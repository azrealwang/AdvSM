#!/usr/bin/env python3
"""
Cross-model **input-gradient label sensitivity** similarity for VQA models in this repo.

Uses teacher-forcing NLL (same objective as ``attack_pgd.py``): for each sample,
g = ∂ NLL / ∂ image (raw, not L2-normalized). Per pixel/channel:

  |g| <= threshold  -> label 0  (gray)
  |g| > threshold, g > 0  -> label 1  (white)
  |g| > threshold, g < 0  -> label -1 (black)

Cross-model similarity is **cosine on flattened label maps** in [-1, 0, 1], averaged over the eval slice
(same logic as ``aml_label/model_gradient_sensitivity_similarity.py``).

Models are loaded **one at a time**; by default each model's label maps are cached on CPU.
Use ``--reload-model-each-batch`` for the legacy per-batch reload, or if the cache would exceed
``--grad-cache-max-gb``.

Example:
  python scripts/vqa_model_gradient_similarity.py \\
    --models clip fare tecoa simclip \\
    --models-config configs/robust_vqa_models.yaml \\
    --data-jsonl data/vqa/vqav2_val.jsonl \\
    --image-root data/coco/val2014 \\
    --subset-file data/vqa/all_correct.ids \\
    --out-dir outputs/sim \\
    --start-idx 0 --end-idx 64 \\
    --batch-size 1 \\
    --threshold 1e-5

Label PNGs are written to ``out_dir/maps/`` by default (use ``--no-save-maps`` to skip).
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import _path  # noqa: F401
from advsm.vlm.data import VQAJsonlDataset, collate_vqa_batch
from advsm.vlm.models import load_model_from_config

LABEL_TO_GRAY = {-1: 0, 0: 128, 1: 255}

DEFAULT_DISPLAY_LABELS: Dict[str, str] = {
    "clip": "CLIP",
    "fare": "FARE",
    "tecoa": "TeCoA",
    "simclip": "SimCLIP",
}


def resolve_plot_labels(model_keys: List[str], label_overrides: List[str] | None) -> List[str]:
    overrides: Dict[str, str] = {}
    for s in label_overrides or []:
        if "=" not in s:
            raise ValueError(f"--model-label must be key=Label, got: {s!r}")
        k, lab = s.split("=", 1)
        k, lab = k.strip(), lab.strip()
        if not k:
            raise ValueError(f"Bad --model-label: {s!r}")
        overrides[k] = lab

    out: List[str] = []
    for k in model_keys:
        if k in overrides:
            out.append(overrides[k])
        else:
            out.append(DEFAULT_DISPLAY_LABELS.get(k, k.replace("_", " ").title()))
    return out


def gradients_to_labels(g: np.ndarray, threshold: float) -> np.ndarray:
    """g: [C, H, W] -> int8 labels in {-1, 0, 1}."""
    g = np.asarray(g, dtype=np.float64)
    abs_g = np.abs(g)
    labels = np.zeros(g.shape, dtype=np.int8)
    low = abs_g <= threshold
    labels[low] = 0
    high = ~low
    labels[high & (g > 0)] = 1
    labels[high & (g < 0)] = -1
    return labels


def labels_to_grayscale_u8(lab: np.ndarray) -> np.ndarray:
    out = np.zeros(lab.shape, dtype=np.uint8)
    for label, gray in LABEL_TO_GRAY.items():
        out[lab == label] = gray
    return out


def save_label_map_png(lab_hw: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(labels_to_grayscale_u8(lab_hw), mode="L").save(out_path)


def label_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Label map shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def _batch_raw_grad_nll(
    model,
    images: torch.Tensor,
    questions: list[str],
    answers: list[str],
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Per-sample raw ∂NLL/∂x, shape [B, C, H, W] on CPU."""
    images = images.detach().clone().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            loss = model.compute_nll_loss(images, questions, answers)
    else:
        loss = model.compute_nll_loss(images, questions, answers)
    loss.backward()
    return images.grad.detach().cpu()


def batch_label_maps(
    model,
    images: torch.Tensor,
    questions: list[str],
    answers: list[str],
    device: torch.device,
    threshold: float,
    amp: bool,
) -> List[np.ndarray]:
    """Per-sample label maps [C, H, W] int8."""
    g = _batch_raw_grad_nll(model, images, questions, answers, device=device, amp=amp)
    b, c, h, w = g.shape
    out: List[np.ndarray] = []
    for i in range(b):
        out.append(gradients_to_labels(g[i].numpy(), threshold))
    return out


def _safe_dir_name(label: str) -> str:
    return label.replace(os.sep, "_").replace(" ", "_")


def _save_matrix_csv(S: np.ndarray, row_labels: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(l.replace(",", ";") for l in row_labels) + "\n")
        for i, lab in enumerate(row_labels):
            f.write(
                lab.replace(",", ";")
                + ","
                + ",".join(f"{S[i, j]:.6f}" for j in range(S.shape[1]))
                + "\n"
            )


def _compute_lim(S: np.ndarray, mode: str, fixed_lim: float, min_lim: float) -> float:
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    vals = np.abs(off[np.isfinite(off)])
    if vals.size == 0:
        return 1.0
    if mode == "fixed":
        lim = float(fixed_lim)
    elif mode == "p95":
        lim = float(np.percentile(vals, 95))
    else:
        lim = float(vals.max())
    return max(min(lim, 1.0), float(min_lim))


def _plot_heatmap(
    S: np.ndarray,
    row_labels: List[str],
    out_path: str,
    *,
    title: str,
    lim: float,
    cmap: str,
    hide_diag: bool,
    text_white_above: float,
) -> None:
    k = S.shape[0]
    # fig, ax = plt.subplots(figsize=(max(6, 0.8 * k), max(5, 0.8 * k)))
    fig, ax = plt.subplots(figsize=(k,k))
    if hide_diag:
        S_plot = S.copy()
        np.fill_diagonal(S_plot, np.nan)
        try:
            c = mpl.colormaps.get_cmap(cmap).copy()
        except AttributeError:
            c = mpl.cm.get_cmap(cmap).copy()
        c.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(S_plot), vmin=-lim, vmax=lim, cmap=c)
    else:
        im = ax.imshow(S, vmin=-lim, vmax=lim, cmap=cmap)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(row_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="x", pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(k):
        for j in range(k):
            if hide_diag and i == j:
                continue
            v = float(S[i, j])
            color = "white" if abs(v) > text_white_above else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="VQA gradient label-sensitivity cosine similarity across models (NLL w.r.t. images)."
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="YAML model keys: clip fare tecoa simclip",
    )
    p.add_argument(
        "--model-label",
        dest="model_labels",
        action="append",
        default=None,
        help="Optional override for plot/CSV axis text: yaml_key=Label (repeatable).",
    )
    p.add_argument("--models-config", required=True)
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--subset-file", default=None)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--amp", action="store_true", help="autocast for forward+NLL (CUDA only)")
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-5,
        help="|g| <= threshold -> label 0; else sign(g) -> +/-1 (same as aml_label grad sensitivity).",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="vqa_grad_sim")
    p.add_argument(
        "--save-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-model/channel grayscale label PNGs under out_dir/maps/ (default: on).",
    )
    p.add_argument(
        "--save-labels-npy",
        action="store_true",
        help="Save [C,H,W] int8 label arrays under out_dir/labels/.",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--plot-scale", type=str, default="maxabs", choices=["maxabs", "p95", "fixed"])
    p.add_argument("--fixed-lim", type=float, default=1.0)
    p.add_argument("--min-lim", type=float, default=0.05)
    p.add_argument(
        "--title",
        type=str,
        default="VQA Encoder Similarity (Cosine)",
        help="Figure title (ignored with --no-plot).",
    )
    p.add_argument(
        "--text-white-above",
        "--text-white-below",
        dest="text_white_above",
        type=float,
        default=0.2,
        metavar="T",
        help="White annotation text when |similarity| exceeds T.",
    )
    p.add_argument("--hide-diag", action="store_true")
    p.add_argument(
        "--reload-model-each-batch",
        action="store_true",
        help="Load every model on every batch (slow). Default caches label maps per model.",
    )
    p.add_argument(
        "--grad-cache-max-gb",
        type=float,
        default=12.0,
        help="If estimated CPU cache for all label maps exceeds this, use per-batch model reload.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")
    if len(args.models) < 2:
        raise ValueError("Need at least 2 models.")

    os.makedirs(args.out_dir, exist_ok=True)
    model_keys = [k.strip() for k in args.models if k.strip()]
    if len(set(model_keys)) != len(model_keys):
        raise ValueError("--models contains duplicates.")
    plot_labels = resolve_plot_labels(model_keys, args.model_labels)
    K = len(model_keys)
    print("Models (YAML keys):", model_keys)
    print("Plot labels:", plot_labels)
    print(f"Label threshold: {args.threshold:g}")

    ds = VQAJsonlDataset(
        args.data_jsonl,
        image_root=args.image_root,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        subset_file=args.subset_file,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vqa_batch)

    n_samples = len(ds)
    n_batches = int(math.ceil(n_samples / max(1, args.batch_size)))
    max_side = 336
    bytes_per_label = 3 * max_side * max_side  # int8 C*H*W upper bound
    est_cache_bytes = K * n_batches * max(1, args.batch_size) * bytes_per_label
    use_stream = bool(args.reload_model_each_batch) or (
        est_cache_bytes > float(args.grad_cache_max_gb) * (1024**3)
    )
    if use_stream and not args.reload_model_each_batch:
        print(
            f"Note: estimated label-map cache {est_cache_bytes / (1024**3):.2f} GiB > "
            f"--grad-cache-max-gb {args.grad_cache_max_gb}; using per-batch model reload."
        )

    maps_root = os.path.join(args.out_dir, "maps")
    labels_root = os.path.join(args.out_dir, "labels")
    global_idx = args.start_idx

    if use_stream:
        sum_mat = np.zeros((K, K), dtype=np.float64)
        cnt_mat = np.zeros((K, K), dtype=np.float64)
        for batch in tqdm(dl, desc="batches(stream)", unit="batch"):
            imgs = batch["image"]
            qs = batch["question"]
            ans = batch["answer"]
            bsz = imgs.shape[0]
            n_ch = imgs.shape[1]
            per_model: List[List[np.ndarray]] = []
            for key in tqdm(model_keys, desc="models/batch", leave=False):
                m = load_model_from_config(key, args.models_config)
                dev_str = getattr(m, "_device", "cuda")
                dev = torch.device("cpu" if dev_str == "cuda" and not torch.cuda.is_available() else dev_str)
                m = m.to(dev)
                m.eval()
                try:
                    labs = batch_label_maps(m, imgs, qs, ans, dev, args.threshold, bool(args.amp))
                finally:
                    del m
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                per_model.append(labs)

            for bi in range(bsz):
                img_idx = global_idx + bi
                for mi, plab in enumerate(plot_labels):
                    lab = per_model[mi][bi]
                    if args.save_maps:
                        safe = _safe_dir_name(plab)
                        for c in range(n_ch):
                            png = os.path.join(maps_root, safe, f"img_{img_idx:05d}_ch{c}.png")
                            save_label_map_png(lab[c], png)
                    if args.save_labels_npy:
                        safe = _safe_dir_name(plab)
                        npy = os.path.join(labels_root, safe, f"img_{img_idx:05d}.npy")
                        os.makedirs(os.path.dirname(npy), exist_ok=True)
                        np.save(npy, lab)
                for i in range(K):
                    for j in range(i, K):
                        s = label_cosine_similarity(per_model[i][bi], per_model[j][bi])
                        sum_mat[i, j] += s
                        cnt_mat[i, j] += 1
                        if j != i:
                            sum_mat[j, i] += s
                            cnt_mat[j, i] += 1
            global_idx += bsz

        S = sum_mat / np.maximum(cnt_mat, 1.0)
        np.fill_diagonal(S, 1.0)
        n_counted = int(cnt_mat[0, 0]) if cnt_mat.size else 0
    else:
        label_maps: List[List[np.ndarray]] = [[] for _ in range(K)]
        for mi, key in enumerate(tqdm(model_keys, desc="models(load once)")):
            m = load_model_from_config(key, args.models_config)
            dev_str = getattr(m, "_device", "cuda")
            dev = torch.device("cpu" if dev_str == "cuda" and not torch.cuda.is_available() else dev_str)
            m = m.to(dev)
            m.eval()
            img_idx = args.start_idx
            try:
                for batch in tqdm(dl, desc=f"  batches[{key}]", leave=False, unit="batch"):
                    imgs = batch["image"]
                    qs = batch["question"]
                    ans = batch["answer"]
                    n_ch = imgs.shape[1]
                    labs = batch_label_maps(m, imgs, qs, ans, dev, args.threshold, bool(args.amp))
                    for bi, lab in enumerate(labs):
                        label_maps[mi].append(lab)
                        if args.save_maps or args.save_labels_npy:
                            safe = _safe_dir_name(plot_labels[mi])
                            if args.save_maps:
                                for c in range(n_ch):
                                    png = os.path.join(maps_root, safe, f"img_{img_idx:05d}_ch{c}.png")
                                    save_label_map_png(lab[c], png)
                            if args.save_labels_npy:
                                npy = os.path.join(labels_root, safe, f"img_{img_idx:05d}.npy")
                                os.makedirs(os.path.dirname(npy), exist_ok=True)
                                np.save(npy, lab)
                        img_idx += 1
            finally:
                del m
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        nb = len(label_maps[0])
        for i in range(K):
            if len(label_maps[i]) != nb:
                raise RuntimeError(f"Inconsistent sample count for model {i}: {len(label_maps[i])} vs {nb}")

        S = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            S[i, i] = 1.0
            for j in range(i + 1, K):
                sims = [
                    label_cosine_similarity(label_maps[i][t], label_maps[j][t]) for t in range(nb)
                ]
                s = float(np.mean(sims)) if sims else 0.0
                S[i, j] = S[j, i] = s
        n_counted = nb

    end_s = args.end_idx if args.end_idx is not None else "end"
    tag = f"label_cos_{args.start_idx}-{end_s}_n{n_counted}_th{args.threshold:g}"
    npy_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.npy")
    csv_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.csv")
    np.save(npy_path, S)
    _save_matrix_csv(S, plot_labels, csv_path)

    print("Label cosine matrix:\n", S)
    print("Saved:", npy_path)
    print("Saved:", csv_path)
    if args.save_maps:
        print("Maps:", maps_root)
    if args.save_labels_npy:
        print("Labels:", labels_root)

    if args.no_plot:
        return

    lim = _compute_lim(S, args.plot_scale, args.fixed_lim, args.min_lim)
    png_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.png")
    _plot_heatmap(
        S,
        plot_labels,
        png_path,
        title=f"{args.title}",
        lim=lim,
        cmap=args.cmap,
        hide_diag=args.hide_diag,
        text_white_above=args.text_white_above,
    )
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
