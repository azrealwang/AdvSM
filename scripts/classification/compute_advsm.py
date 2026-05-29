#!/usr/bin/env python3
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from advsm._paths import ensure_repo_on_path, ensure_third_party_on_path

ensure_repo_on_path()
ensure_third_party_on_path()

"""
Per-model input-gradient label maps and cross-model label cosine similarity.

All channels, per pixel:
  |g| <= threshold:  label 0  -> gray
  |g| > threshold, g < 0:  label -1  -> black
  |g| > threshold, g > 0:  label 1   -> white

Saves grayscale maps per model/image/channel; cosine on flattened [C,H,W] labels in [-1, 1].
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from advsm.classification.utils import load_one_model, load_samples

from advsm.classification.similarity import chunk_tensor, parse_model_specs
from advsm.classification.gradients import batch_raw_gradients

LABEL_TO_GRAY = {-1: 0, 0: 128, 1: 255}


def gradients_to_labels(g: np.ndarray, threshold: float) -> np.ndarray:
    """g: [C, H, W] -> labels int8 in {-1, 0, 1}, same shape."""
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
    """[H, W] labels -> uint8 grayscale."""
    out = np.zeros(lab.shape, dtype=np.uint8)
    for label, gray in LABEL_TO_GRAY.items():
        out[lab == label] = gray
    return out


def save_label_map_png(lab_hw: np.ndarray, out_path: str) -> None:
    """Save exact uint8 gray levels (0/128/255); avoid matplotlib colormap re-quantization."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(labels_to_grayscale_u8(lab_hw), mode="L").save(out_path)


def label_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine on flattened label maps; range [-1, 1]."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def batch_label_maps(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
    threshold: float,
    amp: bool = False,
) -> List[np.ndarray]:
    """Per-sample label maps [C, H, W]."""
    g = batch_raw_gradients(model, xb, yb, device=device, amp=amp)
    b, d = g.shape
    c, h, w = xb.shape[1], xb.shape[2], xb.shape[3]
    if c * h * w != d:
        raise ValueError(f"Cannot reshape grad dim {d} to {c}x{h}x{w}")
    g = g.view(b, c, h, w).detach().cpu().numpy()
    return [gradients_to_labels(g[i], threshold) for i in range(b)]


def _save_matrix_csv(S: np.ndarray, row_labels: List[str], path: str) -> None:
    with open(path, "w") as f:
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
    text_white_below: float,
) -> None:
    k = S.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * k), max(5, 0.8 * k)))
    S_plot = S.copy()
    if hide_diag:
        np.fill_diagonal(S_plot, np.nan)
        c = mpl.cm.get_cmap(cmap).copy()
        c.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(S_plot), vmin=-lim, vmax=lim, cmap=c)
    else:
        im = ax.imshow(S, vmin=-lim, vmax=lim, cmap=cmap)
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(k):
        for j in range(k):
            if hide_diag and i == j:
                continue
            v = float(S[i, j])
            color = "white" if abs(v) > text_white_below else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Label sensitivity maps + label cosine across models.")
    p.add_argument("--data", type=str, required=True, choices=["cifar10", "imagenet"])
    p.add_argument("--threat_model", type=str, default="Linf")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=100, help="AdvSM sample count (default 100 clean-correct images)")
    p.add_argument("--model", dest="models", action="append", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--threshold", type=float, default=1e-5)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="grad_sens")
    p.add_argument("--save_labels_npy", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--plot_scale", type=str, default="maxabs", choices=["maxabs", "p95", "fixed"])
    p.add_argument("--fixed_lim", type=float, default=1.0)
    p.add_argument("--min_lim", type=float, default=0.05)
    p.add_argument("--text_white_below", type=float, default=0.1)
    p.add_argument(
        "--hide_diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask diagonal in heatmap (default: hide). Use --no-hide_diag to show 1.0 on diagonal.",
    )
    p.add_argument("--title", type=str, default="Classifier Similarity (Cosine)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if len(args.models) < 2:
        raise ValueError("Need at least 2 models.")

    model_names, model_labels = parse_model_specs(args.models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_np, y_np = load_samples(args.input, args.start_idx, args.end_idx)
    x = Tensor(x_np).to(device)
    y = Tensor(y_np).long().to(device)
    if args.batch_size is None:
        args.batch_size = x.shape[0]

    models = [
        load_one_model(args.data, name, threat_model=args.threat_model).to(device).eval()
        for name in model_names
    ]
    k = len(models)
    n_ch = x.shape[1]

    # label_maps[model][sample] -> [C, H, W]
    label_maps: List[List[np.ndarray]] = [[] for _ in range(k)]
    maps_root = os.path.join(args.out_dir, "maps")
    labels_root = os.path.join(args.out_dir, "labels")
    global_idx = args.start_idx

    for xb, yb in zip(chunk_tensor(x, args.batch_size), chunk_tensor(y, args.batch_size)):
        per_model = [
            batch_label_maps(m, xb, yb, device, args.threshold, amp=args.amp) for m in models
        ]
        for bi in range(xb.shape[0]):
            img_idx = global_idx + bi
            for mi, name in enumerate(model_labels):
                lab = per_model[mi][bi]  # [C, H, W]
                label_maps[mi].append(lab)
                safe = name.replace(os.sep, "_").replace(" ", "_")
                for c in range(n_ch):
                    png = os.path.join(maps_root, safe, f"img_{img_idx:05d}_ch{c}.png")
                    save_label_map_png(lab[c], png)
                if args.save_labels_npy:
                    npy = os.path.join(labels_root, safe, f"img_{img_idx:05d}.npy")
                    os.makedirs(os.path.dirname(npy), exist_ok=True)
                    np.save(npy, lab)
        global_idx += xb.shape[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_samples = len(label_maps[0])
    S = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        S[i, i] = 1.0
        for j in range(i + 1, k):
            sims = [
                label_cosine_similarity(label_maps[i][t], label_maps[j][t])
                for t in range(n_samples)
            ]
            s = float(np.mean(sims)) if sims else 0.0
            S[i, j] = S[j, i] = s

    tag = f"label_cos_{args.data}_{args.threat_model}_{args.start_idx}-{args.end_idx}_th{args.threshold:g}"
    npy_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.npy")
    csv_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.csv")
    np.save(npy_path, S)
    _save_matrix_csv(S, model_labels, csv_path)

    print("Models:", model_names)
    print(f"Samples: {n_samples}, channels: {n_ch}, threshold: {args.threshold:g}")
    print("Label cosine matrix:\n", S)
    print("Saved:", npy_path, csv_path)
    print("Maps:", maps_root)

    if args.no_plot:
        return

    lim = _compute_lim(S, args.plot_scale, args.fixed_lim, args.min_lim)
    png_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.png")
    _plot_heatmap(
        S,
        model_labels,
        png_path,
        title=f"{args.title}",
        lim=lim,
        cmap=args.cmap,
        hide_diag=args.hide_diag,
        text_white_below=args.text_white_below,
    )
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
