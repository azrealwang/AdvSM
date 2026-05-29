#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import _path  # noqa: F401

"""
Defense mask label maps and cross-defense cosine similarity.

Exclusive regions (from build_masks) -> per-pixel labels on [C, H, W]:
  smooth (green, no change):     0  -> gray (128)
  inv (yellow):                 -1  -> black (0)
  transfer + unstable (blue):    1  -> white (255)

Saves grayscale maps per defense/image/channel; cosine on flattened labels in [-1, 1].
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from advsm.classification.utils import load_samples, smart_cast
from advsm.purification import Purifier
from advsm.purification.mask import build_masks

LABEL_TO_GRAY = {-1: 0, 0: 128, 1: 255}
ALLOWED_GRAY = {0, 128, 255}


# -------------------------
# Parsing
# -------------------------
@dataclass
class DefenseSpec:
    name: str
    settings: Dict[str, object]


def parse_defense_specs(def_args: List[str]) -> List[DefenseSpec]:
    specs: List[DefenseSpec] = []
    for s in def_args:
        parts = s.strip().split()
        if not parts:
            continue
        name = parts[0]
        settings: Dict[str, object] = {}
        for kv in parts[1:]:
            if "=" not in kv:
                raise ValueError(f"Bad token {kv!r} in --def {s!r}. Use k=v.")
            k, v = kv.split("=", 1)
            settings[k] = smart_cast(v)
        specs.append(DefenseSpec(name=name, settings=settings))
    if len(specs) < 2:
        raise ValueError("Need at least 2 defenses for cross similarity.")
    return specs


# -------------------------
# Label maps & similarity
# -------------------------
def masks_to_label_map(
    smooth: np.ndarray,
    inv: np.ndarray,
    transfer: np.ndarray,
    unstable: np.ndarray,
) -> np.ndarray:
    """
    Exclusive bool masks [C, H, W] -> int8 labels in {-1, 0, 1}.
    """
    lab = np.zeros(smooth.shape, dtype=np.int8)
    lab[smooth] = 0
    lab[inv] = -1
    lab[transfer] = 1
    lab[unstable] = 1
    return lab


def sample_label_map(masks: Dict[str, torch.Tensor], batch_idx: int) -> np.ndarray:
    """One sample: [C, H, W] int8 labels."""
    return masks_to_label_map(
        masks["smooth"][batch_idx].cpu().numpy(),
        masks["inv"][batch_idx].cpu().numpy(),
        masks["transfer"][batch_idx].cpu().numpy(),
        masks["unstable"][batch_idx].cpu().numpy(),
    )


def labels_to_grayscale_u8(lab: np.ndarray) -> np.ndarray:
    out = np.full(lab.shape, 128, dtype=np.uint8)
    out[lab == -1] = LABEL_TO_GRAY[-1]
    out[lab == 0] = LABEL_TO_GRAY[0]
    out[lab == 1] = LABEL_TO_GRAY[1]
    return out


def save_label_map_png(lab_hw: np.ndarray, out_path: str) -> None:
    img = labels_to_grayscale_u8(lab_hw)
    u = set(np.unique(img).tolist())
    if not u.issubset(ALLOWED_GRAY):
        raise ValueError(f"Unexpected gray levels {u} in {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(img, mode="L").save(out_path)


def label_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def label_map_stats(lab: np.ndarray) -> str:
    n = lab.size
    f = {v: float((lab == v).sum()) / n for v in (-1, 0, 1)}
    return f"inv={f[-1]:.1%} smooth={f[0]:.1%} other={f[1]:.1%}"


# -------------------------
# Batching
# -------------------------
def chunk_tensor(x: torch.Tensor, bs: int) -> List[torch.Tensor]:
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    return [x[i : i + bs] for i in range(0, x.shape[0], bs)]


def merge_masks(mask_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not mask_list:
        raise ValueError("Empty mask_list")
    keys = mask_list[0].keys()
    return {k: torch.cat([m[k] for m in mask_list], dim=0) for k in keys}


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


def column_average_excluding_diag(S: np.ndarray) -> np.ndarray:
    k = S.shape[0]
    col_avg = np.zeros((k,), dtype=np.float64)
    for j in range(k):
        vals = [S[i, j] for i in range(k) if i != j]
        col_avg[j] = float(np.mean(vals)) if vals else float("nan")
    return col_avg


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


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Defense label maps + label cosine similarity.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=32)
    p.add_argument(
        "--def",
        dest="defs",
        action="append",
        required=True,
        help="Defense spec: '<name> k=v ...' (repeat)",
    )
    p.add_argument("--eps", type=int, default=16, help="noise budget 0-255")
    p.add_argument("--thres", type=float, default=2, help="threshold 0-255")
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--baseline_mode", type=str, default="purify_mean", choices=["purify_mean", "identity"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="defense_sim")
    p.add_argument("--save_labels_npy", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--plot_scale", type=str, default="maxabs", choices=["maxabs", "p95", "fixed"])
    p.add_argument("--fixed_lim", type=float, default=1.0)
    p.add_argument("--min_lim", type=float, default=0.05)
    p.add_argument("--text_white_below", type=float, default=0.35)
    p.add_argument("--hide_diag", action="store_true")
    p.add_argument("--title", type=str, default="Defense label-map cosine")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    specs = parse_defense_specs(args.defs)
    defense_labels = [s.name for s in specs]

    x, _ = load_samples(args.input, args.start_idx, args.end_idx)
    x = Tensor(x).to("cuda")
    if args.batch_size is None:
        args.batch_size = len(x)
    n_ch = x.shape[1]

    # label_maps[defense][sample] -> [C, H, W]
    label_maps: List[List[np.ndarray]] = [[] for _ in range(len(specs))]
    maps_root = os.path.join(args.out_dir, "maps")
    labels_root = os.path.join(args.out_dir, "labels")
    stats_logged = [False] * len(specs)

    for di, spec in enumerate(specs):
        purifier = Purifier(spec.name, spec.settings)
        chunk_masks: List[Dict[str, torch.Tensor]] = []
        for xb in chunk_tensor(x, args.batch_size):
            chunk_masks.append(
                build_masks(
                    xb,
                    purifier,
                    eps=args.eps / 255.0,
                    thres=args.thres / 255.0,
                    M=args.M,
                    N=args.N,
                    baseline_mode=args.baseline_mode,
                )
            )
        masks = merge_masks(chunk_masks)
        safe = spec.name.replace(os.sep, "_").replace(" ", "_")

        for bi in range(masks["smooth"].shape[0]):
            img_idx = args.start_idx + bi
            lab = sample_label_map(masks, bi)
            label_maps[di].append(lab)

            if not stats_logged[di]:
                print(f"  [{spec.name}] ch0 {label_map_stats(lab[0])}")
                stats_logged[di] = True

            for c in range(n_ch):
                png = os.path.join(maps_root, safe, f"img_{img_idx:05d}_ch{c}.png")
                save_label_map_png(lab[c], png)
            if args.save_labels_npy:
                npy = os.path.join(labels_root, safe, f"img_{img_idx:05d}.npy")
                os.makedirs(os.path.dirname(npy), exist_ok=True)
                np.save(npy, lab)

    n_samples = len(label_maps[0])
    k = len(specs)
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

    col_avg = column_average_excluding_diag(S)
    tag = f"label_cos_{args.start_idx}-{args.end_idx}_eps{args.eps}_th{args.thres:g}"
    npy_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.npy")
    csv_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.csv")
    avg_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}_colavg.npy")
    np.save(npy_path, S)
    np.save(avg_path, col_avg)
    with open(csv_path, "w") as f:
        f.write("," + ",".join(d.replace(",", ";") for d in defense_labels) + "\n")
        for i, d in enumerate(defense_labels):
            f.write(d.replace(",", ";") + "," + ",".join(f"{S[i, j]:.6f}" for j in range(k)) + "\n")
        f.write("COL_AVG," + ",".join(f"{col_avg[j]:.6f}" for j in range(k)) + "\n")

    print("Defenses:", defense_labels)
    print(f"Samples: {n_samples}, channels: {n_ch}")
    print("Label cosine matrix:\n", S)
    print("Column average (excl. diag):", col_avg)
    print("Saved:", npy_path, csv_path, avg_path)
    print("Maps:", maps_root)

    if args.no_plot:
        return

    lim = _compute_lim(S, args.plot_scale, args.fixed_lim, args.min_lim)
    png_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.png")
    _plot_heatmap(
        S,
        defense_labels,
        png_path,
        title=args.title,
        lim=lim,
        cmap=args.cmap,
        hide_diag=args.hide_diag,
        text_white_below=args.text_white_below,
    )
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
