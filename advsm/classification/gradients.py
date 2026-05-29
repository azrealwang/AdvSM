"""
Plot per-model input-gradient value distributions (bar charts).

One subplot per model; each uses its own x/y range.
Default: signed g on x in [-A, +A] with A = p99(|g|) per model (tails go to edge bins).
Y label is density (fraction per bin, sums to 1). Layout: 2 rows.
"""

import argparse
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from advsm.classification.utils import load_one_model, load_samples
from advsm.classification.similarity import chunk_tensor, parse_model_specs


def batch_raw_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    if not isinstance(yb, torch.Tensor):
        yb = torch.as_tensor(yb)
    if not isinstance(xb, torch.Tensor):
        xb = torch.as_tensor(xb)
    y = yb.to(device).long()
    x = xb.detach().clone().to(device).requires_grad_(True)

    model.zero_grad(set_to_none=True)
    if amp:
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
    else:
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

    loss.backward()
    return x.grad.detach().view(x.shape[0], -1)


def collect_grad_values(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
    use_abs: bool,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for xb, yb in zip(chunk_tensor(x, batch_size), chunk_tensor(y, batch_size)):
        g = batch_raw_gradients(model, xb, yb, device=device, amp=amp)
        v = g.detach().cpu().numpy().ravel()
        if use_abs:
            v = np.abs(v)
        parts.append(v)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _subplot_grid(n: int) -> Tuple[int, int]:
    """Always 2 rows; columns fit all models."""
    nrow = 2
    ncol = int(math.ceil(n / nrow))
    return nrow, ncol


def _x_tick_fmt(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    if abs(x) < 0.001:
        return f"{x:.1e}"
    return f"{x:.3f}"


def _y_tick_fmt(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    return f"{x:.2f}"


def _apply_x_tick_format(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(_x_tick_fmt))


def _apply_y_tick_format(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(_y_tick_fmt))


def _cap_half_width(v: np.ndarray, cap_percentile: float) -> float:
    """Half-width A from |g| percentiles so extreme signed tails do not stretch the axis."""
    if cap_percentile >= 100.0:
        return max(float(np.max(np.abs(v))), 1e-30)
    a = float(np.percentile(np.abs(v), cap_percentile))
    return max(a, 1e-30)


def _cap_bounds(v: np.ndarray, use_abs: bool, cap_percentile: float) -> Tuple[float, float]:
    """
    Signed: symmetric [-A, A] with A = percentile(|g|, cap).
    |g| mode: [0, A] with the same A.
    """
    a = _cap_half_width(v, cap_percentile)
    if use_abs:
        return 0.0, a
    return -a, a


def histogram_rate(
    v: np.ndarray,
    bins: int,
    use_abs: bool,
    cap_percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin counts / total => rate per bin (sums to 1). Outliers fall in end bins."""
    lo, hi = _cap_bounds(v, use_abs, cap_percentile)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(v, bins=edges)
    rate = counts.astype(np.float64) / max(counts.sum(), 1)
    return rate, edges


def plot_grid(
    *,
    values_list: List[np.ndarray],
    labels: List[str],
    bins: int,
    use_abs: bool,
    cap_percentile: float,
    out_path: str,
    title: str,
) -> None:
    nrow, ncol = _subplot_grid(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)

    for idx, (v, label) in enumerate(zip(values_list, labels)):
        ax = axes[idx // ncol][idx % ncol]
        if v.size == 0:
            ax.set_title(f"{label}\n(empty)")
            continue

        rate, edges = histogram_rate(v, bins, use_abs, cap_percentile)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.9
        ax.bar(
            centers,
            rate,
            width=width,
            align="center",
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
        )
        ymax = float(rate.max()) if rate.size else 1.0
        x_lo, x_hi = float(edges[0]), float(edges[-1])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("|gradient|" if use_abs else "gradient")
        ax.set_ylabel("density")
        _apply_x_tick_format(ax)
        _apply_y_tick_format(ax)

    for idx in range(len(labels), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, choices=["cifar10", "imagenet"])
    p.add_argument("--threat_model", type=str, default="Linf")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=1)
    p.add_argument("--model", dest="models", action="append", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="grad_dist")
    p.add_argument("--title", type=str, default="Gradient value distributions")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--abs", action="store_true", help="Use |g| on x>=0 (default: signed ±g).")
    p.add_argument(
        "--cap_percentile",
        type=float,
        default=99.0,
        help="A = percentile(|g|, cap); signed x is [-A,A]. 100 => A=max|g|.",
    )
    p.add_argument("--no_cap", action="store_true", help="Same as --cap_percentile 100.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_names, labels = parse_model_specs(args.models)
    use_abs = args.abs
    cap_percentile = 100.0 if args.no_cap else float(args.cap_percentile)

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

    values_list: List[np.ndarray] = []
    for m, label in zip(models, labels):
        print(f"{label} ...")
        v = collect_grad_values(m, x, y, args.batch_size, device, args.amp, use_abs)
        values_list.append(v)
        lo, hi = _cap_bounds(v, use_abs, cap_percentile)
        a = _cap_half_width(v, cap_percentile)
        print(
            f"  n={v.size}, full [{v.min():.2e}, {v.max():.2e}], "
            f"cap p{cap_percentile:g}(|g|) A={a:.2e} => x [{lo:.2e}, {hi:.2e}]"
        )

    tag = f"{args.data}_img{args.start_idx}_{'abs' if use_abs else 'signed'}"
    if cap_percentile < 100.0:
        tag = f"{tag}_cap{cap_percentile:g}"
    png_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.png")
    plot_grid(
        values_list=values_list,
        labels=labels,
        bins=args.bins,
        use_abs=use_abs,
        cap_percentile=cap_percentile,
        out_path=png_path,
        title=args.title,
    )
    print("Saved:", png_path)


if __name__ == "__main__":
    main()
