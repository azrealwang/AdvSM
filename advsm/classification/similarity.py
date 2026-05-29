import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

from torch import Tensor
from advsm.classification.utils import load_samples, load_one_model

# -------------------------
# Batching helpers
# -------------------------
def chunk_tensor(x: torch.Tensor, bs: int) -> List[torch.Tensor]:
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    return [x[i:i + bs] for i in range(0, x.shape[0], bs)]


# -------------------------
# Core: per-image gradient cosine similarity
# -------------------------
def batch_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    """
    Per-sample input gradients of CE(model(x), y), flattened to [B, D],
    and L2-normalized per sample for cosine in [-1, 1].
    """
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
    g = x.grad.detach().view(x.shape[0], -1)
    g = F.normalize(g, p=2, dim=1, eps=1e-12)
    return g


# -------------------------
# Args / label parsing
# -------------------------
def parse_model_specs(model_specs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Each --model can be:
      - "ModelCode"                     -> label = ModelCode
      - "ModelCode=Pretty Label"        -> label = Pretty Label
    """
    names, labels = [], []
    for s in model_specs:
        if "=" in s:
            name, label = s.split("=", 1)
            name = name.strip()
            label = label.strip()
        else:
            name, label = s.strip(), s.strip()
        if not name:
            raise ValueError(f"Bad --model spec: {s!r}")
        names.append(name)
        labels.append(label)
    return names, labels


def _compute_lim(S: np.ndarray, mode: str, fixed_lim: float, min_lim: float) -> float:
    """
    Choose symmetric plot range ±lim based on off-diagonal magnitudes (diagonal ignored).
    """
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    vals = np.abs(off[np.isfinite(off)])
    if vals.size == 0:
        return 1.0

    if mode == "fixed":
        lim = float(fixed_lim)
    elif mode == "p95":
        lim = float(np.percentile(vals, 95))
    else:  # maxabs
        lim = float(vals.max())

    lim = max(lim, float(min_lim))
    lim = min(lim, 1.0)
    return lim


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, required=True, choices=["cifar10", "imagenet"])
    p.add_argument("--threat_model", type=str, default="Linf", help="for robustbench models")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=256)

    p.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Repeat for multiple models. Format: 'ModelCode' or 'ModelCode=Label'.",
    )
    p.add_argument("--batch_size", type=int, default=None, help="outer batch size for sample chunking")

    p.add_argument("--amp", action="store_true", help="use autocast for speed (fp16)")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="model_sim")
    p.add_argument("--no_plot", action="store_true")

    # plot controls (better for mostly ~0 values)
    p.add_argument("--cmap", type=str, default="RdBu_r", help="diverging cmap for cosine in [-1,1]")
    p.add_argument("--plot_scale", type=str, default="maxabs", choices=["maxabs", "p95", "fixed"])
    p.add_argument("--fixed_lim", type=float, default=1.0)
    p.add_argument("--min_lim", type=float, default=0.05, help="avoid too tiny range")
    p.add_argument("--text_white_below", type=float, default=0.2)
    p.add_argument("--hide_diag", action="store_true", help="hide diagonal 1.0 (recommended)")
    p.add_argument("--title", type=str, default="Model Gradient Similarity (Cosine)")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.models) < 2:
        raise ValueError("Need at least 2 models to compute cross similarity.")

    model_names, labels = parse_model_specs(args.models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_np, y_np = load_samples(args.input, args.start_idx, args.end_idx)
    x = Tensor(x_np).to(device)
    y = Tensor(y_np).long().to(device)

    if args.batch_size is None:
        args.batch_size = len(x)

    models: List[nn.Module] = []
    for name in model_names:
        m = load_one_model(args.data, name, threat_model=args.threat_model).to(device).eval()
        models.append(m)

    K = len(models)
    sum_mat = np.zeros((K, K), dtype=np.float64)
    cnt_mat = np.zeros((K, K), dtype=np.float64)

    for xb, yb in zip(chunk_tensor(x, args.batch_size), chunk_tensor(y, args.batch_size)):
        grads: List[torch.Tensor] = []
        for m in models:
            grads.append(batch_gradients(m, xb, yb, device=device, amp=args.amp))

        b = grads[0].shape[0]
        for i in range(K):
            gi = grads[i]
            for j in range(i, K):
                gj = grads[j]
                cos = (gi * gj).sum(dim=1)  # [B]
                s = float(cos.sum().item())
                sum_mat[i, j] += s
                cnt_mat[i, j] += b
                if j != i:
                    sum_mat[j, i] += s
                    cnt_mat[j, i] += b

        del grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    S = sum_mat / np.maximum(cnt_mat, 1.0)
    np.fill_diagonal(S, 1.0)

    # outputs (no COL_AVG)
    tag = f"cos_{args.data}_{args.threat_model}_{args.start_idx}-{args.end_idx}"
    npy_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.npy")
    np.save(npy_path, S)

    csv_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join([l.replace(",", ";") for l in labels]) + "\n")
        for i, l in enumerate(labels):
            f.write(l.replace(",", ";") + "," + ",".join([f"{S[i, j]:.6f}" for j in range(K)]) + "\n")

    print("Model names:", model_names)
    print("Labels:", labels)
    print("Saved:", npy_path)
    print("Saved:", csv_path)

    if args.no_plot:
        return

    lim = _compute_lim(S, args.plot_scale, args.fixed_lim, args.min_lim)

    fig = plt.figure(figsize=(max(6, 0.8 * K), max(5, 0.8 * K)))
    ax = plt.gca()

    if args.hide_diag:
        S_plot = S.copy()
        np.fill_diagonal(S_plot, np.nan)
        cmap = mpl.cm.get_cmap(args.cmap).copy()
        cmap.set_bad(color="white")
        im = ax.imshow(np.ma.masked_invalid(S_plot), vmin=-lim, vmax=lim, cmap=cmap)
    else:
        im = ax.imshow(S, vmin=-lim, vmax=lim, cmap=args.cmap)

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", pad=6)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(K):
        for j in range(K):
            if args.hide_diag and i == j:
                continue
            v = float(S[i, j])
            txt_color = "white" if v > args.text_white_below else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_title(f"{args.title}  |  scale=±{lim:.2f}")

    plt.tight_layout()
    png_path = os.path.join(args.out_dir, f"{args.prefix}_{tag}.png")
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()