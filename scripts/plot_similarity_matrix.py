#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _path  # noqa: F401

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_similarity_matrix_from_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Supports formats:

    1) With explicit header row (from defense_similarity.py):
       ,label1,label2,...
       label1,1.0,0.2,...
       ...

    2) Column headers only on first row (no leading empty cell), then data rows:
       label1,label2,label3,...
       label1,1.0,0.2,...
       label2,0.2,1.0,...
       (row i has: row_label, then K numeric cells matching K column headers)

    3) With first column being row labels only (no separate header row):
       label1,1.0,0.2,...
       label2,0.2,1.0,...
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    rows = [[c.strip() for c in ln.split(",")] for ln in raw_lines]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # Detect header format: first row first cell is empty string.
    first_row = rows[0]
    header_format = len(first_row) >= 2 and first_row[0] == ""

    if header_format:
        labels = [c for c in first_row[1:] if c != ""]
        K = len(labels)
        S = np.zeros((K, K), dtype=np.float64)

        if len(rows) != K + 1:
            # Some CSV writers might include fewer/extra whitespace; be permissive.
            pass

        # Remaining rows should be K rows
        data_rows = rows[1:]
        if len(data_rows) < K:
            raise ValueError(f"Not enough rows for K={K} in {path}")
        for i in range(K):
            r = data_rows[i]
            row_label = r[0]
            # tolerate mismatch of row_label order
            vals = r[1 : 1 + K]
            if len(vals) != K:
                raise ValueError(f"Row {i} length mismatch in {path}")
            S[i, :] = np.asarray([float(v) for v in vals], dtype=np.float64)

        return S, labels

    # Column-header-only first row: row0 has K names; each following row is
    # row_label + K floats (so len(row_{i+1}) == K + 1). Avoids mis-parsing
    # row0 as "label + string targets" when the leading comma is missing.
    if len(rows) >= 2:
        K_hdr = len(rows[0])
        r1 = rows[1]
        if (
            K_hdr >= 1
            and len(r1) == K_hdr + 1
            and all(_is_float(r1[j]) for j in range(1, len(r1)))
        ):
            labels = [c for c in rows[0] if c != ""]
            K = len(labels)
            if K_hdr != K:
                raise ValueError(
                    f"Header row has {K_hdr} cells but only {K} non-empty labels in {path}"
                )
            if len(rows) < K + 1:
                raise ValueError(f"Not enough data rows for K={K} in {path}")
            S = np.zeros((K, K), dtype=np.float64)
            for i in range(K):
                r = rows[i + 1]
                if len(r) != K + 1:
                    raise ValueError(f"Row {i + 1} length mismatch in {path}")
                if not all(_is_float(r[j]) for j in range(1, K + 1)):
                    raise ValueError(
                        f"Non-numeric value in data row {i + 1} of {path}: {r!r}"
                    )
                S[i, :] = np.asarray([float(r[j]) for j in range(1, K + 1)], dtype=np.float64)
            return S, labels

    # Otherwise: first column is row labels, remaining cells are numeric
    labels = []
    first = rows[0]
    if len(first) < 2:
        raise ValueError(f"Bad matrix CSV format: {path}")

    K = len(first) - 1
    S = np.zeros((K, K), dtype=np.float64)
    for i, r in enumerate(rows):
        if len(r) != K + 1:
            raise ValueError(f"Row {i} has {len(r)} columns but expected {K+1} in {path}")
        label = r[0]
        labels.append(label)
        vals = r[1:]
        try:
            S[i, :] = np.asarray([float(v) for v in vals], dtype=np.float64)
        except ValueError as e:
            raise ValueError(
                f"Could not parse numeric row {i} in {path}. "
                f"If the first row is column headers, ensure it has a leading comma "
                f"(,col1,col2,...) or exactly K header names with no row-label column. "
                f"Row: {r!r}"
            ) from e

    if len(labels) != K:
        # If CSV has fewer rows than K+1, it's still likely incorrect.
        raise ValueError(f"Expected {K} labeled rows, got {len(labels)} in {path}")

    return S, labels


def column_average_excluding_diag(S: np.ndarray) -> np.ndarray:
    K = S.shape[0]
    col_avg = np.zeros((K,), dtype=np.float64)
    for j in range(K):
        vals = [S[i, j] for i in range(K) if i != j]
        col_avg[j] = float(np.mean(vals)) if vals else float("nan")
    return col_avg


def plot_heatmap(
    *,
    S: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
    vmin: Optional[float],
    vmax: Optional[float],
    cmap: str = "Reds",
    xlabel_rotation: float = 45.0,
    with_colavg_in_xlabel: bool = True,
    text_white_above: float = 0.35,
    avg_bar: bool = False,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got shape={S.shape}")
    K = S.shape[0]
    if len(labels) != K:
        raise ValueError(f"labels length {len(labels)} != K {K}")

    col_avg = column_average_excluding_diag(S)
    if with_colavg_in_xlabel:
        xlabels = [f"{labels[j]}\n{col_avg[j]:.2f}" for j in range(K)]
    else:
        xlabels = [f"{labels[j]}" for j in range(K)]

    if avg_bar:
        fig = plt.figure(figsize=(max(6, 0.8 * K), max(6, 0.8 * K)))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig = plt.figure(figsize=(max(6, 0.8 * K), max(5, 0.8 * K)))
        ax = plt.gca()
        ax_bar = None

    S_plot = S.copy()
    np.fill_diagonal(S_plot, np.nan)

    # Defense-style: sequential colormap.
    if vmin is None:
        vmin = float(np.nanmin(S))
    if vmax is None:
        vmax = float(np.nanmax(S))
    cmap_obj = mpl.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    im = ax.imshow(np.ma.masked_invalid(S_plot), vmin=vmin, vmax=vmax, cmap=cmap_obj)
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticklabels(xlabels, ha="right", rotation=xlabel_rotation, rotation_mode="anchor", fontsize=9)
    ax.tick_params(axis="x", pad=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(cbar_label)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            v = float(S[i, j])
            txt_color = "white" if abs(v) > text_white_above else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_title(title)

    if ax_bar is not None:
        cmap_single = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap_single(norm(float(v))) for v in col_avg]
        ax_bar.bar(np.arange(K), col_avg, color=colors)
        ax_bar.set_ylim(vmin, vmax)
        ax_bar.set_ylabel("COL_AVG", fontsize=9)
        ax_bar.set_xticks(np.arange(K))
        ax_bar.set_xticklabels(xlabels, rotation=xlabel_rotation, ha="center", rotation_mode="anchor", fontsize=9)
        ax_bar.tick_params(axis="x", pad=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True, help="CSV containing similarity matrix with labels.")
    p.add_argument("--out_path", type=str, required=True, help="Output PNG path.")
    p.add_argument("--title", type=str, default="", help="Plot title.")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--cmap", type=str, default="Reds", help="Reds, RdBu_r.")
    p.add_argument("--xlabel_rotation", type=float, default=45.0)
    p.add_argument("--no-with_colavg_in_xlabel", action="store_true", help="Disable col-avg in x labels.")
    p.add_argument("--text_white_above", type=float, default=0.35)
    p.add_argument("--avg_bar", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    S, labels = load_similarity_matrix_from_csv(args.csv_path)
    with_colavg_in_xlabel = not getattr(args, "no_with_colavg_in_xlabel", False)
    plot_heatmap(
        S=S,
        labels=labels,
        out_path=args.out_path,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        xlabel_rotation=args.xlabel_rotation,
        with_colavg_in_xlabel=with_colavg_in_xlabel,
        text_white_above=args.text_white_above,
        avg_bar=bool(args.avg_bar),
    )


if __name__ == "__main__":
    main()

