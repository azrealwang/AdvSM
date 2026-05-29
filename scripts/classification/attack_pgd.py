#!/usr/bin/env python3
"""Untargeted Linf PGD on paper ImageNet classifiers."""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from advsm._paths import ensure_repo_on_path, ensure_third_party_on_path

ensure_repo_on_path()
ensure_third_party_on_path()

import argparse
from time import time

import torch
from tqdm import tqdm

from advsm.classification.models import _PAPER_MODEL_KEYS, load_classifier
from advsm.classification.utils import load_samples, save_all_images
from advsm.purification.attacks import PGDTransfer


def parse_args():
    p = argparse.ArgumentParser(description="PGD attack on ImageNet classifiers.")
    p.add_argument("--model", type=str, required=True, choices=_PAPER_MODEL_KEYS)
    p.add_argument("--eps", type=float, default=4, help="Linf budget ε in 1/255 units (default 4 → 4/255)")
    p.add_argument("--max_iter", type=int, default=10, help="PGD iterations (default 10)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eps = args.eps / 255.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_classifier(args.model)
    attack = PGDTransfer(
        model=model,
        targeted=False,
        n_iter=args.max_iter,
        norm="Linf",
        eps=eps,
        eot_iter=1,
        eot_mode="iter",
        bpda_mode="skip",
        device=device,
        seed=args.seed,
    )

    x_test, y_test = load_samples(args.input, args.start_idx, args.end_idx)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    os.makedirs(args.output, exist_ok=True)
    t0 = time()
    n = len(y_test)
    bs = max(1, args.batch_size)
    starts = range(0, n, bs)
    for start in tqdm(starts, desc=f"attack {args.model}", unit="batch"):
        end = min(start + bs, n)
        x_adv = attack.perturb(x_test[start:end], y_test[start:end])
        save_all_images(x_adv.cpu(), y_test[start:end], args.output, args.start_idx + start)
    print(f"Saved {n} adversarial images to {args.output} in {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
