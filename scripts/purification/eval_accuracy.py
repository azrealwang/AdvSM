#!/usr/bin/env python3
"""Accuracy of a classifier behind an optional purifier (paper purification eval)."""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from advsm._paths import ensure_repo_on_path, ensure_third_party_on_path

ensure_repo_on_path()
ensure_third_party_on_path()

import argparse

import torch
from torch import Tensor

from advsm.classification.utils import load_one_model, load_samples, predict, smart_cast
from advsm.purification import Purifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Load images, optionally purify, classify, report accuracy."
    )
    p.add_argument("--data", type=str, required=True, choices=("cifar10", "imagenet"))
    p.add_argument("--target", type=str, required=True, help="Classifier (paper name or RobustBench id)")
    p.add_argument("--input", type=str, required=True, help="Directory of NNNNN_label.png images")
    p.add_argument("--defense", type=str, default=None, help="Purifier name (omit for raw images)")
    p.add_argument(
        "--d_settings",
        nargs="+",
        action="append",
        metavar=("NAME", "VAL"),
        default=None,
        help="Purifier kwargs as name value pairs, e.g. data imagenet timesteps 150 denoise_steps 10",
    )
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d_settings = {k: smart_cast(v) for k, v in args.d_settings} if args.d_settings else {}

    target = load_one_model(args.data, args.target, threat_model="Linf").eval()
    x_test, y_test = load_samples(args.input, args.start_idx, args.end_idx)
    x_test = Tensor(x_test)
    y_test = Tensor(y_test).long()
    batch_size = args.batch_size or len(y_test)

    if args.defense:
        purifier = Purifier(args.defense, d_settings, batch_size, args.seed)
        x_purified = []
        for start in range(0, len(y_test), batch_size):
            idx = slice(start, min(start + batch_size, len(y_test)))
            x_purified.append(purifier.purify(x_test[idx]).detach().cpu())
        x_eval = torch.cat(x_purified, 0)
    else:
        x_eval = x_test

    predictions = predict(target, x_eval, batch_size=batch_size)
    acc = (predictions.max(1)[1] == y_test).float().mean()
    label = "Robust accuracy" if args.defense else "Accuracy"
    print(f"{label}: {acc:.4f}")


if __name__ == "__main__":
    main()
