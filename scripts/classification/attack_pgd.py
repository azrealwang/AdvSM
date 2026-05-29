#!/usr/bin/env python3
"""Untargeted Linf PGD on paper ImageNet classifiers (Table 1)."""
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

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from advsm.classification.models import _PAPER_MODEL_KEYS, load_classifier
from advsm.classification.utils import load_samples, save_all_images

from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eps = args.eps / 255.0

    model = load_classifier(args.model)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )

    x_test, y_test = load_samples(args.input, args.start_idx, args.end_idx)
    y_test = y_test.astype(np.int64)

    attack = ProjectedGradientDescentPyTorch(
        estimator=classifier,
        eps=eps,
        max_iter=args.max_iter,
        targeted=False,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output, exist_ok=True)
    t0 = time()
    for i in range(len(y_test)):
        x = x_test[[i]]
        y = y_test[[i]]
        x_adv = attack.generate(x=x, y=y)
        save_all_images(Tensor(x_adv), Tensor(y), args.output, args.start_idx + i)
    print(f"Saved {len(y_test)} adversarial images to {args.output} in {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
