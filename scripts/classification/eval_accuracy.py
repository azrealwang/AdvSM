#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import _path  # noqa: F401

import argparse
import torch

from advsm.classification.models import _PAPER_MODEL_KEYS, load_classifier
from advsm.classification.utils import load_samples, predict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True, choices=_PAPER_MODEL_KEYS)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target = load_classifier(args.target)
    x_test, y_test = load_samples(args.input, args.start_idx, args.end_idx)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    bs = args.batch_size or len(y_test)
    preds = predict(target, x_test, batch_size=bs)
    acc = (preds.max(1)[1] == y_test.long()).float().mean()
    print(f"Accuracy: {acc.item():.4f}")
