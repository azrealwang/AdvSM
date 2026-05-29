#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import _path  # noqa: F401

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Print first rows of evaluation CSVs.")
    p.add_argument("csv_path", type=str)
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    path = Path(args.csv_path)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= args.n:
                break
            print(row)


if __name__ == "__main__":
    main()
