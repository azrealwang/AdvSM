#!/usr/bin/env python3
"""Ensure repo root is on sys.path for `import advsm.vlm`."""
from pathlib import Path
import sys


def bootstrap():
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    sys.path.insert(0, str(repo_root))


bootstrap()
