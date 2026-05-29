"""Repository root, ``checkpoints/``, and third-party path helpers."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def repo_root() -> str:
    return _REPO_ROOT


def checkpoints_dir() -> str:
    return os.path.join(_REPO_ROOT, "checkpoints")


def checkpoint_path(*parts: str) -> str:
    return os.path.join(checkpoints_dir(), *parts)


def third_party_root() -> str:
    return os.path.join(_REPO_ROOT, "third_party")


def ensure_repo_on_path() -> str:
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return _REPO_ROOT


def ensure_third_party_on_path() -> None:
    """Vendored purifiers and baseline attack code (DiffAttack, DiffHammer, DiffBreak)."""
    ensure_repo_on_path()
    tp = third_party_root()
    cdp_src = os.path.join(tp, "ContrastDiffPurification", "src")
    for p in (
        tp,
        cdp_src,
        os.path.join(tp, "DiffAttack"),
        os.path.join(tp, "DiffHammer"),
        os.path.join(tp, "DiffBreak"),
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
