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


def require_checkpoint(*parts: str) -> str:
    """Return ``checkpoints/<parts>`` and raise if the file is missing."""
    path = checkpoint_path(*parts)
    if os.path.isfile(path):
        return path
    hint = "Run from the repository root, e.g.\n  bash scripts/download_guided_diffusion.sh"
    if parts and parts[0] == "score_sde":
        hint += "\n  (CIFAR score_sde weights go under checkpoints/score_sde/cifar10/)"
    raise FileNotFoundError(f"Missing checkpoint: {path}\n{hint}")


def imagenet_guided_diffusion_ckpt() -> str:
    """OpenAI 256×256 unconditional diffusion (DiffPure, DDIM, SSNI, DC, MimicDiffusion, ContrastDiff)."""
    return require_checkpoint("guided_diffusion", "imagenet", "256x256_diffusion_uncond.pt")


def cifar10_score_sde_ckpt(filename: str = "checkpoint_8.pth") -> str:
    """Score-SDE checkpoint for CIFAR-10 diffpure runners (default: checkpoint_8.pth)."""
    return require_checkpoint("score_sde", "cifar10", filename)


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
