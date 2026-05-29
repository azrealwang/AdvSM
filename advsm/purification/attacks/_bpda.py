import torch
import torch.nn as nn
from typing import Optional, Any


class BPDAModel(nn.Module):
    """
    Wrap a classifier with an optional purifier, using STE/BPDA if requested.

    - If purifier is None: logits = model(x)
    - If bpda_mode == "none": logits = model(purifier(x)) and backprop through purifier
    - If bpda_mode == "ste": forward uses purifier(x.detach()), backward treats purifier as identity
    """

    def __init__(
        self,
        model: nn.Module,
        purifier: Optional[Any] = None,
        *,
        bpda_mode: str = "ste",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.purifier = purifier
        self.bpda_mode = str(bpda_mode).lower()
        if self.bpda_mode not in ("ste", "none", "skip"):
            raise ValueError("bpda_mode must be one of: 'ste', 'none', 'skip'")
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def _apply_defense(self, x: torch.Tensor) -> torch.Tensor:
        if self.purifier is None or self.bpda_mode == "skip":
            return x
        if self.bpda_mode == "none":
            return self.purifier.purify(x)
        with torch.no_grad():
            x_p = self.purifier.purify(x.detach())
        return x_p + (x - x.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.clip_min, self.clip_max)
        x_p = self._apply_defense(x)
        return self.model(x_p)

