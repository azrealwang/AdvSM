"""
FARE-style robust CLIP visual encoder via OpenCLIP ViT.
Checkpoint must be compatible with the OpenCLIP visual trunk after removing the last transformer block.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_openclip_arch(vision_arch: str) -> tuple[str, str]:
    va = vision_arch.strip()
    mapping = {
        "ViT-L-14": ("ViT-L-14", "openai"),
        "ViT-L-14-336": ("ViT-L-14-336", "openai"),
        "ViT-B-32": ("ViT-B-32", "openai"),
        "ViT-B-16": ("ViT-B-16", "openai"),
    }
    if va not in mapping:
        raise ValueError(
            f"Unsupported vision_arch for FARE/OpenCLIP loader: {vision_arch!r}. "
            f"Supported: {list(mapping)}."
        )
    return mapping[va]


def _openclip_model_name_for_resolution(vision_arch: str, clip_input_resolution: int | None) -> tuple[str, str]:
    """Use ViT-L-14-336 when LLaVA feeds ~336px crops so patch grid matches positional embeddings."""
    model_name, pretrained = _parse_openclip_arch(vision_arch)
    if clip_input_resolution is None:
        return model_name, pretrained
    if vision_arch.strip() == "ViT-L-14" and clip_input_resolution >= 320:
        return "ViT-L-14-336", pretrained
    return model_name, pretrained


def _interpolate_openclip_positional_embedding(pos_embed: torch.Tensor, new_side: int) -> torch.Tensor:
    """Resize CLS + spatial PE from a sqrt(N)-sided grid to new_side x new_side (OpenCLIP ViT layout)."""
    ntok, c = pos_embed.shape
    if ntok == new_side * new_side + 1:
        return pos_embed
    cls = pos_embed[0:1]
    patches = pos_embed[1:]
    old_side = int(round(math.sqrt(patches.shape[0])))
    if old_side * old_side != patches.shape[0]:
        raise ValueError(f"Cannot reshape positional_embedding length {patches.shape[0]} to a square grid.")
    x = patches.reshape(old_side, old_side, c).permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(x.float(), size=(new_side, new_side), mode="bicubic", align_corners=False).to(dtype=pos_embed.dtype)
    y = y.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, c)
    return torch.cat([cls, y], dim=0)


class OpenClipTokenVisual(nn.Module):
    """
    Wraps an OpenCLIP ViT `visual` with the last residual block removed (Robust-LLaVA convention)
    and returns patch(+cls) tokens [B, N, C].
    """

    def __init__(self, visual: nn.Module, token_dim: int):
        super().__init__()
        self.visual = visual
        self.token_dim = int(token_dim)

    def forward(self, images_normalized: torch.Tensor) -> torch.Tensor:
        if hasattr(self.visual, "output_tokens"):
            self.visual.output_tokens = True
        out = self.visual(images_normalized)
        if isinstance(out, tuple):
            a, b = out[0], out[1]
            flat = torch.cat([a.flatten(1), b.flatten(1)], dim=1)
        else:
            flat = out if out.dim() == 2 else out.flatten(1)
        bsz, fd = flat.shape
        if fd % self.token_dim != 0:
            raise RuntimeError(
                f"Encoder output feature dim {fd} is not divisible by token_dim={self.token_dim}. "
                "Checkpoint / architecture mismatch."
            )
        ntok = fd // self.token_dim
        return flat.view(bsz, ntok, self.token_dim)

    @property
    def hidden_size(self) -> int:
        return self.token_dim


def build_fare_visual(
    vision_arch: str,
    checkpoint: str | Path,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
    clip_input_resolution: int | None = None,
) -> OpenClipTokenVisual:
    import open_clip

    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"FARE encoder checkpoint not found: {ckpt}. "
            "Provide a valid encoder_checkpoint trained for the chosen OpenCLIP architecture."
        )
    model_name, pretrained = _openclip_model_name_for_resolution(vision_arch, clip_input_resolution)
    kwargs = {}
    if pretrained == "openai":
        kwargs["force_quick_gelu"] = True
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, **kwargs)
    except TypeError:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    visual = model.visual
    if hasattr(visual, "proj") and visual.proj is not None:
        visual.proj = None
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        if len(blocks) < 2:
            raise RuntimeError("OpenCLIP ViT has too few blocks to drop the last one.")
        visual.transformer.resblocks = blocks[:-1]
    token_dim = int(getattr(visual, "width", 1024))
    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        stripped = {}
        for k, v in sd.items():
            nk = k
            for pref in ("module.", "model.", "visual.", "model.visual."):
                while nk.startswith(pref):
                    nk = nk[len(pref) :]
            stripped[nk] = v
        sd = stripped
    pe_key = "positional_embedding"
    if pe_key in sd and sd[pe_key].shape[0] != visual.positional_embedding.shape[0]:
        if clip_input_resolution is None:
            raise RuntimeError(
                f"Checkpoint positional_embedding length {sd[pe_key].shape[0]} does not match OpenCLIP visual "
                f"({visual.positional_embedding.shape[0]}). Pass clip_input_resolution (LLaVA crop size) when "
                "using encoder_replace with 336px LLaVA and 224px RobustVLM/FARE weights."
            )
        ps = getattr(visual, "patch_size", 14)
        patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        new_side = max(1, clip_input_resolution // patch)
        sd = {**sd, pe_key: _interpolate_openclip_positional_embedding(sd[pe_key], new_side)}
    visual.load_state_dict(sd, strict=False)
    enc = OpenClipTokenVisual(visual, token_dim)
    enc = enc.to(device=device, dtype=dtype)
    return enc
