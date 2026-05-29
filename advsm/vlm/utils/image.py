"""
Differentiable image preprocessing for attacks (no PIL in the gradient path).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_shortest_edge(images: torch.Tensor, shortest: int) -> torch.Tensor:
    """images: [B,3,H,W] in [0,1]. Resize so shorter side == shortest (bilinear)."""
    b, _, h, w = images.shape
    if h == w == shortest:
        return images
    if h < w:
        new_h = shortest
        new_w = int(round(w * (shortest / float(h))))
    else:
        new_w = shortest
        new_h = int(round(h * (shortest / float(w))))
    return F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)


def center_crop_square(images: torch.Tensor, crop: int) -> torch.Tensor:
    """Center crop to (crop, crop). images [B,3,H,W]."""
    _, _, h, w = images.shape
    if h < crop or w < crop:
        images = F.interpolate(images, size=(max(crop, h), max(crop, w)), mode="bilinear", align_corners=False)
        _, _, h, w = images.shape
    top = (h - crop) // 2
    left = (w - crop) // 2
    return images[:, :, top : top + crop, left : left + crop]


def clip_normalize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """images [B,3,H,W] after resize/crop in [0,1] range -> normalized."""
    mean = mean.view(1, 3, 1, 1).to(device=images.device, dtype=images.dtype)
    std = std.view(1, 3, 1, 1).to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def differentiable_clip_preprocess(
    images_01: torch.Tensor,
    shortest_edge: int,
    crop_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Match HF CLIPImageProcessor: resize shortest edge, center crop, scale to [0,1] then normalize.
    images_01 is already [0,1] RGB [B,3,H,W].
    """
    x = _resize_shortest_edge(images_01, shortest_edge)
    x = center_crop_square(x, crop_size)
    return clip_normalize(x, mean, std)


def pad_to_square(images_01: torch.Tensor, fill_value: float) -> torch.Tensor:
    """Pad [B,3,H,W] to square with constant fill (differentiable)."""
    _, _, h, w = images_01.shape
    if h == w:
        return images_01
    s = max(h, w)
    out = torch.full(
        (images_01.shape[0], 3, s, s),
        fill_value,
        device=images_01.device,
        dtype=images_01.dtype,
    )
    pad_h = (s - h) // 2
    pad_w = (s - w) // 2
    out[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = images_01
    return out


def llava_style_preprocess(
    images_01: torch.Tensor,
    image_aspect_ratio: str | None,
    shortest_edge: int,
    crop_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    pad_fill: float = 0.0,
) -> torch.Tensor:
    if image_aspect_ratio == "pad":
        x = pad_to_square(images_01, pad_fill)
        x = F.interpolate(x, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        return clip_normalize(x, mean, std)
    return differentiable_clip_preprocess(images_01, shortest_edge, crop_size, mean, std)


def openclip_default_mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype)
    return mean, std
