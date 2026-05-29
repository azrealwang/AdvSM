import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

@torch.no_grad()
def build_masks(
    x: torch.Tensor,          # [B,C,H,W]
    purifier,
    eps: float,
    thres: float,
    M: int = 5,
    N: int = 10,
    baseline_mode: str = "purify_mean",  # "purify_mean" or "identity"
    clamp: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    channelwise_noise: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Channel-wise masks for masked PGD: each mask is [B,C,H,W] bool.

    Baseline:
      y0 = mean_n purify(x)

    For each noise m:
      y_mean[m] = mean_n purify(x + noise_m)
      delta[m]  = y_mean[m] - y0

    Regions (per pixel/channel), mutually exclusive:
      purified:
        all m: |delta_m| <= thres

      smooth:
        all m: |delta_m| > thres AND sign(delta_m) all same
        AND sign(noise_m) is NOT all same across m (i.e., contains both + and -)

      sensitive (transfer | unstable):
        transfer: all m: |delta_m| > thres AND sign(delta_m) == sign(noise_m)
        unstable: everything else with |delta_m| > thres
    """
    assert x.ndim == 4, f"x must be [B,C,H,W], got {x.shape}"
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    def purify_runs(z: torch.Tensor, runs: int) -> torch.Tensor:
        outs = []
        for _ in range(runs):
            outs.append(purifier.purify(z))
        return torch.stack(outs, dim=0).to(z.device)  # [runs,*,C,H,W]

    # --- baseline: avg purified(x) OR identity ---
    # baseline_mode:
    #   - "purify_mean": y0 = mean_n purify(x)
    #   - "identity":   y0 = x (no purifier baseline)
    if baseline_mode == "purify_mean":
        y0 = purify_runs(x, N).mean(dim=0)  # [B,C,H,W]
    elif baseline_mode == "identity":
        y0 = x.clone()
    else:
        raise ValueError(f"baseline_mode must be 'purify_mean' or 'identity', got {baseline_mode!r}")

    # --- noises in {+eps, -eps} ---
    if channelwise_noise:
        noise_sign = torch.randint(0, 2, (B, M, C, H, W), device=device, dtype=torch.int8) * 2 - 1
    else:
        noise_sign = torch.randint(0, 2, (B, M, 1, H, W), device=device, dtype=torch.int8) * 2 - 1
        noise_sign = noise_sign.expand(B, M, C, H, W)

    noise = noise_sign.to(dtype=dtype) * eps                 # [B,M,C,H,W]
    x_noised = x[:, None, ...] + noise                       # [B,M,C,H,W]
    if clamp:
        x_noised = x_noised.clamp(clamp_min, clamp_max)

    # --- avg purified(x+noise) ---
    x_noised_flat = x_noised.reshape(B * M, C, H, W)          # [B*M,C,H,W]
    y_mean = purify_runs(x_noised_flat, N).mean(dim=0).reshape(B, M, C, H, W)

    # --- delta vs baseline ---
    delta = y_mean - y0[:, None, ...]                         # [B,M,C,H,W]
    abs_delta = delta.abs()
    s_delta = torch.sign(delta)                               # [-1,0,+1]
    s_noise = torch.sign(noise)                               # [-1,+1]

    purified = (abs_delta <= thres).all(dim=1)                # [B,C,H,W]

    all_large = (abs_delta > thres).all(dim=1)                # [B,C,H,W]

    smax = s_delta.max(dim=1).values
    smin = s_delta.min(dim=1).values
    same_delta_sign = (smax == smin)                          # [B,C,H,W]

    nmax = s_noise.max(dim=1).values
    nmin = s_noise.min(dim=1).values
    noise_diff = (nmax != nmin)                               # [B,C,H,W]

    sign_match = (s_delta == s_noise)                         # [B,M,C,H,W]
    transfer = all_large & sign_match.all(dim=1)              # [B,C,H,W]

    smooth = all_large & same_delta_sign & noise_diff         # [B,C,H,W]

    unstable = ~(transfer | smooth | purified)                # [B,C,H,W]
    sensitive = transfer | unstable                         # [B,C,H,W]

    return {
        "purified": purified,
        "smooth": smooth,
        "sensitive": sensitive,
    }

@torch.no_grad()
def build_rand_masks(
    x: torch.Tensor,          # [B,C,H,W]
    sparsity: float = 0.01,   # fraction of True entries (e.g., 0.01 = 1%)
    per_sample: bool = True,  # if False, same mask shared across batch
    clamp: bool = True,       # if True, ensure x is within [clamp_min, clamp_max] (optional)
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Random boolean mask with EXACT (or as exact as possible) sparsity.

    Returns:
        mask: bool tensor with same shape as x: [B,C,H,W]
              Exactly round(sparsity * numel_per_sample) True entries per sample
              (or per batch if per_sample=False).
    """
    assert x.ndim == 4, f"x must be [B,C,H,W], got {x.shape}"
    assert 0.0 <= sparsity <= 1.0, f"sparsity must be in [0,1], got {sparsity}"

    if clamp:
        _ = x.clamp(clamp_min, clamp_max)

    B, C, H, W = x.shape
    device = x.device

    if per_sample:
        # exact k per sample
        n = C * H * W
        k = int(round(sparsity * n))
        k = max(0, min(k, n))

        mask = torch.zeros((B, n), device=device, dtype=torch.bool)

        if k > 0:
            # sample k indices per sample without replacement (via topk over random scores)
            r = torch.rand((B, n), device=device, generator=generator)
            idx = torch.topk(r, k, dim=1, largest=True).indices  # [B,k]
            mask.scatter_(1, idx, True)

        return mask.view(B, C, H, W)

    else:
        # exact k over the whole batch tensor
        n = B * C * H * W
        k = int(round(sparsity * n))
        k = max(0, min(k, n))

        flat = torch.zeros((n,), device=device, dtype=torch.bool)
        if k > 0:
            r = torch.rand((n,), device=device, generator=generator)
            idx = torch.topk(r, k, dim=0, largest=True).indices  # [k]
            flat[idx] = True
        
        return flat.view(B, C, H, W)

@torch.no_grad()
def save_region_overlays(
    x: torch.Tensor,                    # [B,C,H,W]
    masks: Dict[str, torch.Tensor],     # each [B,1,H,W] OR [B,C,H,W] bool
    out_dir: str,
    b: int = 10,
    alpha: float = 1.0,
    overlay_scale: float = 1.0,
    prefix: str = "viz",
    indices: Optional[list] = None,
    normalize: bool = False,
    base_mode: str = "gray",            # "rgb" or "gray"
    mask_reduce: str = "any",           # for overlay only: "any" or "all" if masks are [B,C,H,W]
):
    """
    Overlay priority (highest wins): sensitive -> smooth -> purified
    Overlay is computed on [H,W] after reducing channel-wise masks.
    TXT scores are computed on FULL [C,H,W] (no reduction), as exclusive partition using same priority.
    """
    os.makedirs(out_dir, exist_ok=True)

    assert x.ndim == 4, f"x must be [B,C,H,W], got {x.shape}"
    B, C, H, W = x.shape
    assert C in (1, 3), "For visualization, C should be 1 or 3."
    assert base_mode in ("rgb", "gray"), f"base_mode must be 'rgb' or 'gray', got {base_mode}"
    assert mask_reduce in ("any", "all"), f"mask_reduce must be 'any' or 'all', got {mask_reduce}"

    COLORS = {
        "purified":  np.array([0,   255, 0  ], dtype=np.float32),  # green
        "smooth":    np.array([255, 255, 0  ], dtype=np.float32),  # yellow
        "sensitive": np.array([0,   128, 255], dtype=np.float32),  # blue
    }
    ALPHA_W = {
        "purified":  0.5,
        "smooth":    0.7,
        "sensitive": 0.4,
    }
    OVERLAY_ORDER = ["purified", "smooth", "sensitive"]

    if indices is None:
        indices = list(range(min(b, B)))
    else:
        indices = indices[:b]

    # checks
    for k in COLORS.keys():
        if k not in masks:
            raise KeyError(f"masks missing key '{k}'. Have: {list(masks.keys())}")
        mk = masks[k]
        if not (mk.ndim == 4 and mk.shape[0] == B and mk.shape[2] == H and mk.shape[3] == W):
            raise AssertionError(f"masks['{k}'] must be [B,1,H,W] or [B,C,H,W]; got {tuple(mk.shape)}")
        if mk.shape[1] not in (1, C):
            raise AssertionError(f"masks['{k}'] channel dim must be 1 or match x's C={C}; got {mk.shape[1]}")

    def to_uint8_base(img_chw: torch.Tensor) -> np.ndarray:
        img = img_chw.detach().float().cpu()
        if normalize:
            mn, mx = img.min(), img.max()
            img = (img - mn) / (mx - mn + 1e-12)
        img = img.clamp(0, 1)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if base_mode == "gray":
            r, g, b = img[0], img[1], img[2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            img = gray.unsqueeze(0).repeat(3, 1, 1)
        return (img.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)

    def reduce_to_hw(m_chw: torch.Tensor) -> np.ndarray:
        """m_chw is [1,H,W] or [C,H,W]. Return [H,W] bool for overlay."""
        m = m_chw.detach().cpu().bool()
        if m.ndim != 3:
            raise ValueError(f"Expected [*,H,W], got {tuple(m.shape)}")
        if m.shape[0] == 1:
            return m[0].numpy()
        if mask_reduce == "any":
            return m.any(dim=0).numpy()
        else:
            return m.all(dim=0).numpy()

    # pre-scale overlay colors for softer tint
    COLORS_SCALED = {k: (v * overlay_scale) for k, v in COLORS.items()}

    for idx in indices:
        base_u8 = to_uint8_base(x[idx])                 # [H,W,3] uint8
        base = base_u8.astype(np.float32)               # [H,W,3] float32

        # ---- overlay drawing (HW-based) ----
        out = base.copy().astype(np.float32)
        
        # Paint in order; later regions overwrite earlier ones.
        for region in OVERLAY_ORDER:
            m_hw = reduce_to_hw(masks[region][idx])  # [H,W] bool
            if not m_hw.any():
                continue
        
            a = float(alpha) * float(ALPHA_W.get(region, 1.0))
            a = max(0.0, min(1.0, a))  # clamp
        
            col = COLORS_SCALED[region]  # (3,)
            out[m_hw] = (1.0 - a) * out[m_hw] + a * col
        
        out = np.clip(out, 0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{prefix}_{idx:05d}.png")
        plt.imsave(path, out)

        # ---- scores on FULL [C,H,W] (or [1,H,W]) with exclusive partition ----
        def to_chw_bool(m_bchw: torch.Tensor) -> torch.Tensor:
            m = m_bchw.detach().bool()
            if m.shape[0] == 1 and C != 1:
                m = m.expand(C, H, W)
            return m

        m_sensitive = to_chw_bool(masks["sensitive"][idx])
        m_smooth = to_chw_bool(masks["smooth"][idx])
        m_purified = to_chw_bool(masks["purified"][idx])

        ex_sensitive = m_sensitive
        ex_smooth = m_smooth & ~ex_sensitive
        ex_purified = m_purified & ~(ex_sensitive | ex_smooth)

        denom = float(C * H * W)
        frac_sensitive = float(ex_sensitive.float().sum().item() / denom)
        frac_smooth = float(ex_smooth.float().sum().item() / denom)
        frac_purified = float(ex_purified.float().sum().item() / denom)

        meta_path = os.path.join(out_dir, f"{prefix}_{idx:05d}.txt")
        with open(meta_path, "w") as f:
            f.write(f"purified: {frac_purified:.6f}\n")
            f.write(f"smooth: {frac_smooth:.6f}\n")
            f.write(f"sensitive: {frac_sensitive:.6f}\n")
            f.write(f"sum: {(frac_purified + frac_smooth + frac_sensitive):.6f}\n")