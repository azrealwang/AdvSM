import os
import types
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from advsm._paths import checkpoint_path, third_party_root
from SSNI.utils import dict2namespace, clf2diff, diff2clf
from SSNI.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def _get_beta_schedule(beta_start: float = 1e-4, beta_end: float = 2e-2, num_steps: int = 1000) -> Tensor:
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    return torch.from_numpy(betas).float()


class SSNIDefense:
    """
    Simplified SSNI-style diffusion purifier.

    - Uses the SSNI guided-diffusion config/checkpoint.
    - Applies a single denoising schedule whose noise level is controlled by (adv_eps, tau, bias).
    - Interface matches other defenses: `imgs -> imgs_purified` in [0,1].
    """

    def __init__(
        self,
        *,
        data: str,
        timesteps,
        denoise_steps,
        tau: float,
        bias: float,
        adv_eps: float,
        seed: Optional[int],
        device: torch.device,
    ) -> None:
        self.data = str(data)
        self.device = device
        self.seed = seed

        # Scalar hyperparameters controlling noise injection
        self.tau = float(tau)
        if self.tau <= 0:
            # avoid division by zero; fall back to 1.0
            self.tau = 1.0
        self.bias = float(bias)
        self.adv_eps = float(adv_eps)

        # Normalize timesteps / denoise_steps to ints
        if isinstance(timesteps, (int, float)):
            self.T = int(timesteps)
        else:
            ts_list = list(timesteps)
            self.T = int(ts_list[0]) if ts_list else 1000

        if isinstance(denoise_steps, (int, float)):
            self.N = int(denoise_steps)
        else:
            ds_list = list(denoise_steps)
            if not ds_list:
                self.N = 100
            elif isinstance(ds_list[0], (int, float)):
                self.N = len(ds_list)
            else:
                self.N = len(list(ds_list[0]))
        if self.N <= 0:
            self.N = 1

        # Beta schedule & diffusion model
        self.betas = _get_beta_schedule().to(self.device)

        # Load SSNI guided-diffusion config and model, but force full float32 (no fp16)
        if self.data == "imagenet":
            cfg_path = os.path.join(third_party_root(), "SSNI", "configs", "imagenet.yml")
            model_src = checkpoint_path("guided_diffusion", "imagenet", "256x256_diffusion_uncond.pt")
        elif self.data == "cifar10":
            cfg_path = os.path.join(third_party_root(), "SSNI", "configs", "cifar10.yml")
            model_src = checkpoint_path("score_sde", "cifar10", "checkpoint_35.pth")
        else:
            raise ValueError(f"SSNIDefense currently supports 'imagenet' or 'cifar10', got {self.data!r}")

        import yaml

        with open(cfg_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config_ns = dict2namespace(config_dict)

        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config_ns.model))
        # Critical: disable fp16 to avoid dtype mismatches
        model_config["use_fp16"] = False

        diffusion, _ = create_model_and_diffusion(**model_config)
        state = torch.load(model_src, map_location="cpu")
        diffusion.load_state_dict(state)
        diffusion = diffusion.to(self.device).float().eval()

        self.diffusion = diffusion
        self.is_imagenet = (self.data == "imagenet")

    # ---- core diffusion helpers (mirroring SSNI PurificationModule/PurificationForward) ----

    def init_hyperparam(self) -> None:
        seed = self.seed if self.seed is not None else time.time()
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)

    def _compute_alpha(self, t: Tensor) -> Tensor:
        beta = torch.cat(
            [torch.zeros(1, device=self.betas.device), self.betas],
            dim=0,
        )
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def _get_noised_x(self, x: Tensor, t: int) -> Tensor:
        e = torch.randn_like(x)
        if isinstance(t, int):
            t = (torch.ones(x.shape[0], device=x.device) * t).long()
        else:
            t = t.to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        return x * a.sqrt() + e * (1.0 - a).sqrt()

    def _denoising_process(self, x: Tensor, seq: list[int]) -> Tensor:
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        eta = 0.0  # DDIM-style (matching other defenses)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n, device=x.device) * i)
            next_t = (torch.ones(n, device=x.device) * j)
            at = self._compute_alpha(t.long())
            at_next = self._compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt
    
    def purify(self, imgs: Tensor) -> Tensor:
        """
        Purify a batch of images using a SSNI-style noise scale:
        reweighted_t = sigmoid((adv_eps - eps_mu) / tau) * (T + bias),
        with eps_mu approximated as adv_eps / 2.
        """
        x_in = imgs.to(self.device).float()
        _, _, H, W = x_in.shape

        # Seed like other defenses (e.g. MimicDiffusion), but allow grad control via context.
        self.init_hyperparam()

        # Approximate dataset EPS mean; here we tie it to adv_eps
        eps_mu = 0.5 * self.adv_eps
        scaled = torch.sigmoid(
            torch.tensor((self.adv_eps - eps_mu) / self.tau, device=self.device)
        ) * (self.T + self.bias)
        t_max = int(torch.round(scaled).item())
        t_max = max(0, min(t_max, self.T - 1 if self.T > 1 else 0))

        if self.N <= 1 or t_max <= 0:
            seq = [0]
        else:
            seq = np.linspace(0, t_max, self.N, dtype=np.int32).tolist()
            seq = [int(s) for s in seq]

        ctx = torch.enable_grad() if imgs.requires_grad else torch.no_grad()
        with ctx, torch.cuda.amp.autocast(enabled=False):
            if self.is_imagenet:
                x_proc = F.interpolate(x_in, size=(256, 256), mode="bilinear", align_corners=False)
            else:
                x_proc = x_in
            x_diff = clf2diff(x_proc)
            noised_x = self._get_noised_x(x_diff, t_max)
            x_diff = self._denoising_process(noised_x, seq)
            x_pur = diff2clf(x_diff)

        if self.is_imagenet and (H, W) != x_pur.shape[-2:]:
            x_pur = F.interpolate(x_pur, size=(H, W), mode="bilinear", align_corners=False)
        return x_pur.clamp(0.0, 1.0).to(imgs.device, dtype=imgs.dtype)

