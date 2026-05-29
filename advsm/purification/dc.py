import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import yaml

from advsm._paths import imagenet_guided_diffusion_ckpt, third_party_root
from DC.utils import clf2diff, diff2clf
from DC.purification import PurificationForward, get_ddim_steps, get_beta_schedule, extract
from DC.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class DCCore(torch.nn.Module):
    """
    Thin, classifier-free variant of DC's PurificationForward.

    - Keeps DC's DDIM-style schedule and posterior update.
    - Drops classifier-based attention masks; instead, uses a uniform mask (no spatial selection),
      so behavior reduces to DC's diffusion manipulation without per-region filtering.
    """

    def __init__(
        self,
        *,
        data: str,
        strength_l: float,
        strength_s: float,
        forward_noise_steps: int,
        ddim_steps: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.is_imagenet = (data == "imagenet")

        # Diffusion hyperparams (mirroring DC script/README names)
        # strength_l: large-scale strength, strength_s: small-scale strength
        self.strength_l = float(strength_l)
        self.strength_s = float(strength_s)
        self.forward_noise_steps = int(forward_noise_steps)
        self.num_train_timesteps = 1000

        # Build DDIM timestep schedule as in DC
        self.ddim_steps = int(ddim_steps)
        self.timesteps = get_ddim_steps(self.num_train_timesteps, self.ddim_steps, self.strength_l)
        self.eta = 0.0

        betas = get_beta_schedule(1e-4, 2e-2, 1000)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def diffuse_t_steps(self, x0: Tensor, t: int) -> Tensor:
        alpha_bar = self.alphas_cumprod[t]
        noise = torch.randn_like(x0, device=self.device)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def diffuse_one_step(self, x: Tensor, t: Tensor) -> Tensor:
        noise = torch.randn_like(x, device=self.device)
        return extract(self.sqrt_alphas, t, x.shape) * x + extract(self.sqrt_one_minus_alphas, t, x.shape) * noise

    def diffuse_one_step_from_now(self, x_t: Tensor, t: int, steps: int) -> tuple[Tensor, int]:
        n = x_t.shape[0]
        for i in range(steps):
            x_t = self.diffuse_one_step(x_t, (torch.ones(n, device=self.device) * (t + i + 1)))
        return x_t, t + steps

    def denoising_step(self, x: Tensor, t: int, diffusion: nn.Module) -> tuple[Tensor, Tensor]:
        n = x.shape[0]
        t_tensor = (torch.ones(n, device=self.device) * t)
        model_output = diffusion(x, t_tensor)
        if self.is_imagenet:
            model_output, _ = torch.split(model_output, 3, dim=1)

        pred_xstart = (
            extract(self.sqrt_recip_alphas_cumprod, t_tensor, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t_tensor, x.shape) * model_output
        )
        pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)

        mean = (
            extract(self.posterior_mean_coef1, t_tensor, x.shape) * pred_xstart
            + extract(self.posterior_mean_coef2, t_tensor, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t_tensor, x.shape)

        noise = torch.randn_like(x, device=self.device)
        mask = (t_tensor != 0).float().view(-1, *([1] * (x.dim() - 1)))
        sample = mean + mask * torch.sqrt(posterior_variance) * noise
        return pred_xstart.float(), sample.float()

    def forward(self, x: Tensor, diffusion: nn.Module) -> Tensor:
        """
        Core DC denoising path, but with a uniform mask (no classifier-guided attention).
        """
        # All-pure mask (no region dropping)
        B, _, H, W = x.shape
        mask = torch.ones(B, 1, H, W, device=self.device)

        time_steps_b = self.strength_s * self.num_train_timesteps
        n = x.shape[0]

        x_t = self.diffuse_t_steps(x, int(self.timesteps[0]))
        for t, tau in list(zip(self.timesteps[:-1], self.timesteps[1:])):
            if not np.isclose(self.eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - self.alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - self.alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - self.alphas[t]
                sigma_t = self.eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
                sigma_t = torch.tensor(sigma_t, device=self.device)
            else:
                sigma_t = torch.zeros(1, device=self.device)

            if tau >= time_steps_b:
                x_t_ori = self.diffuse_t_steps(x, t)
                x_t = x_t * mask + x_t_ori * (1.0 - mask)
                x_t, t = self.diffuse_one_step_from_now(x_t, t, steps=self.forward_noise_steps)

            # DDIM sampling
            pred_noise = diffusion(x_t, (torch.ones(n, device=self.device) * t))
            if self.is_imagenet:
                pred_noise, _ = torch.split(pred_noise, 3, dim=1)

            alphas_cumprod_tau = extract(self.alphas_cumprod, (torch.ones(n, device=self.device) * tau), x.shape)
            sqrt_alphas_cumprod_tau = torch.sqrt(alphas_cumprod_tau)
            alphas_cumprod_t = extract(self.alphas_cumprod, (torch.ones(n, device=self.device) * t), x.shape)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)

            first_term = sqrt_alphas_cumprod_tau * (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            coeff = torch.sqrt(torch.clamp(1.0 - alphas_cumprod_tau - sigma_t**2, min=0.0))
            second_term = coeff * pred_noise
            x_t = first_term + second_term

        return x_t


class DCDefense:
    """
    DC defense wrapper using the current guided-diffusion model.

    - Shares the same backbone weights as your other defenses.
    - Follows DC's DDIM-like schedule and forward/denoise procedure (without classifier attention masks).
    """

    def __init__(
        self,
        *,
        data: str,
        strength_l: float,
        strength_s: float,
        forward_noise_steps: int,
        ddim_steps: int,
        seed: Optional[int],
        device: torch.device,
    ) -> None:
        self.data = str(data)
        self.device = device
        self.seed = seed

        # Load guided-diffusion model (same as other defenses; force float32).
        # For ImageNet, DC uses a guided_diffusion UNet whose config is in diffusion_configs/imagenet.yml.
        if self.data == "imagenet":
            cfg_path = os.path.join(third_party_root(), "DC", "diffusion_configs", "imagenet.yml")
            model_src = imagenet_guided_diffusion_ckpt()

            with open(cfg_path, "r") as f:
                config_dict = yaml.safe_load(f)

            model_config = model_and_diffusion_defaults()
            # Align UNet architecture with DC's imagenet config.
            model_config.update(config_dict["model"])
            # Ensure fp32 for consistency with other defenses.
            model_config["use_fp16"] = False

            diffusion, _ = create_model_and_diffusion(**model_config)
            state = torch.load(model_src, map_location="cpu")
            diffusion.load_state_dict(state)
            diffusion = diffusion.to(self.device).float().eval()
        elif self.data == "cifar10":
            # DC's CIFAR-10 pipeline uses a score_sde model, not guided_diffusion.
            # Supporting that end-to-end would require a separate loading path; for now we
            # restrict this wrapper to ImageNet to avoid state_dict shape mismatches.
            raise ValueError("DCDefense wrapper currently supports ImageNet only (guided_diffusion backbone).")
        else:
            raise ValueError(f"DCDefense currently supports 'imagenet', got {self.data!r}")

        self.diffusion = diffusion
        self.is_imagenet = (self.data == "imagenet")

        self.core = DCCore(
            data=self.data,
            strength_l=strength_l,
            strength_s=strength_s,
            forward_noise_steps=forward_noise_steps,
            ddim_steps=ddim_steps,
            device=self.device,
        ).to(self.device)

    def init_hyperparam(self) -> None:
        seed = self.seed if self.seed is not None else time.time()
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)

    def purify(self, imgs: Tensor) -> Tensor:
        x_in = imgs.to(self.device).float()
        B, _, H, W = x_in.shape

        self.init_hyperparam()

        ctx = torch.enable_grad() if imgs.requires_grad else torch.no_grad()
        with ctx, torch.cuda.amp.autocast(enabled=False):
            if self.is_imagenet:
                x_proc = F.interpolate(x_in, size=(256, 256), mode="bilinear", align_corners=False)
            else:
                x_proc = x_in
            x_diff = clf2diff(x_proc)
            x_diff = self.core(x_diff, self.diffusion)
            x_pur = diff2clf(x_diff)

        if self.is_imagenet and (H, W) != x_pur.shape[-2:]:
            x_pur = F.interpolate(x_pur, size=(H, W), mode="bilinear", align_corners=False)
        return x_pur.clamp(0.0, 1.0).to(imgs.device, dtype=imgs.dtype)

