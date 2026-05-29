"""
ContrastDiff defense: diffusion purification with contrastive guidance (ContrastDiffPurification).
Uses SDE reverse sampling + ContrastiveGuidedDiffusion wrapper; same checkpoint as other defenses.
"""
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from advsm._paths import ensure_third_party_on_path, repo_root, third_party_root

ensure_third_party_on_path()
_CDP_SRC = os.path.join(third_party_root(), "ContrastDiffPurification", "src")
if _CDP_SRC not in sys.path:
    sys.path.insert(0, _CDP_SRC)

import torchsde
import yaml


class ContrastDiffDefense:
    """
    Contrastive-guided diffusion purification (ContrastDiffPurification).
    - data: 'imagenet' (and optionally 'cifar10' if supported later).
    - t: forward noise level (e.g. 150).
    - sample_step: number of reverse SDE iterations (e.g. 1).
    - contrastive_classifier: e.g. 'imagenet-resnet18'.
    - n_classes: 1000 for ImageNet.
    - optim_lr, drift_min, drift_max, tau_plus, beta, temperature: contrastive params.
    """

    def __init__(
        self,
        *,
        data: str,
        t: int,
        sample_step: int = 1,
        contrastive_classifier: str = "imagenet-resnet18",
        n_classes: int = 1000,
        optim_lr: float = 0.01,
        drift_min: int = 40,
        drift_max: int = 100000,
        tau_plus: float = 0.1,
        beta: float = 1.0,
        temperature: float = 1.0,
        use_bm: bool = False,
        seed: Optional[int] = None,
        device: torch.device = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = str(data).lower()
        self.device = device
        self.seed = seed
        self.t = int(t)
        self.sample_step = int(sample_step)
        self.use_bm = bool(use_bm)

        if self.data != "imagenet":
            raise ValueError("ContrastDiffDefense currently supports data='imagenet' only.")

        # Load config (ContrastDiffPurification uses data.dataset)
        cfg_path = os.path.join(_CDP_SRC, "configs", "imagenet.yml")
        with open(cfg_path, "r") as f:
            config_dict = yaml.safe_load(f)
        from ContrastDiffPurification.src.utils import dict2namespace
        config = dict2namespace(config_dict)

        # Build guided diffusion model (same checkpoint as other defenses)
        from ContrastDiffPurification.src.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        model_config["use_fp16"] = False
        model, _ = create_model_and_diffusion(**model_config)
        from advsm._paths import checkpoint_path

        model_src = checkpoint_path("guided_diffusion", "imagenet", "256x256_diffusion_uncond.pt")
        state = torch.load(model_src, map_location="cpu")
        model.load_state_dict(state)
        model = model.eval().to(self.device).float()

        # Contrastive wrapper (classifier for guidance)
        from ContrastDiffPurification.src.utils import get_image_classifier
        from ContrastDiffPurification.src.contrastive_guided_diffusion import ContrastiveGuidedDiffusion
        clf = get_image_classifier(contrastive_classifier).eval().to(self.device)
        img_shape = (3, 256, 256)
        self._contrastive_model = ContrastiveGuidedDiffusion(
            model,
            clf,
            n_classes=n_classes,
            batch_size=64,  # used only for contrastive loss; we pass variable batch
            img_shape=img_shape,
            optim_lr=optim_lr,
            enable_drift=True,
            drift_counter_min=drift_min,
            drift_counter_max=drift_max,
            tau_plus=tau_plus,
            beta=beta,
            temperature=temperature,
        ).to(self.device).eval()

        # RevVPSDE (reverse SDE with our wrapped model)
        from ContrastDiffPurification.src.runners.diffpure_sde import RevVPSDE
        self._rev_vpsde = RevVPSDE(
            model=self._contrastive_model,
            score_type="guided_diffusion",
            img_shape=img_shape,
        ).to(self.device)
        self._betas = self._rev_vpsde.discrete_betas.float().to(self.device)

    def init_hyperparam(self) -> None:
        seed = self.seed if self.seed is not None else time.time()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _image_editing_sample(self, img: Tensor) -> Tensor:
        """Single batch: img in [-1,1] (B,3,256,256). Returns (B,3,256,256) in [-1,1]."""
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))
        img = img.to(self.device)
        x0 = img
        xs = []
        for it in range(self.sample_step):
            e = torch.randn_like(x0, device=self.device)
            a = (1 - self._betas).cumprod(dim=0).to(self.device)
            total_noise_levels = self.t
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0 = 1 - self.t * 1.0 / 1000 + epsilon_dt0
            t1 = 1 - epsilon_dt1
            ts = torch.linspace(t0, t1, 2).to(self.device)
            x_ = x.view(batch_size, -1)
            if self.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
                xs_ = torchsde.sdeint_adjoint(self._rev_vpsde, x_, ts, method="euler", bm=bm)
            else:
                xs_ = torchsde.sdeint_adjoint(self._rev_vpsde, x_, ts, method="euler")
            x0 = xs_[-1].view(x.shape)
            xs.append(x0)
        out = torch.cat(xs, dim=0)
        if self.sample_step == 1:
            return out
        return out[-batch_size:].clone()  # last iteration only

    def purify(self, imgs: Tensor) -> Tensor:
        x_in = imgs.to(self.device).float()
        _, _, H, W = x_in.shape
        self.init_hyperparam()
        ctx = torch.enable_grad() if imgs.requires_grad else torch.no_grad()
        try:
            with ctx, torch.cuda.amp.autocast(enabled=False):
                x_proc = F.interpolate(x_in, size=(256, 256), mode="bilinear", align_corners=False)
                x_diff = (x_proc - 0.5) * 2.0
                x_pur = self._image_editing_sample(x_diff)
                if x_pur.shape[0] != x_in.shape[0]:
                    x_pur = x_pur[: x_in.shape[0]]
                x_pur = (x_pur / 2.0) + 0.5
            if hasattr(self._contrastive_model, "reset_drift_counter"):
                self._contrastive_model.reset_drift_counter()
            if (H, W) != x_pur.shape[-2:]:
                x_pur = F.interpolate(x_pur, size=(H, W), mode="bilinear", align_corners=False)
            return x_pur.clamp(0.0, 1.0).to(imgs.device, dtype=imgs.dtype)
        except ValueError as e:
            if "NaN loss" in str(e):
                import warnings
                warnings.warn("ContrastDiff: NaN contrastive loss; returning input unmodified for this batch.")
                return imgs.to(imgs.device, dtype=imgs.dtype)
            raise
