import types
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from advsm._paths import imagenet_guided_diffusion_ckpt
from MimicDiffusion.load_model import load_models as md_load_models
from MimicDiffusion.purification import PurificationForward_mimic
from MimicDiffusion.utils import clf2diff, diff2clf


def _build_denoise_seq(timesteps_val: int, denoise_steps_val: int) -> list[int]:
    """Build ascending sequence of denoise_steps_val steps from 0 to timesteps_val-1.
    MimicDiffusion's denoising_process uses reversed(seq), so high t -> low t (noisy -> clean).
    """
    if denoise_steps_val <= 0:
        denoise_steps_val = 1
    if timesteps_val <= 1:
        return [0]
    seq = np.linspace(0, timesteps_val - 1, denoise_steps_val, dtype=np.int32).tolist()
    return [int(s) for s in seq]


class MimicDiffusionDefense:
    """
    Diffusion-based purifier using MimicDiffusion, with settings aligned to DDIM:

    - data:          'imagenet' or 'cifar10'
    - timesteps:     int (e.g. 1000) = total diffusion steps
    - denoise_steps: int (e.g. 100) = number of denoising steps in the schedule
    """

    def __init__(
        self,
        *,
        data: str,
        timesteps,
        denoise_steps,
        seed: int,
        device: torch.device,
    ) -> None:
        self.data = str(data)
        self.seed = seed
        self.device = device

        # Total diffusion steps (e.g. 1000)
        if isinstance(timesteps, (int, float)):
            T = int(timesteps)
        else:
            ts_list = list(timesteps)
            T = int(ts_list[0]) if ts_list else 1000

        # Build denoising schedule: seq = [0, ..., T-1] with N steps (e.g. 100 steps from 0 to 999)
        if isinstance(denoise_steps, (int, float)):
            N = int(denoise_steps)
            seq = _build_denoise_seq(T, N)
        else:
            ds_list = list(denoise_steps)
            if not ds_list:
                seq = _build_denoise_seq(T, 100)
            elif isinstance(ds_list[0], (int, float)):
                # Flat list of step indices
                seq = [int(x) for x in ds_list]
            else:
                # List of lists: use first inner list
                seq = [int(x) for x in list(ds_list[0])]
        if not seq:
            seq = _build_denoise_seq(T, 100)

        # One round of purification with this schedule (MimicDiffusion uses reversed(seq) = high t -> low t)
        max_timestep = [seq[-1]]
        attack_steps = [seq]

        args_ns = types.SimpleNamespace(dataset=self.data)
        if self.data != "imagenet":
            raise ValueError("MimicDiffusionDefense currently supports data='imagenet' only.")
        model_src = imagenet_guided_diffusion_ckpt()
        _clf, diffusion = md_load_models(args_ns, model_src, self.device)
        diffusion = diffusion.to(self.device).float()
        dummy_clf = nn.Identity().to(self.device).eval()

        self._purifier = PurificationForward_mimic(
            clf=dummy_clf,
            diffusion=diffusion,
            max_timestep=max_timestep,
            attack_steps=attack_steps,
            sampling_method="ddim",
            is_imagenet=(self.data == "imagenet"),
            device=self.device,
        ).to(self.device).eval()
    
    def init_hyperparam(self):
        seed = self.seed if self.seed is not None else time.time()
        # print(f"Defense Seed = {seed}")
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)
    
    def purify(self, imgs: Tensor) -> Tensor:
        x_in = imgs.to(self.device).float()
        _, _, H, W = x_in.shape
        self.init_hyperparam()
        # Purification uses noised input then denoise; guidance uses .sum() (patched) so batch is fine.
        ctx = torch.enable_grad() if imgs.requires_grad else torch.no_grad()
        # print(ctx)
        with ctx, torch.cuda.amp.autocast(enabled=False):
            if self.data == "imagenet":
                x_in = F.interpolate(x_in, size=(256, 256), mode="bilinear", align_corners=False)
            x_diff = clf2diff(x_in)
            t = self._purifier.max_timestep[0]
            noised_x = self._purifier.get_noised_x(x_diff, t)
            x_diff = self._purifier.denoising_process(
                noised_x, self._purifier.attack_steps[0], ref=x_diff
            )
            x_pur = diff2clf(x_diff)
        if (H, W) != x_pur.shape[-2:]:
            x_pur = F.interpolate(x_pur, size=(H, W), mode="bilinear", align_corners=False)
        return x_pur.clamp(0.0, 1.0).to(imgs.device, dtype=imgs.dtype)

