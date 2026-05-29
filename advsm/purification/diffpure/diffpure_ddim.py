import numpy as np
import torch
import torch.nn.functional as F

import yaml
import os
import time

from advsm._paths import checkpoint_path

from .utils import dict2namespace
from .guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from .score_sde.losses import get_optimizer
from .score_sde.models import utils as mutils
from .score_sde.models.ema import ExponentialMovingAverage


def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [max_timesteps]
    num_denoising_steps_list = [num_denoising_steps]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps


class DiffPureDDIM():
    def __init__(
            self,
            data: str, # option for celeba_hq, cifar10, imagenet_256
            timesteps: int = 100,
            denoise_steps: int = 20,
            seed: int = None,
    ):  
        self.data = data
        self.timesteps, self.denoise_steps = get_diffusion_params(timesteps,denoise_steps)
        self.seed = seed
        self.eta = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.device = device
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        
        # Diffusion Model
        if data == 'cifar10':
            with open(os.path.join(current_dir, 'configs', 'cifar10.yml'), 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            model_dir = checkpoint_path("score_sde")
            # print(f'model_config: {config}')
            model = mutils.create_model(config)
            optimizer = get_optimizer(config, model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(f'{model_dir}/cifar10/checkpoint_8.pth', state, device)
            ema.copy_to(model.parameters())
            model.eval().to(device)
            
        elif data == 'imagenet':
            with open(os.path.join(current_dir, 'configs', 'imagenet.yml'), 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            model_ckpt = checkpoint_path("guided_diffusion", "imagenet", "256x256_diffusion_uncond.pt")
            model_config = model_and_diffusion_defaults()
            model_config.update(vars(config.model))
            model, _ = create_model_and_diffusion(**model_config)
            model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
            # model.requires_grad_(False).eval().to(self.device)
            model.eval().to(device)
            if model_config['use_fp16']:
                model.convert_to_fp16()

        else:
            raise NotImplementedError('unknown data type')
        
        self.config = config
        self.model = model
    
    def init_hyperparam(self):
        seed = self.seed if self.seed is not None else time.time()
        # print(f"Defense Seed = {seed}")
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        if self.return_mid:
            self.mid_x = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.model(xt, t)
            if self.data == 'imagenet':
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            if self.return_mid:
                self.mid_x.append(xt)

        return xt

    def purify(self, x, return_mid = False):
        self.init_hyperparam()
        self.return_mid = return_mid
        _, _, h_0, w_0 = x.shape
        x = x.to(self.device)
        ctx = torch.enable_grad() if x.requires_grad else torch.no_grad()
        # print(ctx)
        with ctx:
            if w_0 != self.config.data.image_size:
                x = F.interpolate(x, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)

            for i in range(len(self.timesteps)):
                noised_x = self.get_noised_x((x - 0.5) * 2, self.timesteps[i])
                
                if return_mid:
                    ori_x = []
                    for t_tmp in reversed(self.denoise_steps[i][:-1]):
                        ori_x.append(self.get_noised_x((x - 0.5) * 2, t_tmp))
                    ori_x.append(x)
                
                x_re = self.denoising_process(noised_x, self.denoise_steps[i])

            if w_0 != self.config.data.image_size:
                x_re = F.interpolate(x_re, size=(h_0, w_0), mode='bilinear', align_corners=False)

        if return_mid:
            return (x_re + 1) * 0.5, self.mid_x, ori_x
        else:
            return (x_re + 1) * 0.5
