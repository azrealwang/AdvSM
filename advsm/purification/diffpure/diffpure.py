# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import yaml
import os
import time
from argparse import Namespace

import torch
import torch.nn.functional as F

from .utils import dict2namespace
from .runners.diffpure_ddpm import Diffusion
from .runners.diffpure_guided import GuidedDiffusion
from .runners.diffpure_sde import RevGuidedDiffusion


class DiffPure():
    def __init__(
            self,
            data: str, # option for celeba_hq, cifar10, imagenet_256
            timesteps: int = 100,
            sample_step: int = 1,
            use_bm: bool = True,
            seed: int = None,
    ):  
        args = Namespace()
        args.t = timesteps
        args.sample_step = sample_step
        args.use_bm = use_bm
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # diffusion model
        if data == 'imagenet':
            args.score_type = 'guided_diffusion'
            with open(os.path.join(current_dir, 'configs', 'imagenet.yml'), 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            config.device = device
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif data == 'cifar10':
            args.score_type = 'score_sde'
            with open(os.path.join(current_dir, 'configs', 'cifar10.yml'), 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            config.device = device
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif data == 'celeba_hq':
            args.score_type = 'guided_diffusion'
            with open(os.path.join(current_dir, 'configs', 'celeba.yml'), 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            config.device = device
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown data type')
        
        self.args = args
        self.config = config
        self.tag = None
        self.seed = seed
        

    # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
    # and you may want to change the freq)
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def purify(self, x, return_mid = False):
        _, _, h_0, w_0 = x.shape
        counter = self.counter.item()
        # if counter % 5 == 0:
        #     print(f'diffusion times: {counter}')

        if w_0 != self.config.data.image_size:
            x = F.interpolate(x, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)

        if return_mid:
            x_re, mid_x, ori_x = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, return_mid=return_mid, seed=self.seed)
        else:
            x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, seed=self.seed)
        if w_0 != self.config.data.image_size:
            x_re = F.interpolate(x_re, size=(h_0, w_0), mode='bilinear', align_corners=False)

        # if counter % 5 == 0:
        #     print(f'x shape (before diffusion models): {x.shape}')
        #     print(f'x shape (before resnet): {x_re.shape}')

        self.counter += 1
        
        if return_mid:
            return (x_re + 1) * 0.5, mid_x, ori_x
        else:
            return (x_re + 1) * 0.5