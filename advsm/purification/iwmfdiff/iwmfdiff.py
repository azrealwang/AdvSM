import os
import math
import time
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

# def imgs_resize(
#         imgs: Tensor,
#         shape: Tuple[int, int],
#     ) -> Tensor:
#     from torchvision.transforms import Resize
#     transform = Resize(shape,antialias=True)
#     imgs_resize = transform(imgs)
    
#     return imgs_resize

def restore_checkpoint(ckpt_dir, state, device):
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']

class IWMFDiff():
    def __init__(
            self,
            lambda_0: float = 0,
            s: int = 3,
            data: str = None, # option for celeba_hq, cifar10, imagenet_256
            denoise_steps: float = 20,
            timesteps: int = 1, # default 20
            batch_size: int = None,
            seed: int = None,
    ):
        self.lambda_0 = lambda_0
        self.data = data
        self.timesteps = timesteps
        self.denoise_steps = denoise_steps
        self.s = s
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def init_hyperparam(self):
        seed = self.seed if self.seed is not None else time.time()
        # print(f"Defense Seed = {seed}")
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)


    def purify(self, imgs: Tensor) -> Tensor:
        if self.lambda_0 > 0:
            imgs = self.iwmf(imgs)
        if self.timesteps > 1:
            #print("**********************denoising images******************************")
            imgs = self.ddrm(imgs)
        
        return imgs

    def iwmf(self, imgs: Tensor) -> Tensor:
        self.init_hyperparam()
        device_0 = imgs.device
        i, c, h, w = imgs.shape
        imgs_purified = imgs.clone().to(self.device)
        s_left = math.floor(self.s/2)
        s_right = math.ceil(self.s/2)
        for _ in range(int(self.lambda_0*h*w)):
            x = torch.randint(s_left,h-s_left,())
            y = torch.randint(s_left,w-s_left,())
            array_conv = imgs_purified[:,:,x-s_left:x+s_right,y-s_left:y+s_right].reshape(i,c,self.s*self.s)
            fixed_value = torch.mean(array_conv,dim=2,keepdim=True)
            imgs_purified[:,:,x-s_left:x+s_right,y-s_left:y+s_right] = fixed_value.unsqueeze(3).repeat(1,1,self.s,self.s)
        
        return imgs_purified.to(device_0)

    def ddrm(self, imgs: Tensor, exp: str = "models") -> Tensor:
        import yaml
        from .ddrm.functions.denoising import efficient_generalized_steps, get_beta_schedule, dict2namespace
        from .ddrm.functions.ckpt_util import get_ckpt_path, download
        from .ddrm.functions.svd_replacement import Denoising
        from .ddrm.models import diffusion
        from .ddrm.guided_diffusion.script_util import create_model
        from .ddrm.score_sde.losses import get_optimizer
        from .ddrm.score_sde.models import utils as mutils
        from .ddrm.score_sde.models.ema import ExponentialMovingAverage

        if self.batch_size is None:
            self.batch_size = len(imgs)
        self.init_hyperparam()
        assert self.data in ["celeba_hq", "cifar10_ddpm", "cifar10", "imagenet_256"]
        device_0 = imgs.device
        _, _, h_0, w_0 = imgs.shape
        # load config
        if self.data == 'celeba_hq':
            config_name: str = "celeba_hq.yml"
        elif self.data == 'cifar10_ddpm':
            config_name: str = "cifar10_ddpm.yml"
        elif self.data == 'cifar10':
            config_name: str = "cifar10.yml"
        elif self.data == 'imagenet_256':
            config_name: str = "imagenet_256.yml"
        else:
            raise ValueError
        with open(os.path.join(self.current_dir, "configs", config_name), "r") as f:
            config_tmp = yaml.safe_load(f)
        config = dict2namespace(config_tmp)
        config.device = self.device

        imgs = imgs.to(self.device)
        if w_0 != config.data.image_size:
            imgs = F.interpolate(imgs, size=(config.data.image_size, config.data.image_size), mode='bilinear', align_corners=False)
        
        # load betas
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float().to(self.device)    
        # Calculate alpha and cumulative product of alphas
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # Compute sigma = sqrt(1 - alpha_bar)
        sigma_y = torch.sqrt(1.0 - alpha_bars[self.timesteps-1])

        # other parameters
        # num_timesteps = betas.shape[0]
        skip = self.timesteps // self.denoise_steps
        seq = range(0, self.timesteps, skip)
        if self.data == 'celeba_hq':
            model = diffusion.Model(config)
            ckpt = os.path.join(exp, "ddpm/celaba/celeba_hq.ckpt")
            if not os.path.exists(ckpt):
                download('https://huggingface.co/gwang-kim/DiffusionCLIP-CelebA_HQ/resolve/main/celeba_hq.ckpt', ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            
        elif self.data == 'cifar10_ddpm':
            model = diffusion.Model(config)
            ckpt = get_ckpt_path(f"ema_cifar10", prefix=exp)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)

        elif self.data == 'cifar10':
            model_dir = 'models/score_sde'
            # print(f'model_config: {config}')
            model = mutils.create_model(config)
            optimizer = get_optimizer(config, model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(f'{model_dir}/cifar10/checkpoint_8.pth', state, self.device)
            ema.copy_to(model.parameters())
            model.to(self.device)

        elif self.data == 'imagenet_256':
            config_dict = vars(config.model)
            model = create_model(**config_dict)
            if config.model.use_fp16:
                model.convert_to_fp16()
            if config.model.class_cond:
                ckpt = os.path.join(exp, 'guided_diffusion/imagenet/%dx%d_diffusion.pt' % (config.data.image_size, config.data.image_size))
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (config.data.image_size, config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(exp, "guided_diffusion/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)

        else:
            raise ValueError
            
        # load deg
        H_funcs = Denoising(config.data.channels, config.data.image_size, self.device)
        
        # initial x 
        l = imgs.shape[0]
        x = torch.randn(
            l,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
            )
        
        # process denoising in batch
        for i in range(math.ceil(l/self.batch_size)):
            if i == math.ceil(l/self.batch_size)-1:
                x[i*self.batch_size:l], _ = efficient_generalized_steps(x[i*self.batch_size:l], seq, model, betas, H_funcs, imgs[i*self.batch_size:l], sigma_0=sigma_y, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None)
            else:
                x[i*self.batch_size:(i+1)*self.batch_size],_ = efficient_generalized_steps(x[i*self.batch_size:(i+1)*self.batch_size], seq, model, betas, H_funcs, imgs[i*self.batch_size:(i+1)*self.batch_size], sigma_0=sigma_y, etaB=1, etaA=0.85, etaC=0.85, cls_fn=None, classes=None)
        
        if w_0 != config.data.image_size:
            x = F.interpolate(x, size=(h_0, w_0), mode='bilinear', align_corners=False)

        return x.to(device_0)