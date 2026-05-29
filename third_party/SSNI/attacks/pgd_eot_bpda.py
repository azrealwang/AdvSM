import numpy as np
import torch
import torch.nn.functional as F
from eps_standard import load_dataset_eps
from eps_calculation import reweight_t
from tqdm import tqdm


class BPDA:
    def __init__(self, get_logit, attack_steps=200, eps=8./255., step_size=0.007, target=None, eot=20):
        self.target = target
        self.clamp = (0,1)
        self.eps = eps
        self.step_size = step_size
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        for _ in tqdm(range(self.attack_steps), desc="BPDA+EOT Attacking:"):
            grad = torch.zeros_like(x_adv)
            
            for _ in range(self.eot):
                with torch.no_grad():
                    preprocessed_x = self.get_logit.preprocess(x_adv)
                preprocessed_x.requires_grad = True
                
                # Classification
                logits = self.get_logit.classify(preprocessed_x)
                
                # Calculate loss
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                grad += torch.autograd.grad(loss, [preprocessed_x])[0].detach()
                x_adv = x_adv.detach()

            grad /= self.eot
            grad = grad.sign()
            x_adv = x_adv + self.step_size * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv
    

class Whitebox_BPDA:
    def __init__(self, diffusion, args, config, get_logit, attack_steps=100, eps=8./255., step_size=0.007, target=None, eot=20, is_linear=False):
        self.target = target
        self.clamp = (0,1)
        self.eps = eps
        self.step_size = step_size
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot
        self.diffusion = diffusion
        self.args = args
        self.config = config
        self.is_linear = is_linear

        self.eps_standard = load_dataset_eps(self.args)

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        for _ in tqdm(range(self.attack_steps), desc="Adaptive BPDA+EOT Attacking:"):
            grad = torch.zeros_like(x_adv)

            reweight_range, adv_eps = reweight_t(x_adv, self.diffusion, self.eps_standard, self.args, self.config)
            
            for _ in range(self.eot):
                with torch.no_grad():
                    if self.is_linear:
                        preprocessed_x = self.get_logit.preprocess(x_adv, reweight_range, adv_eps)
                    else:
                        preprocessed_x = self.get_logit.preprocess(x_adv, self.eps_standard, adv_eps)
                preprocessed_x.requires_grad = True
                
                # Classification
                logits = self.get_logit.classify(preprocessed_x)
                
                # Calculate loss
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                grad += torch.autograd.grad(loss, [preprocessed_x])[0].detach()
                x_adv = x_adv.detach()

            grad /= self.eot
            grad = grad.sign()
            x_adv = x_adv + self.step_size * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv