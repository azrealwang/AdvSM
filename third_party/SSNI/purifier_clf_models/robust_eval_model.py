import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from utils import diff2clf, clf2diff
import torchvision.utils as tvu
import os



def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()

class PurificationModule(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        assert sampling_method in ['ddim', 'ddpm']
        self.eta = 0 if sampling_method == 'ddim' else 1
        self.is_imagenet = is_imagenet

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def classify(self, x):
        logits = self.clf(x)
        return logits

    def get_noised_x(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denoising_process(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def preprocess(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def forward(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    

class LinearPurificationForward(PurificationModule):
    def __init__(self, args, config, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__(clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device)
        self.args = args
        self.config = config


    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        assert x.shape[0] == t.shape[0]
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x
    

    def denoising_process(self, x, seq):
        assert x.size(0) == seq.shape[0]
        n = x.size(0)
        # seq_next = [-1] + list(seq[:-1])
        minus = -1 * torch.ones((seq.size(0), 1)).long()
        seq = seq.to(minus.device) # load to cpu
        seq_next = torch.cat((minus, seq[:,:-1]), dim=1)
        xt = x

        for i, j in zip(torch.flip(seq, dims=[1]).T, torch.flip(seq_next, dims=[1]).T):
            t = i.to(x.device)
            next_t = j.to(x.device)
            assert t.size(0) == next_t.size(0)
            assert t.size(0) == n
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt


    def preprocess(self, x, reweight_range, adv_eps):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        if self.args.adaptive_defense_eval or self.args.whitebox_defense_eval:
            att_num_denoising_steps = [int(i) for i in self.args.def_num_denoising_steps.split(',')]
        else:
            att_num_denoising_steps = [int(i) for i in self.args.att_num_denoising_steps.split(',')]
        for i in range(len(self.max_timestep)):
            target_range = self.max_timestep[i]
            (max_eps, min_eps) = reweight_range
            zoom_ratio = target_range / (max_eps - min_eps)
            bias = self.args.bias
            reweighted_t = (adv_eps - min_eps) * zoom_ratio + (self.max_timestep[i]/2 - target_range/2 + bias)
            reweighted_t = torch.round(reweighted_t).to(torch.int64).to(x_diff.device)
            self.attack_steps[i] = get_diffusion_params(reweighted_t, att_num_denoising_steps[i])
            reweighted_t = torch.where((reweighted_t-1) < 0, torch.tensor(0, dtype=torch.int64), (reweighted_t-1))
            noised_x = self.get_noised_x(x_diff, reweighted_t)
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf
    
    
    def forward(self, x, reweight_range, adv_eps):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
            
        x_diff = clf2diff(x)
        if self.args.adaptive_defense_eval or self.args.whitebox_defense_eval:
            att_num_denoising_steps = [int(i) for i in self.args.def_num_denoising_steps.split(',')]
        else:
            att_num_denoising_steps = [int(i) for i in self.args.att_num_denoising_steps.split(',')]
        for i in range(len(self.max_timestep)):
            target_range = self.max_timestep[i]
            (max_eps, min_eps) = reweight_range
            zoom_ratio = target_range / (max_eps - min_eps)
            bias = self.args.bias
            reweighted_t = (adv_eps - min_eps) * zoom_ratio + (self.max_timestep[i]/2 - target_range/2 + bias)
            reweighted_t = torch.round(reweighted_t).to(torch.int64).to(x_diff.device)
            self.attack_steps[i] = get_diffusion_params(reweighted_t, att_num_denoising_steps[i])
            reweighted_t = torch.where((reweighted_t-1) < 0, torch.tensor(0, dtype=torch.int64), (reweighted_t-1))
            if self.args.show_t:
                print(reweighted_t)
            noised_x = self.get_noised_x(x_diff, reweighted_t)
            x_diff = self.denoising_process(noised_x, (self.attack_steps[i] + bias))
        # classifier part
        if self.is_imagenet:
            x_clf = diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False))
        else:
            x_clf = diff2clf(x_diff)
        if self.args.save_purify_img:
            torch.save(x_clf, os.path.join(self.args.adv_data_dir, "adv_data", f"purified_examples_batch_{self.args.batch_seed}.pt"))
        logits = self.clf(x_clf)
        return logits



class PurificationForward(PurificationModule):
    def __init__(self, args, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__(clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device)
        self.args = args

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        assert x.shape[0] == t.shape[0] 
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x


    def denoising_process(self, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt
    
    
    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf
    

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])

            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            # x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
            #     224, 224), mode='bilinear', align_corners=False)))
            x_clf = diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False))
        else:
            x_clf = diff2clf(x_diff)

        if self.args.save_purify_img:
            torch.save(x_clf, os.path.join(self.args.adv_data_dir, "adv_data", f"purified_examples_batch_{self.args.batch_seed}.pt"))
        logits = self.clf(x_clf)
        return logits
    



class AdaptivePurificationForward(PurificationModule):
    def __init__(self, args, config, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device):
        super().__init__(clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device)
        self.args = args
        self.config = config


    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        assert x.shape[0] == t.shape[0] 
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x
    

    def denoising_process(self, x, seq):
        assert x.size(0) == seq.shape[0]
        n = x.size(0)
        # seq_next = [-1] + list(seq[:-1])
        minus = -1 * torch.ones((seq.size(0), 1)).long()
        seq = seq.to(minus.device) # load to cpu
        seq_next = torch.cat((minus, seq[:,:-1]), dim=1)
        xt = x

        for i, j in zip(torch.flip(seq, dims=[1]).T, torch.flip(seq_next, dims=[1]).T):
            t = i.to(x.device)
            next_t = j.to(x.device)
            assert t.size(0) == next_t.size(0)
            assert t.size(0) == n
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt


    def preprocess(self, x, eps_standard, adv_eps):
        eps_standard = torch.tensor(eps_standard)
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
            eps_mu = torch.mean(eps_standard)
        else:
            # eps_mu = torch.max(eps_standard)
            eps_mu = torch.mean(eps_standard)
        x_diff = clf2diff(x)
        if self.args.adaptive_defense_eval or self.args.whitebox_defense_eval:
            num_denoising_steps = [int(i) for i in self.args.def_num_denoising_steps.split(',')]
        else:
            num_denoising_steps = [int(i) for i in self.args.att_num_denoising_steps.split(',')]

        for i in range(len(self.max_timestep)):
            tau = self.args.tau
            bias = self.args.bias
            reweighted_t = torch.sigmoid(input=((adv_eps - eps_mu)/tau)) * (self.max_timestep[i] + bias)
            reweighted_t = torch.round(reweighted_t).to(torch.int64).to(x_diff.device)
            self.attack_steps[i] = get_diffusion_params(reweighted_t, (num_denoising_steps[i] + bias))
            reweighted_t = torch.where((reweighted_t-1) < 0, torch.tensor(0, dtype=torch.int64), (reweighted_t-1))
            if self.args.show_t:
                print(reweighted_t)
            noised_x = self.get_noised_x(x_diff, reweighted_t)
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])
        # classifier part
        if self.is_imagenet:
            x_clf = diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False))
        else:
            x_clf = diff2clf(x_diff)
        return x_clf

    


    def forward(self, x, eps_data, adv_eps):
        eps_data = torch.tensor(eps_data)
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        eps_mu = torch.mean(eps_data)
            
        x_diff = clf2diff(x)
        if self.args.adaptive_defense_eval or self.args.whitebox_defense_eval:
            num_denoising_steps = [int(i) for i in self.args.def_num_denoising_steps.split(',')]
        else:
            num_denoising_steps = [int(i) for i in self.args.att_num_denoising_steps.split(',')]
        for i in range(len(self.max_timestep)):
            tau = self.args.tau
            bias = self.args.bias
            reweighted_t = torch.sigmoid(input=((adv_eps - eps_mu)/tau)) * (self.max_timestep[i] + bias)
            reweighted_t = torch.round(reweighted_t).to(torch.int64).to(x_diff.device)
            if self.args.adaptive_defense_eval or self.args.whitebox_defense_eval:
                self.attack_steps[i] = get_diffusion_params(reweighted_t, (num_denoising_steps[i] + bias))
            else:
                self.attack_steps[i] = get_diffusion_params(reweighted_t, num_denoising_steps[i])
            reweighted_t = torch.where((reweighted_t-1) < 0, torch.tensor(0, dtype=torch.int64), (reweighted_t-1))
            if self.args.show_t:
                print(reweighted_t)
            noised_x = self.get_noised_x(x_diff, reweighted_t)
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])
        # classifier part
        if self.is_imagenet:
            x_clf = diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False))
        else:
            x_clf = diff2clf(x_diff)
        if self.args.save_purify_img:
            torch.save(x_clf, os.path.join(self.args.adv_data_dir, "adv_data", f"purified_examples_batch_{self.args.batch_seed}.pt"))
        logits = self.clf(x_clf)
        return logits


    
def get_diffusion_params(t_values, num_denoising_steps):
    # Round up to ensure we get num_denoising_steps steps
    num_denoising_steps += 1
    if num_denoising_steps == 2: #edge case where t=1
        return torch.where(torch.stack([t_values - 1],dim=-1) == -1, torch.tensor(0, dtype=torch.int64), torch.stack([t_values - 1],dim=-1))
    t_values = torch.ceil(t_values).to(torch.int64)
    
    # Initialize the tensor to store diffusion steps
    diffusion_steps = torch.zeros((t_values.size(0), num_denoising_steps), dtype=torch.int64)
    
    # Calculate diffusion steps for each value in t_values
    for i, t in enumerate(t_values):
        steps = torch.linspace(0, t, steps=num_denoising_steps).ceil().to(torch.int64)
        diffusion_steps[i, :] = steps[:]
    
    diffusion_steps -= 1
    diffusion_steps = torch.where(diffusion_steps == -1, torch.tensor(0, dtype=torch.int64), diffusion_steps)
    
    return diffusion_steps[:, 1:]
