
import os
import time
import numpy as np

import torch


def get_eps(x, model, args, config):
    start_time = time.time()
    if args.dataset =="cifar10":
        img_shape = (3, 32, 32)
    elif args.dataset =="imagenet":
        img_shape = (3, 256, 256)
    rev_vpsde = RevVPSDE(model=model, score_type=args.score_type, img_shape= img_shape, model_kwargs= None).to(config.device)
    sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
    score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)
    score_sum = torch.zeros_like(x, device=x.device)
    score_adv_list = []
    score_adv_lists = []
    with torch.no_grad():
        for value in range(1,args.diffuse_t+1):
            t_value = value/1000
            curr_t_temp = torch.tensor(t_value,device=x.device)

            z = torch.randn_like(x, device=x.device)
            mean_x_adv, std_x_adv = sde.marginal_prob(2*x-1, curr_t_temp.expand(x.shape[0])) #Here the input should be in magnitude of [0,1], converted to [-1,1]
            perturbed_data = mean_x_adv + std_x_adv[:, None, None, None] * z
            score = score_fn(perturbed_data, curr_t_temp.expand(x.shape[0]))

            if args.dataset=='imagenet':
                score, _ = torch.split(score, score.shape[1]//2, dim=1)
                assert x.shape == score.shape, f'{x.shape}, {score.shape}'
            if args.detection_ensattack_norm_flag:
                score_adv_list.append(score.detach().view(x.shape[0],-1).norm(dim=-1).unsqueeze(0))
            elif args.single_vector_norm_flag:
                # score_adv_list.append(score.detach().view(1, *x_adv.shape))
                score_sum += score.detach()
            else:
                score_adv_list.append(score.detach())
            # print(f"{value}th eps calculation time: {time.time() - start_time}")

        if args.detection_ensattack_norm_flag:
            score_adv_lists.append(torch.cat(score_adv_list, dim=0))
        elif args.single_vector_norm_flag:
            score_adv_lists.append(score_sum/value)
        else:
            score_adv_lists.append(torch.cat(score_adv_list, dim=0).view(len(score_adv_list),*x.shape).cpu())
    if args.show_eps_range_info:
        print(f'diffuison time: {time.time() - start_time}')
    adv_score_tensor = torch.cat(score_adv_lists, dim=1) if not args.single_vector_norm_flag else torch.cat(score_adv_lists, dim=0)
    return adv_score_tensor

from score_sde import sde_lib
from score_sde.models import utils as mutils
from runners.diffpure_sde import RevVPSDE
def get_score(x, model, args, config):
    start_time = time.time()
    if args.dataset =="cifar10":
        img_shape = (3, 32, 32)
    elif args.dataset =="imagenet":
        img_shape = (3, 256, 256)
    rev_vpsde = RevVPSDE(model=model, score_type=args.score_type, img_shape= img_shape, model_kwargs= None).to(config.device)
    sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
    score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)
    score_sum = torch.zeros_like(x, device=x.device)
    score_adv_list = []
    score_adv_lists = []
    with torch.no_grad():
        t_value = 20/1000
        curr_t_temp = torch.tensor(t_value,device=x.device)
        score = score_fn(2*x-1, curr_t_temp.expand(x.shape[0]))

        if args.dataset=='imagenet':
            score, _ = torch.split(score, score.shape[1]//2, dim=1)
            assert x.shape == score.shape, f'{x.shape}, {score.shape}'

        if args.detection_ensattack_norm_flag:
            score_adv_list.append(score.detach().view(x.shape[0],-1).norm(dim=-1).unsqueeze(0))
            score_adv_lists.append(torch.cat(score_adv_list, dim=0))
        elif args.single_vector_norm_flag:
            score_sum += score.detach()
            score_adv_lists.append(score_sum)
        else:
            score_adv_list.append(score.detach())
            score_adv_lists.append(torch.cat(score_adv_list, dim=0).view(len(score_adv_list),*x.shape).cpu())

    if args.show_eps_range_info:
        print(f'diffuison time: {time.time() - start_time}')
    adv_score_tensor = torch.cat(score_adv_lists, dim=1) if not args.single_vector_norm_flag else torch.cat(score_adv_lists, dim=0)
    return adv_score_tensor



def reweight_t(x_test, model, eps_data, args, config):
    if args.show_eps_range_info:
        print(f"min in nat is: {min(eps_data)}")
        print(f"max in nat is: {max(eps_data)}")

    if args.use_score:
        x_adv_eps = get_score(x_test, model, args, config)
    else:
        x_adv_eps = get_eps(x_test, model, args, config)

    adv_eps_list = []
    for adv_score in x_adv_eps:
        adv_eps_list.append(torch.norm(adv_score.flatten(), p=2).cpu().item())


    min_eps = min(min(adv_eps_list), min(eps_data))
    max_eps = max(max(adv_eps_list), max(eps_data))
    if args.show_eps_range_info:
        print(f"min in adv is: {min(adv_eps_list)}")
        print(f"max in adv is: {max(adv_eps_list)}")

    EPS_range = (max_eps, min_eps)
    adv_eps_list = torch.from_numpy(np.array(adv_eps_list))

    return EPS_range, adv_eps_list