import os
import time
import numpy as np
import torch
import yaml
import logging
import argparse

import utils
from utils import load_data_from_train, set_all_seed, str2bool, load_diffusion
from eps_calculation import get_eps, get_score
from path import *
from tqdm import tqdm

def calculate_threshold(rank, args, config):
    # Load the training data
    x, _ = load_data_from_train(args)
    data_loader = torch.utils.data.DataLoader(x,
                                             batch_size=args.batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             drop_last=False)

    all_eps = []  # To store eps for all batches
    model_src = diffusion_model_path[args.dataset]
    model = load_diffusion(args, model_src, device=config.device)
    for x_batch in tqdm(data_loader, desc="Batch: "):
        x_batch = x_batch.to(config.device)
        if args.use_score:
            eps = get_score(x_batch, model, args, config)
        else:
            eps = get_eps(x_batch, model, args, config)
        all_eps.append(eps)

    all_eps_tensor = torch.cat(all_eps)
    print(f"The saved tensor got a shape: {all_eps_tensor.shape}")
    save_path = os.path.join(args.eps_norm_dir, f'{all_eps_tensor.shape[0]}_cifar10_score.pt')
    torch.save(all_eps_tensor, save_path)
    print(f"Saved eps tensor to {save_path}")


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for subset data.')
    parser.add_argument("--use_cuda", action='store_true', help="Whether use gpu or not")
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--dataset', type=str, default='cifar10', help='which domain: cifar10, imagenet')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--eps_reweight', type=str2bool, default=False, help='Decide if reweight the perturbation range based on EPS')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--denoise_step_size', type=float, default=1e-3, help='Full gradient or surrogate process')
    parser.add_argument('--defense_method', type=str, default="diffpure", help='baselines + (linear/non-linear) our method.')
    parser.add_argument('--tau', type=int, default=0, help='temperature parameter')

    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--eot', type=int, default=20)
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['pgd', 'pgdl2', 'bpda'])
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')

    # Purification hyperparameters in defense
    parser.add_argument("--def_max_timesteps", type=str, default="", help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_num_denoising_steps', type=str, default="", help='The number of denoising steps for each purification step in defense')
    parser.add_argument('--def_sampling_method', type=str, default='', choices=['ddpm', 'ddim', ''], help='Sampling method for the purification in defense')

    # Purification hyperparameters in attack generation
    parser.add_argument("--att_max_timesteps", type=str, default="", help='The number of forward steps for each purification step in attack')
    parser.add_argument('--att_num_denoising_steps', type=str, default="", help='The number of denoising steps for each purification step in attack')
    parser.add_argument('--att_sampling_method', type=str, default='', choices=['ddpm', 'ddim',''], help='Sampling method for the purification in attack')

    parser.add_argument('--num_ensemble_runs', type=int, default=1, help='The number of ensemble runs for purification in defense')
    #eps
    parser.add_argument('--diffuse_t', type=int,default=20) # CIFAR10:20, Imagenet: 50
    parser.add_argument('--bias', type=int, default=5, help="It determines the bias added onto the scale function.")
    parser.add_argument('--single_vector_norm_flag', action='store_true')	
    parser.add_argument('--detection_ensattack_norm_flag', action='store_true')
    parser.add_argument('--show_eps_range_info', action='store_true')
    parser.add_argument('--show_t', action='store_true')
    parser.add_argument('--use_score', action='store_true')


    parser.add_argument('--adaptive_defense_eval', action='store_true')
    parser.add_argument('--whitebox_defense_eval', action='store_true')
    parser.add_argument('--ablation', action='store_true')

    #Torch DDP
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='Number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='Rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='Address for master')
    parser.add_argument('--port', type=str, default='1234', help='Port number for torch ddp')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)

    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)


    args.eps_norm_dir = os.path.join("loaded_data", args.dataset, f"sd{str(args.seed)}")
    os.makedirs(args.eps_norm_dir, exist_ok=True)

    args.world_size = 1

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    set_all_seed(args)

    torch.backends.cudnn.benchmark = True

    return args, new_config


# Usage
if __name__ == "__main__":
    # Assuming args and config are defined somewhere
    args, config = parse_args_and_config()
    calculate_threshold(0, args, config)
