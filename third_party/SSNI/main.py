from attacks.pgd_eot import PGD, Whitebox_PGD
from attacks.pgd_eot_l2 import PGDL2, Whitebox_PGDL2
from attacks.pgd_eot_bpda import BPDA, Whitebox_BPDA
import argparse
import utils
from utils import get_image_classifier, load_diffusion, set_all_seed, str2bool
import logging
import yaml
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import torchvision.utils as tvu

import os
import datetime

from load_data import load_dataset_by_name
from purifier_clf_models.sde_adv_model import SDE_Adv_Model
from purifier_clf_models.robust_eval_model import PurificationForward, AdaptivePurificationForward, LinearPurificationForward
from tqdm import tqdm
from path import *
from eps_standard import load_dataset_eps
from eps_calculation import reweight_t

def get_diffusion_params(max_timesteps, num_denoising_steps):
    max_timestep_list = [int(i) for i in max_timesteps.split(',')]
    num_denoising_steps_list = [int(i) for i in num_denoising_steps.split(',')]
    assert len(max_timestep_list) == len(num_denoising_steps_list)

    diffusion_steps = []
    for i in range(len(max_timestep_list)):
        diffusion_steps.append([i - 1 for i in range(max_timestep_list[i] // num_denoising_steps_list[i],
                               max_timestep_list[i] + 1, max_timestep_list[i] // num_denoising_steps_list[i])])
        max_timestep_list[i] = max_timestep_list[i] - 1

    return max_timestep_list, diffusion_steps

def load_adv(args, config, diffusion, model, x, y):
    if args.attack_type == "pgd":
        eps = 8./255.
        if args.dataset == "imagenet":
            eps = 4./255.
        if "ours" in args.defense_method:
            attack = Whitebox_PGD(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = False)
            print('[Attack] Whitebox_PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        elif "linear" in args.defense_method:
            attack = Whitebox_PGD(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = True)
            print('[Attack] Whitebox_PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        else:
            attack = PGD(model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] Normal PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
    elif args.attack_type == "pgdl2":
        eps = 0.5
        if "ours" in args.defense_method:
            attack = Whitebox_PGDL2(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = False)
            print('[Attack] Whitebox_PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        elif "linear" in args.defense_method:
            attack = Whitebox_PGDL2(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = True)
            print('[Attack] Whitebox_PGD L2 | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        else:
            attack = PGDL2(model, attack_steps=args.n_iter,
                        eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] Normal PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                args.n_iter, eps, args.eot))
    elif args.attack_type == "bpda":
        eps = 8./255.
        if "ours" in args.defense_method:
            attack = Whitebox_BPDA(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = False)
            print('[Attack] Whitebox_BPDA+EOT | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        elif "linear" in args.defense_method:
            attack = Whitebox_BPDA(diffusion, args, config, model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot, is_linear = True)
            print('[Attack] Whitebox_BPDA+EOT | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
        else:
            attack = BPDA(model, attack_steps=args.n_iter,
                            eps=eps, step_size=0.007, eot=args.eot)
            print('[Attack] BPDA+EOT Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                args.n_iter, eps, args.eot))
    else:
        raise NotImplementedError("No more attacks.")
    
    x_adv = attack(x, y)
    # os.makedirs(os.path.join(args.adv_data_dir, "adv_data", args.classifier_name+args.attack_type+str(args.batch_seed)), exist_ok=True)
    # torch.save(x_adv, os.path.join(args.adv_data_dir, "adv_data", args.classifier_name+args.attack_type+str(args.batch_seed), f"adversarial_examples_batch_{idx}.pt"))
    return x_adv

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for subset data.')
    parser.add_argument("--use_cuda", action='store_true', help="Whether use gpu or not")
    parser.add_argument('--batch_size', type=int, default=16)

    #use these for origin_diffpure
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
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
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
    parser.add_argument('--tau_ablation', action='store_true')
    parser.add_argument('--save_as_pdf', action='store_true')
    parser.add_argument('--save_purify_img', action='store_true')

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

    if args.ablation:
        args.adv_data_dir = os.path.join("ablation_data", "seed" + str(args.seed), args.defense_method, args.classifier_name + args.attack_type)
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            args.adv_data_dir = os.path.join("ablation_data", "seed" + str(121), args.defense_method, args.classifier_name + args.attack_type)
    elif args.tau_ablation:
        args.adv_data_dir = os.path.join("ablation_data", "seed" + str(args.seed), "t"+str(args.tau), args.attack_type)
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            args.adv_data_dir = os.path.join("ablation_data", "seed" + str(121), "t"+str(args.tau), args.attack_type)
    elif args.use_score:
        args.adv_data_dir = os.path.join("additional", args.dataset, "seed" + str(args.seed), args.defense_method, args.classifier_name + args.attack_type)
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            args.adv_data_dir = os.path.join("additional", args.dataset, "seed" + str(121), args.defense_method, args.classifier_name + args.attack_type)
    else:
        args.adv_data_dir = os.path.join("adversarial_examples", args.dataset, "seed" + str(args.seed), args.defense_method, args.classifier_name + args.attack_type)
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            args.adv_data_dir = os.path.join("adversarial_examples", args.dataset, "seed" + str(121), args.defense_method, args.classifier_name + args.attack_type)
    os.makedirs(args.adv_data_dir, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    set_all_seed(args)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def DBP_eval(rank, gpu, args, config):
    print('rank {} | gpu {} started'.format(rank, gpu))
    assert 512 % args.batch_size == 0

    is_imagenet = True if args.dataset == 'imagenet' else False
    dataset_root = imagenet_path if is_imagenet else './dataset'
    num_classes = 1000 if is_imagenet else 10
    num_of_data = 128 if is_imagenet else 512
    testset = load_dataset_by_name(args.dataset, args.data_seed, dataset_root, num_of_data)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=testsampler,
                                             drop_last=False)
    
    correct_nat = torch.tensor([0]).to(config.device)
    correct_adv = torch.tensor([0]).to(config.device)
    nat_total = torch.tensor([0]).to(config.device)
    adv_total = torch.tensor([0]).to(config.device)

    model_src = diffusion_model_path[args.dataset]
    clf = get_image_classifier(args).to(config.device)
    diffusion = load_diffusion(args, model_src, device=config.device)

    if args.defense_method == "origin_diffpure":
        # load data & diffusion model
        model = SDE_Adv_Model(args, config)
        model = model.eval().to(config.device)

        for idx, (x, y) in tqdm(enumerate(testLoader), desc="One batch starts."):
            args.batch_seed = idx
            x = x.to(config.device)
            y = y.to(config.device)
            eps_data = load_dataset_eps(args)

            # attack part, generate adversarial examples.
            adv_data_pt = os.path.join(args.adv_data_dir, "adv_data", f"adversarial_examples_batch_{idx}.pt")
            if os.path.exists(adv_data_pt):
                print(f"Batch {idx} exists, loading x_adv and y data.")
                data = torch.load(adv_data_pt)
                x_adv = data['x_adv']
                y_adv = data['y']
                print(f"The loaded x_adv has batchsize: {x_adv.shape[0]}")
            else:
                x_adv = load_adv(args, config, diffusion, model, x, y)
                gathered_x_adv = [torch.zeros_like(x_adv) for _ in range(args.world_size)]
                gathered_x = [torch.zeros_like(x) for _ in range(args.world_size)]
                gathered_y = [torch.zeros_like(y) for _ in range(args.world_size)]
                
                dist.all_gather(gathered_x_adv, x_adv)
                dist.all_gather(gathered_x, x)
                dist.all_gather(gathered_y, y)
                
                if rank == 0:
                    all_x_adv = torch.cat(gathered_x_adv, dim=0)
                    all_x = torch.cat(gathered_x, dim=0)
                    all_y = torch.cat(gathered_y, dim=0)
                    os.makedirs('{}/imgs'.format(args.adv_data_dir), exist_ok=True)
                    os.makedirs(os.path.join(args.adv_data_dir, "adv_data"), exist_ok=True)

                    torch.save({'x_adv': all_x_adv, 'y': all_y}, 
                            os.path.join(args.adv_data_dir, "adv_data", f"adversarial_examples_batch_{idx}.pt"))
                    tvu.save_image(all_x, '{}/imgs/{}_x.png'.format(args.adv_data_dir, idx))
                    tvu.save_image(all_x_adv, '{}/imgs/{}_x_adv.png'.format(args.adv_data_dir, idx))
                
                dist.barrier()
                        
            # defense part
            if args.adaptive_defense_eval:
                print(f"Evaluating Adversarial Examples: pixel min - {torch.min(x_adv)}, pixel max - {torch.max(x_adv)}")
                with torch.no_grad():
                    pred_nat = predict(x, args, config, model, diffusion, eps_data, num_classes)
                    correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

                    pred_adv = predict(x_adv, args, config, model, diffusion, eps_data, num_classes)
                    correct_adv += pred_adv.eq(y_adv.view_as(pred_adv)).sum().item()

                nat_total += x.shape[0]
                adv_total += x_adv.shape[0]

                print('rank {} | {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
                    rank, idx, nat_total.item(), adv_total.item(), (correct_nat / nat_total * 100).item(), (correct_adv / adv_total * 100).item()))

            elif args.whitebox_defense_eval:
                print(f"pixel min: {torch.min(x)}, pixel max: {torch.max(x)}")
                with torch.no_grad():
                    pred_nat = predict_whitebox(x, args, model, num_classes)
                    correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

                    pred_adv = predict_whitebox(x_adv, args, model, num_classes)
                    correct_adv += pred_adv.eq(y_adv.view_as(pred_adv)).sum().item()

                nat_total += x.shape[0]
                adv_total += x_adv.shape[0]

                print('rank {} | {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
                    rank, idx, nat_total.item(), adv_total.item(), (correct_nat / nat_total *
                                            100).item(), (correct_adv / adv_total * 100).item()
                ))
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            print('rank {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
            rank, nat_total.item(), adv_total.item(), (correct_nat / nat_total *
                                100).item(), (correct_adv / adv_total * 100).item()))


    else:

        # Specify adversarial attack target.
        att_max_timesteps, att_diffusion_steps = get_diffusion_params(
            args.att_max_timesteps, args.att_num_denoising_steps)
        if "ours" in args.defense_method: #Our method using a adaptive forward call.
            attack_forward = AdaptivePurificationForward(
                args, config, clf, diffusion, att_max_timesteps, att_diffusion_steps, args.att_sampling_method, is_imagenet, device=config.device)
            print("Attack type: adaptive white-box attack to our non-linear method!")
        elif "linear" in args.defense_method:
            attack_forward = LinearPurificationForward(
                args, config, clf, diffusion, att_max_timesteps, att_diffusion_steps, args.att_sampling_method, is_imagenet, device=config.device)
            print("Attack type: adaptive white-box attack to our linear method!")
        else:
            attack_forward = PurificationForward(
                args, clf, diffusion, att_max_timesteps, att_diffusion_steps, args.att_sampling_method, is_imagenet, device=config.device)
            print("Attack type: white-box attack to baselines!")
        for idx, (x, y) in tqdm(enumerate(testLoader), desc="One batch starts."):
            args.batch_seed = idx
            x = x.to(config.device)
            y = y.to(config.device)
            eps_data = load_dataset_eps(args)

            # attack part, generate adversarial examples.
            if args.dataset == "imagenet":
                adv_data_pt = os.path.join(args.adv_data_dir, "adv_data", f"ds_{args.data_seed}", f"adversarial_examples_batch_{idx}.pt")
            else:
                adv_data_pt = os.path.join(args.adv_data_dir, "adv_data", f"adversarial_examples_batch_{idx}.pt")
            nat_data_pt = os.path.join(args.adv_data_dir, "adv_data", f"nat_examples_batch_{idx}.pt")
            # torch.save(x, nat_data_pt)
            if os.path.exists(adv_data_pt):
                print(f"Batch {idx} exists, loading x_adv and y data.")
                data = torch.load(adv_data_pt)
                x_adv = data['x_adv']
                y_adv = data['y']
                print(f"The loaded x_adv has batchsize: {x_adv.shape[0]}")
            else:
                x_adv = load_adv(args, config, diffusion, attack_forward, x, y)
                gathered_x_adv = [torch.zeros_like(x_adv) for _ in range(args.world_size)]
                gathered_x = [torch.zeros_like(x) for _ in range(args.world_size)]
                gathered_y = [torch.zeros_like(y) for _ in range(args.world_size)]
                
                dist.all_gather(gathered_x_adv, x_adv)
                dist.all_gather(gathered_x, x)
                dist.all_gather(gathered_y, y)
                
                if rank == 0:
                    all_x_adv = torch.cat(gathered_x_adv, dim=0)
                    all_x = torch.cat(gathered_x, dim=0)
                    all_y = torch.cat(gathered_y, dim=0)
                    os.makedirs('{}/imgs/{}'.format(args.adv_data_dir, f"ds_{args.data_seed}"), exist_ok=True)
                    os.makedirs(os.path.join(args.adv_data_dir, "adv_data", f"ds_{args.data_seed}"), exist_ok=True)

                    torch.save({'x_adv': all_x_adv, 'y': all_y}, 
                            os.path.join(args.adv_data_dir, "adv_data", f"ds_{args.data_seed}", f"adversarial_examples_batch_{idx}.pt"))
                    tvu.save_image(all_x, '{}/imgs/{}/{}_x.png'.format(args.adv_data_dir, f"ds_{args.data_seed}", idx))
                    tvu.save_image(all_x_adv, '{}/imgs/{}/{}_x_adv.png'.format(args.adv_data_dir, f"ds_{args.data_seed}", idx))
                
                dist.barrier()
                        
            # defense part
            if args.adaptive_defense_eval:
                def_max_timesteps, def_diffusion_steps = get_diffusion_params(
                    args.def_max_timesteps, args.def_num_denoising_steps)
                if "ours" in args.defense_method:
                    advanced_defense_forward = AdaptivePurificationForward(args, config, clf, diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, is_imagenet, device=config.device)
                elif "linear" in args.defense_method:
                    advanced_defense_forward = LinearPurificationForward(args, config, clf, diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, is_imagenet, device=config.device)
                print(f"Evaluating Adversarial Examples: pixel min - {torch.min(x_adv)}, pixel max - {torch.max(x_adv)}")
                with torch.no_grad():
                    pred_nat = predict(x, args, config, advanced_defense_forward, diffusion, eps_data, num_classes)
                    correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

                    pred_adv = predict(x_adv, args, config, advanced_defense_forward, diffusion, eps_data, num_classes)
                    correct_adv += pred_adv.eq(y_adv.view_as(pred_adv)).sum().item()

                nat_total += x.shape[0]
                adv_total += x_adv.shape[0]

                print('rank {} | {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
                    rank, idx, nat_total.item(), adv_total.item(), (correct_nat / nat_total * 100).item(), (correct_adv / adv_total * 100).item()))

        
            elif args.whitebox_defense_eval:
                def_max_timesteps, def_diffusion_steps = get_diffusion_params(
                    args.def_max_timesteps, args.def_num_denoising_steps)

                defense_forward = PurificationForward(args, clf, diffusion, def_max_timesteps, def_diffusion_steps, args.def_sampling_method, is_imagenet, device=config.device)


                print(f"Evaluating Adversarial Examples: pixel min - {torch.min(x_adv)}, pixel max - {torch.max(x_adv)}")
                with torch.no_grad():
                    pred_nat = predict_whitebox(x, args, defense_forward, num_classes)
                    correct_nat += pred_nat.eq(y.view_as(pred_nat)).sum().item()

                    pred_adv = predict_whitebox(x_adv, args, defense_forward, num_classes)
                    correct_adv += pred_adv.eq(y_adv.view_as(pred_adv)).sum().item()

                nat_total += x.shape[0]
                adv_total += x_adv.shape[0]


                print('rank {} | {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
                    rank, idx, nat_total.item(), adv_total.item(), (correct_nat / nat_total *
                                            100).item(), (correct_adv / adv_total * 100).item()
                ))
        if args.whitebox_defense_eval or args.adaptive_defense_eval:
            print('rank {} | num_x_samples: {} | num_adv_samples: {} | acc_nat: {:.3f}% | acc_adv: {:.3f}%'.format(
            rank, nat_total.item(), adv_total.item(), (correct_nat / nat_total *
                                100).item(), (correct_adv / adv_total * 100).item()))

def predict(x, args, config, model, diffusion, eps_data, num_classes):
    ensemble = torch.zeros(x.shape[0], num_classes).to(x.device)
    for _ in range(args.num_ensemble_runs):
        _x = x.clone()
        eps_range, adv_eps = reweight_t(_x, diffusion, eps_data, args, config)
        if "ours" in args.defense_method:
            logits = model(_x, eps_data, adv_eps)
        else:
            logits = model(_x, eps_range, adv_eps)
        pred = logits.max(1, keepdim=True)[1]
        
        for idx in range(x.shape[0]):
            ensemble[idx, pred[idx]] += 1

    pred = ensemble.max(1, keepdim=True)[1]
    return pred

def predict_whitebox(x, args, defense_forward, num_classes):
    ensemble = torch.zeros(x.shape[0], num_classes).to(x.device)
    for _ in range(args.num_ensemble_runs):
        _x = x.clone()

        logits = defense_forward(_x)
        pred = logits.max(1, keepdim=True)[1]
        
        for idx in range(x.shape[0]):
            ensemble[idx, pred[idx]] += 1

    pred = ensemble.max(1, keepdim=True)[1]
    return pred

def cleanup():
    dist.destroy_process_group()


def init_processes(rank, size, fn, args, config):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size,
                            timeout=datetime.timedelta(hours=4))
    fn(rank, gpu, args, config)
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args, config = parse_args_and_config()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    print(args)
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            p = Process(target=init_processes, args=(
                global_rank, global_size, DBP_eval, args, config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, size, DBP_eval, args, config)




