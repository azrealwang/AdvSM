import random
# import dataset
import argparse
import yaml
from typing import Any
import numpy as np
from robustbench import load_model
from autoattack import AutoAttack

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .score_sde.models import utils as mutils
from .score_sde.losses import get_optimizer
from .score_sde.models.ema import ExponentialMovingAverage

from .guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def diff2clf(x, is_imagenet=False): 
    # [-1, 1] to [0, 1]
    return (x / 2) + 0.5 

def clf2diff(x):
    # [0, 1] to [-1, 1]
    return (x - 0.5) * 2

def normalize(x):
    # Normalization for ImageNet
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(x)




def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def update_state_dict(state_dict, idx_start=9):
    '''
    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

def get_image_classifier(args):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x) #将处理过的image传给resnet
    if 'imagenet' in args.classifier_name:
        if 'resnet18' in args.classifier_name:
            print('using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif 'resnet50' in args.classifier_name:
            print('using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif 'resnet101' in args.classifier_name:
            print('using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif 'wideresnet-50-2' in args.classifier_name:
            print('using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        elif 'deit-s' in args.classifier_name:
            print('using imagenet deit-s...')
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {args.classifier_name}')

        wrapper_resnet = _Wrapper_ResNet(model)
        
    elif 'cifar10' in args.classifier_name:
        if 'wrn-70-16-dropout' in args.classifier_name:
            print('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            from classifiers.cifar10_resnet import WideResNet_70_16_dropout
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wrn-70-16-dropout/weights.pt'
            print(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded wrn-70-16-dropout checkpoint")

        elif 'resnet-50' in args.classifier_name:
            print('using cifar10 resnet-50...')
            from classifiers.cifar10_resnet import ResNet50
            model = ResNet50()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/resnet-50/weights.pt'
            print(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded resnet-50 checkpoint")
        elif 'wideresnet-28-10' in args.classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]
        else:
            raise NotImplementedError(f'unknown {args.classifier_name}')
        wrapper_resnet = model
    else:
        raise NotImplementedError(f'unknown {args.classifier_name}')
    return wrapper_resnet

def load_diffusion(args, model_src, device):
    if args.dataset == 'cifar10':
        # Diffusion model
        with open('./configs/cifar10.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        diffusion = mutils.create_model(config)
        optimizer = get_optimizer(config, diffusion.parameters())
        ema = ExponentialMovingAverage(
            diffusion.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=diffusion, ema=ema)
        restore_checkpoint(model_src, state, device)
        ema.copy_to(diffusion.parameters())
        diffusion.eval().to(device)

    elif args.dataset == 'imagenet':
        with open('./configs/imagenet.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict2namespace(config)
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        diffusion, _ = create_model_and_diffusion(**model_config)
        diffusion.load_state_dict(torch.load(model_src, map_location='cpu'))
        diffusion.convert_to_fp16()
        diffusion.eval().to(device)

    return diffusion


def attack(args, config, x_val, y_val):
    attack_version = args.attack_version  # ['standard', 'rand', 'custom']
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')
    # ---------------- apply the attack to classifier ----------------
    print(f'apply the attack to classifier [{args.lp_norm}]...')
    classifier = get_image_classifier(args).to(config.device)
    adversary_resnet = AutoAttack(classifier, norm=args.lp_norm, eps=args.adv_eps,
                                  version=attack_version, attacks_to_run=attack_list,
                                  log_path=f'{args.log_dir}/log_{args.classifier_name}_sd{args.seed}_batch{0}.txt', device=config.device)
    adversary_resnet.apgd.eot_iter = args.eot_iter
    print(f'{args.lp_norm}, epsilon: {args.adv_eps}')
    x_adv_resnet = adversary_resnet.run_standard_evaluation(x_val, y_val, bs=args.adv_batch_size)
    print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
    torch.save([x_adv_resnet, y_val], f'{args.log_dir}/adv_images_{args.classifier_name}_sd{args.seed}_batch{0}.pt')

def load_data(args):
    if 'imagenet' in args.dataset:
        val_dir = './dataset/imagenet_lmdb/val'  # using imagenet lmdb data
        val_transform = dataset.get_transform(args.dataset, 'imval', base_size=224)
        val_data = dataset.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=args.num_sub, data_seed=0)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=1)
        x_val, y_val = next(iter(val_loader))
    elif 'cifar10' in args.dataset:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = dataset.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=0)
        n_samples = len(val_data)
        print('length of validation data is: ', len(val_data))
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=1)
        x_val, y_val = next(iter(val_loader))
    else:
        raise NotImplementedError(f'Unknown domain: {args.dataset}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val

def load_data_from_train(args):
    if 'imagenet' in args.dataset:
        val_dir = '/data/datasets/Imagenet/ILSVRC/Data/CLS-LOC/train'  # using imagenet lmdb data
        val_transform = dataset.get_transform(args.dataset, 'imtrain', base_size=224)
        val_data = dataset.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=5000, data_seed=0)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=1)
        x_val, y_val = next(iter(val_loader))
    elif 'cifar10' in args.dataset:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = dataset.cifar10_train_data_sub(args, data_dir, transform=transform,
                                            num_sub=5000, data_seed=0)
        n_samples = len(val_data)
        print('length of validation data is: ', len(val_data))
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=1)
        x_val, y_val = next(iter(val_loader))
    else:
        raise NotImplementedError(f'Unknown domain: {args.dataset}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val

def set_all_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)