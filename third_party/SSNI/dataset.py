import os, sys

import io
import lmdb

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import folder, ImageFolder

###### transformation functions ######

def get_transform(dataset, transform_type, base_size=256):
    if dataset.lower() == "celebahq":
        assert base_size == 256, base_size

        if transform_type == 'imtrain':
            return transforms.Compose([
                transforms.Resize(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif transform_type == 'imval':
            return transforms.Compose([
                transforms.Resize(base_size),
                # no horizontal flip for standard validation
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif transform_type == 'imcolor':
            return transforms.Compose([
                transforms.Resize(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=.05, contrast=.05,
                                       saturation=.05, hue=.05),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif transform_type == 'imcrop':
            return transforms.Compose([
                # 1024 + 32, or 256 + 8
                transforms.Resize(int(1.03125 * base_size)),
                transforms.RandomCrop(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif transform_type == 'tensorbase':
            # dummy transform for compatibility with other datasets
            return transforms.Lambda(lambda x: x)
        else:
            raise NotImplementedError

    elif "imagenet" in dataset.lower():
        assert base_size == 224, base_size

        if transform_type == 'imtrain':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif transform_type == 'imval':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(base_size),
                # no horizontal flip for standard validation
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


################################################################################
# CIFAR-10
###############################################################################

def cifar10_dataset_sub(root, transform=None, num_sub=-1, data_seed=0):
    val_data = torchvision.datasets.CIFAR10(root=root, transform=transform, download=True, train=False)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(val_data), num_sub, replace=False)
        val_data = Subset(val_data, partition_idx)

    return val_data

def cifar10_train_data_sub(args, root, transform=None, num_sub=-1, data_seed=0):
    train_data = torchvision.datasets.CIFAR10(root=root, transform=transform, download=True, train=True)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(train_data), num_sub, replace=False)
        train_data = Subset(train_data, partition_idx)

    return train_data

################################################################################
# ImageNet - LMDB
###############################################################################

def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
    valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    """

    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        './dataset/imagenet_lmdb/val' + '_faster_imagefolder.lmdb.pt') #TODO: VAL AND TRAIN
    lmdb_path = os.path.join(
        './dataset/imagenet_lmdb/val' + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        print("This line is executed.")
        data_set = ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set


def imagenet_lmdb_dataset_sub(
        root, transform=None, target_transform=None,
        loader=lmdb_loader, num_sub=-1, data_seed=0):
    data_set = imagenet_lmdb_dataset(
        root, transform=transform, target_transform=target_transform,
        loader=loader)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(data_set), num_sub, replace=False)
        data_set = Subset(data_set, partition_idx)

    return data_set