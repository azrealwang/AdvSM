
from PIL import Image
import numpy as np
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from dataset import imagenet_lmdb_dataset_sub, get_transform



def cifar10_dataset_sub(root, transform=None, num_sub=512, data_seed=0):
    dataset = torchvision.datasets.CIFAR10(
        root=root, transform=transform, download=True, train=False)
    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(
            len(dataset), num_sub, replace=False)
        dataset = Subset(dataset, partition_idx)
    return dataset


def load_cifar10_sub(root, num_sub, data_seed=0):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = cifar10_dataset_sub(
        root, transform=transform, num_sub=num_sub, data_seed=data_seed)
    return dataset



def load_dataset_by_name(dataset, data_seed, root='./dataset', num_sub=512):
    if dataset == 'cifar10':
        dataset = load_cifar10_sub(root, num_sub)
    elif dataset == 'imagenet':
        val_transform = get_transform(dataset, 'imval', base_size=224)
        dataset = imagenet_lmdb_dataset_sub(root, transform=val_transform,
                                                  num_sub=num_sub, data_seed = data_seed)
    print('dataset len: ', len(dataset))
    return dataset
