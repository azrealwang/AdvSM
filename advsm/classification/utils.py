import os
import numpy as np
import torch
import random
import math
from typing import Optional, Tuple, Any
from torch import Tensor
from torchvision.utils import save_image
from PIL import Image

from advsm.classification.models import load_classifier

# -------------------------
# Model loading helpers
# -------------------------
class NormalizeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean, std):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

def load_one_model(dataset: str, name: str, threat_model: str = "Linf") -> torch.nn.Module:
    """Load a classifier by config name (ImageNet) or RobustBench id."""
    if dataset == "imagenet":
        try:
            return load_classifier(name)
        except KeyError:
            from robustbench import load_model

            return load_model(name, dataset="imagenet", threat_model=threat_model).eval()

    if dataset == "cifar10":
        from robustbench import load_model

        if name == "R56_cifar10":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
            ).eval()
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2023, 0.1994, 0.2010)
            return NormalizeWrapper(model, cifar10_mean, cifar10_std).eval()
        return load_model(name, dataset="cifar10", threat_model=threat_model).eval()

    raise ValueError(f"Unsupported dataset: {dataset!r}")

def imgs_resize(
        imgs: Tensor,
        shape: Tuple[int, int],
        ) -> Tensor:
    from torchvision.transforms import Resize
    transform = Resize(shape,antialias=True)
    images = transform(imgs)
    
    return images

def save_all_images(
        imgs: Any,
        labels: Any,
        output_path: str,
        start_idx: Optional[int] = 0,
        ) -> None:
    amount = imgs.shape[0]
    for i in range(amount):
        save_image(imgs[i], f'{output_path}/%05d_%d.png'%(i+start_idx,labels[i]))

def robust_samples(classifier, x, y):
    # Select 100% accurate samples for testing
    preds = classifier.predict(x)
    results = np.argmax(preds, axis=1) == y
    idx_robust = np.where(results==True)[0]
    x_robust = x[idx_robust]
    y_robust = y[idx_robust]
        
    return x_robust, y_robust

def load_samples(
    path: str,
    start_idx: Optional[int] = 0,
    end_idx: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Any]:   

    images, labels = [], []
    basepath = r""
    samplepath = os.path.join(basepath, f"{path}")
    files = os.listdir(path)
    if end_idx is None:
        end_idx = len(files)

    for i in range(start_idx,end_idx):
        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)
        
        if shape is not None:
            image = image.resize(shape)
        
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3
        
        image = np.transpose(image, (2, 0, 1))
        
        images.append(image)
        labels.append(label)
    
    images_ = np.stack(images)
    labels_ = np.array(labels)

    images_ = images_ / 255
    images_ = np.ascontiguousarray(images_)
    
    return images_, labels_

def l2norm(x):
    
    return (Tensor(x) ** 2).reshape(x.shape[0], -1).sum(-1).sqrt().cpu().numpy()

def rand_perturbs(x,eps):
    eps_arr = [0,-eps,eps]
    perturbs = np.array([random.choice(eps_arr) for _ in range(np.prod(x.shape))])
    perturbs = np.reshape(perturbs,x.shape)
    x_adv = x + perturbs
    x_adv[x_adv>1] = 1
    x_adv[x_adv<0] = 0
    
    return x_adv

def compress(
    input_path: str,
    output_path: str,
    format: Optional[str] = 'jpeg',
    quality: Optional[int] = 5,
    )-> None:
    files = os.listdir(input_path)
    amount = len(files)
    for i in range(amount):
        # open file
        in_file = os.path.join(input_path, files[i])
        image = Image.open(in_file)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out_file = os.path.join(output_path, f"{files[i].split('.')[0]}.{format}")
        if format == 'jpeg':
            image.save(out_file, format=format, quality=quality*10)
        elif format =='png':
            image.save(out_file, format=format, compress_level=quality)
        elif format =='j2k':
            image.save(out_file, format='JPEG2000',quality_mode="rates",quality_layers=[10/quality])
        else:
            raise ValueError(
            "Unsupported compression format"
        )

def perspective_transform(
    input_path: str,
    output_path: str,
    data: str
    )-> None:
    from torchvision.transforms.functional import perspective,rotate
    files = os.listdir(input_path)
    amount = len(files)
    # 20% distortion and 10 degree rotation
    if data == 'cifar10':
        width = 32
        height = 32
        # (32*x_2+x_1*y_2-x_2*y_1)/2
        topleft = [0,0]
        topright = [30,4]
        botright = [28,28]
        botleft = [0,height-1]
    elif data == 'imagenet':
        width = 224
        height = 224
        # (224*x_2+x_1*y_2-x_2*y_1)/2
        topleft = [0,0]
        topright = [200,40]
        botright = [180,190]
        botleft = [0,height-1]
    else:
        raise ValueError("Unsupported data set")
    
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    for i in range(amount):
        # open file
        in_file = os.path.join(input_path, files[i])
        image = Image.open(in_file)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out_file = os.path.join(output_path, f"{files[i]}")
        perspective_imgs = perspective(image,startpoints=startpoints, endpoints=endpoints)
        perspective_imgs=rotate(perspective_imgs,angle=10)
        perspective_imgs.save(out_file)

def predict(
    model,
    imgs: Tensor,
    batch_size: int=100,
    )-> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count = len(imgs)
    batches = math.ceil(count/batch_size)
    logits = Tensor([])
    for b in range(batches):
        if b == batches-1:
            idx = range(b*batch_size,count)
        else:
            idx = range(b*batch_size,(b+1)*batch_size)
        with torch.no_grad():
            logits_batch = model.to(device)(imgs[idx].to(device)).cpu()
        logits = torch.cat((logits, logits_batch), 0)
    
    return logits

def smart_cast(val):
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            continue
    return val  # fallback to string


def parse_d_settings_arg(d_settings) -> dict:
    """Parse ``--d_settings`` tokens from argparse ``nargs='+'`` + ``action='append'``."""
    if not d_settings:
        return {}
    flat = []
    for group in d_settings:
        if isinstance(group, (list, tuple)):
            flat.extend(group)
        else:
            flat.append(group)
    if len(flat) % 2 != 0:
        raise ValueError(
            f"--d_settings must be name/value pairs, got {len(flat)} tokens: {flat!r}"
        )
    return {flat[i]: smart_cast(flat[i + 1]) for i in range(0, len(flat), 2)}