import time
from io import BytesIO
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

# Paper purifiers: Mean, Gaussian, JPEG + DiffPure, DDIM, MimicDiffusion, ContrastDiff, SSNI, DCDefense.
_PAPER_PURIFIERS = frozenset(
    {
        "Mean",
        "Gaussian",
        "JPEG",
        "DiffPure",
        "DDIM",
        "MimicDiffusion",
        "ContrastDiff",
        "SSNI",
        "DCDefense",
    }
)


class Purifier:
    def __init__(
        self,
        method: str,
        settings: Dict,
        batch_size: int = None,
        seed: int = None,
    ):
        if method not in _PAPER_PURIFIERS:
            raise ValueError(
                f"Unknown purifier {method!r}. Supported: {sorted(_PAPER_PURIFIERS)}"
            )
        self.method = method
        self.settings = settings
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_hyperparam(self) -> None:
        seed = self.seed if self.seed is not None else time.time()
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def mean_filter(self, imgs: Tensor, kernel_size: int = 3) -> Tensor:
        device_0 = imgs.device
        kernel = torch.ones(imgs.shape[1], 1, kernel_size, kernel_size, device=self.device)
        kernel /= kernel_size**2

        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        imgs_padded = F.pad(
            imgs.to(self.device), (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
        )
        out = F.conv2d(imgs_padded, kernel, padding=0, groups=imgs.shape[1])
        return out.to(device_0)

    def gaussian_filter(self, imgs: Tensor, kernel_size: int = 3, sigma_k: float = 1.0) -> Tensor:
        device_0 = imgs.device
        coords = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma_k**2))
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(imgs.shape[1], 1, 1, 1)

        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        imgs_padded = F.pad(
            imgs.to(self.device), (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
        )
        out = F.conv2d(imgs_padded, kernel, padding=0, groups=imgs.shape[1])
        return out.to(device_0)

    def gaussian_noise_gaussian_filter(
        self, imgs: Tensor, kernel_size: int = 3, sigma_y: float = 0.1, sigma_k: float = 1.0
    ) -> Tensor:
        self.init_hyperparam()
        device_0 = imgs.device
        imgs = imgs.to(self.device)
        imgs_noised = imgs + torch.randn_like(imgs) * sigma_y
        return self.gaussian_filter(imgs_noised, kernel_size, sigma_k).to(device_0)

    def jpeg_compress(self, imgs: Tensor, quality: int = 50) -> Tensor:
        device_0 = imgs.device
        imgs = imgs.clamp(0, 1)
        imgs_cpu = (imgs.detach().cpu() * 255).to(torch.uint8).permute(0, 2, 3, 1)

        compressed = []
        for img in imgs_cpu:
            pil_img = Image.fromarray(img.numpy(), mode="RGB")
            with BytesIO() as buf:
                pil_img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                rec = Image.open(buf).convert("RGB")
                rec_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(rec.tobytes()))
                rec_tensor = (
                    rec_tensor.view(rec.size[1], rec.size[0], 3).permute(2, 0, 1).float() / 255.0
                )
                compressed.append(rec_tensor)

        return torch.stack(compressed).to(device_0)

    def purify(self, imgs: Tensor, return_mid=False) -> Tensor:
        if self.batch_size is None:
            self.batch_size = len(imgs)

        if self.method == "Mean":
            imgs_purified = self.mean_filter(imgs, self.settings["kernel"])
        elif self.method == "Gaussian":
            imgs_purified = self.gaussian_noise_gaussian_filter(
                imgs,
                self.settings["kernel"],
                self.settings["sigma_y"],
                self.settings["sigma_k"],
            )
        elif self.method == "JPEG":
            imgs_purified = self.jpeg_compress(imgs, self.settings["quality"])
        elif self.method == "DiffPure":
            from .diffpure import DiffPure

            purifier = DiffPure(
                data=self.settings["data"],
                timesteps=self.settings["timesteps"],
                seed=self.seed,
            )
            purifier.reset_counter()
            purifier.set_tag()
            imgs_purified = purifier.purify(imgs)
        elif self.method == "DDIM":
            from .diffpure import DiffPureDDIM

            purifier = DiffPureDDIM(
                data=self.settings["data"],
                timesteps=self.settings["timesteps"],
                denoise_steps=self.settings["denoise_steps"],
                seed=self.seed,
            )
            imgs_purified = purifier.purify(imgs, return_mid)
        elif self.method == "MimicDiffusion":
            from .mimicdiffusion import MimicDiffusionDefense

            purifier = MimicDiffusionDefense(
                data=self.settings["data"],
                timesteps=self.settings["timesteps"],
                denoise_steps=self.settings["denoise_steps"],
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == "SSNI":
            from .ssni import SSNIDefense

            data = self.settings["data"]
            default_adv_eps = 4.0 / 255.0 if data == "imagenet" else 8.0 / 255.0
            purifier = SSNIDefense(
                data=data,
                timesteps=self.settings["timesteps"],
                denoise_steps=self.settings["denoise_steps"],
                tau=self.settings.get("tau", 1.0),
                bias=self.settings.get("bias", 50),
                adv_eps=self.settings.get("adv_eps", default_adv_eps),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == "DCDefense":
            from .dc import DCDefense

            purifier = DCDefense(
                data=self.settings["data"],
                strength_l=self.settings.get("strength_l", 0.2),
                strength_s=self.settings.get("strength_s", 0.1),
                forward_noise_steps=self.settings.get("forward_noise_steps", 1),
                ddim_steps=self.settings.get("timesteps", 150),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == "ContrastDiff":
            from .contrastdiff import ContrastDiffDefense

            purifier = ContrastDiffDefense(
                data=self.settings["data"],
                t=self.settings.get("timesteps", 150),
                sample_step=self.settings.get("sample_step", 1),
                contrastive_classifier=self.settings.get(
                    "contrastive_classifier", "imagenet-resnet50"
                ),
                n_classes=self.settings.get("n_classes", 1000),
                optim_lr=self.settings.get("optim_lr", 0.01),
                drift_min=self.settings.get("drift_min", 40),
                drift_max=self.settings.get("drift_max", 100000),
                tau_plus=self.settings.get("tau_plus", 0.1),
                beta=self.settings.get("beta", 1.0),
                temperature=self.settings.get("temperature", 1.0),
                use_bm=self.settings.get("use_bm", False),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        else:
            raise ValueError(f"Unsupported defense: {self.method!r}")

        return imgs_purified
