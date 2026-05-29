import time
from io import BytesIO
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image, ImageEnhance, ImageFilter


class Purifier:
    def __init__(
            self,
            method: str,
            settings: Dict,
            batch_size: int = None,
            seed: int = None,
    ):
        self.method = method
        self.settings = settings
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_hyperparam(self):
        seed = self.seed if self.seed is not None else time.time()
        # print(f"Defense Seed = {seed}")
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)
    
    # ---------- PIL helpers (0-1 Tensor <-> PIL) ----------
    def _tensor01_to_pil_list(self, imgs: Tensor):
        """
        imgs: (B,C,H,W) float in [0,1]
        return: list[PIL.Image] in RGB
        """
        imgs = imgs.detach().clamp(0, 1).cpu()
        imgs_u8 = (imgs * 255.0).to(torch.uint8).permute(0, 2, 3, 1)  # B,H,W,C
        pil_list = [Image.fromarray(img.numpy(), mode="RGB") for img in imgs_u8]
        return pil_list

    def _pil_list_to_tensor01(self, pil_list, device, dtype):
        """
        pil_list: list[PIL.Image] in RGB
        return: (B,C,H,W) float in [0,1] on given device/dtype
        """
        out = []
        for pil_img in pil_list:
            pil_img = pil_img.convert("RGB")
            t = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_img.tobytes()))
            t = t.view(pil_img.size[1], pil_img.size[0], 3).permute(2, 0, 1).float() / 255.0
            out.append(t)
        out = torch.stack(out).to(device=device, dtype=dtype).clamp(0, 1)
        return out

    # 1. Mean Filter (Differentiable)
    def mean_filter(self, imgs: Tensor, kernel_size: int = 3) -> Tensor:
        device_0 = imgs.device
        kernel = torch.ones(imgs.shape[1], 1, kernel_size, kernel_size, device=self.device)
        kernel /= kernel_size ** 2

        # Calculate padding to preserve spatial dimensions
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        imgs_padded = F.pad(imgs.to(self.device), (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        out = F.conv2d(imgs_padded, kernel, padding=0, groups=imgs.shape[1])

        return out.to(device_0)


    # 2. Median Filter (Non-Differentiable)
    def median_filter(self, imgs: Tensor, kernel_size: int = 3) -> Tensor:
        device_0 = imgs.device
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        pad = kernel_size // 2
        unfolded = F.unfold(imgs, kernel_size=kernel_size, padding=pad)
        unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)

        return unfolded.median(dim=2).values.to(device_0)


    # 3. Gaussian Filter (Differentiable)
    def gaussian_filter(self, imgs: Tensor, kernel_size: int = 3, sigma_k: float = 1.0) -> Tensor:
        device_0 = imgs.device
        coords = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma_k**2))
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(imgs.shape[1], 1, 1, 1)

        # Compute exact padding to maintain input size
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        imgs_padded = F.pad(imgs.to(self.device), (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        out = F.conv2d(imgs_padded, kernel, padding=0, groups=imgs.shape[1])

        return out.to(device_0)


    # 4. Pepper Noise + Median Filter (Non-Differentiable)
    def pepper_noise_median_filter(self, imgs: Tensor, kernel_size: int = 3, amount: float = 0.1) -> Tensor:
        self.init_hyperparam()
        device_0 = imgs.device
        imgs = imgs.to(self.device)
        imgs_noised = imgs.clone()
        pepper_mask = torch.rand_like(imgs_noised) < amount
        imgs_noised[pepper_mask] = 0.0

        return self.median_filter(imgs_noised, kernel_size).to(device_0)


    # 5. Gaussian Noise + Gaussian Filter (Differentiable)
    def gaussian_noise_gaussian_filter(self, imgs: Tensor, kernel_size: int = 3, sigma_y: float = 0.1, sigma_k: float = 1.0) -> Tensor:
        self.init_hyperparam()
        device_0 = imgs.device
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs) * sigma_y
        imgs_noised = imgs + noise

        return self.gaussian_filter(imgs_noised, kernel_size, sigma_k).to(device_0)


    # 6. JPEG Compression (Non-Differentiable)
    def jpeg_compress(self, imgs: Tensor, quality: int = 50) -> Tensor:
        device_0 = imgs.device
        imgs = imgs.clamp(0, 1)
        imgs_cpu = (imgs.detach().cpu() * 255).to(torch.uint8).permute(0, 2, 3, 1)  # B, H, W, C

        compressed = []
        for img in imgs_cpu:
            pil_img = Image.fromarray(img.numpy(), mode='RGB')
            with BytesIO() as buf:
                pil_img.save(buf, format='JPEG', quality=quality)
                buf.seek(0)
                rec = Image.open(buf).convert("RGB")
                rec_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(rec.tobytes()))
                rec_tensor = rec_tensor.view(rec.size[1], rec.size[0], 3).permute(2, 0, 1).float() / 255.0
                compressed.append(rec_tensor)

        return torch.stack(compressed).to(device_0)
        
    # 7. Brightness
    def adjust_brightness(self, imgs: Tensor, factor: float = 1.0) -> Tensor:
        """
        factor > 1.0 -> brighter, factor < 1.0 -> darker
        input/output: (B,C,H,W) float in [0,1]
        """
        device_0, dtype_0 = imgs.device, imgs.dtype
        pil_list = self._tensor01_to_pil_list(imgs)
        pil_list = [ImageEnhance.Brightness(p).enhance(float(factor)) for p in pil_list]
        return self._pil_list_to_tensor01(pil_list, device_0, dtype_0)

    # 8. Contrast
    def adjust_contrast(self, imgs: Tensor, factor: float = 1.0) -> Tensor:
        """
        factor > 1.0 -> higher contrast, factor < 1.0 -> lower contrast
        input/output: (B,C,H,W) float in [0,1]
        """
        device_0, dtype_0 = imgs.device, imgs.dtype
        pil_list = self._tensor01_to_pil_list(imgs)
        pil_list = [ImageEnhance.Contrast(p).enhance(float(factor)) for p in pil_list]
        return self._pil_list_to_tensor01(pil_list, device_0, dtype_0)

    # 9. Sharpen (Unsharp Mask)
    def sharpen(self, imgs: Tensor, radius: float = 2.0, percent: int = 150, threshold: int = 3) -> Tensor:
        """
        UnsharpMask: tends to make fine noise / texture more visible.
        input/output: (B,C,H,W) float in [0,1]
        """
        device_0, dtype_0 = imgs.device, imgs.dtype
        pil_list = self._tensor01_to_pil_list(imgs)
        pil_list = [
            p.filter(ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=int(threshold)))
            for p in pil_list
        ]
        return self._pil_list_to_tensor01(pil_list, device_0, dtype_0)

    # 10. Edges (Find edges)
    def edges(self, imgs: Tensor) -> Tensor:
        """
        Edge map visualization via PIL FIND_EDGES.
        input/output: (B,C,H,W) float in [0,1]
        """
        device_0, dtype_0 = imgs.device, imgs.dtype
        pil_list = self._tensor01_to_pil_list(imgs)
        pil_list = [p.filter(ImageFilter.FIND_EDGES) for p in pil_list]
        return self._pil_list_to_tensor01(pil_list, device_0, dtype_0)

    def purify(self, imgs: Tensor, return_mid = False) -> Tensor:
        if self.batch_size is None:
            self.batch_size = len(imgs)
        if self.method == 'Mean':
            imgs_purified = self.mean_filter(imgs, self.settings['kernel'])
        elif self.method == 'median':
            imgs_purified = self.median_filter(imgs, self.settings['kernel'])
        elif self.method == 'Gaussian--':
            imgs_purified = self.gaussian_filter(imgs, self.settings['kernel'], self.settings['sigma_k'])
        elif self.method == 'pepper_median':
            imgs_purified = self.pepper_noise_median_filter(imgs, self.settings['kernel'], self.settings['amount'])
        elif self.method == 'Gaussian':
            imgs_purified = self.gaussian_noise_gaussian_filter(imgs, self.settings['kernel'], self.settings['sigma_y'], self.settings['sigma_k'])
        elif self.method == 'JPEG':
            imgs_purified = self.jpeg_compress(imgs, self.settings['quality'])
        elif self.method == 'brightness':
            imgs_purified = self.adjust_brightness(imgs, self.settings['factor'])
        elif self.method == 'contrast':
            imgs_purified = self.adjust_contrast(imgs, self.settings['factor'])
        elif self.method == 'sharpen':
            imgs_purified = self.sharpen(
                imgs,
                radius=self.settings.get('radius', 2.0),
                percent=self.settings.get('percent', 150),
                threshold=self.settings.get('threshold', 3),
            )
        elif self.method == 'edges':
            imgs_purified = self.edges(imgs)
        # elif self.method == 'iwmf':
        #     from .iwmfdiff import IWMFDiff
        #     purifier = IWMFDiff(
        #         lambda_0 = self.settings['lambda_0'],
        #         seed = self.seed,
        #         )
        #     imgs_purified = purifier.iwmf(imgs)
        # elif self.method == 'iwmfdiff':
        #     from .iwmfdiff import IWMFDiff
        #     purifier = IWMFDiff(
        #         data = self.settings['data'],
        #         lambda_0 = self.settings['lambda_0'],
        #         timesteps = self.settings['timesteps'],
        #         denoise_steps = self.settings['denoise_steps'],
        #         batch_size = self.batch_size,
        #         seed = self.seed,
        #         )
        #     imgs_purified = purifier.purify(imgs)
        # elif self.method == 'diffpure_ddrm':
        #     from .iwmfdiff import IWMFDiff
        #     purifier = IWMFDiff(
        #         data = self.settings['data'],
        #         timesteps = self.settings['timesteps'],
        #         denoise_steps = self.settings['denoise_steps'],
        #         batch_size = self.batch_size,
        #         seed = self.seed,
        #         )
        #     imgs_purified = purifier.ddrm(imgs)
        elif self.method == 'DiffPure':
            from .diffpure import DiffPure
            purifier = DiffPure(
                data=self.settings['data'],
                timesteps=self.settings['timesteps'],
                seed=self.seed,
            )
            purifier.reset_counter()
            purifier.set_tag()
            imgs_purified = purifier.purify(imgs)
        elif self.method == 'DDIM':
            from .diffpure import DiffPureDDIM
            purifier = DiffPureDDIM(
                data=self.settings['data'],
                timesteps=self.settings['timesteps'],
                denoise_steps=self.settings['denoise_steps'],
                seed=self.seed,
            )
            imgs_purified = purifier.purify(imgs, return_mid)
        elif self.method == 'MimicDiffusion':
            from .mimicdiffusion import MimicDiffusionDefense
            purifier = MimicDiffusionDefense(
                data=self.settings['data'],
                timesteps=self.settings['timesteps'],
                denoise_steps=self.settings['denoise_steps'],
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == 'SSNI':
            from .ssni import SSNIDefense
            # Use same basic settings interface as DDIM/MimicDiffusion, plus SSNI-specific hyperparams.
            data = self.settings['data']
            # Default adv_eps follows common PGD budgets: 4/255 for ImageNet, 8/255 for CIFAR-10.
            default_adv_eps = 4.0 / 255.0 if data == 'imagenet' else 8.0 / 255.0
            purifier = SSNIDefense(
                data=data,
                timesteps=self.settings['timesteps'],
                denoise_steps=self.settings['denoise_steps'],
                tau=self.settings.get('tau', 1.0),
                bias=self.settings.get('bias', 50),
                adv_eps=self.settings.get('adv_eps', default_adv_eps),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == 'DCDefense':
            from .dc import DCDefense
            data = self.settings['data']
            # DC-specific hyperparameters, with reasonable defaults taken from their scripts.
            purifier = DCDefense(
                data=data,
                strength_l=self.settings.get('strength_l', 0.2),
                strength_s=self.settings.get('strength_s', 0.1),
                forward_noise_steps=self.settings.get('forward_noise_steps', 1),
                ddim_steps=self.settings.get('timesteps', 150),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        elif self.method == 'ContrastDiff':
            from .contrastdiff import ContrastDiffDefense
            data = self.settings['data']
            purifier = ContrastDiffDefense(
                data=data,
                t=self.settings.get('timesteps', 150),
                sample_step=self.settings.get('sample_step', 1),
                contrastive_classifier=self.settings.get('contrastive_classifier', 'imagenet-resnet50'),
                n_classes=self.settings.get('n_classes', 1000),
                optim_lr=self.settings.get('optim_lr', 0.01),
                drift_min=self.settings.get('drift_min', 40),
                drift_max=self.settings.get('drift_max', 100000),
                tau_plus=self.settings.get('tau_plus', 0.1),
                beta=self.settings.get('beta', 1.0),
                temperature=self.settings.get('temperature', 1.0),
                use_bm=self.settings.get('use_bm', False),
                seed=self.seed,
                device=self.device,
            )
            imgs_purified = purifier.purify(imgs)
        else:
            raise ValueError("Unsupported defense")

        return imgs_purified