import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any

from ._bpda import BPDAModel


class PGDTransfer:
    def __init__(
        self,
        model: nn.Module,
        *,
        targeted: bool = False,
        n_iter: int = 40,
        norm: str = "Linf",          # "Linf", "L2", "L1"
        eps: float = 8 / 255,
        alpha: Optional[float] = 1/255,
        seed: Optional[int] = None,
        eot_iter: int = 1,
        eot_mode: str = "repeat",      # "repeat" (parallel) or "iter" (sequential, low-mem)
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        rand_init: bool = True,
        defense: Optional[Any] = None,
        bpda_mode: str = "ste",        # "ste" (BPDA, low-mem), "none" (true grad), "skip" (no defense)
    ) -> None:
        self.model = model
        self.targeted = targeted
        self.n_iter = int(n_iter)
        self.norm = self._canon_norm(norm)
        self.eps = float(eps)
        self.alpha = float(alpha) if alpha is not None else None
        self.seed = seed
        self.eot_iter = int(eot_iter)
        self.eot_mode = str(eot_mode).lower()
        if self.eot_mode not in ("repeat", "iter"):
            raise ValueError("eot_mode must be 'repeat' or 'iter'")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.rand_init = bool(rand_init)
        self.defense = defense
        self.bpda_mode = str(bpda_mode).lower()
        if self.bpda_mode not in ("ste", "none", "skip"):
            raise ValueError("bpda_mode must be one of: 'ste', 'none', 'skip'")

        if self.n_iter <= 0:
            raise ValueError("n_iter must be > 0")
        if self.eot_iter <= 0:
            raise ValueError("eot_iter must be > 0")
        if self.eps < 0:
            raise ValueError("eps must be >= 0")

        if self.alpha is None:
            self.alpha = {"Linf": self.eps / 4.0, "L2": self.eps / 3.0, "L1": self.eps / 3.0}[self.norm]

        self.model = self.model.to(self.device)
        self.model.eval()
        self._wrapped = BPDAModel(
            self.model,
            self.defense,
            bpda_mode=self.bpda_mode,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        ).to(self.device).eval()

    @staticmethod
    def _canon_norm(norm: str) -> str:
        n = norm.upper()
        if n in ("LINF", "L∞"):
            return "Linf"
        if n == "L2":
            return "L2"
        if n == "L1":
            return "L1"
        raise ValueError("norm must be one of: 'Linf', 'L2', 'L1'")

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def _broadcast_mask(mask: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Return mask broadcasted to x shape (N,C,H,W). If mask is None, returns ones."""
        if mask is None:
            return torch.ones_like(x)
        m = mask.to(device=x.device, dtype=x.dtype)
        if m.dim() == 2:  # (H,W)
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:  # (N,H,W) or (C,H,W)
            if m.shape[0] == x.shape[0]:
                m = m.unsqueeze(1)  # (N,1,H,W)
            else:
                m = m.unsqueeze(0)  # (1,C,H,W)
        elif m.dim() == 4:
            pass
        else:
            raise ValueError("mask must be None, (H,W), (N,H,W), (N,1,H,W), or (N,C,H,W)")
        # broadcast to (N,C,H,W)
        if m.shape[0] == 1 and x.shape[0] != 1:
            m = m.expand(x.shape[0], -1, -1, -1)
        if m.shape[1] == 1 and x.shape[1] != 1:
            m = m.expand(-1, x.shape[1], -1, -1)
        if m.shape != x.shape:
            raise ValueError(f"mask broadcast failed: mask {tuple(m.shape)} vs x {tuple(x.shape)}")
        return m

    @staticmethod
    def _project_linf(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
        return x + torch.clamp(x_adv - x, min=-eps, max=eps)

    @staticmethod
    def _project_l2(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
        delta = x_adv - x
        flat = delta.view(delta.shape[0], -1)
        n = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factor = torch.clamp(eps / n, max=1.0)
        return x + (flat * factor).view_as(delta)

    @staticmethod
    def _l1_ball_projection(v: torch.Tensor, eps: float) -> torch.Tensor:
        if eps <= 0:
            return torch.zeros_like(v)
        abs_v = v.abs()
        u, _ = torch.sort(abs_v, dim=1, descending=True)
        cssv = torch.cumsum(u, dim=1) - eps
        ind = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype).view(1, -1)
        cond = u - cssv / ind > 0
        rho = cond.sum(dim=1).clamp_min(1)
        rho_idx = (rho - 1).long().view(-1, 1)
        theta = cssv.gather(1, rho_idx) / rho.view(-1, 1).to(v.dtype)
        return torch.sign(v) * torch.clamp(abs_v - theta, min=0.0)

    @classmethod
    def _project_l1(cls, x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
        delta = x_adv - x
        flat = delta.view(delta.shape[0], -1)
        proj = cls._l1_ball_projection(flat, eps).view_as(delta)
        return x + proj

    @classmethod
    def _rand_init(cls, x: torch.Tensor, norm: str, eps: float) -> torch.Tensor:
        if norm == "Linf":
            return x + torch.empty_like(x).uniform_(-eps, eps)
        if norm == "L2":
            delta = torch.randn_like(x)
            flat = delta.view(delta.shape[0], -1)
            n = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            r = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return x + (flat / n * (r * eps)).view_as(x)
        if norm == "L1":
            delta = torch.randn_like(x)
            flat = delta.view(delta.shape[0], -1)
            l1 = torch.norm(flat, p=1, dim=1, keepdim=True).clamp_min(1e-12)
            flat = flat / l1
            r = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return x + (flat * (r * eps)).view_as(x)
        raise ValueError(f"Unsupported norm: {norm}")

    def _project(self, x_adv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "Linf":
            return self._project_linf(x_adv, x, self.eps)
        if self.norm == "L2":
            return self._project_l2(x_adv, x, self.eps)
        return self._project_l1(x_adv, x, self.eps)

    def perturb(self, x: torch.Tensor, y: torch.Tensor, best_adv: bool = True, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Attack a batch: x (N,C,H,W), y (N,)
        EOT can be computed either in parallel by repeating the batch (fast, higher memory) or sequentially (low memory) depending on `eot_mode`.
        """
        self.model.eval()
        self._set_seed(self.seed)

        if x.dim() != 4:
            raise ValueError("x must be (N,C,H,W)")
        if y.dim() != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be (N,) matching x")

        x = x.to(self.device)
        y = y.to(self.device)
        mask_b = self._broadcast_mask(mask, x)

        x_adv = self._rand_init(x, self.norm, self.eps) if self.rand_init else x.clone()
        x_adv = self._clamp(x_adv, self.clip_min, self.clip_max)
        x_adv = x + (x_adv - x) * mask_b
        x_adv = self._project(x_adv, x)
        x_adv = x + (x_adv - x) * mask_b
        x_adv = self._clamp(x_adv, self.clip_min, self.clip_max).detach()
        
        K = self.eot_iter
        N = x.shape[0]
        
        best_score = -float("inf")     # after sign flip, always maximize
        best_x_adv = x_adv.detach().clone()
        
        for it in range(self.n_iter):
            x_adv.requires_grad_(True)
        
            if self.eot_mode == "repeat":
                x_rep = x_adv.repeat_interleave(K, dim=0)
                y_rep = y.repeat_interleave(K, dim=0)
        
                logits = self._wrapped(x_rep)
                ce = F.cross_entropy(logits, y_rep, reduction="none")   # (N*K,)
        
                ce_per_img = ce.view(N, K).mean(dim=1)                  # (N,)
                score = ce_per_img.mean()                               # scalar
                score = -score if self.targeted else score              # maximize always
        
                grad = torch.autograd.grad(score, x_adv, create_graph=False)[0]
        
            else:
                grad_accum = torch.zeros_like(x_adv)
                score_accum = 0.0
        
                for _k in range(K):
                    logits = self._wrapped(x_adv)
                    # scalar score_k
                    score_k = F.cross_entropy(logits, y, reduction="none").mean()
                    score_k = -score_k if self.targeted else score_k
                    score_accum = score_accum + score_k
                    gk = torch.autograd.grad(score_k, x_adv, create_graph=False)[0]
                    grad_accum = grad_accum + gk
        
                score = score_accum / float(K)
                grad = grad_accum / float(K)
            
            # ---- track best (no revert) ----
            if best_adv and (score.item() > best_score):
                best_score = score.item()
                best_x_adv = x_adv.detach().clone()
            # print(f"[{it+1:03d}/{self.n_iter}] Best: {best_score}; this: {score.item()}")
            
            with torch.no_grad():
                if self.norm == "Linf":
                    x_adv = x_adv + self.alpha * grad.sign()
                elif self.norm == "L2":
                    g = grad.view(N, -1)
                    g_norm = torch.norm(g, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    x_adv = x_adv + self.alpha * (g / g_norm).view_as(grad)
                else:
                    x_adv = x_adv + self.alpha * grad.sign()
        
                x_adv = x + (x_adv - x) * mask_b
                x_adv = self._project(x_adv, x)
                x_adv = x + (x_adv - x) * mask_b
                x_adv = self._clamp(x_adv, self.clip_min, self.clip_max).detach()
        
        return best_x_adv if best_adv else x_adv
