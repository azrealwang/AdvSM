import torch
import torch.nn as nn
from typing import Optional, Any

from types import SimpleNamespace
from ._bpda import BPDAModel


class _DiffAttackModelAdapter(nn.Module):
    """
    Adapter for DiffAttack losses.

    DiffAttack's proposal loss may call `model(x, return_mid=True)` and expect
    (logits, mid_x, ori_x). For diffusion defenses (e.g., DiffPure/DDIM), our
    purifier can provide these via `purify(x, return_mid=True)`.
    """

    def __init__(self, model: nn.Module, purifier: Optional[Any], bpda_mode: str = "ste"):
        super().__init__()
        self.model = model
        self.purifier = purifier
        self.bpda_mode = str(bpda_mode).lower()
        # Default path: no mid info, just logits
        self._bpda = BPDAModel(model, purifier, bpda_mode=self.bpda_mode).eval()

    def forward(self, x: torch.Tensor, return_mid: bool = False):
        if not return_mid:
            return self._bpda(x)

        # Proposal mode: follow original DiffAttack behavior conceptually by treating
        # `purifier + classifier` as a single defended model and backpropagating
        # through both (no BPDA STE trick here).
        x_p, mid_x, ori_x = self.purifier.purify(x, return_mid=True)
        logits = self.model(x_p.clamp(0.0, 1.0))
        return logits, mid_x, ori_x


class DiffAttackBaseline:
    """
    Wrapper around the DiffAttack repo. Keeps DiffAttack's proposal (diffusion-aware
    loss: CE + MSE on mid-layer) with PGD as the inner backbone (fixed step, no APGD
    schedule). Uses the repo's 'apgd-pgd-ce' option.

    Requires a defense that supports return_mid (e.g. DDIM). args.t and t_interval are
    taken from defense.settings (timesteps / denoise_steps) so the proposal loss matches
    the purifier schedule. If return_mid is not supported, a RuntimeError is raised.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        norm: str = "Linf",
        n_iter: int = 100,
        eps: float = 8 / 255,
        eot_iter: Optional[int] = None,
        t_interval: Optional[int] = None,
        defense: Optional[Any] = None,
        bpda_mode: str = "ste",
        device: Optional[torch.device] = None,
    ) -> None:
        from DiffAttack.DiffAttack_Score_Based.diffattack.DiffAttack import DiffAttack

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.defense = defense
        self.bpda_mode = str(bpda_mode).lower()

        if defense is None:
            raise ValueError(
                "DiffAttack requires a defense that supports return_mid (e.g. DDIM). "
                "The core algorithm uses diffusion mid-layer outputs; without return_mid it cannot run."
            )

        # args.t = length of mid_x/ori_x. Only DDIM returns mid; its mid list length = denoise_steps.
        settings = getattr(defense, "settings", None)
        if not isinstance(settings, dict):
            settings = {}
        denoise = settings.get("denoise_steps", None)
        if denoise is not None:
            t_val = int(denoise[0]) if isinstance(denoise, (list, tuple)) else int(denoise)
        else:
            t_val = settings.get("timesteps", 100)
            t_val = int(t_val[0]) if isinstance(t_val, (list, tuple)) else int(t_val)
        # ~10 indices for proposal loss: when t_val>10 use t_val//10, else step every 1.
        if t_interval is None:
            t_interval_eff = max(1, t_val // 10) if t_val > 10 else 1
        else:
            t_interval_eff = int(t_interval)
        args_for_loss = SimpleNamespace(t=t_val, t_interval=t_interval_eff)

        wrapped = _DiffAttackModelAdapter(self.model, self.defense, bpda_mode=self.bpda_mode).to(self.device).eval()
        self._engine = DiffAttack(
            model=wrapped,
            norm=str(norm),
            eps=float(eps),
            seed=None,
            verbose=False,
            attacks_to_run=["apgd-pgd-ce"],
            version="custom",
            is_tf_model=False,
            device=str(self.device),
            log_path=None,
            args=args_for_loss,
        )
        # Ensure apgd_pgd always has args so use_proposal is True in _compute_grad_and_loss.
        self._engine.apgd_pgd.args = args_for_loss
        self._engine.apgd_pgd.eot_iter = int(eot_iter if eot_iter is not None else 20)
        self._engine.apgd_pgd.n_iter = int(n_iter)

    def perturb(self, x: torch.Tensor, y: torch.Tensor, best_adv: bool = True, mask: Optional[torch.Tensor] = None):
        x = x.to(self.device)
        y = y.to(self.device)

        # DiffAttack expects plain labels
        adv = self._engine.run_standard_evaluation(x_orig=x, y_orig=y, bs=max(1, x.shape[0]), return_labels=False)

        if mask is not None:
            m = mask.to(device=adv.device, dtype=adv.dtype)
            if m.shape != adv.shape:
                m = m.expand_as(adv)
            adv = x + (adv - x) * m

        return adv.clamp(0.0, 1.0).detach()

