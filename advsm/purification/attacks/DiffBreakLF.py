import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict

from ._bpda import BPDAModel


class _DiffBreakLikeModel(nn.Module):
    """
    Minimal model adapter to satisfy DiffBreak's attack APIs (LF/AutoAttack/etc).
    """

    def __init__(
        self,
        model: nn.Module,
        purifier: Optional[Any],
        bpda_mode: str,
        *,
        targeted: bool,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self._wrapped = BPDAModel(model, purifier, bpda_mode=bpda_mode, clip_min=clip_min, clip_max=clip_max)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.targeted = bool(targeted)

    def to(self, device):
        self._wrapped = self._wrapped.to(device)
        return self

    def eval(self):
        self._wrapped.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._wrapped(x)

    # Preserve callable behavior for places that call model(x) directly
    __call__ = forward

    def get_loss_fn(self):
        """
        DiffBreak's LF optimizer *minimizes* this loss.
        - targeted=False (untargeted attack): we want to maximize CE -> minimize -CE
        - targeted=True  (targeted attack):  minimize CE to the target label
        """
        if self.targeted:
            return lambda logits, y: F.cross_entropy(logits, y, reduction="none")
        return lambda logits, y: -F.cross_entropy(logits, y, reduction="none")

    def set_y_orig(self, _y_init):
        # Some DiffBreak losses use this; LF doesn't need it.
        return

    def eval_attack(self, x: torch.Tensor, label: torch.Tensor, targeted: bool = True):
        # DiffBreak wrapper returns (success_rate, successful_attack, successful_attack_single)
        with torch.no_grad():
            logits = self._wrapped(x)
            pred = logits.argmax(dim=1)
            if targeted:
                success = (pred == label).float()
            else:
                success = (pred != label).float()
            success_rate = success.mean()
            successful_attack_single = float(success.any().item())
            successful_attack = successful_attack_single
        return success_rate, float(successful_attack), float(successful_attack_single)

class DiffBreakLF:
    """
    Adaptive wrapper that reuses DiffBreak's original LF implementation and
    its official hyperparameters from `DiffBreak.DiffBreak.utils.registry.Registry`.

    Behavior:
    - All LF settings (eps, max_iterations, optimizer_args, filter_args, etc.)
      come from `Registry.attack_params(dataset_name, "LF")`.
    - Only CLI-controlled knobs from `attack_adaptive.py` override registry:
        * norm / eps (LPIPS vs Linf, bound value)
        * n_iter (max_iterations)
        * eot_iter (eot_iters)
        * bpda_mode (how we treat the purifier: "ste", "none", "skip")
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        norm: str = "LPIPS",
        eps: float = 0.05,
        n_iter: int = 100,
        eot_iter: int = 1,
        defense: Optional[Any] = None,
        bpda_mode: str = "ste",
        device: Optional[torch.device] = None,
        lf_optimizer_args: Optional[Dict[str, Any]] = None,
        lf_filter_args: Optional[Dict[str, Any]] = None,
        binary_search_steps: int = 1,
        abort_early: bool = False,
    ) -> None:
        from DiffBreak.DiffBreak import Registry
        import DiffBreak.DiffBreak.attacks.lf as lf_mod

        LF = lf_mod.LF

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.defense = defense
        # keep bpda_mode for API compatibility but do not use our BPDA wrapper
        self.bpda_mode = str(bpda_mode).lower()
        self._use_lpips = str(norm).upper() == "LPIPS"

        # Infer dataset name coarsely from classifier output dimension so we can
        # query the appropriate LF_params from the registry.
        num_classes = 10
        try:
            dummy = self.model(torch.zeros(1, 3, 32, 32, device=self.device))
            num_classes = int(dummy.shape[-1])
        except Exception:
            pass
        dataset_name = "cifar10" if num_classes <= 10 else "imagenet"

        attack_params = Registry.attack_params(dataset_name, "LF")

        # Respect CLI overrides while otherwise using registry defaults.
        if self._use_lpips and eps is not None:
            attack_params["eps"] = float(eps)
        if n_iter is not None:
            attack_params["max_iterations"] = int(n_iter)
        if eot_iter is not None:
            attack_params["eot_iters"] = int(eot_iter)

        # These are official optimizer/filter args unless explicitly overridden.
        if lf_optimizer_args is not None:
            attack_params["optimizer_args"] = lf_optimizer_args
        if lf_filter_args is not None:
            attack_params["filter_args"] = lf_filter_args

        # `attack_name` is used only by Runner; LF itself doesn't need it.
        attack_params.pop("attack_name", None)

        # LF expects `binary_search_steps` and `abort_early` among its kwargs.
        attack_params["binary_search_steps"] = int(binary_search_steps)
        attack_params["abort_early"] = bool(abort_early)

        model_fn = _DiffBreakLikeModel(
            self.model,
            self.defense,
            self.bpda_mode,
            targeted=False,
        ).to(self.device).eval()

        self._lf = LF(
            model=model_fn,
            **attack_params,
        ).to(self.device).eval()

    def perturb(self, x: torch.Tensor, y: torch.Tensor, best_adv: bool = True, mask: Optional[torch.Tensor] = None):
        x = x.to(self.device)
        y = y.to(self.device)

        # DiffBreak LF signature: (x, label, y_init). We pass y_init as one-hot of y.
        y_init = torch.nn.functional.one_hot(y, num_classes=int(self._lf.model(x[:1]).shape[-1])).float()

        adv, _succ, _single = self._lf(x, y, y_init)

        if mask is not None:
            m = mask.to(device=adv.device, dtype=adv.dtype)
            if m.shape != adv.shape:
                m = m.expand_as(adv)
            adv = x + (adv - x) * m

        return adv.clamp(0.0, 1.0).detach()

