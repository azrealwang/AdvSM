import argparse
import logging
from types import SimpleNamespace
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._bpda import BPDAModel


class _NullLogger:
    def info(self, *_args, **_kwargs):
        return

    def debug(self, *_args, **_kwargs):
        return

    def warning(self, *_args, **_kwargs):
        return


class _DiffHammerModelAdapter(nn.Module):
    """
    Provide the minimal `forward()` + `gradient()` interface expected by DiffHammer's attacker code.
    """

    def __init__(self, model: nn.Module, purifier: Optional[Any], bpda_mode: str, clip_min: float = 0.0, clip_max: float = 1.0):
        super().__init__()
        self._wrapped = BPDAModel(model, purifier, bpda_mode=bpda_mode, clip_min=clip_min, clip_max=clip_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._wrapped(x)

    def gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn,
        grad_mode: str,
        seed,
        aug=None,
        em_grad=None,
    ):
        # DiffHammer's code uses `seed` to control stochastic defenses; we set torch RNGs.
        # Some call sites may pass lists/tensors; normalize to an int.
        if isinstance(seed, (list, tuple)):
            seed_val = int(seed[0])
        elif isinstance(seed, torch.Tensor):
            seed_val = int(seed.detach().cpu().item())
        else:
            seed_val = int(seed)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        x_in = x
        if aug is not None:
            x_in = aug(x_in)

        x_in = x_in.detach().requires_grad_(True)
        logits = self._wrapped(x_in)
        # DiffHammer losses typically return per-sample loss
        loss_vec = loss_fn(logits, y)
        if loss_vec.dim() == 0:
            loss_vec = loss_vec.view(1)
        loss = loss_vec.sum()
        grad = torch.autograd.grad(loss, x_in, create_graph=False, retain_graph=False)[0].detach()
        return grad, logits.detach(), loss_vec.detach()


class DiffHammerEM:
    """
    DiffHammer EM attack: their proposed EM (gradient selection / refinement) with
    PGD as the inner update, consistent with other baselines (DiffAttack, etc.).
    We use DiffHammer's PGD class with EM=True so base Attacker.perturb() runs the
    EM path; PGD's native init_params and update_adv are used (no APGD/monkey-patch).

    Final behavior:
    - iteration, eps (Linf), and eot are customizable from attack_adaptive.py (--max_iter,
      --eps, --norm, --eot_iter). Linf eps is passed as eps/255. --eot_iter sets N_EVAL
      (number of gradient samples for EM to choose from); N_EOT is fixed to 1 to match
      original DiffHammer EM (attack step uses a single gradient selected by EM, not
      an average over EOT).
    - Always runs DiffHammerEM algorithm with PGD backbone (fixed step 1.2*eps/n_iter).
    - No OOM: gradients come from autograd.grad(loss, x_in) with retain_graph=False,
      and defense is wrapped in BPDA, so no full backward through the diffusion model.
    - Logic matches original DiffHammer EM except the inner update is PGD (not APGD);
      N_EOT=1 and N_EVAL from config, EM(), em_grad/attack_seeds, and BPDA gradient flow
      are unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        n_iter: int = 100,
        norm: str = "Linf",
        eps: float = 8 / 255,
        n_eval: int = 10,
        defense: Optional[Any] = None,
        bpda_mode: str = "ste",
        device: Optional[torch.device] = None,
    ) -> None:
        from DiffHammer.attacker import PGD as DiffHammerPGD
        from DiffHammer.utils import get_loss_func

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device).eval()
        self.defense = defense
        self.bpda_mode = str(bpda_mode).lower()

        # Use PGD class with EM=True: same base perturb() with EM path, native PGD update.
        # Original DiffHammer EM uses N_EOT=1 (attack step = single gradient from EM) and
        # N_EVAL = number of candidate gradients for EM to select/refine. So eot_iter sets
        # N_EVAL only; N_EOT is fixed to 1 to match the paper/repo.
        attack_cfg = {
            "RESUME": False,
            "METHOD": "pgd",
            "N_ITERS": [int(n_iter)],
            "NORM": "inf" if str(norm).upper() == "LINF" else "2",
            "N_RESTART": 1,
            "RESTART_THR": 0.0,
            "N_EOT": 1,
            "N_EVAL": int(max(1, n_eval)),
            "EPS": float(eps),
            "LOSS_NAMES": ["CE"],
            "GRAD_MODE": "bpda",
            "PGD_CMD": "",
            "PGD_STEP_SIZE": 0.0,
            "EM": True,
            "EM_ALPHA": 0.5,
            "EM_LAM": 5.0,
            "EM_STEPS": 5,
        }
        data_cfg = SimpleNamespace(BATCH_SIZE=1, NUM=1)
        cfg = SimpleNamespace(ATTACK=attack_cfg, DATA=data_cfg, OUTPUT_DIR="", EXP="", DEFENSE=SimpleNamespace(METHOD=""), NAME="")

        self._loss = get_loss_func("CE")
        self._dh_model = _DiffHammerModelAdapter(self._model, self.defense, self.bpda_mode).to(self.device).eval()
        self._logger = _NullLogger()
        self._seeder = iter(range(int(1e9)))
        self._attacker_class = DiffHammerPGD
        self._cfg = cfg

    def perturb(self, x: torch.Tensor, y: torch.Tensor, best_adv: bool = True, mask: Optional[torch.Tensor] = None):
        from DiffHammer.utils import judge_success, set_seed

        x = x.to(self.device)
        y = y.to(self.device)
        B = x.shape[0]

        self._cfg.DATA.BATCH_SIZE = int(B)
        self._cfg.DATA.NUM = int(B)

        attacker = self._attacker_class(self._dh_model, self._cfg, self._logger, self._seeder)
        attacker.loss = self._loss
        attacker.n_iter = int(self._cfg.ATTACK["N_ITERS"][0])
        attacker.n_eval = int(self._cfg.ATTACK["N_EVAL"])
        attacker.n_eot = int(self._cfg.ATTACK["N_EOT"])
        attacker.eps = float(self._cfg.ATTACK["EPS"])
        attacker.em = True
        attacker.grad_mode = "bpda"
        assert attacker.em, "DiffHammerEM must run with EM enabled (attacker.em True)"

        # PGD class already has init_params and update_adv; no monkey-patch needed.
        # Monkey-patch: simple update_batch_metrics that doesn't rely on attacker.data_log,
        # since in this wrapper we don't track full experiment metrics. We still want
        # x_adv_best[idx] to be updated for all idxs.
        def simple_update_batch_metrics(success, loss, idxs):
            return [(i, idx) for i, idx in enumerate(idxs)]

        attacker.update_batch_metrics = simple_update_batch_metrics

        attacker.batch = 0
        attacker.restart = 0
        attacker.B = B
        attacker.N = B
        attacker.n_batch = 1

        attacker.x_adv = attacker.initialize(x)

        # Build initial log with EM (grads + loss) so base perturb() can run EM()
        log = {}
        attacker.eval_dict = {"seeds": [], "grads": [], "loss": []}
        for _ in range(attacker.n_eval):
            eval_seed = next(attacker.seeder)
            attacker.eval_dict["seeds"].append(eval_seed)
            g, logits, loss_vec = attacker.model.gradient(attacker.x_adv, y, attacker.loss, "bpda", eval_seed)
            attacker.eval_dict["grads"].append(g.view(B, -1))
            attacker.eval_dict["loss"].append(loss_vec.detach())
            success = judge_success(logits, y)
            log = attacker.log_metrics(log, logits, loss_vec, success)
        attacker.eval_dict["grads"] = torch.stack(attacker.eval_dict["grads"])
        attacker.eval_dict["loss"] = torch.stack(attacker.eval_dict["loss"])
        attacker.EM(attacker.eval_dict)
        # EM() sets attacker.em_grad and attacker.attack_seeds for the first perturb() iteration.
        assert hasattr(attacker, "em_grad") and hasattr(attacker, "attack_seeds"), "EM attack requires em_grad and attack_seeds from initial EM()"

        # Some DiffHammer versions expect `metric_str` on the attacker for logging;
        # provide a minimal placeholder to avoid attribute errors.
        if not hasattr(attacker, "metric_str"):
            attacker.metric_str = ""

        idxs = list(range(B))
        x_adv, _log = attacker.perturb(x, y, log, idxs, y_target=None)

        if mask is not None:
            m = mask.to(device=x_adv.device, dtype=x_adv.dtype)
            if m.shape != x_adv.shape:
                m = m.expand_as(x_adv)
            x_adv = x + (x_adv - x) * m

        return x_adv.clamp(0.0, 1.0).detach()

