#!/usr/bin/env python3
"""Adaptive attacks on purification pipelines (paper Table 4 / 11)."""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from advsm._paths import ensure_repo_on_path, ensure_third_party_on_path

ensure_repo_on_path()
ensure_third_party_on_path()

import argparse
from time import time

import torch
from torch import Tensor
from tqdm import tqdm

from advsm.classification.utils import (
    load_one_model,
    load_samples,
    parse_d_settings_arg,
    predict,
    save_all_images,
)
from advsm.purification import Purifier
from advsm.purification.attacks import (
    DiffAttackBaseline,
    DiffBreakLF,
    DiffHammerEM,
    PGDTransfer,
)

PURIFIER_ATTACKS = (
    "PGD",
    "BPDA_EOT",
    "DiffPGD",
    "DiffAttack",
    "DiffHammer",
    "DiffBreak",
    "PGDTransfer",
)


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=PURIFIER_ATTACKS,
        help="PGD, BPDA_EOT, DiffPGD, DiffAttack, DiffHammer, DiffBreak, PGDTransfer",
    )
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=4, help="Linf ε in 1/255 units (default 4 → 4/255)")
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--eot_iter", type=int, default=5)
    parser.add_argument("--n_eval", type=int, default=3, help="DiffHammer N_EVAL")
    parser.add_argument("--t_interval", type=int, default=None, help="DiffAttack proposal interval")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--defense", type=str, default=None)
    parser.add_argument(
        "--d_settings",
        nargs="+",
        default=None,
        metavar="KV",
        help="Purifier kwargs as name value pairs, e.g. data imagenet timesteps 150 denoise_steps 3",
    )
    parser.add_argument("--data", type=str, required=True, choices=("cifar10", "imagenet"))
    parser.add_argument("--target", type=str, required=True, help="Classifier behind purifier (RobustBench id or paper name)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_attack(args, target, purifier, eps):
    common = dict(
        model=target,
        targeted=args.targeted,
        n_iter=args.max_iter,
        norm=args.norm,
        eps=eps,
    )
    if args.attack == "PGD":
        return PGDTransfer(**common, bpda_mode="skip")
    if args.attack == "BPDA_EOT":
        return PGDTransfer(
            **common,
            eot_iter=args.eot_iter,
            eot_mode="iter",
            defense=purifier,
            bpda_mode="ste",
        )
    if args.attack == "DiffPGD":
        return PGDTransfer(**common, defense=purifier, bpda_mode="none")
    if args.attack == "PGDTransfer":
        return PGDTransfer(
            **common,
            eot_iter=args.eot_iter,
            eot_mode="iter",
            defense=purifier,
            bpda_mode="none",
        )
    if args.attack == "DiffAttack":
        return DiffAttackBaseline(
            model=target,
            n_iter=args.max_iter,
            norm=args.norm,
            eps=eps,
            eot_iter=args.eot_iter,
            t_interval=args.t_interval,
            defense=purifier,
            bpda_mode="none",
        )
    if args.attack == "DiffHammer":
        return DiffHammerEM(
            model=target,
            n_iter=args.max_iter,
            norm=args.norm,
            eps=eps,
            n_eval=args.n_eval,
            defense=purifier,
            bpda_mode="none",
        )
    if args.attack == "DiffBreak":
        eps_db = float(args.eps) if args.norm == "LPIPS" else eps
        return DiffBreakLF(
            model=target,
            norm=args.norm,
            eps=eps_db,
            n_iter=args.max_iter,
            eot_iter=args.eot_iter,
            defense=purifier,
            bpda_mode="none",
        )
    raise ValueError(args.attack)


def main() -> None:
    args = parse_args_and_config()
    eps = args.eps / 255 if args.norm == "Linf" else args.eps
    d_settings = parse_d_settings_arg(args.d_settings)

    target = load_one_model(args.data, args.target, threat_model="Linf").eval()
    x_test, y_test = load_samples(args.input, args.start_idx, args.end_idx)
    x_test = Tensor(x_test)
    y_test = Tensor(y_test).long()
    batch_size = args.batch_size or 1

    if args.defense:
        purifier = Purifier(args.defense, d_settings, batch_size, args.seed)
        x_purified = []
        for start in tqdm(
            range(0, len(y_test), batch_size),
            desc=f"purify clean ({args.defense})",
            unit="batch",
        ):
            idx = slice(start, min(start + batch_size, len(y_test)))
            x_purified.append(purifier.purify(x_test[idx]).detach().cpu())
        x_purified = torch.cat(x_purified, 0)
    else:
        purifier = None
        x_purified = x_test.clone()

    if args.targeted:
        num_classes = {"cifar10": 10, "imagenet": 1000}[args.data]
        eval_labels = (y_test + 1) % num_classes
    else:
        eval_labels = y_test

    predictions = predict(target, x_purified, batch_size=batch_size)
    acc = (predictions.max(1)[1] == eval_labels).float().mean()
    print(f"Clean accuracy (after purifier): {acc:.4f}")

    attack = build_attack(args, target, purifier, eps)
    os.makedirs(args.output, exist_ok=True)
    t0 = time()
    attack_label = f"{args.attack}"
    if args.defense:
        attack_label += f" / {args.defense}"
    for start in tqdm(
        range(0, len(y_test), batch_size),
        desc=f"attack {attack_label}",
        unit="batch",
    ):
        end = min(start + batch_size, len(y_test))
        idx = slice(start, end)
        x_adv = attack.perturb(x=x_test[idx], y=eval_labels[idx]).detach().cpu()
        save_all_images(x_adv, eval_labels[idx], args.output, args.start_idx + start)
    print(f"Attack time: {time() - t0:.1f}s")

    x_adv_load, _ = load_samples(args.output, args.start_idx, args.end_idx)
    x_adv_load = Tensor(x_adv_load)
    d_linf = (x_adv_load - x_test).abs().max() * 255
    if args.defense:
        x_pur = []
        for start in tqdm(
            range(0, len(y_test), batch_size),
            desc=f"purify adv ({args.defense})",
            unit="batch",
        ):
            idx = slice(start, min(start + batch_size, len(y_test)))
            x_pur.append(purifier.purify(x_adv_load[idx]).detach().cpu())
        x_eval = torch.cat(x_pur, 0)
    else:
        x_eval = x_adv_load
    predictions = predict(target, x_eval, batch_size=batch_size)
    acc = (predictions.max(1)[1] == eval_labels).float().mean()
    print(f"Robust accuracy: {acc:.4f}; Linf max perturbation: {d_linf:.4f}")


if __name__ == "__main__":
    main()
