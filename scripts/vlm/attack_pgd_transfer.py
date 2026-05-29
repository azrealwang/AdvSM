#!/usr/bin/env python3
"""PGDTransfer: image-space PGD on LVLM VQA pipelines."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import _path  # noqa: F401
from advsm.vlm.attacks.pgd_vqa import pgd_linf_vqa
from advsm.vlm.data import VQAJsonlDataset, collate_vqa_batch, safe_vqa_id_for_filename
from advsm.vlm.eval.vqa_metrics import exact_match_correct
from advsm.vlm.models import load_model_from_config
from advsm.vlm.utils.config import save_yaml
from advsm.vlm.utils.logging import get_logger

log = get_logger()

ADV_IMAGES_SUBDIR = "adv_images"


def parse_frac(s: str) -> float:
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def parse_args():
    p = argparse.ArgumentParser(description="PGD attack in image space (teacher-forcing NLL).")
    p.add_argument("--source", required=True)
    p.add_argument("--models-config", required=True)
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eps", type=parse_frac, default=4 / 255, help="Linf ε (default 4/255)")
    p.add_argument("--alpha", type=parse_frac, default=1 / 255)
    p.add_argument("--steps", type=int, default=100, help="PGD iterations (default 100)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--attack-clean-correct-only", action="store_true")
    p.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="0-based start index among accepted jsonl rows (after subset); inclusive.",
    )
    p.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="0-based end index among accepted rows; exclusive (same as Python slice). "
        "Omit to run through end of data. E.g. --start-idx 0 --end-idx 10 → 10 rows.",
    )
    p.add_argument(
        "--subset-file",
        default=None,
        metavar="PATH",
        help="If set, only keep jsonl rows whose question id is listed in this file "
        "(one id per line; leading token before whitespace, or JSON object with \"id\").",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-random-start",
        action="store_true",
        help="Disable random Linf initialization for PGD (default: random start on).",
    )
    p.add_argument(
        "--log_loss",
        action="store_true",
        help="Log teacher-forcing NLL at every PGD step (verbose).",
    )
    return p.parse_args()


def _to_bgr_uint8(x01_chw: torch.Tensor) -> np.ndarray:
    x = (x01_chw.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))[:, :, ::-1]
    return x


def main():
    args = parse_args()
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)
    adv_img_root = out_dir / ADV_IMAGES_SUBDIR
    adv_img_root.mkdir(parents=True, exist_ok=True)
    model = load_model_from_config(args.source, args.models_config)
    model.assert_attackable()
    device_str = getattr(model, "_device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(device_str)

    ds = VQAJsonlDataset(
        args.data_jsonl,
        image_root=args.image_root,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        subset_file=args.subset_file,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vqa_batch)
    random_start = not args.no_random_start

    meta_rows = []
    batch_idx = 0
    for batch in tqdm(dl, desc=f"attack {args.source}", unit="batch"):
        imgs = batch["image"].to(dev)
        qs = batch["question"]
        ans = batch["answer"]
        answers_list = batch["answers"]
        with torch.no_grad():
            clean_preds = model.generate_answer(imgs, qs)
        mask = []
        for i, cp in enumerate(clean_preds):
            cc = exact_match_correct(cp, answers_list[i])
            mask.append(cc if args.attack_clean_correct_only else True)
        if args.attack_clean_correct_only and not any(mask):
            continue
        if args.attack_clean_correct_only:
            sub = [i for i, m in enumerate(mask) if m]
            imgs_a = imgs[sub]
            qs_a = [qs[i] for i in sub]
            ans_a = [ans[i] for i in sub]
            answers_multi_a = [answers_list[i] for i in sub]
            ids_a = [batch["id"][i] for i in sub]
            paths_a = [batch["image_path"][i] for i in sub]
            clean_preds_a = [clean_preds[i] for i in sub]
        else:
            imgs_a, qs_a, ans_a, answers_multi_a = imgs, qs, ans, answers_list
            ids_a, paths_a = batch["id"], batch["image_path"]
            clean_preds_a = list(clean_preds)

        adv = pgd_linf_vqa(
            model,
            imgs_a,
            qs_a,
            ans_a,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            random_start=random_start,
            log_each_step=args.log_loss,
            log_prefix=f"batch={batch_idx} ",
        )
        with torch.no_grad():
            adv_preds = model.generate_answer(adv, qs_a)

        for i in range(len(qs_a)):
            ext = Path(paths_a[i]).suffix or ".png"
            dest = adv_img_root / f"{safe_vqa_id_for_filename(ids_a[i])}{ext}"
            cv2.imwrite(str(dest), _to_bgr_uint8(adv[i]))
            adv_rel = dest.relative_to(out_dir).as_posix()
            linf = (adv[i] - imgs_a[i]).abs().max().item()
            l2 = torch.norm((adv[i] - imgs_a[i]).flatten(), p=2).item()
            cc = exact_match_correct(clean_preds_a[i], answers_multi_a[i])
            ac = exact_match_correct(adv_preds[i], answers_multi_a[i])
            meta_rows.append(
                {
                    "id": ids_a[i],
                    "image_path": paths_a[i],
                    "adv_image_rel": adv_rel,
                    "question": qs_a[i],
                    "answer": ans_a[i],
                    "source_model": args.source,
                    "source_clean_pred": clean_preds_a[i],
                    "source_clean_correct": int(cc),
                    "source_adv_pred": adv_preds[i],
                    "source_adv_correct": int(ac),
                    "linf": linf,
                    "l2": l2,
                }
            )

        batch_idx += 1

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "image_path",
                "adv_image_rel",
                "question",
                "answer",
                "source_model",
                "source_clean_pred",
                "source_clean_correct",
                "source_adv_pred",
                "source_adv_correct",
                "linf",
                "l2",
            ],
        )
        w.writeheader()
        w.writerows(meta_rows)

    save_yaml(
        {
            "source": args.source,
            "eps": float(args.eps),
            "alpha": float(args.alpha),
            "steps": args.steps,
            "random_start": random_start,
            "log_loss": args.log_loss,
            "attack_clean_correct_only": args.attack_clean_correct_only,
            "batch_size": args.batch_size,
            "data_jsonl": str(args.data_jsonl),
            "image_root": args.image_root,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "subset_file": args.subset_file,
            "num_batches": batch_idx,
            "adv_images_subdir": ADV_IMAGES_SUBDIR,
            "adv_images_layout": "per_question_id",
        },
        out_dir / "attack_config.yaml",
    )
    log.info("Wrote %d batches under %s (adv_images/ one file per question id)", batch_idx, out_dir)


if __name__ == "__main__":
    main()
