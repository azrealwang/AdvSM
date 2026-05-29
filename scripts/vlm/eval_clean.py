#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import _path  # noqa: F401
from advsm.vlm.data import VQAJsonlDataset, collate_vqa_batch
from advsm.vlm.eval.vqa_metrics import exact_match_correct, vqav2_soft_score
from advsm.vlm.models import load_model_from_config
from advsm.vlm.utils.logging import get_logger

log = get_logger()


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VQA on a jsonl. --image-root is always the image directory; "
        "use --adv to load per-question adversarial files ({id}{suffix}) as written by attack_pgd."
    )
    p.add_argument("--models", nargs="+", required=True, help="One or more model keys from the YAML config.")
    p.add_argument("--models-config", required=True)
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument(
        "--adv",
        action="store_true",
        help="Load images as image_root/{safe_id(id)}{suffix from jsonl image field} (attack adv_images/).",
    )
    p.add_argument("--output", required=True)
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
        help="0-based end index among accepted rows; exclusive. Omit for rest of data.",
    )
    p.add_argument(
        "--subset-file",
        default=None,
        metavar="PATH",
        help="If set, only keep jsonl rows whose question id is listed in this file "
        "(one id per line; leading token before whitespace, or JSON object with \"id\").",
    )
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    ds = VQAJsonlDataset(
        args.data_jsonl,
        image_root=args.image_root,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        subset_file=args.subset_file,
        adv=args.adv,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vqa_batch)

    summaries = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        log.info("Loading %s", model_name)
        model = load_model_from_config(model_name, args.models_config)
        device_str = getattr(model, "_device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        else:
            dev = torch.device(device_str)
        rows = []
        correct = 0
        tot_score = 0.0
        n = 0
        for batch in tqdm(dl, desc=f"eval:{model_name}", unit="batch"):
            imgs = batch["image"].to(dev)
            preds = model.generate_answer(imgs, batch["question"])
            for i, pred in enumerate(preds):
                ok = exact_match_correct(pred, batch["answers"][i])
                sc = vqav2_soft_score(pred, batch["answers"][i])
                if ok:
                    correct += 1
                tot_score += sc
                n += 1
                rows.append(
                    {
                        "id": batch["id"][i],
                        "image_path": batch["image_path"][i],
                        "question": batch["question"][i],
                        "answer": batch["answer"][i],
                        "prediction": pred,
                        "correct": int(ok),
                        "score": sc,
                    }
                )
        per_path = out_path.parent / f"{out_path.stem}_{model_name}{out_path.suffix}"
        with open(per_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["id", "image_path", "question", "answer", "prediction", "correct", "score"],
            )
            w.writeheader()
            w.writerows(rows)
        acc = correct / max(n, 1)
        summaries.append(
            {"model": model_name, "accuracy": acc, "avg_score": tot_score / max(n, 1), "num_samples": n}
        )
        log.info("model=%s accuracy=%.4f avg_score=%.4f n=%d", model_name, acc, tot_score / max(n, 1), n)
        del model
        torch.cuda.empty_cache()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "avg_score", "num_samples"])
        w.writeheader()
        w.writerows(summaries)


if __name__ == "__main__":
    main()
