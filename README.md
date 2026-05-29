# AdvSM

Code for **Adversarial Robustness Optimization Increases Attack Transferability and Weakens Defense Isolation** (IEEE S&P).

- **AdvSM** (Adversarial Sensitivity Maps): quantify alignment among robustness-optimized defenses.
- **PGDTransfer**: PGD + EOT + DDIM-surrogate adaptive attack for transfer across defenses in the same family.

## Supported systems

| Setting | Models | Generate AdvSM | Attack |
|--------|--------|----------------|--------|
| **Classification** | 4 standard + 4 robust ImageNet classifiers (Table 1) | `scripts/classification/compute_advsm.py` | `scripts/classification/attack_pgd.py` (**PGD only**) |
| **Purification** | Mean, Gaussian, JPEG, DiffPure, DDIM, MimicDiffusion, ContrastDiff, DCDefense, SSNI | `scripts/purification/compute_advsm.py` | `scripts/purification/attack_pgd_transfer.py` (**7 attacks**) |
| **LVLM / VQA** | CLIP, FARE, TeCoA, SimCLIP | `scripts/vlm/compute_advsm.py` | `scripts/vlm/attack_pgd_transfer.py` (**PGDTransfer only**) |

Classifier names and RobustBench ids: `configs/classifiers.yaml`.  
VLM encoders: `configs/vlm_models.yaml` (`clip`, `fare`, `tecoa`, `simclip`).

All downloadable weights go under **`checkpoints/`** — see [checkpoints/README.md](checkpoints/README.md).

## Install

```bash
pip install -r requirements.txt
```

**LVLM:** install [LLaVA](https://github.com/haotian-liu/LLaVA) (`pip install -e /path/to/LLaVA`) or set `LLAVA_ROOT`.

```bash
bash scripts/vlm/download_llava_v1_5_weights.sh
bash scripts/vlm/download_encoder_replace_weights.sh
bash scripts/download_guided_diffusion.sh
```

## Classification

**AdvSM** (example: two models):

```bash
python scripts/classification/compute_advsm.py \
  --data imagenet \
  --model "ResNet-50" \
  --model "ResNet-50 (Robust)" \
  --input /path/to/images \
  --out_dir outputs/advsm/classifiers \
  --start_idx 0 --end_idx 100
```

**PGD attack** (`--model` must be one of the eight keys in `configs/classifiers.yaml`):

```bash
python scripts/classification/attack_pgd.py \
  --model "ViT-B (Robust)" \
  --eps 4 \
  --input /path/to/images \
  --output outputs/attacks/vitb_robust_pgd
```

**Eval:**

```bash
python scripts/classification/eval_accuracy.py \
  --target "ViT-B (Robust)" \
  --input outputs/attacks/vitb_robust_pgd
```

## Purification

Seven attacks: `PGD`, `BPDA_EOT`, `DiffPGD`, `DiffAttack`, `DiffHammer`, `DiffBreak`, `PGDTransfer`.

```bash
python scripts/purification/attack_pgd_transfer.py \
  --data imagenet \
  --target Hendrycks2020AugMix_ResNeXt \
  --defense DDIM \
  --d_settings data imagenet timesteps 150 denoise_steps 10 \
  --attack PGDTransfer \
  --eps 4 --max_iter 40 --eot_iter 5 \
  --input /path/to/images \
  --output outputs/attacks/pgdtransfer
```

## LVLM / VQA

```bash
python scripts/vlm/attack_pgd_transfer.py \
  --source fare \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqav2_val.jsonl \
  --image-root data/coco/val2014 \
  --output-dir outputs/attacks/vlm_fare \
  --eps 4/255 --steps 40 --batch-size 1
```

```bash
python scripts/vlm/eval_clean.py \
  --models fare tecoa simclip \
  --adv \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqav2_val.jsonl \
  --image-root outputs/attacks/vlm_fare/adv_images \
  --output outputs/eval_adv.csv
```

## Repository layout

```
advsm/              # Python package
configs/            # classifiers.yaml, vlm_models.yaml
scripts/            # CLIs
third_party/        # DC, SSNI, MimicDiffusion, ContrastDiffPurification, DiffAttack, DiffHammer, DiffBreak
checkpoints/        # all local weights (gitignored blobs)
```

## Citation

```bibtex
@inproceedings{advsm2026,
  title={Adversarial Robustness Optimization Increases Attack Transferability and Weakens Defense Isolation},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2026}
}
```
