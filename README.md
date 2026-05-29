# AdvSM

Code for **Adversarial Robustness Optimization Increases Attack Transferability and Weakens Defense Isolation** (IEEE S&P 2026).

- **AdvSM** (Adversarial Sensitivity Maps): quantify alignment among robustness-optimized defenses.
- **PGDTransfer**: PGD + EOT + DDIM-surrogate adaptive attack for transfer across defenses in the same family.

**Paper settings** below follow §6.1 and Tables 5–6 of the submission PDF (*Rethinking Transferability* draft). Primary attack protocol: **untargeted**, **ℓ∞ ε = 4/255**. **500** clean-correct samples per attack/eval run; **100** for AdvSM maps.

## Supported systems

| Setting | Models (paper) | AdvSM | Attack | Eval |
|--------|----------------|-------|--------|------|
| **Classification** | 8 ImageNet classifiers (Table 1) | `scripts/classification/compute_advsm.py` | `scripts/classification/attack_pgd.py` | `scripts/classification/eval_accuracy.py` |
| **Purification** | 9 purifiers + ResNet-50 (Tables 2, 5, 11) | `scripts/purification/compute_advsm.py` | `scripts/purification/attack_pgd_transfer.py` | `scripts/purification/eval_accuracy.py` |
| **LVLM / VQA** | CLIP + FARE / TeCoA / SimCLIP (Table 1, §6.5) | `scripts/vlm/compute_advsm.py` | `scripts/vlm/attack_pgd_transfer.py` | `scripts/vlm/eval_vqa.py` |

Configs: `configs/classifiers.yaml` (Table 1), `configs/vlm_models.yaml` (`clip`, `fare`, `tecoa`, `simclip`).

### Table 6 — attack hyperparameters (ε = 4/255)

| Setting | ε | step α | iterations T | EOT K |
|--------|---|--------|--------------|-------|
| Classifier (`attack_pgd.py`, untargeted) | 4/255 | ε/4 → **1/255** | **10** | — |
| Classifier (TransferAttack baselines, Table 3) | 4/255 | ε/4 → **1/255** | per [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) | — |
| Purifier (**PGDTransfer**, Table 11) | 4/255 | **1/255** | **40** | **5** |
| VQA (**PGDTransfer**, §6.5) | 4/255 | **1/255** | **100** | — |

## Install

```bash
pip install -r requirements.txt
```

**LVLM — LLaVA Python package** (weights are separate):

```bash
git clone https://github.com/haotian-liu/LLaVA.git
pip install -e /path/to/LLaVA
# or: export LLAVA_ROOT=/path/to/LLaVA
python -c "from llava.model.builder import load_pretrained_model"
```

**Checkpoints** (`checkpoints/`, gitignored):

```bash
bash scripts/download_guided_diffusion.sh
bash scripts/vlm/download_llava_v1_5_weights.sh
bash scripts/vlm/download_encoder_replace_weights.sh
```

```
checkpoints/
├── llava-v1.5-7b/
├── encoder_replace/{fare,tecoa,simclip}/
└── guided_diffusion/imagenet/256x256_diffusion_uncond.pt
```

Table 1 classifiers load via [RobustBench](https://github.com/RobustBench/robustbench) on first use.

## Classification

**Data:** NIPS 2017 adversarial-defense ImageNet subset — `data/imagenet/clean-correct-500.zip` (500 images, all Table 1 models correct on clean). Unzip to `data/imagenet/NNNNN_<class_id>.png` (224×224 RGB).

```bash
unzip -q -o data/imagenet/clean-correct-500.zip -d data/imagenet
mv data/imagenet/imagenet/*.png data/imagenet/ && rmdir data/imagenet/imagenet
```

**Models (Table 1):** ResNet-50, ConvNeXt-B, ViT-B, Swin-B + four RobustBench robust counterparts (`configs/classifiers.yaml`).

**AdvSM (§6.1):** gradient threshold **10⁻⁵**, **100** samples:

```bash
python scripts/classification/compute_advsm.py \
  --data imagenet \
  --model "ResNet-50" --model "ResNet-50 (Robust)" \
  --input data/imagenet --out_dir outputs/advsm/classifiers \
  --start_idx 0 --end_idx 100
```

**Attack / eval (500 samples):** untargeted **ℓ∞ ε = 4/255**, **10** PGD steps (`attack_pgd.py` defaults). Table 3 transferable attacks: [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).

```bash
python scripts/classification/attack_pgd.py \
  --model "ViT-B (Robust)" \
  --eps 4 --max_iter 10 \
  --input data/imagenet --output outputs/attacks/vitb_robust_pgd \
  --start_idx 0 --end_idx 500

python scripts/classification/eval_accuracy.py \
  --target "ViT-B (Robust)" \
  --input outputs/attacks/vitb_robust_pgd \
  --start_idx 0 --end_idx 500
```

## Purification

**Data:** same 500-image `data/imagenet/` subset.

**Pipeline (§6.1):** fixed downstream classifier **non-robust ResNet-50**; vary purifier. **PGDTransfer** uses **DDIM** surrogate (Table 5: `timesteps=150`, `denoise_steps=3`). Table 11: ε = 4/255, T = 40, K = 5.

**Table 5 — purifier settings (target purifiers):**

| Purifier | Key settings |
|----------|----------------|
| Mean | `kernel=5` |
| Gaussian | `noise std=0.015`, `kernel=5`, `sigma=1.5` |
| JPEG | `quality=20%` |
| DiffPure | `timesteps=150` |
| MimicDiffusion | `timesteps=1000`, `denoise_steps=100` |
| ContrastDiff | `timesteps=150`, `sample_steps=1` |
| SSNI | `timesteps=150`, `denoise_steps=150` |
| DCDefense | `timesteps=150`, `forward_noise_steps=1`, `strength_l=0.2`, `strength_s=0.1` |
| DDIM (surrogate) | `timesteps=150`, `denoise_steps=3` |

**AdvSM (§6.1):** random-sign probes, **ε = 16/255**, response threshold **θ = 2/255**, **M = 5**, **N = 10** (defaults in `compute_advsm.py`), **100** samples.

**Attack** (prints clean / robust accuracy):

```bash
python scripts/purification/attack_pgd_transfer.py \
  --data imagenet \
  --target "ResNet-50" \
  --defense DDIM \
  --d_settings data imagenet timesteps 150 denoise_steps 3 \
  --attack PGDTransfer \
  --eps 4 --max_iter 40 --eot_iter 5 \
  --input data/imagenet --output outputs/attacks/pgdtransfer_ddim \
  --start_idx 0 --end_idx 500
```

**Eval:**

```bash
python scripts/purification/eval_accuracy.py \
  --data imagenet --target "ResNet-50" \
  --defense DDIM \
  --d_settings data imagenet timesteps 150 denoise_steps 3 \
  --input data/imagenet --start_idx 0 --end_idx 500

python scripts/purification/eval_accuracy.py \
  --data imagenet --target "ResNet-50" \
  --defense DDIM \
  --d_settings data imagenet timesteps 150 denoise_steps 3 \
  --input outputs/attacks/pgdtransfer_ddim --start_idx 0 --end_idx 500
```

## LVLM / VQA

**Data:** VQAv2 val + COCO val2014:

```bash
bash scripts/vlm/download_and_prepare_vqav2.sh
```

→ `data/vqa/vqav2_val.jsonl`, `data/coco/val2014/`. Bundled **`data/vqa/all_correct.ids`** lists clean-correct **question_id**s (use with `--subset-file`; take **500** rows via `--end_idx 500` per §6.1).

**Models (§6.5):** LLaVA-1.5-7B + shared projector; encoders **CLIP ViT-L/14** (non-robust) and **FARE / TeCoA / SimCLIP** (robust, Table 1). **PGDTransfer:** ε = 4/255, α = 1/255, **T = 100**, untargeted; surrogate **fare** for transfer experiments (Table 13).

**AdvSM:** threshold **10⁻⁵**, **100** samples (`--end_idx 100`).

**Attack (500 questions):**

```bash
python scripts/vlm/attack_pgd_transfer.py \
  --source fare \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --image-root data/coco/val2014 \
  --subset-file data/vqa/all_correct.ids \
  --output-dir outputs/attacks/vlm_fare \
  --eps 4/255 --alpha 1/255 --steps 100 --batch-size 1 \
  --start-idx 0 --end-idx 500
```

**Eval** (`eval_vqa.py` — clean vs adversarial via `--adv`):

```bash
python scripts/vlm/eval_vqa.py \
  --models clip fare tecoa simclip \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --image-root data/coco/val2014 \
  --subset-file data/vqa/all_correct.ids \
  --start-idx 0 --end-idx 500 \
  --output outputs/vqa_results.csv

python scripts/vlm/eval_vqa.py \
  --models clip fare tecoa simclip --adv \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --image-root outputs/attacks/vlm_fare/adv_images \
  --subset-file data/vqa/all_correct.ids \
  --start-idx 0 --end-idx 500 \
  --output outputs/vqa_results_adv.csv
```

## Repository layout

```
advsm/              # Python package
configs/            # classifiers.yaml, vlm_models.yaml
data/               # imagenet/, vqa/, coco/
scripts/            # CLIs per setting
third_party/        # purifier / attack baselines
checkpoints/        # downloaded weights
```
