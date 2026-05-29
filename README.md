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

## Install

```bash
pip install -r requirements.txt
```

**LVLM:** install [LLaVA](https://github.com/haotian-liu/LLaVA) (`pip install -e /path/to/LLaVA`) or set `LLAVA_ROOT`.

Download pretrained weights (into `checkpoints/`; see [Checkpoints](#checkpoints)):

```bash
bash scripts/vlm/download_llava_v1_5_weights.sh
bash scripts/vlm/download_encoder_replace_weights.sh
bash scripts/download_guided_diffusion.sh
```

## Data

Evaluation inputs live under **`data/`** at the repository root.

```
data/
├── imagenet/                    # classification / purification (bundled)
│   ├── clean-correct-500.zip
│   └── *.png
├── vqa/                         # LVLM / VQA
│   ├── all_correct.ids          # bundled: clean-correct question ids
│   ├── vqav2_val.jsonl          # from download script
│   └── raw/                     # extracted VQA json (from download script)
├── coco/val2014/                # from download script
└── _downloads/                  # temp zips (from download script)
```

### ImageNet (classification & purification)

Bundled **`data/imagenet/clean-correct-500.zip`** — **500** clean RGB PNGs on which all eight paper classifiers (Table 1) are correct. Extract once from the repo root:

```bash
unzip -q -o data/imagenet/clean-correct-500.zip -d data/imagenet
mv data/imagenet/imagenet/*.png data/imagenet/
rmdir data/imagenet/imagenet
```

Filenames use sequential index + ground-truth class id: `00000_305.png`, `00001_559.png`, … `00499_<class_id>.png`.

- **Format:** RGB PNG; **224×224** recommended (matches classifier input in the attack scripts).
- **Labels:** the suffix after `_` is the ground-truth class id (0–999), used for saving/eval.
- Scripts load via `--input data/imagenet` with `--start_idx` / `--end_idx` (half-open range on the `00000`…`00499` indices).

Example (full 500-image subset):

```bash
python scripts/classification/compute_advsm.py \
  --data imagenet \
  --model "ResNet-50" \
  --model "ResNet-50 (Robust)" \
  --input data/imagenet \
  --out_dir outputs/advsm/classifiers \
  --start_idx 0 --end_idx 500
```

**Classifiers** are pulled from [RobustBench](https://github.com/RobustBench/robustbench) on first use (Hugging Face / torch hub cache), not duplicated under `checkpoints/`.

### VQAv2 + COCO val2014 (LVLM / VQA)

VQA images and annotations are **not** bundled. From the repo root, run:

```bash
bash scripts/vlm/download_and_prepare_vqav2.sh
```

This downloads VQAv2 val questions/annotations and COCO val2014 images (~6 GB for images), then writes:

| Path | Role |
|------|------|
| `data/vqa/vqav2_val.jsonl` | One JSON object per question (for eval / attacks) |
| `data/vqa/raw/` | Extracted official VQA val JSON |
| `data/coco/val2014/` | COCO images referenced by the jsonl |

**`data/vqa/all_correct.ids`** (bundled) — one **VQAv2 `question_id`** per line (**1,108** ids). Val questions where all evaluated LVLM encoders are correct on clean inputs. Pass as `--subset-file data/vqa/all_correct.ids` (filters jsonl rows by `id`; `start_idx` / `end_idx` apply after filtering).

**Options:**

```bash
bash scripts/vlm/download_and_prepare_vqav2.sh --data-root /path/to/data
bash scripts/vlm/download_and_prepare_vqav2.sh --max-samples 5000
bash scripts/vlm/download_and_prepare_vqav2.sh --skip-coco      # VQA zips + jsonl only (images already present)
bash scripts/vlm/download_and_prepare_vqav2.sh --skip-download  # convert only (zips already extracted)
```

Requires `curl` or `wget`, `unzip`, and `python3`. Sources: [COCO](https://cocodataset.org/#download), [VQAv2](https://visualqa.org/download.html).

After preparation, use `--data-jsonl`, `--image-root`, and optionally `--subset-file` (see [LVLM / VQA](#lvlm--vqa)).

## Checkpoints

All pretrained weights for this repo live under **`checkpoints/`** at the repository root.  
Scripts download here by default; Python loaders resolve paths via `advsm._paths.checkpoint_path(...)`.

### Layout

```
checkpoints/
├── llava-v1.5-7b/              # LLaVA-1.5-7B merged (CLIP backbone) — `bash scripts/vlm/download_llava_v1_5_weights.sh`
├── encoder_replace/
│   ├── fare/fare_eps_4.pt
│   ├── tecoa/tecoa_eps_4.pt
│   └── simclip/simclip4.pt     # `bash scripts/vlm/download_encoder_replace_weights.sh`
├── guided_diffusion/
│   └── imagenet/
│       └── 256x256_diffusion_uncond.pt   # DiffPure / DDIM / DC / SSNI / MimicDiffusion / ContrastDiff
└── score_sde/                  # optional, CIFAR-10 diffusion purifiers
    └── cifar10/
        └── checkpoint_35.pth
```

ImageNet **classifiers** use RobustBench caches (see [Data](#data)). Keys are in `configs/classifiers.yaml`.

### Diffusion weights

```bash
bash scripts/download_guided_diffusion.sh
```

Installs `checkpoints/guided_diffusion/imagenet/256x256_diffusion_uncond.pt` (OpenAI [guided-diffusion](https://github.com/openai/guided-diffusion) ImageNet unconditional model, used by DiffPure / DDIM and other purifiers in this repo).

Optional env: `FORCE=1` to re-download; `IMAGENET_URL=...` to override the URL (see script header).

## Classification

**AdvSM** (example: two models):

```bash
python scripts/classification/compute_advsm.py \
  --data imagenet \
  --model "ResNet-50" \
  --model "ResNet-50 (Robust)" \
  --input data/imagenet \
  --out_dir outputs/advsm/classifiers \
  --start_idx 0 --end_idx 500
```

**PGD attack** (`--model` must be one of the eight keys in `configs/classifiers.yaml`):

```bash
python scripts/classification/attack_pgd.py \
  --model "ViT-B (Robust)" \
  --eps 4 \
  --input data/imagenet \
  --output outputs/attacks/vitb_robust_pgd \
  --start_idx 0 --end_idx 500
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
  --input data/imagenet \
  --output outputs/attacks/pgdtransfer \
  --start_idx 0 --end_idx 500
```

## LVLM / VQA

Prepare data first: `bash scripts/vlm/download_and_prepare_vqav2.sh` (see [Data](#data)). Use `data/vqa/all_correct.ids` as the clean-correct question subset.

**Attack:**

```bash
python scripts/vlm/attack_pgd_transfer.py \
  --source fare \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --image-root data/coco/val2014 \
  --subset-file data/vqa/all_correct.ids \
  --output-dir outputs/attacks/vlm_fare \
  --eps 4/255 --steps 40 --batch-size 1
```

**Eval:**

```bash
python scripts/vlm/eval_clean.py \
  --models fare tecoa simclip \
  --adv \
  --models-config configs/vlm_models.yaml \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --image-root outputs/attacks/vlm_fare/adv_images \
  --subset-file data/vqa/all_correct.ids \
  --output outputs/eval_adv.csv
```

## Repository layout

```
advsm/              # Python package
configs/            # classifiers.yaml, vlm_models.yaml
data/               # imagenet/; vqa/ (+ coco/ from download script)
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
