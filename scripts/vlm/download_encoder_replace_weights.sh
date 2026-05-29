#!/usr/bin/env bash
# Download robust vision encoder .pt files for fare / tecoa / simclip
# into ./checkpoints/encoder_replace/.
#
# Encoder sources:
#   FARE 4 ViT-L/14:      https://github.com/chs20/RobustVLM
#   TeCoA 4 ViT-L/14:     RobustVLM
#   SimCLIP 4 ViT-L/14: https://github.com/speedlab-git/SimCLIP  (HF weights)
#
# Projector: configs/vlm_models.yaml points encoder-replace entries at
#   checkpoints/llava-v1.5-7b/mm_projector.bin
# (stock LLaVA 1.5 merge — same as [Robust-LLaVA](https://github.com/HashmatShadab/Robust-LLaVA) plug-and-play for FARE/SimCLIP ViT-L/14).
# It comes with `bash scripts/download_llava_v1_5_weights.sh` (full snapshot). If that file is missing alone, use:
#   FETCH_MM_PROJECTOR=1 bash scripts/download_encoder_replace_weights.sh
#
# Prereq: curl (or wget). Large files (~0.3–1.2 GB each).
#
# Usage:
#   bash scripts/download_encoder_replace_weights.sh
#
# Optional:
#   DRY_RUN=1 bash scripts/download_encoder_replace_weights.sh
#   FETCH_MM_PROJECTOR=1 bash scripts/download_encoder_replace_weights.sh   # only if mm_projector.bin missing under llava-v1.5-7b/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/checkpoints/encoder_replace"
LLAVA_DIR="${ROOT}/checkpoints/llava-v1.5-7b"
HF_MM_URL="https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/mm_projector.bin"

download() {
  local url="$1" out="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would fetch: $url -> $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --continue-at - -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" -c "$url"
  else
    echo "Need curl or wget to download."
    exit 1
  fi
}

mkdir -p "${DEST}"

FARE_L14_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/jnQ2qmp9tst8kyQ/download/fare_eps_4.pt"
TECOA_L14_EPS4_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/92req4Pak5i56tX/download/tecoa_eps_4.pt"
SIMCLIP4_URL="https://huggingface.co/hossainzarif19/SimCLIP/resolve/main/simclip4.pt"

echo "Saving encoders under: ${DEST}/"
download "${FARE_L14_URL}" "${DEST}/fare/fare_eps_4.pt"
download "${TECOA_L14_EPS4_URL}" "${DEST}/tecoa/tecoa_eps_4.pt"
download "${SIMCLIP4_URL}" "${DEST}/simclip/simclip4.pt"

if [[ "${FETCH_MM_PROJECTOR:-0}" == "1" ]]; then
  MM="${LLAVA_DIR}/mm_projector.bin"
  if [[ -f "$MM" ]]; then
    echo "Already present: $MM"
  else
    mkdir -p "${LLAVA_DIR}"
    echo "Downloading mm_projector.bin -> ${MM}"
    download "${HF_MM_URL}" "${MM}"
  fi
fi

echo ""
echo "Done."
echo "  Encoders: ${DEST}/{fare,tecoa,simclip}/*.pt"
echo "  Projector (shared): ${LLAVA_DIR}/mm_projector.bin  (run download_llava_v1_5_weights.sh, or FETCH_MM_PROJECTOR=1 with this script)"
