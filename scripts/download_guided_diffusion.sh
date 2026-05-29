#!/usr/bin/env bash
# Download OpenAI guided-diffusion ImageNet weights used by DiffPure, DDIM, DC, SSNI, MimicDiffusion, ContrastDiff.
# → checkpoints/guided_diffusion/imagenet/256x256_diffusion_uncond.pt
#
# Source: https://github.com/openai/guided-diffusion
# URL:    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
#
# Usage (from repository root):
#   bash scripts/download_guided_diffusion.sh
#
# Optional env:
#   DRY_RUN=1              print commands only
#   FORCE=1                re-download even if the file exists
#   IMAGENET_URL=...       override download URL

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT}/checkpoints/guided_diffusion/imagenet"
DEST_FILE="${DEST_DIR}/256x256_diffusion_uncond.pt"
IMAGENET_URL="${IMAGENET_URL:-https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt}"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "Already present (skip): ${DEST_FILE}"
  echo "Set FORCE=1 to re-download."
  exit 0
fi

download() {
  local url="$1"
  local out="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would download:"
    echo "  ${url}"
    echo "  -> ${out}"
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --continue-at - -o "${out}.part" "${url}"
    mv "${out}.part" "${out}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${out}.part" "${url}"
    mv "${out}.part" "${out}"
  else
    echo "Need curl or wget to download weights."
    exit 1
  fi
}

echo "Downloading ImageNet guided diffusion (~2.1 GB)..."
echo "  ${IMAGENET_URL}"
echo "  -> ${DEST_FILE}"
download "${IMAGENET_URL}" "${DEST_FILE}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ ! -s "${DEST_FILE}" ]]; then
    echo "Download failed or empty file: ${DEST_FILE}" >&2
    exit 1
  fi
  echo "Done. Loaders use: checkpoints/guided_diffusion/imagenet/256x256_diffusion_uncond.pt"
fi
