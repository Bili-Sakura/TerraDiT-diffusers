#!/bin/bash
# Shared settings for scripts/train_{alpha,sigma,omega,omega_base}.sh.
# Select a preset with PRESET=1gpu (default) or PRESET=4gpu; both keep the paper's
# effective batch of 256 (1 GPU: 64 x 4 gradient-accumulation steps).
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET="${PRESET:-1gpu}"
DATA_ROOT="${DATA_ROOT:-data/git10m}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$DATA_ROOT/hf}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-${TERRADIT_DINOV3_WEIGHTS:-}}"   # required unless --no-use-repa
REPORT_TO="${REPORT_TO:-wandb}"                                    # or none

case "$PRESET" in
  1gpu) CONFIG=configs/accelerate_1gpu.yaml; BATCH=64;  ACCUM=4; WORKERS=8  ;;
  4gpu) CONFIG=configs/accelerate_4gpu.yaml; BATCH=256; ACCUM=1; WORKERS=8  ;;
  *) echo "PRESET must be 1gpu or 4gpu"; exit 1 ;;
esac

COMMON=(--data-root "$DATA_ROOT" --hf-cache-dir "$HF_CACHE_DIR"
        --batch-size "$BATCH" --gradient-accumulation-steps "$ACCUM" --num-workers "$WORKERS"
        --allow-tf32 --mixed-precision fp16 --report-to "$REPORT_TO"
        --checkpointing-steps 10000 --sampling-steps 5000)
if [ -n "$DINOV3_WEIGHTS" ]; then COMMON+=(--dinov3-weights "$DINOV3_WEIGHTS"); fi
# add --vae-on-the-fly if you did not run scripts/encode_latents.py
if [ "${VAE_ON_THE_FLY:-0}" = "1" ]; then COMMON+=(--vae-on-the-fly); fi

# EXTRA_ARGS: extra trainer flags as one string, e.g. EXTRA_ARGS="--max-samples 512 --resume"
# shellcheck disable=SC2086
launch() { accelerate launch --config_file "$CONFIG" terradit/train.py "$@" "${COMMON[@]}" ${EXTRA_ARGS:-}; }
