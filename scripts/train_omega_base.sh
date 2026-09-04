#!/bin/bash
# TerraDiT-omega base model (SiT-B/2 + GALA), trained from scratch for 400k steps at lr 2e-5
# (the ablation backbone in the paper). Fits a single 24 GB GPU with PRESET=1gpu.
source "$(dirname "$0")/train_common.sh"
STEPS="${STEPS:-400000}"; LR="${LR:-2e-5}"
launch --family omega --arch SiT-B/2 --omega-attn GALA \
  --max-train-steps "$STEPS" --learning-rate "$LR" --exp-name "${EXP_NAME:-omega-base}"
