#!/bin/bash
# TerraDiT-sigma (text + geolocation + point prompts, SiT-XL/2), warm-started from alpha.
#   paper recipe: INIT=alpha_xl (new point modules start random), ~50k steps at lr 1e-5.
source "$(dirname "$0")/train_common.sh"
INIT="${INIT:-alpha_xl}"; STEPS="${STEPS:-50000}"; LR="${LR:-1e-5}"
launch --family sigma --arch SiT-XL/2 --init-from "$INIT" \
  --max-train-steps "$STEPS" --learning-rate "$LR" --exp-name "${EXP_NAME:-sigma-finetune}"
