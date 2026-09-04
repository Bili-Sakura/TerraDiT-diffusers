#!/bin/bash
# TerraDiT-omega (text + geolocation + geospatial primitives via GALA, SiT-XL/2).
#   paper recipe: warm-start from alpha (INIT=alpha_xl), ~210k steps at lr 2e-5.
#   continue from the released omega instead: INIT=omega_xl
source "$(dirname "$0")/train_common.sh"
INIT="${INIT:-alpha_xl}"; STEPS="${STEPS:-210000}"; LR="${LR:-2e-5}"
launch --family omega --arch SiT-XL/2 --omega-attn GALA --init-from "$INIT" \
  --max-train-steps "$STEPS" --learning-rate "$LR" --exp-name "${EXP_NAME:-omega-finetune}"
