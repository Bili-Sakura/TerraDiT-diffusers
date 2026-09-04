#!/bin/bash
# TerraDiT-alpha (text -> image, SiT-XL/2).
#   from the released weights (recommended):  bash scripts/train_alpha.sh
#   from scratch:                             INIT=none STEPS=1500000 bash scripts/train_alpha.sh
source "$(dirname "$0")/train_common.sh"
INIT="${INIT:-alpha_xl}"; STEPS="${STEPS:-100000}"; LR="${LR:-1e-5}"
INIT_ARGS=(); [ "$INIT" != "none" ] && INIT_ARGS=(--init-from "$INIT")
launch --family alpha --arch SiT-XL/2 "${INIT_ARGS[@]}" \
  --max-train-steps "$STEPS" --learning-rate "$LR" --exp-name "${EXP_NAME:-alpha-finetune}"
