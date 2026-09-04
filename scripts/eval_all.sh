#!/bin/bash
# Full paper-protocol evaluation grid. Random + spatial splits; dense is an opt-in subset
# of random (DENSE=1). Results land in output/results/*.json.
#   DATA_ROOT=data/git10m HF_CACHE_DIR=data/git10m/hf bash scripts/eval_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${DATA_ROOT:-data/git10m}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$DATA_ROOT/hf}"
SPLITS=(random spatial); [ "${DENSE:-0}" = "1" ] && SPLITS+=(dense)
COMMON=(--data-root "$DATA_ROOT" --hf-cache-dir "$HF_CACHE_DIR" --batch-size "${BATCH:-16}" ${LIMIT:+--limit $LIMIT})
# Weights resolve by release name (checkpoints/<name>, else download from MVRL/TerraDiT).
# Before publishing, point WEIGHTS_ROOT at local exports, e.g. WEIGHTS_ROOT=release/weights.
W="${WEIGHTS_ROOT:+$WEIGHTS_ROOT/}"

for split in "${SPLITS[@]}"; do
  for cond in omega box point; do
    python terradit/evaluate.py --ckpt "${W}omega_xl"   --split "$split" --condition-type "$cond" "${COMMON[@]}"
  done
  python terradit/evaluate.py --ckpt "${W}omega_base" --split "$split" --condition-type omega "${COMMON[@]}"
  python terradit/evaluate.py --ckpt "${W}alpha_xl"   --split "$split" "${COMMON[@]}"
  # sigma: the bundled splits carry omega instance geometry; sigma point prompts are
  # sampled from points.pack for tiles it covers (see docs/DATA.md).
  python terradit/evaluate.py --ckpt "${W}sigma_xl"   --split "$split" "${COMMON[@]}"
done
python - <<'PY'
import glob, json
rows=[json.load(open(f)) for f in sorted(glob.glob("output/results/*.json"))]
print(f"{'run':40s} {'n':>6s} {'FID':>8s} {'CLIP':>7s} {'LPIPS':>7s} {'SSIM':>7s}")
for r in rows:
    print(f"{r['run']:40s} {r['num_tiles']:6d} {r.get('fid',float('nan')):8.2f} {r.get('clip_score',float('nan')):7.2f} "
          f"{r.get('lpips',float('nan')):7.3f} {r.get('ssim',float('nan')):7.3f}")
PY
