# Training

One trainer (`terradit/train.py`) serves all three families via `--family {alpha,sigma,omega}`.
The preset scripts in `scripts/` wrap it with the paper hyper-parameters.

## Prerequisites

1. **Data** (see [DATA.md](DATA.md)): Git-10M cache + `scripts/download_data.py --family <f>`.
2. **Latents**: `python scripts/encode_latents.py --data-root data/git10m --hf-cache-dir data/git10m/hf --family <f>`
   (multi-GPU: prefix with `accelerate launch --num_processes N`). Skip and pass
   `VAE_ON_THE_FLY=1` to encode inside the training loop instead (roughly one extra VAE
   forward per step; fine for fine-tuning, slow for from-scratch runs).
3. **DINOv3 SAT-493M weights** for REPA alignment. Request from Meta
   (<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>), then
   `export TERRADIT_DINOV3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
   (or pass `--dinov3-weights`). The architecture is pulled from `torch.hub`
   (`facebookresearch/dinov3`), so the first run needs network access or a warm hub cache.
   Without the weights, train with `--no-use-repa` (this changes the recipe; the released
   models all used REPA with `--proj-coeff 0.5`, `--encoder-depth 8`).
4. `pip install -e ".[train]"` for wandb logging (`REPORT_TO=none` to disable).

## Presets

Both presets keep the paper's **effective batch of 256**:

| `PRESET` | GPUs | per-step batch | grad accumulation | config |
| --- | --- | --- | --- | --- |
| `1gpu` (default) | 1 x 24-80 GB | 64 | 4 | `configs/accelerate_1gpu.yaml` |
| `4gpu` | 4 | 256 (64 each) | 1 | `configs/accelerate_4gpu.yaml` |

```bash
PRESET=4gpu bash scripts/train_omega.sh            # paper setup
bash scripts/train_omega_base.sh                   # single GPU, SiT-B/2 from scratch
INIT=omega_xl STEPS=20000 EXP_NAME=my-ft bash scripts/train_omega.sh   # short fine-tune
```

Environment overrides: `DATA_ROOT`, `HF_CACHE_DIR`, `DINOV3_WEIGHTS`, `REPORT_TO`,
`VAE_ON_THE_FLY`, `INIT`, `STEPS`, `LR`, `EXP_NAME`, and `EXTRA_ARGS` (extra trainer flags
as one string, appended last so they override the preset). Checkpoints go to `exps/<EXP_NAME>/checkpoints/<step>.pt`.

## Recipes (what produced the released weights)

| model | init | steps | lr | notes |
| --- | --- | --- | --- | --- |
| `alpha_xl` | class-conditional SiT-XL/2 on Git-10M (~800k) -> text (~560k) | +100k | 1e-5 | final stage fixes the adaLN pooled-text bug (`legacy=False`) |
| `sigma_xl` | `alpha_xl` | 50k | 1e-5 | new point-prompt modules start random |
| `omega_xl` | `alpha_xl` | ~210k | 2e-5 | GALA attention, `--omega-attn GALA` |
| `omega_base` | scratch | 400k | 2e-5 | SiT-B/2 + GALA (ablation backbone) |

`--init-from` accepts a release name (`alpha_xl`, auto-downloaded), a directory with
`model.safetensors`, or a training `.pt`. It loads weights with `strict=False` (new modules
stay random), starts a fresh optimizer, and resets the step counter to 0.

## Useful flags

* `--resume` continues the latest checkpoint of `--exp-name` (optimizer state included when
  present); `--resume-step N` picks a specific one.
* `--max-samples N` trains on the first N metadata rows (subset fine-tunes, smoke tests).
* `--checkpointing-steps`, `--sampling-steps` (wandb sample grid), `--log-every`.
* `--legacy` / `--no-legacy` override the adaLN conditioning flag (only for loading the
  pre-fix weights; all released models are `legacy=False`).
* `--freeze-except <substr> ...` trains only parameters whose name contains a substring
  (e.g. `--freeze-except point_ instance_encoder` for adapter-style fine-tunes).

## Smoke test

```bash
REPORT_TO=none STEPS=20 EXP_NAME=smoke EXTRA_ARGS="--max-samples 512 --checkpointing-steps 10" \
  bash scripts/train_alpha.sh
```

## Troubleshooting

* `JITCallable._set_src() takes 1 positional argument` (or any Triton error) while importing
  diffusers: a stale `xformers` wheel built for another torch version is being imported
  opportunistically. TerraDiT does not use xformers (attention is torch SDPA);
  `pip uninstall xformers` fixes it.

## Memory

SiT-XL/2 at 256x256 with per-step batch 64, fp16 autocast, and the frozen DINOv3-L
encoder fits in ~40 GB. Drop to `--batch-size 32 --gradient-accumulation-steps 8` for
24 GB cards. The SiT-B/2 base model fits comfortably at batch 64 on 24 GB.
