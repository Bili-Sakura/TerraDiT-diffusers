---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: text-to-image
tags: [satellite-imagery, remote-sensing, diffusion-transformer, geospatial]
datasets: [lcybuaa/Git-10M, MVRL/TerraDiT-data]
---

# TerraDiT-Σ (sigma_xl)

| | |
| --- | --- |
| Family | `sigma` |
| Conditioning | text + RANGE+ geolocation (1280-d) + up to 50 point prompts (pixel position + OSM tag) |
| Backbone | SiT-XL/2 with adaptive local attention around point prompts |
| Parameters | 1.10B (EMA, fp16, 2.20 GB) |
| Resolution | 256x256 latents 32x32x4 (SDXL VAE) |
| Sampler | Euler, 100 steps, no CFG |

## Training

Warm-started from `alpha_xl`; new point-prompt modules randomly initialised; 50k steps, lr 1e-5, effective batch 256, REPA, caption dropout 0.5, location dropout 0.7 (a zero location embedding is in-distribution). Data: Git-10M snapshot `29f192b8` + [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data).

## Use

```bash
python terradit/sigma_demo.py --random-points 12 --lat 40.71 --lon -74.01
```

```python
from terradit.generation import build_inference_model
model = build_inference_model("sigma", None, "sigma_xl", "cuda")
```

Files: `model.safetensors` (state dict, keys as in `terradit/models/sit.py`, REPA projectors
removed) and `config.json` (construction flags read by `build_inference_model`).

## Evaluation

Reproduce with `python terradit/evaluate.py --ckpt sigma_xl --split random` (and `spatial`);
metrics are FID, CLIP score, LPIPS, SSIM against the held-out Git-10M tiles. Numbers: see
the paper and `output/results/` after running `scripts/eval_all.sh`.

## License

CC-BY-NC-4.0 (non-commercial research use): trained on Git-10M (CC-BY-NC-ND-4.0) with
OpenStreetMap-derived conditioning (ODbL, © OpenStreetMap contributors). Code is Apache 2.0.

## Limitations

Synthetic imagery; coverage follows Git-10M and OpenStreetMap. Not for presenting outputs as
real observations. Citation and full details: [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT).
