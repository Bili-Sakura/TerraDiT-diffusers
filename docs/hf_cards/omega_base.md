---
license: cc-by-nc-4.0
library_name: diffusers
pipeline_tag: text-to-image
tags: [satellite-imagery, remote-sensing, diffusion-transformer, geospatial]
datasets: [lcybuaa/Git-10M, MVRL/TerraDiT-data]
---

# TerraDiT-Ω base (omega_base)

| | |
| --- | --- |
| Family | `omega` |
| Conditioning | same as omega_xl |
| Backbone | SiT-B/2 + GALA |
| Parameters | 299M (EMA, fp16, 0.60 GB) |
| Resolution | 256x256 latents 32x32x4 (SDXL VAE) |
| Sampler | Euler, 100 steps, no CFG |

## Training

Trained from scratch for 400k steps, lr 2e-5, effective batch 256, REPA. This is the backbone used for the conditioning-mechanism ablations in the TerraDiT-Ω paper; use it when memory or latency matter. Data: Git-10M snapshot `29f192b8` + [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data).

## Use

```bash
python terradit/omega_demo.py --ckpt omega_base
```

```python
from terradit import TerraDiTOmegaPipeline
pipe = TerraDiTOmegaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-Omega-B")
image = pipe("A small town crossed by a river.").images[0]
```

Files: `model.safetensors` (state dict, keys as in `terradit/models/sit.py`, REPA projectors
removed) and `config.json`, or a converted Diffusers folder from
`scripts/convert_to_diffusers.py`.

## Evaluation

Reproduce with `python terradit/evaluate.py --ckpt omega_base --split random` (and `spatial`);
metrics are FID, CLIP score, LPIPS, SSIM against the held-out Git-10M tiles. Numbers: see
the paper and `output/results/` after running `scripts/eval_all.sh`.

## License

CC-BY-NC-4.0 (non-commercial research use): trained on Git-10M (CC-BY-NC-ND-4.0) with
OpenStreetMap-derived conditioning (ODbL, © OpenStreetMap contributors). Code is Apache 2.0.

## Limitations

Synthetic imagery; coverage follows Git-10M and OpenStreetMap. Not for presenting outputs as
real observations. Citation and full details: [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT).
