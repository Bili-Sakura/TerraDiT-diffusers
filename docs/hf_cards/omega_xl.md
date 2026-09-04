---
license: cc-by-nc-4.0
library_name: diffusers
pipeline_tag: text-to-image
tags: [satellite-imagery, remote-sensing, diffusion-transformer, geospatial]
datasets: [lcybuaa/Git-10M, MVRL/TerraDiT-data]
---

# TerraDiT-Ω (omega_xl)

| | |
| --- | --- |
| Family | `omega` |
| Conditioning | text + RANGE+ geolocation + up to 64 instances, each a polygon, polyline, bounding box, or point with an OSM tag (GALA: Geometry-Aware Local Attention) |
| Backbone | SiT-XL/2 + GALA |
| Parameters | 1.18B (EMA, fp16, 2.36 GB) |
| Resolution | 256x256 latents 32x32x4 (SDXL VAE) |
| Sampler | Euler, 100 steps, no CFG |

## Training

Warm-started from the TerraDiT-α text model; ~210k steps, lr 2e-5, effective batch 256, REPA. Modality dropout during training enables `--condition-type omega|box|point` at inference. Data: Git-10M snapshot `29f192b8` + [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data).

## Use

```bash
python terradit/omega_demo.py --condition-type omega
```

```python
from terradit import TerraDiTOmegaPipeline
pipe = TerraDiTOmegaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-omega")
image = pipe("A small town crossed by a river.", condition_type="omega").images[0]
```

Files: `model.safetensors` (state dict, keys as in `terradit/models/sit.py`, REPA projectors
removed) and `config.json`, or a converted Diffusers folder from
`scripts/convert_to_diffusers.py`.

## Evaluation

Reproduce with `python terradit/evaluate.py --ckpt omega_xl --split random` (and `spatial`);
metrics are FID, CLIP score, LPIPS, SSIM against the held-out Git-10M tiles. Numbers: see
the paper and `output/results/` after running `scripts/eval_all.sh`.

## License

CC-BY-NC-4.0 (non-commercial research use): trained on Git-10M (CC-BY-NC-ND-4.0) with
OpenStreetMap-derived conditioning (ODbL, © OpenStreetMap contributors). Code is Apache 2.0.

## Limitations

Synthetic imagery; coverage follows Git-10M and OpenStreetMap. Not for presenting outputs as
real observations. Citation and full details: [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT).
