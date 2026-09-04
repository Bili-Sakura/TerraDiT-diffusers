---
license: cc-by-nc-4.0
library_name: diffusers
pipeline_tag: text-to-image
tags: [satellite-imagery, remote-sensing, diffusion-transformer, geospatial]
datasets: [lcybuaa/Git-10M, MVRL/TerraDiT-data]
---

# TerraDiT-α (alpha_xl)

| | |
| --- | --- |
| Family | `alpha` |
| Conditioning | text (LongCLIP, 144 tokens) |
| Backbone | SiT-XL/2 |
| Parameters | 828M (EMA, fp16, 1.66 GB) |
| Resolution | 256x256 latents 32x32x4 (SDXL VAE) |
| Sampler | Euler, 100 steps, no CFG |

## Training

Class-conditional SiT-XL/2 pre-training on Git-10M (~800k steps), text-conditional training (~560k steps), then 100k steps with the adaLN pooled-text fix (`legacy=False`), lr 1e-5, effective batch 256, REPA (DINOv3 SAT-493M, depth 8, coeff 0.5). Data: Git-10M snapshot `29f192b8` + [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data).

## Use

```bash
python terradit/alpha_demo.py --prompt "a coastal town with a marina and red-roofed houses"
```

```python
from terradit import TerraDiTPipeline
pipe = TerraDiTPipeline.from_checkpoint("alpha_xl")
image = pipe("a coastal town with a marina and red-roofed houses").images[0]
```

Files: `model.safetensors` (state dict, keys as in `terradit/models/sit.py`, REPA projectors
removed) and `config.json`, or a converted Diffusers folder from
`scripts/convert_to_diffusers.py`.

## Evaluation

Reproduce with `python terradit/evaluate.py --ckpt alpha_xl --split random` (and `spatial`);
metrics are FID, CLIP score, LPIPS, SSIM against the held-out Git-10M tiles. Numbers: see
the paper and `output/results/` after running `scripts/eval_all.sh`.

## License

CC-BY-NC-4.0 (non-commercial research use): trained on Git-10M (CC-BY-NC-ND-4.0) with
OpenStreetMap-derived conditioning (ODbL, © OpenStreetMap contributors). Code is Apache 2.0.

## Limitations

Synthetic imagery; coverage follows Git-10M and OpenStreetMap. Not for presenting outputs as
real observations. Citation and full details: [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT).
