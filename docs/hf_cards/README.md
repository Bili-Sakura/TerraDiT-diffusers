---
license: cc-by-nc-4.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - satellite-imagery
  - remote-sensing
  - diffusion-transformer
  - controllable-generation
  - geospatial
datasets:
  - lcybuaa/Git-10M
  - MVRL/TerraDiT-data
---

# TerraDiT / TerraDiT-Ω weights

Official weights for **TerraDiT** (α, Σ) and **TerraDiT-Ω** (ECCV 2026), diffusion
transformers for satellite image synthesis at 256x256 conditioned on text, geolocation,
point prompts, and arbitrary geospatial primitives.

[Code](https://github.com/mvrl/TerraDiT) ·
[Collection](https://huggingface.co/collections/MVRL/terradit-6a9a248fef20075fe3b1967c) ·
[Data](https://huggingface.co/datasets/MVRL/TerraDiT-data) ·
[TerraDiT-Ω paper](https://arxiv.org/abs/2606.31029) · [project page](https://brian-j-wei.github.io/terradit-omega/index.html) ·
[TerraDiT paper](https://arxiv.org/abs/2603.02172) · [project page](https://vishu26.github.io/terradit/index.html)

| folder | model | conditioning | backbone | params | size |
| --- | --- | --- | --- | --- | --- |
| `alpha_xl/` | TerraDiT-α | text | SiT-XL/2 | 828M | 1.66 GB |
| `sigma_xl/` | TerraDiT-Σ | text + geolocation + point prompts | SiT-XL/2 | 1.10B | 2.20 GB |
| `omega_xl/` | TerraDiT-Ω | text + geolocation + polygons/polylines/boxes/points | SiT-XL/2 | 1.18B | 2.36 GB |
| `omega_base/` | TerraDiT-Ω (base) | same as Ω | SiT-B/2 | 299M | 0.60 GB |

Each folder holds `model.safetensors` (EMA weights, fp16, REPA projector heads removed) and
`config.json` (family, architecture, conditioning flags, training steps, parameter count).

## Usage

```bash
git clone --recursive https://github.com/mvrl/TerraDiT.git && cd TerraDiT && pip install -e .
python terradit/omega_demo.py            # downloads omega_xl on first use
```

```python
from terradit import TerraDiTOmegaPipeline
pipe = TerraDiTOmegaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-Omega-XL")
pipe = pipe.to("cuda")
image = pipe("A small town crossed by a river and a road bridge.").images[0]
```

A converted Diffusers folder is self-contained (one `pipeline.py` +
`transformer/transformer_terradit_omega.py` + `geolocation_encoder/` on Σ / Ω) and loads
without installing this repo::

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "BiliSakura/TerraDiT", subfolder="TerraDiT-Omega-XL", trust_remote_code=True,
    )
Sampling: `FlowMatchEulerDiscreteScheduler`, 100 steps, no classifier-free guidance, SDXL VAE decoder. Text encoder:
LongCLIP (`zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14`, 144 tokens). Geolocation:
RANGE+ (1280-d, `geolocation_encoder/`, location encoder from
[`MVRL/satclip-loc-enc-vit16-l40`](https://huggingface.co/MVRL/satclip-loc-enc-vit16-l40)). See the [GitHub repo](https://github.com/mvrl/TerraDiT) for demos,
evaluation, and training.

## Training data

Git-10M imagery (snapshot `29f192b8`, 256x256, ~1 m/px) with derived OSM conditioning and
RANGE+ embeddings from [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data).

## License and attribution

Weights: **CC-BY-NC-4.0** (non-commercial research use), because the training imagery,
[Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M), is CC-BY-NC-ND-4.0. Code:
Apache 2.0. Conditioning data derives from OpenStreetMap: contains information from
OpenStreetMap, made available under the Open Database License (ODbL), © OpenStreetMap
contributors. Full component list: `NOTICE.md` in the code repository.

## Intended use and limitations

Research on controllable remote-sensing image synthesis, data augmentation, and map-to-image
translation. Outputs are synthetic and must not be presented as real imagery. The models
inherit Git-10M's geographic and semantic coverage; OSM tag coverage is uneven across regions.

## Citation

```bibtex
@inproceedings{wei2026terraditomega,
  title     = {TerraDiT-{\Omega}: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive},
  author    = {Wei, Brian and Sastry, Srikumar and Cher, Daniel and Xing, Eric and Jacobs, Nathan},
  booktitle = {European Conference on Computer Vision}, year = {2026}
}
@article{sastry2026terradit,
  title   = {TerraDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis},
  author  = {Sastry, Srikumar and Cher, Daniel and Wei, Brian and Dhakal, Aayush and Khanal, Subash and Gupta, Dev and Jacobs, Nathan},
  journal = {arXiv:2603.02172}, year = {2026}
}
```
