<div align="center">

# TerraDiT

<sub>Official implementation of **TerraDiT-α (alpha)**, **TerraDiT-Σ (Sigma)**, and **TerraDiT-Ω (Omega)**, unifying two papers:</sub><br/>
<sub>• **TerraDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis** — α + Σ</sub><br/>
<sub>• **TerraDiT-Ω: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive** — Ω</sub>

<br/>

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Collection-orange)](https://huggingface.co/collections/MVRL/terradit-6a9a248fef20075fe3b1967c)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/MVRL/TerraDiT)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue)](https://huggingface.co/datasets/MVRL/TerraDiT-data)

| Model | Conditioning | Backbone | Weights | Paper |
| --- | --- | --- | --- | --- |
| **TerraDiT-α** | text | SiT-XL/2 | `alpha_xl` (1.7 GB) | TerraDiT |
| **TerraDiT-Σ** | text + geolocation + point prompts | SiT-XL/2 | `sigma_xl` (2.2 GB) | TerraDiT |
| **TerraDiT-Ω** | text + geolocation + polygons / polylines / boxes / points | SiT-XL/2 | `omega_xl` (2.4 GB) | TerraDiT-Ω |
| **TerraDiT-Ω base** | same as Ω | SiT-B/2 | `omega_base` (0.6 GB) | TerraDiT-Ω |

</div>

---

<div align="center">

## TerraDiT-Ω: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive (ECCV 2026)

<img src="images/omega_teaser.png" width="640">

[![arXiv](https://img.shields.io/badge/arXiv-2606.31029-red)](https://arxiv.org/abs/2606.31029)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://brian-j-wei.github.io/terradit-omega/index.html)

[Brian Wei\*](https://brian-j-wei.github.io/),
[Srikumar Sastry\*](https://vishu26.github.io/),
[Dan Cher\*](https://dcher95.github.io/),
[Eric Xing](https://ericx003.github.io/),
[Nathan Jacobs](https://jacobsn.github.io/)
&nbsp;(\*equal contribution)

</div>

TerraDiT-Ω generalises point-conditioned control to **any geospatial primitive** (points,
bounding boxes, polylines, polygons) through Geometry-Aware Local Attention (GALA). One model
accepts heterogeneous instance geometry plus a global caption and geolocation, giving precise
per-instance spatial control without dense maps.

<div align="center">

## TerraDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis (TerraBytes II, ECCV 2026)

<img src="images/sigma_teaser.jpg" width="600">

[![arXiv](https://img.shields.io/badge/arXiv-2603.02172-red)](https://arxiv.org/abs/2603.02172)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vishu26.github.io/terradit/index.html)

[Srikumar Sastry\*](https://vishu26.github.io/),
[Dan Cher\*](https://dcher95.github.io/),
[Brian Wei\*](https://brian-j-wei.github.io/),
[Aayush Dhakal](https://sites.wustl.edu/aayush/),
[Subash Khanal](https://subash-khanal.github.io/),
Dev Gupta,
[Nathan Jacobs](https://jacobsn.github.io/)
&nbsp;(\*equal contribution)

</div>

TerraDiT generates satellite images from **point queries** (spatial points paired with short
OSM descriptions) through adaptive local attention (Σ), on top of a text-to-image base (α).

---

## Quickstart

```bash
git clone --recursive https://github.com/mvrl/TerraDiT.git && cd TerraDiT
conda create -n terradit python=3.11 -y && conda activate terradit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA
pip install -e .
```

Inference uses one Diffusers pipeline class per family
(`TerraDiTAlphaPipeline`, `TerraDiTSigmaPipeline`, `TerraDiTOmegaPipeline`)
with `FlowMatchEulerDiscreteScheduler` (100 Euler steps, no CFG, SDXL VAE).
Converted folders live on [BiliSakura/TerraDiT](https://huggingface.co/BiliSakura/TerraDiT)
as `TerraDiT-Alpha-XL` / `TerraDiT-Sigma-XL` / `TerraDiT-Omega-XL` / `TerraDiT-Omega-B`.
Each folder is a self-contained [SiT-diffusers](https://huggingface.co/BiliSakura/SiT-diffusers)-style
repo: one `pipeline.py`, one `transformer/transformer_terradit_{alpha|sigma|omega}.py`, and on
Σ / Ω a `geolocation_encoder/` folder for RANGE+ (SatCLIP location tower + retrieval DB).
VAE / LongCLIP / tokenizer configs are included so those weights can be uploaded beside them.
Write the four Hub folders (code + configs, no weights) with
`python scripts/convert_to_diffusers.py --skeleton --out release/TerraDiT`.
Convert a training checkpoint with
`python scripts/convert_to_diffusers.py --ckpt omega_xl --out release/TerraDiT --repo-layout`.

Coordinates below are tile pixels in `[0, 256)`, x to the right and y down. Tags follow
the OSM `"<key> <value>"` convention (vocabulary in `osm/tag_vocab.pt`).
`height` / `width` default to 256; other multiples of 16 interpolate the trained
positional embeddings.

### TerraDiT-α — text

```python
import torch
from diffusers import DiffusionPipeline

# no `terradit` install required (SiT-diffusers-style folder)
pipe = DiffusionPipeline.from_pretrained(
    "BiliSakura/TerraDiT",
    subfolder="TerraDiT-Alpha-XL",
    trust_remote_code=True,
).to("cuda")
# or: from terradit import TerraDiTAlphaPipeline
# pipe = TerraDiTAlphaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-Alpha-XL")
image = pipe(
    prompt="The satellite image shows a coastal town with a marina and red-roofed houses.",
    height=256,
    width=256,
    num_inference_steps=100,
    num_images_per_prompt=1,
    guidance_scale=0.0,          # released default: no CFG
    # negative_prompt="",        # used only when guidance_scale > 1
    # prompt_embeds=y,           # or pass LongCLIP token + pooled embeds together
    # pooled_prompt_embeds=yp,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="pil",
).images[0]
image.save("terradit_alpha.png")
```

### TerraDiT-Σ — text + geolocation + point prompts

```python
import torch
from terradit import TerraDiTSigmaPipeline

pipe = TerraDiTSigmaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-Sigma-XL")
pipe = pipe.to("cuda")
image = pipe(
    prompt="The satellite image shows a suburban neighborhood with a school and a pond.",
    lat=38.65,                   # WGS84; RANGE+ encodes these when loc_embed is omitted
    lon=-90.31,
    # loc_embed=loc,             # or pass a precomputed RANGE+ vector (1280,) / (B, 1280)
    # range_model=range_model,   # optional live RANGE+ encoder
    points=[                     # [x, y, "osm tag"] in tile pixels [0, 256)
        [40, 40, "building house"],
        [90, 170, "amenity school"],
        [205, 215, "natural water"],
    ],
    # point_prompts=..., pos=..., mask=...,   # or precomputed sigma tensors (eval / dataset)
    height=256,
    width=256,
    num_inference_steps=100,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

### TerraDiT-Ω — text + geolocation + any geospatial primitive

```python
import torch
from terradit import TerraDiTOmegaPipeline

pipe = TerraDiTOmegaPipeline.from_pretrained("BiliSakura/TerraDiT", subfolder="TerraDiT-Omega-XL")
# SiT-B/2 variant: subfolder="TerraDiT-Omega-B"
pipe = pipe.to("cuda")
image = pipe(
    prompt="A small town crossed by a river and a road bridge.",
    lat=48.86,
    lon=2.35,
    # loc_embed=loc, range_model=range_model,
    instances=[
        {"type": "polygon",  "coords": [[20, 20], [110, 15], [120, 90], [30, 100]], "tag": "leisure park"},
        {"type": "polyline", "coords": [[0, 30], [60, 110], [140, 170], [255, 235]], "tag": "waterway river"},
        {"type": "bbox",     "coords": [[20, 170], [70, 215]], "tag": "building house"},
        {"type": "point",    "coords": [200, 200], "tag": "amenity parking"},
    ],
    condition_type="omega",      # "omega" = all primitives; "box" = boxes+points; "point" = points
    # dropout_probs=[...],       # explicit GALA dropout mask; overrides condition_type
    # inst_text_embed=..., polygon_xy=..., polygon_xy_mask=...,
    # polyline_xy=..., polyline_xy_mask=..., bbox_xyxy=..., point_xy=...,
    # format_mask=..., instance_mask=...,    # or precomputed omega tensors (eval / dataset)
    height=256,
    width=256,
    num_inference_steps=100,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

```bash
# α: text -> image (built-in prompts; edit PROMPTS in the file or pass --prompt)
python terradit/alpha_demo.py

# Σ: text + lat/lon + point prompts (EXAMPLE in the file; or --random-points 12; or --points JSON)
python terradit/sigma_demo.py
python terradit/sigma_demo.py --prompt "a dense residential neighborhood" \
    --points '[[120,80,"building house"],[60,200,"waterway river"]]' --lat 40.71 --lon -74.01

# Ω: text + lat/lon + any geospatial primitive (EXAMPLE has one of each; or --example-json)
python terradit/omega_demo.py
python terradit/omega_demo.py --subfolder TerraDiT-Omega-B --condition-type box
```

Each demo writes 4 samples plus an overlay of the conditioning to `samples/<family>/`.

**Geolocation.** Σ and Ω encode WGS84 `lat` / `lon` with RANGE+ (1280-d). Converted Hub
folders ship that encoder as its own component::

    geolocation_encoder/
      modeling_geolocation.py
      config.json
      satclip-vit16-l40.ckpt    # microsoft/SatCLIP-ViT16-L40 (location tower)
      range_db.npz              # mvrl/RANGE-database (`range_db_large.npz`)

`DiffusionPipeline.from_pretrained(..., trust_remote_code=True)` loads the code from that
subfolder. Drop the two weight files in and `pipe(lat=..., lon=...)` runs RANGE+ with no
extra install. Without those files (or without coordinates) the location embedding is
zeros — in-distribution for Σ, which trained with location dropout. The
[RANGE+](https://github.com/mvrl/RANGE) git submodule remains a fallback for the in-repo
demos (`git submodule update --init`).

## Downloads

| | Hub | local path | how |
| --- | --- | --- | --- |
| Weights (4 models) | [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT) | `checkpoints/<name>/` | automatic, or `python scripts/download_weights.py` |
| Derived data | [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data) | `data/git10m/` | `python scripts/download_data.py --data-root data/git10m [--family omega]` |
| Imagery | [lcybuaa/Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) @ `29f192b8` | `data/git10m/hf/` | `python -c "from terradit.hf import load_git10m; load_git10m('data/git10m/hf')"` |

All released weights are EMA, fp16 safetensors with a `config.json` beside them; every family
shares one SiT backbone (`terradit/models/sit.py`, shipped on the Hub as
`transformer/transformer_terradit_{alpha|sigma|omega}.py` like
[SiT-diffusers](https://huggingface.co/BiliSakura/SiT-diffusers))
and the Diffusers flow-matching Euler scheduler (100 steps, no classifier-free guidance).
Convert them to a one-stop Diffusers folder (`pipeline.py`,
`transformer/transformer_terradit_*.py`, `geolocation_encoder/modeling_geolocation.py` on
Σ / Ω, VAE / LongCLIP / tokenizer configs, `scheduler/scheduler_config.json`,
`model_index.json`) with `scripts/convert_to_diffusers.py`
(`--skeleton` writes code + configs only; `--repo-layout` writes `TerraDiT-Alpha-XL` /
`TerraDiT-Sigma-XL` / `TerraDiT-Omega-XL` / `TerraDiT-Omega-B`; `--include-aux` also
writes the SDXL VAE and LongCLIP weights; `--satclip-ckpt` / `--range-db` attach RANGE+
weights into `geolocation_encoder/`).
The derived data (tile ids + coordinates + `hf_idx`, OSM point rasters, instance geometry,
RANGE+ embeddings, test splits) mirrors the layout the code expects; see
[docs/DATA.md](docs/DATA.md) for the layout, provenance, and custom data.

**Imagery is not re-hosted.** Download the exact Git-10M snapshot we used: every metadata row
carries an `hf_idx` that indexes revision `29f192b8d2aa28b5d4d8c8d7f0f608cdc61fb52f` of the
`train` split (10.5M rows, 787 Arrow shards, ~394 GB). All loaders pin this revision and
verify `img_name` on read, and captions are read from it. Demos do not need it; evaluation and training do.

## Evaluation

```bash
python terradit/evaluate.py --ckpt BiliSakura/TerraDiT --family omega --split random \
    --condition-type omega --data-root data/git10m --hf-cache-dir data/git10m/hf
bash scripts/eval_all.sh          # full grid (random + spatial; DENSE=1 adds the dense subset)
```

`evaluate.py` generates the split once (`output/<run>/`), fetches ground-truth tiles by
`hf_idx`, and reports FID, CLIP score, LPIPS and SSIM to `output/results/<run>.json`. Splits:
`random` (14,932 tiles), `spatial` (14,387 geographically held-out tiles), and `dense`
(the 3,426 `random` tiles with >= 15 instances; a strict subset, so the protocol is
random + spatial).

## Training

```bash
export TERRADIT_DINOV3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   # REPA target
python scripts/encode_latents.py --data-root data/git10m --hf-cache-dir data/git10m/hf --family omega

PRESET=4gpu bash scripts/train_omega.sh        # paper setup: 4 GPUs, effective batch 256
bash scripts/train_omega.sh                    # 1 GPU: batch 64 x 4 grad-accum (same effective batch)
INIT=omega_xl STEPS=20000 bash scripts/train_omega.sh   # fine-tune the released model
```

One trainer (`terradit/train.py --family alpha|sigma|omega`) covers all families; `--init-from`
accepts a release name. REPA alignment uses the satellite DINOv3 ViT-L/16, whose weights you
obtain from Meta under their license. See [docs/TRAINING.md](docs/TRAINING.md) for the
recipes, presets, memory notes, and the `--no-use-repa` / `--vae-on-the-fly` options.

## Repository layout

```
terradit/
  alpha_demo.py  sigma_demo.py  omega_demo.py   demos (edit the EXAMPLE / PROMPTS at the top)
  evaluate.py  generate.py  eval/metrics.py    test-split generation + FID/CLIP/LPIPS/SSIM
  train.py  training/                           unified trainer (REPA loss, EMA, accelerate)
  pipelines/  models/  conditioning.py  viz.py  Diffusers pipeline, SiT + GALA, overlays
  data/dataset.py  data/preprocessing/          data_root layout, packers, hf_idx tooling
  hf.py                                         pinned Hub repos/revision, weight loading
  RANGE/                                        RANGE+ geolocation encoder (git submodule)
scripts/   download_*.py  encode_latents.py  convert_to_diffusers.py  export_weights.py  train_*.sh  eval_all.sh  upload_hf.py
configs/   accelerate presets (1 GPU / 4 GPU)
docs/      DATA.md  TRAINING.md  hf_cards/
```

## Citation

```bibtex
@inproceedings{wei2026terraditomega,
  title     = {TerraDiT-{\Omega}: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive},
  author    = {Wei, Brian and Sastry, Srikumar and Cher, Daniel and Xing, Eric and Jacobs, Nathan},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
```bibtex
@article{sastry2026terradit,
  title   = {TerraDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis},
  author  = {Sastry, Srikumar and Cher, Daniel and Wei, Brian and Dhakal, Aayush and Khanal, Subash and Gupta, Dev and Jacobs, Nathan},
  journal = {arXiv:2603.02172},
  year    = {2026}
}
```

## Acknowledgements

Built on [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA),
[RANGE](https://github.com/mvrl/RANGE), the [SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae),
[LongCLIP](https://huggingface.co/zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14), and
[DINOv3](https://github.com/facebookresearch/dinov3). Imagery from
[Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M).

## License

* **Code**: Apache 2.0 ([LICENSE](LICENSE)).
* **Weights**: CC-BY-NC-4.0, non-commercial research use, because the training imagery
  (Git-10M) is CC-BY-NC-ND-4.0.
* **Derived data** ([MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data)):
  Open Database License (ODbL) 1.0. Contains information from OpenStreetMap, made available
  under ODbL, © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
* **Git-10M** imagery and captions are not redistributed; download them from the original
  repository under its CC-BY-NC-ND-4.0 terms. Our metadata holds only tile identifiers,
  coordinates, and row indices, and the code reads captions from your copy.

See [NOTICE.md](NOTICE.md) for every third-party component and its license.

## Additional Links

Check out our lab website for other work:
* Multimodal Vision Research Lab (MVRL) — [Link](https://mvrl.cse.wustl.edu/)
* Related works from MVRL — [Link](https://mvrl.cse.wustl.edu/publications/)
