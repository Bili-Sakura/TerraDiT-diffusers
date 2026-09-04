"""Convert TerraDiT release / training checkpoints to a Diffusers pipeline folder.

Each output directory is a self-contained variant that
``DiffusionPipeline.from_pretrained`` / ``TerraDiTPipeline.from_pretrained`` can
load. The scheduler folder contains only ``scheduler_config.json``.

    # released Hub weights (auto-download) -> Diffusers layout
    python scripts/convert_to_diffusers.py --ckpt omega_xl --out release/diffusers

    # training .pt
    python scripts/convert_to_diffusers.py --ckpt exps/run/checkpoints/0400000.pt \
        --family omega --arch SiT-B/2 --out release/diffusers/omega_base

    # include SDXL VAE + LongCLIP so the folder is one-stop
    python scripts/convert_to_diffusers.py --ckpt alpha_xl --out release/diffusers --include-aux

Without ``--include-aux`` the folder holds the transformer, scheduler, model_index,
and a ``pipeline.py`` re-export. ``TerraDiTPipeline.from_checkpoint`` then loads
the VAE and LongCLIP from their Hub ids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import diffusers
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import MODELS, VAE_ID
from terradit.models.transformer import TerraDiTTransformer2DModel
from terradit.pipelines.pipeline_terradit import TerraDiTPipeline

PIPELINE_PY = '''# Copyright 2026 The TerraDiT Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Diffusers pipeline entry point for a converted TerraDiT folder.

Install the ``terradit`` package, then load with::

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_dir,
        custom_pipeline=f"{model_dir}/pipeline.py",
        trust_remote_code=True,
    )
"""

from terradit.pipelines.pipeline_terradit import TerraDiTPipeline

__all__ = ["TerraDiTPipeline"]
'''


def _write_model_index(out_dir: str, include_aux: bool) -> None:
    index = {
        "_class_name": "TerraDiTPipeline",
        "_diffusers_version": diffusers.__version__,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["terradit.models.transformer", "TerraDiTTransformer2DModel"],
    }
    if include_aux:
        index.update(
            vae=["diffusers", "AutoencoderKL"],
            text_encoder=["transformers", "CLIPTextModel"],
            tokenizer=["transformers", "CLIPTokenizer"],
        )
    with open(os.path.join(out_dir, "model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def convert_checkpoint(
    ckpt: str,
    out_dir: str,
    *,
    family: str | None = None,
    arch: str | None = None,
    legacy: bool | None = None,
    include_aux: bool = False,
    dtype: torch.dtype = torch.float16,
) -> str:
    """Write a Diffusers pipeline directory and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    transformer = TerraDiTTransformer2DModel.from_legacy_checkpoint(
        ckpt, family=family, arch=arch, legacy=legacy, torch_dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    transformer.save_pretrained(os.path.join(out_dir, "transformer"))
    scheduler.save_pretrained(os.path.join(out_dir, "scheduler"))

    extra = [n for n in os.listdir(os.path.join(out_dir, "scheduler")) if n != "scheduler_config.json"]
    for name in extra:
        os.remove(os.path.join(out_dir, "scheduler", name))

    if include_aux:
        pipe = TerraDiTPipeline.from_checkpoint(
            ckpt, family=family, arch=arch, legacy=legacy, load_aux=True,
            scheduler=scheduler,
        )
        pipe.transformer = transformer
        pipe.save_pretrained(out_dir, safe_serialization=True)
        # save_pretrained rewrites model_index using registered modules; keep it.
    else:
        _write_model_index(out_dir, include_aux=False)

    with open(os.path.join(out_dir, "pipeline.py"), "w") as f:
        f.write(PIPELINE_PY)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="release name, safetensors dir/file, training .pt, or Diffusers folder")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--family", default=None, choices=["alpha", "sigma", "omega"])
    ap.add_argument("--arch", default=None)
    ap.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--include-aux", action="store_true",
                    help=f"also serialize SDXL VAE ({VAE_ID}) and LongCLIP")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    if args.ckpt in MODELS and args.family is None:
        args.family = MODELS[args.ckpt]["family"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = convert_checkpoint(
        args.ckpt, args.out, family=args.family, arch=args.arch,
        legacy=args.legacy, include_aux=args.include_aux, dtype=dtype,
    )
    print(f"[convert] {args.ckpt} -> {out}")
    print(f"[convert] load with: TerraDiTPipeline.from_pretrained({out!r})")


if __name__ == "__main__":
    main()
