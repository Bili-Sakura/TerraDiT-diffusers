"""Convert TerraDiT release / training checkpoints to a Diffusers pipeline folder.

Each output directory is a self-contained family variant that
``DiffusionPipeline.from_pretrained(..., trust_remote_code=True)`` can load
without installing this repository. Layout matches ``BiliSakura/SiT-diffusers``::

    TerraDiT-Alpha-XL/
      model_index.json          # _class_name=["pipeline", "TerraDiTAlphaPipeline"]
      pipeline.py               # family pipeline (relative imports only)
      pipeline_common.py
      pipeline_conditioning.py
      constants.py
      scheduler/scheduler_config.json
      transformer/
        config.json
        diffusion_pytorch_model.safetensors
        transformer_sit.py      # TerraDiTTransformer2DModel
        sit.py localattn.py omega.py

Hub layout::

    BiliSakura/TerraDiT/TerraDiT-Alpha-XL
    BiliSakura/TerraDiT/TerraDiT-Sigma-XL
    BiliSakura/TerraDiT/TerraDiT-Omega-XL
    BiliSakura/TerraDiT/TerraDiT-Omega-B

    # released Hub weights (auto-download) -> family folder
    python scripts/convert_to_diffusers.py --ckpt omega_xl --out release/TerraDiT-Omega-XL

    # write the four Hub subfolders under a repo root
    python scripts/convert_to_diffusers.py --ckpt alpha_xl --out release/TerraDiT --repo-layout

    # training .pt
    python scripts/convert_to_diffusers.py --ckpt exps/run/checkpoints/0400000.pt \
        --family omega --arch SiT-B/2 --out release/TerraDiT-Omega-B

    # include SDXL VAE + LongCLIP so the folder is one-stop
    python scripts/convert_to_diffusers.py --ckpt alpha_xl --out release/TerraDiT-Alpha-XL --include-aux
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, CLIPTextModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import FAMILY_HUB_SUBFOLDER, MODELS, VARIANT_HUB_SUBFOLDER, VAE_ID
from terradit.models.legacy import load_legacy_transformer
from terradit.pipelines.constants import PIPELINE_CLASS_NAME, TOKENIZER_ID
from terradit.pipelines.hub_export import write_self_contained_repo
from terradit.pipelines.pipeline_terradit import pipeline_class_for_family


def resolve_out_dir(out_dir: str, family: str, repo_layout: bool, variant: str | None = None) -> str:
    """Optionally nest ``TerraDiT-Alpha-XL`` / … under a Hub repo root."""
    if not repo_layout:
        return out_dir
    sub = VARIANT_HUB_SUBFOLDER.get(variant or "", FAMILY_HUB_SUBFOLDER[family])
    if os.path.basename(os.path.normpath(out_dir)) == sub:
        return out_dir
    return os.path.join(out_dir, sub)


def convert_checkpoint(
    ckpt: str,
    out_dir: str,
    *,
    family: str | None = None,
    arch: str | None = None,
    legacy: bool | None = None,
    include_aux: bool = False,
    dtype: torch.dtype = torch.float16,
    repo_layout: bool = False,
) -> str:
    """Write a Diffusers pipeline directory and return its path."""
    if family is None and ckpt in MODELS:
        family = MODELS[ckpt]["family"]
    if family is None:
        raise ValueError("Pass --family alpha|sigma|omega or a release name (alpha_xl, …).")

    out_dir = resolve_out_dir(out_dir, family, repo_layout, variant=ckpt if ckpt in VARIANT_HUB_SUBFOLDER else None)
    os.makedirs(out_dir, exist_ok=True)
    transformer = load_legacy_transformer(
        ckpt, family=family, arch=arch, legacy=legacy, torch_dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    transformer.save_pretrained(os.path.join(out_dir, "transformer"))
    scheduler.save_pretrained(os.path.join(out_dir, "scheduler"))

    extra = [n for n in os.listdir(os.path.join(out_dir, "scheduler")) if n != "scheduler_config.json"]
    for name in extra:
        os.remove(os.path.join(out_dir, "scheduler", name))

    pipe_cls = pipeline_class_for_family(family)
    if include_aux:
        pipe = pipe_cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=AutoencoderKL.from_pretrained(VAE_ID),
            text_encoder=CLIPTextModel.from_pretrained(TOKENIZER_ID),
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID),
        )
        pipe.save_pretrained(out_dir, safe_serialization=True)

    write_self_contained_repo(out_dir, family)
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
    ap.add_argument("--repo-layout", action="store_true",
                    help="write under <out>/TerraDiT-Alpha-XL|Sigma-XL|Omega-XL|Omega-B")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    if args.ckpt in MODELS and args.family is None:
        args.family = MODELS[args.ckpt]["family"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = convert_checkpoint(
        args.ckpt, args.out, family=args.family, arch=args.arch,
        legacy=args.legacy, include_aux=args.include_aux, dtype=dtype,
        repo_layout=args.repo_layout,
    )
    class_name = PIPELINE_CLASS_NAME[args.family]
    print(f"[convert] {args.ckpt} -> {out}")
    print(
        f"[convert] load with: DiffusionPipeline.from_pretrained({out!r}, trust_remote_code=True)  "
        f"# or {class_name}.from_pretrained({out!r})"
    )


if __name__ == "__main__":
    main()
