"""TerraDiT-α demo: text -> satellite image.

Edit PROMPTS below or pass --prompt. Weights download automatically on first use.

    python terradit/alpha_demo.py                                   # built-in prompts
    python terradit/alpha_demo.py --prompt "a coastal city with a harbor" --num-images 4
    python terradit/alpha_demo.py --data-root data/git10m --hf-cache-dir data/git10m/hf --index 0 1 2
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import DIFFUSERS_REPO, FAMILY_HUB_SUBFOLDER
from terradit.pipelines import TerraDiTAlphaPipeline
from terradit.viz import save_images

# --------------------------------------------------------------------------- #
# Example inputs. Git-10M captions are long and descriptive; the model responds best
# to the same style (scene type, layout, colours, notable features).
# --------------------------------------------------------------------------- #
PROMPTS = [
    "The satellite image shows a dense residential neighborhood with rows of houses "
    "with grey and brown rooftops, narrow tree-lined streets forming a regular grid, "
    "and a small park with green lawns near the center.",
    "The satellite image depicts an agricultural landscape with large rectangular "
    "farmland plots in shades of green and brown, separated by dirt tracks, with a "
    "single farmstead and a cluster of trees in the upper left.",
    "The satellite image shows an industrial area with several large warehouses with "
    "flat white roofs, parking lots filled with trucks, and a railway line running "
    "diagonally across the lower part of the image.",
    "The satellite image shows a coastal town where a marina with many small boats "
    "meets the dark blue sea, with a sandy beach along the shore and low-rise "
    "buildings with red roofs behind it.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DIFFUSERS_REPO,
                    help=f"Hub repo ({DIFFUSERS_REPO}), release name (alpha_xl), or a safetensors/.pt path")
    ap.add_argument("--subfolder", default=None,
                    help=f"Hub subfolder (default: {FAMILY_HUB_SUBFOLDER['alpha']})")
    ap.add_argument("--arch", default=None, help="override the arch in the release config")
    ap.add_argument("--prompt", nargs="+", default=None, help="free-text caption(s); default: PROMPTS above")
    ap.add_argument("--index", type=int, nargs="+", default=None,
                    help="use the real Git-10M captions of these <data-root>/metadata/alpha.json rows")
    ap.add_argument("--data-root", default=None, help="required with --index")
    ap.add_argument("--hf-cache-dir", default=None, help="Git-10M cache (captions are read from it)")
    ap.add_argument("--num-images", type=int, default=4, help="variations per caption")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--out-dir", default="samples/alpha")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=None,
                    help="override the legacy flag (only for old pre-fix weights)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.index is not None:
        if args.data_root is None:
            ap.error("--index requires --data-root")
        from terradit.data.dataset import build_dataset
        from terradit.hf import load_git10m
        ds = build_dataset("alpha", args.data_root, hf_dataset=load_git10m(args.hf_cache_dir), split="test")
        prompts = [ds.caption(i) for i in args.index]
    else:
        prompts = args.prompt or PROMPTS
    for c, cap in enumerate(prompts):
        print(f"[caption {c}] {cap[:140]}{'...' if len(cap) > 140 else ''}")

    pipe = TerraDiTAlphaPipeline.from_checkpoint(
        args.ckpt, family="alpha", arch=args.arch, legacy=args.legacy, device=device,
        subfolder=args.subfolder,
    )
    V = args.num_images
    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=prompts,
        num_images_per_prompt=V,
        num_inference_steps=args.num_steps,
        guidance_scale=0.0,
        generator=generator,
        output_type="pt",
    )
    imgs = out.images
    B = imgs.shape[0] if torch.is_tensor(imgs) else len(imgs)

    paths = [os.path.join(args.out_dir, f"alpha_c{c:02d}_v{v:02d}.png")
             for c in range(len(prompts)) for v in range(V)]
    save_images(imgs, paths)
    with open(os.path.join(args.out_dir, "alpha_prompts.txt"), "w") as f:
        for c, cap in enumerate(prompts):
            f.write(f"c{c:02d}: {cap}\n")
    print(f"wrote {B} images to {args.out_dir} ({len(prompts)} caption(s) x {V} variation(s))")


if __name__ == "__main__":
    main()
