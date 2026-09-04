"""TerraDiT-Σ demo: text + geolocation + point prompts -> satellite image.

Point prompts are (x, y, "osm tag") with x, y in tile pixels [0, 256). Edit EXAMPLE
below, pass --points, or let the demo scatter random tagged points (--random-points K).
Geolocation is optional: with --lat/--lon the RANGE+ submodule encodes it live;
otherwise a zero embedding is used (in-distribution: sigma trained with location dropout).

    python terradit/sigma_demo.py                                   # EXAMPLE below
    python terradit/sigma_demo.py --random-points 12 --seed 3       # random tags/positions
    python terradit/sigma_demo.py --prompt "..." --points '[[120,80,"building"],[60,200,"waterway river"]]' \
        --lat 40.71 --lon -74.01
    python terradit/sigma_demo.py --data-root data/git10m --hf-cache-dir data/git10m/hf --index 0
"""
import os
import sys
import json
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.pipelines import TerraDiTPipeline
from terradit.conditioning import (caption_embeds, load_range_model, pack_sigma_points,
                                   resolve_sample_index, tag_pooled_embeds)
from terradit.viz import render_sigma_inputs, save_images

# --------------------------------------------------------------------------- #
# Example input. Tags follow the OSM "key value" convention used in training
# (see osm/tag_vocab.pt in MVRL/TerraDiT-data for the full 1.7k-entry vocabulary).
# --------------------------------------------------------------------------- #
EXAMPLE = {
    "caption": "The satellite image shows a suburban neighborhood with detached houses "
               "along curving residential streets, a school with a large sports field, "
               "and a small pond surrounded by trees in the lower right.",
    "lat": 38.65, "lon": -90.31,          # St. Louis, MO (set both to None to skip geolocation)
    "points": [                           # (x, y, tag), x to the right, y down
        (40, 40, "building house"), (70, 55, "building house"), (100, 40, "building house"),
        (130, 60, "building house"), (160, 45, "building house"), (200, 50, "building house"),
        (128, 100, "highway residential"), (30, 150, "highway residential"), (220, 140, "highway residential"),
        (90, 170, "amenity school"), (110, 205, "leisure pitch"),
        (205, 215, "natural water"), (190, 235, "natural wood"),
    ],
}

# tags used by --random-points (a common subset of the vocabulary)
COMMON_TAGS = [
    "building", "building house", "building apartments", "building industrial", "building commercial",
    "highway residential", "highway service", "highway tertiary", "highway secondary", "highway primary",
    "landuse residential", "landuse farmland", "landuse grass", "landuse industrial", "landuse forest",
    "natural water", "natural wood", "waterway river", "waterway stream", "amenity parking",
    "leisure park", "leisure pitch", "amenity school", "railway rail", "power line",
]


def random_points(k, rng):
    return [(rng.randrange(8, 248), rng.randrange(8, 248), rng.choice(COMMON_TAGS)) for _ in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="sigma_xl", help="sigma_xl (auto-download) or a safetensors/.pt path")
    ap.add_argument("--arch", default=None, help="override the arch in the release config")
    # manual inputs
    ap.add_argument("--prompt", default=None, help="global caption (default: EXAMPLE caption)")
    ap.add_argument("--points", default=None, help='JSON list of [x, y, "tag"] (default: EXAMPLE points)')
    ap.add_argument("--random-points", type=int, default=None, metavar="K",
                    help="ignore EXAMPLE/--points and scatter K random tagged points")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--no-range", action="store_true", help="skip RANGE+; use a zero location embedding")
    # dataset mode
    ap.add_argument("--data-root", default=None, help="dataset mode: pull a real tile's conditioning")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--img-name", default=None, help="target a specific tile by name (overrides --index)")
    # sampling / output
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--out-dir", default="samples/sigma")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-show-inputs", action="store_true", help="skip the point-overlay PNG/TXT")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    B = args.num_images

    pipe = TerraDiTPipeline.from_checkpoint(
        args.ckpt, family="sigma", arch=args.arch, legacy=args.legacy, device=device,
    )
    tokenizer, clip = pipe.tokenizer, pipe.text_encoder

    if args.data_root is None:
        # ---- manual mode (EXAMPLE / CLI) ----
        caption = args.prompt if args.prompt is not None else EXAMPLE["caption"]
        if args.random_points is not None:
            points = random_points(args.random_points, rng)
        elif args.points is not None:
            points = [(p[0], p[1], p[2] if len(p) > 2 else "") for p in json.loads(args.points)]
        else:
            points = list(EXAMPLE["points"])
        lat = args.lat if args.lat is not None else EXAMPLE["lat"]
        lon = args.lon if args.lon is not None else EXAMPLE["lon"]
        if args.lat is None and args.lon is None and args.prompt is not None:
            lat = lon = None  # a custom prompt without coords -> no location
        range_model = None
        if not args.no_range and lat is not None and lon is not None:
            range_model = load_range_model(device)
        coords_raw, mask_raw, point_tags = pack_sigma_points(points, torch.device("cpu"))
        srcdesc = (f"manual ({len(points)} points, "
                   f"{'RANGE+ ' + str((lat, lon)) if range_model is not None else 'no geolocation'})")
        pipe_kwargs = dict(prompt=caption, points=points, lat=lat, lon=lon, range_model=range_model)
    else:
        # ---- dataset mode ----
        from terradit.hf import load_git10m
        from terradit.data.dataset import build_dataset
        hf = load_git10m(args.hf_cache_dir)
        ds = build_dataset("sigma", args.data_root, hf_dataset=hf, split="test",
                           vae_on_the_fly=True, use_packed_points=True)
        idx = resolve_sample_index(ds, args.img_name, args.index)
        (_img, _lat, cap_ids, cap_attn, _loc, p_ids, p_attn, pos, mask) = ds[idx]
        coords_raw, mask_raw = pos.clone(), mask.clone()
        point_tags = tokenizer.batch_decode(p_ids, skip_special_tokens=True)
        if args.prompt is not None:
            y, y_pooled, _ = caption_embeds(clip, tokenizer, [args.prompt], device)
        else:
            out = clip(cap_ids.unsqueeze(0).to(device), cap_attn.unsqueeze(0).to(device))
            y = F.normalize(out.last_hidden_state, dim=-1)
            y_pooled = F.normalize(out.pooler_output, dim=-1)
        loc = F.normalize(torch.from_numpy(ds.location_embeddings[idx]).float(), dim=-1)
        point_embed = tag_pooled_embeds(clip, p_ids.to(device), p_attn.to(device))
        pipe_kwargs = dict(
            prompt_embeds=y, pooled_prompt_embeds=y_pooled,
            loc_embed=loc.unsqueeze(0).to(device),
            point_prompts=point_embed.unsqueeze(0),
            pos=pos.unsqueeze(0).to(device),
            mask=mask.unsqueeze(0).to(device),
        )
        caption = args.prompt or ds.caption(idx)
        srcdesc = f"dataset sample {idx} ({ds.metadata[idx].get('img_name')})"

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        num_images_per_prompt=B,
        num_inference_steps=args.num_steps,
        guidance_scale=0.0,
        generator=generator,
        output_type="pt",
        **pipe_kwargs,
    )
    imgs = out.images

    paths = [os.path.join(args.out_dir, f"sigma_{i:02d}.png") for i in range(B)]
    save_images(imgs, paths)
    print(f"wrote {B} images to {args.out_dir} ({srcdesc})")

    if not args.no_show_inputs:
        viz_path = os.path.join(args.out_dir, "sigma_inputs.png")
        n = render_sigma_inputs(imgs[0], coords_raw, mask_raw, point_tags, viz_path)
        with open(os.path.join(args.out_dir, "sigma_inputs.txt"), "w") as f:
            f.write(f"caption: {caption}\nsource: {srcdesc}\n\npoint prompts ({n}):\n")
            for i in range(coords_raw.shape[0]):
                if int(mask_raw[i]) != 0:
                    continue
                r, c = int(coords_raw[i, 0]), int(coords_raw[i, 1])
                f.write(f"  (x={c:3d}, y={r:3d})  {point_tags[i]!r}\n")
        print(f"wrote inputs overlay: {viz_path} ({n} points)")


if __name__ == "__main__":
    main()
