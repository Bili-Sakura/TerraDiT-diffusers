"""TerraDiT-Ω demo: text + geolocation + any geospatial primitive -> satellite image.

Instances are {"type": polygon|polyline|bbox|point, "coords": [[x, y], ...], "tag": "..."}
with x, y in tile pixels [0, 256) (x right, y down). A polygon/polyline implies its bbox
and centre point; a bbox implies its centre point, so specify only the richest primitive.
Edit EXAMPLE below, or pass --example-json / --instances. Geolocation is optional:
with lat/lon the RANGE+ submodule encodes it live; otherwise zeros are used.

    python terradit/omega_demo.py                                    # EXAMPLE below, omega_xl
    python terradit/omega_demo.py --ckpt omega_base                  # SiT-B/2 GALA model
    python terradit/omega_demo.py --condition-type box               # drop polygons/polylines -> boxes+points
    python terradit/omega_demo.py --example-json my_scene.json --lat 51.5 --lon -0.12
    python terradit/omega_demo.py --data-root data/git10m --hf-cache-dir data/git10m/hf --index 0

--condition-type selects which modalities the model sees via the GALA dropout mask:
omega = all primitives, box = bboxes + points only, point = points only.
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.generation import (build_inference_model, load_text_vae, caption_embeds,
                               tag_pooled_embeds, decode_latents, save_images,
                               resolve_sample_index)
from terradit.conditioning import build_conditioning, pack_omega_instances, load_range_model
from terradit.sampling.samplers import euler_sampler
from terradit.viz import render_omega_inputs, active_from_dropouts

# --------------------------------------------------------------------------- #
# Example input: one of every primitive. Tags follow the OSM "key value" convention.
# --------------------------------------------------------------------------- #
EXAMPLE = {
    "caption": "The satellite image shows a small town where a river with dark water runs "
               "from the top left to the bottom right, crossed by a road bridge. A park with "
               "green lawns and trees lies north of the river, a large warehouse with a flat "
               "grey roof sits in the east, and residential houses with red roofs fill the "
               "south-west.",
    "lat": 48.86, "lon": 2.35,            # Paris (set both to None to skip geolocation)
    "instances": [
        # polygon: closed outline (first point repeated automatically)
        {"type": "polygon", "coords": [[20, 20], [110, 15], [120, 90], [30, 100]], "tag": "leisure park"},
        {"type": "polygon", "coords": [[150, 60], [240, 60], [240, 130], [150, 130]], "tag": "building warehouse"},
        # polyline: open path
        {"type": "polyline", "coords": [[0, 30], [60, 110], [140, 170], [255, 235]], "tag": "waterway river"},
        {"type": "polyline", "coords": [[128, 0], [128, 255]], "tag": "highway secondary"},
        # bbox: two corners (x0, y0), (x1, y1)
        {"type": "bbox", "coords": [[20, 170], [70, 215]], "tag": "building house"},
        {"type": "bbox", "coords": [[80, 190], [120, 230]], "tag": "building house"},
        # point: a single (x, y)
        {"type": "point", "coords": [200, 200], "tag": "amenity parking"},
        {"type": "point", "coords": [60, 140], "tag": "natural tree"},
    ],
}

CONDITION_DROPOUTS = {"omega": None, "box": [1.0, 1.0, 0.0, 0.0], "point": [1.0, 1.0, 1.0, 0.0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="omega_xl", help="omega_xl | omega_base (auto-download) or a safetensors/.pt path")
    ap.add_argument("--arch", default=None, help="override the arch in the release config")
    # manual inputs
    ap.add_argument("--example-json", default=None,
                    help="JSON file with {caption, lat, lon, instances} (default: EXAMPLE above)")
    ap.add_argument("--prompt", default=None, help="override the caption")
    ap.add_argument("--instances", default=None, help="override instances: JSON list of {type, coords, tag}")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--no-range", action="store_true", help="skip RANGE+; use a zero location embedding")
    ap.add_argument("--condition-type", default="omega", choices=list(CONDITION_DROPOUTS))
    # dataset mode
    ap.add_argument("--data-root", default=None, help="dataset mode: pull a real tile's instance geometry")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--img-name", default=None, help="target a specific tile by name (overrides --index)")
    # sampling / output
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--out-dir", default="samples/omega")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-show-inputs", action="store_true", help="skip the geometry-overlay PNG/TXT")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    B = args.num_images
    dropouts = CONDITION_DROPOUTS[args.condition_type]

    model = build_inference_model("omega", args.arch, args.ckpt, device, legacy=args.legacy)
    tokenizer, clip, vae = load_text_vae(device)

    if args.data_root is None:
        # ---- manual mode (EXAMPLE / JSON / CLI) ----
        spec = dict(EXAMPLE) if args.example_json is None else json.load(open(args.example_json))
        if args.prompt is not None:
            spec["caption"] = args.prompt
        if args.instances is not None:
            spec["instances"] = json.loads(args.instances)
        if args.lat is not None:
            spec["lat"] = args.lat
        if args.lon is not None:
            spec["lon"] = args.lon
        lat, lon = spec.get("lat"), spec.get("lon")
        range_model = None
        if not args.no_range and lat is not None and lon is not None:
            range_model = load_range_model(device)
        instances = spec["instances"]
        cond = build_conditioning("omega", spec, clip, tokenizer, device, range_model=range_model,
                                  num_images=B, dropout_probs=dropouts)
        g = pack_omega_instances(instances, torch.device("cpu"))
        viz = dict(polygon_xy=g["polygon_xy"], polygon_xy_mask=g["polygon_xy_mask"],
                   polyline_xy=g["polyline_xy"], polyline_xy_mask=g["polyline_xy_mask"],
                   bbox_xyxy=g["bbox_xyxy"], point_xy=g["point_xy"],
                   instance_mask=g["instance_mask"], tags=g["tags"])
        caption = spec.get("caption", "")
        srcdesc = (f"manual ({len(instances)} instances, "
                   f"{'RANGE+ ' + str((lat, lon)) if range_model is not None else 'no geolocation'})")
    else:
        # ---- dataset mode ----
        from terradit.hf import load_git10m
        from terradit.data.dataset import build_dataset
        hf = load_git10m(args.hf_cache_dir)
        ds = build_dataset("omega", args.data_root, hf_dataset=hf, split="test", vae_on_the_fly=True)
        idx = resolve_sample_index(ds, args.img_name, args.index)
        (_img, _lat, cap_ids, cap_attn, _loc, polygon_xy, polygon_xy_mask, polyline_xy,
         polyline_xy_mask, bbox_xyxy, point_xy, format_mask, instance_mask,
         inst_tag_ids, inst_tag_attn) = ds[idx][:15]
        if args.prompt is not None:
            y, y_pooled, _ = caption_embeds(clip, tokenizer, [args.prompt], device)
        else:
            out = clip(cap_ids.unsqueeze(0).to(device), cap_attn.unsqueeze(0).to(device))
            y = F.normalize(out.last_hidden_state, dim=-1)
            y_pooled = F.normalize(out.pooler_output, dim=-1)
        inst_text_embed = tag_pooled_embeds(clip, inst_tag_ids.to(device), inst_tag_attn.to(device))
        loc = F.normalize(torch.from_numpy(ds.location_embeddings[idx]).float(), dim=-1)

        def rb(t):
            return t.unsqueeze(0).to(device).repeat(B, *([1] * t.dim()))
        cond = dict(y=y.repeat(B, 1, 1), y_pooled=y_pooled.repeat(B, 1),
                    loc_embed=loc.unsqueeze(0).to(device).repeat(B, 1),
                    inst_text_embed=inst_text_embed.unsqueeze(0).repeat(B, 1, 1),
                    polygon_xy=rb(polygon_xy), polygon_xy_mask=rb(polygon_xy_mask),
                    polyline_xy=rb(polyline_xy), polyline_xy_mask=rb(polyline_xy_mask),
                    bbox_xyxy=rb(bbox_xyxy), point_xy=rb(point_xy), format_mask=rb(format_mask),
                    instance_mask=rb(instance_mask), dropout_probs=dropouts)
        viz = dict(polygon_xy=polygon_xy, polygon_xy_mask=polygon_xy_mask,
                   polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask,
                   bbox_xyxy=bbox_xyxy, point_xy=point_xy, instance_mask=instance_mask,
                   tags=tokenizer.batch_decode(inst_tag_ids, skip_special_tokens=True))
        caption = args.prompt or ds.caption(idx)
        srcdesc = f"dataset sample {idx} ({ds.metadata[idx].get('img_name')})"

    xT = torch.randn(B, model.in_channels, 32, 32, device=device)
    with torch.no_grad():
        samples = euler_sampler(model, xT, num_steps=args.num_steps, cfg_scale=0.0,
                                path_type="linear", **cond).to(torch.float32)
        imgs = decode_latents(vae, samples, device)

    paths = [os.path.join(args.out_dir, f"omega_{args.condition_type}_{i:02d}.png") for i in range(B)]
    save_images(imgs, paths)
    print(f"wrote {B} images to {args.out_dir} ({srcdesc}, condition={args.condition_type})")

    if not args.no_show_inputs:
        active = active_from_dropouts(dropouts)
        viz_path = os.path.join(args.out_dir, f"omega_{args.condition_type}_inputs.png")
        counts = render_omega_inputs(imgs[0], active=active, out_path=viz_path, **viz)
        with open(os.path.join(args.out_dir, f"omega_{args.condition_type}_inputs.txt"), "w") as f:
            f.write(f"caption: {caption}\nsource: {srcdesc}\ncondition-type: {args.condition_type}  "
                    f"active={[k for k, v in active.items() if v]}\n")
            f.write(f"drawn primitives: {counts}\n\ninstances:\n")
            im = viz["instance_mask"]
            for j in range(int(im.shape[0])):
                if float(im[j]) < 0.5:
                    continue
                kind = ("polygon" if (active["polygon"] and viz["polygon_xy_mask"][j].any())
                        else "polyline" if (active["polyline"] and viz["polyline_xy_mask"][j].any())
                        else "bbox" if active["bbox"] else "point" if active["point"] else "-")
                f.write(f"  inst {j:2d}: {kind:8s}  {viz['tags'][j]!r}\n")
        print(f"wrote inputs overlay: {viz_path} {counts}")


if __name__ == "__main__":
    main()
