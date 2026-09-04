"""Generate images for a held-out test split, for any model family.

Loops a family over a test split (random / spatial / dense), reusing the trainer's
build_model_kwargs so the conditioning matches training/inference exactly, and writes
one image per tile (named by the tile's img_name) for downstream metric computation.

Split files live under <data-root>/splits/<split>/ as a canonical triplet
(metadata.json, inst_metadata.npz, range_plus.npz), downloaded by
scripts/download_data.py; override any path explicitly with --metadata etc.

    python terradit/generate.py --family omega --split random \
        --ckpt BiliSakura/TerraDiT --subfolder TerraDiT-Omega-XL \
        --data-root data/git10m --hf-cache-dir data/git10m/hf \
        --condition-type omega --out-dir output/omega_random --batch-size 16

The "dense" split is the min-15-instances variant of the random split.
"""
import os
import sys
import argparse

import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.pipelines import pipeline_class_for_family
from terradit.conditioning import OMEGA_CONDITION_DROPOUTS
from terradit.data.dataset import build_dataset
from terradit.training.train_terradit import build_model_kwargs
from terradit.hf import DIFFUSERS_REPO, load_git10m, GIT10M_REPO, GIT10M_REVISION
from terradit.viz import save_images

# omega condition-type -> GALA dropout mask (omega=all, box=boxes+points, point=points)
CONDITION_DROPOUTS = OMEGA_CONDITION_DROPOUTS


def split_paths(data_root, split):
    d = os.path.join(data_root, "splits", split)
    return dict(metadata_path=os.path.join(d, "metadata.json"),
                inst_meta_path=os.path.join(d, "inst_metadata.npz"),
                loc_embed_path=os.path.join(d, "range_plus.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["alpha", "sigma", "omega"])
    ap.add_argument("--split", default="random", choices=["random", "spatial", "dense"])
    ap.add_argument("--ckpt", required=True,
                    help=f"Hub repo ({DIFFUSERS_REPO}) or a local Diffusers folder")
    ap.add_argument("--subfolder", default=None,
                    help="Hub subfolder (TerraDiT-Alpha-XL / TerraDiT-Sigma-XL / TerraDiT-Omega-XL / TerraDiT-Omega-B)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--hf-repo-id", default=GIT10M_REPO)
    ap.add_argument("--hf-revision", default=GIT10M_REVISION)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition-type", default="omega", choices=["omega", "box", "point"],
                    help="omega only: which geometry modalities the model sees")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap number of tiles (debug)")
    ap.add_argument("--seed", type=int, default=42)
    # explicit path overrides (bypass the canonical splits/<split>/ layout)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--inst-meta", default=None)
    ap.add_argument("--loc-embed", default=None)
    ap.add_argument("--points-pack", default=None,
                    help="sigma: dense OSM point rasters for the split (paper protocol); "
                         "default derives points from the split's instance geometry")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    sp = split_paths(args.data_root, args.split)
    metadata_path = args.metadata or sp["metadata_path"]
    inst_meta_path = args.inst_meta or sp["inst_meta_path"]
    loc_embed_path = args.loc_embed or sp["loc_embed_path"]
    # sigma test points: the shipped splits carry instance geometry, not dense OSM rasters,
    # so point prompts are sampled from instance points unless a packed raster is given.
    sigma_kwargs = {}
    if args.family == "sigma":
        if args.points_pack:
            sigma_kwargs = dict(point_source="pack", points_pack_path=args.points_pack)
        else:
            sigma_kwargs = dict(point_source="instances")

    pipe_cls = pipeline_class_for_family(args.family)
    pipe = pipe_cls.from_pretrained(args.ckpt, subfolder=args.subfolder)
    pipe._ensure_aux(device)
    pipe = pipe.to(device)
    clip = pipe.text_encoder

    hf = load_git10m(args.hf_cache_dir, repo_id=args.hf_repo_id, revision=args.hf_revision)
    ds = build_dataset(args.family, args.data_root, hf_dataset=hf, split="test",
                       vae_on_the_fly=True, metadata_path=metadata_path,
                       inst_meta_path=inst_meta_path, loc_embed_path=loc_embed_path,
                       **sigma_kwargs)

    N = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[generate] family={args.family} split={args.split} tiles={N} -> {args.out_dir}")

    written = 0
    for start in range(0, N, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, N)))
        samples = [ds[i] for i in idxs]
        # collate only the tensor fields (omega test rows carry a trailing img_name string)
        batch = default_collate([[x for x in s if torch.is_tensor(x)] for s in samples])
        _, _, kwargs = build_model_kwargs(args.family, batch, clip, device, 0, 1)
        pipe_kwargs = dict(
            prompt_embeds=kwargs["y"],
            pooled_prompt_embeds=kwargs["y_pooled"],
            loc_embed=kwargs.get("loc_embed"),
            point_prompts=kwargs.get("point_prompts"),
            pos=kwargs.get("pos"),
            mask=kwargs.get("mask"),
            inst_text_embed=kwargs.get("inst_text_embed"),
            polygon_xy=kwargs.get("polygon_xy"),
            polygon_xy_mask=kwargs.get("polygon_xy_mask"),
            polyline_xy=kwargs.get("polyline_xy"),
            polyline_xy_mask=kwargs.get("polyline_xy_mask"),
            bbox_xyxy=kwargs.get("bbox_xyxy"),
            point_xy=kwargs.get("point_xy"),
            format_mask=kwargs.get("format_mask"),
            instance_mask=kwargs.get("instance_mask"),
        )
        if args.family == "omega":
            pipe_kwargs["dropout_probs"] = CONDITION_DROPOUTS[args.condition_type]
            pipe_kwargs["condition_type"] = args.condition_type

        b = len(idxs)
        out = pipe(
            num_inference_steps=args.num_steps,
            guidance_scale=0.0,
            output_type="pt",
            **{k: v for k, v in pipe_kwargs.items() if v is not None},
        )
        imgs = out.images

        paths = []
        for i in idxs:
            name = os.path.splitext(ds.metadata[i]["img_name"])[0]
            paths.append(os.path.join(args.out_dir, name + ".png"))
        save_images(imgs, paths)
        written += b
        if start // args.batch_size % 20 == 0:
            print(f"  {written}/{N}")

    print(f"[generate] done: {written} images in {args.out_dir}")


if __name__ == "__main__":
    main()
