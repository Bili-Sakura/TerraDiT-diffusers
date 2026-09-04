"""Generate a test split with a released model and score it (FID, CLIP, LPIPS, SSIM).

One command per (model, split, condition) cell of the paper tables:

    python terradit/evaluate.py --ckpt omega_xl --split random --condition-type omega \
        --data-root data/git10m --hf-cache-dir data/git10m/hf
    python terradit/evaluate.py --ckpt omega_xl --split spatial --condition-type box
    python terradit/evaluate.py --ckpt alpha_xl --split random
    python terradit/evaluate.py --ckpt sigma_xl --split random

Splits: ``random`` and ``spatial`` are the protocol; ``dense`` is the >=15-instance subset
of random (opt-in). Generated tiles go to ``<out-root>/<name>_<split>_<cond>/``, results to
``<out-root>/results/<name>_<split>_<cond>.json``. Re-running skips generation when every
tile already exists (``--regenerate`` to force). See scripts/eval_all.sh for the full grid.
"""
import os
import sys
import json
import argparse
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import (MODELS, resolve_checkpoint, load_weights, load_git10m,
                         GIT10M_REPO, GIT10M_REVISION)
from terradit.eval.metrics import (load_split_rows, ensure_gt_tiles, evaluate_folder,
                                 attach_captions, ALL_METRICS, stem)


def infer_family(ckpt):
    if ckpt in MODELS:
        return MODELS[ckpt]["family"]
    _, config = load_weights(resolve_checkpoint(ckpt))
    if config and "family" in config:
        return config["family"]
    raise ValueError("cannot infer --family from checkpoint; pass --family explicitly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="release name (omega_xl ...) or safetensors/.pt path")
    ap.add_argument("--family", default=None, choices=["alpha", "sigma", "omega"])
    ap.add_argument("--arch", default=None)
    ap.add_argument("--split", default="random", choices=["random", "spatial", "dense"])
    ap.add_argument("--condition-type", default="omega", choices=["omega", "box", "point"],
                    help="omega only: which geometry modalities the model sees")
    ap.add_argument("--data-root", default="data/git10m")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--hf-repo-id", default=GIT10M_REPO)
    ap.add_argument("--hf-revision", default=GIT10M_REVISION)
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--name", default=None, help="run name (default: ckpt name)")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="first N tiles only (debug)")
    ap.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=list(ALL_METRICS))
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--metrics-only", action="store_true", help="skip generation entirely")
    # explicit split overrides (e.g. a sigma-aligned split)
    ap.add_argument("--metadata", default=None); ap.add_argument("--inst-meta", default=None)
    ap.add_argument("--loc-embed", default=None)
    args = ap.parse_args()

    family = args.family or infer_family(args.ckpt)
    name = args.name or (args.ckpt if args.ckpt in MODELS else stem(os.path.basename(args.ckpt.rstrip("/"))))
    cond = args.condition_type if family == "omega" else "text"
    run = f"{name}_{args.split}_{cond}"
    gen_dir = os.path.join(args.out_root, run)
    res_path = os.path.join(args.out_root, "results", run + ".json")

    rows = load_split_rows(args.data_root, args.split, args.metadata)
    if args.limit:
        rows = rows[:args.limit]

    have = sum(os.path.exists(os.path.join(gen_dir, stem(r["img_name"]) + ".png")) for r in rows)
    if not args.metrics_only and (args.regenerate or have < len(rows)):
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "generate.py"),
               "--family", family, "--split", args.split, "--ckpt", args.ckpt,
               "--data-root", args.data_root, "--out-dir", gen_dir,
               "--num-steps", str(args.num_steps), "--batch-size", str(args.batch_size),
               "--seed", str(args.seed), "--hf-repo-id", args.hf_repo_id, "--hf-revision", args.hf_revision]
        if args.hf_cache_dir:
            cmd += ["--hf-cache-dir", args.hf_cache_dir]
        if args.arch:
            cmd += ["--arch", args.arch]
        if family == "omega":
            cmd += ["--condition-type", args.condition_type]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        for flag, val in (("--metadata", args.metadata), ("--inst-meta", args.inst_meta),
                          ("--loc-embed", args.loc_embed)):
            if val:
                cmd += [flag, val]
        print("[evaluate] generating:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"[evaluate] {have}/{len(rows)} tiles present in {gen_dir}; skipping generation")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf = load_git10m(args.hf_cache_dir, repo_id=args.hf_repo_id, revision=args.hf_revision)
    attach_captions(rows, hf)   # captions live in Git-10M, not in the shipped metadata
    gt_dir = ensure_gt_tiles(rows, os.path.join(args.data_root, "gt", args.split), hf=hf)
    res = evaluate_folder(gen_dir, rows, gt_dir, device, tuple(args.metrics))
    res.update(run=run, ckpt=args.ckpt, family=family, split=args.split, condition_type=cond,
               num_steps=args.num_steps, seed=args.seed, gen_dir=gen_dir)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    json.dump(res, open(res_path, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[evaluate] -> {res_path}")


if __name__ == "__main__":
    main()
