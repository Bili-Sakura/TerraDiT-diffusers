"""Image-quality metrics for a generated test split (the paper protocol).

Given a folder of generated tiles named ``<img_name stem>.png`` and the split's
metadata, computes:

* **FID**   -- torchmetrics FrechetInceptionDistance (Inception-v3, 2048-d), real = GT tiles.
* **CLIP**  -- openai/clip-vit-large-patch14 image-text logit between each tile and its caption.
* **LPIPS** -- torchmetrics LPIPS (SqueezeNet) between generated and GT tile, inputs in [-1, 1].
* **SSIM**  -- torchmetrics SSIM (data_range=1) between generated and GT tile.

Ground-truth tiles are fetched from the pinned Git-10M snapshot by ``hf_idx`` and cached
as PNGs under ``<gt-cache>/<split>/`` the first time (``--gt-dir`` to point at an existing
folder instead). Generated tiles are resized to the GT size if they differ.

    python -m terradit.eval.metrics --gen-dir output/omega_random --split random \
        --data-root data/git10m --hf-cache-dir data/git10m/hf --out results/omega_random.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from terradit.hf import load_git10m, GIT10M_REPO, GIT10M_REVISION

CLIP_ID = "openai/clip-vit-large-patch14"
ALL_METRICS = ("fid", "clip", "lpips", "ssim")


def stem(name):
    return os.path.splitext(name)[0]


def load_split_rows(data_root, split, metadata_path=None):
    path = metadata_path or os.path.join(data_root, "splits", split, "metadata.json")
    return json.load(open(path))


def attach_captions(rows, hf):
    """Fill ``row["caption"]`` from the Git-10M row (the shipped metadata carries no text)."""
    for r in rows:
        if not r.get("caption") and "hf_idx" in r:
            r["caption"] = hf[int(r["hf_idx"])].get("caption", "")
    return rows


def ensure_gt_tiles(rows, gt_dir, hf=None, hf_cache_dir=None, hf_repo_id=GIT10M_REPO,
                    hf_revision=GIT10M_REVISION):
    """Write ``<gt_dir>/<stem>.png`` for every row missing on disk (from Git-10M by hf_idx)."""
    os.makedirs(gt_dir, exist_ok=True)
    missing = [r for r in rows if not os.path.exists(os.path.join(gt_dir, stem(r["img_name"]) + ".png"))]
    if not missing:
        return gt_dir
    print(f"[gt] fetching {len(missing)} ground-truth tiles from {hf_repo_id}@{hf_revision[:8]} -> {gt_dir}")
    if hf is None:
        hf = load_git10m(hf_cache_dir, repo_id=hf_repo_id, revision=hf_revision)
    for r in tqdm(missing):
        row = hf[int(r["hf_idx"])]
        if row.get("img_name") not in (None, r["img_name"]):
            raise ValueError(f"hf_idx mismatch: {row.get('img_name')} != {r['img_name']}")
        row["image"].convert("RGB").save(os.path.join(gt_dir, stem(r["img_name"]) + ".png"))
    return gt_dir


def _pairs(rows, gen_dir, gt_dir):
    """(row, gen_path, gt_path) for every row whose generated tile exists."""
    out = []
    for r in rows:
        s = stem(r["img_name"])
        gen = os.path.join(gen_dir, s + ".png")
        if not os.path.exists(gen):
            gen = os.path.join(gen_dir, s + ".jpg")
        gt = os.path.join(gt_dir, s + ".png")
        if not os.path.exists(gt):
            gt = os.path.join(gt_dir, s + ".jpg")
        if os.path.exists(gen) and os.path.exists(gt):
            out.append((r, gen, gt))
    return out


def _load_u8(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)


@torch.no_grad()
def compute_fid(pairs, device, batch_size=64):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048).to(device)
    for is_real, col in ((True, 2), (False, 1)):
        paths = [p[col] for p in pairs]
        gt_size = Image.open(pairs[0][2]).size
        for i in tqdm(range(0, len(paths), batch_size), desc=f"fid[{'real' if is_real else 'gen'}]"):
            x = torch.stack([_load_u8(p, gt_size) for p in paths[i:i + batch_size]]).to(device)
            fid.update(x, real=is_real)
    return float(fid.compute())


@torch.no_grad()
def compute_clip(pairs, device, batch_size=32):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="clip"):
        chunk = pairs[i:i + batch_size]
        imgs = [Image.open(p[1]).convert("RGB") for p in chunk]
        caps = [p[0].get("caption", "") for p in chunk]
        inp = proc(text=caps, images=imgs, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp)
        # per-sample image<->own caption logit (diagonal), as in the paper's script
        scores.extend(torch.diagonal(out.logits_per_image).float().cpu().tolist())
    return float(np.mean(scores))


@torch.no_grad()
def compute_pairwise(pairs, device, which=("lpips", "ssim"), batch_size=32):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device) if "lpips" in which else None
    ss = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device) if "ssim" in which else None
    lpips_scores, ssim_scores = [], []
    gt_size = Image.open(pairs[0][2]).size
    for i in tqdm(range(0, len(pairs), batch_size), desc="lpips/ssim"):
        chunk = pairs[i:i + batch_size]
        gen = torch.stack([_load_u8(p[1], gt_size) for p in chunk]).to(device).float() / 255.0
        gt = torch.stack([_load_u8(p[2]) for p in chunk]).to(device).float() / 255.0
        if lp is not None:
            for g, t in zip(gen, gt):   # torchmetrics LPIPS returns a mean; do per-sample
                lpips_scores.append(float(lp(2 * g[None] - 1, 2 * t[None] - 1)))
        if ss is not None:
            ssim_scores.extend(ss(gen, gt).float().cpu().tolist())
    out = {}
    if lp is not None:
        out["lpips"] = float(np.mean(lpips_scores))
    if ss is not None:
        out["ssim"] = float(np.mean(ssim_scores))
    return out


def evaluate_folder(gen_dir, rows, gt_dir, device, metrics=ALL_METRICS):
    pairs = _pairs(rows, gen_dir, gt_dir)
    if not pairs:
        raise FileNotFoundError(f"no (generated, gt) pairs found under {gen_dir} / {gt_dir}")
    res = {"num_tiles": len(pairs), "num_rows": len(rows)}
    if len(pairs) < len(rows):
        print(f"[metrics] warning: {len(rows) - len(pairs)} rows have no generated tile; "
              f"metrics computed on {len(pairs)}")
    if "fid" in metrics:
        res["fid"] = compute_fid(pairs, device)
    if "clip" in metrics:
        res["clip_score"] = compute_clip(pairs, device)
    pw = tuple(m for m in ("lpips", "ssim") if m in metrics)
    if pw:
        res.update(compute_pairwise(pairs, device, pw))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--split", default="random", choices=["random", "spatial", "dense"])
    ap.add_argument("--data-root", default="data/git10m")
    ap.add_argument("--metadata", default=None, help="override <data-root>/splits/<split>/metadata.json")
    ap.add_argument("--gt-dir", default=None, help="existing GT folder; default <gt-cache>/<split>")
    ap.add_argument("--gt-cache", default="data/git10m/gt")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--hf-repo-id", default=GIT10M_REPO)
    ap.add_argument("--hf-revision", default=GIT10M_REVISION)
    ap.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=list(ALL_METRICS))
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (debug)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_split_rows(args.data_root, args.split, args.metadata)
    if args.limit:
        rows = rows[:args.limit]
    hf = load_git10m(args.hf_cache_dir, repo_id=args.hf_repo_id, revision=args.hf_revision)
    attach_captions(rows, hf)
    gt_dir = args.gt_dir or ensure_gt_tiles(rows, os.path.join(args.gt_cache, args.split), hf=hf)
    res = evaluate_folder(args.gen_dir, rows, gt_dir, device, tuple(args.metrics))
    res.update(gen_dir=args.gen_dir, split=args.split, gt_dir=gt_dir)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
