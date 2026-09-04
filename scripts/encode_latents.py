"""Precompute SDXL-VAE latent moments for training (one .npy per tile).

Writes ``<data-root>/latents/<img_name>.npy`` holding ``[8, 32, 32]`` = mean||std of the
SDXL VAE posterior (fp16), exactly what ``terradit/training/train_terradit.py`` samples from
when ``--vae-on-the-fly`` is not set. Skips tiles that already exist, so it is resumable
and can be sharded across jobs/GPUs.

    # all tiles for a family, single GPU
    python scripts/encode_latents.py --data-root data/git10m --hf-cache-dir data/git10m/hf --family omega

    # 4 GPUs on one node (each process takes an interleaved shard)
    accelerate launch --num_processes 4 scripts/encode_latents.py --data-root data/git10m \
        --hf-cache-dir data/git10m/hf --family omega

    # explicit row range (e.g. for an array job)
    python scripts/encode_latents.py ... --start 0 --end 500000

Imagery comes from the pinned Git-10M snapshot via ``hf_idx``; ~2M tiles x 16 KB = ~33 GB
in fp16. Fine-tuning on a subset? Use --max-samples in the trainer and --end here.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import load_git10m, GIT10M_REPO, GIT10M_REVISION


class TileImages(Dataset):
    def __init__(self, rows, hf, images_root=None):
        self.rows, self.hf, self.images_root = rows, hf, images_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.hf is not None and "hf_idx" in r:
            img = self.hf[int(r["hf_idx"])]["image"]
        else:
            from PIL import Image
            img = Image.open(os.path.join(self.images_root, r["img_name"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
        return x, r["img_name"]


@torch.no_grad()
def encode_moments(vae, x):
    img = x.float() / 127.5 - 1.0
    post = vae.encode(img).latent_dist
    return torch.cat([post.mean, post.std], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--family", default="omega", choices=["alpha", "sigma", "omega"],
                    help="which metadata file enumerates the tiles (omega covers the full 2M)")
    ap.add_argument("--metadata", default=None, help="explicit metadata json (overrides --family)")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--hf-repo-id", default=GIT10M_REPO)
    ap.add_argument("--hf-revision", default=GIT10M_REVISION)
    ap.add_argument("--images-root", default=None, help="disk fallback when rows lack hf_idx")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    # optional accelerate sharding (works as a plain script too)
    try:
        from accelerate import PartialState
        state = PartialState()
        rank, world, device = state.process_index, state.num_processes, state.device
    except Exception:
        rank, world, device = 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = args.metadata or os.path.join(args.data_root, "metadata", f"{args.family}.json")
    rows = json.load(open(meta_path))
    rows = rows[args.start: args.end]
    out_dir = os.path.join(args.data_root, "latents")
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["img_name"] + ".npy"))]
    todo = todo[rank::world]
    if rank == 0:
        print(f"[latents] {len(rows)} rows, {len(todo) * world} to encode -> {out_dir} "
              f"({world} process(es), {args.dtype})")
    if not todo:
        return

    hf = load_git10m(args.hf_cache_dir, repo_id=args.hf_repo_id, revision=args.hf_revision) \
        if any("hf_idx" in r for r in todo) else None
    loader = DataLoader(TileImages(todo, hf, args.images_root), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device).eval()
    save_dtype = np.float16 if args.dtype == "fp16" else np.float32

    for x, names in tqdm(loader, disable=rank != 0):
        m = encode_moments(vae, x.to(device)).cpu().numpy().astype(save_dtype)
        for k, name in enumerate(names):
            np.save(os.path.join(out_dir, name + ".npy"), m[k])
    if rank == 0:
        print("[latents] done")


if __name__ == "__main__":
    main()
