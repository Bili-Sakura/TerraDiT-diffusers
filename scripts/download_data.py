"""Download the TerraDiT derived-data artifacts (MVRL/TerraDiT-data) into a data_root.

    python scripts/download_data.py --data-root data/git10m                 # eval-only (small)
    python scripts/download_data.py --data-root data/git10m --family omega  # + omega training
    python scripts/download_data.py --data-root data/git10m --family all    # everything

The repo mirrors the ``data_root`` layout documented in terradit/data/dataset.py::

    metadata/{alpha,sigma,omega}.json          img_name, Google_location, lat/lon, hf_idx
    splits/{random,spatial,dense}/             held-out test splits (eval)
    osm/tag_vocab.pt                           OSM tag vocabulary
    instance_metadata/inst_metadata.npz        omega instance geometry (2.1 GB)
    points.pack                                sigma point supervision (1.2 GB)
    range_plus/omega.npz                       RANGE+ per-tile embeddings for omega (fp16)
    range_plus/sigma.npy                       RANGE+ per-tile embeddings for sigma (fp16)

Imagery is NOT here: it comes from ``lcybuaa/Git-10M`` (pinned snapshot, ~394 GB) via
``--hf-cache-dir`` at train/eval time. SDXL latents are not distributed either; build
them with scripts/encode_latents.py or train with ``--vae-on-the-fly``.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import DATA_REPO

# what each selection needs beyond the always-included eval core
CORE = ["metadata/*", "splits/**", "osm/*", "README.md"]
FAMILY_EXTRA = {
    "alpha": [],
    "sigma": ["points.pack", "range_plus/sigma.npy"],
    "omega": ["instance_metadata/inst_metadata.npz", "range_plus/omega.npz"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/git10m")
    ap.add_argument("--family", nargs="+", default=[], choices=["alpha", "sigma", "omega", "all"],
                    help="also fetch training artifacts for these families (default: eval core only)")
    ap.add_argument("--repo-id", default=DATA_REPO)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    fams = ["alpha", "sigma", "omega"] if "all" in args.family else args.family
    patterns = list(CORE)
    for f in fams:
        patterns += FAMILY_EXTRA[f]

    from huggingface_hub import snapshot_download
    os.makedirs(args.data_root, exist_ok=True)
    print(f"[download] {args.repo_id} -> {args.data_root}\n  patterns: {patterns}")
    snapshot_download(args.repo_id, repo_type="dataset", revision=args.revision,
                      local_dir=args.data_root, allow_patterns=patterns)
    print("[download] done. Layout:")
    for root, dirs, files in os.walk(args.data_root):
        depth = root[len(args.data_root):].count(os.sep)
        if depth > 2:
            continue
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"  {os.path.relpath(p, args.data_root):50s} {os.path.getsize(p)/1e9:7.2f} GB")


if __name__ == "__main__":
    main()
