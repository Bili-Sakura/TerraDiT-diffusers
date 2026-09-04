"""Build points.pack from the legacy per-tile .pt / .json sigma supervision.

Packs only the tiles referenced by the sigma metadata (their Google_location),
not the full pixel_tensors directory. Tiles with no OSM coverage are skipped (the
reader returns an all -1 tensor for any tile absent from the pack).

The released points.pack (MVRL/TerraDiT-data) was built with this script; you only
need it to pack your own point supervision. Run once (heavy: hundreds of thousands
of tiles):
    python -m terradit.data.preprocessing.pack_points \
        --metadata  metadata_cities.json \
        --pixel-dir osm_data/pixel_tensors \
        --tag-dir   osm_data/list_tags \
        --out       data/git10m/points.pack
"""
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from terradit.data.preprocessing.points_pack import PointPackWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="sigma metadata json (enumerates Google_location)")
    ap.add_argument("--pixel-dir", required=True, help="legacy pixel_tensors dir of <gloc>.pt")
    ap.add_argument("--tag-dir", required=True, help="legacy list_tags dir of <gloc>.json")
    ap.add_argument("--out", required=True, help="output points.pack path")
    ap.add_argument("--limit", type=int, default=None, help="cap #tiles (for testing)")
    args = ap.parse_args()

    metadata = json.load(open(args.metadata, "r"))
    seen, glocs = set(), []
    for s in metadata:
        g = s["Google_location"]
        if g not in seen:
            seen.add(g)
            glocs.append(g)
    if args.limit:
        glocs = glocs[: args.limit]
    print(f"{len(glocs)} unique tiles referenced by metadata")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_added = n_missing = 0
    with PointPackWriter(args.out) as w:
        for g in tqdm(glocs):
            pt_path = os.path.join(args.pixel_dir, g + ".pt")
            if not os.path.exists(pt_path):
                n_missing += 1
                continue
            pixel_tensor = torch.load(pt_path).long()
            tags_path = os.path.join(args.tag_dir, g + ".json")
            tags = json.load(open(tags_path, "r")) if os.path.exists(tags_path) else None
            w.add(g, pixel_tensor, tags)
            n_added += 1

    size_gb = os.path.getsize(args.out) / 1e9
    print(f"wrote {args.out}: {n_added} tiles packed, {n_missing} missing/skipped, {size_gb:.3f} GB")


if __name__ == "__main__":
    main()
