"""Compute RANGE+ geolocation embeddings for a metadata file (for training on your own tiles).

Reads ``latitude`` / ``longitude`` from each metadata row and writes an ``.npz`` with
``embeddings`` [N, 1280] (fp16) and ``locs`` [N, 2] (lon, lat), row-aligned with the
metadata, i.e. the same layout as ``range_plus/omega.npz`` in MVRL/TerraDiT-data.
Requires the RANGE submodule (``git submodule update --init``); SatCLIP weights and
the RANGE database are fetched from the Hub on first use.

    python scripts/precompute_range_plus.py --metadata data/git10m/metadata/my.json \
        --out data/git10m/range_plus/my.npz
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.conditioning import load_range_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lat-key", default="latitude"); ap.add_argument("--lon-key", default="longitude")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rm = load_range_model(device)
    if rm is None:
        sys.exit("RANGE+ could not be loaded; see the message above.")

    rows = json.load(open(args.metadata))
    locs = np.array([[float(r[args.lon_key]), float(r[args.lat_key])] for r in rows], dtype=np.float64)
    emb = np.zeros((len(rows), 1280), dtype=np.float16)
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size)):
            coords = torch.from_numpy(locs[i:i + args.batch_size]).to(device)
            e = rm(coords)
            e = torch.from_numpy(e) if isinstance(e, np.ndarray) else e
            emb[i:i + args.batch_size] = e.float().cpu().numpy().astype(np.float16)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, embeddings=emb, locs=locs)
    print(f"wrote {args.out}: embeddings {emb.shape} fp16, locs {locs.shape}")


if __name__ == "__main__":
    main()
