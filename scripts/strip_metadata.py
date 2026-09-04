"""Reduce metadata files to the fields we may redistribute.

Git-10M is released under CC-BY-NC-ND-4.0, so its captions and per-image statistics are
not redistributed here. The shipped metadata keeps only tile identifiers and coordinates;
the code reads captions from the user's own Git-10M download via ``hf_idx``.

Kept fields: img_name, Google_location, latitude, longitude, hf_idx.
Tile coordinates are the north-west corner of the zoom-17 Web-Mercator tile named in
``Google_location`` (zoom_x_y), i.e. a function of the tile index, not Git-10M content.

    python scripts/strip_metadata.py release/data/metadata/*.json release/data/splits/*/metadata.json
    python scripts/strip_metadata.py in.json --out out.json
"""
import os
import sys
import json
import argparse

KEEP = ("img_name", "Google_location", "latitude", "longitude", "hf_idx")


def strip(path, out=None):
    rows = json.load(open(path))
    dropped = sorted({k for r in rows for k in r} - set(KEEP))
    slim = [{k: r[k] for k in KEEP if k in r} for r in rows]
    out = out or path
    json.dump(slim, open(out, "w"))
    print(f"{path} -> {out}: {len(slim)} rows, kept {list(KEEP)}, dropped {dropped}, "
          f"{os.path.getsize(out)/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None, help="output path (single input only); default: in place")
    args = ap.parse_args()
    if args.out and len(args.paths) != 1:
        sys.exit("--out needs exactly one input")
    for p in args.paths:
        strip(p, args.out)


if __name__ == "__main__":
    main()
