"""Augment alpha/sigma metadata with an `hf_idx` field so images load from the
Hugging Face dataset (lcybuaa/Git-10M), mirroring omega.

Two modes:
  reference  -- join img_name against a metadata that already has (img_name, hf_idx),
                e.g. omega's metadata_with_hf_idx.json. Fast; covers overlapping tiles.
  scan       -- iterate the HF dataset's img_name column to build the full map.
                Robust (covers everything) but loads the whole column.

Rows whose img_name is not found are left without hf_idx (they fall back to disk).

The HF dataset is always the pinned Git-10M snapshot (see terradit/hf.py), so hf_idx
values are stable across machines.

Examples:
  python -m terradit.data.preprocessing.add_hf_idx --mode reference \
      --in-metadata  my_metadata.json \
      --reference    data/git10m/metadata/omega.json \
      --out-metadata data/git10m/metadata/sigma.json

  python -m terradit.data.preprocessing.add_hf_idx --mode scan \
      --in-metadata  my_metadata.json --hf-cache-dir data/git10m/hf \
      --out-metadata data/git10m/metadata/alpha.json
"""
import os
import json
import argparse


def build_map_reference(reference_path):
    ref = json.load(open(reference_path, "r"))
    return {r["img_name"]: int(r["hf_idx"]) for r in ref if "hf_idx" in r and "img_name" in r}


def build_map_scan(repo_id, cache_dir):
    from terradit.hf import load_git10m
    ds = load_git10m(cache_dir, repo_id=repo_id)
    names = ds["img_name"]  # whole column
    return {name: i for i, name in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "scan"], required=True)
    ap.add_argument("--in-metadata", required=True)
    ap.add_argument("--out-metadata", required=True)
    ap.add_argument("--reference", help="metadata json with (img_name, hf_idx) [reference mode]")
    from terradit.hf import GIT10M_REPO
    ap.add_argument("--hf-repo-id", default=GIT10M_REPO)
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap input rows (for quick tests)")
    args = ap.parse_args()

    if args.mode == "reference":
        assert args.reference, "--reference required in reference mode"
        name2hf = build_map_reference(args.reference)
    else:
        name2hf = build_map_scan(args.hf_repo_id, args.hf_cache_dir)
    print(f"hf_idx map: {len(name2hf)} entries")

    meta = json.load(open(args.in_metadata, "r"))
    if args.limit:
        meta = meta[: args.limit]
    matched = 0
    for r in meta:
        hi = name2hf.get(r.get("img_name"))
        if hi is not None:
            r["hf_idx"] = hi
            matched += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out_metadata)), exist_ok=True)
    json.dump(meta, open(args.out_metadata, "w"))
    print(f"wrote {args.out_metadata}: {matched}/{len(meta)} rows matched "
          f"({100*matched/max(len(meta),1):.1f}% have hf_idx)")


if __name__ == "__main__":
    main()
