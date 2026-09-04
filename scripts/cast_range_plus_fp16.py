"""One-off: cast the RANGE+ training embedding tables to fp16 for distribution.

The source tables are float64 (omega: 2M x 1280 -> 20.6 GB, sigma:
330k x 1280 -> 3.4 GB). fp16 keeps the downstream L2-normalised vectors within ~1e-3 and
cuts the download to ~5.1 GB + 0.85 GB. Needs RAM for the source table (~21 GB for omega).

    python scripts/cast_range_plus_fp16.py \
        --omega-in  range_plus_omega_fp64.npz --sigma-in range_plus_sigma_fp64.npy \
        --splits-in data/git10m/splits \
        --out release/data/range_plus  --splits-out release/data/splits
"""
import os
import argparse

import numpy as np


def cast_npz(src, dst, key="embeddings"):
    z = np.load(src)
    out = {}
    for k in z.files:
        arr = z[k]
        out[k] = arr.astype(np.float16) if (k == key and arr.dtype.kind == "f") else arr
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, **out)
    e = out[key]
    print(f"{src} -> {dst}: {e.shape} {e.dtype} ({os.path.getsize(dst)/1e9:.2f} GB), keys={list(out)}")
    return e


def cast_npy(src, dst):
    a = np.load(src, mmap_mode="r")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=a.shape)
    step = 50_000
    for i in range(0, a.shape[0], step):
        out[i:i + step] = a[i:i + step].astype(np.float16)
    out.flush()
    print(f"{src} -> {dst}: {a.shape} float16 ({os.path.getsize(dst)/1e9:.2f} GB)")


def check_normalised_error(a64, a16, n=1000):
    idx = np.random.default_rng(0).choice(a64.shape[0], size=min(n, a64.shape[0]), replace=False)
    x = np.asarray(a64[idx], dtype=np.float64); y = np.asarray(a16[idx], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True); y /= np.linalg.norm(y, axis=1, keepdims=True)
    print(f"  normalised max|diff| over {len(idx)} rows: {np.abs(x - y).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega-in"); ap.add_argument("--sigma-in")
    ap.add_argument("--splits-in", help="dir with <split>/range_plus.npz (random/spatial/dense)")
    ap.add_argument("--out", default="release/data/range_plus")
    ap.add_argument("--splits-out", default="release/data/splits")
    args = ap.parse_args()

    if args.sigma_in:
        dst = os.path.join(args.out, "sigma.npy")
        cast_npy(args.sigma_in, dst)
        check_normalised_error(np.load(args.sigma_in, mmap_mode="r"), np.load(dst, mmap_mode="r"))
    if args.omega_in:
        dst = os.path.join(args.out, "omega.npz")
        e16 = cast_npz(args.omega_in, dst)
        check_normalised_error(np.load(args.omega_in)["embeddings"], e16)
    if args.splits_in:
        for split in ("random", "spatial", "dense"):
            src = os.path.join(args.splits_in, split, "range_plus.npz")
            if os.path.exists(src):
                cast_npz(src, os.path.join(args.splits_out, split, "range_plus.npz"))


if __name__ == "__main__":
    main()
