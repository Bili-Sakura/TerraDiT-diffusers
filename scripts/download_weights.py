"""Download released TerraDiT weights from the Hugging Face Hub.

    python scripts/download_weights.py                      # all four models
    python scripts/download_weights.py --models omega_xl    # one model

Files land in ``checkpoints/<name>/{model.safetensors,config.json}``, which is where
every demo / eval / training script looks by default. The demos also auto-download on
first use, so running this is optional.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import MODELS, MODEL_REPO, download_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(MODELS), choices=sorted(MODELS))
    ap.add_argument("--checkpoints-root", default="checkpoints")
    ap.add_argument("--repo-id", default=MODEL_REPO)
    ap.add_argument("--revision", default=None, help="Hub revision (default: main)")
    args = ap.parse_args()
    for name in args.models:
        path = download_weights(name, args.checkpoints_root, repo_id=args.repo_id,
                                revision=args.revision)
        print(f"[download] {name}: {path} ({os.path.getsize(path)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
