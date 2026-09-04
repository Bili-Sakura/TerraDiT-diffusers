"""TerraDiT training entry point.

Thin wrapper around terradit.training.train_terradit. Use --family to pick the model
(alpha / sigma / omega):

    accelerate launch terradit/train.py --family omega --arch SiT-XL/2 \
        --data-root data/git10m --hf-cache-dir data/git10m/hf \
        --dinov3-weights /path/to/dinov3_vitl16_pretrain_sat493m.pth \
        --init-from alpha_xl --exp-name my-run

See scripts/train_*.sh for the 1-GPU and 4-GPU presets.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.training.train_terradit import main

if __name__ == "__main__":
    main()
