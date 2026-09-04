"""Hugging Face constants and loaders shared by training, evaluation, and the demos.

Everything that touches the Hub goes through here so the dataset snapshot, the model
repo, and the on-disk layout are pinned in exactly one place.

Imagery:  ``lcybuaa/Git-10M`` at a fixed revision. The ``hf_idx`` field in every
          metadata file is a row index into the ``train`` split of *that* snapshot.
Weights:  ``MVRL/TerraDiT`` -> ``<name>/model.safetensors`` + ``<name>/config.json``.
Data:     ``MVRL/TerraDiT-data`` -> mirrors the ``data_root`` layout described in
          ``terradit/data/dataset.py``.
"""
import os
import json

import torch

# --------------------------------------------------------------------------- #
# Pinned repositories
# --------------------------------------------------------------------------- #
GIT10M_REPO = "lcybuaa/Git-10M"
GIT10M_REVISION = "29f192b8d2aa28b5d4d8c8d7f0f608cdc61fb52f"   # 2025-06-28, 10,503,567 rows
MODEL_REPO = "MVRL/TerraDiT"
DATA_REPO = "MVRL/TerraDiT-data"
VAE_ID = "stabilityai/sdxl-vae"

# Diffusers-format Hub repo (one folder per family pipeline).
DIFFUSERS_REPO = "BiliSakura/TerraDiT"
FAMILY_HUB_SUBFOLDER = {
    "alpha": "TerraDiT-Alpha-XL",
    "sigma": "TerraDiT-Sigma-XL",
    "omega": "TerraDiT-Omega-XL",
}
# Release name -> Hub subfolder (omega_base is the SiT-B/2 Ω variant).
VARIANT_HUB_SUBFOLDER = {
    "alpha_xl": "TerraDiT-Alpha-XL",
    "sigma_xl": "TerraDiT-Sigma-XL",
    "omega_xl": "TerraDiT-Omega-XL",
    "omega_base": "TerraDiT-Omega-B",
}

# Released weights: name -> construction spec (also stored in each config.json).
MODELS = {
    "alpha_xl":   dict(family="alpha", arch="SiT-XL/2"),
    "sigma_xl":   dict(family="sigma", arch="SiT-XL/2"),
    "omega_xl":   dict(family="omega", arch="SiT-XL/2"),
    "omega_base": dict(family="omega", arch="SiT-B/2"),
}
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

# keys present in training checkpoints that inference never needs
_DROP_PREFIXES = ("projectors.",)     # REPA alignment heads


# --------------------------------------------------------------------------- #
# Imagery
# --------------------------------------------------------------------------- #
def load_git10m(cache_dir=None, repo_id=GIT10M_REPO, revision=GIT10M_REVISION):
    """The Git-10M ``train`` split at the pinned snapshot (downloads on first use, ~394 GB)."""
    from datasets import load_dataset
    return load_dataset(repo_id, cache_dir=cache_dir, revision=revision)["train"]


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def is_diffusers_pipeline_dir(path):
    """True when ``path`` is a Diffusers folder (has ``model_index.json``)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "model_index.json"))


def checkpoint_dir(name, checkpoints_root="checkpoints"):
    return os.path.join(checkpoints_root, name)


def download_weights(name, checkpoints_root="checkpoints", repo_id=MODEL_REPO, revision=None):
    """Fetch ``<name>/model.safetensors`` + ``config.json`` into ``checkpoints/<name>/``."""
    from huggingface_hub import hf_hub_download
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    out = checkpoint_dir(name, checkpoints_root)
    os.makedirs(out, exist_ok=True)
    for fname in (CONFIG_FILE, WEIGHTS_FILE):
        hf_hub_download(repo_id, f"{name}/{fname}", revision=revision,
                        local_dir=checkpoints_root)
    return os.path.join(out, WEIGHTS_FILE)


def resolve_checkpoint(name_or_path, checkpoints_root="checkpoints", auto_download=True):
    """Turn a release name (``omega_xl``), a directory, or a file path into a weights file."""
    if os.path.isfile(name_or_path):
        return name_or_path
    if os.path.isdir(name_or_path):
        if is_diffusers_pipeline_dir(name_or_path):
            transformer_dir = os.path.join(name_or_path, "transformer")
            for fname in ("diffusion_pytorch_model.safetensors", WEIGHTS_FILE):
                cand = os.path.join(transformer_dir, fname)
                if os.path.isfile(cand):
                    return cand
            raise FileNotFoundError(f"{name_or_path} is a Diffusers folder but has no transformer weights")
        cand = os.path.join(name_or_path, WEIGHTS_FILE)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"{name_or_path} has no {WEIGHTS_FILE}")
    if name_or_path in MODELS:
        cand = os.path.join(checkpoint_dir(name_or_path, checkpoints_root), WEIGHTS_FILE)
        if os.path.isfile(cand):
            return cand
        if auto_download:
            print(f"[hf] {name_or_path} not found locally; downloading from {MODEL_REPO}")
            try:
                return download_weights(name_or_path, checkpoints_root)
            except Exception as e:  # RepositoryNotFoundError, GatedRepoError, offline, ...
                raise FileNotFoundError(
                    f"could not fetch {MODEL_REPO}/{name_or_path} ({type(e).__name__}). "
                    f"If the weights are not published yet, or you are offline, pass a local "
                    f"path instead (e.g. --ckpt release/weights/{name_or_path} or "
                    f"checkpoints/{name_or_path}).") from e
        raise FileNotFoundError(f"{cand} missing; run scripts/download_weights.py --models {name_or_path}")
    raise FileNotFoundError(f"checkpoint {name_or_path!r} is neither a file nor a release name "
                            f"{sorted(MODELS)}")


def _strip_module(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


def _drop_training_only(sd):
    return {k: v for k, v in sd.items() if not k.startswith(_DROP_PREFIXES)}


def load_weights(path, *, drop_training_only=True):
    """Load inference weights from a release ``.safetensors`` or a training ``.pt``.

    Returns ``(state_dict, config)``. ``config`` is the release config dict for
    safetensors files (read from the sidecar ``config.json`` or the file metadata),
    or ``None`` for raw training checkpoints. Training ``.pt`` files yield the EMA
    weights; the ``module.`` DDP prefix and REPA projector heads are stripped.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        sd = load_file(path, device="cpu")
        config = None
        sidecar = os.path.join(os.path.dirname(path), CONFIG_FILE)
        if os.path.isfile(sidecar):
            config = json.load(open(sidecar))
        else:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
            if "config" in meta:
                config = json.loads(meta["config"])
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sd = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
        config = None
    sd = _strip_module(sd)
    if drop_training_only:
        sd = _drop_training_only(sd)
    return sd, config


def training_checkpoint_to_release(ckpt_path, *, name=None, family=None, arch=None, dtype=torch.float16):
    """Convert a training ``.pt`` into ``(state_dict fp16, config)`` for the Hub.

    Construction flags are read from the checkpoint's saved args when present and can
    be overridden. The result is what ``scripts/export_weights.py`` writes out.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    args = vars(args) if args is not None and not isinstance(args, dict) else (args or {})
    sd = _drop_training_only(_strip_module(ckpt["ema"]))
    sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}

    fam = family or args.get("family")
    if fam is None:   # legacy REPA/omega trainers did not record a family
        fam = "omega" if any(k.startswith("instance_encoder") for k in sd) else \
              "sigma" if any(k.startswith("geo_embedder") for k in sd) else "alpha"
    config = {
        "name": name,
        "family": fam,
        "arch": arch or args.get("arch") or args.get("model"),
        "legacy": bool(args.get("legacy", False)),
        "loc_dim": int(args.get("loc_dim", 1280)),
        "omega_attn": args.get("omega_attn", "GALA"),
        "encoder_depth": int(args.get("encoder_depth", 8)),
        "resolution": int(args.get("resolution", 256)),
        "num_classes": int(args.get("num_classes", 1000)),
        "dtype": str(dtype).replace("torch.", ""),
        "training_steps": int(ckpt.get("steps", 0)) if isinstance(ckpt, dict) else None,
        "source_checkpoint": os.path.abspath(ckpt_path),
        "num_parameters": int(sum(v.numel() for v in sd.values())),
    }
    return sd, config


def save_release(sd, config, out_dir):
    """Write ``model.safetensors`` (+ config in metadata) and ``config.json`` to ``out_dir``."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(out_dir, WEIGHTS_FILE), metadata={"config": json.dumps(config)})
    json.dump(config, open(os.path.join(out_dir, CONFIG_FILE), "w"), indent=2)
    return os.path.join(out_dir, WEIGHTS_FILE)
