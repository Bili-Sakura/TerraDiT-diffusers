# Copyright 2026 The TerraDiT Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write one self-contained Diffusers folder per TerraDiT family.

Each Hub subfolder has custom Python files (SiT-diffusers style)::

    pipeline.py
    transformer/transformer_terradit_{family}.py
    geolocation_encoder/modeling_geolocation.py   # Σ / Ω only (RANGE+)

plus component configs so VAE / LongCLIP / tokenizer / RANGE+ weights can be dropped in::

    model_index.json
    scheduler/scheduler_config.json
    vae/config.json                          # upload diffusion_pytorch_model.safetensors
    text_encoder/config.json                 # upload model.safetensors
    tokenizer/tokenizer_config.json          # vocab.json + merges.txt filled by convert
    transformer/config.json                  # upload diffusion_pytorch_model.safetensors
    geolocation_encoder/config.json          # Σ / Ω: upload location-encoder safetensors + range_db.npz
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .constants import (
    PIPELINE_CLASS_NAME,
    TOKENIZER_ID,
    VARIANT_HUB_SUBFOLDER,
)

FAMILY_PIPELINE_FILE = {
    "alpha": "pipeline_terradit_alpha.py",
    "sigma": "pipeline_terradit_sigma.py",
    "omega": "pipeline_terradit_omega.py",
}

TRANSFORMER_MODULE_NAME = {
    "alpha": "transformer_terradit_alpha",
    "sigma": "transformer_terradit_sigma",
    "omega": "transformer_terradit_omega",
}

LEGACY_ROOT_PY = (
    "pipeline_common.py",
    "pipeline_conditioning.py",
    "constants.py",
    "hub_export.py",
)
LEGACY_TRANSFORMER_PY = (
    "sit.py",
    "localattn.py",
    "omega.py",
    "transformer_sit.py",
    "transformer.py",
)

VARIANT_FAMILY_ARCH = {
    "alpha_xl": ("alpha", "SiT-XL/2"),
    "sigma_xl": ("sigma", "SiT-XL/2"),
    "omega_xl": ("omega", "SiT-XL/2"),
    "omega_base": ("omega", "SiT-B/2"),
}

TOKENIZER_REMOTE_FILES = ("vocab.json", "merges.txt")


def transformer_module_name(family: str) -> str:
    key = (family or "alpha").lower()
    try:
        return TRANSFORMER_MODULE_NAME[key]
    except KeyError as exc:
        raise ValueError(f"unknown TerraDiT family {family!r}") from exc


def _strip_future(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.startswith("from __future__ import"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _strip_relative_imports(source: str) -> str:
    """Drop ``from .foo import ...`` (including parenthesized multi-line imports)."""
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("from .") or stripped.startswith("import ."):
            depth = stripped.count("(") - stripped.count(")")
            i += 1
            while depth > 0 and i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _inline_modules(paths: list[Path], banner: str) -> str:
    parts = [banner.rstrip(), "", "from __future__ import annotations", ""]
    for path in paths:
        text = _strip_relative_imports(_strip_future(path.read_text(encoding="utf-8")))
        parts.append(f"# === {path.name} ===")
        parts.append(text.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _write_inlined_pipeline(output_path: Path, family: str) -> None:
    src = Path(__file__).resolve().parent
    banner = f'''# Copyright 2026 The TerraDiT Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained TerraDiT-{family} Diffusers pipeline.

This is the only custom pipeline module in the folder. Load with::

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(model_dir, trust_remote_code=True)
"""
'''
    (output_path / "pipeline.py").write_text(
        _inline_modules(
            [
                src / "constants.py",
                src / "pipeline_conditioning.py",
                src / "pipeline_common.py",
                src / FAMILY_PIPELINE_FILE[family],
            ],
            banner,
        ),
        encoding="utf-8",
    )


def _write_inlined_transformer(output_path: Path, family: str) -> None:
    src = Path(__file__).resolve().parent.parent / "models"
    module = transformer_module_name(family)
    banner = f'''# Copyright 2026 The TerraDiT Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained TerraDiT-{family} transformer (SiT + family conditioning).

Diffusers loads this file via model_index.json::

    "transformer": ["{module}", "TerraDiTTransformer2DModel"]
"""
'''
    dest = output_path / "transformer"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{module}.py").write_text(
        _inline_modules(
            [
                src / "localattn.py",
                src / "omega.py",
                src / "sit.py",
                src / "transformer.py",
            ],
            banner,
        ),
        encoding="utf-8",
    )


def _copy_asset_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.name.startswith(".") or path.suffix in {".lock", ".metadata"}:
            continue
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def _ensure_tokenizer_vocab(tokenizer_dir: Path) -> None:
    missing = [name for name in TOKENIZER_REMOTE_FILES if not (tokenizer_dir / name).is_file()]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    for name in missing:
        try:
            path = hf_hub_download(TOKENIZER_ID, name)
            shutil.copy2(path, tokenizer_dir / name)
        except Exception:
            return


def _resolve_location_weights(satclip_ckpt: str | Path | None) -> Path | None:
    satclip_ckpt = satclip_ckpt or os.environ.get("SATCLIP_CKPT") or os.environ.get("TERRADIT_SATCLIP_CKPT")
    if not satclip_ckpt:
        return None
    if str(satclip_ckpt).lower() in {"hub", "download"}:
        from huggingface_hub import hf_hub_download
        from terradit.models.geolocation import LOC_ENCODER_FILENAME, LOC_ENCODER_REPO

        return Path(hf_hub_download(LOC_ENCODER_REPO, LOC_ENCODER_FILENAME))
    path = Path(satclip_ckpt)
    if path.is_file():
        return path
    if path.is_dir():
        from terradit.models.geolocation import GEO_WEIGHT_FILENAMES

        for name in GEO_WEIGHT_FILENAMES:
            candidate = path / name
            if candidate.is_file():
                return candidate
    return None


def write_geolocation_encoder(
    output_path: str | Path,
    family: str,
    *,
    satclip_ckpt: str | Path | None = None,
    range_db: str | Path | None = None,
) -> Path | None:
    """Write RANGE+ code + config for Σ / Ω. Optionally attach location-encoder weights and the DB."""
    family = (family or "alpha").lower()
    if family == "alpha":
        return None
    output_path = Path(output_path)
    geo_dir = output_path / "geolocation_encoder"
    geo_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent.parent / "models" / "geolocation.py"
    (geo_dir / "modeling_geolocation.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assets = Path(__file__).resolve().parent / "hub_assets" / "geolocation_encoder"
    if assets.is_dir():
        _copy_asset_tree(assets, geo_dir)
    weight_path = _resolve_location_weights(satclip_ckpt)
    if weight_path is not None:
        from terradit.models.geolocation import TerraDiTGeolocationModel

        model = TerraDiTGeolocationModel.from_location_weights(weight_path)
        model.save_pretrained(geo_dir)
        (geo_dir / "modeling_geolocation.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    if range_db and Path(range_db).is_file():
        dest = geo_dir / "range_db.npz"
        if Path(range_db).resolve() != dest.resolve():
            shutil.copy2(range_db, dest)
    return geo_dir


def write_component_configs(output_path: str | Path, *, family: str | None = None, arch: str | None = None) -> None:
    """Write VAE / LongCLIP / tokenizer / scheduler / RANGE+ configs (no weight files)."""
    output_path = Path(output_path)
    assets = Path(__file__).resolve().parent / "hub_assets"
    for name in ("vae", "text_encoder", "tokenizer", "scheduler"):
        _copy_asset_tree(assets / name, output_path / name)
    _ensure_tokenizer_vocab(output_path / "tokenizer")

    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    config_path = transformer_dir / "config.json"
    if not config_path.is_file() and family is not None:
        config_path.write_text(
            json.dumps(default_transformer_config(family, arch=arch), indent=2) + "\n",
            encoding="utf-8",
        )
    if family is not None and family != "alpha":
        write_geolocation_encoder(output_path, family)


def default_transformer_config(family: str, arch: str | None = None) -> dict:
    arch = arch or "SiT-XL/2"
    return {
        "_class_name": "TerraDiTTransformer2DModel",
        "family": family,
        "arch": arch,
        "input_size": 32,
        "in_channels": 4,
        "sample_size": 32,
        "resolution": 256,
        "num_classes": 1000,
        "encoder_depth": 8,
        "geolocation_dim": 1280,
        "loc_dim": 1280,
        "omega_attn": "GALA",
        "legacy": False,
        "use_repa": False,
        "use_cfg": False,
        "fused_attn": True,
        "qk_norm": False,
    }


def _cleanup_legacy_py(output_path: Path, family: str) -> None:
    keep_transformer = f"{transformer_module_name(family)}.py"
    for name in LEGACY_ROOT_PY:
        path = output_path / name
        if path.is_file():
            path.unlink()
    transformer_dir = output_path / "transformer"
    if transformer_dir.is_dir():
        for name in LEGACY_TRANSFORMER_PY:
            path = transformer_dir / name
            if path.is_file():
                path.unlink()
        for path in transformer_dir.glob("transformer_terradit_*.py"):
            if path.name != keep_transformer:
                path.unlink()


def write_model_index(output_path: str | Path, family: str) -> None:
    output_path = Path(output_path)
    index_path = output_path / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index["_class_name"] = ["pipeline", PIPELINE_CLASS_NAME[family]]
    index["transformer"] = [transformer_module_name(family), "TerraDiTTransformer2DModel"]
    index["scheduler"] = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    index["vae"] = ["diffusers", "AutoencoderKL"]
    index["text_encoder"] = ["transformers", "CLIPTextModel"]
    index["tokenizer"] = ["transformers", "CLIPTokenizer"]
    geo_dir = output_path / "geolocation_encoder"
    has_geo_weights = False
    if geo_dir.is_dir():
        from terradit.models.geolocation import geolocation_folder_has_weights

        has_geo_weights = geolocation_folder_has_weights(geo_dir)
    if family != "alpha" and has_geo_weights:
        index["geolocation_encoder"] = ["modeling_geolocation", "TerraDiTGeolocationModel"]
    else:
        index.pop("geolocation_encoder", None)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def write_self_contained_repo(
    output_path: str | Path,
    family: str,
    *,
    arch: str | None = None,
    satclip_ckpt: str | Path | None = None,
    range_db: str | Path | None = None,
) -> None:
    """Write the custom Python files plus component configs."""
    family = (family or "alpha").lower()
    if family not in FAMILY_PIPELINE_FILE:
        raise ValueError(f"unknown TerraDiT family {family!r}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_inlined_pipeline(output_path, family)
    _write_inlined_transformer(output_path, family)
    write_component_configs(output_path, family=family, arch=arch)
    if family != "alpha":
        write_geolocation_encoder(output_path, family, satclip_ckpt=satclip_ckpt, range_db=range_db)
    write_model_index(output_path, family)
    _cleanup_legacy_py(output_path, family)


def write_repo_skeletons(root: str | Path, *, repo_layout: bool = True) -> list[str]:
    """Write every Hub subfolder with code + configs and no weight files."""
    root = Path(root)
    written: list[str] = []
    for variant, (family, arch) in VARIANT_FAMILY_ARCH.items():
        out = root / VARIANT_HUB_SUBFOLDER[variant] if repo_layout else root / variant
        write_self_contained_repo(out, family, arch=arch)
        written.append(str(out))
    return written
