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

"""Load a released / training checkpoint into ``TerraDiTTransformer2DModel``.

Kept out of ``transformer.py`` so converted Hub folders can ship a copy of that
file without ``terradit.hf`` imports.
"""

from __future__ import annotations

import torch
from diffusers.utils import logging

from terradit.hf import MODELS, load_weights, resolve_checkpoint
from terradit.models.transformer import TerraDiTTransformer2DModel

logger = logging.get_logger(__name__)


def load_legacy_transformer(
    name_or_path: str,
    *,
    family: str | None = None,
    arch: str | None = None,
    checkpoints_root: str = "checkpoints",
    legacy: bool | None = None,
    loc_dim: int | None = None,
    omega_attn: str | None = None,
    encoder_depth: int | None = None,
    resolution: int | None = None,
    num_classes: int | None = None,
    torch_dtype: torch.dtype | None = None,
) -> TerraDiTTransformer2DModel:
    """Build the transformer from a release name, safetensors directory/file, or training ``.pt``."""
    path = resolve_checkpoint(name_or_path, checkpoints_root)
    sd, config = load_weights(path)
    config = dict(config or {})
    if name_or_path in MODELS:
        config = {**MODELS[name_or_path], **config}

    family = family or config.get("family")
    arch = arch or config.get("arch")
    if family is None or arch is None:
        raise ValueError(f"family/arch not given and not found in a config next to {path}")

    resolution = resolution if resolution is not None else int(config.get("resolution", 256))
    input_size = resolution // 8
    geo_dim = loc_dim if loc_dim is not None else int(config.get("loc_dim", config.get("geolocation_dim", 1280)))
    model = TerraDiTTransformer2DModel(
        family=family,
        arch=arch,
        input_size=input_size,
        sample_size=input_size,
        resolution=resolution,
        num_classes=num_classes if num_classes is not None else int(config.get("num_classes", 1000)),
        encoder_depth=encoder_depth if encoder_depth is not None else int(config.get("encoder_depth", 8)),
        geolocation_dim=geo_dim,
        loc_dim=geo_dim,
        omega_attn=omega_attn or config.get("omega_attn", "GALA"),
        legacy=config.get("legacy", False) if legacy is None else legacy,
        use_repa=False,
        use_cfg=False,
    )
    sd = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("projectors")]
    logger.info(
        "Loaded %s (%s, %s, legacy=%s): missing=%s unexpected=%s",
        path,
        family,
        arch,
        model.config.legacy,
        len(missing),
        len(result.unexpected_keys),
    )
    if missing:
        logger.warning("First missing keys: %s", missing[:8])
    if result.unexpected_keys:
        logger.warning("First unexpected keys: %s", result.unexpected_keys[:8])
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model.eval()
    return model
