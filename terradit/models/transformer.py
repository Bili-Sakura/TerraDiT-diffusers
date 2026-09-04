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

"""Diffusers ``ModelMixin`` wrapper around the TerraDiT SiT backbone.

The wrapped module keeps the original SiT state-dict key names so released
``model.safetensors`` files load without remapping. Construction flags live in
the Diffusers config and are written by ``save_pretrained``.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging

from terradit.families import FAMILY_CONSTRUCT
from terradit.hf import MODELS, load_weights, resolve_checkpoint
from terradit.models.sit import SiT, SiT_models

logger = logging.get_logger(__name__)

# Attributes copied off the constructed SiT so ``SiT.forward`` can run on this instance.
_SIT_ATTRS = (
    "path_type",
    "in_channels",
    "out_channels",
    "patch_size",
    "num_heads",
    "use_cfg",
    "num_classes",
    "z_dims",
    "encoder_depth",
    "condition_type",
    "geolocation",
    "point_prompts",
    "legacy",
    "use_repa",
    "omega",
    "omega_attn",
)


def _build_sit(
    family: str,
    arch: str,
    *,
    input_size: int,
    num_classes: int,
    encoder_depth: int,
    geolocation_dim: int,
    omega_attn: str,
    legacy: bool,
    use_repa: bool,
    use_cfg: bool,
    fused_attn: bool,
    qk_norm: bool,
) -> SiT:
    if family not in FAMILY_CONSTRUCT:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(FAMILY_CONSTRUCT)}")
    if arch not in SiT_models:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(SiT_models)}")
    construct = dict(FAMILY_CONSTRUCT[family])
    construct["legacy"] = legacy
    return SiT_models[arch](
        input_size=input_size,
        num_classes=num_classes,
        use_cfg=use_cfg,
        z_dims=[0],
        encoder_depth=encoder_depth,
        geolocation_dim=geolocation_dim,
        omega_attn=omega_attn,
        use_repa=use_repa,
        fused_attn=fused_attn,
        qk_norm=qk_norm,
        **construct,
    )


class TerraDiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    Diffusers-compatible TerraDiT transformer (SiT backbone + family conditioning).

    Parameters:
        family (`str`, defaults to `"alpha"`):
            Conditioning family: `"alpha"` (text), `"sigma"` (text + location + points),
            or `"omega"` (text + location + geospatial primitives).
        arch (`str`, defaults to `"SiT-XL/2"`):
            SiT architecture key from [`SiT_models`].
        input_size (`int`, defaults to 32):
            Spatial size of the latent grid (image resolution / VAE scale factor).
        in_channels (`int`, defaults to 4):
            Number of latent channels (SDXL VAE).
        sample_size (`int`, defaults to 32):
            Alias of `input_size` kept for DiT-style configs.
        resolution (`int`, defaults to 256):
            Native decoded image size in pixels.
        num_classes (`int`, defaults to 1000):
            Unused class-table size kept for SiT constructor compatibility.
        encoder_depth (`int`, defaults to 8):
            Block index used as the REPA feature tap during training.
        geolocation_dim (`int`, defaults to 1280):
            RANGE+ location embedding dimension.
        loc_dim (`int`, defaults to 1280):
            Alias of `geolocation_dim` matching the released `config.json` field.
        omega_attn (`str`, defaults to `"GALA"`):
            Omega instance-attention implementation. Only `"GALA"` is released.
        legacy (`bool`, defaults to `False`):
            If `True`, adaLN is driven by the timestep embedding alone (pre-fix weights).
        use_repa (`bool`, defaults to `False`):
            Inference always constructs without REPA projector heads.
        use_cfg (`bool`, defaults to `False`):
            SiT constructor flag; classifier-free guidance is applied in the pipeline.
        fused_attn (`bool`, defaults to `True`):
            Use fused scaled-dot-product attention where available.
        qk_norm (`bool`, defaults to `False`):
            QK-normalization flag forwarded to SiT blocks.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        family: str = "alpha",
        arch: str = "SiT-XL/2",
        input_size: int = 32,
        in_channels: int = 4,
        sample_size: int = 32,
        resolution: int = 256,
        num_classes: int = 1000,
        encoder_depth: int = 8,
        geolocation_dim: int = 1280,
        loc_dim: int = 1280,
        omega_attn: str = "GALA",
        legacy: bool = False,
        use_repa: bool = False,
        use_cfg: bool = False,
        fused_attn: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        geo_dim = geolocation_dim if geolocation_dim is not None else loc_dim
        sit = _build_sit(
            family,
            arch,
            input_size=input_size,
            num_classes=num_classes,
            encoder_depth=encoder_depth,
            geolocation_dim=geo_dim,
            omega_attn=omega_attn,
            legacy=legacy,
            use_repa=use_repa,
            use_cfg=use_cfg,
            fused_attn=fused_attn,
            qk_norm=qk_norm,
        )
        self._adopt_sit(sit)

    def _adopt_sit(self, sit: SiT) -> None:
        """Move SiT children/parameters onto this module so state-dict keys stay native."""
        for name, module in list(sit.named_children()):
            self.add_module(name, module)
        for name, param in list(sit.named_parameters(recurse=False)):
            self.register_parameter(name, param)
        for name, buf in list(sit.named_buffers(recurse=False)):
            persistent = name not in getattr(sit, "_non_persistent_buffers_set", set())
            self.register_buffer(name, buf, persistent=persistent)
        for name in _SIT_ATTRS:
            setattr(self, name, getattr(sit, name))
        # Bound SiT helpers so ``SiT.forward(self, ...)`` can run on this instance.
        self.unpatchify = SiT.unpatchify.__get__(self, type(self))
        if getattr(self, "instance_encoder", None) is None:
            self.instance_encoder = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple:
        r"""
        Predict the flow-matching velocity for noisy latents.

        Parameters:
            hidden_states (`torch.Tensor`):
                Noisy latents of shape `(batch, in_channels, height, width)`.
            timestep (`torch.Tensor`):
                Flow-matching time in `[0, 1]`, shape `(batch,)`.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Caption token embeddings `(batch, seq, dim)`. Alias of `y`.
            y (`torch.Tensor`, *optional*):
                Caption token embeddings (TerraDiT native name).
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`Transformer2DModelOutput`].
            **kwargs:
                Family-specific conditioning (`y_pooled`, `loc_embed`, point prompts,
                omega geometry tensors, ...). See [`SiT.forward`].

        Returns:
            [`Transformer2DModelOutput`] or `tuple`:
                Predicted velocity in `.sample` (and unused REPA features when present).
        """
        cond = y if y is not None else encoder_hidden_states
        if cond is None:
            raise ValueError("`y` or `encoder_hidden_states` is required")
        sample, zs = SiT.forward(self, hidden_states, timestep, cond, **kwargs)
        if not return_dict:
            return (sample, zs)
        return Transformer2DModelOutput(sample=sample)

    @classmethod
    def from_legacy_checkpoint(
        cls,
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
    ) -> "TerraDiTTransformer2DModel":
        r"""
        Build the transformer from a release name, safetensors directory/file, or training `.pt`.

        Parameters:
            name_or_path (`str`):
                Release name (`omega_xl`), a directory with `model.safetensors`, or a file path.
            family / arch:
                Override construction flags when the sidecar config does not provide them.
            checkpoints_root (`str`):
                Local cache directory used when `name_or_path` is a release name.
            legacy, loc_dim, omega_attn, encoder_depth, resolution, num_classes:
                Optional overrides of the release config.
            torch_dtype (`torch.dtype`, *optional*):
                Cast floating weights after load.

        Returns:
            [`TerraDiTTransformer2DModel`]:
                Transformer in eval mode with released EMA weights.
        """
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
        model = cls(
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
