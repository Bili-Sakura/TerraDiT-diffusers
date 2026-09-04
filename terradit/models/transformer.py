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

The layout follows ``BiliSakura/SiT-diffusers`` (``SiTTransformer2DModel`` in
``transformer_sit.py``): this file is copied into a converted folder as
``transformer/transformer_sit.py`` and must stay free of ``terradit`` imports.

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

from .sit import SiT, SiT_models

# Inlined from ``terradit.families`` so a Hub copy of this file is self-contained.
_FAMILY_CONSTRUCT = {
    "alpha": dict(condition_type="text", geolocation=False, point_prompts=False, omega=False, legacy=False),
    "sigma": dict(condition_type="text", geolocation=True, point_prompts=True, omega=False, legacy=False),
    "omega": dict(condition_type="text", geolocation=True, point_prompts=False, omega=True, legacy=False),
}

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
    if family not in _FAMILY_CONSTRUCT:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_FAMILY_CONSTRUCT)}")
    if arch not in SiT_models:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(SiT_models)}")
    construct = dict(_FAMILY_CONSTRUCT[family])
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
            Native decoded image size in pixels. Other sizes interpolate the
            frozen 2D sin-cos positional embeddings.
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
