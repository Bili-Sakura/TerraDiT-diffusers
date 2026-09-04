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

"""RANGE+ lat/lon encoder used by TerraDiT-Σ and TerraDiT-Ω.

This file is copied to converted Hub folders as
``geolocation_encoder/modeling_geolocation.py``. It must not import ``terradit``.

Hub folder (sigma / omega)::

    geolocation_encoder/
      config.json
      modeling_geolocation.py
      satclip-vit16-l40.ckpt     # upload: microsoft/SatCLIP-ViT16-L40
      range_db.npz               # upload: mvrl/RANGE-database (range_db_large.npz)

``forward`` takes ``(lon, lat)`` in degrees, shape ``(batch, 2)``, and returns a
1280-d RANGE+ embedding (1024-d retrieved image features + 256-d SatCLIP).
Without a retrieval database the encoder returns zeros so generation still runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

GEO_WEIGHT_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
)


def _associated_legendre_polynomial(degree: int, order: int, x: torch.Tensor) -> torch.Tensor:
    pmm = torch.ones_like(x)
    if order > 0:
        somx2 = torch.sqrt(((1 - x) * (1 + x)).clamp(min=0))
        fact = 1.0
        for _ in range(1, order + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if degree == order:
        return pmm
    pmmp1 = x * (2.0 * order + 1.0) * pmm
    if degree == order + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ell in range(order + 2, degree + 1):
        pll = ((2.0 * ell - 1.0) * x * pmmp1 - (ell + order - 1.0) * pmm) / (ell - order)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _sh_renormalization(degree: int, order: int) -> float:
    return math.sqrt(
        (2.0 * degree + 1.0)
        * math.exp(math.lgamma(degree - order + 1) - math.lgamma(degree + order + 1))
        / (4.0 * math.pi)
    )


def _spherical_harmonic(order: int, degree: int, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.cos(theta)
    if order == 0:
        return _sh_renormalization(degree, 0) * _associated_legendre_polynomial(degree, 0, cos_theta)
    if order > 0:
        return (
            math.sqrt(2.0)
            * _sh_renormalization(degree, order)
            * torch.cos(order * phi)
            * _associated_legendre_polynomial(degree, order, cos_theta)
        )
    return (
        math.sqrt(2.0)
        * _sh_renormalization(degree, -order)
        * torch.sin(-order * phi)
        * _associated_legendre_polynomial(degree, -order, cos_theta)
    )


class SphericalHarmonics(nn.Module):
    """SatCLIP spherical-harmonic positional encoding (expects lon/lat in degrees)."""

    def __init__(self, legendre_polys: int = 40):
        super().__init__()
        self.L = int(legendre_polys)
        self.embedding_dim = self.L * self.L

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        # Official SatCLIP applies deg2rad(lon+180) / deg2rad(lat+90) here even when
        # callers already converted to radians. Keep that convention so extracted
        # location-encoder weights stay aligned with the RANGE+ database.
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        terms = []
        for degree in range(self.L):
            for order in range(-degree, degree + 1):
                terms.append(_spherical_harmonic(order, degree, phi, theta))
        return torch.stack(terms, dim=-1)


class _Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class _Siren(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, w0: float = 1.0, is_first: bool = False, dropout: bool = False):
        super().__init__()
        self.dropout = dropout
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        w_std = (1 / dim_in) if is_first else (math.sqrt(6 / dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        bias.uniform_(-w_std, w_std)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation = _Sine(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    """SatCLIP / RANGE SirenNet (``nnet.layers.*`` / ``nnet.last_layer.*`` keys)."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, num_layers: int, w0: float = 1.0, w0_initial: float = 30.0):
        super().__init__()
        layers = []
        for index in range(num_layers):
            layers.append(
                _Siren(
                    dim_in if index == 0 else dim_hidden,
                    dim_hidden,
                    w0=w0_initial if index == 0 else w0,
                    is_first=index == 0,
                    dropout=True,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.last_layer = _Siren(dim_hidden, dim_out, w0=w0, dropout=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class SatCLIPLocationEncoder(nn.Module):
    def __init__(self, legendre_polys: int, hidden_size: int, out_dim: int, num_layers: int):
        super().__init__()
        self.posenc = SphericalHarmonics(legendre_polys)
        self.nnet = SirenNet(self.posenc.embedding_dim, hidden_size, out_dim, num_layers)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        encoded = self.posenc(lonlat.double()).to(dtype=lonlat.dtype)
        return self.nnet(encoded)


def _lonlat_to_xyz(lonlat: torch.Tensor) -> torch.Tensor:
    lon = torch.deg2rad(lonlat[:, 0])
    lat = torch.deg2rad(lonlat[:, 1])
    return torch.stack(
        [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)],
        dim=-1,
    )


def _as_1d(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().float().reshape(-1)
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor([float(v) for v in value], dtype=torch.float32)
    else:
        tensor = torch.tensor([float(value)], dtype=torch.float32)
    return tensor


def _coords_from_latlon(coords, latitudes, longitudes) -> torch.Tensor:
    if coords is not None:
        tensor = torch.as_tensor(coords, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != 2:
            raise ValueError(f"`coords` must be (batch, 2) lon/lat degrees, got {tuple(tensor.shape)}")
        return tensor
    if latitudes is None or longitudes is None:
        raise ValueError("Provide `coords` or both `latitudes` and `longitudes`.")
    lat = _as_1d(latitudes)
    lon = _as_1d(longitudes)
    if lat.numel() != lon.numel():
        raise ValueError(f"latitudes ({lat.numel()}) and longitudes ({lon.numel()}) must have the same length")
    return torch.stack([lon, lat], dim=-1)


def _strip_location_prefix(key: str) -> str:
    for prefix in ("model.location.", "location.", "satclip."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _remap_location_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map official SatCLIP location keys onto ``SatCLIPLocationEncoder``."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = _strip_location_prefix(str(key))
        if name.startswith("visual.") or name.startswith("logit_scale") or name.startswith("head."):
            continue
        name = name.replace("nnet.net.", "nnet.layers.")
        name = name.replace(".linear.weight", ".weight")
        name = name.replace(".linear.bias", ".bias")
        remapped[name] = value
    return remapped


def geolocation_folder_has_weights(folder: str | Path) -> bool:
    folder = Path(folder)
    if not folder.is_dir():
        return False
    names = list(GEO_WEIGHT_FILENAMES) + ["satclip-vit16-l40.ckpt"]
    config_path = folder / "config.json"
    if config_path.is_file():
        try:
            extra = json.loads(config_path.read_text(encoding="utf-8")).get("satclip_filename")
            if extra:
                names.append(str(extra))
        except Exception:
            pass
    return any((folder / name).is_file() for name in names)


class TerraDiTGeolocationModel(ModelMixin, ConfigMixin):
    """RANGE+ geolocation encoder (SatCLIP query + retrieval database)."""

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 1280,
        satclip_dim: int = 256,
        image_dim: int = 1024,
        legendre_polys: int = 40,
        nnet_hidden: int = 256,
        nnet_layers: int = 2,
        beta: float = 0.5,
        semantic_temp: float = 12.0,
        geo_temp: float = 40.0,
        database_filename: str = "range_db.npz",
        satclip_filename: str = "satclip-vit16-l40.ckpt",
    ) -> None:
        super().__init__()
        if embedding_dim != satclip_dim + image_dim:
            raise ValueError(
                f"embedding_dim must be satclip_dim + image_dim ({satclip_dim + image_dim}), got {embedding_dim}"
            )
        self.satclip = SatCLIPLocationEncoder(legendre_polys, nnet_hidden, satclip_dim, nnet_layers)
        self.satclip.eval()
        for param in self.satclip.parameters():
            param.requires_grad = False
        self.beta = float(beta)
        self.semantic_temp = float(semantic_temp)
        self.geo_temp = float(geo_temp)
        self.register_buffer("db_satclip", torch.zeros(1, satclip_dim), persistent=False)
        self.register_buffer("db_image", torch.zeros(1, image_dim), persistent=False)
        self.register_buffer("db_xyz", torch.zeros(1, 3), persistent=False)
        self._has_database = False

    def attach_database(self, path: str | Path) -> None:
        archive = np.load(path)
        sat = archive["satclip_embeddings"].astype(np.float32)
        sat = sat / np.linalg.norm(sat, axis=1, keepdims=True).clip(min=1e-8)
        image = archive["image_embeddings"].astype(np.float32)
        locs = archive["locs"].astype(np.float32)
        xyz = _lonlat_to_xyz(torch.from_numpy(locs)).numpy()
        self.db_satclip = torch.from_numpy(sat)
        self.db_image = torch.from_numpy(image)
        self.db_xyz = torch.from_numpy(xyz.astype(np.float32))
        self._has_database = True

    def _maybe_attach_database_from_dir(self, folder: str | Path | None) -> None:
        if folder is None or self._has_database:
            return
        folder = Path(folder)
        for name in (self.config.database_filename, "range_db.npz", "range_db_large.npz"):
            candidate = folder / name
            if candidate.is_file():
                self.attach_database(candidate)
                return

    def _load_satclip_weights(self, state: dict[str, torch.Tensor]) -> None:
        remapped = _remap_location_state_dict(state)
        self.satclip.load_state_dict(remapped, strict=False)

    @classmethod
    def from_satclip_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        database: str | Path | None = None,
        beta: float = 0.5,
        torch_dtype: torch.dtype | None = None,
    ) -> "TerraDiTGeolocationModel":
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = dict(ckpt.get("hyper_parameters") or {})
        model = cls(
            embedding_dim=1280,
            satclip_dim=int(hparams.get("embed_dim", 256)),
            image_dim=1024,
            legendre_polys=int(hparams.get("legendre_polys", 40)),
            nnet_hidden=int(hparams.get("capacity", 256)),
            nnet_layers=int(hparams.get("num_hidden_layers", 2)),
            beta=beta,
        )
        state = ckpt.get("state_dict") or ckpt
        if isinstance(state, dict):
            model._load_satclip_weights(state)
        if database is not None:
            model.attach_database(database)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model.eval()
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | None = None, **kwargs):
        folder = Path(pretrained_model_name_or_path) if pretrained_model_name_or_path else None
        if folder is not None and folder.is_dir():
            config_guess: dict[str, Any] = {}
            config_path = folder / "config.json"
            if config_path.is_file():
                config_guess = json.loads(config_path.read_text(encoding="utf-8"))
            ckpt_name = kwargs.pop("satclip_filename", None) or config_guess.get(
                "satclip_filename", "satclip-vit16-l40.ckpt"
            )
            ckpt_path = folder / str(ckpt_name)
            has_weights = any((folder / name).is_file() for name in GEO_WEIGHT_FILENAMES)
            if ckpt_path.is_file() and not has_weights:
                model = cls.from_satclip_checkpoint(
                    ckpt_path,
                    database=None,
                    beta=float(config_guess.get("beta", 0.5)),
                    torch_dtype=kwargs.get("torch_dtype"),
                )
                model._maybe_attach_database_from_dir(folder)
                return model
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model._maybe_attach_database_from_dir(pretrained_model_name_or_path)
        return model

    @torch.no_grad()
    def forward(
        self,
        coords: torch.Tensor | None = None,
        latitudes: Any = None,
        longitudes: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode ``(lon, lat)`` degrees to a RANGE+ embedding of shape ``(batch, 1280)``."""
        latitudes = kwargs.pop("lat", latitudes)
        longitudes = kwargs.pop("lon", longitudes)
        coords = _coords_from_latlon(coords, latitudes, longitudes)
        device = self.satclip.nnet.last_layer.weight.device
        coords = coords.to(device=device)
        if not self._has_database:
            return torch.zeros(coords.shape[0], self.config.embedding_dim, device=device, dtype=torch.float32)

        sat = F.normalize(self.satclip(coords).float(), dim=-1)
        db_sat = self.db_satclip.to(device=device, dtype=sat.dtype)
        db_image = self.db_image.to(device=device, dtype=sat.dtype)
        db_xyz = self.db_xyz.to(device=device, dtype=sat.dtype)
        semantic = torch.softmax(sat @ db_sat.T * self.semantic_temp, dim=-1)
        high_res = semantic @ db_image
        xyz = _lonlat_to_xyz(coords.float())
        angular = torch.softmax(xyz @ db_xyz.T * self.geo_temp, dim=-1)
        angular_high = angular @ db_image
        mixed = (1.0 - self.beta) * angular_high + self.beta * high_res
        return torch.cat([mixed, sat], dim=-1)
