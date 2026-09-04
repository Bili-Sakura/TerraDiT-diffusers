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
      model.safetensors          # upload: MVRL/satclip-loc-enc-vit16-l40
      range_db.npz               # upload: mvrl/RANGE-database (range_db_large.npz)

The location tower is the extracted SatCLIP ViT16-L40 encoder (~5 MB safetensors),
not the original full SatCLIP checkpoint (ViT image tower included). RANGE+ still
needs the retrieval database.

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

LOC_ENCODER_REPO = "MVRL/satclip-loc-enc-vit16-l40"
LOC_ENCODER_FILENAME = "model.safetensors"
RANGE_DB_REPO = "mvrl/RANGE-database"
RANGE_DB_FILENAME = "range_db_large.npz"

GEO_WEIGHT_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "diffusion_pytorch_model.bin",
    "pytorch_model.bin",
    "satclip-vit16-l40.ckpt",
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
    """Map SatCLIP / RANGE location keys onto ``SatCLIPLocationEncoder``."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = _strip_location_prefix(str(key))
        if name.startswith("visual.") or name.startswith("logit_scale") or name.startswith("head."):
            continue
        name = name.replace("nnet.net.", "nnet.layers.")
        name = name.replace(".linear.weight", ".weight")
        name = name.replace(".linear.bias", ".bias")
        remapped[name] = value
    return remapped


def _tensor_state(blob: Any) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return {key: value for key, value in blob.items() if torch.is_tensor(value)}
    return {}


def _is_modelmixin_location_state(state: dict[str, Any]) -> bool:
    return any(str(key).startswith("satclip.") for key in state)


def _is_raw_location_state(state: dict[str, Any]) -> bool:
    if _is_modelmixin_location_state(state):
        return False
    return any(
        str(key).startswith(("location.", "model.location.", "nnet.", "posenc."))
        for key in state
    )


def _infer_location_hparams(state: dict[str, torch.Tensor]) -> dict[str, int]:
    remapped = _remap_location_state_dict(state)
    layer_ids: list[int] = []
    hidden = None
    in_dim = None
    out_dim = None
    for key, value in remapped.items():
        if key.startswith("nnet.layers.") and key.endswith(".weight") and value.ndim == 2:
            try:
                layer_ids.append(int(key.split(".")[2]))
            except (IndexError, ValueError):
                pass
            if key == "nnet.layers.0.weight":
                hidden, in_dim = int(value.shape[0]), int(value.shape[1])
        elif key == "nnet.last_layer.weight" and value.ndim == 2:
            out_dim = int(value.shape[0])
            if hidden is None:
                hidden = int(value.shape[1])
    legendre = int(round(in_dim ** 0.5)) if in_dim else 40
    return {
        "satclip_dim": out_dim or 256,
        "nnet_hidden": hidden or 512,
        "nnet_layers": (max(layer_ids) + 1) if layer_ids else 2,
        "legendre_polys": legendre,
    }


def _load_weight_blob(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return {"state_dict": load_file(str(path), device="cpu"), "hyper_parameters": {}}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "hyper_parameters" in ckpt):
        state = ckpt.get("state_dict") or _tensor_state(ckpt)
        return {"state_dict": state, "hyper_parameters": dict(ckpt.get("hyper_parameters") or {})}
    if isinstance(ckpt, dict):
        return {"state_dict": _tensor_state(ckpt), "hyper_parameters": {}}
    raise ValueError(f"Unrecognized location-encoder weights at {path}")


def _config_weight_names(folder: str | Path) -> list[str]:
    names = list(GEO_WEIGHT_FILENAMES)
    config_path = Path(folder) / "config.json"
    if config_path.is_file():
        try:
            extra = json.loads(config_path.read_text(encoding="utf-8")).get("satclip_filename")
            if extra:
                names.append(str(extra))
        except Exception:
            pass
    return names


def _find_weight_file(folder: str | Path, names: list[str] | None = None) -> Path | None:
    folder = Path(folder)
    for name in names or _config_weight_names(folder):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def geolocation_folder_has_weights(folder: str | Path) -> bool:
    folder = Path(folder)
    if not folder.is_dir():
        return False
    return _find_weight_file(folder) is not None


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
        nnet_hidden: int = 512,
        nnet_layers: int = 2,
        beta: float = 0.5,
        semantic_temp: float = 12.0,
        geo_temp: float = 40.0,
        database_filename: str = "range_db.npz",
        satclip_filename: str = LOC_ENCODER_FILENAME,
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
    def from_location_weights(
        cls,
        checkpoint: str | Path,
        *,
        database: str | Path | None = None,
        beta: float = 0.5,
        torch_dtype: torch.dtype | None = None,
        **config_overrides: Any,
    ) -> "TerraDiTGeolocationModel":
        """Load the location encoder from safetensors or a legacy Lightning ckpt."""
        blob = _load_weight_blob(checkpoint)
        state = blob["state_dict"]
        hparams = dict(blob.get("hyper_parameters") or {})
        inferred = _infer_location_hparams(state)
        satclip_dim = int(config_overrides.get("satclip_dim", hparams.get("embed_dim", inferred["satclip_dim"])))
        image_dim = int(config_overrides.get("image_dim", 1024))
        model = cls(
            embedding_dim=int(config_overrides.get("embedding_dim", satclip_dim + image_dim)),
            satclip_dim=satclip_dim,
            image_dim=image_dim,
            legendre_polys=int(
                config_overrides.get("legendre_polys", hparams.get("legendre_polys", inferred["legendre_polys"]))
            ),
            nnet_hidden=int(config_overrides.get("nnet_hidden", hparams.get("capacity", inferred["nnet_hidden"]))),
            nnet_layers=int(
                config_overrides.get("nnet_layers", hparams.get("num_hidden_layers", inferred["nnet_layers"]))
            ),
            beta=float(config_overrides.get("beta", beta)),
            satclip_filename=Path(checkpoint).name,
        )
        if state:
            model._load_satclip_weights(state)
        if database is not None:
            model.attach_database(database)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model.eval()
        return model

    @classmethod
    def from_satclip_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        database: str | Path | None = None,
        beta: float = 0.5,
        torch_dtype: torch.dtype | None = None,
    ) -> "TerraDiTGeolocationModel":
        """Backward-compatible alias of :meth:`from_location_weights`."""
        return cls.from_location_weights(
            checkpoint, database=database, beta=beta, torch_dtype=torch_dtype
        )

    @classmethod
    def from_hub(
        cls,
        *,
        database: bool | str | Path = True,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        beta: float = 0.5,
        torch_dtype: torch.dtype | None = None,
        revision: str | None = None,
    ) -> "TerraDiTGeolocationModel":
        """Download the ViT16-L40 location encoder (and optionally the RANGE database)."""
        from huggingface_hub import hf_hub_download

        download_kw: dict[str, Any] = {}
        if cache_dir is not None:
            download_kw["cache_dir"] = str(cache_dir)
        if local_dir is not None:
            download_kw["local_dir"] = str(local_dir)
        if revision is not None:
            download_kw["revision"] = revision
        weights = hf_hub_download(LOC_ENCODER_REPO, LOC_ENCODER_FILENAME, **download_kw)
        db_path: str | Path | None = None
        if database is True:
            db_kw = dict(download_kw)
            db_path = hf_hub_download(RANGE_DB_REPO, RANGE_DB_FILENAME, repo_type="dataset", **db_kw)
        elif database:
            db_path = database
        return cls.from_location_weights(weights, database=db_path, beta=beta, torch_dtype=torch_dtype)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | None = None, **kwargs):
        folder = Path(pretrained_model_name_or_path) if pretrained_model_name_or_path else None
        name = str(pretrained_model_name_or_path or "")
        if name in {LOC_ENCODER_REPO, f"{LOC_ENCODER_REPO}/{LOC_ENCODER_FILENAME}"}:
            database = kwargs.pop("database", True)
            return cls.from_hub(
                database=database,
                cache_dir=kwargs.pop("cache_dir", None),
                local_dir=kwargs.pop("local_dir", None),
                beta=float(kwargs.pop("beta", 0.5)),
                torch_dtype=kwargs.get("torch_dtype"),
                revision=kwargs.pop("revision", None),
            )
        if folder is not None and folder.is_dir():
            config_guess: dict[str, Any] = {}
            config_path = folder / "config.json"
            if config_path.is_file():
                config_guess = json.loads(config_path.read_text(encoding="utf-8"))
            ckpt_name = kwargs.pop("satclip_filename", None) or config_guess.get(
                "satclip_filename", LOC_ENCODER_FILENAME
            )
            names = [str(ckpt_name), *GEO_WEIGHT_FILENAMES]
            weight_path = _find_weight_file(folder, names)
            if weight_path is not None:
                blob = _load_weight_blob(weight_path)
                state = blob["state_dict"]
                if _is_raw_location_state(state) or weight_path.suffix in {".ckpt", ".pt"}:
                    overrides = {
                        key: config_guess[key]
                        for key in (
                            "embedding_dim",
                            "satclip_dim",
                            "image_dim",
                            "legendre_polys",
                            "nnet_hidden",
                            "nnet_layers",
                        )
                        if key in config_guess
                    }
                    # Prefer shapes from the weight file over a stale skeleton config.
                    overrides.update(_infer_location_hparams(state))
                    satclip_dim = int(overrides["satclip_dim"])
                    image_dim = int(overrides.get("image_dim", 1024))
                    overrides["satclip_dim"] = satclip_dim
                    overrides["image_dim"] = image_dim
                    if int(overrides.get("embedding_dim", satclip_dim + image_dim)) != satclip_dim + image_dim:
                        overrides["embedding_dim"] = satclip_dim + image_dim
                    model = cls.from_location_weights(
                        weight_path,
                        database=None,
                        beta=float(config_guess.get("beta", kwargs.get("beta", 0.5))),
                        torch_dtype=kwargs.get("torch_dtype"),
                        **overrides,
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
