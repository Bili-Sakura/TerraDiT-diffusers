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

"""Inference-only conditioning helpers for converted TerraDiT folders.

Copied next to ``pipeline.py``. Do not import ``terradit``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .constants import CAPTION_MAX_LEN, LOC_DIM, TAG_MAX_LEN

MAX_INSTANCES = 64
MAX_POINTS = 64
SIGMA_MAX_POINTS = 50
OMEGA_CONDITION_DROPOUTS = {"omega": None, "box": [1.0, 1.0, 0.0, 0.0], "point": [1.0, 1.0, 1.0, 0.0]}


def _default_range_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("TERRADIT_RANGE_DIR")
    candidates = [env, os.path.join(os.path.dirname(here), "RANGE"), os.path.join(here, "RANGE")]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "range")):
            return candidate
    return env or os.path.join(os.path.dirname(here), "RANGE")


def load_range_model(device, *, beta=0.5, range_dir=None):
    """Lazily load RANGE+ for live lat/lon -> embedding. Returns None if unavailable."""
    range_dir = range_dir or _default_range_dir()
    if not os.path.isdir(os.path.join(range_dir, "range")):
        print(
            f"[range] RANGE submodule not found at {range_dir}.\n"
            "        Fetch it with:  git submodule update --init --recursive\n"
            "        (or set TERRADIT_RANGE_DIR). Geolocation falls back to zeros."
        )
        return None
    try:
        if range_dir not in sys.path:
            sys.path.insert(0, range_dir)
        from range.load_model import load_model  # noqa: E402
        from huggingface_hub import hf_hub_download

        cache = os.path.join(range_dir, "pretrained")
        ckpt = hf_hub_download(
            "microsoft/SatCLIP-ViT16-L40",
            "satclip-vit16-l40.ckpt",
            repo_type="model",
            local_dir=cache,
        )
        db = hf_hub_download(
            "mvrl/RANGE-database",
            "range_db_large.npz",
            repo_type="dataset",
            local_dir=cache,
        )
        return load_model(
            model_name="RANGE+",
            pretrained_path=ckpt,
            device=device,
            db_path=db,
            beta=beta,
        )
    except Exception as exc:
        print(
            f"[range] RANGE+ unavailable ({type(exc).__name__}: {exc}); "
            "geolocation will fall back to borrow/zeros."
        )
        return None


def loc_embed_from_latlon(range_model, lat, lon, device):
    """lat/lon -> normalized ``[1, LOC_DIM]`` RANGE+ embedding; zeros if no coords/model."""
    if range_model is not None and lat is not None and lon is not None:
        coords = torch.tensor([[float(lon), float(lat)]], dtype=torch.float64, device=device)
        loc = range_model(coords)
        loc = torch.from_numpy(loc).to(device) if isinstance(loc, np.ndarray) else loc.to(device)
        return torch.nn.functional.normalize(loc, dim=-1).float()
    return torch.zeros((1, LOC_DIM), dtype=torch.float32, device=device)


@torch.no_grad()
def caption_embeds(clip, tokenizer, prompts, device):
    """Encode caption text(s) -> (y [B,T,D], y_pooled [B,D], attn_mask [B,T])."""
    tok = tokenizer(
        prompts,
        padding="max_length",
        max_length=CAPTION_MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )
    out = clip(tok.input_ids.to(device), tok.attention_mask.to(device))
    y = torch.nn.functional.normalize(out.last_hidden_state, dim=-1)
    y_pooled = torch.nn.functional.normalize(out.pooler_output, dim=-1)
    return y, y_pooled, tok.attention_mask.to(device)


@torch.no_grad()
def tag_pooled_embeds(clip, ids, attn):
    """Pooled, normalized CLIP embeddings for tag tokens. ids/attn: [M, TAG_MAX_LEN]."""
    out = clip(ids, attn).pooler_output
    return torch.nn.functional.normalize(out, dim=-1)


def pack_sigma_points(points, device, *, max_points=SIGMA_MAX_POINTS):
    """[(x, y, tag), ...] -> (coords [P,2] as (row,col), mask [P], tags [P])."""
    coords = torch.zeros(max_points, 2, dtype=torch.float32, device=device)
    mask = torch.ones(max_points, dtype=torch.long, device=device)
    tags = [""] * max_points
    n = min(len(points), max_points)
    for i in range(n):
        x, y, tag = points[i]
        coords[i] = torch.tensor([float(y), float(x)], device=device)
        mask[i] = 0
        tags[i] = tag or ""
    return coords, mask, tags


def pack_omega_instances(instances, device):
    """[{type, coords, tag}, ...] -> omega geometry tensors + per-instance tags."""
    polygon_xy = torch.zeros(MAX_INSTANCES, MAX_POINTS, 2, device=device)
    polygon_xy_mask = torch.zeros(MAX_INSTANCES, MAX_POINTS, dtype=torch.bool, device=device)
    polyline_xy = torch.zeros(MAX_INSTANCES, MAX_POINTS, 2, device=device)
    polyline_xy_mask = torch.zeros(MAX_INSTANCES, MAX_POINTS, dtype=torch.bool, device=device)
    bbox_xyxy = torch.zeros(MAX_INSTANCES, 4, device=device)
    point_xy = torch.zeros(MAX_INSTANCES, 2, device=device)
    format_mask = torch.zeros(MAX_INSTANCES, 4, device=device)
    instance_mask = torch.zeros(MAX_INSTANCES, device=device)

    n = min(len(instances), MAX_INSTANCES)
    tags = [""] * MAX_INSTANCES
    for j, inst in enumerate(instances[:n]):
        itype = inst.get("type", "point")
        tags[j] = inst.get("tag", inst.get("prompt", "")) or ""
        raw = inst.get("coords", [])
        if raw and not isinstance(raw[0], (list, tuple)):
            raw = [raw]
        coords = [(float(x), float(y)) for (x, y) in raw]
        length = min(len(coords), MAX_POINTS)

        if itype == "polygon" and length >= 3:
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
                length = min(len(coords), MAX_POINTS)
            c = torch.tensor(coords[:length], device=device)
            polygon_xy[j, :length] = c
            polygon_xy_mask[j, :length] = True
            xs, ys = c[:, 0], c[:, 1]
            bbox_xyxy[j] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device)
            point_xy[j] = torch.stack([xs.mean(), ys.mean()])
            format_mask[j, 0] = format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif itype == "polyline" and length >= 2:
            c = torch.tensor(coords[:length], device=device)
            polyline_xy[j, :length] = c
            polyline_xy_mask[j, :length] = True
            xs, ys = c[:, 0], c[:, 1]
            bbox_xyxy[j] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device)
            point_xy[j] = c[length // 2]
            format_mask[j, 1] = format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif itype == "bbox" and length >= 2:
            (x0, y0), (x1, y1) = coords[0], coords[1]
            bbox_xyxy[j] = torch.tensor([x0, y0, x1, y1], device=device)
            point_xy[j] = torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1)], device=device)
            format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif length >= 1:
            point_xy[j] = torch.tensor(coords[0], device=device)
            format_mask[j, 3] = 1.0
        instance_mask[j] = 1.0

    return dict(
        polygon_xy=polygon_xy,
        polygon_xy_mask=polygon_xy_mask,
        polyline_xy=polyline_xy,
        polyline_xy_mask=polyline_xy_mask,
        bbox_xyxy=bbox_xyxy,
        point_xy=point_xy,
        format_mask=format_mask,
        instance_mask=instance_mask,
        tags=tags,
    )


def _resolve_location(spec, range_model, device):
    if spec.get("loc_embed") is not None:
        loc = spec["loc_embed"]
        loc = torch.as_tensor(loc, device=device).float()
        return loc.view(1, -1)
    return loc_embed_from_latlon(range_model, spec.get("lat"), spec.get("lon"), device)


def build_conditioning(
    family,
    spec,
    clip,
    tokenizer,
    device,
    *,
    range_model=None,
    num_images=1,
    dropout_probs=None,
):
    """Build transformer conditioning kwargs (batched to ``num_images``) for a family."""
    batch = num_images
    y, y_pooled, _ = caption_embeds(clip, tokenizer, [spec.get("caption", "")], device)
    cond = dict(y=y.repeat(batch, 1, 1), y_pooled=y_pooled.repeat(batch, 1))

    if family == "alpha":
        return cond

    loc = _resolve_location(spec, range_model, device)
    cond["loc_embed"] = loc.repeat(batch, 1)

    if family == "sigma":
        coords, mask, tags = pack_sigma_points(spec.get("points", []), device)
        tok = tokenizer(tags, padding="max_length", max_length=TAG_MAX_LEN, truncation=True, return_tensors="pt")
        pe = tag_pooled_embeds(clip, tok["input_ids"].to(device), tok["attention_mask"].to(device))
        cond["point_prompts"] = pe.unsqueeze(0).repeat(batch, 1, 1)
        cond["pos"] = coords.unsqueeze(0).repeat(batch, 1, 1)
        cond["mask"] = mask.unsqueeze(0).repeat(batch, 1)
        return cond

    if family == "omega":
        geometry = pack_omega_instances(spec.get("instances", []), device)
        tok = tokenizer(
            geometry["tags"],
            padding="max_length",
            max_length=TAG_MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        ite = tag_pooled_embeds(clip, tok["input_ids"].to(device), tok["attention_mask"].to(device))
        cond["inst_text_embed"] = ite.unsqueeze(0).repeat(batch, 1, 1)
        for key in (
            "polygon_xy",
            "polygon_xy_mask",
            "polyline_xy",
            "polyline_xy_mask",
            "bbox_xyxy",
            "point_xy",
            "format_mask",
            "instance_mask",
        ):
            cond[key] = geometry[key].unsqueeze(0).repeat(batch, *([1] * geometry[key].dim()))
        cond["dropout_probs"] = dropout_probs
        return cond

    raise ValueError(f"unknown family {family!r}")
