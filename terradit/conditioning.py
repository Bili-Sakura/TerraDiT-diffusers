"""Unified conditioning builder for the TerraDiT pipeline and demos.

Turns a human-friendly *spec* into the keyword arguments the Diffusers pipeline
expects, so the same code path serves both input sources:

  * dataset mode  -- conditioning pulled from a (HF-cached) dataset row, and
  * manual  mode  -- conditioning typed/clicked by a user.

Spec format (coordinates are tile pixels in [0, 256], the size of the decoded tile;
user-facing geometry is (x, y), matching a canvas click):

    alpha:  {"caption": "..."}
    sigma:  {"caption": "...",
             "points":   [(x, y, "tag"), ...],          # OSM point prompts
             "lat": <float|None>, "lon": <float|None>,  # geolocation (optional)
             "loc_embed": <Tensor|None>}                # or a precomputed embedding
    omega:  {"caption": "...",
             "instances": [{"type": "polygon|polyline|bbox|point",
                            "coords": [[x, y], ...] | [x, y],
                            "tag": "..."}, ...],
             "lat": ..., "lon": ..., "loc_embed": ...}

A polygon/polyline implies its bbox + a point, so the user only ever specifies the
richest primitive; pack_omega_instances() derives the rest (the inverse of the
derive-down visualization in viz.py). Geolocation is optional: sigma was trained
with location dropout so a zeroed location is in-distribution; omega prefers a real
one (pass lat/lon with RANGE+ loaded, or borrow an embedding from a dataset tile).
"""
import os
import sys

import numpy as np
import torch

from terradit.data.dataset import CAPTION_MAX_LEN, TAG_MAX_LEN

MAX_INSTANCES = 64
MAX_POINTS = 64
SIGMA_MAX_POINTS = 50
LOC_DIM = 1280
OMEGA_CONDITION_DROPOUTS = {"omega": None, "box": [1.0, 1.0, 0.0, 0.0], "point": [1.0, 1.0, 1.0, 0.0]}


# --------------------------------------------------------------------------- #
# Geolocation
# --------------------------------------------------------------------------- #
def load_range_model(device, *, beta=0.5, range_dir=None):
    """Lazily load RANGE+ for live lat/lon -> embedding. Returns None if unavailable.

    RANGE lives as an optional submodule; if it (or its weights) can't be loaded we
    return None and callers fall back to a borrowed/zeroed location. Set
    TERRADIT_RANGE_DIR to override the submodule location.
    """
    range_dir = range_dir or os.environ.get(
        "TERRADIT_RANGE_DIR",
        os.path.join(os.path.dirname(__file__), "RANGE"))
    if not os.path.isdir(os.path.join(range_dir, "range")):
        print(f"[range] RANGE submodule not found at {range_dir}.\n"
              "        Fetch it with:  git submodule update --init --recursive\n"
              "        (or set TERRADIT_RANGE_DIR). Geolocation falls back to zeros.")
        return None
    try:
        if range_dir not in sys.path:
            sys.path.insert(0, range_dir)
        from range.load_model import load_model  # noqa: E402  (submodule import)
        from huggingface_hub import hf_hub_download
        cache = os.path.join(range_dir, "pretrained")
        ckpt = hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt",
                               repo_type="model", local_dir=cache)
        db = hf_hub_download("mvrl/RANGE-database", "range_db_large.npz",
                             repo_type="dataset", local_dir=cache)
        return load_model(model_name="RANGE+", pretrained_path=ckpt, device=device,
                          db_path=db, beta=beta)
    except Exception as e:  # missing submodule, no network, etc.
        print(f"[range] RANGE+ unavailable ({type(e).__name__}: {e}); "
              f"geolocation will fall back to borrow/zeros.")
        return None


def loc_embed_from_latlon(range_model, lat, lon, device):
    """lat/lon -> normalized [1, LOC_DIM] RANGE+ embedding; zeros if no coords/model."""
    if range_model is not None and lat is not None and lon is not None:
        coords = torch.tensor([[float(lon), float(lat)]], dtype=torch.float64, device=device)
        loc = range_model(coords)
        loc = torch.from_numpy(loc).to(device) if isinstance(loc, np.ndarray) else loc.to(device)
        return torch.nn.functional.normalize(loc, dim=-1).float()
    return torch.zeros((1, LOC_DIM), dtype=torch.float32, device=device)


def loc_embed_from_dataset(ds, idx, device):
    """Borrow a real, normalized RANGE embedding from a dataset tile -> [1, LOC_DIM]."""
    loc = torch.from_numpy(ds.location_embeddings[idx]).float()
    return torch.nn.functional.normalize(loc, dim=-1).unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
# Text embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def caption_embeds(clip, tokenizer, prompts, device):
    """Encode caption text(s) -> (y [B,T,D], y_pooled [B,D], attn_mask [B,T])."""
    tok = tokenizer(prompts, padding="max_length", max_length=CAPTION_MAX_LEN,
                    truncation=True, return_tensors="pt")
    out = clip(tok.input_ids.to(device), tok.attention_mask.to(device))
    y = torch.nn.functional.normalize(out.last_hidden_state, dim=-1)
    y_pooled = torch.nn.functional.normalize(out.pooler_output, dim=-1)
    return y, y_pooled, tok.attention_mask.to(device)


@torch.no_grad()
def tag_pooled_embeds(clip, ids, attn):
    """Pooled, normalized CLIP embeddings for tag tokens. ids/attn: [M, TAG_MAX_LEN]."""
    out = clip(ids, attn).pooler_output
    return torch.nn.functional.normalize(out, dim=-1)


def resolve_sample_index(ds, img_name=None, index=0):
    """Resolve a dataset row: by exact img_name if given, else the raw index."""
    if img_name is None:
        return index
    for i, row in enumerate(ds.metadata):
        if row.get("img_name") == img_name:
            return i
    raise ValueError(f"img_name {img_name!r} not found in this family's metadata")


# --------------------------------------------------------------------------- #
# Geometry packing (manual mode)
# --------------------------------------------------------------------------- #
def pack_sigma_points(points, device, *, max_points=SIGMA_MAX_POINTS):
    """[(x, y, tag), ...] -> (coords [P,2] as (row,col), mask [P], tags [P]).

    mask is 0 for a real point and 1 for a padding slot (matching the dataset). The
    user gives (x, y); the model's positional encoding uses (row, col) = (y, x).
    """
    coords = torch.zeros(max_points, 2, dtype=torch.float32, device=device)
    mask = torch.ones(max_points, dtype=torch.long, device=device)  # 1 = padding
    tags = [""] * max_points
    n = min(len(points), max_points)
    for i in range(n):
        x, y, tag = points[i]
        coords[i] = torch.tensor([float(y), float(x)], device=device)  # (row, col)
        mask[i] = 0
        tags[i] = tag or ""
    return coords, mask, tags


def pack_omega_instances(instances, device):
    """[{type, coords, tag}, ...] -> omega geometry tensors + per-instance tags.

    Derives bbox + point from a polygon/polyline and a point from a bbox, so only the
    richest primitive needs specifying. Mirrors the dataset's format_mask semantics.
    """
    polygon_xy = torch.zeros(MAX_INSTANCES, MAX_POINTS, 2, device=device)
    polygon_xy_mask = torch.zeros(MAX_INSTANCES, MAX_POINTS, dtype=torch.bool, device=device)
    polyline_xy = torch.zeros(MAX_INSTANCES, MAX_POINTS, 2, device=device)
    polyline_xy_mask = torch.zeros(MAX_INSTANCES, MAX_POINTS, dtype=torch.bool, device=device)
    bbox_xyxy = torch.zeros(MAX_INSTANCES, 4, device=device)
    point_xy = torch.zeros(MAX_INSTANCES, 2, device=device)
    format_mask = torch.zeros(MAX_INSTANCES, 4, device=device)  # [polygon, polyline, bbox, point]
    instance_mask = torch.zeros(MAX_INSTANCES, device=device)

    N = min(len(instances), MAX_INSTANCES)
    tags = [""] * MAX_INSTANCES  # pad to MAX_INSTANCES so inst_text_embed N matches geometry
    for j, inst in enumerate(instances[:N]):
        itype = inst.get("type", "point")
        tags[j] = inst.get("tag", inst.get("prompt", "")) or ""
        raw = inst.get("coords", [])
        # a bare point may be given as [x, y]; normalize to a list of pairs
        if raw and not isinstance(raw[0], (list, tuple)):
            raw = [raw]
        coords = [(float(x), float(y)) for (x, y) in raw]
        L = min(len(coords), MAX_POINTS)

        if itype == "polygon" and L >= 3:
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
                L = min(len(coords), MAX_POINTS)
            c = torch.tensor(coords[:L], device=device)
            polygon_xy[j, :L] = c
            polygon_xy_mask[j, :L] = True
            xs, ys = c[:, 0], c[:, 1]
            bbox_xyxy[j] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device)
            point_xy[j] = torch.stack([xs.mean(), ys.mean()])
            format_mask[j, 0] = format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif itype == "polyline" and L >= 2:
            c = torch.tensor(coords[:L], device=device)
            polyline_xy[j, :L] = c
            polyline_xy_mask[j, :L] = True
            xs, ys = c[:, 0], c[:, 1]
            bbox_xyxy[j] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device)
            point_xy[j] = c[L // 2]
            format_mask[j, 1] = format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif itype == "bbox" and L >= 2:
            (x0, y0), (x1, y1) = coords[0], coords[1]
            bbox_xyxy[j] = torch.tensor([x0, y0, x1, y1], device=device)
            point_xy[j] = torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1)], device=device)
            format_mask[j, 2] = format_mask[j, 3] = 1.0
        elif L >= 1:  # point (or fallback)
            point_xy[j] = torch.tensor(coords[0], device=device)
            format_mask[j, 3] = 1.0
        instance_mask[j] = 1.0

    return dict(polygon_xy=polygon_xy, polygon_xy_mask=polygon_xy_mask,
                polyline_xy=polyline_xy, polyline_xy_mask=polyline_xy_mask,
                bbox_xyxy=bbox_xyxy, point_xy=point_xy, format_mask=format_mask,
                instance_mask=instance_mask, tags=tags)


# --------------------------------------------------------------------------- #
# Spec -> pipeline / transformer kwargs
# --------------------------------------------------------------------------- #
def _resolve_location(spec, range_model, device):
    if spec.get("loc_embed") is not None:
        loc = spec["loc_embed"]
        loc = torch.as_tensor(loc, device=device).float()
        return loc.view(1, -1)
    return loc_embed_from_latlon(range_model, spec.get("lat"), spec.get("lon"), device)


def build_conditioning(family, spec, clip, tokenizer, device, *,
                       range_model=None, num_images=1, dropout_probs=None):
    """Build transformer conditioning kwargs (batched to num_images) for a family.

    Returns a dict ready to splat into ``TerraDiTPipeline(..., **cond)`` or the
    SiT forward: ``y``, ``y_pooled``, and family-specific geometry / location.
    """
    B = num_images
    y, y_pooled, _ = caption_embeds(clip, tokenizer, [spec.get("caption", "")], device)
    cond = dict(y=y.repeat(B, 1, 1), y_pooled=y_pooled.repeat(B, 1))

    if family == "alpha":
        return cond

    loc = _resolve_location(spec, range_model, device)
    cond["loc_embed"] = loc.repeat(B, 1)

    if family == "sigma":
        coords, mask, tags = pack_sigma_points(spec.get("points", []), device)
        tok = tokenizer(tags, padding="max_length", max_length=TAG_MAX_LEN,
                        truncation=True, return_tensors="pt")
        pe = tag_pooled_embeds(clip, tok["input_ids"].to(device), tok["attention_mask"].to(device))
        cond["point_prompts"] = pe.unsqueeze(0).repeat(B, 1, 1)
        cond["pos"] = coords.unsqueeze(0).repeat(B, 1, 1)
        cond["mask"] = mask.unsqueeze(0).repeat(B, 1)
        return cond

    if family == "omega":
        g = pack_omega_instances(spec.get("instances", []), device)
        tok = tokenizer(g["tags"], padding="max_length", max_length=TAG_MAX_LEN,
                        truncation=True, return_tensors="pt")
        ite = tag_pooled_embeds(clip, tok["input_ids"].to(device), tok["attention_mask"].to(device))
        cond["inst_text_embed"] = ite.unsqueeze(0).repeat(B, 1, 1)
        for k in ("polygon_xy", "polygon_xy_mask", "polyline_xy", "polyline_xy_mask",
                  "bbox_xyxy", "point_xy", "format_mask", "instance_mask"):
            cond[k] = g[k].unsqueeze(0).repeat(B, *([1] * g[k].dim()))
        cond["dropout_probs"] = dropout_probs
        return cond

    raise ValueError(f"unknown family {family!r}")
