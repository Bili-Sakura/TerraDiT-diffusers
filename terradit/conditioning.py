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
import torch

from terradit.data.dataset import CAPTION_MAX_LEN, TAG_MAX_LEN
from terradit.pipelines.pipeline_conditioning import (
    LOC_DIM,
    MAX_INSTANCES,
    MAX_POINTS,
    OMEGA_CONDITION_DROPOUTS,
    SIGMA_MAX_POINTS,
    build_conditioning,
    caption_embeds,
    loc_embed_from_latlon,
    load_range_model,
    pack_omega_instances,
    pack_sigma_points,
    tag_pooled_embeds,
)

__all__ = [
    "CAPTION_MAX_LEN",
    "TAG_MAX_LEN",
    "LOC_DIM",
    "MAX_INSTANCES",
    "MAX_POINTS",
    "SIGMA_MAX_POINTS",
    "OMEGA_CONDITION_DROPOUTS",
    "load_range_model",
    "loc_embed_from_latlon",
    "loc_embed_from_dataset",
    "caption_embeds",
    "tag_pooled_embeds",
    "resolve_sample_index",
    "pack_sigma_points",
    "pack_omega_instances",
    "build_conditioning",
]


def loc_embed_from_dataset(ds, idx, device):
    """Borrow a real, normalized RANGE embedding from a dataset tile -> [1, LOC_DIM]."""
    loc = torch.from_numpy(ds.location_embeddings[idx]).float()
    return torch.nn.functional.normalize(loc, dim=-1).unsqueeze(0).to(device)


def resolve_sample_index(ds, img_name=None, index=0):
    """Resolve a dataset row: by exact img_name if given, else the raw index."""
    if img_name is None:
        return index
    for i, row in enumerate(ds.metadata):
        if row.get("img_name") == img_name:
            return i
    raise ValueError(f"img_name {img_name!r} not found in this family's metadata")
