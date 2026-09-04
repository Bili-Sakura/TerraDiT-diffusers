"""Unified TerraDiT dataset for the alpha / sigma / omega model families.

All artifact paths resolve under a single ``data_root`` with a fixed internal
layout, so the only difference between a local mirror and a Hugging Face download is
where ``data_root`` points. No absolute paths are baked
in. Each family preserves its original ``__getitem__`` return signature (the
training/model code depends on those tuples).

Layout under ``data_root``::

    metadata/{alpha,sigma,omega}.json  # img_name, Google_location, latitude, longitude, hf_idx
    latents/<img_name>.npy            # optional when vae_on_the_fly=True
    range_plus/sigma.npy              # RANGE+ embeddings aligned to sigma metadata (1280-d, fp16)
    range_plus/omega.npz              # RANGE+ embeddings aligned to omega metadata (1280-d, fp16)
    osm/tag_vocab.pt
    points.pack                       # sigma point supervision (packed tiles)
    instance_metadata/inst_metadata.npz   # omega instance geometry
    images/                           # optional disk fallback (primary = HF)
"""
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import AutoTokenizer

TOKENIZER_ID = "zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14"
CAPTION_MAX_LEN = 144
TAG_MAX_LEN = 5
LOC_DIM = 1280


def sample_points(x, num_points, max_points):
    """Sample point-prompt pixel coordinates from a dense 256x256 tag tensor.

    Verbatim behaviour from the original REPA sigma dataset: returns
    (coords [max_points, 2], tag-index values [max_points], mask [max_points]).
    """
    valid_indices = torch.nonzero(x != -1, as_tuple=False)
    if valid_indices.numel() == 0:
        valid_indices = torch.nonzero(x == -1, as_tuple=False)
        idx = valid_indices[torch.randint(len(valid_indices), (max_points,))]
        point_prompts = x[idx[:, 0], idx[:, 1]]
        mask = torch.ones((max_points,), dtype=torch.long)
        return idx, point_prompts, mask
    else:
        idx = valid_indices[torch.randint(len(valid_indices), (num_points,))]
        if num_points < max_points:
            extra_idx = torch.randint(256, (max_points - num_points, 2))
            point_prompts = torch.cat(
                [x[idx[:, 0], idx[:, 1]],
                 -1 * torch.ones((max_points - num_points,), dtype=torch.long)], dim=0)
            idx = torch.cat([idx, extra_idx], dim=0)
            mask = torch.cat(
                [torch.zeros((num_points,), dtype=torch.long),
                 torch.ones((max_points - num_points,), dtype=torch.long)], dim=0)
        else:
            point_prompts = x[idx[:, 0], idx[:, 1]]
            mask = torch.zeros((max_points,), dtype=torch.long)
        return idx, point_prompts, mask


# --------------------------------------------------------------------------- #
# Point sources for sigma (dense 256x256 tag tensor + per-tile taglist).
# Two interchangeable implementations behind .get(google_location) ->
# (pixel_tensor [256,256] long, index_to_taglist dict|None). A missing tile
# yields an all -1 tensor (no OSM coverage).
# --------------------------------------------------------------------------- #
class LegacyPixelTensorSource:
    """Reads the original per-tile ``.pt`` tensors + ``.json`` taglists."""

    def __init__(self, pixel_dir, tag_dir):
        self.pixel_dir = pixel_dir
        self.tag_dir = tag_dir

    def get(self, google_location):
        pt_path = os.path.join(self.pixel_dir, google_location + ".pt")
        if not os.path.exists(pt_path):
            return -1 * torch.ones((256, 256), dtype=torch.long), None
        pixel_tensor = torch.load(pt_path).long()
        index_to_taglist = json.load(
            open(os.path.join(self.tag_dir, google_location + ".json"), "r"))
        return pixel_tensor, index_to_taglist


class PackedPointSource:
    """Reads the packed ``points.pack`` produced by preprocessing/pack_points.py.

    Exposes the same ``.get`` interface as the legacy source so SigmaDataset is
    agnostic to which one it uses.
    """

    def __init__(self, pack_path):
        from .preprocessing.points_pack import PointPackReader  # local import
        self.reader = PointPackReader(pack_path)

    def get(self, google_location):
        return self.reader.get(google_location)


class InstancePointSource:
    """Point rasters derived from omega instance geometry (``inst_metadata.npz``).

    Used for the held-out test splits, which ship instance geometry but no dense OSM
    rasters. Each instance is rasterised into a 256x256 tag-id raster the way the OSM
    rasters were built (polygons filled, polylines drawn 2 px wide, plus the
    representative point), so ``sample_points`` draws sigma point prompts from the same
    kind of tensor as in training. Same ``.get(google_location)`` interface as the other
    sources; tiles without instances yield an all -1 raster (no point prompts).
    """

    def __init__(self, inst_meta_path, metadata):
        m = np.load(inst_meta_path)
        self.geom_type, self.point_xy = m["geom_type"], m["point_xy"]
        self.bbox_xyxy = m["bbox_xyxy"]
        self.verts_xy, self.verts_ptr = m["verts_xy"], m["verts_ptr"]
        self.tag_ids, self.tag_ids_ptr = m["tag_ids"], m["tag_ids_ptr"]
        self.start, self.end = m["tile_inst_start"], m["tile_inst_end"]
        self.row_of = {r["Google_location"]: i for i, r in enumerate(metadata)}

    def get(self, google_location):
        from PIL import ImageDraw
        i = self.row_of.get(google_location)
        if i is None or self.start[i] < 0 or self.end[i] <= self.start[i]:
            return -1 * torch.ones((256, 256), dtype=torch.long), None
        canvas = Image.new("I", (256, 256), -1)
        draw = ImageDraw.Draw(canvas)
        used = set()
        # paint large footprints first so small features (buildings, roads) stay on top
        insts = list(range(int(self.start[i]), int(self.end[i])))
        bb = self.bbox_xyxy[insts]
        area = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
        insts = [insts[k] for k in np.argsort(-area, kind="stable")]
        for inst in insts:
            t0, t1 = int(self.tag_ids_ptr[inst]), int(self.tag_ids_ptr[inst + 1])
            if t1 <= t0:
                continue
            tag = int(self.tag_ids[t0])
            used.add(tag)
            verts = self.verts_xy[int(self.verts_ptr[inst]):int(self.verts_ptr[inst + 1])]
            pts = [(float(x), float(y)) for x, y in verts]
            if int(self.geom_type[inst]) == 1 and len(pts) >= 3:      # polygon: filled
                draw.polygon(pts, fill=tag)
            elif len(pts) >= 2:                                        # polyline: 2 px stroke
                draw.line(pts, fill=tag, width=2)
            px, py = self.point_xy[inst]
            draw.point((float(px), float(py)), fill=tag)
        pixel = torch.from_numpy(np.array(canvas, dtype=np.int64))
        return pixel, {str(t): [t] for t in sorted(used)}


# --------------------------------------------------------------------------- #
# Shared base: metadata, tokenizer, image (HF or disk), caption, latent.
# --------------------------------------------------------------------------- #
class _TerraDiTBase(Dataset):
    def __init__(self, metadata_path, *, hf_dataset=None, images_root=None,
                 latent_root=None, vae_on_the_fly=False, split="train",
                 max_samples=2_000_000):
        self.metadata = json.load(open(metadata_path, "r"))[:max_samples]
        self.hf_dataset = hf_dataset
        self.images_root = images_root
        self.latent_root = latent_root
        self.vae_on_the_fly = vae_on_the_fly
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def __len__(self):
        return len(self.metadata)

    def _load_row(self, idx):
        """Load the tile image (uint8 CHW) and its caption.

        Primary path: the Hugging Face dataset (``lcybuaa/Git-10M``) keyed by the metadata
        row's ``hf_idx``; the caption is read from the same row, so the shipped metadata
        never has to carry Git-10M text. Falls back to ``images_root/<img_name>`` on disk
        plus a ``caption`` field in the metadata, if present.
        """
        sample = self.metadata[idx]
        caption = sample.get("caption", "")
        if self.hf_dataset is not None and "hf_idx" in sample:
            hf = self.hf_dataset[int(sample["hf_idx"])]
            if hf.get("img_name") not in (None, sample.get("img_name")):
                raise ValueError(
                    f"HF sample {hf.get('img_name')} != metadata {sample.get('img_name')}")
            img = hf["image"]
            caption = hf.get("caption", caption) or caption
        else:
            if self.images_root is None:
                raise RuntimeError(
                    "No image source: provide hf_dataset (with hf_idx metadata) "
                    "or images_root for disk fallback.")
            name = sample["img_name"]
            path = os.path.join(self.images_root, name)
            if not os.path.exists(path) and not name.lower().endswith((".jpg", ".png")):
                path = path + ".jpg"
            img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1), caption

    def caption(self, idx):
        """Caption for a row without decoding the image (HF row if available, else metadata)."""
        sample = self.metadata[idx]
        if self.hf_dataset is not None and "hf_idx" in sample:
            return self.hf_dataset[int(sample["hf_idx"])].get("caption", "") or sample.get("caption", "")
        return sample.get("caption", "")

    def _load_latent(self, sample):
        """Precomputed SDXL latent (.npy). Empty when computing latents on the fly."""
        if self.vae_on_the_fly or self.latent_root is None:
            return torch.empty(0)
        path = os.path.join(self.latent_root, sample["img_name"] + ".npy")
        return torch.from_numpy(np.load(path)).float()

    def _caption_tokens(self, caption, drop_prob):
        if self.split == "train" and torch.rand(1) <= drop_prob:
            caption = ""
        out = self.tokenizer(caption or "", padding="max_length",
                             max_length=CAPTION_MAX_LEN, truncation=True,
                             return_tensors="pt")
        return out["input_ids"][0], out["attention_mask"][0]


# --------------------------------------------------------------------------- #
# alpha: text -> image
# --------------------------------------------------------------------------- #
class AlphaDataset(_TerraDiTBase):
    CAPTION_DROP = 0.1

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        image, caption = self._load_row(idx)
        latent = self._load_latent(sample)
        cap_ids, cap_attn = self._caption_tokens(caption, self.CAPTION_DROP)
        return image, latent, cap_ids, cap_attn


# --------------------------------------------------------------------------- #
# sigma: text + geolocation + point prompts -> image
# --------------------------------------------------------------------------- #
class SigmaDataset(_TerraDiTBase):
    CAPTION_DROP = 0.5
    LOC_DROP = 0.7  # fraction of samples with a zeroed geolocation embedding

    def __init__(self, metadata_path, *, point_source, tag_vocab_path,
                 loc_embed_path, max_points=50, **base_kwargs):
        super().__init__(metadata_path, **base_kwargs)
        self.point_source = point_source
        tag_vocab = torch.load(tag_vocab_path)
        self.tag_index = {int(v): k for k, v in tag_vocab.items()}
        self.tag_index[-1] = ""
        self.location_embeddings = _load_loc_embeddings(loc_embed_path)
        self.max_points = max_points

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        image, caption = self._load_row(idx)
        latent = self._load_latent(sample)

        if torch.rand(1) <= self.LOC_DROP:
            loc_embed = torch.zeros(LOC_DIM, dtype=torch.float)
        else:
            loc_embed = torch.from_numpy(np.asarray(self.location_embeddings[idx], dtype=np.float32))
            loc_embed = torch.nn.functional.normalize(loc_embed, dim=-1)

        cap_ids, cap_attn = self._caption_tokens(caption, self.CAPTION_DROP)

        pixel_tensor, index_to_taglist = self.point_source.get(sample["Google_location"])
        num_points = torch.randint(1, self.max_points, (1,)).item()
        coords, point_vals, mask = sample_points(pixel_tensor, num_points, self.max_points)
        if index_to_taglist is None:
            point_tags = [""] * len(point_vals)
        else:
            point_tags = [self.tag_index[random.choice(index_to_taglist[str(v.item())])]
                          if v.item() != -1 else "" for v in point_vals]
        point_prompts = self.tokenizer(point_tags, padding="max_length",
                                       max_length=TAG_MAX_LEN, truncation=True,
                                       return_tensors="pt")
        return (image, latent, cap_ids, cap_attn, loc_embed,
                point_prompts["input_ids"], point_prompts["attention_mask"], coords, mask)


# --------------------------------------------------------------------------- #
# omega: text + geolocation + instance geometry (GALA) -> image
# --------------------------------------------------------------------------- #
class OmegaDataset(_TerraDiTBase):
    CAPTION_DROP = 0.1

    def __init__(self, metadata_path, *, inst_meta_path, tag_vocab_path,
                 loc_embed_path, max_instances=64, max_points=64, **base_kwargs):
        super().__init__(metadata_path, **base_kwargs)
        self.max_instances = max_instances
        self.max_points = max_points
        self.location_embeddings = _load_loc_embeddings(loc_embed_path)
        tag_vocab = torch.load(tag_vocab_path)
        self.tag_index = {int(v): k for k, v in tag_vocab.items()}
        self.tag_index[-1] = ""

        m = np.load(inst_meta_path)
        self.geom_type = m["geom_type"]
        self.point_xy = m["point_xy"]
        self.bbox_xyxy = m["bbox_xyxy"]
        self.verts_xy = m["verts_xy"]
        self.verts_ptr = m["verts_ptr"]
        self.tag_ids = m["tag_ids"]
        self.tag_ids_ptr = m["tag_ids_ptr"]
        self.tile_inst_start = m["tile_inst_start"]
        self.tile_inst_end = m["tile_inst_end"]

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        image, caption = self._load_row(idx)
        latent = self._load_latent(sample)
        loc_embed = torch.from_numpy(np.asarray(self.location_embeddings[idx], dtype=np.float32))
        loc_embed = torch.nn.functional.normalize(loc_embed, dim=-1).float()
        cap_ids, cap_attn = self._caption_tokens(caption, self.CAPTION_DROP)

        N, P = self.max_instances, self.max_points
        polygon_xy = torch.zeros(N, P, 2, dtype=torch.float32)
        polygon_xy_mask = torch.zeros(N, P, dtype=torch.bool)
        polyline_xy = torch.zeros(N, P, 2, dtype=torch.float32)
        polyline_xy_mask = torch.zeros(N, P, dtype=torch.bool)
        bbox_xyxy = torch.zeros(N, 4, dtype=torch.float32)
        point_xy = torch.zeros(N, 2, dtype=torch.float32)
        format_mask = torch.zeros(N, 4, dtype=torch.float32)
        instance_mask = torch.zeros(N, dtype=torch.float32)
        inst_tag_texts = [""] * N

        inst_start = int(self.tile_inst_start[idx])
        inst_end = int(self.tile_inst_end[idx])
        num_inst = 0 if (inst_start < 0 or inst_end <= inst_start) else (inst_end - inst_start)
        num_inst = min(num_inst, N)

        for j in range(num_inst):
            inst_id = inst_start + j
            gtype = int(self.geom_type[inst_id])  # 1 = polygon, 0 = polyline
            v0 = int(self.verts_ptr[inst_id])
            v1 = int(self.verts_ptr[inst_id + 1])
            verts = self.verts_xy[v0:v1]
            L = min(len(verts), P)
            if L > 0:
                verts_t = torch.from_numpy(verts[:L]).float()
                if gtype == 1:
                    polygon_xy[j, :L] = verts_t
                    polygon_xy_mask[j, :L] = True
                else:
                    polyline_xy[j, :L] = verts_t
                    polyline_xy_mask[j, :L] = True
            bbox_xyxy[j] = torch.from_numpy(self.bbox_xyxy[inst_id]).float()
            point_xy[j] = torch.from_numpy(self.point_xy[inst_id]).float()
            format_mask[j, 0 if gtype == 1 else 1] = 1.0  # polygon / polyline
            format_mask[j, 2] = 1.0  # bbox
            format_mask[j, 3] = 1.0  # point
            instance_mask[j] = 1.0
            t0 = int(self.tag_ids_ptr[inst_id])
            t1 = int(self.tag_ids_ptr[inst_id + 1])
            inst_tag_texts[j] = self.tag_index.get(int(self.tag_ids[t0]), "") if t1 > t0 else ""

        tok = self.tokenizer(inst_tag_texts, padding="max_length", max_length=TAG_MAX_LEN,
                             truncation=True, return_tensors="pt")
        out = (image, latent, cap_ids, cap_attn, loc_embed,
               polygon_xy, polygon_xy_mask, polyline_xy, polyline_xy_mask,
               bbox_xyxy, point_xy, format_mask, instance_mask,
               tok["input_ids"], tok["attention_mask"])
        if self.split == "test":
            return out + (sample["img_name"],)
        return out


def _load_loc_embeddings(path):
    """RANGE+ geolocation embeddings (fp16 or fp32): .npz -> ['embeddings'], else raw .npy.

    .npy files are memory-mapped; .npz files are decompressed into RAM (~10 GB for the
    full omega table in fp16). Rows are cast to float32 when read.
    """
    if path.endswith(".npz"):
        return np.load(path)["embeddings"]
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_dataset(family, data_root, *, hf_dataset=None, split="train",
                  vae_on_the_fly=False, max_points=None,
                  max_instances=64, use_packed_points=True,
                  metadata_path=None, inst_meta_path=None, loc_embed_path=None,
                  max_samples=None, point_source="pack", points_pack_path=None):
    """Construct the dataset for a family using the canonical data_root layout.

    metadata_path / inst_meta_path / loc_embed_path override the default in-tree
    locations, used to point a family at a specific test split (random/spatial/dense).
    max_samples caps the metadata to its first N rows (subset fine-tuning / smoke tests).
    point_source (sigma): "pack" (points.pack dense OSM rasters; training and the paper
    protocol), "instances" (derive from inst_meta_path; the shipped test splits), or
    "legacy" (per-tile .pt/.json under osm/).
    """
    p = lambda *parts: os.path.join(data_root, *parts)
    base = dict(hf_dataset=hf_dataset, images_root=p("images"),
                latent_root=p("latents"), vae_on_the_fly=vae_on_the_fly, split=split)
    if max_samples is not None:
        base["max_samples"] = max_samples

    if family == "alpha":
        return AlphaDataset(metadata_path or p("metadata", "alpha.json"), **base)

    if family == "sigma":
        if not use_packed_points:
            point_source = "legacy"
        meta_path = metadata_path or p("metadata", "sigma.json")
        if point_source == "pack":
            src = PackedPointSource(points_pack_path or p("points.pack"))
        elif point_source == "instances":
            src = InstancePointSource(inst_meta_path or p("instance_metadata", "inst_metadata.npz"),
                                      json.load(open(meta_path)))
        elif point_source == "legacy":
            src = LegacyPixelTensorSource(p("osm", "pixel_tensors"), p("osm", "list_tags"))
        else:
            raise ValueError(f"unknown point_source {point_source!r}")
        return SigmaDataset(
            meta_path, point_source=src,
            tag_vocab_path=p("osm", "tag_vocab.pt"),
            loc_embed_path=loc_embed_path or p("range_plus", "sigma.npy"),
            max_points=(max_points or 50), **base)

    if family == "omega":
        return OmegaDataset(
            metadata_path or p("metadata", "omega.json"),
            inst_meta_path=inst_meta_path or p("instance_metadata", "inst_metadata.npz"),
            tag_vocab_path=p("osm", "tag_vocab.pt"),
            loc_embed_path=loc_embed_path or p("range_plus", "omega.npz"),
            max_instances=max_instances, max_points=(max_points or 64), **base)

    raise ValueError(f"unknown family {family!r} (expected alpha|sigma|omega)")
