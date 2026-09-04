---
license: odbl
pretty_name: TerraDiT derived conditioning data
tags:
  - satellite-imagery
  - remote-sensing
  - openstreetmap
  - geospatial
size_categories:
  - 1M<n<10M
---

# TerraDiT-data

Derived conditioning artifacts for training and evaluating
[TerraDiT / TerraDiT-Ω](https://github.com/mvrl/TerraDiT). **No imagery is included**: tiles
come from [`lcybuaa/Git-10M`](https://huggingface.co/datasets/lcybuaa/Git-10M) at revision
`29f192b8d2aa28b5d4d8c8d7f0f608cdc61fb52f`; every metadata row's `hf_idx` indexes the `train`
split of that snapshot.

```bash
python scripts/download_data.py --data-root data/git10m --family omega
```

## Contents (mirrors the `data_root` layout the code expects)

| path | rows / size | description |
| --- | --- | --- |
| `metadata/alpha.json` | 2.0M | img_name, Google_location, latitude, longitude, hf_idx (no Git-10M text; captions are read from your Git-10M copy) |
| `metadata/sigma.json` | 330k | cities subset used for TerraDiT-Σ |
| `metadata/omega.json` | 2.0M | tiles with instance geometry (TerraDiT-Ω) |
| `splits/random/` | 14,932 tiles | held-out random test split: `metadata.json`, `inst_metadata.npz`, `range_plus.npz` |
| `splits/spatial/` | 14,387 tiles | geographically held-out test split (same files) |
| `splits/dense/` | 3,426 tiles | `random` tiles with >= 15 instances (strict subset) |
| `osm/tag_vocab.pt` | 1,711 tags | OSM `"key value"` tag -> id |
| `instance_metadata/inst_metadata.npz` | 2.1 GB | per-instance `geom_type`, `point_xy`, `bbox_xyxy`, `verts_xy(+ptr)`, `tag_ids(+ptr)`, per-tile `tile_inst_start/end` |
| `points.pack` | 1.2 GB | 256x256 OSM tag-id rasters per cities tile (Σ point supervision) |
| `range_plus/omega.npz` | ~5 GB | RANGE+ embeddings `[2.0M, 1280]` fp16 + `locs [N, 2]` (lon, lat), row-aligned with `metadata/omega.json` |
| `range_plus/sigma.npy` | ~0.9 GB | RANGE+ embeddings `[330k, 1280]` fp16, row-aligned with `metadata/sigma.json` |

Not included: SDXL latents (build with `scripts/encode_latents.py`) and DINOv3 weights
(obtain from Meta).

## License and provenance

This dataset is released under the **Open Database License (ODbL) 1.0**.

* **OpenStreetMap-derived** tags, point rasters, and instance geometry (`osm/`, `points.pack`,
  `instance_metadata/`, `splits/*/inst_metadata.npz`): contains information from OpenStreetMap,
  made available under ODbL, © OpenStreetMap contributors (https://www.openstreetmap.org/copyright).
* **Metadata** (`metadata/`, `splits/*/metadata.json`): tile identifiers, tile-corner coordinates (a function of the tile index),
  and row indices into the pinned Git-10M snapshot. No Git-10M captions or statistics are
  redistributed; Git-10M itself is CC-BY-NC-ND-4.0 and must be obtained from its authors' repo.
* **RANGE+ embeddings** (`range_plus/`, `splits/*/range_plus.npz`): computed from tile coordinates
  with [RANGE](https://github.com/mvrl/RANGE) (SatCLIP + RANGE database).

## Citation

See [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT).
