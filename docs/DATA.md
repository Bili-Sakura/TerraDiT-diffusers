# Data

TerraDiT trains on **Git-10M** imagery plus a set of *derived conditioning artifacts* that
we distribute separately. Nothing in this repo re-hosts Git-10M images.

## 1. Imagery: Git-10M (pinned snapshot)

| | |
| --- | --- |
| Repo | [`lcybuaa/Git-10M`](https://huggingface.co/datasets/lcybuaa/Git-10M) |
| Revision | `29f192b8d2aa28b5d4d8c8d7f0f608cdc61fb52f` (2025-06-28) |
| Split / rows | `train`, 10,503,567 |
| On disk | 787 Arrow shards, ~394 GB |

Every metadata row we ship carries an `hf_idx`: the **row index into the `train` split of
that exact revision**. All loaders (`terradit/hf.py:load_git10m`) pin the revision, and the
dataset asserts the row's `img_name` matches, so a shifted index fails loudly instead of
silently training on the wrong tile.

The whole split is downloaded on first use (`datasets.load_dataset`); there is no partial
download without breaking row indices. Set `--hf-cache-dir` (e.g. `data/git10m/hf`) to
control where it lives, or pre-populate the cache with

```bash
python -c "from terradit.hf import load_git10m; load_git10m('data/git10m/hf')"
```

## 2. Derived artifacts: `MVRL/TerraDiT-data`

```bash
python scripts/download_data.py --data-root data/git10m                 # eval core (~1 GB)
python scripts/download_data.py --data-root data/git10m --family omega  # + omega training
python scripts/download_data.py --data-root data/git10m --family all
```

Layout under `data_root` (this is what every script expects):

```
data/git10m/
  metadata/{alpha,sigma,omega}.json      img_name, Google_location, latitude/longitude, hf_idx (no captions)
  splits/{random,spatial,dense}/
      metadata.json                      test tiles (14932 / 14387 / 3426)
      inst_metadata.npz                  omega instance geometry for those tiles
      range_plus.npz                     RANGE+ embeddings, row-aligned with metadata
  osm/tag_vocab.pt                       1,711 OSM "key value" tags -> id
  instance_metadata/inst_metadata.npz    omega instance geometry for the 2M training tiles (2.1 GB)
  points.pack                            sigma dense point supervision, 330k cities tiles (1.2 GB)
  range_plus/omega.npz                   RANGE+ [2M, 1280] fp16 (~5 GB), + locs [2M, 2] (lon, lat)
  range_plus/sigma.npy                   RANGE+ [330k, 1280] fp16 (~0.9 GB)
  latents/<img_name>.npy                 NOT distributed; see below
  gt/<split>/<stem>.png                  written by the evaluator on first use
```

Which family needs what:

| artifact | alpha | sigma | omega |
| --- | :-: | :-: | :-: |
| metadata/<family>.json | x | x | x |
| latents/ (or `--vae-on-the-fly`) | x | x | x |
| osm/tag_vocab.pt | | x | x |
| points.pack + range_plus/sigma.npy | | x | |
| instance_metadata + range_plus/omega.npz | | | x |

### Provenance

* **Imagery and captions**: Git-10M (zoom 17, 256x256, ~1 m/px), CC-BY-NC-ND-4.0, never
  redistributed here; captions are read from the user's Git-10M copy at run time.
  Coordinates are the north-west corner of the Web-Mercator tile named in the file (zoom_x_y),
  i.e. derived from the tile index.
* **OSM tags / points / instances**: OpenStreetMap features rasterised or vectorised per
  tile. © OpenStreetMap contributors, ODbL 1.0; these derived databases are released under ODbL. Tags use the `"<key> <value>"` convention (`building house`, `highway residential`).
  Sigma's `points.pack` stores a 256x256 tag-id raster per tile from which point prompts are
  sampled at train time; omega's `inst_metadata.npz` stores per-instance geometry
  (`geom_type`, `point_xy`, `bbox_xyxy`, `verts_xy`) plus the tag.
* **Geolocation**: [RANGE+](https://github.com/mvrl/RANGE) embeddings (1280-d = SatCLIP
  256 + RANGE retrieval 1024) of each tile's stored coordinate. Stored fp16; L2-normalised at load.
  `scripts/precompute_range_plus.py` regenerates them for new tiles.
* **Latents**: SDXL VAE (`stabilityai/sdxl-vae`) posterior `mean||std`, `[8, 32, 32]`
  per tile, fp16. Build with `scripts/encode_latents.py` (~33 GB for 2M tiles) or train
  with `--vae-on-the-fly`.

### Test splits

* `random`: 14,932 tiles sampled at random from the held-out set.
* `spatial`: 14,387 tiles from geographically held-out regions (no OSM point rasters exist
  for these tiles; sigma point prompts are derived from instance geometry there).
* `dense`: the 3,426 `random` tiles with >= 15 instances (a strict subset of `random`).

Ground-truth tiles for metrics are fetched from Git-10M by `hf_idx` and cached under
`data/git10m/gt/<split>/`.

## 3. Using your own tiles

1. Write a metadata json with `img_name`, `latitude`, `longitude` per tile (plus `caption` if the
   tiles are not Git-10M) and
   either `hf_idx` (Git-10M tiles; `python -m terradit.data.preprocessing.add_hf_idx --mode scan`)
   or an `images/<img_name>` folder under `data_root`.
2. `python scripts/precompute_range_plus.py --metadata ... --out data/git10m/range_plus/<name>.npz`
3. `python scripts/encode_latents.py --metadata ... --data-root data/git10m` (or on-the-fly).
4. Sigma: build a `points.pack` with `terradit.data.preprocessing.pack_points`; omega: an
   `inst_metadata.npz` with the keys above.
