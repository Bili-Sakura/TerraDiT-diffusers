# Third-party components and licenses

TerraDiT code in this repository is released under the Apache License 2.0 (see LICENSE).
The released **model weights** are provided for non-commercial research use
(CC-BY-NC-4.0), because they were trained on imagery distributed under a non-commercial
license. The components below carry their own terms.

## Data

| component | what we use | license / terms |
| --- | --- | --- |
| [Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) | training and evaluation imagery, captions, tile coordinates | CC-BY-NC-ND-4.0. **Not redistributed here.** Users download it from the original source; our metadata carries only tile identifiers, coordinates, and row indices, and the code reads captions from the user's copy. |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | feature tags, point rasters, and instance geometry per tile (`points.pack`, `inst_metadata.npz`, `tag_vocab.pt`, split geometry) | © OpenStreetMap contributors, [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). These derived databases are redistributed under ODbL in [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data). |
| [RANGE database](https://huggingface.co/datasets/mvrl/RANGE-database) | retrieval database for RANGE+ geolocation embeddings | MVRL; see the RANGE repository. |

## Models and code

| component | role | license |
| --- | --- | --- |
| [SiT](https://github.com/willisma/SiT) | diffusion transformer backbone and flow-matching sampler | MIT |
| [REPA](https://github.com/sihyun-yu/REPA) | representation-alignment training loss and trainer structure | MIT |
| [RANGE](https://github.com/mvrl/RANGE) | RANGE+ geolocation encoder (git submodule at `terradit/RANGE`) | MVRL; see the RANGE repository |
| [SatCLIP location encoder ViT16-L40](https://huggingface.co/MVRL/satclip-loc-enc-vit16-l40) | RANGE+ location tower (safetensors extracted from [microsoft/SatCLIP-ViT16-L40](https://huggingface.co/microsoft/SatCLIP-ViT16-L40)) | MIT |
| [SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae) | latent encoder / decoder | MIT |
| [LongCLIP KO-LITE](https://huggingface.co/zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14) | text encoder | MIT |
| [DINOv3](https://github.com/facebookresearch/dinov3) SAT-493M ViT-L/16 | frozen REPA target encoder (training only) | DINOv3 License (Meta). Weights are obtained by the user from Meta and are not redistributed. |
| [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | CLIP-score evaluation metric | MIT |
| torchmetrics / torch-fidelity | FID, LPIPS, SSIM metrics | Apache 2.0 |

## Attribution text

When you publish results or derived data, include:

> Contains information from OpenStreetMap, which is made available under the Open Database
> License (ODbL). © OpenStreetMap contributors. Imagery and captions from Git-10M
> (CC-BY-NC-ND-4.0), used under its terms.
