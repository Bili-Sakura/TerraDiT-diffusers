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

"""Hub-safe constants copied next to ``pipeline.py`` in a converted folder.

This module must not import ``terradit`` so Diffusers can load a converted
checkpoint without installing this repository.
"""

TOKENIZER_ID = "zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14"
CAPTION_MAX_LEN = 144
TAG_MAX_LEN = 5
LOC_DIM = 1280
VAE_ID = "stabilityai/sdxl-vae"

DIFFUSERS_REPO = "BiliSakura/TerraDiT"
FAMILY_HUB_SUBFOLDER = {
    "alpha": "TerraDiT-Alpha-XL",
    "sigma": "TerraDiT-Sigma-XL",
    "omega": "TerraDiT-Omega-XL",
}
VARIANT_HUB_SUBFOLDER = {
    "alpha_xl": "TerraDiT-Alpha-XL",
    "sigma_xl": "TerraDiT-Sigma-XL",
    "omega_xl": "TerraDiT-Omega-XL",
    "omega_base": "TerraDiT-Omega-B",
}
PIPELINE_CLASS_NAME = {
    "alpha": "TerraDiTAlphaPipeline",
    "sigma": "TerraDiTSigmaPipeline",
    "omega": "TerraDiTOmegaPipeline",
}
