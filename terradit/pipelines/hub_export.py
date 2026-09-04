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

"""Write a SiT-diffusers-style self-contained Diffusers folder.

Layout matches ``BiliSakura/SiT-diffusers``::

    pipeline.py
    pipeline_common.py
    pipeline_conditioning.py
    constants.py
    scheduler/scheduler_config.json
    transformer/transformer_sit.py
    transformer/sit.py
    transformer/localattn.py
    transformer/omega.py
    model_index.json
        "_class_name": ["pipeline", "<FamilyPipeline>"]
        "transformer": ["transformer_sit", "TerraDiTTransformer2DModel"]

``DiffusionPipeline.from_pretrained(folder, trust_remote_code=True)`` then loads
the copied Python without installing this repository.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .constants import PIPELINE_CLASS_NAME

FAMILY_PIPELINE_FILE = {
    "alpha": "pipeline_terradit_alpha.py",
    "sigma": "pipeline_terradit_sigma.py",
    "omega": "pipeline_terradit_omega.py",
}

HUB_PIPELINE_SIBLINGS = (
    "pipeline_common.py",
    "pipeline_conditioning.py",
    "constants.py",
)

HUB_TRANSFORMER_FILES = (
    ("transformer.py", "transformer_sit.py"),
    ("sit.py", "sit.py"),
    ("localattn.py", "localattn.py"),
    ("omega.py", "omega.py"),
)


def write_self_contained_repo(output_path: str | Path, family: str) -> None:
    """Copy Hub-safe pipeline / SiT sources and point ``model_index.json`` at them."""
    family = (family or "alpha").lower()
    if family not in FAMILY_PIPELINE_FILE:
        raise ValueError(f"unknown TerraDiT family {family!r}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    src_pipelines = Path(__file__).resolve().parent
    src_models = src_pipelines.parent / "models"

    shutil.copy2(src_pipelines / FAMILY_PIPELINE_FILE[family], output_path / "pipeline.py")
    for name in HUB_PIPELINE_SIBLINGS:
        shutil.copy2(src_pipelines / name, output_path / name)

    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in HUB_TRANSFORMER_FILES:
        shutil.copy2(src_models / src_name, transformer_dir / dest_name)

    index_path = output_path / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index["_class_name"] = ["pipeline", PIPELINE_CLASS_NAME[family]]
    index["transformer"] = ["transformer_sit", "TerraDiTTransformer2DModel"]
    index.setdefault("scheduler", ["diffusers", "FlowMatchEulerDiscreteScheduler"])
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
