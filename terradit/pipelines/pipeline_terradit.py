# Copyright 2026 The TerraDiT Authors and The HuggingFace Team. All rights reserved.
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

"""Re-exports for the family-specific TerraDiT pipelines."""

from terradit.pipelines.pipeline_common import (
    TerraDiTPipelineBase,
    flow_match_euler_denoise,
    paper_flow_sigmas,
)
from terradit.pipelines.pipeline_terradit_alpha import TerraDiTAlphaPipeline
from terradit.pipelines.pipeline_terradit_omega import TerraDiTOmegaPipeline
from terradit.pipelines.pipeline_terradit_sigma import TerraDiTSigmaPipeline


def pipeline_class_for_family(family: str):
    """Return the family pipeline class (`TerraDiTAlphaPipeline`, …)."""
    key = (family or "alpha").lower()
    try:
        return {
            "alpha": TerraDiTAlphaPipeline,
            "sigma": TerraDiTSigmaPipeline,
            "omega": TerraDiTOmegaPipeline,
        }[key]
    except KeyError as exc:
        raise ValueError(f"unknown TerraDiT family {family!r}") from exc


__all__ = [
    "TerraDiTAlphaPipeline",
    "TerraDiTSigmaPipeline",
    "TerraDiTOmegaPipeline",
    "TerraDiTPipelineBase",
    "paper_flow_sigmas",
    "flow_match_euler_denoise",
    "pipeline_class_for_family",
]
