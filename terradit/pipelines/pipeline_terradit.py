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

"""Compatibility exports. Prefer the family-specific pipeline classes."""

from __future__ import annotations

from typing import Any

from terradit.hf import FAMILY_HUB_SUBFOLDER, MODELS, VARIANT_HUB_SUBFOLDER, is_diffusers_pipeline_dir
from terradit.pipelines.pipeline_common import (
    TerraDiTPipelineBase,
    flow_match_euler_denoise,
    paper_flow_sigmas,
    pipeline_class_for_family,
)
from terradit.pipelines.pipeline_terradit_alpha import TerraDiTAlphaPipeline
from terradit.pipelines.pipeline_terradit_omega import TerraDiTOmegaPipeline
from terradit.pipelines.pipeline_terradit_sigma import TerraDiTSigmaPipeline

__all__ = [
    "TerraDiTPipeline",
    "TerraDiTAlphaPipeline",
    "TerraDiTSigmaPipeline",
    "TerraDiTOmegaPipeline",
    "TerraDiTPipelineBase",
    "paper_flow_sigmas",
    "flow_match_euler_denoise",
    "pipeline_class_for_family",
]


def _infer_family(pretrained_model_name_or_path: str | None, family: str | None, subfolder: str | None) -> str:
    if family:
        return family
    if pretrained_model_name_or_path in MODELS:
        return MODELS[pretrained_model_name_or_path]["family"]
    if pretrained_model_name_or_path in FAMILY_HUB_SUBFOLDER:
        return pretrained_model_name_or_path
    if pretrained_model_name_or_path in VARIANT_HUB_SUBFOLDER:
        return MODELS.get(pretrained_model_name_or_path, {}).get("family", "omega" if "omega" in str(pretrained_model_name_or_path) else "alpha")
    if subfolder:
        for key, name in FAMILY_HUB_SUBFOLDER.items():
            if subfolder == name or subfolder.startswith(f"{name}-"):
                return key
        if "omega" in subfolder:
            return "omega"
        if "sigma" in subfolder:
            return "sigma"
        if "alpha" in subfolder:
            return "alpha"
    if pretrained_model_name_or_path and is_diffusers_pipeline_dir(pretrained_model_name_or_path):
        import json
        import os

        index_path = os.path.join(pretrained_model_name_or_path, "model_index.json")
        with open(index_path) as f:
            index = json.load(f)
        class_name = index.get("_class_name", "")
        for key, name in {
            "alpha": "TerraDiTAlphaPipeline",
            "sigma": "TerraDiTSigmaPipeline",
            "omega": "TerraDiTOmegaPipeline",
        }.items():
            if class_name == name:
                return key
        cfg_path = os.path.join(pretrained_model_name_or_path, "transformer", "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                return json.load(f).get("family", "alpha")
    return "alpha"


class TerraDiTPipeline:
    """Family-dispatching loader. Prefer `TerraDiTAlpha/Sigma/OmegaPipeline`."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        family = kwargs.pop("family", None)
        subfolder = kwargs.get("subfolder")
        family = _infer_family(pretrained_model_name_or_path, family, subfolder)
        return pipeline_class_for_family(family).from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_checkpoint(cls, pretrained_model_name_or_path: str, family: str | None = None, **kwargs: Any):
        subfolder = kwargs.get("subfolder")
        family = _infer_family(pretrained_model_name_or_path, family, subfolder)
        return pipeline_class_for_family(family).from_checkpoint(
            pretrained_model_name_or_path, family=family, **kwargs
        )

    def __new__(cls, *args, **kwargs):
        transformer = kwargs.get("transformer")
        if transformer is None and args:
            transformer = args[0]
        family = kwargs.get("family")
        if family is None and transformer is not None:
            family = getattr(getattr(transformer, "config", None), "family", None)
        real_cls = pipeline_class_for_family(family or "alpha")
        return real_cls(*args, **kwargs)
