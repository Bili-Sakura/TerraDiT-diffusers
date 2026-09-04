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

"""TerraDiT-Ω text + geolocation + GALA instance-geometry pipeline."""

from __future__ import annotations

from typing import Any, Callable

import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import replace_example_docstring

from .constants import DIFFUSERS_REPO, FAMILY_HUB_SUBFOLDER, VARIANT_HUB_SUBFOLDER
from .pipeline_common import TerraDiTPipelineBase

EXAMPLE_DOC_STRING = r"""
    Examples:
        Load the converted Ω folder from the Hub:

        ```py
        >>> import torch
        >>> from terradit import TerraDiTOmegaPipeline

        >>> pipe = TerraDiTOmegaPipeline.from_pretrained(
        ...     "BiliSakura/TerraDiT",
        ...     subfolder="TerraDiT-Omega-XL",
        ...     torch_dtype=torch.float32,
        ... )
        >>> pipe = pipe.to("cuda")
        >>> image = pipe(
        ...     prompt="A small town crossed by a river and a road bridge.",
        ...     lat=48.86,
        ...     lon=2.35,
        ...     instances=[{"type": "point", "coords": [128, 128], "tag": "amenity parking"}],
        ...     condition_type="omega",
        ... ).images[0]
        >>> image.save("terradit_omega.png")
        ```

        The SiT-B/2 Ω variant uses the same pipeline class and a different subfolder:

        ```py
        >>> pipe = TerraDiTOmegaPipeline.from_pretrained(
        ...     "BiliSakura/TerraDiT",
        ...     subfolder="TerraDiT-Omega-B",
        ... )
        ```

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "BiliSakura/TerraDiT",
        ...     subfolder="TerraDiT-Omega-XL",
        ...     trust_remote_code=True,
        ... )
        ```
"""


class TerraDiTOmegaPipeline(TerraDiTPipelineBase):
    r"""
    Pipeline for instance-conditioned satellite image generation with TerraDiT-Ω.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Ω adds heterogeneous instance geometry (polygons, polylines, boxes, points) through GALA.
    The SiT-B/2 `omega_base` weights use this same class with Hub subfolder
    ``TerraDiT-Omega-B``.

    Parameters:
        transformer ([`TerraDiTTransformer2DModel`]):
            SiT backbone that predicts flow-matching velocity.
        scheduler ([`FlowMatchEulerDiscreteScheduler`] or a Karras-compatible scheduler):
            Scheduler used to denoise the encoded image latents. Released sampling is
            Euler flow-matching with a linear interpolant.
        vae ([`AutoencoderKL`], *optional*):
            SDXL VAE used to decode latents. Loaded from `stabilityai/sdxl-vae` when omitted.
        text_encoder ([`CLIPTextModel`], *optional*):
            LongCLIP text encoder. Loaded from the tokenizer id when omitted.
        tokenizer ([`CLIPTokenizer`], *optional*):
            LongCLIP tokenizer (144-token captions).
        geolocation_encoder ([`TerraDiTGeolocationModel`], *optional*):
            RANGE+ lat/lon encoder loaded from ``geolocation_encoder/``. Used when
            `lat` / `lon` are passed and `loc_embed` is omitted.
    """

    family = "omega"
    hub_repo_id = DIFFUSERS_REPO
    hub_subfolder = FAMILY_HUB_SUBFOLDER["omega"]
    hub_subfolder_base = VARIANT_HUB_SUBFOLDER["omega_base"]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 100,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 0.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        lat: float | None = None,
        lon: float | None = None,
        loc_embed: torch.Tensor | None = None,
        range_model: Any = None,
        instances: list | None = None,
        inst_text_embed: torch.Tensor | None = None,
        polygon_xy: torch.Tensor | None = None,
        polygon_xy_mask: torch.Tensor | None = None,
        polyline_xy: torch.Tensor | None = None,
        polyline_xy_mask: torch.Tensor | None = None,
        bbox_xyxy: torch.Tensor | None = None,
        point_xy: torch.Tensor | None = None,
        format_mask: torch.Tensor | None = None,
        instance_mask: torch.Tensor | None = None,
        condition_type: str = "omega",
        dropout_probs: list[float] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guidance_rescale: float = 0.0,
        clip_skip: int | None = None,
        callback_on_step_end: Callable[..., None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        callback: Callable[..., None] | None = None,
        callback_steps: int | None = None,
    ) -> ImagePipelineOutput | tuple:
        r"""
        Generate satellite images from text, geolocation, and GALA instance geometry.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Global caption(s). Git-10M-style long descriptions work best.
            height (`int`, *optional*):
                Output height in pixels. Defaults to the trained resolution (256).
                Other sizes (multiples of 16) interpolate positional embeddings.
            width (`int`, *optional*):
                Output width in pixels. Defaults to the trained resolution (256).
            num_inference_steps (`int`, defaults to 100):
                Euler steps. Released weights were sampled with 100 steps and no CFG.
            timesteps (`list[int]`, *optional*):
                Custom scheduler timesteps. Mutually exclusive with `sigmas` in some schedulers.
            sigmas (`list[float]`, *optional*):
                Custom flow-matching sigmas. Defaults to the paper `linspace(1, 0, steps+1)` schedule.
            guidance_scale (`float`, defaults to 0.0):
                Classifier-free guidance scale. Disabled at the released default of 0.
            negative_prompt (`str` or `list[str]`, *optional*):
                Negative captions used when `guidance_scale > 1`.
            num_images_per_prompt (`int`, defaults to 1):
                Variations drawn per caption.
            eta (`float`, defaults to 0.0):
                DDIM eta; ignored by the default flow-matching Euler scheduler.
            generator (`torch.Generator` or list, *optional*):
                RNG for latent initialization (and schedulers that accept it).
            latents (`torch.Tensor`, *optional*):
                Pre-sampled noise. Shape `(batch, 4, h/8, w/8)`.
            prompt_embeds / pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed LongCLIP embeddings. Both must be passed together.
            negative_prompt_embeds / negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed unconditional embeddings for guidance.
            lat / lon (`float`, *optional*):
                WGS84 coordinates. Encoded with RANGE+ when `loc_embed` is not given.
            loc_embed (`torch.Tensor`, *optional*):
                Precomputed RANGE+ embedding `(batch, 1280)` or `(1280,)`.
            range_model:
                Optional live RANGE+ encoder. Defaults to `geolocation_encoder`, then
                downloads `MVRL/satclip-loc-enc-vit16-l40` plus the RANGE database.
            instances (`list[dict]`, *optional*):
                Ω primitives: `{"type": "polygon|polyline|bbox|point", "coords": ..., "tag": ...}`.
            inst_text_embed / polygon_xy / ... / instance_mask:
                Precomputed omega tensors (eval / dataset mode).
            condition_type (`str`, defaults to `"omega"`):
                GALA dropout preset: `"omega"` (all), `"box"`, or `"point"`.
            dropout_probs (`list[float]`, *optional*):
                Explicit GALA dropout mask; overrides `condition_type` when set.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] instead of a bare tuple.
            cross_attention_kwargs / guidance_rescale:
                Accepted for Stable Diffusion signature compatibility; unused.
            clip_skip (`int`, *optional*):
                Unused (LongCLIP uses the last hidden state).
            callback_on_step_end:
                Called after each scheduler step with `(pipeline, step_index, timestep, callback_kwargs)`.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names included in `callback_kwargs`. Defaults to `["latents"]`.
            callback / callback_steps:
                Deprecated SD callback pair; use `callback_on_step_end`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                Generated images. NSFW flags are always absent (`None` if a tuple is requested).
        """
        self._warn_unused_sd_kwargs(cross_attention_kwargs, guidance_rescale, callback, callback_steps)
        height, width, batch_size, callback_on_step_end_tensor_inputs = self._resolve_call_batch(
            prompt, prompt_embeds, height, width, guidance_scale, callback_on_step_end_tensor_inputs,
        )
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds,
            negative_prompt_embeds, pooled_prompt_embeds, callback_on_step_end_tensor_inputs,
        )
        device = self._execution_device
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=clip_skip,
        )
        cond_kwargs = self._prepare_omega_kwargs(
            batch_size * num_images_per_prompt,
            device,
            prompt_embeds.dtype,
            lat=lat,
            lon=lon,
            loc_embed=loc_embed,
            range_model=range_model,
            instances=instances,
            inst_text_embed=inst_text_embed,
            polygon_xy=polygon_xy,
            polygon_xy_mask=polygon_xy_mask,
            polyline_xy=polyline_xy,
            polyline_xy_mask=polyline_xy_mask,
            bbox_xyxy=bbox_xyxy,
            point_xy=point_xy,
            format_mask=format_mask,
            instance_mask=instance_mask,
            dropout_probs=dropout_probs,
            condition_type=condition_type,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        return self._generate(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            cond_kwargs=cond_kwargs,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        )
