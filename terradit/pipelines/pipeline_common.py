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

"""Shared TerraDiT encode / latents / Euler / decode helpers for family pipelines."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizerBase

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor

from terradit.conditioning import OMEGA_CONDITION_DROPOUTS, build_conditioning, loc_embed_from_latlon, load_range_model
from terradit.data.dataset import CAPTION_MAX_LEN, TOKENIZER_ID
from terradit.hf import (
    DIFFUSERS_REPO,
    FAMILY_HUB_SUBFOLDER,
    MODELS,
    VARIANT_HUB_SUBFOLDER,
    VAE_ID,
    is_diffusers_pipeline_dir,
)
from terradit.models.transformer import TerraDiTTransformer2DModel

try:
    from diffusers.utils import is_torch_xla_available
except ImportError:  # older Diffusers
    def is_torch_xla_available() -> bool:
        return False

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

PIPELINE_CLASS_NAME = {
    "alpha": "TerraDiTAlphaPipeline",
    "sigma": "TerraDiTSigmaPipeline",
    "omega": "TerraDiTOmegaPipeline",
}
PIPELINE_MODULE = {
    "alpha": "terradit.pipelines.pipeline_terradit_alpha",
    "sigma": "terradit.pipelines.pipeline_terradit_sigma",
    "omega": "terradit.pipelines.pipeline_terradit_omega",
}


def paper_flow_sigmas(num_inference_steps: int) -> list[float]:
    r"""
    Sigma schedule matching the released TerraDiT Euler sampler.

    The original sampler used ``linspace(1, 0, steps + 1)``. Diffusers' flow-match
    scheduler appends the terminal 0, so this returns the leading ``steps`` values.

    Parameters:
        num_inference_steps (`int`):
            Number of denoising steps. Must be >= 1.

    Returns:
        `list[float]`:
            Sigmas from 1.0 down to ``1 / num_inference_steps``.
    """
    if num_inference_steps < 1:
        raise ValueError(f"`num_inference_steps` must be >= 1, got {num_inference_steps}")
    return np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).tolist()


def _unwrap_transformer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output.sample


@torch.no_grad()
def flow_match_euler_denoise(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    *,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
    num_inference_steps: int = 100,
    extra_step_kwargs: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> torch.Tensor:
    r"""
    Paper-faithful flow-matching Euler loop using [`FlowMatchEulerDiscreteScheduler`].

    Used by the pipeline denoising stage and by the trainer's wandb sample grid so
    both paths share one scheduler implementation.

    Parameters:
        transformer (`nn.Module`):
            Velocity model. Accepts `(latents, timestep, **model_kwargs)` and returns a
            tensor or a Diffusers output with `.sample`.
        latents (`torch.Tensor`):
            Initial noise of shape `(batch, channels, h, w)` at time 1.
        scheduler ([`FlowMatchEulerDiscreteScheduler`], *optional*):
            Scheduler used for `set_timesteps` / `step`. A shift-1 Euler scheduler is
            constructed when omitted.
        num_inference_steps (`int`, defaults to 100):
            Number of Euler steps (released sampling uses 100).
        extra_step_kwargs (`dict`, *optional*):
            Extra arguments forwarded to `scheduler.step` (e.g. `generator`).
        **model_kwargs:
            Conditioning tensors forwarded to the transformer (`y`, `y_pooled`, ...).

    Returns:
        `torch.Tensor`:
            Denoised latents at time 0, same shape as `latents`.
    """
    if scheduler is None:
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    extra_step_kwargs = extra_step_kwargs or {}
    sigmas = paper_flow_sigmas(num_inference_steps)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, sigmas=sigmas, device=latents.device)
    latents = latents.to(dtype=torch.float32)
    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        timestep = torch.full((latents.shape[0],), sigma, device=latents.device, dtype=latents.dtype)
        model_output = _unwrap_transformer_output(transformer(latents, timestep, **model_kwargs))
        latents = scheduler.step(model_output, t, latents, return_dict=False, **extra_step_kwargs)[0]
        if XLA_AVAILABLE:
            xm.mark_step()
    return latents


def pipeline_class_for_family(family: str):
    """Return the family pipeline class (`TerraDiTAlphaPipeline`, …)."""
    from terradit.pipelines.pipeline_terradit_alpha import TerraDiTAlphaPipeline
    from terradit.pipelines.pipeline_terradit_omega import TerraDiTOmegaPipeline
    from terradit.pipelines.pipeline_terradit_sigma import TerraDiTSigmaPipeline

    key = (family or "alpha").lower()
    try:
        return {
            "alpha": TerraDiTAlphaPipeline,
            "sigma": TerraDiTSigmaPipeline,
            "omega": TerraDiTOmegaPipeline,
        }[key]
    except KeyError as exc:
        raise ValueError(f"unknown TerraDiT family {family!r}") from exc


def resolve_hub_load(pretrained_model_name_or_path: str | os.PathLike | None, subfolder: str | None, default_subfolder: str):
    """Default ``subfolder`` when loading ``BiliSakura/TerraDiT`` from the Hub."""
    if pretrained_model_name_or_path is None:
        return None, subfolder
    path = os.fspath(pretrained_model_name_or_path)
    if os.path.exists(path):
        return path, subfolder
    if path in (DIFFUSERS_REPO, "BiliSakura/TerraDiT"):
        return path, subfolder if subfolder is not None else default_subfolder
    if path in VARIANT_HUB_SUBFOLDER:
        return DIFFUSERS_REPO, subfolder if subfolder is not None else VARIANT_HUB_SUBFOLDER[path]
    if path in FAMILY_HUB_SUBFOLDER:
        return DIFFUSERS_REPO, subfolder if subfolder is not None else FAMILY_HUB_SUBFOLDER[path]
    return path, subfolder


class TerraDiTPipelineBase(DiffusionPipeline):
    r"""
    Shared SD-style encode / latents / Euler / decode path for TerraDiT family pipelines.

    α / Σ / Ω each subclass this with their own pipeline name and `__call__`.
    """

    family: str = "alpha"
    hub_repo_id: str = DIFFUSERS_REPO
    hub_subfolder: str = "TerraDiT-alpha"
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["vae", "text_encoder", "tokenizer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: TerraDiTTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers | FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL | None = None,
        text_encoder: CLIPTextModel | None = None,
        tokenizer: PreTrainedTokenizerBase | CLIPTokenizer | None = None,
        family: str | None = None,
        caption_max_length: int = CAPTION_MAX_LEN,
        tokenizer_id: str = TOKENIZER_ID,
        vae_id: str = VAE_ID,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        vae_scale = 8
        if vae is not None and getattr(vae, "config", None) is not None:
            blocks = getattr(vae.config, "block_out_channels", None)
            if blocks:
                vae_scale = 2 ** (len(blocks) - 1)
        self.vae_scale_factor = vae_scale
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(
            family=family or getattr(self, "family", None) or getattr(transformer.config, "family", "alpha"),
            caption_max_length=caption_max_length,
            tokenizer_id=tokenizer_id,
            vae_id=vae_id,
        )

    def _ensure_aux(self, device: torch.device | str | None = None) -> None:
        """Load the default VAE / LongCLIP components when they were not serialized."""
        device = device or self._execution_device
        if self.vae is None:
            vae = AutoencoderKL.from_pretrained(VAE_ID)
            self.register_modules(vae=vae)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if self.tokenizer is None:
            self.register_modules(tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID))
        if self.text_encoder is None:
            self.register_modules(text_encoder=CLIPTextModel.from_pretrained(TOKENIZER_ID))
        self.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None = None, **kwargs):
        """Load a Diffusers folder. Hub ids default to this family's ``hub_subfolder``."""
        subfolder = kwargs.pop("subfolder", None)
        pretrained_model_name_or_path, subfolder = resolve_hub_load(
            pretrained_model_name_or_path, subfolder, cls.hub_subfolder
        )
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        pretrained_model_name_or_path: str,
        *,
        family: str | None = None,
        arch: str | None = None,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        checkpoints_root: str = "checkpoints",
        legacy: bool | None = None,
        load_aux: bool = True,
        vae: AutoencoderKL | None = None,
        text_encoder: CLIPTextModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        scheduler: FlowMatchEulerDiscreteScheduler | None = None,
        subfolder: str | None = None,
        **kwargs: Any,
    ) -> "TerraDiTPipelineBase":
        r"""
        Load a pipeline from a Diffusers folder, Hub id, release name, or a legacy checkpoint.

        Parameters:
            pretrained_model_name_or_path (`str`):
                Diffusers directory, Hub id (`BiliSakura/TerraDiT`), release name
                (`omega_xl`), or a `model.safetensors` / training `.pt` path.
            family / arch:
                Overrides used when constructing a transformer from a legacy checkpoint.
            torch_dtype (`torch.dtype`, *optional*):
                Cast floating modules after load.
            device (`str` or `torch.device`, *optional*):
                Device to move the pipeline to.
            checkpoints_root (`str`):
                Local cache for auto-downloaded release weights.
            legacy (`bool`, *optional*):
                Override the adaLN legacy flag stored in the release config.
            load_aux (`bool`, defaults to `True`):
                Load the SDXL VAE and LongCLIP when they are not part of the folder.
            vae / text_encoder / tokenizer / scheduler:
                Optional pre-constructed components (used by tests and conversion).
            subfolder (`str`, *optional*):
                Hub subfolder. Defaults to this class's ``hub_subfolder``
                (``TerraDiT-alpha`` / ``TerraDiT-sigma`` / ``TerraDiT-omega``).
            **kwargs:
                Forwarded to [`DiffusionPipeline.from_pretrained`] for Diffusers folders.

        Returns:
            A family pipeline (`TerraDiTAlphaPipeline`, `TerraDiTSigmaPipeline`, or
            `TerraDiTOmegaPipeline`).
        """
        family = family or cls.family
        path = pretrained_model_name_or_path
        if is_diffusers_pipeline_dir(path):
            pipe = cls.from_pretrained(path, torch_dtype=torch_dtype, **kwargs)
            if load_aux:
                pipe._ensure_aux(device)
            if device is not None:
                pipe.to(device)
            return pipe

        hub_id, hub_sub = resolve_hub_load(path, subfolder, cls.hub_subfolder)
        use_hub = (
            not os.path.exists(path)
            and not str(path).endswith((".pt", ".safetensors"))
            and (
                path in (DIFFUSERS_REPO, cls.hub_repo_id, "BiliSakura/TerraDiT")
                or (subfolder is not None and hub_id == DIFFUSERS_REPO)
                or (path not in MODELS and "/" in str(path))
            )
        )
        if use_hub:
            pipe = cls.from_pretrained(hub_id, subfolder=hub_sub, torch_dtype=torch_dtype, **kwargs)
            if load_aux:
                pipe._ensure_aux(device)
            if device is not None:
                pipe.to(device)
            return pipe

        transformer = TerraDiTTransformer2DModel.from_legacy_checkpoint(
            pretrained_model_name_or_path,
            family=family,
            arch=arch,
            checkpoints_root=checkpoints_root,
            legacy=legacy,
            torch_dtype=torch_dtype,
        )
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        if load_aux:
            if vae is None:
                vae = AutoencoderKL.from_pretrained(VAE_ID)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
            if text_encoder is None:
                text_encoder = CLIPTextModel.from_pretrained(TOKENIZER_ID)
        pipe = cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            family=family,
        )
        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)
        if device is not None:
            pipe.to(device)
        return pipe

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        clip_skip: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""
        Encode captions with LongCLIP and L2-normalize token / pooled embeddings.

        Parameters:
            prompt (`str` or `list[str]`, *optional*):
                Caption string(s). Ignored when `prompt_embeds` is provided.
            device (`torch.device`):
                Device of the returned embeddings.
            num_images_per_prompt (`int`, defaults to 1):
                Copies of each prompt embedding in the batch dimension.
            do_classifier_free_guidance (`bool`, defaults to `False`):
                If `True`, also encode `negative_prompt` (empty string when omitted).
            negative_prompt (`str` or `list[str]`, *optional*):
                Negative captions used only when guidance is enabled.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed token embeddings `(batch, seq, dim)`.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed pooled embeddings `(batch, dim)`. Required with `prompt_embeds`.
            negative_prompt_embeds / negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed unconditional embeddings.
            clip_skip (`int`, *optional*):
                Unused; accepted for Stable Diffusion call-signature compatibility.

        Returns:
            `tuple`:
                `(prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds)`.
        """
        if clip_skip is not None:
            logger.warning("`clip_skip` is ignored by TerraDiT (LongCLIP last-layer embeddings).")

        if prompt_embeds is None:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError("Passing `prompt` requires `text_encoder` and `tokenizer`.")
            if prompt is None:
                raise ValueError("Provide `prompt` or `prompt_embeds`.")
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            tok = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=CAPTION_MAX_LEN,
                truncation=True,
                return_tensors="pt",
            )
            out = self.text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device))
            prompt_embeds = F.normalize(out.last_hidden_state, dim=-1)
            pooled_prompt_embeds = F.normalize(out.pooler_output, dim=-1)
        else:
            if pooled_prompt_embeds is None:
                raise ValueError("`pooled_prompt_embeds` is required when `prompt_embeds` is passed.")
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        batch = prompt_embeds.shape[0]
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        neg_embeds = neg_pooled = None
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if self.text_encoder is None or self.tokenizer is None:
                    raise ValueError("Guidance with string prompts requires `text_encoder` and `tokenizer`.")
                if negative_prompt is None:
                    negatives = [""] * batch
                elif isinstance(negative_prompt, str):
                    negatives = [negative_prompt] * batch
                else:
                    if len(negative_prompt) != batch:
                        raise ValueError("`negative_prompt` length must match `prompt` / `prompt_embeds`.")
                    negatives = list(negative_prompt)
                tok = self.tokenizer(
                    negatives,
                    padding="max_length",
                    max_length=CAPTION_MAX_LEN,
                    truncation=True,
                    return_tensors="pt",
                )
                out = self.text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device))
                negative_prompt_embeds = F.normalize(out.last_hidden_state, dim=-1)
                negative_pooled_prompt_embeds = F.normalize(out.pooler_output, dim=-1)
            neg_embeds = negative_prompt_embeds.to(device).repeat_interleave(num_images_per_prompt, dim=0)
            neg_pooled = negative_pooled_prompt_embeds.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled

    def run_safety_checker(
        self,
        image: torch.Tensor | np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | np.ndarray, None]:
        r"""
        No-op safety checker (TerraDiT has no NSFW head).

        Parameters:
            image (`torch.Tensor` or `np.ndarray`):
                Decoded images.
            device (`torch.device`):
                Unused; kept for Stable Diffusion helper compatibility.
            dtype (`torch.dtype`):
                Unused; kept for Stable Diffusion helper compatibility.

        Returns:
            `tuple`:
                `(image, None)` — the `None` stands for absent NSFW flags.
        """
        return image, None

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        r"""
        Decode latents to a numpy image batch in `[0, 1]`.

        Parameters:
            latents (`torch.Tensor`):
                Denoised latents `(batch, 4, h, w)` in the SDXL-scaled space.

        Returns:
            `np.ndarray`:
                Images of shape `(batch, height, width, 3)`, float32 in `[0, 1]`.
        """
        deprecation_message = "Use `VaeImageProcessor.postprocess` via the pipeline `output_type` argument instead."
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)
        if self.vae is None:
            raise ValueError("VAE is required to decode latents.")
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator: torch.Generator | list[torch.Generator] | None, eta: float) -> dict[str, Any]:
        r"""
        Extra kwargs accepted by `scheduler.step` (eta / generator when supported).

        Parameters:
            generator (`torch.Generator` or list, *optional*):
                RNG forwarded when the scheduler's `step` accepts `generator`.
            eta (`float`):
                DDIM eta; ignored by flow-matching Euler.

        Returns:
            `dict`:
                Keyword arguments safe to splat into `scheduler.step`.
        """
        extra: dict[str, Any] = {}
        params = inspect.signature(self.scheduler.step).parameters
        if "eta" in params:
            extra["eta"] = eta
        if "generator" in params:
            extra["generator"] = generator
        return extra

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        height: int,
        width: int,
        callback_steps: int | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        r"""
        Validate generation arguments before the denoising loop.

        Parameters:
            prompt / negative_prompt:
                String captions; mutually exclusive with the matching `*_embeds`.
            height / width (`int`):
                Output pixel size. Must be positive multiples of `vae_scale_factor`.
            callback_steps (`int`, *optional*):
                Deprecated positive-int callback interval.
            prompt_embeds / negative_prompt_embeds / pooled_prompt_embeds:
                Precomputed embeddings. `prompt_embeds` requires `pooled_prompt_embeds`.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names forwarded to `callback_on_step_end`; must be a subset of
                `_callback_tensor_inputs`.
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"`height` and `width` must be positive, got {height}x{width}")
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got {height}x{width}"
            )
        native = int(getattr(self.transformer.config, "resolution", 256))
        if height != native or width != native:
            logger.warning(
                "Requested %sx%s but the transformer was trained at %sx%s; positional embeddings are not interpolated.",
                height,
                width,
                native,
                native,
            )

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` has to be a positive integer, got {callback_steps}")
        if callback_on_step_end_tensor_inputs is not None:
            extra = set(callback_on_step_end_tensor_inputs) - set(self._callback_tensor_inputs)
            if extra:
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has unknown names {sorted(extra)}; "
                    f"allowed: {self._callback_tensor_inputs}"
                )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise TypeError(f"`prompt` must be str or list[str], got {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError("`pooled_prompt_embeds` is required when `prompt_embeds` is provided.")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Sample or validate the initial noise (flow-matching time = 1).

        Parameters:
            batch_size (`int`):
                Number of images to generate.
            num_channels_latents (`int`):
                Transformer input channels (4 for SDXL latents).
            height / width (`int`):
                Pixel size; latent spatial size is divided by `vae_scale_factor`.
            dtype / device:
                Dtype and device of the returned tensor.
            generator (`torch.Generator` or list, *optional*):
                RNG for `randn_tensor`.
            latents (`torch.Tensor`, *optional*):
                User-supplied initial noise; moved to `device`/`dtype` when given.

        Returns:
            `torch.Tensor`:
                Latents of shape `(batch, channels, h / scale, w / scale)`.
        """
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape {tuple(latents.shape)}, expected {shape}")
            latents = latents.to(device=device, dtype=dtype)
        return latents

    def _repeat_cond(self, value: Any, batch_size: int) -> Any:
        if value is None or not torch.is_tensor(value):
            return value
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.repeat(batch_size, *([1] * (value.ndim - 1)))
        raise ValueError(f"Conditioning batch {value.shape[0]} does not match image batch {batch_size}")

    def _text_cond_kwargs(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "y": prompt_embeds.to(device=device, dtype=dtype),
            "y_pooled": pooled_prompt_embeds.to(device=device, dtype=dtype),
        }

    def _prepare_loc_embed(
        self,
        *,
        lat: float | None,
        lon: float | None,
        loc_embed: torch.Tensor | None,
        range_model: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if loc_embed is None:
            if range_model is None and lat is not None and lon is not None:
                range_model = load_range_model(device)
            loc_embed = loc_embed_from_latlon(range_model, lat, lon, device)
        loc_embed = loc_embed.to(device=device, dtype=dtype)
        if loc_embed.ndim == 1:
            loc_embed = loc_embed.unsqueeze(0)
        return self._repeat_cond(F.normalize(loc_embed.float(), dim=-1).to(dtype), batch_size)

    def _prepare_sigma_kwargs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        lat: float | None,
        lon: float | None,
        loc_embed: torch.Tensor | None,
        range_model: Any,
        points: list | None,
        point_prompts: torch.Tensor | None,
        pos: torch.Tensor | None,
        mask: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> dict[str, Any]:
        kwargs = self._text_cond_kwargs(prompt_embeds, pooled_prompt_embeds, device, dtype)
        kwargs["loc_embed"] = self._prepare_loc_embed(
            lat=lat, lon=lon, loc_embed=loc_embed, range_model=range_model,
            batch_size=batch_size, device=device, dtype=dtype,
        )
        if point_prompts is None:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError("Sigma point prompts as (x, y, tag) require `text_encoder` and `tokenizer`.")
            spec = {"caption": "", "points": points or [], "loc_embed": kwargs["loc_embed"][:1]}
            packed = build_conditioning("sigma", spec, self.text_encoder, self.tokenizer, device, num_images=1)
            point_prompts, pos, mask = packed["point_prompts"], packed["pos"], packed["mask"]
        kwargs["point_prompts"] = self._repeat_cond(point_prompts.to(device=device, dtype=dtype), batch_size)
        kwargs["pos"] = self._repeat_cond(pos.to(device=device), batch_size)
        kwargs["mask"] = self._repeat_cond(mask.to(device=device), batch_size)
        return kwargs

    def _prepare_omega_kwargs(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        lat: float | None,
        lon: float | None,
        loc_embed: torch.Tensor | None,
        range_model: Any,
        instances: list | None,
        inst_text_embed: torch.Tensor | None,
        polygon_xy: torch.Tensor | None,
        polygon_xy_mask: torch.Tensor | None,
        polyline_xy: torch.Tensor | None,
        polyline_xy_mask: torch.Tensor | None,
        bbox_xyxy: torch.Tensor | None,
        point_xy: torch.Tensor | None,
        format_mask: torch.Tensor | None,
        instance_mask: torch.Tensor | None,
        dropout_probs: list[float] | None,
        condition_type: str,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> dict[str, Any]:
        kwargs = self._text_cond_kwargs(prompt_embeds, pooled_prompt_embeds, device, dtype)
        kwargs["loc_embed"] = self._prepare_loc_embed(
            lat=lat, lon=lon, loc_embed=loc_embed, range_model=range_model,
            batch_size=batch_size, device=device, dtype=dtype,
        )
        if inst_text_embed is None:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError("Omega instances as dicts require `text_encoder` and `tokenizer`.")
            spec = {"caption": "", "instances": instances or [], "loc_embed": kwargs["loc_embed"][:1]}
            packed = build_conditioning(
                "omega", spec, self.text_encoder, self.tokenizer, device, num_images=1, dropout_probs=dropout_probs,
            )
            inst_text_embed = packed["inst_text_embed"]
            polygon_xy = packed["polygon_xy"]
            polygon_xy_mask = packed["polygon_xy_mask"]
            polyline_xy = packed["polyline_xy"]
            polyline_xy_mask = packed["polyline_xy_mask"]
            bbox_xyxy = packed["bbox_xyxy"]
            point_xy = packed["point_xy"]
            format_mask = packed["format_mask"]
            instance_mask = packed["instance_mask"]
        tensor_map = {
            "inst_text_embed": inst_text_embed,
            "polygon_xy": polygon_xy,
            "polygon_xy_mask": polygon_xy_mask,
            "polyline_xy": polyline_xy,
            "polyline_xy_mask": polyline_xy_mask,
            "bbox_xyxy": bbox_xyxy,
            "point_xy": point_xy,
            "format_mask": format_mask,
            "instance_mask": instance_mask,
        }
        for key, tensor in tensor_map.items():
            if tensor is None:
                raise ValueError(f"Omega conditioning is missing `{key}`.")
            kwargs[key] = self._repeat_cond(tensor.to(device=device), batch_size)
        kwargs["inst_text_embed"] = kwargs["inst_text_embed"].to(dtype=dtype)
        if dropout_probs is None:
            dropout_probs = OMEGA_CONDITION_DROPOUTS.get(condition_type)
        kwargs["dropout_probs"] = dropout_probs
        return kwargs

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def _warn_unused_sd_kwargs(self, cross_attention_kwargs, guidance_rescale, callback, callback_steps) -> None:
        if callback is not None or callback_steps is not None:
            deprecate(
                "callback / callback_steps",
                "1.0.0",
                "Use `callback_on_step_end` instead of `callback`/`callback_steps`.",
                standard_warn=False,
            )
        if cross_attention_kwargs:
            logger.warning("`cross_attention_kwargs` is unused by TerraDiT.")
        if guidance_rescale:
            logger.warning("`guidance_rescale` is unused by TerraDiT.")

    def _resolve_call_batch(
        self,
        prompt,
        prompt_embeds,
        height,
        width,
        guidance_scale,
        callback_on_step_end_tensor_inputs,
    ):
        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
        native = int(getattr(self.transformer.config, "resolution", 256))
        height = native if height is None else height
        width = native if width is None else width
        self._guidance_scale = guidance_scale
        self._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        return height, width, batch_size, callback_on_step_end_tensor_inputs

    @torch.no_grad()
    def _generate(
        self,
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        cond_kwargs: dict[str, Any],
        height: int,
        width: int,
        num_inference_steps: int,
        timesteps,
        sigmas,
        eta: float,
        generator,
        latents,
        output_type: str | None,
        return_dict: bool,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs: list[str],
        batch_size: int,
        num_images_per_prompt: int,
    ):
        device = self._execution_device
        if sigmas is None and timesteps is None:
            sigmas = paper_flow_sigmas(num_inference_steps)
        try:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                timesteps=timesteps,
                sigmas=sigmas,
            )
        except ValueError:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                timesteps=timesteps,
            )
        self._num_timesteps = len(timesteps)

        num_channels = int(getattr(self.transformer, "in_channels", 4))
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                sigma = self.scheduler.sigmas[i]
                latent_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                step_kwargs = dict(cond_kwargs)
                if self.do_classifier_free_guidance:
                    step_kwargs["y"] = torch.cat([cond_kwargs["y"], negative_prompt_embeds], dim=0)
                    step_kwargs["y_pooled"] = torch.cat([cond_kwargs["y_pooled"], negative_pooled_prompt_embeds], dim=0)
                    for key, value in list(step_kwargs.items()):
                        if key in ("y", "y_pooled", "dropout_probs") or not torch.is_tensor(value):
                            continue
                        if value.shape[0] == latents.shape[0]:
                            step_kwargs[key] = torch.cat([value, value], dim=0)
                timestep = torch.full((latent_input.shape[0],), sigma, device=device, dtype=latent_input.dtype)
                model_output = _unwrap_transformer_output(self.transformer(latent_input, timestep, **step_kwargs))
                if self.do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = model_output.chunk(2)
                    model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.scheduler.step(model_output, t, latents, return_dict=False, **extra_step_kwargs)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs:
                        latents = callback_outputs.get("latents", latents)
                        prompt_embeds = callback_outputs.get("prompt_embeds", prompt_embeds)
                        pooled_prompt_embeds = callback_outputs.get("pooled_prompt_embeds", pooled_prompt_embeds)

                if callback is not None and callback_steps and i % callback_steps == 0:
                    callback(i, t, latents)
                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            if self.vae is None:
                raise ValueError("VAE is required unless `output_type='latent'`.")
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image, _ = self.run_safety_checker(image, device, latents.dtype)
            do_denorm = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denorm)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
