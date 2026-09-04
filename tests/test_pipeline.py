"""Diffusers-compatibility checks for TerraDiT family pipelines (CPU, random weights)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from safetensors.torch import save_file

from terradit.hf import DIFFUSERS_REPO, FAMILY_HUB_SUBFOLDER, VARIANT_HUB_SUBFOLDER, is_diffusers_pipeline_dir
from terradit.models.sit import interpolate_2d_pos_embed
from terradit.models.transformer import TerraDiTTransformer2DModel
from terradit.pipelines.pipeline_common import (
    flow_match_euler_denoise,
    paper_flow_sigmas,
    pipeline_class_for_family,
    resolve_hub_load,
)
from terradit.pipelines.pipeline_terradit_alpha import TerraDiTAlphaPipeline
from terradit.pipelines.pipeline_terradit_omega import TerraDiTOmegaPipeline
from terradit.pipelines.pipeline_terradit_sigma import TerraDiTSigmaPipeline
import importlib.util

_CONVERT = os.path.join(os.path.dirname(__file__), "..", "scripts", "convert_to_diffusers.py")
_spec = importlib.util.spec_from_file_location("convert_to_diffusers", _CONVERT)
_convert_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_convert_mod)
convert_checkpoint = _convert_mod.convert_checkpoint

FAMILY_CLS = {
    "alpha": TerraDiTAlphaPipeline,
    "sigma": TerraDiTSigmaPipeline,
    "omega": TerraDiTOmegaPipeline,
}


def _tiny_transformer(family: str = "alpha", input_size: int = 4) -> TerraDiTTransformer2DModel:
    return TerraDiTTransformer2DModel(
        family=family,
        arch="SiT-B/2",
        input_size=input_size,
        sample_size=input_size,
        resolution=input_size * 8,
        encoder_depth=2,
        use_repa=False,
    )


def _tiny_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        latent_channels=4,
        sample_size=32,
        scaling_factor=0.13025,
    )


def _prompt_embeds(batch: int = 1, seq: int = 8, dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.nn.functional.normalize(torch.randn(batch, seq, dim), dim=-1)
    yp = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)
    return y, yp


def _pipeline(family: str = "alpha"):
    transformer = _tiny_transformer(family)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    return FAMILY_CLS[family](transformer=transformer, scheduler=scheduler, vae=_tiny_vae())


def test_paper_sigmas_match_released_linspace():
    steps = 100
    expected = torch.linspace(1.0, 0.0, steps + 1)[:-1].tolist()
    got = paper_flow_sigmas(steps)
    assert np.allclose(got, expected, atol=1e-6)


def test_flow_match_euler_matches_manual_step():
    torch.manual_seed(0)
    model = _tiny_transformer("alpha")
    model.eval()
    y, yp = _prompt_embeds()
    latents = torch.randn(1, 4, 4, 4)
    steps = 3
    out = flow_match_euler_denoise(model, latents.clone(), num_inference_steps=steps, y=y, y_pooled=yp)

    x = latents.to(torch.float32)
    t_steps = torch.linspace(1.0, 0.0, steps + 1)
    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            t = torch.full((1,), t_cur, dtype=x.dtype)
            v = model(x, t, y=y, y_pooled=yp, return_dict=False)[0]
            x = x + (t_next - t_cur) * v
    assert torch.allclose(out, x, atol=1e-4, rtol=1e-4)


def test_interpolate_pos_embed_identity_and_resize():
    torch.manual_seed(0)
    pos = torch.randn(1, 4, 8)
    assert torch.equal(interpolate_2d_pos_embed(pos, 2, 2), pos)
    resized = interpolate_2d_pos_embed(pos, 4, 2)
    assert resized.shape == (1, 8, 8)


def test_variable_resolution_pos_interpolation():
    pipe = _pipeline("alpha")
    y, yp = _prompt_embeds()
    out = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=64,
        width=64,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape == (1, 4, 8, 8)
    wide = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=64,
        width=32,
        num_inference_steps=2,
        output_type="latent",
    )
    assert wide.images.shape == (1, 4, 8, 4)


def test_call_latent_output_and_reproducibility():
    pipe = _pipeline("alpha")
    y, yp = _prompt_embeds()
    height = width = pipe.transformer.config.resolution
    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    a = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=height,
        width=width,
        num_inference_steps=2,
        generator=gen1,
        output_type="latent",
        return_dict=True,
    )
    b = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=height,
        width=width,
        num_inference_steps=2,
        generator=gen2,
        output_type="latent",
        return_dict=False,
    )
    assert isinstance(a, ImagePipelineOutput)
    assert a.images.shape == (1, 4, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    assert torch.allclose(a.images, b[0])


def test_scheduler_swap_heun():
    pipe = _pipeline("alpha")
    pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipe.scheduler.config)
    y, yp = _prompt_embeds()
    height = width = pipe.transformer.config.resolution
    out = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=height,
        width=width,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape[0] == 1


@pytest.mark.parametrize("family,cls", list(FAMILY_CLS.items()))
def test_save_load_roundtrip_each_family(tmp_path, family, cls):
    pipe = _pipeline(family)
    out_dir = tmp_path / family
    pipe.save_pretrained(out_dir)
    assert (out_dir / "model_index.json").is_file()
    assert (out_dir / "scheduler" / "scheduler_config.json").is_file()
    extra = [p for p in (out_dir / "scheduler").iterdir() if p.name != "scheduler_config.json"]
    assert extra == []
    index = json.loads((out_dir / "model_index.json").read_text())
    assert index["_class_name"] == cls.__name__
    loaded = cls.from_pretrained(out_dir)
    assert loaded.transformer.config.family == family
    assert isinstance(loaded, cls)
    y, yp = _prompt_embeds()
    height = width = loaded.transformer.config.resolution
    call_kw = dict(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=height,
        width=width,
        num_inference_steps=2,
        output_type="latent",
    )
    if family != "alpha":
        call_kw["loc_embed"] = torch.nn.functional.normalize(torch.randn(1, 1280), dim=-1)
    if family == "sigma":
        call_kw.update(
            point_prompts=torch.nn.functional.normalize(torch.randn(1, 2, 768), dim=-1),
            pos=torch.tensor([[[10, 12], [40, 50]]]),
            mask=torch.zeros(1, 2),
        )
    if family == "omega":
        n, p = 4, 64
        call_kw.update(
            inst_text_embed=torch.nn.functional.normalize(torch.randn(1, n, 768), dim=-1),
            polygon_xy=torch.rand(1, n, p, 2) * 256,
            polygon_xy_mask=torch.zeros(1, n, p, dtype=torch.bool),
            polyline_xy=torch.rand(1, n, p, 2) * 256,
            polyline_xy_mask=torch.zeros(1, n, p, dtype=torch.bool),
            bbox_xyxy=torch.tensor([[[10.0, 10.0, 40.0, 40.0]] * n]),
            point_xy=torch.rand(1, n, 2) * 256,
            format_mask=torch.ones(1, n, 4),
            instance_mask=torch.ones(1, n),
            condition_type="point",
        )
    out = loaded(**call_kw)
    assert out.images.ndim == 4


def test_omega_tensor_conditioning():
    pipe = _pipeline("omega")
    y, yp = _prompt_embeds()
    n, p = 4, 64
    kw = dict(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        loc_embed=torch.nn.functional.normalize(torch.randn(1, 1280), dim=-1),
        inst_text_embed=torch.nn.functional.normalize(torch.randn(1, n, 768), dim=-1),
        polygon_xy=torch.rand(1, n, p, 2) * 256,
        polygon_xy_mask=torch.zeros(1, n, p, dtype=torch.bool),
        polyline_xy=torch.rand(1, n, p, 2) * 256,
        polyline_xy_mask=torch.zeros(1, n, p, dtype=torch.bool),
        bbox_xyxy=torch.tensor([[[10.0, 10.0, 40.0, 40.0]] * n]),
        point_xy=torch.rand(1, n, 2) * 256,
        format_mask=torch.ones(1, n, 4),
        instance_mask=torch.ones(1, n),
        height=pipe.transformer.config.resolution,
        width=pipe.transformer.config.resolution,
        num_inference_steps=2,
        output_type="latent",
        condition_type="point",
    )
    out = pipe(**kw)
    assert out.images.shape[0] == 1


def test_convert_legacy_safetensors(tmp_path):
    transformer = _tiny_transformer("alpha")
    src = tmp_path / "legacy"
    src.mkdir()
    sd = {k: v.contiguous().half() if v.is_floating_point() else v for k, v in transformer.state_dict().items()}
    save_file(sd, src / "model.safetensors")
    config = {
        "family": "alpha",
        "arch": "SiT-B/2",
        "legacy": False,
        "loc_dim": 1280,
        "omega_attn": "GALA",
        "encoder_depth": 2,
        "resolution": 32,
        "num_classes": 1000,
    }
    (src / "config.json").write_text(json.dumps(config))
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="alpha", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    assert is_diffusers_pipeline_dir(str(dest))
    assert (dest / "pipeline.py").is_file()
    assert (dest / "transformer" / "config.json").is_file()
    sched_files = os.listdir(dest / "scheduler")
    assert sched_files == ["scheduler_config.json"]
    index = json.loads((dest / "model_index.json").read_text())
    assert index["_class_name"] == "TerraDiTAlphaPipeline"
    pipeline_src = (dest / "pipeline.py").read_text()
    assert "TerraDiTAlphaPipeline" in pipeline_src
    loaded = TerraDiTAlphaPipeline.from_pretrained(str(dest), vae=_tiny_vae())
    y, yp = _prompt_embeds()
    out = loaded(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=32,
        width=32,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape == (1, 4, 4, 4)


def test_convert_repo_layout_writes_hub_subfolder(tmp_path):
    transformer = _tiny_transformer("sigma")
    src = tmp_path / "legacy"
    src.mkdir()
    sd = {k: v.contiguous().half() if v.is_floating_point() else v for k, v in transformer.state_dict().items()}
    save_file(sd, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps({
        "family": "sigma",
        "arch": "SiT-B/2",
        "legacy": False,
        "loc_dim": 1280,
        "omega_attn": "GALA",
        "encoder_depth": 2,
        "resolution": 32,
        "num_classes": 1000,
    }))
    dest = tmp_path / "TerraDiT"
    out = convert_checkpoint(
        str(src), str(dest), family="sigma", arch="SiT-B/2",
        include_aux=False, dtype=torch.float32, repo_layout=True,
    )
    assert os.path.basename(out) == FAMILY_HUB_SUBFOLDER["sigma"]
    index = json.loads(open(os.path.join(out, "model_index.json")).read())
    assert index["_class_name"] == "TerraDiTSigmaPipeline"


def test_resolve_hub_load_defaults_family_subfolder():
    assert resolve_hub_load(DIFFUSERS_REPO, None, "TerraDiT-Alpha-XL") == (DIFFUSERS_REPO, "TerraDiT-Alpha-XL")
    assert resolve_hub_load(DIFFUSERS_REPO, "TerraDiT-Omega-B", "TerraDiT-Omega-XL") == (
        DIFFUSERS_REPO, "TerraDiT-Omega-B",
    )
    assert resolve_hub_load("TerraDiT-Alpha-XL", None, "TerraDiT-Sigma-XL") == (
        DIFFUSERS_REPO, "TerraDiT-Alpha-XL",
    )
    assert VARIANT_HUB_SUBFOLDER["omega_base"] == "TerraDiT-Omega-B"
    assert FAMILY_HUB_SUBFOLDER == {
        "alpha": "TerraDiT-Alpha-XL",
        "sigma": "TerraDiT-Sigma-XL",
        "omega": "TerraDiT-Omega-XL",
    }


def test_pipeline_init_has_no_family_arg():
    import inspect

    params = inspect.signature(TerraDiTAlphaPipeline.__init__).parameters
    assert "family" not in params
    assert not hasattr(TerraDiTAlphaPipeline, "from_checkpoint")


def test_pipeline_class_for_family():
    assert pipeline_class_for_family("alpha") is TerraDiTAlphaPipeline
    assert pipeline_class_for_family("sigma") is TerraDiTSigmaPipeline
    assert pipeline_class_for_family("omega") is TerraDiTOmegaPipeline
    with pytest.raises(ValueError):
        pipeline_class_for_family("beta")


def test_old_inference_modules_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__("terradit.generation")
    with pytest.raises(ModuleNotFoundError):
        __import__("terradit.sampling.samplers")
