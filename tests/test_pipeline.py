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
from terradit.models.geolocation import TerraDiTGeolocationModel
from terradit.models.sit import interpolate_2d_pos_embed
from terradit.models.transformer import TerraDiTTransformer2DModel
from terradit.pipelines.pipeline_common import (
    flow_match_euler_denoise,
    paper_flow_sigmas,
    resolve_hub_load,
)
from terradit.pipelines.pipeline_terradit import pipeline_class_for_family
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
    kw["height"] = kw["width"] = 64
    out64 = pipe(**kw)
    assert out64.images.shape == (1, 4, 8, 8)


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
    assert not (dest / "pipeline_common.py").exists()
    assert (dest / "transformer" / "config.json").is_file()
    assert (dest / "transformer" / "transformer_terradit_alpha.py").is_file()
    assert not (dest / "transformer" / "sit.py").exists()
    assert (dest / "vae" / "config.json").is_file()
    assert (dest / "text_encoder" / "config.json").is_file()
    assert (dest / "tokenizer" / "tokenizer_config.json").is_file()
    sched_files = os.listdir(dest / "scheduler")
    assert sched_files == ["scheduler_config.json"]
    index = json.loads((dest / "model_index.json").read_text())
    assert index["_class_name"] == ["pipeline", "TerraDiTAlphaPipeline"]
    assert index["transformer"] == ["transformer_terradit_alpha", "TerraDiTTransformer2DModel"]
    pipeline_src = (dest / "pipeline.py").read_text()
    assert "TerraDiTAlphaPipeline" in pipeline_src
    assert "class TerraDiTPipelineBase" in pipeline_src
    loaded = TerraDiTAlphaPipeline.from_pretrained(
        str(dest), vae=_tiny_vae(), text_encoder=None, tokenizer=None, trust_remote_code=True,
    )
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
    assert index["_class_name"] == ["pipeline", "TerraDiTSigmaPipeline"]
    assert index["transformer"] == ["transformer_terradit_sigma", "TerraDiTTransformer2DModel"]
    assert "geolocation_encoder" not in index
    assert (tmp_path / "TerraDiT" / FAMILY_HUB_SUBFOLDER["sigma"] / "transformer" / "transformer_terradit_sigma.py").is_file()
    assert (tmp_path / "TerraDiT" / FAMILY_HUB_SUBFOLDER["sigma"] / "geolocation_encoder" / "modeling_geolocation.py").is_file()
    assert (tmp_path / "TerraDiT" / FAMILY_HUB_SUBFOLDER["sigma"] / "geolocation_encoder" / "config.json").is_file()


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


def _write_tiny_aux_weights(dest) -> None:
    """Drop tiny VAE / LongCLIP weights so DiffusionPipeline.from_pretrained can load the folder."""
    from transformers import CLIPTextConfig, CLIPTextModel

    dest = os.fspath(dest)
    _tiny_vae().save_pretrained(os.path.join(dest, "vae"))
    text_cfg = CLIPTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=49408,
        max_position_embeddings=16,
    )
    CLIPTextModel(text_cfg).save_pretrained(os.path.join(dest, "text_encoder"))


def _write_tiny_legacy(tmp_path, family="alpha"):
    transformer = _tiny_transformer(family)
    src = tmp_path / "legacy"
    src.mkdir()
    sd = {k: v.contiguous().half() if v.is_floating_point() else v for k, v in transformer.state_dict().items()}
    save_file(sd, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps({
        "family": family,
        "arch": "SiT-B/2",
        "legacy": False,
        "loc_dim": 1280,
        "omega_attn": "GALA",
        "encoder_depth": 2,
        "resolution": 32,
        "num_classes": 1000,
    }))
    return src


@pytest.mark.parametrize(
    "family,expected_py",
    [
        ("alpha", ["pipeline.py", "transformer/transformer_terradit_alpha.py"]),
        (
            "sigma",
            [
                "geolocation_encoder/modeling_geolocation.py",
                "pipeline.py",
                "transformer/transformer_terradit_sigma.py",
            ],
        ),
    ],
)
def test_converted_folder_has_no_terradit_imports(tmp_path, family, expected_py):
    src = _write_tiny_legacy(tmp_path, family=family)
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family=family, arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    import re

    py_files = list(dest.rglob("*.py"))
    assert py_files
    import_re = re.compile(r"^\s*(?:from terradit|import terradit|from \.)\b", re.MULTILINE)
    for path in py_files:
        text = path.read_text()
        assert import_re.search(text) is None, f"{path} still has package/relative imports"
    names = sorted(p.relative_to(dest).as_posix() for p in py_files)
    assert names == expected_py


def test_converted_folder_loads_via_diffusers_custom_code(tmp_path):
    from diffusers import DiffusionPipeline

    src = _write_tiny_legacy(tmp_path)
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="alpha", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    _write_tiny_aux_weights(dest)
    pipe = DiffusionPipeline.from_pretrained(str(dest), trust_remote_code=True)
    assert pipe.__class__.__name__ == "TerraDiTAlphaPipeline"
    y, yp = _prompt_embeds()
    out = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        height=32,
        width=32,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape == (1, 4, 4, 4)


def test_converted_folder_loads_without_terradit_on_path(tmp_path):
    import subprocess
    import sys

    src = _write_tiny_legacy(tmp_path)
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="alpha", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    _write_tiny_aux_weights(dest)
    script = (
        "import os, sys\n"
        "sys.path = [p for p in sys.path if p and 'workspace' not in os.path.abspath(p)]\n"
        "try:\n"
        "    import terradit\n"
        "except ImportError:\n"
        "    terradit = None\n"
        "else:\n"
        "    raise SystemExit(f'terradit still importable from {terradit.__file__}')\n"
        "from diffusers import DiffusionPipeline\n"
        f"pipe = DiffusionPipeline.from_pretrained({str(dest)!r}, trust_remote_code=True, local_files_only=True)\n"
        "assert pipe.__class__.__name__ == 'TerraDiTAlphaPipeline'\n"
        "assert pipe.transformer.config.family == 'alpha'\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_hub_skeleton_writes_family_transformer_module(tmp_path):
    from terradit.pipelines.hub_export import write_repo_skeletons, write_self_contained_repo

    write_self_contained_repo(tmp_path / "alpha", "alpha", arch="SiT-XL/2")
    assert (tmp_path / "alpha" / "pipeline.py").is_file()
    assert (tmp_path / "alpha" / "transformer" / "transformer_terradit_alpha.py").is_file()
    assert (tmp_path / "alpha" / "vae" / "config.json").is_file()
    assert (tmp_path / "alpha" / "text_encoder" / "config.json").is_file()
    index = json.loads((tmp_path / "alpha" / "model_index.json").read_text())
    assert index["transformer"] == ["transformer_terradit_alpha", "TerraDiTTransformer2DModel"]
    assert index["vae"] == ["diffusers", "AutoencoderKL"]
    assert index["text_encoder"] == ["transformers", "CLIPTextModel"]
    assert "geolocation_encoder" not in index
    assert not (tmp_path / "alpha" / "geolocation_encoder").exists()
    assert not list((tmp_path / "alpha").rglob("*.safetensors"))

    write_self_contained_repo(tmp_path / "sigma", "sigma", arch="SiT-XL/2")
    assert (tmp_path / "sigma" / "geolocation_encoder" / "modeling_geolocation.py").is_file()
    assert (tmp_path / "sigma" / "geolocation_encoder" / "config.json").is_file()
    sigma_index = json.loads((tmp_path / "sigma" / "model_index.json").read_text())
    assert "geolocation_encoder" not in sigma_index
    geo_cfg = json.loads((tmp_path / "sigma" / "geolocation_encoder" / "config.json").read_text())
    assert geo_cfg["_class_name"] == "TerraDiTGeolocationModel"
    assert geo_cfg["embedding_dim"] == 1280
    assert geo_cfg["nnet_hidden"] == 512
    assert geo_cfg["satclip_filename"] == "model.safetensors"

    paths = write_repo_skeletons(tmp_path / "repo")
    names = {os.path.basename(p) for p in paths}
    assert names == set(FAMILY_HUB_SUBFOLDER.values()) | {"TerraDiT-Omega-B"}
    assert (tmp_path / "repo" / "TerraDiT-Omega-B" / "transformer" / "transformer_terradit_omega.py").is_file()
    assert (tmp_path / "repo" / "TerraDiT-Omega-B" / "geolocation_encoder" / "modeling_geolocation.py").is_file()
    assert not (tmp_path / "repo" / "TerraDiT-Alpha-XL" / "geolocation_encoder").exists()
    omega_cfg = json.loads((tmp_path / "repo" / "TerraDiT-Omega-B" / "transformer" / "config.json").read_text())
    assert omega_cfg["arch"] == "SiT-B/2"
    assert omega_cfg["family"] == "omega"


def _tiny_geolocation(**kwargs) -> TerraDiTGeolocationModel:
    defaults = dict(
        embedding_dim=16,
        satclip_dim=8,
        image_dim=8,
        legendre_polys=2,
        nnet_hidden=8,
        nnet_layers=1,
    )
    defaults.update(kwargs)
    return TerraDiTGeolocationModel(**defaults)


def test_geolocation_encoder_zeros_without_database():
    model = _tiny_geolocation()
    out = model(torch.tensor([[-90.31, 38.65]]))
    assert out.shape == (1, 16)
    assert torch.equal(out, torch.zeros_like(out))
    via_kwargs = model(latitudes=38.65, longitudes=-90.31)
    assert torch.equal(via_kwargs, out)


def test_geolocation_encoder_retrieves_with_dummy_database(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    model = _tiny_geolocation()
    locs = np.array([[-90.31, 38.65], [0.0, 0.0], [2.35, 48.86]], dtype=np.float32)
    archive = tmp_path / "range_db.npz"
    np.savez(
        archive,
        satclip_embeddings=np.random.randn(3, 8).astype(np.float32),
        image_embeddings=np.random.randn(3, 8).astype(np.float32),
        locs=locs,
    )
    model.attach_database(archive)
    out = model(torch.tensor([[-90.31, 38.65]]))
    assert out.shape == (1, 16)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_geolocation_encoder_save_load_roundtrip(tmp_path):
    model = _tiny_geolocation()
    dest = tmp_path / "geo"
    model.save_pretrained(dest)
    loaded = TerraDiTGeolocationModel.from_pretrained(dest)
    assert loaded.config.embedding_dim == 16
    assert loaded.config.legendre_polys == 2
    assert not loaded._has_database


def test_geolocation_loads_mvrl_style_safetensors(tmp_path):
    torch.manual_seed(0)
    model = _tiny_geolocation()
    sd = {f"location.{key}": value.contiguous() for key, value in model.satclip.state_dict().items()}
    weights = tmp_path / "model.safetensors"
    save_file(sd, weights)
    loaded = TerraDiTGeolocationModel.from_location_weights(weights)
    assert loaded.config.nnet_hidden == 8
    assert loaded.config.satclip_dim == 8
    assert loaded.config.legendre_polys == 2
    assert loaded.config.nnet_layers == 1
    folder = tmp_path / "geo"
    folder.mkdir()
    save_file(sd, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps({
        "_class_name": "TerraDiTGeolocationModel",
        "embedding_dim": 16,
        "satclip_dim": 8,
        "image_dim": 8,
        "legendre_polys": 2,
        "nnet_hidden": 8,
        "nnet_layers": 1,
    }))
    via_folder = TerraDiTGeolocationModel.from_pretrained(folder)
    assert via_folder.config.embedding_dim == 16
    assert via_folder.config.satclip_dim == 8
    for key in model.satclip.state_dict():
        assert torch.allclose(loaded.satclip.state_dict()[key], model.satclip.state_dict()[key])


def test_geolocation_infers_vit16_l40_hparams(tmp_path):
    sd = {
        "location.nnet.layers.0.weight": torch.zeros(512, 1600),
        "location.nnet.layers.0.bias": torch.zeros(512),
        "location.nnet.layers.1.weight": torch.zeros(512, 512),
        "location.nnet.layers.1.bias": torch.zeros(512),
        "location.nnet.last_layer.weight": torch.zeros(256, 512),
        "location.nnet.last_layer.bias": torch.zeros(256),
    }
    path = tmp_path / "model.safetensors"
    save_file(sd, path)
    loaded = TerraDiTGeolocationModel.from_location_weights(path)
    assert loaded.config.nnet_hidden == 512
    assert loaded.config.nnet_layers == 2
    assert loaded.config.legendre_polys == 40
    assert loaded.config.satclip_dim == 256
    assert loaded.config.embedding_dim == 1280
    alias = TerraDiTGeolocationModel.from_satclip_checkpoint(path)
    assert alias.config.nnet_hidden == 512


def test_sigma_pipeline_uses_geolocation_encoder():
    geo = TerraDiTGeolocationModel(
        embedding_dim=1280,
        satclip_dim=256,
        image_dim=1024,
        legendre_polys=2,
        nnet_hidden=8,
        nnet_layers=1,
    )
    pipe = FAMILY_CLS["sigma"](
        transformer=_tiny_transformer("sigma"),
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0),
        vae=_tiny_vae(),
        geolocation_encoder=geo,
    )
    y, yp = _prompt_embeds()
    out = pipe(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        lat=38.65,
        lon=-90.31,
        point_prompts=torch.nn.functional.normalize(torch.randn(1, 2, 768), dim=-1),
        pos=torch.tensor([[[10, 12], [40, 50]]]),
        mask=torch.zeros(1, 2),
        height=pipe.transformer.config.resolution,
        width=pipe.transformer.config.resolution,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape[0] == 1


def test_converted_sigma_loads_with_optional_geolocation(tmp_path):
    src = _write_tiny_legacy(tmp_path, family="sigma")
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="sigma", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    loaded = TerraDiTSigmaPipeline.from_pretrained(
        str(dest),
        vae=_tiny_vae(),
        text_encoder=None,
        tokenizer=None,
        geolocation_encoder=None,
        trust_remote_code=True,
    )
    assert loaded.geolocation_encoder is None
    assert (dest / "geolocation_encoder" / "modeling_geolocation.py").is_file()
    y, yp = _prompt_embeds()
    out = loaded(
        prompt_embeds=y,
        pooled_prompt_embeds=yp,
        loc_embed=torch.nn.functional.normalize(torch.randn(1, 1280), dim=-1),
        point_prompts=torch.nn.functional.normalize(torch.randn(1, 2, 768), dim=-1),
        pos=torch.tensor([[[10, 12], [40, 50]]]),
        mask=torch.zeros(1, 2),
        height=32,
        width=32,
        num_inference_steps=2,
        output_type="latent",
    )
    assert out.images.shape == (1, 4, 4, 4)


def test_write_geolocation_encoder_from_safetensors(tmp_path):
    from terradit.pipelines.hub_export import write_geolocation_encoder, write_model_index

    geo = _tiny_geolocation()
    sd = {f"location.{key}": value.contiguous() for key, value in geo.satclip.state_dict().items()}
    weights = tmp_path / "model.safetensors"
    save_file(sd, weights)
    dest = tmp_path / "sigma"
    write_geolocation_encoder(dest, "sigma", satclip_ckpt=weights)
    write_model_index(dest, "sigma")
    assert (dest / "geolocation_encoder" / "modeling_geolocation.py").is_file()
    loaded = TerraDiTGeolocationModel.from_pretrained(dest / "geolocation_encoder")
    assert loaded.config.nnet_hidden == 8
    assert loaded.config.satclip_dim == 8
    index = json.loads((dest / "model_index.json").read_text())
    assert index["geolocation_encoder"] == ["modeling_geolocation", "TerraDiTGeolocationModel"]
    src = _write_tiny_legacy(tmp_path, family="sigma")
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="sigma", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    geo = _tiny_geolocation()
    geo.save_pretrained(dest / "geolocation_encoder")
    # Keep the Hub modeling file next to the tiny weights.
    from terradit.pipelines.hub_export import write_geolocation_encoder

    write_geolocation_encoder(dest, "sigma")
    geo.save_pretrained(dest / "geolocation_encoder")
    loaded = TerraDiTSigmaPipeline.from_pretrained(
        str(dest),
        vae=_tiny_vae(),
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    )
    assert loaded.geolocation_encoder is not None
    assert loaded.geolocation_encoder.config.embedding_dim == 16


def test_converted_sigma_attaches_mvrl_safetensors(tmp_path):
    src = _write_tiny_legacy(tmp_path, family="sigma")
    dest = tmp_path / "diffusers"
    convert_checkpoint(str(src), str(dest), family="sigma", arch="SiT-B/2", include_aux=False, dtype=torch.float32)
    geo = _tiny_geolocation()
    sd = {f"location.{key}": value.contiguous() for key, value in geo.satclip.state_dict().items()}
    save_file(sd, dest / "geolocation_encoder" / "model.safetensors")
    (dest / "geolocation_encoder" / "config.json").write_text(json.dumps({
        "_class_name": "TerraDiTGeolocationModel",
        "embedding_dim": 16,
        "satclip_dim": 8,
        "image_dim": 8,
        "legendre_polys": 2,
        "nnet_hidden": 8,
        "nnet_layers": 1,
        "database_filename": "range_db.npz",
        "satclip_filename": "model.safetensors",
    }))
    from terradit.pipelines.hub_export import write_model_index

    write_model_index(dest, "sigma")
    loaded = TerraDiTSigmaPipeline.from_pretrained(
        str(dest),
        vae=_tiny_vae(),
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    )
    assert loaded.geolocation_encoder is not None
    assert loaded.geolocation_encoder.config.satclip_dim == 8
    assert loaded.geolocation_encoder.config.nnet_hidden == 8
