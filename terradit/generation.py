"""Shared inference helpers for the TerraDiT demos.

Reproduces the released sampling: build the family model (use_repa=False), load the
EMA weights, encode conditioning with LongCLIP, run euler_sampler (cfg_scale=0, no
classifier-free guidance), and decode with the SDXL VAE.
"""
import os
import torch
import torch.nn.functional as F
import torchvision

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel

from terradit.models.sit import SiT_models
from terradit.families import FAMILY_CONSTRUCT
from terradit.data.dataset import TOKENIZER_ID, CAPTION_MAX_LEN, TAG_MAX_LEN
from terradit.hf import resolve_checkpoint, load_weights, MODELS

LATENTS_SCALE = torch.tensor([0.13025] * 4).view(1, 4, 1, 1)
LATENTS_BIAS = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1)


def build_inference_model(family, arch, ckpt, device, *, legacy=None, loc_dim=None,
                          omega_attn=None, encoder_depth=None, resolution=None,
                          num_classes=None, checkpoints_root="checkpoints"):
    """Construct a family model and load inference weights.

    ``ckpt`` is a release name (``omega_xl``; auto-downloaded from the Hub), a directory
    holding ``model.safetensors`` + ``config.json``, or a path to a ``.safetensors`` /
    training ``.pt`` file. Construction flags default to the release config when one is
    present; explicit arguments override it. ``family``/``arch`` may be ``None`` when the
    config supplies them.
    """
    path = resolve_checkpoint(ckpt, checkpoints_root)
    sd, config = load_weights(path)
    config = config or {}
    if ckpt in MODELS:  # name given: fill family/arch from the registry as a fallback
        config = {**MODELS[ckpt], **config}

    family = family or config.get("family")
    arch = arch or config.get("arch")
    if family is None or arch is None:
        raise ValueError(f"family/arch not given and not found in a config next to {path}")
    legacy = config.get("legacy", False) if legacy is None else legacy
    loc_dim = loc_dim if loc_dim is not None else config.get("loc_dim", 1280)
    omega_attn = omega_attn or config.get("omega_attn", "GALA")
    encoder_depth = encoder_depth if encoder_depth is not None else config.get("encoder_depth", 8)
    resolution = resolution if resolution is not None else config.get("resolution", 256)
    num_classes = num_classes if num_classes is not None else config.get("num_classes", 1000)

    construct = dict(FAMILY_CONSTRUCT[family])  # condition_type/geolocation/point_prompts/omega/legacy
    construct["legacy"] = legacy
    model = SiT_models[arch](
        input_size=resolution // 8, num_classes=num_classes, use_cfg=False, z_dims=[0],
        encoder_depth=encoder_depth, geolocation_dim=loc_dim,
        omega_attn=omega_attn, use_repa=False, fused_attn=True, qk_norm=False,
        **construct,
    ).to(device)
    sd = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("projectors")]
    print(f"[load] {os.path.basename(path)} ({family}, {arch}, legacy={legacy}): "
          f"missing={len(missing)} unexpected={len(result.unexpected_keys)}")
    if missing:
        print(f"[load]   first missing: {missing[:8]}")
    if result.unexpected_keys:
        print(f"[load]   first unexpected: {result.unexpected_keys[:8]}")
    model.eval()
    return model


def resolve_sample_index(ds, img_name=None, index=0):
    """Resolve a dataset row: by exact img_name if given, else the raw index.

    Lets the sigma/omega demos target the *same physical tile* across families,
    which otherwise sit at different rows (sigma uses the 330k cities subset).
    """
    if img_name is None:
        return index
    for i, row in enumerate(ds.metadata):
        if row.get("img_name") == img_name:
            return i
    raise ValueError(f"img_name {img_name!r} not found in this family's metadata")


def load_text_vae(device):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    clip = CLIPTextModel.from_pretrained(TOKENIZER_ID).to(device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device).eval()
    return tokenizer, clip, vae


@torch.no_grad()
def caption_embeds(clip, tokenizer, prompts, device):
    """Encode caption text(s) -> (y [B,T,D], y_pooled [B,D], attn_mask [B,T])."""
    tok = tokenizer(prompts, padding="max_length", max_length=CAPTION_MAX_LEN,
                    truncation=True, return_tensors="pt")
    out = clip(tok.input_ids.to(device), tok.attention_mask.to(device))
    y = F.normalize(out.last_hidden_state, dim=-1)
    y_pooled = F.normalize(out.pooler_output, dim=-1)
    return y, y_pooled, tok.attention_mask.to(device)


@torch.no_grad()
def tag_pooled_embeds(clip, ids, attn):
    """Pooled, normalized CLIP embeddings for tag tokens. ids/attn: [M, TAG_MAX_LEN]."""
    out = clip(ids, attn).pooler_output
    return F.normalize(out, dim=-1)


@torch.no_grad()
def decode_latents(vae, samples, device):
    scale = LATENTS_SCALE.to(device).to(vae.dtype)
    bias = LATENTS_BIAS.to(device).to(vae.dtype)
    imgs = vae.decode((samples.to(vae.dtype) - bias) / scale).sample
    return ((imgs + 1) / 2).clamp(0, 1)


def save_images(imgs, paths):
    for img, path in zip(imgs, paths):
        torchvision.utils.save_image(img, path)
