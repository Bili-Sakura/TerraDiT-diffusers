"""REPA alignment encoder for TerraDiT.

The released models all use the satellite-pretrained DINOv3 ViT-L/16 (SAT-493M) as the
REPA target encoder. The encoder is frozen and run on the fly during training to
produce the patch-token targets the diffusion features are aligned to. It is only
needed for training with ``--use-repa`` (the default); inference never loads it.

The weights are distributed by Meta under the DINOv3 license and are not
redistributed here. Request them at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
and pass the file via ``--dinov3-weights`` (or set ``TERRADIT_DINOV3_WEIGHTS``).
"""
import os
import torch
from torchvision.transforms import Normalize

# satellite normalization statistics (DINOv3 SAT-493M)
SAT_DEFAULT_MEAN = (0.430, 0.411, 0.296)
SAT_DEFAULT_STD = (0.213, 0.156, 0.143)

DINOV3_HUB_REPO = "facebookresearch/dinov3"
DINOV3_HUB_MODEL = "dinov3_vitl16"
DINOV3_FILENAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
ENV_VAR = "TERRADIT_DINOV3_WEIGHTS"


def resolve_dinov3_weights(path=None):
    """CLI path > $TERRADIT_DINOV3_WEIGHTS > error with instructions."""
    path = path or os.environ.get(ENV_VAR)
    if path and os.path.isfile(path):
        return path
    raise FileNotFoundError(
        "DINOv3 SAT-493M weights are required for REPA training but were not found"
        + (f" at {path!r}" if path else "") + ".\n"
        f"  1. Request the weights from Meta (DINOv3 license): "
        "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
        f"  2. Pass --dinov3-weights /path/to/{DINOV3_FILENAME} or export {ENV_VAR}=...\n"
        "  Or train without representation alignment via --no-use-repa (changes the recipe).")


def load_encoders(enc_type, device, weights_path=None, resolution=256):
    """Load the REPA encoder(s). Returns (encoders, encoder_types, architectures).

    Only ``dinov3-vit-l`` is supported in the public release. The architecture is
    pulled from torch.hub (network access or a populated hub cache is required).
    """
    assert "dinov3" in enc_type, (
        f"only dinov3 encoders are supported in this release (got {enc_type!r})")
    weights_path = resolve_dinov3_weights(weights_path)
    encoder = torch.hub.load(DINOV3_HUB_REPO, DINOV3_HUB_MODEL, weights=weights_path)
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return [encoder], ["dinov3-vit-l"], ["vit"]


def preprocess_raw_image(x, enc_type="dinov3-vit-l"):
    """Normalize a uint8 image batch for the DINOv3 encoder."""
    resolution = x.shape[-1]
    x = x / 255.0
    x = Normalize(SAT_DEFAULT_MEAN, SAT_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 256 * (resolution // 256), mode="bicubic")
    return x


def encode_repa_targets(encoders, encoder_types, raw_image):
    """Run the frozen encoder(s) to get REPA target patch tokens (list of [B, T, D])."""
    zs = []
    for encoder, etype in zip(encoders, encoder_types):
        x = preprocess_raw_image(raw_image, etype)
        z = encoder.forward_features(x)
        if "dinov3" in etype or "dinov2" in etype:
            z = z["x_norm_patchtokens"]
        zs.append(z)
    return zs
