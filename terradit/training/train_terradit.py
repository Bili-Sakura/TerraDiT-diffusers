"""Unified TerraDiT trainer for the alpha / sigma / omega families.

One training loop; ``--family`` selects the dataset, the model's conditioning
flags, and the per-step model_kwargs. Reconciled from the original
train_text2image (alpha), train_points2image (sigma), and train_omega (omega).

Latents are read from precomputed SDXL moments by default, or computed on the fly
(``--vae-on-the-fly``) so training needs only imagery + the DINOv3 encoder. The
DINOv3 REPA target is always computed on the fly (frozen encoder).

Launch via terradit/train.py, e.g.:
    accelerate launch terradit/train.py --family omega --arch SiT-XL/2 \
        --data-root data/git10m --exp-name my-omega-run
"""
import os
import math
import argparse
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel
from torchvision.utils import make_grid

from terradit.models.sit import SiT_models
from terradit.models.encoders import load_encoders, encode_repa_targets
from terradit.training.loss import SILoss
from terradit.sampling.samplers import euler_sampler
from terradit.data.dataset import build_dataset, TOKENIZER_ID
from terradit.families import FAMILY_CONSTRUCT
from terradit.hf import (load_git10m, resolve_checkpoint, load_weights,
                       GIT10M_REPO, GIT10M_REVISION)

# SDXL latent scaling (shared by all families)
LATENTS_SCALE = torch.tensor([0.13025] * 4).view(1, 4, 1, 1)
LATENTS_BIAS = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1)


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * latents_scale + latents_bias


@torch.no_grad()
def vae_encode_moments(vae, raw_image):
    """uint8 [B,3,H,W] -> SDXL latent moments [B,8,H/8,W/8] (mean||std)."""
    img = raw_image.float() / 127.5 - 1.0
    posterior = vae.encode(img).latent_dist
    return torch.cat([posterior.mean, posterior.std], dim=1)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad_(flag)


def get_annealed_dropout(step, total_steps, start_p=0.5, end_p=0.1, warmup_ratio=0.33, cooldown_ratio=0.15):
    progress = step / max(total_steps, 1)
    if progress < warmup_ratio:
        p = start_p
    elif progress > (1 - cooldown_ratio):
        p = end_p
    else:
        anneal = (progress - warmup_ratio) / (1 - warmup_ratio - cooldown_ratio)
        p = end_p + (start_p - end_p) * 0.5 * (1 + math.cos(math.pi * anneal))
    return [p, p, p]


def _encode_caption(clip, ids, attn):
    out = clip(ids, attn)
    y = F.normalize(out.last_hidden_state, dim=-1)
    y_pooled = F.normalize(out.pooler_output, dim=-1)
    return y, y_pooled


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@torch.no_grad()
def sample_grid(model, vae, viz_kwargs, latents_scale, latents_bias, latent_size, device, num_steps):
    """Generate images from fixed conditioning (for periodic wandb logging). fp32, no CFG."""
    was_training = model.training
    model.eval()
    y = viz_kwargs["y"]
    xT = torch.randn(y.shape[0], 4, latent_size, latent_size, device=device)
    kw = {k: v for k, v in viz_kwargs.items() if k != "y"}
    samples = euler_sampler(model, xT, y, num_steps=num_steps, cfg_scale=0.0,
                            path_type="linear", **kw).to(torch.float32)
    imgs = vae.decode((samples - latents_bias) / latents_scale).sample
    if was_training:
        model.train()
    return ((imgs + 1) / 2).clamp(0, 1)


def build_model_kwargs(family, batch, clip, device, step, total_steps):
    """Unpack a family batch, run the text encoder, and return (raw_image, latent, kwargs)."""
    batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
    if family == "alpha":
        raw_image, latent, cap_ids, cap_attn = batch
        y, y_pooled = _encode_caption(clip, cap_ids, cap_attn)
        kwargs = dict(y=y, y_pooled=y_pooled)

    elif family == "sigma":
        raw_image, latent, cap_ids, cap_attn, loc_embed, p_ids, p_attn, pos, mask = batch
        y, y_pooled = _encode_caption(clip, cap_ids, cap_attn)
        L = p_ids.shape[-1]
        p = clip(p_ids.view(-1, L), p_attn.view(-1, L)).pooler_output
        p = F.normalize(p.view(raw_image.shape[0], -1, p.shape[-1]), dim=-1)
        kwargs = dict(y=y, y_pooled=y_pooled, loc_embed=loc_embed.float(),
                      point_prompts=p, pos=pos, mask=mask)

    elif family == "omega":
        (raw_image, latent, cap_ids, cap_attn, loc_embed, polygon_xy, polygon_xy_mask,
         polyline_xy, polyline_xy_mask, bbox_xyxy, point_xy, format_mask, instance_mask,
         inst_tag_ids, inst_tag_attn) = batch
        y, y_pooled = _encode_caption(clip, cap_ids, cap_attn)
        B, N, Lt = inst_tag_ids.shape
        inst = clip(inst_tag_ids.view(B * N, Lt), inst_tag_attn.view(B * N, Lt)).pooler_output
        inst_text_embed = F.normalize(inst, dim=-1).view(B, N, -1)
        kwargs = dict(y=y, y_pooled=y_pooled, loc_embed=loc_embed.float(), mask=cap_attn,
                      inst_text_embed=inst_text_embed, polygon_xy=polygon_xy,
                      polygon_xy_mask=polygon_xy_mask, polyline_xy=polyline_xy,
                      polyline_xy_mask=polyline_xy_mask, bbox_xyxy=bbox_xyxy, point_xy=point_xy,
                      format_mask=format_mask, instance_mask=instance_mask,
                      dropout_probs=get_annealed_dropout(step, total_steps))
    else:
        raise ValueError(family)
    return raw_image, latent, kwargs


def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir,
                                          logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with=(None if args.report_to == "none" else args.report_to),
                              project_config=project_config)
    device = accelerator.device
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed + accelerator.process_index)

    latent_size = args.resolution // 8
    latents_scale = LATENTS_SCALE.to(device)
    latents_bias = LATENTS_BIAS.to(device)

    # REPA encoder (frozen DINOv3) -> z_dims
    if args.use_repa:
        encoders, encoder_types, _ = load_encoders(args.enc_type, device, args.dinov3_weights)
        z_dims = [encoders[0].embed_dim]
    else:
        encoders, encoder_types, z_dims = [], [], [0]

    construct = dict(FAMILY_CONSTRUCT[args.family])  # condition_type/geolocation/point_prompts/omega/legacy
    if args.legacy is not None:                       # explicit override of the family default
        construct["legacy"] = args.legacy
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.arch](
        input_size=latent_size, num_classes=args.num_classes, use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims, encoder_depth=args.encoder_depth, geolocation_dim=args.loc_dim,
        omega_attn=args.omega_attn, use_repa=args.use_repa,
        **construct, **block_kwargs,
    ).to(device)

    # warm-start from an existing checkpoint (e.g. fine-tune alpha->sigma, or legacy fix):
    # loads model weights strict=False (new modules stay randomly initialized), fresh optimizer.
    if args.init_from:
        # accepts a release name (alpha_xl -> auto-download), a safetensors dir/file, or a .pt
        init_path = resolve_checkpoint(args.init_from)
        sd, _ = load_weights(init_path, drop_training_only=False)
        sd = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        if accelerator.is_main_process:
            new = [k for k in res.missing_keys if not k.startswith("projectors")]
            print(f"[init-from] {args.init_from}: {len(new)} new (random) params, "
                  f"{len(res.unexpected_keys)} unused ckpt keys")

    # optional partial fine-tune: train only parameters whose names match
    if args.freeze_except:
        for name, p in model.named_parameters():
            p.requires_grad = any(s in name for s in args.freeze_except)
        if accelerator.is_main_process:
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[freeze] training only params matching {args.freeze_except}: {n:,} params")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval(); requires_grad(vae, False)
    clip = CLIPTextModel.from_pretrained(TOKENIZER_ID).to(device).eval()
    requires_grad(clip, False)

    loss_fn = SILoss(prediction=args.prediction, path_type=args.path_type, encoders=encoders,
                     accelerator=accelerator, latents_scale=latents_scale,
                     latents_bias=latents_bias, weighting=args.weighting)

    # data (imagery from the pinned Git-10M snapshot; everything else under --data-root)
    hf = load_git10m(args.hf_cache_dir, repo_id=args.hf_repo_id, revision=args.hf_revision)
    dataset = build_dataset(args.family, args.data_root, hf_dataset=hf,
                            vae_on_the_fly=args.vae_on_the_fly, split="train",
                            max_samples=args.max_samples)
    if not args.vae_on_the_fly:
        probe = os.path.join(args.data_root, "latents", dataset.metadata[0]["img_name"] + ".npy")
        if not os.path.exists(probe):
            raise FileNotFoundError(
                f"no precomputed latent at {probe}. Run scripts/encode_latents.py first, "
                "or pass --vae-on-the-fly to encode in the training loop.")
    # --batch-size is the EFFECTIVE (global) batch, split across processes (matches REPA/omega)
    local_batch_size = max(1, args.batch_size // accelerator.num_processes)
    loader = DataLoader(dataset, batch_size=local_batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    ckpt_dir = os.path.join(args.output_dir, args.exp_name, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    # resume: --resume picks the latest checkpoint in this experiment; --resume-step picks one
    global_step = 0
    resume_step = args.resume_step
    if args.resume and resume_step == 0:
        found = sorted(int(f[:-3]) for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f[:-3].isdigit()) \
            if os.path.isdir(ckpt_dir) else []
        resume_step = found[-1] if found else 0
        if accelerator.is_main_process and not found:
            print(f"[resume] no checkpoints in {ckpt_dir}; starting fresh")
    if resume_step > 0:
        ckpt = torch.load(os.path.join(ckpt_dir, f"{resume_step:07d}.pt"),
                          map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
        ema.load_state_dict(ckpt["ema"], strict=False)
        if "opt" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["opt"])
            except Exception as e:  # e.g. resuming a model+EMA-only final checkpoint
                if accelerator.is_main_process:
                    print(f"[resume] optimizer state not restored ({type(e).__name__}); fresh optimizer")
        global_step = ckpt["steps"]
        if accelerator.is_main_process:
            print(f"[resume] {args.exp_name} from step {global_step}")

    # logging + a fixed batch for periodic sample images
    log_on = args.report_to != "none"
    viz_kwargs = viz_gt = None
    if log_on:
        accelerator.init_trackers(project_name="TerraDiT", config=vars(args),
                                  init_kwargs={"wandb": {"name": args.exp_name}})
        if accelerator.is_main_process:
            from torch.utils.data import default_collate
            n_viz = min(16, len(dataset))
            vb = default_collate([dataset[i] for i in range(n_viz)])
            vimg, vlat, viz_kwargs = build_model_kwargs(args.family, vb, clip, device, 0, args.max_train_steps)
            if "dropout_probs" in viz_kwargs:
                viz_kwargs["dropout_probs"] = None  # show all instance conditioning in the viz
            if not args.vae_on_the_fly and torch.is_tensor(vlat) and vlat.numel() > 0:
                m = (vlat.squeeze(1) if vlat.dim() == 5 else vlat).to(device)
                zgt = sample_posterior(m, latents_scale=latents_scale, latents_bias=latents_bias)
                viz_gt = ((vae.decode((zgt - latents_bias) / latents_scale).sample + 1) / 2).clamp(0, 1)

    model.train()
    progress = tqdm(total=args.max_train_steps, initial=global_step,
                    disable=not accelerator.is_main_process)
    done = False
    while not done:
        for batch in loader:
            raw_image, latent, model_kwargs = build_model_kwargs(
                args.family, batch, clip, device, global_step, args.max_train_steps)

            # latent moments: precomputed or on-the-fly VAE encode
            if args.vae_on_the_fly:
                moments = vae_encode_moments(vae, raw_image)
            else:
                moments = latent.squeeze(1) if latent.dim() == 5 else latent
            x = sample_posterior(moments, latents_scale=latents_scale, latents_bias=latents_bias)

            zs = None
            if args.use_repa:
                with torch.no_grad(), accelerator.autocast():
                    zs = encode_repa_targets(encoders, encoder_types, raw_image)

            grad_norm = None
            with accelerator.accumulate(model):
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs, repa=args.use_repa)
                total = loss.mean() + args.proj_coeff * (proj_loss.mean() if torch.is_tensor(proj_loss) else proj_loss)
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(model))

            global_step += 1
            progress.update(1)

            if global_step % args.log_every == 0:
                logs = {"loss": float(loss.mean()),
                        "proj_loss": float(proj_loss.mean()) if torch.is_tensor(proj_loss) else 0.0}
                if grad_norm is not None:
                    logs["grad_norm"] = float(grad_norm)
                if accelerator.is_main_process:
                    progress.set_postfix(**logs)
                if log_on:
                    accelerator.log(logs, step=global_step)

            # periodic sample images (main process only)
            if (log_on and viz_kwargs is not None
                    and (global_step == 1 or global_step % args.sampling_steps == 0)):
                import wandb
                imgs = sample_grid(accelerator.unwrap_model(model), vae, viz_kwargs,
                                   latents_scale, latents_bias, latent_size, device, args.sampling_num_steps)
                grids = {"samples": wandb.Image(array2grid(imgs))}
                if viz_gt is not None:
                    grids["gt"] = wandb.Image(array2grid(viz_gt))
                accelerator.log(grids, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0 and accelerator.is_main_process:
                torch.save({"model": accelerator.unwrap_model(model).state_dict(),
                            "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                            "args": args, "steps": global_step},
                           os.path.join(ckpt_dir, f"{global_step:07d}.pt"))

            if global_step >= args.max_train_steps:
                done = True
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({"model": accelerator.unwrap_model(model).state_dict(), "ema": ema.state_dict(),
                    "args": args, "steps": global_step}, os.path.join(ckpt_dir, f"{global_step:07d}.pt"))
    if log_on:
        accelerator.end_training()


def parse_args(input_args=None):
    p = argparse.ArgumentParser(description="Unified TerraDiT trainer")
    # family + architecture
    p.add_argument("--family", required=True, choices=["alpha", "sigma", "omega"])
    p.add_argument("--arch", type=str, default="SiT-XL/2")
    p.add_argument("--omega-attn", type=str, default="GALA")
    p.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=None,
                   help="override family legacy flag, wired to the model (default = family value)")
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start model+EMA from this checkpoint (strict=False, fresh optimizer, step 0)")
    p.add_argument("--freeze-except", nargs="+", default=None,
                   help="freeze all params except those whose name contains one of these substrings")
    # logging / io
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest checkpoint in <output-dir>/<exp-name>/checkpoints")
    p.add_argument("--resume-step", type=int, default=0, help="resume from a specific step instead")
    p.add_argument("--max-samples", type=int, default=None,
                   help="use only the first N metadata rows (subset fine-tuning / smoke tests)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpointing-steps", type=int, default=5000)
    p.add_argument("--report-to", type=str, default="wandb", help="'wandb' or 'none'")
    p.add_argument("--sampling-steps", type=int, default=5000, help="log sample images every N steps")
    p.add_argument("--sampling-num-steps", type=int, default=50, help="euler steps for the logged samples")
    # model
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-repa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loc-dim", type=int, default=1280)
    p.add_argument("--enc-type", type=str, default="dinov3-vit-l")
    p.add_argument("--dinov3-weights", type=str, default=None,
                   help="DINOv3 SAT-493M .pth for REPA (or set $TERRADIT_DINOV3_WEIGHTS)")
    # data
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--hf-repo-id", type=str, default=GIT10M_REPO)
    p.add_argument("--hf-revision", type=str, default=GIT10M_REVISION,
                   help="Git-10M snapshot; hf_idx in the metadata indexes this revision")
    p.add_argument("--hf-cache-dir", type=str, default=None)
    p.add_argument("--vae-on-the-fly", action="store_true",
                   help="compute SDXL latents in-loop instead of reading precomputed .npy")
    p.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    # precision / optim
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # loss
    p.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--prediction", type=str, default="v", choices=["v"])
    p.add_argument("--cfg-prob", type=float, default=0.1)
    p.add_argument("--proj-coeff", type=float, default=0.5)
    p.add_argument("--weighting", type=str, default="uniform")
    args = p.parse_args(input_args)
    if args.use_repa:  # fail fast, before any model/data is loaded
        from terradit.models.encoders import resolve_dinov3_weights
        args.dinov3_weights = resolve_dinov3_weights(args.dinov3_weights)
    return args


if __name__ == "__main__":
    main()
