"""Convert training checkpoints (.pt with model/ema/opt) into release weights.

Writes ``<out>/<name>/model.safetensors`` (EMA only, fp16, REPA projectors dropped) and
``<out>/<name>/config.json`` for each requested model. Source checkpoints are given with ``--source name=path``.

    python scripts/export_weights.py --models omega_base --out release/weights \
        --source omega_base=exps/omega-base/checkpoints/0400000.pt --verify

``--verify`` reloads the exported file next to the original EMA weights and checks the
predicted velocity on one random batch (GPU recommended; runs on CPU for omega_base).
"""
import os
import sys
import json
import time
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import MODELS, training_checkpoint_to_release, save_release, load_weights

SOURCES = {}   # name -> training checkpoint path; pass with --source name=path


def verify(name, src_path, out_path, config, device):
    """Same random input through original-EMA fp32 model and exported fp16 weights (cast to fp32)."""
    from terradit.generation import build_inference_model
    torch.manual_seed(0)
    fam, arch = config["family"], config["arch"]
    ref = build_inference_model(fam, arch, src_path, device, legacy=config["legacy"],
                                loc_dim=config["loc_dim"], omega_attn=config["omega_attn"])
    new = build_inference_model(fam, arch, out_path, device, legacy=config["legacy"],
                                loc_dim=config["loc_dim"], omega_attn=config["omega_attn"])
    B, D = 2, 768
    x = torch.randn(B, 4, 32, 32, device=device)
    t = torch.rand(B, device=device)
    y = torch.nn.functional.normalize(torch.randn(B, 144, D, device=device), dim=-1)
    yp = torch.nn.functional.normalize(torch.randn(B, D, device=device), dim=-1)
    kw = dict(y=y, y_pooled=yp)
    if fam in ("sigma", "omega"):
        kw["loc_embed"] = torch.nn.functional.normalize(torch.randn(B, config["loc_dim"], device=device), dim=-1)
    if fam == "sigma":   # shapes/dtypes as SigmaDataset (max_points=50, mask 0=valid 1=pad)
        P = 50
        mask = torch.ones(B, P, dtype=torch.long, device=device); mask[:, :12] = 0
        kw.update(point_prompts=torch.nn.functional.normalize(torch.randn(B, P, D, device=device), dim=-1),
                  pos=torch.randint(0, 256, (B, P, 2), device=device).float(), mask=mask)
    if fam == "omega":   # shapes/dtypes as OmegaDataset (max_instances=64, max_points=64)
        N, P = 64, 64
        inst_mask = torch.zeros(B, N, device=device); inst_mask[:, :6] = 1.0
        fmt = torch.zeros(B, N, 4, device=device); fmt[:, :6] = 1.0
        pg_mask = torch.zeros(B, N, P, dtype=torch.bool, device=device); pg_mask[:, :6, :8] = True
        kw.update(inst_text_embed=torch.nn.functional.normalize(torch.randn(B, N, D, device=device), dim=-1),
                  polygon_xy=torch.rand(B, N, P, 2, device=device) * 256, polygon_xy_mask=pg_mask,
                  polyline_xy=torch.rand(B, N, P, 2, device=device) * 256, polyline_xy_mask=pg_mask.clone(),
                  bbox_xyxy=torch.tensor([[[10., 10., 100., 100.]] * N] * B, device=device),
                  point_xy=torch.rand(B, N, 2, device=device) * 256,
                  format_mask=fmt, instance_mask=inst_mask)
    with torch.no_grad():
        a = ref(x, t, **kw)
        b = new(x, t, **kw)
    a = a[0] if isinstance(a, (tuple, list)) else a
    b = b[0] if isinstance(b, (tuple, list)) else b
    diff = (a.float() - b.float()).abs()
    print(f"[verify] {name}: max|dv|={diff.max():.3e} mean|dv|={diff.mean():.3e} "
          f"(ref std {a.float().std():.3f})")
    return float(diff.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"], help=f"{sorted(MODELS)} or 'all'")
    ap.add_argument("--out", default="release/weights")
    ap.add_argument("--source", nargs="*", default=[], metavar="NAME=PATH",
                    help="override a source checkpoint, e.g. omega_xl=/path/to/1570000.pt")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    names = sorted(MODELS) if args.models == ["all"] else args.models
    sources = dict(SOURCES)
    for kv in args.source:
        k, v = kv.split("=", 1)
        sources[k] = v
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    summary = {}
    for name in names:
        spec = MODELS[name]
        src = sources.get(name)
        if not src or not os.path.isfile(src):
            print(f"[skip] {name}: no source checkpoint (pass --source {name}=/path/to.pt)")
            continue
        t0 = time.time()
        print(f"[export] {name} <- {src}")
        sd, config = training_checkpoint_to_release(src, name=name, family=spec["family"],
                                                    arch=spec["arch"], dtype=dtype)
        out_dir = os.path.join(args.out, name)
        out_path = save_release(sd, config, out_dir)
        size = os.path.getsize(out_path) / 1e9
        print(f"[export] {name}: {config['num_parameters']/1e6:.1f}M params, {size:.2f} GB, "
              f"{len(sd)} tensors, {time.time()-t0:.0f}s -> {out_path}")
        summary[name] = {"path": out_path, "size_gb": round(size, 3), **config}
        if args.verify:
            summary[name]["verify_max_abs_diff"] = verify(name, src, out_path, config, args.device)
        del sd

    os.makedirs(args.out, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out, "export_summary.json"), "w"), indent=2)
    print(f"[export] summary -> {os.path.join(args.out, 'export_summary.json')}")


if __name__ == "__main__":
    main()
