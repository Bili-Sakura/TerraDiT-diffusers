# Hub component configs

Copied into every converted / skeleton family folder. Weight files are **not**
stored here — drop them next to these configs when publishing:

| folder | config | upload |
| --- | --- | --- |
| `vae/` | `config.json` (SDXL VAE) | `diffusion_pytorch_model.safetensors` from `stabilityai/sdxl-vae` |
| `text_encoder/` | `config.json` (LongCLIP text tower) | `model.safetensors` from `zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14` |
| `tokenizer/` | tokenizer JSON (+ `vocab.json` / `merges.txt` written by convert) | none (not a weight) |
| `scheduler/` | `scheduler_config.json` | none |
| `transformer/` | written from the checkpoint or skeleton defaults | `diffusion_pytorch_model.safetensors` |
| `geolocation_encoder/` | Σ / Ω only: RANGE+ code + config | `satclip-vit16-l40.ckpt` from `microsoft/SatCLIP-ViT16-L40` and `range_db.npz` from `mvrl/RANGE-database` (`range_db_large.npz`) |
