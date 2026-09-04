"""Upload exported weights and derived data to the Hugging Face Hub (MVRL org).

Requires `huggingface-cli login` with write access to the MVRL organisation.

    # weights: release/weights/<name>/{model.safetensors,config.json} + docs/hf_cards/<name>.md
    python scripts/upload_hf.py weights --models all --private  # -> MVRL/TerraDiT/<name>/... (private)
    python scripts/upload_hf.py weights --models omega_base --dry-run
    python scripts/upload_hf.py weights --cards-only                # refresh READMEs only

    # data: release/data/** (mirrors the data_root layout) + docs/hf_cards/dataset.md
    python scripts/upload_hf.py data                            # -> MVRL/TerraDiT-data

    # collection: one Hub page grouping the model repo, the dataset repo, and both papers
    python scripts/upload_hf.py collection                      # -> huggingface.co/collections/MVRL/terradit-...

Large files go through the Hub's chunked uploader (hf_transfer if installed: pip install
hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from terradit.hf import MODELS, MODEL_REPO, DATA_REPO, WEIGHTS_FILE, CONFIG_FILE

CARDS = os.path.join(os.path.dirname(__file__), "..", "docs", "hf_cards")


def upload_weights(api, names, src_root, repo_id, dry_run, private=False, cards_only=False):
    if not dry_run:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    for name in ([] if cards_only else names):
        d = os.path.join(src_root, name)
        for f in (WEIGHTS_FILE, CONFIG_FILE):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                sys.exit(f"missing {p}; run scripts/export_weights.py --models {name}")
            print(f"[upload] {p} -> {repo_id}/{name}/{f} ({os.path.getsize(p)/1e9:.2f} GB)")
            if not dry_run:
                api.upload_file(path_or_fileobj=p, path_in_repo=f"{name}/{f}", repo_id=repo_id,
                                repo_type="model", commit_message=f"add {name}")
    # repo-level README (the collection card) + per-model cards
    for card, dest in [("README.md", "README.md")] + [(f"{n}.md", f"{n}/README.md") for n in names]:
        p = os.path.join(CARDS, card)
        if os.path.isfile(p):
            print(f"[upload] {p} -> {repo_id}/{dest}")
            if not dry_run:
                api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id=repo_id,
                                repo_type="model", commit_message=f"card: {dest}")
        else:
            print(f"[upload] (no card at {p})")


def upload_data(api, src_root, repo_id, dry_run, private=False):
    if not os.path.isdir(src_root):
        sys.exit(f"{src_root} not found; stage files there in the data_root layout first")
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(src_root) for f in fs)
    print(f"[upload] {src_root} ({total/1e9:.1f} GB) -> {repo_id}")
    for r, _, fs in os.walk(src_root):
        for f in sorted(fs):
            p = os.path.join(r, f)
            print(f"  {os.path.relpath(p, src_root):55s} {os.path.getsize(p)/1e9:7.2f} GB")
    if dry_run:
        return
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_large_folder(folder_path=src_root, repo_id=repo_id, repo_type="dataset")
    card = os.path.join(CARDS, "dataset.md")
    if os.path.isfile(card):
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo_id,
                        repo_type="dataset", commit_message="dataset card")


PAPERS = ["2606.31029", "2603.02172"]   # TerraDiT-Omega, TerraDiT


def create_collection(api, title, namespace, model_repo, data_repo, private, dry_run):
    """One Hub collection grouping the weights, the data, and both papers."""
    items = [(model_repo, "model"), (data_repo, "dataset")] + [(p, "paper") for p in PAPERS]
    print(f"[collection] {namespace}/{title}: " + ", ".join(f"{t}:{i}" for i, t in items))
    if dry_run:
        return
    col = api.create_collection(title=title, namespace=namespace, private=private, exists_ok=True,
                                description="TerraDiT / TerraDiT-Omega: diffusion transformers for "
                                            "satellite image synthesis with text, geolocation, points, "
                                            "and geospatial primitives (ECCV 2026).")
    for item_id, item_type in items:
        try:
            api.add_collection_item(col.slug, item_id=item_id, item_type=item_type, exists_ok=True)
        except Exception as e:
            print(f"[collection] could not add {item_type} {item_id}: {e}")
    print(f"[collection] https://huggingface.co/collections/{col.slug}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["weights", "data", "collection"])
    ap.add_argument("--collection-title", default="TerraDiT")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--weights-root", default="release/weights")
    ap.add_argument("--data-root", default="release/data")
    ap.add_argument("--model-repo", default=MODEL_REPO)
    ap.add_argument("--data-repo", default=DATA_REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the repo as private (flip to public in the Hub settings later)")
    ap.add_argument("--cards-only", action="store_true", help="weights: upload only the README cards (docs/hf_cards)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    if not args.dry_run:
        who = api.whoami()
        print(f"[hf] logged in as {who['name']}; orgs: {[o['name'] for o in who.get('orgs', [])]}")
    if args.what == "weights":
        names = sorted(MODELS) if args.models == ["all"] else args.models
        upload_weights(api, names, args.weights_root, args.model_repo, args.dry_run, args.private, args.cards_only)
    elif args.what == "data":
        upload_data(api, args.data_root, args.data_repo, args.dry_run, args.private)
    else:
        create_collection(api, args.collection_title, args.model_repo.split("/")[0],
                          args.model_repo, args.data_repo, args.private, args.dry_run)


if __name__ == "__main__":
    main()
