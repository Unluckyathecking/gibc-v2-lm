"""Push the release folder to the Hugging Face Hub as a public model repo.

  HF_TOKEN=... uv run python upload_to_hf.py --whoami          # check the token, print account only
  HF_TOKEN=... uv run python upload_to_hf.py --folder hf_release
  uv run modal run modal_axiomic.py::upload                     # same, from Modal with the `huggingface` secret

The token is read from HF_TOKEN and never printed.
"""
import argparse
import os

REPO_ID = "Unluckyexe/gibc-v2-lm"
IGNORE = ["__pycache__/*", "*.pyc", ".DS_Store"]
REQUIRED = ["README.md", "config.json", "configuration_gibc.py", "modeling_gibc.py",
            "model.safetensors", "final.pt", "tokenizer.json", "tokenizer_config.json"]


def whoami(token: str) -> str:
    from huggingface_hub import HfApi
    return HfApi(token=token).whoami()["name"]


def upload(folder: str, repo_id: str = REPO_ID, token: str | None = None) -> str:
    from huggingface_hub import HfApi
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(folder, f))]
    if missing:
        raise SystemExit(f"{folder} is missing {missing}; run scripts/build_hf_release.py first")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model",
                             ignore_patterns=IGNORE, commit_message="gibc-v2-lm: main_vres_anneal final weights")
    return info.commit_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="hf_release")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--whoami", action="store_true", help="only check the token")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    name = whoami(token)
    print(f"authenticated as: {name}")
    if a.whoami:
        return
    if name != a.repo_id.split("/")[0]:
        raise SystemExit(f"token belongs to {name}, not {a.repo_id.split('/')[0]}")
    print(upload(a.folder, a.repo_id, token))


if __name__ == "__main__":
    main()
