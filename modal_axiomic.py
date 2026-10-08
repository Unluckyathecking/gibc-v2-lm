"""Modal app "gibc-axiomic": Axiomic Labs Open SLM Leaderboard evals on the HF release folder.

  uv run modal run modal_axiomic.py::arithmark        # official ArithMark-3 and -2 scripts
  uv run modal run modal_axiomic.py::lm_eval_release  # lm-eval CLI on the release folder
  uv run modal run modal_axiomic.py::upload --whoami-only  # check the `huggingface` secret
  uv run modal run modal_axiomic.py::upload            # build + push to Unluckyexe/gibc-v2-lm

Each job rebuilds the Hugging Face release folder (scripts/build_hf_release.py) from the
run's final.pt on the gibc-runs volume, so what is scored is exactly what gets uploaded,
loaded through AutoModelForCausalLM with trust_remote_code. The ArithMark scripts are the
unmodified files from the AxiomicLabs dataset repos, pinned to a commit. No HF token is
attached: every dataset used here is public.
"""
import json
import os
import shutil
import subprocess
import sys

import modal

from gibc.modal_common import VOLUMES, hf_secret, image

app = modal.App("gibc-axiomic")
GPU = "L4"
RUN = "main_vres_anneal"
REL = "/tmp/hf_release"
img = (image
       .add_local_file("hf_release/configuration_gibc.py", "/root/rel_src/configuration_gibc.py")
       .add_local_file("hf_release/modeling_gibc.py", "/root/rel_src/modeling_gibc.py")
       .add_local_file("scripts/build_hf_release.py", "/root/build_hf_release.py")
       .add_local_file("hf_release/README.md", "/root/rel_src/README.md")
       .add_local_file("upload_to_hf.py", "/root/upload_to_hf.py"))

# name -> (dataset repo, commit, script, data file)
ARITHMARK = {
    "arithmark-3": ("AxiomicLabs/Arithmark-3.0", "6f6e59dd9b7e2c63455f7af7f838f9ecc3d0a746",
                    "bencharithmark-3.py", "arithmark-3.jsonl"),
    "arithmark-2": ("AxiomicLabs/ArithMark-2.0", "5bd986c7305dcccec88432d630aae1ac9280a71f",
                    "benchmark_arithmark-2.0.py", "arithmark_2.0.jsonl"),
}
LB_TASKS = "hellaswag,arc_easy,arc_challenge,piqa"
# model_args: "default" is what a plain `lm_eval --model hf` call does (fp32, the tokenizer's
# own BOS post-processor); "ours" mirrors gibc/evaluate.py (explicit BOS, bf16 autocast).
LM_EVAL_ARGS = {"default": "", "ours": ",add_bos_token=True,mixed_precision_dtype=bfloat16"}


def build_release(run: str = RUN, copy_pt: bool = False) -> str:
    """Release folder (card + code + config + safetensors + tokenizer, .pt optional) at REL."""
    from gibc.configs import RUNS_DIR, tokenizer_path
    sys.path.insert(0, "/root")
    from build_hf_release import build
    shutil.rmtree(REL, ignore_errors=True)
    shutil.copytree("/root/rel_src", REL)
    build(f"{RUNS_DIR}/{run}/final.pt", tokenizer_path("tok32k"), REL, copy_pt=copy_pt)
    return REL


def sh(cmd: list[str], cwd: str) -> str:
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(p.stdout[-4000:], p.stderr[-2000:], sep="\n", flush=True)
    p.check_returncode()
    return p.stdout


@app.function(image=img, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_arithmark(run: str = RUN) -> dict:
    from huggingface_hub import hf_hub_download
    rel, out = build_release(run), {}
    for name, (repo, rev, script, data) in ARITHMARK.items():
        work = f"/tmp/{name}"
        os.makedirs(work, exist_ok=True)
        s = hf_hub_download(repo, script, repo_type="dataset", revision=rev, local_dir=work)
        d = hf_hub_download(repo, data, repo_type="dataset", revision=rev, local_dir=work)
        cmd = ["python", os.path.basename(s), "--model", rel, "--data-path", os.path.basename(d)]
        stdout = sh(cmd, work)
        res_dir = os.path.join(work, "benchmark_cache")
        [res_file] = [f for f in os.listdir(res_dir) if f.endswith(".json")]
        with open(os.path.join(res_dir, res_file)) as f:
            res = json.load(f)
        out[name] = {"repo": repo, "revision": rev, "command": " ".join(cmd).replace(rel, "<hf_release>"),
                     "result": res, "stdout_tail": stdout[-3000:]}
    return out


@app.function(image=img, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_lm_eval_release(mode: str, run: str = RUN) -> dict:
    rel = build_release(run)
    cmd = ["lm_eval", "--model", "hf", "--model_args",
           f"pretrained={rel},trust_remote_code=True" + LM_EVAL_ARGS[mode],
           "--tasks", LB_TASKS, "--num_fewshot", "0", "--batch_size", "32", "--device", "cuda:0",
           "--output_path", "/tmp/lmeval_out"]
    sh(cmd, "/tmp")
    [path] = [os.path.join(dp, f) for dp, _, fs in os.walk("/tmp/lmeval_out") for f in fs
              if f.startswith("results") and f.endswith(".json")]
    with open(path) as f:
        res = json.load(f)
    return {"mode": mode, "command": " ".join(cmd).replace(rel, "<hf_release>"),
            "results": res["results"], "versions": res.get("versions"),
            "lm_eval_version": res.get("lm_eval_version")}


# The only function that attaches the `huggingface` secret (it needs a write token).
@app.function(image=img, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def push(whoami_only: bool = False, run: str = RUN) -> str:
    sys.path.insert(0, "/root")
    import upload_to_hf
    token = os.environ["HF_TOKEN"]
    name = upload_to_hf.whoami(token)
    if whoami_only:
        return f"authenticated as: {name}"
    if name != upload_to_hf.REPO_ID.split("/")[0]:
        raise RuntimeError(f"token belongs to {name}, not {upload_to_hf.REPO_ID}")
    return upload_to_hf.upload(build_release(run, copy_pt=True), token=token)


@app.local_entrypoint()
def upload(whoami_only: bool = False, run: str = RUN):
    print(push.remote(whoami_only, run))


def save(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")


@app.local_entrypoint()
def arithmark(run: str = RUN):
    out = run_arithmark.remote(run)
    for name, r in out.items():
        res = r["result"]["results"]
        print(name, json.dumps({k: v for k, v in next(iter(res.values())).items()
                                if k in ("acc", "acc_norm", "total")}))
    save(out, f"results/main/{run}_arithmark.json")


@app.local_entrypoint()
def lm_eval_release(run: str = RUN, modes: str = "default,ours"):
    names = modes.split(",")
    outs = list(run_lm_eval_release.starmap([(m, run) for m in names]))
    save({m: o for m, o in zip(names, outs)}, f"results/main/{run}_lmeval_release.json")
