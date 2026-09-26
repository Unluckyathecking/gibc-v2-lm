"""Modal app "gibc-data" (CPU only): tokenizers, token shards, manifest, decontam, check.

    modal run --detach modal_data.py::train_tokenizers
    modal run --detach modal_data.py::tokenize_all --tok tok32k   (then --tok tok16k)
    modal run modal_data.py::check --tok tok32k
    modal run --detach modal_data.py::decontam

tokenize_all is resumable: each per-file job writes {source}/meta/{split}_{idx:04d}.json
last and is skipped when that exists; rounds are planned deterministically from those
results, so a rerun rebuilds the same plan. Progress: {shard_dir(tok)}/status.json.
"""
import json
import os
import random
import tempfile
import time

import modal
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from gibc.configs import (DATA_DIR, MIX_MAIN, SOURCES, TOKEN_TARGETS, TOKENIZERS,
                          VAL_TOKENS_PER_SOURCE, WIKITEXT, manifest_path, shard_dir,
                          tokenizer_path)
from gibc.data import load_manifest, read_shard, write_manifest
from gibc.decontam import decontam_report
from gibc.modal_common import VOLUMES, data_vol, hf_secret, image
from gibc.tokenize import (iter_parquet_texts, list_source_files, plan_train_jobs, split_files,
                           tokenize_file_to_shards, train_tokenizer)

app = modal.App("gibc-data")
FN = dict(image=image, volumes=VOLUMES, secrets=[hf_secret])

TOK_TRAIN_BYTES = 3_000_000_000  # text sampled to train each tokenizer
TOK_TRAIN_MIX = {**{s: 0.85 * w for s, w in MIX_MAIN.items()}, "fwedu_hq": 0.15}
# Val docs are chosen by a byte cap (~5M tok32k tokens at ~4.6 bytes/token), not a token cap, so tok32k
# and tok16k get the exact same val documents.
VAL_BYTES_PER_SOURCE = 5 * VAL_TOKENS_PER_SOURCE
DECONTAM_TOKENS_PER_SOURCE = 50_000_000  # ~1.4M tokens/s scanned on one core
HOUR = 3600


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


# --- tokenizers --------------------------------------------------------------
def _sample_texts(source: str, n_bytes: int) -> list[str]:
    """Leading docs of the source's train files until n_bytes of UTF-8 text."""
    out, got = [], 0
    for _, f in split_files(source, list_source_files(source))["train"]:
        with tempfile.TemporaryDirectory() as tmp:
            local = hf_hub_download(SOURCES[source].repo, f, repo_type="dataset", local_dir=tmp)
            for texts in iter_parquet_texts(local, source):
                out += texts
                got += sum(len(t.encode("utf-8")) for t in texts)
                if got >= n_bytes:
                    return out
    return out


@app.function(**FN, cpu=32, memory=131072, timeout=6 * HOUR)
def train_tokenizers():
    texts = []
    for src, w in sorted(TOK_TRAIN_MIX.items()):
        texts += _sample_texts(src, int(w * TOK_TRAIN_BYTES))
        print(f"sampled {src}: {len(texts):,} docs so far", flush=True)
    random.Random(0).shuffle(texts)
    for tok, vocab in TOKENIZERS.items():
        t0 = time.time()
        train_tokenizer(texts, vocab, tokenizer_path(tok))
        data_vol.commit()
        print(f"{tok}: vocab {vocab} trained in {time.time() - t0:.0f}s", flush=True)


# --- tokenization ------------------------------------------------------------
def _meta_path(tok: str, source: str, split: str, idx: int) -> str:
    return f"{shard_dir(tok)}/{source}/meta/{split}_{idx:04d}.json"


@app.function(**FN, cpu=8, memory=16384, timeout=4 * HOUR, max_containers=32, retries=2)
def tokenize_job(args: dict) -> dict:
    """Shard one parquet file; args: tok, source, split, file_idx, file, budget, max_bytes."""
    meta_path = _meta_path(args["tok"], args["source"], args["split"], args["file_idx"])
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    tok = Tokenizer.from_file(tokenizer_path(args["tok"]))
    out = tokenize_file_to_shards(args["source"], args["file_idx"], args["file"], tok,
                                  f"{shard_dir(args['tok'])}/{args['source']}", args["split"],
                                  args.get("budget"), args.get("max_bytes"))
    meta = {**args, **out}
    _write_json(meta_path, meta)
    data_vol.commit()
    return meta


def _job(tok, source, split, file, budget=None, max_bytes=None) -> dict:
    idx, path = file
    return {"tok": tok, "source": source, "split": split, "file_idx": idx, "file": path,
            "budget": budget, "max_bytes": max_bytes}


def _run_round(jobs: list, status: dict, tok: str) -> list:
    outs = list(tokenize_job.map(jobs, return_exceptions=True, order_outputs=False))
    errors = [repr(o) for o in outs if isinstance(o, BaseException)]
    if errors:
        _status(tok, {**status, "state": "failed", "errors": errors[:20]})
        raise RuntimeError(f"{len(errors)} tokenize jobs failed; rerun to resume: {errors[:3]}")
    return outs


def _status(tok: str, status: dict) -> None:
    _write_json(f"{shard_dir(tok)}/status.json", {**status, "time": time.strftime("%FT%TZ")})
    data_vol.commit()


def _manifest(results: dict) -> dict:
    return {src: {split: [s for r in sorted(rs, key=lambda r: r["file_idx"]) for s in r["shards"]]
                  for split, rs in splits.items()}
            for src, splits in results.items()}


def _progress(targets: dict, results: dict) -> dict:
    return {s: {"target": t, "files": len(results[s]["train"]),
                "train_tokens": sum(r["n_tokens"] for r in results[s]["train"]),
                "val_tokens": sum(r["n_tokens"] for r in results[s]["val"])}
            for s, t in targets.items()}


@app.function(**FN, cpu=2, memory=4096, timeout=24 * HOUR)
def tokenize_all(tok: str = "tok32k"):
    """Plan per-file jobs in rounds until every source meets TOKEN_TARGETS[tok], then write
    the manifest. Val: first VAL_BYTES_PER_SOURCE of each source's val file."""
    assert os.path.exists(tokenizer_path(tok)), f"run train_tokenizers first ({tok})"
    targets = TOKEN_TARGETS[tok]
    files = {s: split_files(s, list_source_files(s)) for s in targets}
    results = {s: {"train": [], "val": []} for s in targets}
    jobs = [_job(tok, s, "val", files[s]["val"][0], max_bytes=VAL_BYTES_PER_SOURCE)
            for s in targets]
    for rnd in range(1, 100):
        for s, t in targets.items():
            plan = plan_train_jobs(t, files[s]["train"], results[s]["train"])
            jobs += [_job(tok, s, "train", f, budget=b) for f, b in plan]
        if not jobs:
            break
        status = {"state": "running", "round": rnd, "jobs": len(jobs),
                  "sources": _progress(targets, results)}
        _status(tok, status)
        for meta in _run_round(jobs, status, tok):
            results[meta["source"]][meta["split"]].append(meta)
        for s in targets:
            results[s]["train"].sort(key=lambda r: r["file_idx"])
        jobs = []
    write_manifest(manifest_path(tok), _manifest(results))
    progress = _progress(targets, results)
    short = [s for s, p in progress.items() if p["train_tokens"] < p["target"]]
    _status(tok, {"state": "done", "short_sources": short, "sources": progress})
    print(json.dumps(progress, indent=1))


# --- checks ------------------------------------------------------------------
@app.function(**FN, cpu=8, memory=32768, timeout=3 * HOUR)
def decontam(tok: str = "tok32k"):
    """WikiText-103 test 13-gram overlap with the first train shard of each source."""
    repo, config, split = WIKITEXT
    path = hf_hub_download(repo, f"{config}/{split}-00000-of-00001.parquet", repo_type="dataset")
    bench = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    manifest = load_manifest(tok)
    shards = [manifest[s]["train"][0]["path"] for s in sorted(manifest) if manifest[s]["train"]]
    report = decontam_report(bench, shards, Tokenizer.from_file(tokenizer_path(tok)),
                             DECONTAM_TOKENS_PER_SOURCE)
    _write_json(f"{DATA_DIR}/decontam.json", report)
    data_vol.commit()
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=1))


@app.function(**FN, cpu=2, memory=8192, timeout=HOUR)
def check(tok: str = "tok32k", window: int = 64, seed: int = 0):
    """Manifest totals per source/split vs targets, and 3 decoded random windows per source."""
    manifest = load_manifest(tok)
    tokenizer = Tokenizer.from_file(tokenizer_path(tok))
    targets = TOKEN_TARGETS[tok]
    for src in sorted(manifest):
        for split, entries in manifest[src].items():
            n_tok = sum(e["n_tokens"] for e in entries)
            n_bytes = sum(e["n_bytes"] for e in entries)
            target = targets.get(src, 0) if split == "train" else VAL_TOKENS_PER_SOURCE
            print(f"{src:9s} {split:5s} shards={len(entries):4d} tokens={n_tok:>14,} "
                  f"target={target:>14,} ({n_tok / max(1, target):6.1%}) "
                  f"docs={sum(e['n_docs'] for e in entries):>11,} bytes/token={n_bytes / max(1, n_tok):.3f}")
    rng = random.Random(seed)
    for src in sorted(manifest):
        for _ in range(3):
            toks = read_shard(rng.choice(manifest[src]["train"])["path"])
            start = rng.randrange(max(1, len(toks) - window))
            text = tokenizer.decode(toks[start:start + window].tolist(), skip_special_tokens=False)
            print(f"--- {src} @ {start}: {text!r}")
