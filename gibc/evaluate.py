"""Benchmarks (lm-eval zero-shot) and WikiText-103 test perplexity for trained runs.

Headline numbers are tokenizer-independent: WikiText word perplexity and bits per byte,
counted exactly as lm-eval's `wikitext` task counts them (words and bytes of the raw page).
"""
import glob
import json
import math
import os
import re
import time

import torch

from gibc.configs import BOS_ID, LM_EVAL_TASKS, RUNS_DIR, WIKITEXT, tokenizer_path
from gibc.hf_wrap import load_hf, load_weights

# Metrics we report per task (lm-eval keys are "<metric>,none" and "<metric>_stderr,none").
REPORT_METRICS = {"hellaswag": ["acc_norm"], "arc_easy": ["acc", "acc_norm"],
                  "piqa": ["acc", "acc_norm"], "winogrande": ["acc"]}
N_AVG = 4  # snapshots in the "avg" variant: the last N decay-phase snapshots


def _autocast(device: torch.device):
    """bf16 on CUDA, matching training and train-time val; fp32 elsewhere."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


# --- lm-eval benchmarks ------------------------------------------------------------
def make_hflm(hf_model, tokenizer, batch_size: int = 32):
    """HFLM that prepends BOS to every context and never feeds more than ctx tokens."""
    from lm_eval.models.huggingface import HFLM
    return HFLM(pretrained=hf_model, backend="causal", tokenizer=tokenizer,
                batch_size=batch_size, max_length=hf_model.config.max_position_embeddings,
                add_bos_token=True, prefix_token_id=BOS_ID,
                mixed_precision_dtype=torch.bfloat16 if hf_model.device.type == "cuda" else None)


def pick_metrics(results: dict) -> dict:
    """lm-eval results -> {"task/metric": value, "task/metric_stderr": value}."""
    out = {}
    for task, metrics in REPORT_METRICS.items():
        if task not in results:
            continue
        for m in metrics:
            out[f"{task}/{m}"] = results[task][f"{m},none"]
            out[f"{task}/{m}_stderr"] = results[task][f"{m}_stderr,none"]
    return out


def run_lm_eval(hf_model, tokenizer, tasks=LM_EVAL_TASKS, batch_size: int = 32,
                limit: int | None = None) -> dict:
    """Zero-shot lm-eval on `tasks`; `limit` caps examples per task (first N docs)."""
    from lm_eval import simple_evaluate
    out = simple_evaluate(model=make_hflm(hf_model, tokenizer, batch_size), tasks=list(tasks),
                          num_fewshot=0, limit=limit, log_samples=False)
    return pick_metrics(out["results"])


# --- WikiText-103 test perplexity -----------------------------------------------------
def load_wikitext_pages() -> list[str]:
    """Raw test articles, rebuilt from lines like EleutherAI/wikitext_document_level (the
    source of lm-eval's `wikitext` task): 62 pages, identical except one trailing newline."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, name, split = WIKITEXT
    path = hf_hub_download(repo, f"{name}/{split}-00000-of-00001.parquet", repo_type="dataset")
    pages, cur = [], []
    for line in pq.read_table(path).column("text").to_pylist():
        if re.match(r"^ = [^=].* = \n$", line):  # top-level " = Title = " starts a page
            pages.append("".join(cur))
            cur = []
        cur.append(line or " \n")  # blank lines are "" in the parquet, " \n" in the raw file
    pages.append("".join(cur))
    return [p.removesuffix("\n") for p in pages if p.strip()]


def window_spans(n: int, ctx: int, stride: int) -> list[tuple[int, int, int]]:
    """(begin, end, first_target) windows over a length-n sequence whose position 0 is BOS.

    Targets 1..n-1 are each scored exactly once; after the first window every target
    sees at least ctx - stride tokens of context.
    """
    assert 0 < stride < ctx, "each window needs >= 1 context token before its first target"
    spans, scored, begin = [], 1, 0
    while scored < n:
        end = min(begin + ctx, n)
        spans.append((begin, end, scored))
        scored, begin = end, begin + stride
    return spans


@torch.no_grad()
def score_ids(model, ids: list[int], ctx: int, stride: int) -> float:
    """Summed negative log-likelihood (nats) of ids[1:] given everything before them."""
    x_all = torch.tensor(ids, device=model.device)
    nats = 0.0
    for begin, end, first in window_spans(len(ids), ctx, stride):
        x = x_all[begin:end]
        with _autocast(model.device):
            logits = model(x[None]).logits[0].float()
        nll = torch.nn.functional.cross_entropy(logits[:-1], x[1:], reduction="none")
        nats += nll[first - begin - 1:].sum().item()
    return nats


def wikitext103_ppl(model, tokenizer, ctx: int = 1024, stride: int = 512,
                    pages: list[str] | None = None) -> dict:
    """Sliding-window perplexity over each test article, BOS first, lm-eval's counting:
    the detokenized page is scored; words and bytes are counted on the raw page."""
    from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
    pages = load_wikitext_pages() if pages is None else pages
    nats = n_tokens = n_words = n_bytes = 0
    for page in pages:
        text = wikitext_detokenizer({"page": page})
        ids = [BOS_ID] + tokenizer.encode(text, add_special_tokens=False)
        nats += score_ids(model, ids, ctx, stride)
        n_tokens += len(ids) - 1
        n_words += len(re.split(r"\s+", page))
        n_bytes += len(page.encode("utf-8"))
    return {"total_nats": nats, "n_tokens": n_tokens, "n_words": n_words, "n_bytes": n_bytes,
            "n_docs": len(pages), "ctx": ctx, "stride": stride,
            "token_ppl": math.exp(nats / n_tokens), "word_ppl": math.exp(nats / n_words),
            "bits_per_byte": nats / (math.log(2) * n_bytes)}


# --- checkpoints and whole runs -------------------------------------------------------
def average_checkpoints(paths: list[str]) -> dict:
    """Elementwise fp32 mean of the model state_dicts in `paths`."""
    total = None
    for path in paths:
        state, _ = load_weights(path)
        if total is None:
            total = {k: v.float().clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                total[k] += v.float()
    return {k: v / len(paths) for k, v in total.items()}


def eval_weights(state: dict, mc, device: str, limit: int | None = None,
                 batch_size: int = 32) -> dict:
    """lm-eval benchmarks + WikiText-103 for one set of GPT weights."""
    model, tok = load_hf(state, mc, tokenizer_path(mc.tokenizer), device)
    t0 = time.time()
    res = {"lm_eval": run_lm_eval(model, tok, batch_size=batch_size, limit=limit),
           "wikitext103": wikitext103_ppl(model, tok, ctx=mc.ctx, stride=mc.ctx // 2)}
    return {**res, "eval_s": round(time.time() - t0, 1)}


def variant_weights(rd: str, variant: str, n_avg: int = N_AVG) -> list[str]:
    """Files behind a variant: "final" -> [final.pt]; "avg" -> the last n_avg snapshots."""
    if variant == "final":
        return glob.glob(os.path.join(rd, "final.pt"))
    if variant == "avg":
        return sorted(glob.glob(os.path.join(rd, "snap", "step_*.pt")))[-n_avg:]
    raise ValueError(f"unknown variant {variant!r}")


def evaluate_run(run: str, variants=("final", "avg"), limit: int | None = None,
                 n_avg: int = N_AVG, runs_dir: str = RUNS_DIR, device: str | None = None) -> dict:
    """Evaluate each variant of a finished run and write {run}/evals/{variant}[_limitN].json.

    "avg" is skipped when the run has fewer than 2 snapshots.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rd, out = os.path.join(runs_dir, run), {}
    os.makedirs(os.path.join(rd, "evals"), exist_ok=True)
    for variant in variants:
        paths = variant_weights(rd, variant, n_avg)
        if len(paths) < (2 if variant == "avg" else 1):
            print(f"{run}: skipping {variant} ({len(paths)} weight files)")
            continue
        _, mc = load_weights(paths[-1])
        state = average_checkpoints(paths) if variant == "avg" else load_weights(paths[0])[0]
        rec = {"run": run, "variant": variant, "weights": [os.path.relpath(p, rd) for p in paths],
               "limit": limit, **eval_weights(state, mc, device, limit)}
        name = variant if limit is None else f"{variant}_limit{limit}"
        with open(os.path.join(rd, "evals", f"{name}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        out[variant] = rec
    return out
