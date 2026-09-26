"""CPU tests for gibc/train.py on tiny synthetic shards (6 steps, decay from step 3)."""
import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from gibc.configs import BOS_ID, MIX_ANNEAL, ModelConfig, TrainConfig
from gibc.data import write_shard
from gibc.train import smoke, train

MC = ModelConfig(vocab=256, d_model=32, n_unique_blocks=2, repeats=1, n_head=4,
                 n_kv_head=2, mlp_hidden=64, ctx=32)
# 2 x 2 x 32 = 128 tokens/step, 6 steps; decay 0.5 -> decay starts at step index 3.
TC = TrainConfig("A", total_tokens=6 * 128, mix_anneal=dict(MIX_ANNEAL), micro_bsz=2,
                 grad_accum=2, warmup=0.2, decay=0.5, ckpt_minutes=10_000, snap_every=1,
                 val_every=3, val_tokens=2000, seed=0)
SOURCES = ["fwedu", "fwedu_hq", "dclm", "finemath", "finepdfs"]


def _docs(rng, n_tokens: int) -> np.ndarray:
    toks = rng.integers(3, MC.vocab, n_tokens)
    toks[::50] = BOS_ID  # a document every 50 tokens, stream starts with BOS
    return toks


@pytest.fixture
def manifest(tmp_path) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for src in SOURCES:
        out[src] = {}
        for split, n in [("train", 4000), ("val", 500)]:
            path = str(tmp_path / f"{src}_{split}.bin")
            write_shard(path, _docs(rng, n), vocab=MC.vocab)
            out[src][split] = [{"path": path, "n_tokens": n, "n_bytes": 4 * n, "n_docs": n // 50}]
    return out


def _log(rd) -> list[dict]:
    with open(os.path.join(rd, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def _json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _run(tmp_path, manifest, name, **kw):
    return train(name, runs_dir=str(tmp_path / "runs"), data_manifest=manifest, tc=TC, mc=MC, **kw)


def test_full_run_logs_and_artifacts(tmp_path, manifest):
    st = _run(tmp_path, manifest, "full")
    rd = tmp_path / "runs" / "full"
    log = _log(rd)
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6]
    assert all(math.isfinite(r["loss"]) for r in log)
    assert log[-1]["lr_mult"] == 0.0 and log[-1]["tokens_seen"] == 6 * 128
    assert "val_bpb" in log[2] and "val_bpb" in log[5] and "val_bpb" not in log[3]
    assert math.isfinite(log[5]["val_bpb"]) and log[5]["val_bpb"] > 0
    assert st["state"] == "done" and st["step"] == 6
    assert _json(rd / "status.json")["state"] == "done"
    # decay starts at step index 3 -> snapshots after completed steps 4, 5, 6
    assert sorted(os.listdir(rd / "snap")) == [f"step_00000{n}.pt" for n in (4, 5, 6)]
    final = torch.load(rd / "final.pt", weights_only=True)
    assert set(final) == {"model", "model_cfg"} and final["model_cfg"]["d_model"] == 32
    assert _json(rd / "latest.json")["step"] == 6


def test_resume_matches_uninterrupted(tmp_path, manifest):
    _run(tmp_path, manifest, "ref")
    ref = [r["loss"] for r in _log(tmp_path / "runs/ref")]
    calls = []
    st = _run(tmp_path, manifest, "split", max_steps=3, on_checkpoint=lambda: calls.append(1))
    assert st["state"] == "stopped" and st["step"] == 3 and calls
    _run(tmp_path, manifest, "split")
    got = [r["loss"] for r in _log(tmp_path / "runs/split")]
    assert len(got) == 6
    assert got == pytest.approx(ref, abs=1e-6)


def test_mix_switches_to_anneal_at_decay_start(tmp_path, manifest):
    _run(tmp_path, manifest, "mix", max_steps=3)  # steps 0..2 use the main mix
    ckpt = torch.load(tmp_path / "runs/mix/ckpt/step_000003.pt", weights_only=True)
    assert ckpt["loader_state"]["mix"] == TC.mix_main
    assert ckpt["loader_state"]["cursors"]["fwedu_hq"] == [0, 0]
    _run(tmp_path, manifest, "mix", max_steps=4)  # step 3 is the first decay step
    ckpt = torch.load(tmp_path / "runs/mix/ckpt/step_000004.pt", weights_only=True)
    assert ckpt["loader_state"]["mix"] == MIX_ANNEAL
    assert ckpt["loader_state"]["cursors"]["fwedu_hq"] != [0, 0]


def test_frequent_checkpoints_keep_last_two(tmp_path, manifest):
    train("ck", runs_dir=str(tmp_path / "runs"), data_manifest=manifest,
          tc=replace(TC, ckpt_minutes=0), mc=MC)
    assert sorted(os.listdir(tmp_path / "runs/ck/ckpt")) == ["step_000005.pt", "step_000006.pt"]


def test_smoke_reports_throughput():
    res = smoke("tiny", minutes=0, tc=TC, mc=MC)
    assert res["loss_finite"] and res["steps"] >= 1 and res["tok_per_s"] > 0
