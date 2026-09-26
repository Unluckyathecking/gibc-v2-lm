"""CPU tests for gibc/train.py on tiny synthetic shards (6 steps, decay from step 3)."""
import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from gibc.configs import BOS_ID, MIX_ANNEAL, ModelConfig, TrainConfig
from gibc.data import MixLoader, write_shard
from gibc.train import check_data_supply, smoke, train

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


def test_default_val_tokens_scores_every_shard(tmp_path, manifest):
    assert TrainConfig("A", 1).val_tokens is None
    st = train("allval", runs_dir=str(tmp_path / "runs"), data_manifest=manifest,
               tc=replace(TC, val_tokens=None), mc=MC)
    assert st["state"] == "done" and math.isfinite(st["val_bpb"])


def test_last_step_snapshot_saved_for_averaging(tmp_path, manifest):
    train("snap", runs_dir=str(tmp_path / "runs"), data_manifest=manifest,
          tc=replace(TC, snap_every=4), mc=MC)
    assert sorted(os.listdir(tmp_path / "runs/snap/snap")) == ["step_000004.pt", "step_000006.pt"]


def _set_train_tokens(manifest, tmp_path, src: str, n: int) -> dict:
    path = str(tmp_path / f"{src}_tight.bin")
    write_shard(path, _docs(np.random.default_rng(1), n), vocab=MC.vocab)
    entry = {"path": path, "n_tokens": n, "n_bytes": 4 * n, "n_docs": 1}
    return {**manifest, src: {**manifest[src], "train": [entry]}}


# TC: B=2 rows split fwedu 1 / dclm 1 on the main mix, fwedu_hq 1 / finemath 1 on the
# anneal mix; grad_accum 2 -> fwedu needs 2 rows x 3 main steps = 6 rows of T+1 tokens.
@pytest.mark.parametrize("n,ok", [(6 * 32 + 1, True), (6 * 32, False)])
def test_preflight_is_exact_at_the_boundary(tmp_path, manifest, n, ok):
    tight = _set_train_tokens(manifest, tmp_path, "fwedu", n)
    if ok:
        assert _run(tmp_path, tight, "tight")["state"] == "done"  # the loader agrees
    else:
        with pytest.raises(RuntimeError, match="fwedu: needs 6 rows"):
            _run(tmp_path, tight, "tight")
        assert not os.path.exists(tmp_path / "runs/tight/log.jsonl")  # failed before step 0


def test_preflight_counts_consumed_rows_on_resume(tmp_path, manifest):
    tc = replace(TC, mix_anneal=None)  # fwedu: 2 rows x 6 steps = 12 rows
    tight = _set_train_tokens(manifest, tmp_path, "fwedu", 12 * 32 + 1)
    loader = MixLoader(tight, tc.mix_main, tc.micro_bsz, MC.ctx)
    check_data_supply(loader, tight, tc, total=6, step=0)
    loader.next_batch()  # one fwedu row consumed
    with pytest.raises(RuntimeError, match="fwedu: needs 12 rows"):
        check_data_supply(loader, tight, tc, total=6, step=0)
    loader.next_batch()  # a full step consumed: exactly enough left for steps 1..5
    check_data_supply(loader, tight, tc, total=6, step=1)


def test_preflight_rejects_anneal_source_missing_from_manifest(tmp_path, manifest):
    no_hq = {s: v for s, v in manifest.items() if s != "fwedu_hq"}
    with pytest.raises(RuntimeError, match="fwedu_hq: in the mix but has no train shards"):
        _run(tmp_path, no_hq, "nohq")
