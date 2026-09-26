"""Shard format, manifest, MixLoader and exact val bpb (CPU, no network)."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gibc.configs import BOS_ID, SHARD_MAGIC, SHARD_VERSION
from gibc.data import (IGNORE_INDEX, MixLoader, allocate_rows, read_manifest, read_shard,
                       val_batches, val_bpb, val_shards, write_manifest, write_shard)

T = 8
SOURCE_BASE = {"a": 1000, "b": 2000, "c": 3000}  # token values identify the source


def _entry(path, toks):
    write_shard(str(path), toks)
    return {"path": str(path), "n_tokens": len(toks), "n_bytes": 3 * len(toks), "n_docs": 1}


@pytest.fixture
def manifest(tmp_path):
    """3 sources x 3 train shards of 100 tokens + 1 val shard; values encode (source, shard)."""
    m = {}
    for src, base in SOURCE_BASE.items():
        train = [_entry(tmp_path / f"{src}_t{i}.bin", base + 100 * i + np.arange(100))
                 for i in range(3)]
        val = [_entry(tmp_path / f"{src}_v.bin", base + np.arange(37))]
        m[src] = {"train": train, "val": val}
    return m


def _source_of(row):
    return {v // 1000: k for k, v in SOURCE_BASE.items()}[int(row[0]) // 1000]


def test_shard_round_trip(tmp_path):
    toks = np.array([BOS_ID, 5, 32767, 7, 0], dtype=np.int64)
    p = str(tmp_path / "s.bin")
    write_shard(p, toks, vocab=32768)
    header = np.fromfile(p, dtype="<i4", count=256)
    assert list(header[:3]) == [SHARD_MAGIC, SHARD_VERSION, 5] and not header[3:].any()
    for mmap in (True, False):
        out = read_shard(p, mmap=mmap)
        assert out.dtype == np.uint16 and np.array_equal(out, toks)


def test_shard_rejects_bad_ids_and_headers(tmp_path):
    p = str(tmp_path / "s.bin")
    with pytest.raises(ValueError):
        write_shard(p, [1, 32768], vocab=32768)
    with pytest.raises(ValueError):
        write_shard(p, [65535])
    with pytest.raises(ValueError):
        write_shard(p, [-1])
    write_shard(p, [1, 2, 3])
    with open(p, "r+b") as f:
        f.write(np.int32(123).tobytes())
    with pytest.raises(ValueError):
        read_shard(p)


def test_manifest_paths_relative_on_disk(tmp_path, manifest):
    path = str(tmp_path / "manifest.json")
    write_manifest(path, manifest)
    on_disk = json.load(open(path))
    assert on_disk["a"]["train"][0]["path"] == "a_t0.bin"
    assert read_manifest(path) == manifest


def test_allocate_rows_largest_remainder():
    assert allocate_rows({"a": 0.6, "b": 0.3, "c": 0.1}, 10) == {"a": 6, "b": 3, "c": 1}
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = dict(zip("abcd", rng.random(4)))
        B = int(rng.integers(1, 100))
        rows = allocate_rows(w, B)
        assert sum(rows.values()) == B
        assert all(abs(rows[s] - w[s] / sum(w.values()) * B) < 1 for s in w)
    assert allocate_rows({"a": 1.0, "b": 0.0}, 4) == {"a": 4}


def test_batch_proportions_and_shift(manifest):
    loader = MixLoader(manifest, {"a": 0.6, "b": 0.3, "c": 0.1}, B=10, T=T)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (10, T) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    counts = {}
    for row in x:
        counts[_source_of(row)] = counts.get(_source_of(row), 0) + 1
    assert counts == {"a": 6, "b": 3, "c": 1}


def test_resume_from_state_reproduces_next_batch(manifest):
    mix = {"a": 0.5, "b": 0.5}
    first = MixLoader(manifest, mix, B=4, T=T, seed=3)
    for _ in range(5):
        first.next_batch()
    state = json.loads(json.dumps(first.state()))
    resumed = MixLoader(manifest, {"c": 1.0}, B=4, T=T, state=state, seed=99)
    for _ in range(3):
        (x1, y1), (x2, y2) = first.next_batch(), resumed.next_batch()
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def _grown(manifest, tmp_path):
    """The manifest after a top-up: every source gains a fourth train shard."""
    return {src: {**splits, "train": splits["train"] + [
                _entry(tmp_path / f"{src}_t3.bin", SOURCE_BASE[src] + 300 + np.arange(100))]}
            for src, splits in manifest.items()}


def test_resume_ignores_shards_added_after_checkpoint(manifest, tmp_path):
    mix = {"a": 0.5, "b": 0.5}
    first = MixLoader(manifest, mix, B=4, T=T, seed=3)
    for _ in range(5):
        first.next_batch()
    state = json.loads(json.dumps(first.state()))
    grown = _grown(manifest, tmp_path)
    assert MixLoader(grown, mix, B=4, T=T, seed=3).shards != first.shards  # order would move
    resumed = MixLoader(grown, mix, B=4, T=T, state=state)
    assert resumed.shards == first.shards
    for _ in range(3):
        (x1, y1), (x2, y2) = first.next_batch(), resumed.next_batch()
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_resume_from_state_without_shards_field(manifest):
    mix = {"a": 0.5, "b": 0.5}
    first = MixLoader(manifest, mix, B=4, T=T, seed=3)
    first.next_batch()
    old = {k: v for k, v in first.state().items() if k != "shards"}
    resumed = MixLoader(manifest, mix, B=4, T=T, state=old)
    assert resumed.shards == first.shards
    assert torch.equal(first.next_batch()[0], resumed.next_batch()[0])


def test_set_mix_new_source_after_pinned_restore(manifest, tmp_path):
    first = MixLoader(manifest, {"a": 1.0}, B=2, T=T, seed=3)
    first.next_batch()
    state = first.state()
    grown = _grown(manifest, tmp_path)
    grown["d"] = {"train": [_entry(tmp_path / "d_t0.bin", 4000 + np.arange(100))], "val": []}
    resumed = MixLoader(grown, {"a": 1.0}, B=2, T=T, state=state)
    resumed.set_mix({"d": 1.0})
    x, _ = resumed.next_batch()
    assert torch.equal(x, torch.from_numpy(4000 + np.arange(2 * T).reshape(2, T)))
    assert resumed.shards["d"] == MixLoader(grown, {"d": 1.0}, B=2, T=T, seed=3).shards["d"]


def test_set_mix_keeps_cursors(manifest):
    loader = MixLoader(manifest, {"a": 1.0}, B=2, T=T)
    fresh = MixLoader(manifest, {"a": 1.0}, B=2, T=T)
    loader.next_batch()
    fresh.next_batch()
    loader.set_mix({"c": 1.0})
    xc, _ = loader.next_batch()
    xc_fresh, _ = MixLoader(manifest, {"c": 1.0}, B=2, T=T).next_batch()
    assert torch.equal(xc, xc_fresh)  # c starts at its own cursor 0
    loader.set_mix({"a": 1.0})
    assert torch.equal(loader.next_batch()[0], fresh.next_batch()[0])  # a resumes
    with pytest.raises(KeyError):
        loader.set_mix({"zzz": 1.0})


def test_exhaustion_raises_without_wrapping(manifest):
    loader = MixLoader(manifest, {"b": 1.0}, B=2, T=T)
    rows_per_shard = (100 - 1) // T  # rows of T+1 tokens, stride T
    seen = set()
    with pytest.raises(RuntimeError, match="exhausted"):
        for _ in range(1000):
            x, _ = loader.next_batch()
            seen.update(tuple(r.tolist()) for r in x)
    assert len(seen) == 3 * rows_per_shard - (3 * rows_per_shard) % 2


def test_val_batches_score_every_token_once(manifest):
    shards = [manifest[s]["val"][0] for s in "abc"]
    targets = []
    for x, y in val_batches(shards, B=4, T=T):
        assert x.shape == y.shape == (4, T)
        targets += y[y != IGNORE_INDEX].tolist()
    expected = []
    for e in shards:
        expected += read_shard(e["path"])[1:].tolist()
    assert sorted(targets) == sorted(expected)


def test_val_shards_quota(manifest):
    extra = {**manifest, "a": {**manifest["a"], "val": manifest["a"]["val"] * 3}}
    mix = {"a": 0.5, "b": 0.5}
    assert len(val_shards(extra, None, mix)) == 4
    assert len(val_shards(extra, 80, mix)) == 3  # a: 37+37 >= 40, b: one shard
    assert len(val_shards(extra, 2, mix)) == 2   # every source keeps at least one shard


def test_val_bpb_exact_for_uniform_model(manifest):
    V = 4096

    def uniform(x, y):
        logits = torch.zeros(*x.shape, V)
        return F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=IGNORE_INDEX)

    res = val_bpb(uniform, manifest, B=4, T=T, mix={"a": 0.5, "b": 0.5})
    n_tok = sum(e["n_tokens"] - 1 for s in "ab" for e in manifest[s]["val"])
    n_bytes = sum(e["n_bytes"] for s in "ab" for e in manifest[s]["val"])
    assert res["n_tokens"] == n_tok and res["n_bytes"] == n_bytes
    assert res["loss"] == pytest.approx(math.log(V))
    assert res["bpb"] == pytest.approx(n_tok * math.log(V) / (math.log(2) * n_bytes))
