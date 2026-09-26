"""Tokenizer training, file splits, int_score filter, parquet -> shards, job planning."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from gibc.configs import BOS_ID
from gibc.data import read_shard
from gibc.decontam import bench_ngrams, scan, shard_docs
from gibc.tokenize import (N_HQ_FILES, plan_train_jobs, select_texts, split_files,
                           tokenize_parquet, train_tokenizer)

CORPUS = ["The quick brown fox jumps over the lazy dog. " * 3,
          "Numbers 12345 and 3.14159; code: x = f(y) + 2\n\tindent",
          "Unicode: naïve café, 東京, Ελληνικά, emoji 🙂🚀, math ∑x²."] * 20
HARD = "  héllo\twörld\r\n東京 🙂 x=1;  \n\n trailing  "


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("tok") / "tok" / "tokenizer.json")
    train_tokenizer(iter(CORPUS), 400, path)
    return Tokenizer.from_file(path)  # round trip through the saved file


def test_special_ids_and_lossless(tok):
    assert [tok.token_to_id(t) for t in ("<|bos|>", "<|eos|>", "<|pad|>")] == [0, 1, 2]
    assert 3 + 256 < tok.get_vocab_size() <= 400  # tiny corpus may run out of merges
    for text in CORPUS[:3] + [HARD]:
        ids = tok.encode(text, add_special_tokens=False).ids
        assert BOS_ID not in ids and tok.decode(ids) == text


def test_post_processor_adds_bos(tok):
    assert tok.encode("hello", add_special_tokens=True).ids[0] == BOS_ID
    assert tok.encode("hello", add_special_tokens=False).ids[0] != BOS_ID


def test_int_score_filter():
    table = pa.table({"text": ["a", "b", "c", "", "e"], "int_score": [3, 4, 5, 5, 2]})
    assert select_texts(table, "fwedu_hq") == ["b", "c"]
    assert select_texts(table, "fwedu") == ["a", "b", "c", "e"]
    assert select_texts(table.to_batches()[0], "fwedu_hq") == ["b", "c"]


def test_file_splits_disjoint():
    files = [f"sample/100BT/{i:03d}.parquet" for i in range(140)]
    parts = {k: split_files(k, files) for k in ("fwedu", "fwedu_hq")}
    val = parts["fwedu"]["val"]
    assert val == parts["fwedu_hq"]["val"] == [(139, files[139])]
    fw, hq = set(parts["fwedu"]["train"]), set(parts["fwedu_hq"]["train"])
    assert len(hq) == N_HQ_FILES and not (fw & hq) and not ((fw | hq) & set(val))
    assert len(fw | hq | set(val)) == 140
    other = split_files("dclm", files)
    assert len(other["train"]) == 139 and not set(other["train"]) & set(other["val"])


def _parquet(tmp_path, texts, scores):
    path = str(tmp_path / "toy.parquet")
    pq.write_table(pa.table({"text": texts, "int_score": scores}), path, row_group_size=7)
    return path


def test_tokenize_parquet_shards(tok, tmp_path):
    texts = [f"doc {i}: " + CORPUS[i % 3] for i in range(50)]
    scores = [4 if i % 2 else 3 for i in range(50)]
    path = _parquet(tmp_path, texts, scores)
    out = tokenize_parquet(path, "fwedu_hq", tok, str(tmp_path / "out"), "train_0000",
                           shard_tokens=500)
    kept = texts[1::2]
    assert out["exhausted"] and out["n_docs"] == len(kept) and len(out["shards"]) > 1
    assert out["n_bytes"] == sum(len(t.encode()) for t in kept)
    docs = []
    for s in out["shards"]:
        toks = np.asarray(read_shard(s["path"]))
        assert toks[0] == BOS_ID and len(toks) == s["n_tokens"]
        docs += tok.decode_batch([d.tolist() for d in shard_docs(toks)])
    assert docs == kept
    assert out["n_tokens"] == sum(s["n_tokens"] for s in out["shards"])


def test_tokenize_parquet_budgets(tok, tmp_path):
    texts = [f"doc {i}: " + CORPUS[i % 3] for i in range(50)]
    path = _parquet(tmp_path, texts, [5] * 50)
    full = tokenize_parquet(path, "fwedu", tok, str(tmp_path / "a"), "t")
    part = tokenize_parquet(path, "fwedu", tok, str(tmp_path / "b"), "t", budget=300)
    assert not part["exhausted"] and 300 <= part["n_tokens"] < 300 + 200
    by_bytes = tokenize_parquet(path, "fwedu", tok, str(tmp_path / "c"), "t", max_bytes=1000)
    assert 1000 <= by_bytes["n_bytes"] < 1000 + 200 and by_bytes["n_docs"] < full["n_docs"]


def test_plan_train_jobs():
    files = [(i, f"f{i}") for i in range(10)]
    assert plan_train_jobs(1000, files, []) == [((0, "f0"), 1000)]
    first = [{"n_tokens": 300, "exhausted": True}]
    plan = plan_train_jobs(1000, files, first)
    assert [f for f, _ in plan] == files[1:4] and sum(b for _, b in plan) >= 700
    assert plan_train_jobs(1000, files, first + [{"n_tokens": 700, "exhausted": False}]) == []
    assert plan_train_jobs(10**9, files[:1], first) == []  # out of files


def test_decontam_scan():
    bench = ["the " * 3 + "Alpha @-@ beta gamma delta epsilon zeta eta theta iota kappa lambda mu"]
    targets = bench_ngrams(bench)
    docs = ["unrelated text", "The ALPHA beta, gamma delta epsilon zeta eta theta iota kappa lambda mu!"]
    matched, n_match, _ = scan(docs, targets)
    assert n_match == 1 and len(matched) == 1
