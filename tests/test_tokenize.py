"""Tokenizer training, file splits, int_score filter, parquet -> shards, job planning."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from gibc.configs import BOS_ID
from gibc.data import read_shard
from gibc.decontam import (NgramFilter, bench_item_texts, bench_ngrams, bench_report,
                           item_ngrams, ngrams, normalise_title, scan, shard_docs,
                           wikitext_titles, words)
from gibc.tokenize import (N_HQ_FILES, DocFilter, plan_train_jobs, select_docs, select_texts,
                           split_files, tokenize_parquet, train_tokenizer)

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


# --- WikiText-103 decontamination -------------------------------------------
WIKITEXT_LINES = [
    " = Meridian , Mississippi = \n", " \n",
    " Meridian is the sixth @-@ largest city in the state , with a population of 41 @,@ 148 "
    "at the 2010 census and an area of 45 @.@ 5 square miles . \n",
    " = = History = = \n",
    " The city was founded in 1860 at the junction of two railroads and grew quickly after the war . \n",
    " = 2011 – 12 Michigan Wolverines men 's basketball team = \n",
    " = Kiss You ( One Direction song ) = \n",
]
SECRET = ("the city was founded in 1860 at the junction of two railroads and grew quickly "
          "after the war")


def test_wikitext_titles_and_normalisation():
    titles = wikitext_titles(WIKITEXT_LINES)
    assert titles == ["Meridian , Mississippi", "2011 – 12 Michigan Wolverines men 's basketball team",
                      "Kiss You ( One Direction song )"]
    keys = {normalise_title(t) for t in titles}
    for wiki_title in ("Meridian, Mississippi", "MERIDIAN, mississippi",
                       "2011–12 Michigan Wolverines men's basketball team",
                       "Kiss You (One Direction song)"):
        assert normalise_title(wiki_title) in keys
    assert normalise_title("Meridian, Idaho") not in keys
    assert normalise_title("Sixth @-@ largest  city") == "sixth-largest city"
    assert normalise_title("41 @,@ 148 and 45 @.@ 5") == "41,148 and 45.5"


def test_ngram_filter_matches_exact_scan():
    filt = NgramFilter(WIKITEXT_LINES)
    assert filt.targets == bench_ngrams(WIKITEXT_LINES)
    secret = SECRET.split()
    docs = ["unrelated text " * 20,
            "Intro. The City was founded in 1860, at the junction of two railroads AND grew "
            "quickly after the war!",                                  # 13-gram, punctuation differs
            " ".join(secret[:12]),                                     # 12 words only
            " ".join(secret[:12]) + " zebra quickly after the war",    # unknown word breaks it
            "",
            "the war " + " ".join(secret[:6]) + " banana " + " ".join(secret[6:])]
    batch = docs[:3] + [" ".join(secret[:7])] + [" ".join(secret[7:])] + docs[3:]  # split doc pair
    want = [bool(ngrams(words(d)) & filt.targets) for d in batch]
    assert filt.flags(batch) == want == [False, True, False, False, False, False, False, False]
    assert filt.flags([]) == [] and filt.flags(["short"]) == [False]


def _wiki_parquet(tmp_path, rows):
    path = str(tmp_path / "wiki.parquet")
    pq.write_table(pa.table({"id": [str(i) for i in range(len(rows))],
                             "title": [r[0] for r in rows], "text": [r[1] for r in rows]}), path)
    return path


def test_wiki_docs_dropped_by_title_and_ngram(tok, tmp_path):
    rows = [("Meridian, Mississippi", "Meridian is a city in Lauderdale County."),  # title
            ("Railways of Meridian", "History: " + SECRET + "."),                  # 13-gram
            ("Fox", "The quick brown fox jumps over the lazy dog."),              # kept
            ("Empty", ""),                                                        # skipped
            ("Kiss You (One Direction song)", "A song.")]                         # title, last
    path = _wiki_parquet(tmp_path, rows)
    texts, titles = select_docs(pq.read_table(path), "wiki")
    assert titles == [r[0] for r in rows if r[1]] and len(texts) == 4
    filt = DocFilter(frozenset(normalise_title(t) for t in wikitext_titles(WIKITEXT_LINES)),
                     NgramFilter(WIKITEXT_LINES))
    assert filt.reasons(texts, titles) == ["title", "ngram", None, "title"]
    out = tokenize_parquet(path, "wiki", tok, str(tmp_path / "out"), "train_0000", doc_filter=filt)
    assert out["n_docs"] == 1 and out["n_docs_dropped_title"] == 2
    assert out["n_docs_dropped_ngram"] == 1 and out["exhausted"]
    (shard,) = out["shards"]  # the trailing drop is folded into the last shard
    assert shard["n_docs_dropped_title"] == 2 and shard["n_docs_dropped_ngram"] == 1
    toks = np.asarray(read_shard(shard["path"]))
    assert tok.decode_batch([d.tolist() for d in shard_docs(toks)]) == [rows[2][1]]
    plain = tokenize_parquet(path, "wiki", tok, str(tmp_path / "plain"), "t")
    assert plain["n_docs"] == 4 and plain["n_docs_dropped_title"] == 0


def test_bench_items_and_report():
    rows = {"hellaswag": {"ctx": "A man is", "endings": ["running.", "sitting."]},
            "arc_easy": {"question": "Which is a gas?", "choices": {"text": ["air", "rock"]}},
            "piqa": {"goal": "Open a jar", "sol1": "twist the lid", "sol2": "hit it"},
            "winogrande": {"sentence": "Tom asked Bob because _ knew.",
                           "option1": "Tom", "option2": "Bob"}}
    assert bench_item_texts("hellaswag", rows["hellaswag"]) == ["A man is running.", "A man is sitting."]
    assert bench_item_texts("arc_easy", rows["arc_easy"]) == ["Which is a gas? air", "Which is a gas? rock"]
    assert bench_item_texts("piqa", rows["piqa"])[1] == "Open a jar hit it"
    assert bench_item_texts("winogrande", rows["winogrande"])[0] == "Tom asked Bob because Tom knew."
    a = item_ngrams([" ".join(f"w{i}" for i in range(14))])  # 2 13-grams
    items = {"toy": [a, item_ngrams(["too short"]), item_ngrams([" ".join(f"w{i}" for i in range(1, 14))])]}
    rep = bench_report(items, {" ".join(f"w{i}" for i in range(1, 14))})["toy"]
    assert rep["n_items"] == 3 and rep["n_items_with_ngrams"] == 2 and rep["n_ngrams"] == 2
    assert rep["n_matched_ngrams"] == 1 and rep["n_items_hit"] == 2
