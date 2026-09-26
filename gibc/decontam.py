"""13-gram overlap between WikiText-103 test and a sample of training shards.

N-grams are over normalized words, not tokens, so the check is tokenizer-independent and
robust to WikiText's spaced punctuation (" @-@ ", " , "): text is lowercased and split into
runs of [a-z0-9]. Reports the fraction of distinct WikiText test 13-grams found in the
sample and the number of sampled training docs containing at least one of them.
"""
import re

import numpy as np

from gibc.configs import BOS_ID
from gibc.data import read_shard

NGRAM = 13
WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(ws: list[str], n: int = NGRAM) -> set[str]:
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def bench_ngrams(texts: list[str]) -> set[str]:
    """Distinct 13-grams of the benchmark text, joined into one word stream."""
    return ngrams(words(" ".join(texts)))


def shard_docs(tokens: np.ndarray, max_tokens: int | None = None) -> list[np.ndarray]:
    """Split a [BOS] doc [BOS] doc ... stream into docs (BOS removed), whole docs only,
    stopping at the first doc boundary at or after max_tokens."""
    if max_tokens is not None and len(tokens) > max_tokens:
        later = np.flatnonzero(tokens[max_tokens:] == BOS_ID)
        tokens = tokens[:max_tokens + later[0]] if len(later) else tokens
    starts = np.flatnonzero(tokens == BOS_ID)
    ends = list(starts[1:]) + [len(tokens)]
    return [tokens[s + 1:e] for s, e in zip(starts, ends)]


def scan(docs: list[str], targets: set[str]) -> tuple[set[str], int, int]:
    """(matched target n-grams, docs with any match, words scanned)."""
    matched, n_match, n_words = set(), 0, 0
    for doc in docs:
        ws = words(doc)
        n_words += len(ws)
        hits = ngrams(ws) & targets
        if hits:
            matched |= hits
            n_match += 1
    return matched, n_match, n_words


def decontam_report(bench_texts: list[str], shard_paths: list[str], tok,
                    max_tokens_per_shard: int | None = None) -> dict:
    """Scan (a prefix of) each training shard for benchmark 13-grams; JSON-able summary."""
    targets = bench_ngrams(bench_texts)
    matched, n_docs, n_match, n_words, n_tokens = set(), 0, 0, 0, 0
    for path in shard_paths:
        docs = shard_docs(np.asarray(read_shard(path)), max_tokens_per_shard)
        texts = tok.decode_batch([d.tolist() for d in docs])
        m, k, nw = scan(texts, targets)
        matched |= m
        n_docs, n_match, n_words = n_docs + len(docs), n_match + k, n_words + nw
        n_tokens += sum(len(d) + 1 for d in docs)
    return {"ngram": NGRAM, "n_test_ngrams": len(targets), "n_matched_ngrams": len(matched),
            "frac_test_ngrams_matched": len(matched) / max(1, len(targets)),
            "n_docs_scanned": n_docs, "n_docs_matching": n_match, "n_words_scanned": n_words,
            "n_tokens_scanned": n_tokens, "shards": shard_paths, "examples": sorted(matched)[:5]}
