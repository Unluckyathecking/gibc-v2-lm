"""13-gram decontamination against WikiText-103 and the lm-eval benchmarks.

N-grams are over normalized words, not tokens, so the check is tokenizer-independent and
robust to WikiText's spaced punctuation (" @-@ ", " , "): text is lowercased and split into
runs of [a-z0-9].

Three uses:
- NgramFilter / wikitext_filter: at tokenization time, drop every doc (all sources, train
  and val) sharing >= 1 13-gram with WikiText-103 validation + test.
- wikitext_title_keys: for Wikipedia, also drop articles titled like a val/test article.
- decontam_report / bench_report: overlap REPORTS on a sample of training shards
  (WikiText-103 test; hellaswag, arc_easy, piqa, winogrande eval items). No filtering.
"""
import functools
import re
from itertools import repeat

import numpy as np

from gibc.configs import BOS_ID, WIKITEXT
from gibc.data import read_shard

NGRAM = 13
WORD = re.compile(r"[a-z0-9]+")
WIKITEXT_SPLITS = ("validation", "test")  # both are held out from every training source


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(ws: list[str], n: int = NGRAM) -> set[str]:
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def bench_ngrams(texts: list[str]) -> set[str]:
    """Distinct 13-grams of the benchmark text, joined into one word stream."""
    return ngrams(words(" ".join(texts)))


# --- WikiText titles ---------------------------------------------------------
HEADING = re.compile(r" = ([^=].*?) = ")  # article heading line; sections are " = = x = = "
UNESCAPE = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."))


def normalise_title(title: str) -> str:
    """Matching key for a title: WikiText escapes undone, no space before closing punctuation,
    after "(" or around dashes, whitespace collapsed, casefolded. Applied to both sides."""
    t = " ".join(title.split())
    for a, b in UNESCAPE:
        t = t.replace(a, b)
    t = re.sub(r" ([,.;:!?)\]'])", r"\1", t)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = re.sub(r" ?([-–—]) ?", r"\1", t)
    return t.casefold()


def wikitext_titles(lines: list[str]) -> list[str]:
    """Article titles from WikiText lines (" = Title = \\n" headings only)."""
    return [m.group(1) for line in lines if (m := HEADING.fullmatch(line.rstrip("\n")))]


# --- fast doc filter ---------------------------------------------------------
BASE = 0x9E3779B97F4A7C15  # odd, so invertible mod 2**64


def _powers(b: int, length: int) -> np.ndarray:
    """[1, b, b**2, ...] mod 2**64 as uint64."""
    p = np.full(length, b, dtype=np.uint64)
    p[:1] = 1
    return np.cumprod(p, dtype=np.uint64)


class NgramFilter:
    """Flags docs sharing >= 1 n-gram (over words()) with a reference text.

    Per batch: one C-level pass maps words to reference-vocabulary ids (-1 if absent);
    windows holding an unknown word or crossing a doc boundary are skipped; the rest get a
    64-bit polynomial hash (prefix sums, wrapping uint64) looked up in the sorted reference
    hashes; hash hits are confirmed against the exact n-gram strings.
    """

    def __init__(self, ref_texts: list[str], n: int = NGRAM):
        self.n = n
        self.targets = ngrams(words(" ".join(ref_texts)), n)
        self.vocab: dict[str, int] = {}
        for g in self.targets:
            for w in g.split():
                self.vocab.setdefault(w, len(self.vocab))
        ids = np.array([[self.vocab[w] for w in g.split()] for g in self.targets],
                       dtype=np.uint64).reshape(-1, n)
        pows = _powers(BASE, n)
        self.hashes = np.unique((ids * pows).sum(axis=1, dtype=np.uint64))
        self._pows = self._inv_pows = np.zeros(0, dtype=np.uint64)

    def _window_hashes(self, ids: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """sum_k ids[s+k] * BASE**k (mod 2**64) for each start s, via prefix sums."""
        if len(self._pows) < len(ids):  # powers are cached and grown geometrically
            size = max(len(ids), 2 * len(self._pows))
            self._pows = _powers(BASE, size)
            self._inv_pows = _powers(pow(BASE, -1, 2 ** 64), size)
        prefix = np.zeros(len(ids) + 1, dtype=np.uint64)
        np.cumsum(ids.astype(np.uint64) * self._pows[:len(ids)], dtype=np.uint64, out=prefix[1:])
        return (prefix[starts + self.n] - prefix[starts]) * self._inv_pows[starts]

    def flags(self, texts: list[str]) -> list[bool]:
        """True for each text containing at least one reference n-gram."""
        ws = [words(t) for t in texts]
        get, ids = self.vocab.get, []
        for doc in ws:  # -1 marks unknown words and doc ends, so no valid window crosses one
            ids += map(get, doc, repeat(-1))
            ids.append(-1)
        ids = np.array(ids, dtype=np.int64)
        out = [False] * len(texts)
        if len(ids) < self.n:
            return out
        unknown = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(ids < 0, out=unknown[1:])
        starts = np.flatnonzero(unknown[self.n:] == unknown[:-self.n])
        if not len(starts):
            return out
        h = self._window_hashes(ids, starts)
        pos = np.searchsorted(self.hashes, h).clip(max=len(self.hashes) - 1)
        offs = np.cumsum([0] + [len(doc) + 1 for doc in ws])
        for s in starts[self.hashes[pos] == h]:
            d = int(np.searchsorted(offs, s, side="right")) - 1
            if not out[d]:
                i = s - offs[d]
                out[d] = " ".join(ws[d][i:i + self.n]) in self.targets
        return out


@functools.cache
def wikitext_lines() -> list[str]:
    """WikiText-103 raw validation + test lines, downloaded once per process."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, config, _ = WIKITEXT
    lines = []
    for split in WIKITEXT_SPLITS:
        path = hf_hub_download(repo, f"{config}/{split}-00000-of-00001.parquet",
                               repo_type="dataset")
        lines += pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return lines


@functools.cache
def wikitext_filter() -> NgramFilter:
    return NgramFilter(wikitext_lines())


@functools.cache
def wikitext_title_keys() -> frozenset:
    return frozenset(normalise_title(t) for t in wikitext_titles(wikitext_lines()))


# --- overlap reports ---------------------------------------------------------
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


def scan_shards(targets: set[str], shard_paths: list[str], tok,
                max_tokens_per_shard: int | None = None) -> tuple[set[str], dict]:
    """Scan (a prefix of) each training shard for target n-grams: (matched, scan stats)."""
    matched, n_docs, n_match, n_words, n_tokens = set(), 0, 0, 0, 0
    for path in shard_paths:
        docs = shard_docs(np.asarray(read_shard(path)), max_tokens_per_shard)
        m, k, nw = scan(tok.decode_batch([d.tolist() for d in docs]), targets)
        matched |= m
        n_docs, n_match, n_words = n_docs + len(docs), n_match + k, n_words + nw
        n_tokens += sum(len(d) + 1 for d in docs)
    return matched, {"n_docs_scanned": n_docs, "n_docs_matching": n_match,
                     "n_words_scanned": n_words, "n_tokens_scanned": n_tokens,
                     "shards": shard_paths}


def decontam_report(bench_texts: list[str], shard_paths: list[str], tok,
                    max_tokens_per_shard: int | None = None) -> dict:
    """WikiText test 13-grams found in the shard sample; JSON-able summary."""
    targets = bench_ngrams(bench_texts)
    matched, stats = scan_shards(targets, shard_paths, tok, max_tokens_per_shard)
    return {"ngram": NGRAM, "n_test_ngrams": len(targets), "n_matched_ngrams": len(matched),
            "frac_test_ngrams_matched": len(matched) / max(1, len(targets)), **stats,
            "examples": sorted(matched)[:5]}


# --- lm-eval benchmarks ------------------------------------------------------
# (repo, parquet path) of the split each lm-eval task scores (dataset_path in its yaml).
BENCH_FILES = {
    "hellaswag": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
    "piqa": ("baber/piqa", "piqa_validation.parquet"),
    "winogrande": ("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet"),
}


def bench_item_texts(bench: str, row: dict) -> list[str]:
    """The texts an eval item is scored on: context joined with each candidate answer."""
    if bench == "hellaswag":
        return [f"{row['ctx']} {e}" for e in row["endings"]]
    if bench == "arc_easy":
        return [f"{row['question']} {c}" for c in row["choices"]["text"]]
    if bench == "piqa":
        return [f"{row['goal']} {row['sol1']}", f"{row['goal']} {row['sol2']}"]
    if bench == "winogrande":
        return [row["sentence"].replace("_", row[k]) for k in ("option1", "option2")]
    raise KeyError(bench)


def item_ngrams(texts: list[str]) -> set[str]:
    return set().union(*(ngrams(words(t)) for t in texts))


def bench_report(items: dict[str, list[set[str]]], matched: set[str]) -> dict:
    """Per benchmark: its distinct 13-grams found in the sample and items with any hit."""
    out = {}
    for bench, sets in items.items():
        grams = set().union(*sets)
        hit = sum(1 for s in sets if s & matched)
        out[bench] = {"n_items": len(sets), "n_items_with_ngrams": sum(1 for s in sets if s),
                      "n_ngrams": len(grams), "n_matched_ngrams": len(grams & matched),
                      "frac_ngrams_matched": len(grams & matched) / max(1, len(grams)),
                      "n_items_hit": hit, "frac_items_hit": hit / max(1, len(sets)),
                      "examples": sorted(grams & matched)[:5]}
    return out


def decontam_bench_report(rows: dict[str, list[dict]], shard_paths: list[str], tok,
                          max_tokens_per_shard: int | None = None) -> dict:
    """Benchmark eval-item 13-gram overlap with the shard sample (one scan for all)."""
    items = {b: [item_ngrams(bench_item_texts(b, r)) for r in rs] for b, rs in rows.items()}
    targets = set().union(*(s for sets in items.values() for s in sets))
    matched, stats = scan_shards(targets, shard_paths, tok, max_tokens_per_shard)
    return {"ngram": NGRAM, "benchmarks": bench_report(items, matched), **stats}
