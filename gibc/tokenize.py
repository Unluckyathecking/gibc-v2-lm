"""Tokenizer training, source file listing/splitting, and parquet -> token shard conversion.

Tokenizer: byte-level BPE with the Llama-3 split regex; <|bos|>, <|eos|>, <|pad|> are ids
0, 1, 2. Its post-processor prepends <|bos|> (for lm-eval's add_bos_token); shard writing
encodes with add_special_tokens=False and inserts BOS itself, once per document.

Decontamination: when NGRAM_FILTER is on, every source's docs (train and val) sharing a
13-gram with WikiText-103 validation + test are dropped; TITLE_FILTER sources (Wikipedia)
also drop articles titled like a WikiText-103 val/test article. Counts are recorded per
shard (n_docs_dropped_title, n_docs_dropped_ngram) and per file.
"""
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from gibc.configs import BOS_ID, EOS_ID, PAD_ID, SHARD_TOKENS, SOURCES, SPECIAL_TOKENS
from gibc.data import write_shard
from gibc.decontam import NgramFilter, normalise_title, wikitext_filter, wikitext_title_keys

LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
ENCODE_BATCH = 1024          # docs per parquet batch / encode_batch_fast call
VAL_SHARD_TOKENS = 1_000_000  # small val shards so val_bpb can take whole shards per source
N_VAL_FILES = 1              # last file(s) of every source's list hold its val docs
N_HQ_FILES = 60              # fineweb-edu files before the val file reserved for fwedu_hq
PLAN_SAFETY = 0.9            # plan as if each new file yields 90% of the files seen so far
NGRAM_FILTER = True          # drop docs sharing a 13-gram with WikiText-103 val+test (all sources)
TITLE_FILTER = {"wiki": "title"}  # source -> title column; drop WikiText-103 val/test articles
# Stored in each file's job meta; a job whose meta has a different tag is redone on resume.
FILTER_TAG = "wikitext103-valtest-13gram" if NGRAM_FILTER else None


# --- tokenizer ---------------------------------------------------------------
def train_tokenizer(text_iter, vocab_size: int, out_path: str | None = None) -> Tokenizer:
    """Train byte-level BPE on an iterable of strings; save to out_path if given."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(text_iter, trainer)
    ids = [tok.token_to_id(t) for t in SPECIAL_TOKENS]
    assert ids == [BOS_ID, EOS_ID, PAD_ID], f"special token ids {ids}"
    tok.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS[BOS_ID]} $A", special_tokens=[(SPECIAL_TOKENS[BOS_ID], BOS_ID)])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tok.save(out_path)
    return tok


# --- source files ------------------------------------------------------------
def list_source_files(source_key: str, api=None) -> list[str]:
    """Sorted parquet paths in the source's HF repo matching Source.pattern."""
    from huggingface_hub import HfApi
    src = SOURCES[source_key]
    files = (api or HfApi()).list_repo_files(src.repo, repo_type="dataset")
    return sorted(f for f in files if fnmatchcase(f, src.pattern))


def split_files(source_key: str, files: list[str]) -> dict:
    """{"train": [(idx, path)], "val": [(idx, path)]}, idx = position in the sorted list.

    Val is the last N_VAL_FILES files. fwedu and fwedu_hq share a repo: fwedu_hq trains on
    the N_HQ_FILES files before val, fwedu on everything earlier, so all three are disjoint.
    """
    indexed = list(enumerate(files))
    val, rest = indexed[-N_VAL_FILES:], indexed[:-N_VAL_FILES]
    if source_key in ("fwedu", "fwedu_hq"):
        assert len(rest) > N_HQ_FILES, f"only {len(rest)} fineweb-edu train files"
        rest = rest[-N_HQ_FILES:] if source_key == "fwedu_hq" else rest[:-N_HQ_FILES]
    return {"train": rest, "val": val}


def plan_train_jobs(target: int, files: list, results: list) -> list:
    """Next round of (file, token_budget) for one source; [] when done or out of files.

    results are the finished jobs for files[:len(results)] ({"n_tokens", "exhausted"}).
    Round 1 is the first file with the whole target. Later rounds split what remains over
    enough files that each needs only PLAN_SAFETY of the mean yield of fully-read files.
    """
    remaining = target - sum(r["n_tokens"] for r in results)
    left = files[len(results):]
    if remaining <= 0 or not left:
        return []
    if not results:
        return [(left[0], remaining)]
    full = [r["n_tokens"] for r in results if r["exhausted"]] or [r["n_tokens"] for r in results]
    per_file = max(1.0, PLAN_SAFETY * sum(full) / len(full))
    n = min(len(left), math.ceil(remaining / per_file))
    return [(f, math.ceil(remaining / n)) for f in left[:n]]


# --- parquet -> shards -------------------------------------------------------
def select_docs(table, source_key: str) -> tuple[list[str], list[str] | None]:
    """(texts, titles) of the non-empty rows of a pyarrow Table/RecordBatch with
    int_score >= min_int_score; titles is None unless the source is in TITLE_FILTER."""
    src = SOURCES[source_key]
    if src.min_int_score is not None:
        table = table.filter(pc.greater_equal(table.column("int_score"), src.min_int_score))
    texts = table.column(src.text_col).to_pylist()
    keep = [i for i, t in enumerate(texts) if t]
    title_col = TITLE_FILTER.get(source_key)
    titles = table.column(title_col).to_pylist() if title_col else None
    return [texts[i] for i in keep], titles and [titles[i] for i in keep]


def select_texts(table, source_key: str) -> list[str]:
    """Non-empty texts of a pyarrow Table/RecordBatch, keeping int_score >= min_int_score."""
    return select_docs(table, source_key)[0]


def iter_parquet_docs(path: str, source_key: str):
    """Yield select_docs() of each ENCODE_BATCH rows, streaming the parquet file."""
    src = SOURCES[source_key]
    cols = ([src.text_col] + (["int_score"] if src.min_int_score is not None else [])
            + ([TITLE_FILTER[source_key]] if source_key in TITLE_FILTER else []))
    for batch in pq.ParquetFile(path).iter_batches(batch_size=ENCODE_BATCH, columns=cols):
        yield select_docs(batch, source_key)


def iter_parquet_texts(path: str, source_key: str):
    """Yield lists of selected texts, streaming the parquet file in ENCODE_BATCH rows."""
    for texts, _ in iter_parquet_docs(path, source_key):
        yield texts


@dataclass(frozen=True)
class DocFilter:
    """Drops docs by title (normalised, in title_keys) first, then by n-gram overlap."""
    title_keys: frozenset | None = None
    ngram: NgramFilter | None = None

    def reasons(self, texts: list[str], titles: list[str] | None = None) -> list[str | None]:
        """Per doc: "title", "ngram", or None to keep."""
        out = [None] * len(texts)
        if self.title_keys is not None and titles is not None:
            out = ["title" if normalise_title(t or "") in self.title_keys else None
                   for t in titles]
        if self.ngram is not None:
            idx = [i for i, r in enumerate(out) if r is None]
            for i, hit in zip(idx, self.ngram.flags([texts[i] for i in idx])):
                out[i] = "ngram" if hit else None
        return out


def source_filter(source_key: str) -> DocFilter | None:
    """The WikiText-103 val+test filter for a source (built once per process), or None."""
    titles = wikitext_title_keys() if source_key in TITLE_FILTER else None
    ngram = wikitext_filter() if NGRAM_FILTER else None
    return DocFilter(titles, ngram) if titles is not None or ngram is not None else None


DROP_REASONS = ("title", "ngram")


class ShardWriter:
    """Collects [BOS] + ids docs and writes {out_dir}/{stem}_{part:02d}.bin every shard_tokens.
    Dropped docs are counted against the shard being filled (the last shard at the end)."""

    def __init__(self, out_dir: str, stem: str, vocab: int, shard_tokens: int = SHARD_TOKENS):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.stem, self.vocab, self.shard_tokens = out_dir, stem, vocab, shard_tokens
        self.shards, self.total_tokens, self.total_bytes = [], 0, 0
        self.total_dropped = dict.fromkeys(DROP_REASONS, 0)
        self._reset()

    def _reset(self):
        self.chunks, self.n_tokens, self.n_bytes = [], 0, 0
        self.dropped = dict.fromkeys(DROP_REASONS, 0)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1
        self.total_dropped[reason] += 1

    def add(self, ids: np.ndarray, n_bytes: int) -> None:
        self.chunks.append(ids)
        self.n_tokens += len(ids)
        self.n_bytes += n_bytes
        self.total_tokens += len(ids)
        self.total_bytes += n_bytes
        if self.n_tokens >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        drops = {f"n_docs_dropped_{r}": n for r, n in self.dropped.items()}
        if self.chunks:
            path = os.path.join(self.out_dir, f"{self.stem}_{len(self.shards):02d}.bin")
            write_shard(path, np.concatenate(self.chunks), self.vocab)
            self.shards.append({"path": path, "n_tokens": self.n_tokens,
                                "n_bytes": self.n_bytes, "n_docs": len(self.chunks), **drops})
        elif self.shards:
            self.shards[-1] = {**self.shards[-1],
                               **{k: self.shards[-1][k] + n for k, n in drops.items()}}
        self._reset()


def _full(w: ShardWriter, budget: int | None, max_bytes: int | None) -> bool:
    return ((budget is not None and w.total_tokens >= budget)
            or (max_bytes is not None and w.total_bytes >= max_bytes))


def tokenize_parquet(path: str, source_key: str, tok: Tokenizer, out_dir: str, stem: str,
                     budget: int | None = None, max_bytes: int | None = None,
                     shard_tokens: int = SHARD_TOKENS, doc_filter: DocFilter | None = None) -> dict:
    """Tokenize a local parquet file into shards, whole docs only, skipping docs doc_filter
    drops and stopping once budget tokens or max_bytes text bytes are reached (overshoot <
    one doc). Each batch is encoded in a worker thread (the tokenizer releases the GIL)
    while the filter runs. Returns {"shards": [...], "n_tokens", "n_bytes", "n_docs",
    "n_docs_dropped_title", "n_docs_dropped_ngram", "exhausted"}."""
    w = ShardWriter(out_dir, stem, tok.get_vocab_size(), shard_tokens)
    with ThreadPoolExecutor(1) as pool:
        for texts, titles in iter_parquet_docs(path, source_key):
            if _full(w, budget, max_bytes):
                break
            encs = pool.submit(tok.encode_batch_fast, texts, add_special_tokens=False)
            reasons = doc_filter.reasons(texts, titles) if doc_filter else [None] * len(texts)
            for text, enc, reason in zip(texts, encs.result(), reasons):
                if _full(w, budget, max_bytes):
                    break
                if reason:
                    w.drop(reason)
                    continue
                ids = np.empty(len(enc.ids) + 1, dtype=np.uint16)
                ids[0], ids[1:] = BOS_ID, enc.ids
                w.add(ids, len(text.encode("utf-8")))
    exhausted = not _full(w, budget, max_bytes)
    w.flush()
    return {"shards": w.shards, "n_tokens": w.total_tokens, "n_bytes": w.total_bytes,
            "n_docs": sum(s["n_docs"] for s in w.shards),
            **{f"n_docs_dropped_{r}": n for r, n in w.total_dropped.items()},
            "exhausted": exhausted}


def tokenize_file_to_shards(source_key: str, file_idx: int, file: str, tok: Tokenizer,
                            out_dir: str, split: str, budget: int | None = None,
                            max_bytes: int | None = None) -> dict:
    """Download one HF parquet file to local disk, shard it as {split}_{file_idx:04d}_NN.bin
    in out_dir with the source's decontamination filter, then delete the download. Val
    shards are VAL_SHARD_TOKENS long."""
    from huggingface_hub import hf_hub_download
    shard_tokens = SHARD_TOKENS if split == "train" else VAL_SHARD_TOKENS
    doc_filter = source_filter(source_key)
    with tempfile.TemporaryDirectory() as tmp:
        local = hf_hub_download(SOURCES[source_key].repo, file, repo_type="dataset", local_dir=tmp)
        out = tokenize_parquet(local, source_key, tok, out_dir, f"{split}_{file_idx:04d}",
                               budget, max_bytes, shard_tokens, doc_filter)
    return {**out, "filter": FILTER_TAG}
