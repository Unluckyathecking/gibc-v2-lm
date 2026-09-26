"""Token shards, the manifest that indexes them, the training mix loader and exact val bpb.

Shard file: 256 little-endian int32 header [SHARD_MAGIC, SHARD_VERSION, n_tokens, 0...],
then n_tokens little-endian uint16 ids. Every document is written as [BOS] ids, and a
document never straddles two shards.

Manifest (JSON at manifest_path(tok)), shard paths stored relative to the manifest's dir:
    {source: {"train": [{"path", "n_tokens", "n_bytes", "n_docs"}, ...], "val": [...]}}
n_bytes is the UTF-8 byte count of the source text of the shard's documents.

Val bits-per-byte is exact: val_bpb scores every token of whole val shards except each
shard's leading BOS (all text tokens plus the BOS that opens each later doc), so the bytes
behind the scored tokens are exactly the shards' summed n_bytes:
    bpb = total_nats / (ln 2 * total_bytes)
"""
import json
import math
import os
import zlib

import numpy as np
import torch

from gibc.configs import (PAD_ID, SHARD_HEADER_INTS, SHARD_MAGIC, SHARD_VERSION, VAL_MIX,
                          manifest_path)

MAX_TOKEN_ID = 65535  # uint16 storage; ids must be strictly below this (and below vocab)
IGNORE_INDEX = -100   # y value at padded val positions (F.cross_entropy's default ignore_index)
HEADER_BYTES = SHARD_HEADER_INTS * 4


# --- shard files -------------------------------------------------------------
def write_shard(path: str, tokens, vocab: int | None = None) -> None:
    """Write tokens as a shard. Raises ValueError if any id is < 0 or >= min(vocab, 65535)."""
    tokens = np.asarray(tokens)
    limit = min(vocab or MAX_TOKEN_ID, MAX_TOKEN_ID)
    if tokens.size and (tokens.min() < 0 or tokens.max() >= limit):
        raise ValueError(f"token id out of range [0, {limit}) in {path}")
    header = np.zeros(SHARD_HEADER_INTS, dtype="<i4")
    header[:3] = SHARD_MAGIC, SHARD_VERSION, tokens.size
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:  # write-then-rename so a crash never leaves a half shard
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())
    os.replace(tmp, path)


def read_shard(path: str, mmap: bool = True) -> np.ndarray:
    """Return a shard's tokens as a uint16 array, after validating header and file size."""
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
        raise ValueError(f"bad shard header in {path}")
    n = int(header[2])
    if os.path.getsize(path) != HEADER_BYTES + 2 * n:
        raise ValueError(f"shard {path} size does not match its header ({n} tokens)")
    if mmap:
        return np.memmap(path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(n,))
    return np.fromfile(path, dtype="<u2", count=n, offset=HEADER_BYTES)


# --- manifest ----------------------------------------------------------------
def _map_paths(manifest: dict, fn) -> dict:
    return {src: {split: [{**e, "path": fn(e["path"])} for e in entries]
                  for split, entries in splits.items()}
            for src, splits in manifest.items()}


def write_manifest(path: str, manifest: dict) -> None:
    """Write a manifest with shard paths made relative to its directory."""
    root = os.path.dirname(path)
    rel = _map_paths(manifest, lambda p: os.path.relpath(p, root))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rel, f, indent=1)
    os.replace(tmp, path)


def read_manifest(path: str) -> dict:
    """Read a manifest and resolve shard paths against its directory."""
    root = os.path.dirname(path)
    with open(path) as f:
        return _map_paths(json.load(f), lambda p: os.path.join(root, p))


def load_manifest(tok: str) -> dict:
    return read_manifest(manifest_path(tok))


# --- training loader ---------------------------------------------------------
def allocate_rows(mix: dict, B: int) -> dict:
    """Split B rows across sources by largest remainder on the mix weights (deterministic)."""
    active = sorted(s for s, w in mix.items() if w > 0)
    total = sum(mix[s] for s in active)
    if not active or any(w < 0 for w in mix.values()):
        raise ValueError(f"mix needs non-negative weights with a positive sum: {mix}")
    exact = {s: mix[s] * B / total for s in active}
    rows = {s: math.floor(exact[s]) for s in active}
    by_remainder = sorted(active, key=lambda s: (rows[s] - exact[s], s))
    for s in by_remainder[:B - sum(rows.values())]:
        rows[s] += 1
    return rows


def _shard_order(entries: list, seed: int, src: str, shuffle: bool) -> list:
    """Shard paths for one source; train order is a fixed permutation of (seed, source)."""
    paths = [e["path"] for e in entries]
    if not shuffle:
        return paths
    rng = np.random.default_rng([seed, zlib.crc32(src.encode())])
    return [paths[i] for i in rng.permutation(len(paths))]


class MixLoader:
    """Batches of [B, T] rows drawn from several sources in fixed per-batch proportions.

    Each source is read sequentially through its (seed-shuffled) shard list via its own
    cursor (shard_idx, offset). A row is T+1 contiguous tokens from one shard (a shard tail
    shorter than T+1 is skipped); x = row[:-1], y = row[1:], and the cursor advances by T so
    every token after the first is a target once. A source that runs out raises RuntimeError.
    state() is JSON-able; MixLoader(..., state=s) restores cursors, mix and seed from s
    (overriding the mix and seed arguments) and reproduces the identical next batch.
    state() also pins each source's resolved shard list, so shards added to the manifest
    after a checkpoint cannot reorder a resumed run; sources absent from the pinned lists
    (or every source, for an older state without them) are ordered from the manifest.
    """

    def __init__(self, manifest: dict, mix: dict, B: int, T: int, split: str = "train",
                 state: dict | None = None, seed: int = 0):
        if state is not None:
            mix, seed = state["mix"], state["seed"]
        self.B, self.T, self.split, self.seed = B, T, split, seed
        derived = {src: _shard_order(s[split], seed, src, shuffle=split == "train")
                   for src, s in manifest.items() if s.get(split)}
        pinned = {src: list(p) for src, p in (state or {}).get("shards", {}).items()}
        self.shards = {**derived, **pinned}
        self.cursors = {src: [0, 0] for src in self.shards}
        if state is not None:
            self.cursors.update({src: list(c) for src, c in state["cursors"].items()})
        self._open = {}
        self.set_mix(mix)

    def set_mix(self, mix: dict) -> None:
        """Switch proportions; every source keeps its own cursor (0 if never read)."""
        missing = [s for s, w in mix.items() if w > 0 and s not in self.shards]
        if missing:
            raise KeyError(f"mix sources with no {self.split} shards: {missing}")
        self.mix = dict(mix)
        self.rows = allocate_rows(self.mix, self.B)

    def state(self) -> dict:
        return {"mix": dict(self.mix), "seed": self.seed, "split": self.split,
                "cursors": {src: list(c) for src, c in self.cursors.items()},
                "shards": {src: list(p) for src, p in self.shards.items()}}

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buf = np.empty((self.B, self.T + 1), dtype=np.int64)
        r = 0
        for src in sorted(self.rows):
            for _ in range(self.rows[src]):
                buf[r] = self._take(src)
                r += 1
        return torch.from_numpy(buf[:, :-1].copy()), torch.from_numpy(buf[:, 1:].copy())

    def _tokens(self, path: str) -> np.ndarray:
        if path not in self._open:
            self._open[path] = read_shard(path)
        return self._open[path]

    def _take(self, src: str) -> np.ndarray:
        i, off = self.cursors[src]
        paths, n = self.shards[src], self.T + 1
        while True:
            if i >= len(paths):
                raise RuntimeError(f"source {src!r} exhausted its {len(paths)} {self.split} shards")
            toks = self._tokens(paths[i])
            if off + n <= len(toks):
                break
            i, off = i + 1, 0
        self.cursors[src] = [i, off + self.T]
        return toks[off:off + n]


# --- exact val bits-per-byte -------------------------------------------------
def val_shards(manifest: dict, max_tokens: int | None = None, mix: dict = VAL_MIX) -> list:
    """Whole val shards of the mix's sources. With max_tokens, each source contributes shards
    (in order) until it reaches its weight's share of max_tokens; None takes every shard."""
    total_w = sum(mix.values())
    chosen = []
    for src in sorted(s for s, w in mix.items() if w > 0):
        quota = math.inf if max_tokens is None else max_tokens * mix[src] / total_w
        got = 0
        for e in manifest[src]["val"]:
            if got >= quota:
                break
            chosen.append(e)
            got += e["n_tokens"]
    return chosen


def val_batches(shards: list, B: int, T: int):
    """Yield (x, y) int64 [B, T] batches scoring tokens[1:] of every shard exactly once.
    Short rows are padded with x = PAD_ID, y = IGNORE_INDEX; so are filler rows at the end."""
    xs, ys = [], []
    for e in shards:
        toks = read_shard(e["path"])
        for start in range(0, len(toks) - 1, T):
            seg = np.asarray(toks[start:start + T + 1], dtype=np.int64)
            x, y = np.full(T, PAD_ID, np.int64), np.full(T, IGNORE_INDEX, np.int64)
            x[:len(seg) - 1], y[:len(seg) - 1] = seg[:-1], seg[1:]
            xs.append(x)
            ys.append(y)
            if len(xs) == B:
                yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
                xs, ys = [], []
    if xs:
        pad = B - len(xs)
        xs += [np.full(T, PAD_ID, np.int64)] * pad
        ys += [np.full(T, IGNORE_INDEX, np.int64)] * pad
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def val_bpb(model, manifest: dict, B: int, T: int, max_tokens: int | None = None,
            device="cpu", mix: dict = VAL_MIX) -> dict:
    """Exact bits-per-byte of model(x, y) (mean CE over non-ignored targets) on val_shards.
    Returns {"bpb", "loss" (nats/token), "n_tokens", "n_bytes"}."""
    shards = val_shards(manifest, max_tokens, mix)
    nats, n_tok = 0.0, 0
    for x, y in val_batches(shards, B, T):
        n = int((y != IGNORE_INDEX).sum())
        loss = model(x.to(device), y.to(device))
        nats += float(loss) * n
        n_tok += n
    n_bytes = sum(e["n_bytes"] for e in shards)
    return {"bpb": nats / (math.log(2) * n_bytes), "loss": nats / n_tok,
            "n_tokens": n_tok, "n_bytes": n_bytes}
