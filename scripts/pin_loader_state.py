"""Pin the shard order into an old checkpoint's loader_state so a resume is exact.

Checkpoints written before loader states carried "shards" re-derive each source's order
from the manifest on resume. If shards were added since the run started, that order
changes. This rebuilds the order the run actually trained on (the first OLD_COUNTS[src]
manifest entries of the topped-up sources), proves it against the checkpoint's cursors,
writes {ckpt}.pinned.pt with loader_state["shards"] added, and checks the pinned loader on
the current manifest yields the same batches as the old loader on the old manifest.

Runs locally (the Modal CLI moves files; no Modal function is called):
    uv run python scripts/pin_loader_state.py --run main_vres_anneal --work /path/to/scratch
    ... --upload   also puts the pinned file on gibc-runs, then over the original name
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess

import torch

from gibc.configs import DATA_VOL, RUNS_VOL, ModelConfig, TrainConfig, shard_dir
from gibc.data import MixLoader, _shard_order, allocate_rows, read_manifest
from gibc.optim import decay_start
from gibc.train import check_data_supply, total_steps

TOK = "tok32k"
OLD_COUNTS = {"fwedu": 99, "dclm": 68}  # train shards per source when the run started
VOL_ROOT = shard_dir(TOK)                # "/data/shards/tok32k", as load_manifest resolves


def modal_get(vol: str, remote: str, local: str) -> None:
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        subprocess.run(["modal", "volume", "get", vol, remote, local], check=True,
                       capture_output=True)


def modal_put(vol: str, local: str, remote: str, force: bool = False) -> None:
    cmd = ["modal", "volume", "put", *(["--force"] if force else []), vol, local, remote]
    subprocess.run(cmd, check=True, capture_output=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def file_idx(path: str) -> int:
    return int(os.path.basename(path).split("_")[1])  # train_{file:04d}_{part:02d}.bin


def old_manifest(cur: dict) -> dict:
    return {src: {**splits, "train": splits["train"][:OLD_COUNTS.get(src, len(splits["train"]))]}
            for src, splits in cur.items()}


def rows_consumed(order: list, n_tokens: dict, cursor: list, T: int) -> int:
    i, off = cursor
    return sum((n_tokens[p] - 1) // T for p in order[:i]) + off // T


def check_old_set(cur: dict, work: str, fails: list) -> None:
    """The old set is the first N entries: a file boundary, and older meta files."""
    for src, n in OLD_COUNTS.items():
        paths = [e["path"] for e in cur[src]["train"]]
        idx = [file_idx(p) for p in paths]
        print(f"[{src}] manifest has {len(paths)} train shards; sorted by path: "
              f"{paths == sorted(paths)}; file idx non-decreasing: {idx == sorted(idx)}")
        print(f"  old/new boundary: #{n - 1} {os.path.basename(paths[n - 1])} | "
              f"#{n} {os.path.basename(paths[n])}")
        if idx[n - 1] == idx[n]:
            fails.append(f"{src}: shard {n} splits file {idx[n]} across old and new sets")
        with open(f"{work}/ls_{src}_meta.json") as f:
            meta = {os.path.basename(x["Filename"]): x["Created/Modified"] for x in json.load(f)}
        old_t = [meta[f"train_{k:04d}.json"] for k in sorted(set(idx[:n]))]
        new_t = [meta[f"train_{k:04d}.json"] for k in sorted(set(idx[n:]))]
        print(f"  meta written: old files {min(old_t)}..{max(old_t)}, "
              f"new files {min(new_t)}..{max(new_t)}")
        if max(old_t) >= min(new_t):
            fails.append(f"{src}: old and new meta write times overlap")


def check_cursors(cur, old, state, tc, total, step, T, fails) -> None:
    """Rows the schedule says each source gave must equal the rows its cursor implies."""
    ds = decay_start(total, tc.decay)
    main_rows = allocate_rows(tc.mix_main, tc.micro_bsz)
    ann_rows = allocate_rows(tc.mix_anneal, tc.micro_bsz)
    n_tok = {e["path"]: e["n_tokens"] for s in cur.values() for e in s["train"]}
    print(f"step {step}, decay_start {ds}, total {total}")
    for src, cursor in state["cursors"].items():
        expect = tc.grad_accum * (main_rows.get(src, 0) * min(step, ds)
                                  + ann_rows.get(src, 0) * max(0, step - ds))
        o_old = _shard_order(old[src]["train"], state["seed"], src, shuffle=True)
        o_cur = _shard_order(cur[src]["train"], state["seed"], src, shuffle=True)
        got_old = rows_consumed(o_old, n_tok, cursor, T)
        got_cur = rows_consumed(o_cur, n_tok, cursor, T)
        i, off = cursor
        size = n_tok[o_old[i]]
        print(f"[{src}] cursor {cursor}; old list len {len(o_old)}; shard {os.path.basename(o_old[i])}"
              f" n_tokens {size:,}; rows expected {expect:,}, via old order {got_old:,}, "
              f"via current order {got_cur:,}; same shard at cursor: {o_old[i] == o_cur[i]}")
        if i >= len(o_old) or off % T or off + T + 1 > size or got_old != expect:
            fails.append(f"{src}: cursor {cursor} inconsistent with old order "
                         f"(rows {got_old} vs expected {expect}, shard size {size})")


def shards_needed(loader: MixLoader, n_tok: dict, n_batches: int) -> set:
    """Shards the next n_batches read, simulated from n_tokens without opening files."""
    need = set()
    for src, rows in loader.rows.items():
        i, off = loader.cursors[src]
        for _ in range(rows * n_batches):
            while off + loader.T + 1 > n_tok[loader.shards[src][i]]:
                i, off = i + 1, 0
            need.add(loader.shards[src][i])
            off += loader.T
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work", required=True, help="local scratch directory")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    work, fails = args.work, []

    # --- fetch --------------------------------------------------------------
    modal_get(RUNS_VOL, f"/{args.run}/latest.json", f"{work}/latest.json")
    with open(f"{work}/latest.json") as f:
        rel = json.load(f)["path"]
    name = os.path.basename(rel)
    ckpt_path = f"{work}/{name}"
    modal_get(RUNS_VOL, f"/{args.run}/{rel}", ckpt_path)
    local_root = f"{work}/shards/{TOK}"
    modal_get(DATA_VOL, f"{VOL_ROOT.removeprefix('/data')}/manifest.json",
              f"{local_root}/manifest.json")
    for src in OLD_COUNTS:
        out = subprocess.run(["modal", "volume", "ls", "--json", DATA_VOL,
                              f"{VOL_ROOT.removeprefix('/data')}/{src}/meta"],
                             check=True, capture_output=True, text=True).stdout
        with open(f"{work}/ls_{src}_meta.json", "w") as f:
            f.write(out)
    orig_sha = sha256(ckpt_path)
    print(f"checkpoint {rel} sha256 {orig_sha[:16]}")

    # --- reconstruct and prove the old order -----------------------------------
    cur_local = read_manifest(f"{local_root}/manifest.json")
    to_vol = lambda p: VOL_ROOT + p.removeprefix(local_root)
    to_local = lambda p: local_root + p.removeprefix(VOL_ROOT)
    old_local = old_manifest(cur_local)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["loader_state"]
    if "shards" in state:
        raise SystemExit("loader_state already has pinned shards; nothing to do")
    tc = TrainConfig(**ckpt["train_cfg"])
    mc = ModelConfig(**ckpt["model_cfg"])
    T, total, step = mc.ctx, total_steps(tc, mc), ckpt["step"]
    check_old_set(cur_local, work, fails)
    check_cursors(cur_local, old_local, state, tc, total, step, T, fails)

    # --- pinned state: every source, old order, volume paths --------------------
    pinned = {src: [to_vol(p) for p in _shard_order(s["train"], state["seed"], src, shuffle=True)]
              for src, s in old_manifest(cur_local).items() if s.get("train")}
    new_state = {**state, "shards": pinned}
    new_ckpt = {**ckpt, "loader_state": new_state}
    pinned_path = ckpt_path.removesuffix(".pt") + ".pinned.pt"
    torch.save(new_ckpt, pinned_path)

    # --- verify ---------------------------------------------------------------
    reloaded = torch.load(pinned_path, map_location="cpu", weights_only=True)  # as train.load_latest
    same = lambda a, b: (torch.equal(a, b) if isinstance(a, torch.Tensor) else
                         a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
                         if isinstance(a, dict) else
                         len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
                         if isinstance(a, (list, tuple)) else a == b)
    for k in ckpt:
        ok = same(reloaded[k], new_ckpt[k] if k == "loader_state" else ckpt[k])
        if not ok:
            fails.append(f"pinned checkpoint key {k!r} differs")
    if reloaded["loader_state"] != new_state or set(reloaded) != set(ckpt):
        fails.append("pinned loader_state is not the original plus 'shards'")
    print(f"pinned file reloads with weights_only=True; all {len(ckpt)} keys equal except "
          f"loader_state['shards'] added: {not fails}")

    local_state = {**new_state, "shards": {s: [to_local(p) for p in ps] for s, ps in pinned.items()}}
    mix, B = tc.mix_anneal, tc.micro_bsz
    ref = MixLoader(old_local, mix, B, T, state=state)            # what the run would read
    pin = MixLoader(cur_local, mix, B, T, state=local_state)       # what a resume will read
    naive = MixLoader(cur_local, mix, B, T, state=state)           # the bug
    print(f"pinned loader loads; mix {pin.mix}; rows {pin.rows}")
    for src in OLD_COUNTS:
        i = state["cursors"][src][0]
        print(f"[{src}] shard at cursor: old {os.path.basename(ref.shards[src][i])}, "
              f"pinned {os.path.basename(pin.shards[src][i])}, "
              f"unpinned-current {os.path.basename(naive.shards[src][i])}")
        if pin.shards[src][i] != ref.shards[src][i]:
            fails.append(f"{src}: pinned shard at cursor differs from old order")
    if pin.shards != ref.shards:
        fails.append("pinned shard lists differ from the old loader's")
    print(f"full shard lists identical (pinned vs old): {pin.shards == ref.shards}; "
          f"(unpinned vs old): {naive.shards == ref.shards}")

    n_tok = {e["path"]: e["n_tokens"] for s in cur_local.values() for e in s["train"]}
    need = shards_needed(ref, n_tok, tc.grad_accum) | shards_needed(naive, n_tok, 1)
    for p in sorted(need):
        modal_get(DATA_VOL, to_vol(p).removeprefix("/data"), p)
    diff_naive = False
    for b in range(tc.grad_accum):  # one full optimizer step of micro-batches
        (x1, y1), (x2, y2) = ref.next_batch(), pin.next_batch()
        if b == 0:
            x3, _ = naive.next_batch()
            diff_naive = not torch.equal(x1, x3)
        if not (torch.equal(x1, x2) and torch.equal(y1, y2)):
            fails.append(f"micro-batch {b}: pinned != old")
    if ref.state()["cursors"] != pin.state()["cursors"]:
        fails.append("cursors diverge after the step")
    print(f"next {tc.grad_accum} micro-batches identical (pinned/current vs old/old): "
          f"{not any('micro-batch' in f for f in fails)}; unpinned batch differs: {diff_naive}")

    try:
        check_data_supply(MixLoader(cur_local, mix, B, T, state=local_state), cur_local, tc,
                          total, step)
        print(f"check_data_supply from step {step}: OK")
    except RuntimeError as e:
        fails.append(f"check_data_supply: {e}")
    for p in need:
        os.remove(p)

    if fails:
        print("FAILED, checkpoint NOT replaced:\n  " + "\n  ".join(fails))
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
    if not args.upload:
        return

    # --- upload: pinned copy, then over the original; keep a local .bak --------
    shutil.copy2(ckpt_path, ckpt_path + ".bak")
    remote_dir = f"{args.run}/{os.path.dirname(rel)}"
    modal_put(RUNS_VOL, pinned_path, f"{remote_dir}/{os.path.basename(pinned_path)}")
    modal_put(RUNS_VOL, pinned_path, f"{remote_dir}/{name}", force=True)
    back = f"{work}/roundtrip_{name}"
    subprocess.run(["modal", "volume", "get", "--force", RUNS_VOL, f"/{args.run}/{rel}", back],
                   check=True, capture_output=True)
    ok = sha256(back) == sha256(pinned_path)
    os.remove(back)
    print(f"uploaded; {remote_dir}/{name} round-trip sha256 matches pinned: {ok}; "
          f"original kept at {ckpt_path}.bak (sha256 {orig_sha[:16]})")
    if not ok:
        raise SystemExit("round-trip mismatch after upload")


if __name__ == "__main__":
    main()
