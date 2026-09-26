"""Training loop: bf16 autocast + torch.compile on CUDA, grad accumulation, wall-clock
checkpoints with exact resume, decay-phase snapshots, JSONL log and status file.

Run layout under {runs_dir}/{run}/:
  log.jsonl        one JSON line per step (val_bpb/val_loss on val steps and the last step)
  status.json      state, progress and time/compute totals (all attempts)
  latest.json      {"step", "path"} of the newest full checkpoint
  ckpt/step_N.pt   full state (last KEEP_CKPTS kept)
  snap/step_N.pt   model-only snapshots during LR decay, for checkpoint averaging
  final.pt         model-only weights after the last step
Model-only files are {"model": state_dict, "model_cfg": asdict(ModelConfig)}.
"""
import glob
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import torch

from gibc.configs import MODEL_CONFIGS, RUNS, RUNS_DIR, ModelConfig, TrainConfig
from gibc.data import MixLoader, load_manifest, val_bpb
from gibc.model import GPT, count_params
from gibc.optim import apply_lr, build_optimizers, decay_start, lr_mult

GRAD_CLIP = 1.0
KEEP_CKPTS = 2
VAL_BSZ = 16  # eager val materialises fp32 logits: 16 x 1024 x 32768 x 4B = 2 GB per batch


# --- small helpers -------------------------------------------------------------
def total_steps(tc: TrainConfig, mc: ModelConfig) -> int:
    return tc.total_tokens // (tc.micro_bsz * tc.grad_accum * mc.ctx)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _write_json(path: str, obj: dict) -> None:
    """Atomic: a crash mid-write never leaves a truncated file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_pt(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- one optimizer step ------------------------------------------------------------
def train_step(model, raw, opts, next_batch, grad_accum: int, device) -> float:
    """grad_accum micro-batches, clip, step. Returns the mean micro-batch loss."""
    loss_sum = torch.zeros((), device=device)
    for _ in range(grad_accum):
        x, y = next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device):
            loss = model(x, y)
        (loss / grad_accum).backward()
        loss_sum += loss.detach()
    torch.nn.utils.clip_grad_norm_(raw.parameters(), GRAD_CLIP)
    for opt in opts:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return (loss_sum / grad_accum).item()


def run_val(raw, manifest, tc: TrainConfig, mc: ModelConfig, device) -> dict:
    """Exact bits-per-byte on the val split, uncompiled model in eval mode."""
    raw.eval()
    with torch.no_grad(), _autocast(device):
        res = val_bpb(raw, manifest, min(tc.micro_bsz, VAL_BSZ), mc.ctx, tc.val_tokens, device)
    raw.train()
    return {"val_bpb": res["bpb"], "val_loss": res["loss"]}


# --- checkpoints -----------------------------------------------------------------
def _rng_state(device) -> dict:
    # The loader carries its own cursors and nothing else draws from python/numpy RNG.
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def _set_rng_state(rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def save_checkpoint(rd, raw, opts, loader, prog: dict, mc, tc, device) -> None:
    """Full state -> ckpt/step_N.pt, then latest.json, then prune to KEEP_CKPTS."""
    rel = f"ckpt/step_{prog['step']:06d}.pt"
    _save_pt({"model": raw.state_dict(), "optims": [o.state_dict() for o in opts],
              "step": prog["step"], "tokens_seen": prog["tokens_seen"],
              "elapsed_s": prog["elapsed_s"], "loader_state": loader.state(),
              "model_cfg": asdict(mc), "train_cfg": asdict(tc), "rng": _rng_state(device)},
             os.path.join(rd, rel))
    _write_json(os.path.join(rd, "latest.json"), {"step": prog["step"], "path": rel})
    for old in sorted(glob.glob(os.path.join(rd, "ckpt", "step_*.pt")))[:-KEEP_CKPTS]:
        os.remove(old)


def save_weights(raw, mc: ModelConfig, path: str) -> None:
    _save_pt({"model": raw.state_dict(), "model_cfg": asdict(mc)}, path)


def load_latest(rd: str) -> dict | None:
    latest = _read_json(os.path.join(rd, "latest.json"))
    if latest is None:
        return None
    return torch.load(os.path.join(rd, latest["path"]), map_location="cpu", weights_only=True)


# --- status / log ----------------------------------------------------------------
def _truncate_log(rd: str, step: int) -> None:
    """On resume, drop log lines past the checkpoint; those steps are about to rerun."""
    path = os.path.join(rd, "log.jsonl")
    if not os.path.exists(path):
        return
    with open(path) as f:
        keep = [line for line in f if line.strip() and json.loads(line)["step"] <= step]
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


class RunLog:
    """Appends log.jsonl and rewrites status.json. GPU time accumulates across attempts."""

    def __init__(self, rd: str, total: int):
        self.rd, self.total, self.t0 = rd, total, time.time()
        prev = _read_json(os.path.join(rd, "status.json")) or {}
        self.prev_gpu_s = prev.get("gpu_s", 0.0)
        self.last = {"loss": None, "val_bpb": prev.get("val_bpb")}

    def log(self, rec: dict) -> None:
        with open(os.path.join(self.rd, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.last["loss"] = rec["loss"]
        if "val_bpb" in rec:
            self.last["val_bpb"] = rec["val_bpb"]

    def status(self, state: str, prog: dict, **extra) -> dict:
        gpu_s = self.prev_gpu_s + time.time() - self.t0
        st = {"state": state, "step": prog["step"], "total_steps": self.total,
              "tokens_seen": prog["tokens_seen"], "loss": self.last["loss"],
              "val_bpb": self.last["val_bpb"], "elapsed_s": round(prog["elapsed_s"], 1),
              "gpu_s": round(gpu_s, 1), "gpu_hours": round(gpu_s / 3600, 3),
              "updated": _now(), **extra}
        _write_json(os.path.join(self.rd, "status.json"), st)
        return st


# --- train -------------------------------------------------------------------------
def train(run_name: str, runs_dir: str = RUNS_DIR, data_manifest: dict | None = None,
          max_steps: int | None = None, on_checkpoint=None,
          tc: TrainConfig | None = None, mc: ModelConfig | None = None) -> dict:
    """Train RUNS[run_name] (or the tc/mc overrides), resuming from latest.json if present.

    max_steps stops early at that absolute step (state "stopped", checkpoint saved).
    on_checkpoint() is called after every checkpoint (Modal passes runs_vol.commit).
    Returns the final status dict.
    """
    tc = tc or RUNS[run_name]
    mc = mc or MODEL_CONFIGS[tc.model]
    rd = os.path.join(runs_dir, run_name)
    os.makedirs(rd, exist_ok=True)
    total = total_steps(tc, mc)
    runlog = RunLog(rd, total)
    prog = {"step": 0, "tokens_seen": 0, "elapsed_s": 0.0}
    try:
        return _train(rd, tc, mc, data_manifest, max_steps, on_checkpoint, runlog, prog)
    except BaseException as e:
        runlog.status("failed", prog, error=repr(e))
        raise


def _train(rd, tc, mc, manifest, max_steps, on_checkpoint, runlog, prog) -> dict:
    device = _device()
    manifest = manifest or load_manifest(mc.tokenizer)
    total, ds = runlog.total, decay_start(runlog.total, tc.decay)
    ckpt = load_latest(rd)

    torch.manual_seed(tc.seed)
    raw = GPT(mc).to(device)
    opts = build_optimizers(raw, tc)
    loader = MixLoader(manifest, tc.mix_main, tc.micro_bsz, mc.ctx, split="train",
                       state=ckpt["loader_state"] if ckpt else None, seed=tc.seed)
    if ckpt:
        raw.load_state_dict(ckpt["model"])
        for opt, sd in zip(opts, ckpt["optims"], strict=True):
            opt.load_state_dict(sd)
        prog.update(step=ckpt["step"], tokens_seen=ckpt["tokens_seen"],
                    elapsed_s=ckpt["elapsed_s"])
        _set_rng_state(ckpt["rng"])
        _truncate_log(rd, prog["step"])
        if tc.mix_anneal and prog["step"] > ds:
            loader.set_mix(tc.mix_anneal)
    model = torch.compile(raw, dynamic=False) if device.type == "cuda" else raw
    return _loop(rd, raw, model, opts, loader, manifest, tc, mc, device,
                 min(total, max_steps or total), on_checkpoint, runlog, prog)


def _loop(rd, raw, model, opts, loader, manifest, tc, mc, device, stop, on_checkpoint,
          runlog, prog) -> dict:
    total, ds = runlog.total, decay_start(runlog.total, tc.decay)
    tok_per_step = tc.micro_bsz * tc.grad_accum * mc.ctx
    t_start = t_last = t_ckpt = time.time()
    elapsed0 = prog["elapsed_s"]
    status_every = max(1, min(tc.val_every, 10))

    def checkpoint():
        save_checkpoint(rd, raw, opts, loader, prog, mc, tc, device)
        runlog.status("running", prog)
        if on_checkpoint:
            on_checkpoint()

    while prog["step"] < stop:
        step = prog["step"]
        if tc.mix_anneal and step == ds:
            loader.set_mix(tc.mix_anneal)
        mult = lr_mult(step, total, tc.warmup, tc.decay)
        apply_lr(opts, mult)
        loss = train_step(model, raw, opts, loader.next_batch, tc.grad_accum, device)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss} at step {step}")

        now = time.time()
        prog.update(step=step + 1, tokens_seen=prog["tokens_seen"] + tok_per_step,
                    elapsed_s=elapsed0 + now - t_start)
        rec = {"step": step + 1, "tokens_seen": prog["tokens_seen"], "loss": loss,
               "lr_mult": mult, "tok_per_s": tok_per_step / (now - t_last),
               "elapsed_s": round(prog["elapsed_s"], 2)}
        if (step + 1) % tc.val_every == 0 or step + 1 == total:
            rec.update(run_val(raw, manifest, tc, mc, device))
        runlog.log(rec)

        if step + 1 > ds and (step + 1) % tc.snap_every == 0:
            save_weights(raw, mc, os.path.join(rd, "snap", f"step_{step + 1:06d}.pt"))
        if time.time() - t_ckpt >= tc.ckpt_minutes * 60 and step + 1 < stop:
            checkpoint()
            t_ckpt = time.time()
        elif (step + 1) % status_every == 0:
            runlog.status("running", prog)
        t_last = time.time()  # tok/s covers the train step only, not val or saves
    return _finish(rd, raw, mc, total, checkpoint, runlog, prog)


def _finish(rd, raw, mc, total, checkpoint, runlog, prog) -> dict:
    checkpoint()
    if prog["step"] < total:
        return runlog.status("stopped", prog)
    save_weights(raw, mc, os.path.join(rd, "final.pt"))
    return runlog.status("done", prog, params=count_params(mc))


# --- smoke test: throughput on random tokens -------------------------------------------
def smoke(model_key: str = "A", minutes: float = 5.0,
          tc: TrainConfig | None = None, mc: ModelConfig | None = None) -> dict:
    """Train on random tokens for `minutes`; report tok/s after compile, peak memory."""
    mc = mc or MODEL_CONFIGS[model_key]
    tc = tc or TrainConfig(model_key, total_tokens=0)
    device = _device()
    torch.manual_seed(tc.seed)
    raw = GPT(mc).to(device)
    opts = build_optimizers(raw, tc)
    model = torch.compile(raw, dynamic=False) if device.type == "cuda" else raw

    def next_batch():
        x = torch.randint(0, mc.vocab, (tc.micro_bsz, mc.ctx + 1))
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    t0 = time.time()
    loss = train_step(model, raw, opts, next_batch, tc.grad_accum, device)  # compiles
    compile_s = time.time() - t0
    steps, t1 = 0, time.time()
    while time.time() - t0 < minutes * 60 or steps == 0:
        loss = train_step(model, raw, opts, next_batch, tc.grad_accum, device)
        steps += 1
    run_s = time.time() - t1
    tok_per_step = tc.micro_bsz * tc.grad_accum * mc.ctx
    return {"model": model_key, "device": torch.cuda.get_device_name() if device.type == "cuda"
            else "cpu", "params": count_params(mc), "micro_bsz": tc.micro_bsz,
            "grad_accum": tc.grad_accum, "ctx": mc.ctx, "compile_s": round(compile_s, 1),
            "steps": steps, "tok_per_s": round(steps * tok_per_step / run_s),
            "step_s": round(run_s / steps, 4), "loss": loss, "loss_finite": math.isfinite(loss),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if device.type == "cuda" else None}
