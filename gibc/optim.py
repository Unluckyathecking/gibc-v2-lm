"""Optimizers and the warmup / constant / linear-decay LR schedule.

Steps are 0-indexed: the trainer runs steps 0 .. total_steps-1 and the LR multiplier
is exactly 0 at step total_steps-1.
"""
import torch
import torch.nn as nn

from gibc.configs import TrainConfig


def split_params(model: nn.Module) -> tuple[list, list]:
    """(2D matrices inside blocks, everything else: embedding, norm weights, skip scalars)."""
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    others = [p for p in model.parameters() if id(p) not in matrix_ids]
    return matrices, others


def build_optimizers(model: nn.Module, tc: TrainConfig) -> list[torch.optim.Optimizer]:
    matrices, others = split_params(model)
    fused = all(p.is_cuda for p in model.parameters())
    if tc.opt == "muon":
        muon = torch.optim.Muon(matrices, lr=tc.muon_lr, weight_decay=tc.muon_wd,
                                momentum=0.95, nesterov=True, adjust_lr_fn="original")
        adam = torch.optim.AdamW(others, lr=tc.adam_lr, betas=tc.adam_betas,
                                 weight_decay=tc.adam_wd, fused=fused)
        optimizers = [muon, adam]
    elif tc.opt == "adamw":
        groups = [{"params": matrices, "weight_decay": tc.muon_wd},
                  {"params": others, "weight_decay": 0.0}]
        optimizers = [torch.optim.AdamW(groups, lr=tc.adam_lr, betas=tc.adam_betas,
                                        fused=fused)]
    else:
        raise ValueError(f"unknown opt {tc.opt!r}")
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def decay_start(total_steps: int, decay_frac: float) -> int:
    """First step of the linear decay (the trainer switches to the anneal mix here)."""
    return total_steps - int(decay_frac * total_steps)


def lr_mult(step: int, total_steps: int, warmup_frac: float, decay_frac: float) -> float:
    """Warmup (step+1)/W over the first W steps, then 1, then linear from 1 at
    decay_start to exactly 0 at step total_steps-1."""
    warmup_steps = int(warmup_frac * total_steps)
    warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    start, last = decay_start(total_steps, decay_frac), total_steps - 1
    if step < start:
        return warm
    decay = (last - step) / (last - start) if last > start else 0.0
    return min(warm, max(0.0, decay))


def apply_lr(optimizers: list[torch.optim.Optimizer], mult: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * mult
