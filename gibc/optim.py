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


# --- NorMuon (arXiv 2510.05491), following modded-nanogpt's NorMuonAndAdam -------------
NS_COEFFS = (3.4445, -4.7750, 2.0315)  # same quintic as torch.optim.Muon


def orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz in bf16, written to match torch.optim.Muon's iteration exactly."""
    a, b, c = NS_COEFFS
    tall = g.size(0) > g.size(1)
    x = g.bfloat16()
    x = x.T if tall else x
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if tall else x


def normalize_neurons(o: torch.Tensor, second_moment: torch.Tensor, beta2: float) -> torch.Tensor:
    """EMA the mean square of o per slice of its larger dimension (the smaller one is
    already near-orthonormal after NS), divide by its sqrt, then restore o's Frobenius norm
    so the Muon lr scale is unchanged. Updates second_moment in place."""
    o = o.float()
    reduce_dim = -1 if o.size(0) >= o.size(1) else -2
    second_moment.lerp_(o.square().mean(dim=reduce_dim, keepdim=True).to(second_moment), 1 - beta2)
    out = o * second_moment.float().clamp_min(1e-10).rsqrt()
    return out * (o.norm() / out.norm().clamp_min(1e-10))


def apply_weight_decay(p: torch.Tensor, update: torch.Tensor, decay: float, cautious: bool) -> None:
    """p -= decay * p, where cautious (arXiv 2510.12402) keeps it only on coordinates
    with update * p >= 0; `update` is the step that will be subtracted from p."""
    if cautious:
        p.sub_(p * (update * p >= 0) * decay)
    else:
        p.mul_(1 - decay)


class NorMuon(torch.optim.Optimizer):
    """Single-device NorMuon for 2D params: Nesterov momentum -> Newton-Schulz -> per-neuron
    second-moment normalisation -> step with lr * sqrt(max(1, rows/cols)).
    Defaults are the modded-nanogpt record's normuon_defaults."""

    def __init__(self, params, lr: float = 0.023, momentum: float = 0.95, beta2: float = 0.9,
                 weight_decay: float = 1.2, cautious_wd: bool = False, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay,
                        cautious_wd=cautious_wd, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if any(p.ndim != 2 for p in group["params"]):
                raise ValueError("NorMuon only supports 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    rows, cols = p.shape
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_moment"] = p.new_zeros((rows, 1) if rows >= cols else (1, cols))
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - momentum)
                o = orthogonalize(p.grad.lerp(buf, momentum), group["ns_steps"])
                o = normalize_neurons(o, state["second_moment"], group["beta2"]).to(p.dtype)
                apply_weight_decay(p, o, lr * group["weight_decay"], group["cautious_wd"])
                p.add_(o, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)
        return loss


def build_optimizers(model: nn.Module, tc: TrainConfig) -> list[torch.optim.Optimizer]:
    if tc.cautious_wd and tc.opt != "normuon":
        raise ValueError(f"cautious_wd is only implemented for opt='normuon', not {tc.opt!r}")
    matrices, others = split_params(model)
    fused = all(p.is_cuda for p in model.parameters())
    if tc.opt in ("muon", "normuon"):
        if tc.opt == "muon":
            matrix_opt = torch.optim.Muon(matrices, lr=tc.muon_lr, weight_decay=tc.muon_wd,
                                          momentum=0.95, nesterov=True, adjust_lr_fn="original")
        else:
            matrix_opt = NorMuon(matrices, lr=tc.muon_lr, weight_decay=tc.muon_wd,
                                 cautious_wd=tc.cautious_wd)
        adam = torch.optim.AdamW(others, lr=tc.adam_lr, betas=tc.adam_betas,
                                 weight_decay=tc.adam_wd, fused=fused)
        optimizers = [matrix_opt, adam]
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
