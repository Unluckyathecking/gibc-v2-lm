import pytest
import torch

from gibc.configs import ModelConfig, TrainConfig
from gibc.model import GPT
from gibc.optim import apply_lr, build_optimizers, decay_start, lr_mult

CFG = ModelConfig(vocab=256, d_model=64, n_unique_blocks=2, repeats=2,
                  n_head=4, n_kv_head=2, mlp_hidden=128, ctx=32)


def block_matrix_ids(model):
    return {id(p) for p in model.blocks.parameters() if p.ndim == 2}


@pytest.mark.parametrize("opt", ["muon", "adamw"])
def test_grouping_and_step(opt):
    torch.manual_seed(0)
    model = GPT(CFG)
    optimizers = build_optimizers(model, TrainConfig("A", 10**9, opt=opt))
    seen = [id(p) for o in optimizers for g in o.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())  # each exactly once
    for o in optimizers:
        for g in o.param_groups:
            assert g["base_lr"] == g["lr"]
            if isinstance(o, torch.optim.Muon):
                assert {id(p) for p in g["params"]} == block_matrix_ids(model)
    if opt == "adamw":
        decayed = [g for g in optimizers[0].param_groups if g["weight_decay"] > 0]
        assert {id(p) for p in decayed[0]["params"]} == block_matrix_ids(model)

    idx = torch.randint(0, CFG.vocab, (2, 16))
    model(idx, idx).backward()
    for o in optimizers:
        o.step()
    assert model.blocks[0].attn.wo.weight.abs().sum() > 0  # zero-init weights moved


def test_apply_lr():
    optimizers = build_optimizers(GPT(CFG), TrainConfig("A", 10**9))
    apply_lr(optimizers, 0.5)
    for o in optimizers:
        for g in o.param_groups:
            assert g["lr"] == 0.5 * g["base_lr"]


def test_lr_mult_schedule():
    total, warm, dec = 1000, 0.01, 0.35
    start = decay_start(total, dec)
    assert start == 650
    assert lr_mult(0, total, warm, dec) == pytest.approx(0.1)  # (0+1)/10
    assert lr_mult(9, total, warm, dec) == 1.0                  # end of warmup
    assert lr_mult(400, total, warm, dec) == 1.0
    assert lr_mult(start, total, warm, dec) == 1.0
    assert lr_mult(total - 1, total, warm, dec) == 0.0
    decay = [lr_mult(s, total, warm, dec) for s in range(start, total)]
    assert all(a > b for a, b in zip(decay, decay[1:]))
