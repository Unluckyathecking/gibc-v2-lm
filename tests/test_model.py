import math

import pytest
import torch

from gibc.configs import MODEL_CONFIGS, PARAM_CAP, ModelConfig
from gibc.model import GPT, count_params


def tiny(repeats: int = 1, n_unique_blocks: int = 2) -> ModelConfig:
    return ModelConfig(vocab=256, d_model=64, n_unique_blocks=n_unique_blocks, repeats=repeats,
                       n_head=4, n_kv_head=2, mlp_hidden=128, ctx=32)


@pytest.mark.parametrize("repeats", [1, 2])
def test_forward_backward(repeats):
    torch.manual_seed(0)
    cfg = tiny(repeats)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, cfg.vocab) and logits.dtype == torch.float32
    assert logits.abs().max() <= cfg.softcap
    loss = model(idx, torch.randint(0, cfg.vocab, (2, 16)))
    assert loss.ndim == 0 and loss.dtype == torch.float32
    assert abs(loss.item() - math.log(cfg.vocab)) < 0.1
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_initial_loss_full_size():
    torch.manual_seed(0)
    cfg = MODEL_CONFIGS["A"]
    idx = torch.randint(0, cfg.vocab, (1, 64))
    loss = GPT(cfg)(idx, torch.randint(0, cfg.vocab, (1, 64)))
    assert abs(loss.item() - math.log(cfg.vocab)) < 0.1


def test_causal():
    torch.manual_seed(0)
    model = GPT(tiny())
    for p in model.parameters():  # break the zero-init so attention actually mixes tokens
        torch.nn.init.normal_(p, std=0.1)
    idx = torch.randint(0, 256, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10:] = (idx2[0, 10:] + 1) % 256
    assert torch.allclose(model(idx)[:, :10], model(idx2)[:, :10], atol=1e-5)
    assert not torch.allclose(model(idx)[:, 10:], model(idx2)[:, 10:])


def test_state_dict_has_no_rope_buffers():
    assert not any("rope" in k for k in GPT(tiny()).state_dict())


@pytest.mark.parametrize("name,once,twice", [
    ("A", 49_822_215, None),
    ("alt1", 47_195_720, None),
    ("alt2", 41_433_607, 49_822_215),
])
def test_count_params(name, once, twice):
    counts = count_params(MODEL_CONFIGS[name])
    assert counts["once"] == once and counts["once"] <= PARAM_CAP
    if twice is not None:
        assert counts["twice"] == twice
    assert counts["embedding"] + counts["non_embedding"] == counts["once"]


@pytest.mark.parametrize("n_unique,repeats", [(2, 1), (2, 2), (3, 1), (3, 3)])
def test_skip_count(n_unique, repeats):
    cfg = tiny(repeats, n_unique)
    assert GPT(cfg).skip_w.numel() == cfg.n_eff // 2


def test_repeats_share_weights():
    one, two = count_params(tiny(1)), count_params(tiny(2))
    assert two["once"] - one["once"] == tiny(2).n_eff // 2 - tiny(1).n_eff // 2
    alt1 = MODEL_CONFIGS["alt1"]
    unshared = count_params(ModelConfig(**{**alt1.__dict__, "repeats": 1}))
    assert count_params(alt1)["once"] - unshared["once"] == alt1.n_eff // 2 - 8 // 2
