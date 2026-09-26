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


# --- XSA and value residual --------------------------------------------------
def flagged(xsa=False, value_residual=False, **kw) -> ModelConfig:
    return ModelConfig(**{**tiny(**kw).__dict__, "xsa": xsa, "value_residual": value_residual})


def randomized(model: GPT, seed: int = 0) -> GPT:
    """Break the zero-init (Wo, W_down) so attention and MLPs actually contribute."""
    torch.manual_seed(seed)
    for p in model.blocks.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return model


def test_flags_off_unchanged():
    torch.manual_seed(0)
    a = GPT(tiny())
    torch.manual_seed(0)
    b = GPT(flagged())
    assert a.state_dict().keys() == b.state_dict().keys()
    assert not any("xsa" in k or "vres" in k for k in b.state_dict())
    idx = torch.randint(0, 256, (2, 16))
    assert torch.equal(a(idx), b(idx))
    assert count_params(flagged())["once"] == count_params(tiny())["once"]


@pytest.mark.parametrize("repeats", [1, 2])
def test_flags_on_at_neutral_equal_baseline(repeats):
    """xsa gate 0 (its init) and lambda 1 reproduce the flags-off model exactly."""
    base = randomized(GPT(tiny(repeats)))
    model = GPT(flagged(True, True, repeats=repeats))
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert sorted(missing) == ["vres_lambda", "xsa_alpha"] and not unexpected
    with torch.no_grad():
        model.vres_lambda.fill_(1.0)
    idx = torch.randint(0, 256, (2, 16))
    assert torch.equal(model(idx), base(idx))


@pytest.mark.parametrize("n_head,n_kv_head", [(1, 1), (4, 2)])
def test_xsa_output_orthogonal_to_own_value(n_head, n_kv_head):
    torch.manual_seed(0)
    d = 16 * n_head
    cfg = ModelConfig(vocab=256, d_model=d, n_unique_blocks=1, repeats=1, n_head=n_head,
                      n_kv_head=n_kv_head, mlp_hidden=32, ctx=32, xsa=True)
    model = randomized(GPT(cfg))
    attn = model.blocks[0].attn
    with torch.no_grad():
        attn.wo.weight.copy_(torch.eye(d))  # attention output == pre-Wo y
    x = torch.randn(2, 12, d)
    alpha = torch.full((n_head,), 20.0)  # tanh(20) == 1 in fp32: full exclusion
    y, _ = attn(x, model.rope_cos, model.rope_sin, xsa_alpha=alpha)
    y = y.view(2, 12, n_head, cfg.head_dim)
    v = attn.wv(x).view(2, 12, n_kv_head, cfg.head_dim)
    v = v.repeat_interleave(n_head // n_kv_head, dim=2)  # query head h reads KV head h // group
    vn = torch.nn.functional.normalize(v, dim=-1)
    assert (y * vn).sum(-1).abs().max() < 1e-5
    y_plain, _ = attn(x, model.rope_cos, model.rope_sin)  # gate off: plain attention
    assert (y_plain.view_as(y) * vn).sum(-1).abs().max() > 1e-2  # so the test is not vacuous


def test_value_residual_lambda_zero_uses_layer0_v(monkeypatch):
    model = randomized(GPT(flagged(value_residual=True, repeats=2)))  # 4 effective layers
    seen = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append(v.detach().clone())
        return sdpa(q, k, v, **kw)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    idx = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        model.vres_lambda.zero_()
        model(idx)
        assert len(seen) == 4
        v0 = model.blocks[0].attn.wv(model.blocks[0].norm1(model.wte(idx)))
        assert torch.allclose(seen[0], v0.view(1, 16, 2, 16).transpose(1, 2))
        for v in seen[1:]:
            assert torch.equal(v, seen[0])
        seen.clear()
        model.vres_lambda.fill_(0.5)
        model(idx)
        assert not torch.allclose(seen[1], seen[0])


@pytest.mark.parametrize("name,once", [("A_xsa", 49_822_215 + 14 * 8), ("A_vres", 49_822_215 + 13)])
def test_count_params_flags(name, once):
    counts = count_params(MODEL_CONFIGS[name])
    assert counts["once"] == once and once < PARAM_CAP


def test_new_scalars_route_to_adamw():
    from gibc.optim import split_params
    model = GPT(flagged(True, True, repeats=2))
    matrices, others = split_params(model)
    assert model.xsa_alpha.ndim == 1 and model.vres_lambda.ndim == 1
    assert model.xsa_alpha.numel() == 4 * 4 and model.vres_lambda.numel() == 3
    other_ids = {id(p) for p in others}
    assert id(model.xsa_alpha) in other_ids and id(model.vres_lambda) in other_ids


@pytest.mark.parametrize("repeats", [1, 2])
def test_flags_forward_backward(repeats):
    torch.manual_seed(0)
    cfg = flagged(True, True, repeats=repeats)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab, (2, 16))
    loss = model(idx, torch.randint(0, cfg.vocab, (2, 16)))
    assert abs(loss.item() - math.log(cfg.vocab)) < 0.1  # init is still the no-op stack
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    model.zero_grad()
    randomized(model)(idx, torch.randint(0, cfg.vocab, (2, 16))).backward()
    for p in (model.xsa_alpha, model.vres_lambda):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
