"""Cap-lever options: factorised tied embedding (embed_dim) and BitNet b1.58 ternary QAT."""
import math

import pytest
import torch

from gibc.configs import LEVER_RUNS, MODEL_CONFIGS, PARAM_CAP, RUNS, SWEEP_RUNS, ModelConfig, TrainConfig
from gibc.model import GPT, BitLinear, act_quant8, count_params, ternary_weight
from gibc.optim import build_optimizers, split_params


def tiny(**kw) -> ModelConfig:
    base = dict(vocab=256, d_model=64, n_unique_blocks=2, repeats=1, n_head=4, n_kv_head=2,
                mlp_hidden=128, ctx=32, value_residual=True)
    return ModelConfig(**{**base, **kw})


@pytest.mark.parametrize("name,once,layers", [
    ("fact128", 49_729_051, 19), ("fact256", 49_759_768, 17), ("ternary", 49_822_228, 14)])
def test_lever_param_counts(name, once, layers):
    cfg = MODEL_CONFIGS[name]
    counts = count_params(cfg)
    assert counts["once"] == once and 49_300_000 <= once <= 49_830_000 and once < PARAM_CAP
    assert cfg.n_eff == layers and cfg.value_residual
    if cfg.embed_dim:
        assert counts["embedding"] == cfg.vocab * cfg.embed_dim + cfg.d_model * cfg.embed_dim


def test_ternary_same_count_as_a_vres():
    assert count_params(MODEL_CONFIGS["ternary"]) == count_params(MODEL_CONFIGS["A_vres"])


def test_lever_runs_match_a_vres_recipe():
    ref = RUNS["sweep_A_vres"].__dict__
    for run in LEVER_RUNS:
        assert {k: v for k, v in RUNS[run].__dict__.items() if k != "model"} == \
               {k: v for k, v in ref.items() if k != "model"}
        assert run not in SWEEP_RUNS  # the existing "sweep" group is unchanged


def test_flags_off_identical():
    """embed_dim=0 and ternary=False build exactly the existing model."""
    torch.manual_seed(0)
    a = GPT(tiny())
    torch.manual_seed(0)
    b = GPT(tiny(embed_dim=0, ternary=False))
    assert a.state_dict().keys() == b.state_dict().keys()
    assert not any(isinstance(m, BitLinear) for m in b.modules())
    idx = torch.randint(0, 256, (2, 16))
    assert torch.equal(a(idx), b(idx))
    for name in ("A", "A_vres", "alt1", "alt2"):
        assert MODEL_CONFIGS[name].embed_dim == 0 and not MODEL_CONFIGS[name].ternary


def test_ternary_state_dict_matches_dense():
    dense, tern = GPT(tiny()), GPT(tiny(ternary=True))
    assert {k: v.shape for k, v in dense.state_dict().items()} == \
           {k: v.shape for k, v in tern.state_dict().items()}


def test_factorised_embedding_routes_p_to_muon():
    model = GPT(tiny(embed_dim=16))
    assert model.wte.weight.shape == (256, 16) and model.embed_proj.weight.shape == (64, 16)
    matrices, others = split_params(model)
    assert id(model.embed_proj.weight) in {id(p) for p in matrices}
    assert id(model.wte.weight) in {id(p) for p in others}
    muon, adam = build_optimizers(model, TrainConfig("A", 10**9))
    assert isinstance(muon, torch.optim.Muon)
    assert id(model.embed_proj.weight) in {id(p) for g in muon.param_groups for p in g["params"]}


def test_ternary_weight_values():
    w = torch.randn(32, 16)
    q = ternary_weight(w)
    scale = w.abs().mean()
    assert set((q / scale).round().unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.allclose(q / scale, (q / scale).round(), atol=1e-5)
    assert torch.equal(ternary_weight(torch.zeros(4, 4)), torch.zeros(4, 4))


def test_act_quant8_levels():
    x = torch.randn(3, 5, 64)
    q = act_quant8(x)
    scale = 127.0 / x.abs().amax(-1, keepdim=True)
    levels = q * scale
    assert torch.allclose(levels, levels.round(), atol=1e-3) and levels.abs().max() <= 127.001
    assert (q - x).abs().max() <= (0.5 / scale).max() + 1e-6


@pytest.mark.parametrize("kw", [dict(embed_dim=16), dict(embed_dim=32, repeats=2),
                                dict(ternary=True), dict(ternary=True, ternary_act8=False)])
def test_lever_forward_backward(kw):
    torch.manual_seed(0)
    cfg = tiny(**kw)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, cfg.vocab) and logits.abs().max() <= cfg.softcap
    loss = model(idx, torch.randint(0, cfg.vocab, (2, 16)))
    assert abs(loss.item() - math.log(cfg.vocab)) < 0.1
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_ternary_ste_gradient_nonzero():
    """Latent weights get gradients through round(): the STE passes them straight through."""
    torch.manual_seed(0)
    model = GPT(tiny(ternary=True))
    for p in model.blocks.parameters():  # break the zero-init so every matrix is on the path
        torch.nn.init.normal_(p, std=0.1)
    idx = torch.randint(0, 256, (2, 16))
    model(idx, torch.randint(0, 256, (2, 16))).backward()
    for name, p in model.blocks.named_parameters():
        if p.ndim == 2:
            assert p.grad.abs().sum() > 0, name
    lin = model.blocks[0].mlp.w_up
    x = torch.randn(4, 64)
    assert torch.allclose(lin(x), torch.nn.functional.linear(act_quant8(x), ternary_weight(lin.weight)),
                          atol=1e-5)
