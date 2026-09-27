"""GDN hybrid on CPU: fla's GatedDeltaNet is swapped for a stub with the same parameter
names and shapes (fla 0.5.2), so config arithmetic, init and optimizer grouping are
checked without Triton. The real layer is exercised by the Modal smoke run."""
import math

import pytest
import torch
import torch.nn as nn

import gibc.model as gm
from gibc.configs import MODEL_CONFIGS, PARAM_CAP, RUNS, ModelConfig, TrainConfig
from gibc.optim import build_optimizers, split_params


class StubGDN(nn.Module):
    """Parameter layout of fla.layers.GatedDeltaNet(use_gate=True, use_short_conv=True)."""

    def __init__(self, d, heads, head_dim, expand_v, conv=4):
        super().__init__()
        key, value = heads * head_dim, int(heads * head_dim * expand_v)
        self.q_proj, self.k_proj = nn.Linear(d, key, bias=False), nn.Linear(d, key, bias=False)
        self.v_proj, self.g_proj = nn.Linear(d, value, bias=False), nn.Linear(d, value, bias=False)
        self.a_proj, self.b_proj = nn.Linear(d, heads, bias=False), nn.Linear(d, heads, bias=False)
        self.A_log, self.dt_bias = nn.Parameter(torch.zeros(heads)), nn.Parameter(torch.zeros(heads))
        self.q_conv1d = nn.Conv1d(key, key, conv, groups=key, bias=False)
        self.k_conv1d = nn.Conv1d(key, key, conv, groups=key, bias=False)
        self.v_conv1d = nn.Conv1d(value, value, conv, groups=value, bias=False)
        self.o_norm = nn.RMSNorm(int(head_dim * expand_v))
        self.o_proj = nn.Linear(value, d, bias=False)

    def forward(self, x):  # token-local stand-in; causality is trivially preserved
        o = self.o_norm(self.v_proj(x).unflatten(-1, (-1, self.o_norm.weight.numel())))
        o = o.flatten(-2) * torch.sigmoid(self.g_proj(x))
        return self.o_proj(o), None, None


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(gm, "gdn_mixer", lambda c: StubGDN(c.d_model, c.gdn_heads,
                                                           c.gdn_head_dim, c.gdn_expand_v))


def tiny_gdn(**kw) -> ModelConfig:
    return ModelConfig(vocab=256, d_model=64, n_unique_blocks=4, repeats=1, n_head=4,
                       n_kv_head=2, mlp_hidden=128, ctx=32, gdn_layers=(0, 1, 3),
                       gdn_heads=2, gdn_head_dim=16, **kw)


def test_config():
    cfg = MODEL_CONFIGS["gdn_hybrid"]
    assert [i for i in range(cfg.n_eff) if i not in cfg.gdn_layers] == [3, 7, 11]
    assert cfg.value_residual and cfg.n_eff == 14 and cfg.repeats == 1
    ref, tc = RUNS["sweep_A_vres"], RUNS["sweep_gdn_hybrid"]
    assert tc.model == "gdn_hybrid" and tc.__dict__ == {**ref.__dict__, "model": "gdn_hybrid"}


def test_param_count(stub):
    counts = gm.count_params(MODEL_CONFIGS["gdn_hybrid"])
    assert counts["once"] == 49_544_401 and counts["once"] <= PARAM_CAP
    gdn = 2 * 512 * 256 + 3 * 512 * 320 + 2 * 512 * 4 + 2 * 4 + 4 * (2 * 256 + 320) + 80
    assert gdn == 761_176
    # A_vres minus 11 x (attention - GDN) minus the 11 lambdas that no longer apply.
    assert counts["once"] == 49_822_228 - 11 * (786_432 - gdn) - 11


def test_forward_backward_and_value_residual(stub):
    torch.manual_seed(0)
    model = gm.GPT(tiny_gdn(value_residual=True))
    assert model.attn_layers == [2] and model.vres_lambda.numel() == 0
    model = gm.GPT(ModelConfig(**{**tiny_gdn(value_residual=True).__dict__, "gdn_layers": (1,)}))
    assert model.attn_layers == [0, 2, 3] and model.vres_lambda.numel() == 2
    assert isinstance(model.blocks[1], gm.GDNBlock) and isinstance(model.blocks[0], gm.Block)
    assert model.blocks[1].gdn.o_proj.weight.abs().sum() == 0  # zero-init output, like Wo
    idx = torch.randint(0, 256, (2, 16))
    loss = model(idx, torch.randint(0, 256, (2, 16)))
    assert abs(loss.item() - math.log(256)) < 0.1
    loss.backward()
    unused = ("q_proj", "k_proj", "a_proj", "b_proj", "A_log", "dt_bias", "conv1d")  # by the stub
    for name, p in model.named_parameters():
        if not any(u in name for u in unused):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_repeats_rejected(stub):
    with pytest.raises(ValueError):
        gm.GPT(ModelConfig(**{**tiny_gdn().__dict__, "n_unique_blocks": 2, "repeats": 2,
                              "gdn_layers": (0,)}))


def test_grouping(stub):
    model = gm.GPT(tiny_gdn(value_residual=True))
    matrices, others = split_params(model)
    assert all(p.ndim == 2 for p in matrices)
    gdn = model.blocks[0].gdn
    matrix_ids, other_ids = {id(p) for p in matrices}, {id(p) for p in others}
    for p in (gdn.A_log, gdn.dt_bias, gdn.q_conv1d.weight, gdn.o_norm.weight):
        assert id(p) in other_ids and id(p) not in matrix_ids
    for p in (gdn.q_proj.weight, gdn.o_proj.weight, gdn.a_proj.weight):
        assert id(p) in matrix_ids
    opts = build_optimizers(model, TrainConfig("gdn_hybrid", 10**9))
    seen = [id(p) for o in opts for g in o.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())  # each exactly once
    assert all(p.ndim == 2 for g in opts[0].param_groups for p in g["params"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla needs CUDA")
def test_real_fla_matches_stub_layout():
    real = gm.gdn_mixer(MODEL_CONFIGS["gdn_hybrid"])
    shapes = {n: tuple(p.shape) for n, p in real.named_parameters()}
    ref = StubGDN(512, 4, 64, 1.25)
    assert shapes == {n: tuple(p.shape) for n, p in ref.named_parameters()}
