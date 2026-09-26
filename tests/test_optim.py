import copy

import pytest
import torch

from gibc.configs import RUNS, ModelConfig, TrainConfig
from gibc.model import GPT
from gibc.optim import (NorMuon, apply_lr, apply_weight_decay, build_optimizers, decay_start,
                        lr_mult, orthogonalize)

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


def test_adamw_sweep_arm_decays_matrices_at_0_1():
    (opt,) = build_optimizers(GPT(CFG), RUNS["sweep_A_adamw"])
    matrices, others = opt.param_groups
    assert (matrices["weight_decay"], others["weight_decay"]) == (0.1, 0.0)
    assert matrices["lr"] == others["lr"] == 2e-3


# --- NorMuon / cautious weight decay -------------------------------------------------
def test_normuon_grouping_step_and_loss_decreases():
    torch.manual_seed(0)
    model = GPT(CFG)
    optimizers = build_optimizers(model, TrainConfig("A", 10**9, opt="normuon", cautious_wd=True))
    normuon, adam = optimizers
    assert isinstance(normuon, NorMuon) and isinstance(adam, torch.optim.AdamW)
    seen = [id(p) for o in optimizers for g in o.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())  # each exactly once
    assert {id(p) for g in normuon.param_groups for p in g["params"]} == block_matrix_ids(model)
    for o in optimizers:
        for g in o.param_groups:
            assert g["base_lr"] == g["lr"]
    assert normuon.param_groups[0]["cautious_wd"] is True

    idx = torch.randint(0, CFG.vocab, (4, 32))
    losses = []
    for _ in range(8):
        loss = model(idx, idx)
        losses.append(loss.item())
        loss.backward()
        for o in optimizers:
            o.step()
            o.zero_grad(set_to_none=True)
    assert model.blocks[0].attn.wo.weight.abs().sum() > 0  # zero-init weights moved
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("opt", ["muon", "adamw"])
def test_cautious_wd_rejected_outside_normuon(opt):
    with pytest.raises(ValueError, match="cautious_wd"):
        build_optimizers(GPT(CFG), TrainConfig("A", 10**9, opt=opt, cautious_wd=True))


def test_orthogonalize_matches_torch_muon():
    from torch.optim._muon import _zeropower_via_newtonschulz
    torch.manual_seed(0)
    for shape in [(64, 64), (128, 32), (32, 96)]:
        g = torch.randn(shape)
        ours = orthogonalize(g)
        assert torch.equal(ours, _zeropower_via_newtonschulz(g, (3.4445, -4.7750, 2.0315), 5, 1e-7))
        if shape[0] != shape[1]:  # random square matrices have near-zero singular values
            s = torch.linalg.svdvals(ours.float())
            assert s.min() > 0.5 and s.max() < 1.5  # NS5 gives singular values ~ U(0.5, 1.5)


def test_normuon_step_is_muon_step_with_rows_rescaled():
    """Same momentum/NS/lr scale as torch Muon; only per-row magnitudes differ, Frobenius
    norm of the update is preserved, and every row of the update has equal RMS at step 1."""
    torch.manual_seed(0)
    w0, grad = torch.randn(96, 48), torch.randn(96, 48)
    steps = {}
    for name, cls, kw in [("muon", torch.optim.Muon, dict(nesterov=True, adjust_lr_fn="original")),
                          ("normuon", NorMuon, {})]:
        w = torch.nn.Parameter(w0.clone())
        w.grad = grad.clone()
        cls([w], lr=0.02, momentum=0.95, weight_decay=0.0, **kw).step()
        steps[name] = w0 - w.detach()
    muon, nor = steps["muon"], steps["normuon"]
    assert nor.norm() == pytest.approx(muon.norm().item(), rel=1e-3)
    assert torch.nn.functional.cosine_similarity(nor.flatten(), muon.flatten(), dim=0) > 0.95
    row_scale = nor.norm(dim=1, keepdim=True) / muon.norm(dim=1, keepdim=True)
    assert torch.allclose(muon * row_scale, nor, atol=1e-6)  # each row of Muon's step, rescaled
    row_rms = nor.square().mean(dim=1).sqrt()
    assert torch.allclose(row_rms, row_rms.mean().expand_as(row_rms), rtol=1e-3)


def test_normuon_state_dict_round_trip():
    torch.manual_seed(0)
    w0, grads = torch.randn(32, 64), [torch.randn(32, 64) for _ in range(3)]

    def make():
        w = torch.nn.Parameter(w0.clone())
        return w, NorMuon([w], lr=0.02, weight_decay=0.1, cautious_wd=True)

    def step(w, opt, g):
        w.grad = g.clone()
        opt.step()

    w, opt = make()
    step(w, opt, grads[0])
    step(w, opt, grads[1])
    saved_w, saved_opt = w.detach().clone(), copy.deepcopy(opt.state_dict())
    step(w, opt, grads[2])

    w2, opt2 = make()
    with torch.no_grad():
        w2.copy_(saved_w)
    opt2.load_state_dict(saved_opt)
    step(w2, opt2, grads[2])
    assert torch.equal(w, w2)
    assert opt2.state[w2]["second_moment"].shape == (1, 64)  # wide: one stat per column


def test_cautious_mask_hand_built():
    p = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    update = torch.tensor([[0.5, 0.5, -1.0], [-1.0, 0.0, 2.0]])
    # update * p: [[+, -, -], [+, 0, +]] -> decay where >= 0
    expected = torch.tensor([[0.9, -2.0, 3.0], [-3.6, 4.5, 5.4]])
    q = p.clone()
    apply_weight_decay(q, update, 0.1, cautious=True)
    assert torch.allclose(q, expected)
    q = p.clone()
    apply_weight_decay(q, update, 0.1, cautious=False)
    assert torch.allclose(q, 0.9 * p)


def test_normuon_cautious_decays_only_agreeing_coords():
    torch.manual_seed(0)
    w0, grad = torch.randn(16, 16), torch.randn(16, 16)
    lr, wd = 0.02, 0.5
    out = {}
    for name, kw in [("none", dict(weight_decay=0.0)), ("cwd", dict(weight_decay=wd, cautious_wd=True))]:
        w = torch.nn.Parameter(w0.clone())
        w.grad = grad.clone()
        NorMuon([w], lr=lr, **kw).step()
        out[name] = w.detach()
    update = w0 - out["none"]  # sign of the orthogonalised, normalised update
    mask = (update * w0) >= 0
    assert 0 < mask.sum() < mask.numel()
    assert torch.allclose(out["cwd"], out["none"] - w0 * lr * wd * mask, atol=1e-7)
