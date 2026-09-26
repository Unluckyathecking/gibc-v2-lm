"""Checkpoint averaging and the WikiText sliding-window scorer (CPU, no network)."""
import math
from dataclasses import asdict

import pytest
import torch

from gibc.configs import BOS_ID
from gibc.evaluate import (average_checkpoints, score_ids, variant_weights, wikitext103_ppl,
                           window_spans)
from gibc.model import GPT
from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
from tests.test_hf_wrap import TINY, make_trained


def save(tmp_path, name: str, state: dict) -> str:
    path = tmp_path / name
    torch.save({"model": state, "model_cfg": asdict(TINY)}, path)
    return str(path)


def test_average_of_identical_is_identity(tmp_path):
    state = GPT(TINY).state_dict()
    avg = average_checkpoints([save(tmp_path, "a.pt", state), save(tmp_path, "b.pt", state)])
    for k, v in state.items():
        torch.testing.assert_close(avg[k], v.float(), atol=0, rtol=0)


def test_average_of_two_is_mean(tmp_path):
    torch.manual_seed(0)
    a = {k: torch.randn_like(v) for k, v in GPT(TINY).state_dict().items()}
    b = {k: torch.randn_like(v) for k, v in a.items()}
    avg = average_checkpoints([save(tmp_path, "a.pt", a), save(tmp_path, "b.pt", b)])
    for k in a:
        torch.testing.assert_close(avg[k], (a[k] + b[k]) / 2)


def test_variant_weights_picks_last_snapshots(tmp_path):
    (tmp_path / "snap").mkdir()
    for step in (250, 500, 750, 1000, 1250):
        (tmp_path / "snap" / f"step_{step:06d}.pt").touch()
    names = [p.rsplit("/", 1)[-1] for p in variant_weights(str(tmp_path), "avg", n_avg=3)]
    assert names == ["step_000750.pt", "step_001000.pt", "step_001250.pt"]
    assert variant_weights(str(tmp_path), "final") == []  # no final.pt yet


@pytest.mark.parametrize("n,ctx,stride", [(2, 8, 4), (8, 8, 4), (9, 8, 4), (50, 8, 4),
                                          (50, 8, 7), (51, 16, 5), (1, 8, 4)])
def test_windows_score_each_target_once(n, ctx, stride):
    targets = []
    for i, (begin, end, first) in enumerate(window_spans(n, ctx, stride)):
        assert end - begin <= ctx and begin < first <= end
        if i > 0:
            assert first - begin >= ctx - stride  # every new target has >= ctx-stride context
        targets += range(first, end)
    assert targets == list(range(1, n))


def test_windows_reject_stride_equal_to_ctx():
    with pytest.raises(AssertionError):
        window_spans(50, 8, 8)


def direct_nats(model, ids: list[int]) -> float:
    x = torch.tensor(ids)
    with torch.no_grad():
        logits = model(x[None]).logits[0]
    return torch.nn.functional.cross_entropy(logits[:-1], x[1:], reduction="sum").item()


def test_score_matches_direct_when_one_window(tmp_path):
    _, hf, tok = make_trained(tmp_path)
    ids = [BOS_ID] + tok.encode("the cat sat on the mat. hi there", add_special_tokens=False)
    assert len(ids) <= TINY.ctx
    assert score_ids(hf, ids, TINY.ctx, TINY.ctx // 2) == pytest.approx(direct_nats(hf, ids),
                                                                        rel=1e-5)


def test_sliding_window_equals_sum_of_window_scores(tmp_path):
    _, hf, tok = make_trained(tmp_path)
    ids = [BOS_ID] + tok.encode(" ".join(["the cat sat on the mat."] * 12),
                                add_special_tokens=False)
    ctx, stride = 16, 8
    assert len(ids) > 3 * ctx
    expected = 0.0
    for begin, end, first in window_spans(len(ids), ctx, stride):
        x = torch.tensor(ids[begin:end])
        with torch.no_grad():
            nll = torch.nn.functional.cross_entropy(hf(x[None]).logits[0][:-1], x[1:],
                                                    reduction="none")
        expected += nll[first - begin - 1:].sum().item()
    assert score_ids(hf, ids, ctx, stride) == pytest.approx(expected, rel=1e-5)


def test_wikitext_counts(tmp_path):
    _, hf, tok = make_trained(tmp_path)
    pages = [" = Cat = \n \n the cat sat on the mat . \n", " = Fox = \n \n a quick brown fox . \n"]
    res = wikitext103_ppl(hf, tok, ctx=TINY.ctx, stride=TINY.ctx // 2, pages=pages)
    assert res["n_docs"] == 2
    # re.split(r"\s+") on the raw page, as lm-eval counts: leading/trailing space adds ''s
    assert res["n_words"] == 12 + 10
    detok = [wikitext_detokenizer({"page": p}) for p in pages]
    assert res["n_tokens"] == sum(len(tok.encode(t, add_special_tokens=False)) for t in detok)
    assert res["n_bytes"] == sum(len(p.encode()) for p in pages)
    assert res["word_ppl"] == pytest.approx(math.exp(res["total_nats"] / res["n_words"]))
    assert res["bits_per_byte"] == pytest.approx(
        res["total_nats"] / (math.log(2) * res["n_bytes"]))
