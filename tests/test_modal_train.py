"""Driver failure handling in modal_train.py (no Modal calls: FunctionCall is faked)."""
import pytest

from modal_train import MAX_RETRIES, _on_failure, _poll


class FakeCall:
    def __init__(self, err: Exception | None = None):
        self.err, self.cancelled = err, False

    def get(self, timeout=None):
        raise self.err

    def cancel(self):
        self.cancelled = True


def _state(attempts: int = 1) -> dict:
    return {"state": "running", "attempts": attempts, "errors": []}


def test_poll_returns_the_exception():
    err = OSError("boom")
    assert _poll(FakeCall(err)) == ("failed", err)
    assert _poll(FakeCall(TimeoutError())) == ("running", None)


@pytest.mark.parametrize("err", [FloatingPointError("non-finite loss nan at step 7"),
                                 RuntimeError("source 'dclm' exhausted its 3 train shards"),
                                 RuntimeError("data preflight: training data would be exhausted"),
                                 KeyError("fwedu_hq"), ValueError("bad mix")])
def test_deterministic_failures_are_not_retried(err):
    s, call = _state(), FakeCall()
    assert _on_failure(s, call, err) is False
    assert s["state"] == "failed" and s["errors"] == [repr(err)] and call.cancelled


@pytest.mark.parametrize("err", [RuntimeError("CUDA error: an illegal memory access"),
                                 OSError("connection reset"), Exception("preempted")])
def test_other_failures_cancel_then_retry(err):
    s, call = _state(), FakeCall()
    assert _on_failure(s, call, err) is True
    assert s["state"] == "running" and call.cancelled


def test_retries_are_capped():
    s = _state(attempts=MAX_RETRIES + 1)
    assert _on_failure(s, FakeCall(), OSError("again")) is False and s["state"] == "failed"
