"""Results table in modal_eval.py (no Modal calls)."""
import inspect

from modal_eval import eval_sweep, format_table


def test_table_shows_both_wikitext_windowings():
    wt = lambda ppl, bpb: {"word_ppl": ppl, "bits_per_byte": bpb}  # noqa: E731
    rec = {"run": "sweep_A", "variant": "final", "limit": None, "lm_eval": {},
           "wikitext103_stride512": wt(31.5, 0.91), "wikitext103_lmeval_style": wt(40.25, 0.97)}
    head, row = format_table([rec]).splitlines()
    assert "wt_wppl" in head and "lmwt_wppl" in head
    assert "31.5" in row and "0.9100" in row and "40.25" in row and "0.9700" in row


def test_eval_sweep_defaults_to_final_only():
    fn = eval_sweep.info.raw_f  # the function behind @app.local_entrypoint()
    assert inspect.signature(fn).parameters["variants"].default == "final"
