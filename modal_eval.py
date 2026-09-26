"""Modal app "gibc-eval": lm-eval benchmarks + WikiText-103 perplexity for trained runs.

  modal run modal_eval.py::main --run sweep_A               # full eval, final + avg
  modal run modal_eval.py::main --run sweep_A --variants final --limit 1000
  modal run modal_eval.py::eval_sweep                       # all SWEEP_RUNS, parallel, fast, final
  modal run modal_eval.py::eval_sweep --limit 0             # all SWEEP_RUNS, full

Results land in {run_dir}/evals/{variant}[_limitN].json on the gibc-runs volume.
"""
import modal

from gibc.configs import SWEEP_RUNS
from gibc.modal_common import VOLUMES, hf_secret, image, runs_vol

app = modal.App("gibc-eval")

# A 50M model scoring short multiple-choice requests is nowhere near compute-bound on an
# H100; an L4 (24 GB, bf16) does the job for roughly a fifth of the price.
EVAL_GPU = "L4"
SWEEP_LIMIT = 1000  # fast mode: first 1000 examples per task (hellaswag has 10,042)


@app.function(image=image, gpu=EVAL_GPU, volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 3600)
def eval_run(run: str, variants: str = "final,avg", limit: int | None = None) -> dict:
    """Evaluate one run's variants, write evals/*.json and commit the runs volume."""
    from gibc.evaluate import evaluate_run
    runs_vol.reload()  # see weights committed after this container started
    out = evaluate_run(run, variants=tuple(variants.split(",")), limit=limit)
    runs_vol.commit()
    return out


def format_table(recs: list[dict]) -> str:
    """One line per (run, variant): benchmark accuracies, then WikiText word perplexity and
    bits per byte, sliding stride 512 (wt_*) and lm-eval-style disjoint windows (lmwt_*)."""
    cols = [("hellaswag/acc_norm", "hs_n"), ("arc_easy/acc", "arcE"),
            ("arc_easy/acc_norm", "arcE_n"), ("piqa/acc", "piqa"), ("piqa/acc_norm", "piqa_n"),
            ("winogrande/acc", "wino")]
    head = f"{'run':<18}{'variant':<8}" + "".join(f"{c:>8}" for _, c in cols)
    lines = [head + f"{'wt_wppl':>10}{'wt_bpb':>8}{'lmwt_wppl':>10}{'lmwt_bpb':>9}{'limit':>7}"]
    for r in recs:
        lm, wt, lw = r["lm_eval"], r["wikitext103_stride512"], r["wikitext103_lmeval_style"]
        row = f"{r['run']:<18}{r['variant']:<8}" + "".join(f"{lm.get(k, float('nan')):>8.4f}"
                                                            for k, _ in cols)
        lines.append(row + f"{wt['word_ppl']:>10.5g}{wt['bits_per_byte']:>8.4f}"
                           f"{lw['word_ppl']:>10.5g}{lw['bits_per_byte']:>9.4f}"
                           f"{str(r['limit']):>7}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(run: str, variants: str = "final,avg", limit: int = 0):
    out = eval_run.remote(run, variants, limit or None)
    print(format_table(list(out.values())))


@app.local_entrypoint()
def eval_sweep(variants: str = "final", limit: int = SWEEP_LIMIT):
    """Fan out over SWEEP_RUNS in parallel (one container each); final weights by default."""
    args = [(run, variants, limit or None) for run in SWEEP_RUNS]
    recs = []
    for run, out in zip(SWEEP_RUNS, eval_run.starmap(args, return_exceptions=True)):
        if isinstance(out, Exception):
            print(f"{run}: eval failed: {out!r}")
        else:
            recs.extend(out.values())
    print(format_table(recs))
