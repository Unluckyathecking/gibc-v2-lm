"""Plot training curves from downloaded log.jsonl files into results/*.png (1920x1080).

Usage:
  modal volume get gibc-runs sweep_A/log.jsonl runs/sweep_A/log.jsonl
  uv run python scripts/plot_curves.py runs/*/log.jsonl

Writes results/train_loss.png, results/val_bpb.png, results/tok_per_s.png, all runs
overlaid. A run is labelled by its directory name (runs/sweep_A/log.jsonl -> sweep_A).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = "results"
EMA = 0.98  # smoothing for the (noisy) per-step train loss
plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
                     "legend.fontsize": 14, "lines.linewidth": 2.2})


def read_log(path: str) -> list[dict]:
    """Records sorted by step; after a resume the re-run steps replace the earlier ones."""
    by_step = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_step[rec["step"]] = rec
    return [by_step[s] for s in sorted(by_step)]


def label(path: str) -> str:
    base = os.path.basename(path)
    return os.path.basename(os.path.dirname(os.path.abspath(path))) if base == "log.jsonl" \
        else os.path.splitext(base)[0]


def ema(xs: list[float], decay: float = EMA) -> list[float]:
    out, avg = [], xs[0]
    for x in xs:
        avg = decay * avg + (1 - decay) * x
        out.append(avg)
    return out


def new_axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def save(fig, ax, name: str) -> None:
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_all(logs: dict[str, list[dict]]) -> None:
    fig, ax = new_axes("Training loss", "tokens seen (billions)", "train loss (nats/token)")
    for name, recs in logs.items():
        x = [r["tokens_seen"] / 1e9 for r in recs]
        line, = ax.plot(x, ema([r["loss"] for r in recs]), label=name)
        ax.plot(x, [r["loss"] for r in recs], color=line.get_color(), alpha=0.15, linewidth=1)
    save(fig, ax, "train_loss.png")

    fig, ax = new_axes("Validation bits per byte", "tokens seen (billions)", "val bpb")
    for name, recs in logs.items():
        val = [r for r in recs if "val_bpb" in r]
        ax.plot([r["tokens_seen"] / 1e9 for r in val], [r["val_bpb"] for r in val],
                marker="o", markersize=4, label=name)
    save(fig, ax, "val_bpb.png")

    fig, ax = new_axes("Training throughput", "wall-clock training time (hours)",
                       "million tokens / s")
    top = 0.0
    for name, recs in logs.items():
        tps = [r["tok_per_s"] / 1e6 for r in recs]
        ax.plot([r["elapsed_s"] / 3600 for r in recs], tps, label=name, linewidth=1.2)
        top = max(top, sorted(tps)[int(0.99 * (len(tps) - 1))])
    ax.set_ylim(0, 1.15 * top)  # compile/checkpoint outliers would otherwise set the scale
    save(fig, ax, "tok_per_s.png")


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_all({label(p): read_log(p) for p in paths})


if __name__ == "__main__":
    main(sys.argv[1:])
