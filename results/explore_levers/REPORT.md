# Cap levers: factorised embedding and ternary QAT at 1B tokens

Branch `explore/cap-levers`. Question: is there a better way to spend the 50M trainable-parameter
cap than A_vres's 16.8M-parameter tied embedding plus 14 dense layers? Three 1B-token arms, each
with the exact `sweep_A_vres` recipe (value residual, Muon 0.02 + AdamW 3e-3, WSD 1% warmup / 35%
linear decay, MIX_MAIN, no anneal, seed 1337, 524,288 tokens per step, 1,907 steps, one H100).

## Arms

| arm | change vs A_vres | layers | mlp | trainable params |
|---|---|---|---|---|
| sweep_A_vres (reference) | none | 14 | 1536 | 49,822,228 |
| sweep_fact128 | tied E 32768x128 + P 512x128 | 19 | 1568 | 49,729,051 |
| sweep_fact256 | tied E 32768x256 + P 512x256 | 17 | 1600 | 49,759,768 |
| sweep_ternary | BitNet b1.58 QAT on Wq,Wk,Wv,Wo,W_up,W_down | 14 | 1536 | 49,822,228 |

Factorised embedding (`ModelConfig.embed_dim`): input `x = P e`, output
`logits = (norm_f(h) @ P) @ E^T`. E and P are each a single tensor used on both sides. P is a 2-D
matrix and goes to Muon (`optim.split_params` adds `embed_proj.weight` to the matrix list); E goes
to AdamW like the full embedding does. P init N(0, 0.02), E init N(0, 0.005) as before.

Layer counts: with the block shape unchanged, 19 layers (fact128) and 17 layers (fact256) are the
most that fit, but they land at 49.11M and 48.65M, under the 49.3M floor. One more layer breaks the
cap. The leftover went into MLP width (1568 and 1600, multiples of 32) to bring both arms within
0.2% of A_vres. That makes MLP width a second, small difference from the reference (+2% / +4%).

Ternary (`ModelConfig.ternary`): latent fp32 weights; forward uses `round(W / mean|W|)` clipped to
{-1, 0, 1} times `mean|W|` (per-tensor), and 8-bit absmax fake quantisation of each block matrix's
input per token (`ternary_act8=True`, kept on: no instability). Straight-through gradients for both.
Embedding, norms and scalars stay full precision. No extra SubLN was added. Muon on the latent
matrices was stable (no NaN, no loss spikes beyond the baseline's), so the arm stayed on Muon.

## Results (final weights, full eval sets, zero-shot, accuracies in percent)

| | A_vres | fact128 | fact256 | ternary |
|---|---|---|---|---|
| val bpb | **1.0931** | 1.1264 | 1.0994 | 1.1602 |
| WikiText-103 word ppl, stride 512 | **53.95** | 63.00 | 55.90 | 68.72 |
| WikiText-103 word ppl, lm-eval style | **60.57** | 71.05 | 63.05 | 77.23 |
| HellaSwag acc_norm | **28.71** | 28.25 | 28.30 | 27.71 |
| ARC-Easy acc | **47.18** | 44.15 | 46.17 | 46.25 |
| PIQA acc | **60.83** | 58.81 | 58.92 | 59.41 |
| WinoGrande acc | 50.04 | **52.80** | 48.93 | 51.14 |
| train tok/s (median) | 800k | 731k | 789k | 814k |
| H100 hours (incl. val) | 0.447 | 0.505 | 0.468 | 0.545 |

Val bpb was worse than A_vres at every one of the 8 val points for all three arms
(see `logs/*/log.jsonl`). WinoGrande is at chance for all arms (stderr about 1.4 points).

## Verdict

None of the three beats A_vres, and none earns a 20B run.

- fact128: clearly worse (+0.033 val bpb, +17% WikiText ppl). Cutting the embedding to 128 dims
  costs more than five extra layers recover at this scale.
- fact256: closest, but behind on every metric (+0.006 val bpb, +3.6% WikiText ppl, -1.0 ARC-E,
  -1.9 PIQA). The gap is small and single-seed, but it points the wrong way, and the trend from 128
  to 256 suggests the full 512-d tied embedding is the right choice at 32k vocab and 1B tokens.
- ternary: worst on LM quality (+0.067 val bpb, +27% WikiText ppl). Under a cap that counts
  parameters, ternary weights only remove precision; they would pay off only under a cap counted in
  bits or bytes, where the same budget could hold about 10x more ternary parameters.

A 20B run of fact256 would cost about 8.5 to 9 H100 hours, roughly $35 on Modal, with no evidence it
would beat the submitted model. Not recommended.

## Spend

Training 1.52 H100 hours (about $6), three 2-minute smokes (about $0.7), three full evals on L4
(about $1). Total about $8, under the $20 cap.

## Files

`*_status.json` (final run status and param counts), `logs/<run>/log.jsonl` (per-step log),
`*_final_eval.json` (lm-eval + WikiText), `smoke_*.json` (random-token throughput),
`eval_table.txt` (the three arms plus sweep_A_vres).
