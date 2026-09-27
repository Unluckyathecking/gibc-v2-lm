# Gated DeltaNet hybrid, 1B-token ablation

Branch `explore/gdn-hybrid` only. Nothing here touches `main` or the submitted runs.

## Setup

`MODEL_CONFIGS["gdn_hybrid"]` is Config A_vres with 11 of its 14 attention mixers swapped for
fla's `GatedDeltaNet` (flash-linear-attention 0.5.2), in the Qwen3-Next 3:1 pattern. Softmax
attention (GQA 8/4, RoPE, QK-norm) stays at layers 3, 7 and 11. Everything else is unchanged:
tied 32k embedding, d_model 512, 3x ReLU² MLP in every layer, pre-norm, U-net skips. The run
`sweep_gdn_hybrid` uses the same recipe as `sweep_A_vres` (Muon 0.02 / AdamW 3e-3, 1B tokens,
1% warmup, 35% linear decay, main mix, seed 1337).

GDN mixer: 4 heads, q/k head dim 64, v head dim 80 (`expand_v=1.25`), short convs (size 4),
output gate. It has its own gated RMSNorm on the output, applies no RoPE, and its forward
returns `(o, None, cache)`. Init follows our convention: input projections N(0, 0.02),
`o_proj` zero, so at init the stack is still a no-op. fla's own init is kept for A_log and dt_bias.

- **Parameters: 49,544,401** (66,321,617 counting the tied embedding twice). A_vres has 49,822,228.
  A GDN mixer has 761,176 params and an attention mixer has 786,432. `expand_v=1.5` would
  push the total past the 50M cap, so this is the closest fit without odd head sizes.
- **Value residual** only makes sense where there is a V to mix. The first attention layer
  (3) provides v0, and layers 7 and 11 each learn one lambda (2 scalars instead of 13).
- **Optimizer groups:** every 2D block matrix goes to Muon: q/k/v/g/o and the small a/b gate
  projections. A_log, dt_bias, the conv weights (3D) and norm weights go to AdamW.
  `split_params` now asserts that each param lands in exactly one group. `tests/test_gdn.py`
  checks this with a stub that has fla's exact parameter layout, because fla needs CUDA. All
  107 existing tests still pass.
- **Image:** `flash-linear-attention==0.5.2` plus `triton==3.7.1`. torch 2.11 pins triton
  3.6.0, and fla refuses 3.6.0 on Hopper because its gated `chunk_bwd_dqkwg` gives wrong
  gradients there (fla #640). Triton 3.7.1 is the fixed version.

## Throughput (H100, micro-batch 64 x 1024, grad accum 8)

| | tok/s | peak mem | compile |
|---|---|---|---|
| Config A (smoke) | 863,913 | 20.9 GB | 35 s |
| gdn_hybrid, torch.compile tracing fla | **384,652** | 52.6 GB | 135 s |
| gdn_hybrid, GDN under `torch.compiler.disable` | 351,286 | 54.8 GB | 141 s |
| sweep run, median step (gdn vs A_vres) | 385,995 vs 800,389 | | |

Compile works without the `torch.compiler.disable` wrapper, and the unwrapped version is
faster, so the wrapper was dropped. At ctx 1024 the GDN hybrid trains at 0.48x the speed of
the attention baseline and uses 2.5x the memory.

## Results (1B tokens, final weights, full eval)

| | val bpb | WikiText word ppl (s512) | WT bpb | HellaSwag | ARC-E | PIQA | Winogrande |
|---|---|---|---|---|---|---|---|
| sweep_A_vres | **1.0931** | **53.95** | **1.0760** | **28.71** | 47.18 | **60.83** | **50.04** |
| sweep_gdn_hybrid | 1.0981 | 55.10 | 1.0816 | 28.43 | **48.23** | 60.23 | 49.72 |

Val bpb along the run (GDN vs A_vres): 250 1.4167/1.381, 500 1.2687/1.251,
1000 1.1799/1.1686, 1500 1.1309/1.1222, 1907 1.0981/1.0931. The gap narrows from 0.036 to
0.005 but never closes. All benchmark differences are inside about 1 stderr: HellaSwag ±0.45,
ARC-E ±1.0, PIQA ±1.1 and Winogrande ±1.4 points.

## Verdict

The hybrid did not help. At 1B tokens it is slightly worse on val bpb (+0.005) and WikiText
(+1.1 ppl), and within noise on the benchmarks. It also costs twice the GPU time per token.
That result is expected at our 1024-token context. GDN's advantage is linear-time long
context and a small inference cache, and at ctx 1024 neither matters. Meanwhile fla's
chunked kernels are slower than FlashAttention-backed SDPA at this size. The narrowing gap
suggests the hybrid might roughly match the baseline at 20B tokens, but nothing here says it
would beat it, and the LRs were tuned for attention. Not worth pursuing for this entry.

## Cost

- Spent: about $5 in total. That covers the training run (0.85 H100-h, about $3.40), two
  3-minute smokes including compile (about $1) and the L4 eval (about $0.10).
- A full 20B run at 386k tok/s would take about 14.4 H100-hours, roughly $57. The A_vres main
  run takes about 7 hours at 800k tok/s.
