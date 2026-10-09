# Training-compute efficiency: what we can and cannot claim

Snapshot: Axiomic Labs Open SLM Leaderboard, saved `index.html` (213 rows with a computable Int Index, 125 of them under 50M parameters, 42 organisations). Int Index recomputed with the Space's own formula; it reproduces gibc-v2-lm 16.13 and 50M-pro 17.93. Compute figures were read from each model's Hugging Face card (README plus repo file list) on 2026-10-09. Everything below is self-reported by the model authors and unverified, including our own.

Our run: 49.8M params, 20B tokens, 8.58 H100-hours on one H100 80GB (final run). A separate 12-arm, 1B-token sweep cost 5.23 H100-hours, so the project total is 13.81 H100-hours. Cost is about $18 per 10B tokens on Modal (our own billing figure).

## 1. Short answer

- Disclosure is thin. Of the 115 other sub-50M base runs, 39 name their hardware and 30 give enough (tokens, hardware, time) to derive GPU-hours. 91 give a token count.
- 8.58 H100-h is **above** the median disclosed compute in the bracket, not below it. Do not describe it as small.
- Strongest defensible claim: **among the sub-50M models that report their training compute (n=30), gibc-v2-lm has the highest Int Index (16.13; next is 13.40), and the two nearest reporters used equal or more compute.** It must always carry the "among models that report" qualifier.
- Biggest caveat: the model ranked above us, 50M-pro (17.93), and two more in the top six (ZeroS-Linear-50M 15.62, Linear-50M-exp 14.66) disclose no tokens, hardware or time at all, and Blaze (15.45), BananaMind-2-Medium (15.37) and Veyra2-Apricot (13.39) give tokens but no hardware. We cannot say anything about our compute relative to 50M-pro.

## 2. How disclosure was classified

Unit: unique base runs. Derived rows (Chat, Instruct) were folded into their base. 125 rows became 116 unique runs, of which ours is one, leaving 115 peers. Three cards could not be read (NACR-1.0 and Ant-10M return 401, Mystery-5M-Preview has no card link) and are counted as "none".

| Level | Meaning | Peers (of 115) |
|---|---|---|
| L3 | tokens plus hardware plus time, GPU-hours derivable | 30 (26%) |
| L2 | tokens plus hardware, no time | 8 (7%) |
| L1 | tokens only | 53 (46%) |
| partial | time or hardware or partial tokens, no total token count | 5 (4%) |
| none | nothing found | 19 (17%) |

So 96 of 115 (83%) disclose something, 91 (79%) disclose a token count, 39 (34%) name hardware, 30 (26%) allow GPU-hours to be derived. 37 of 42 organisations disclose something on at least one model. Direct GPU-hour figures exist only for Doge-20M (14 RTX 4090 hours, table in its card) and, via the paper, Ettin (4xH100 node, about 6 days for the 17M pretraining phase). Pythia-14M and Pythia-31M state 300B tokens and A100-40GB in the paper but give GPU-hours only for 70M and up. Cost appears for two models only: SupraGDN-5M (about $2 on Runpod) and Stentor3-20M ($0, free Kaggle TPU).

Method note: cards were scanned by keyword and skimmed, then the top non-disclosers were re-checked through the repo file list. "None" means nothing found, not proof of absence; a linked paper or Discord post could exist.

## 3. Top 10 sub-50M by Int Index

| Rank | Model | Org | Int Index | ARC-E | Tokens | Hardware | Time | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | 50M-pro | fromzero | 17.93 | 45.62 | not disclosed | not disclosed | not disclosed | [card](https://huggingface.co/fromziro/50M-pro) (README lists datasets only; repo has diagnostics.json, no run info) |
| 2 | **gibc-v2-lm** | unluckyexe | 16.13 | 47.85 | 20B | 1x H100 80GB | 8.58 H100-h (plus 5.23 sweep) | [card](https://huggingface.co/Unluckyexe/gibc-v2-lm) |
| 3 | ZeroS-Linear-50M | fromzero | 15.62 | 43.86 | not disclosed | not disclosed | not disclosed | [card](https://huggingface.co/fromziro/ZeroS-Linear-50M) (card is one validation-loss line) |
| 4 | Blaze | surjolabs | 15.45 | 41.84 | ~20.97B | not disclosed | not disclosed | [card](https://huggingface.co/SurjoLabs/Blaze) |
| 5 | BananaMind-2-Medium | bananamind | 15.37 | 43.81 | 50.0B | not disclosed | not disclosed | [card](https://huggingface.co/BananaMind/BananaMind-2-Medium) |
| 6 | Linear-50M-exp | user01110 | 14.66 | 44.02 | not disclosed | not disclosed | not disclosed | [card](https://huggingface.co/User01110/Linear-50M-exp) |
| 7 | BananaMind-2-Medium-Chat | bananamind | 14.07 | 43.43 | SFT of rank 5; 0.42B SFT tokens | not disclosed | not disclosed | [card](https://huggingface.co/BananaMind/BananaMind-2-Medium-Chat) |
| 8 | BananaMind-2.1-Unified | bananamind | 13.40 | 38.51 | 38.0B | 8x RTX PRO 6000 Blackwell | 27,405 s wall (7h37m) = 60.9 GPU-h | [card](https://huggingface.co/BananaMind/BananaMind-2.1-Unified) |
| 9 | Veyra2-Apricot-50M-Base | veyraai | 13.39 | 42.47 | 19.99B | not disclosed | not disclosed | [card](https://huggingface.co/veyra-ai/Veyra2-Apricot-50M-Base), [training_state.json](https://huggingface.co/veyra-ai/Veyra2-Apricot-50M-Base/raw/main/training_state.json) |
| 10-11 | Supra2-Medium-Instruct / -Base | supralabs | 12.16 / 12.09 | 40.28 / 41.67 | 20B (base) | RTX 5060 Ti 16GB + RTX 5060 8GB, DDP | not disclosed | [card](https://huggingface.co/SupraLabs/Supra2-Medium-Base) |

Strict count for ranks 1 to 11 excluding us (8 distinct base runs, the Chat row being a derivative): 5 give tokens, 1 gives hardware with time (BananaMind-2.1-Unified), 1 more gives hardware without time. Three give nothing. No cost is reported by any of them. The next entry that reports full compute is Er-Large-30M (rank 13, II 11.71).

Hugging Face repo ids differ in case from the org names shown on the leaderboard; links are the card URLs stored in the Space.

## 4. Models that report enough to derive GPU-hours (L3, n=31 including us)

The "H100-eq h" column converts to H100-hours with the factors in section 6 and is an estimate for every row except ours. "6ND FLOPs" is 6 x params x tokens, the convention Axiomic's own GPT-X2.5-135M card uses ("Training compute is estimated as 6 x parameters x training tokens"). It needs no hardware assumption.

| # | Model | Org | Params | Int Index | Tokens (B) | Hardware (as reported) | Reported time | H100-eq h (est.) | 6ND FLOPs | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gibc-v2-lm | unluckyexe | 49.8M | 16.13 | 20 | 1x H100 | 8.58 h (own card; 8.58 H100-h) | 8.58 | 6.0e+18 | [card](https://huggingface.co/Unluckyexe/gibc-v2-lm) |
| 2 | BananaMind-2.1-Unified | bananamind | 35.0M | 13.40 | 38 | 8x PRO6000 | 7.61 h (27,405 s wall) | 30.45 | 8.0e+18 | [card](https://huggingface.co/BananaMind/BananaMind-2.1-Unified) |
| 3 | Er-Large-30M | fromzero | 31.0M | 11.71 | 34.8 | 2x 5070 | 116.00 h (assumes 116h wall on 2 GPUs) | 23.20 | 6.5e+18 | [card](https://huggingface.co/fromziro/Er-Large-30M) |
| 4 | pulvis-v2 | benchlabs | 3.0M | 8.62 | 20 | 2x 5090 | 1.72 h (77+26 min) | 0.72 | 3.6e+17 | [card](https://huggingface.co/bench-labs/pulvis-v2) |
| 5 | Er-Medium-12.5M | fromzero | 12.5M | 8.24 | 34.8 | 1x 5060 | 120.00 h (5 days) | 7.20 | 2.6e+18 | [card](https://huggingface.co/fromziro/Er-Medium-12.5M) |
| 6 | DynamicMind-MoE | dedeprogames | 30.2M | 8.17 | 10 | 1x 3060 | 39.50 h | 1.19 | 1.8e+18 | [card](https://huggingface.co/DedeProGames/DynamicMind-MoE) |
| 7 | pulvis-v1 | benchlabs | 3.0M | 7.85 | 20 | 8x 5090 | 0.55 h (25+8 min) | 0.92 | 3.6e+17 | [card](https://huggingface.co/bench-labs/pulvis-v1) |
| 8 | BananaMind-2-Micro | bananamind | 2.9M | 6.01 | 75 | 8x PRO6000 | 3.33 h (11,973 s) | 13.30 | 1.3e+18 | [card](https://huggingface.co/BananaMind/BananaMind-2-Micro) |
| 9 | Stentor3-20M | stentor | 20.0M | 5.81 | 8.5 | 8x TPUv5e | 8.20 h (8 chips, ~8.1-8.7 h) | 13.12 | 1.0e+18 | [card](https://huggingface.co/StentorLabs/Stentor3-20M) |
| 10 | CyberSLM-33M-Base | sabari2005 | 33.5M | 5.66 | 0.786 | 1x A100 | 1.18 h | 0.38 | 1.6e+17 | [card](https://huggingface.co/sabari2005/cyberslm-base) |
| 11 | MiniBananaMind-v2-9M | bananamind | 8.9M | 5.42 | 3.54 | 1x 5070Ti | 4.57 h | 0.55 | 1.9e+17 | [card](https://huggingface.co/BananaMind/MiniBananaMind-v2-9M) |
| 12 | BananaMind-2.1-Pico-Preview | bananamind | 1.5M | 5.32 | 2 | 1x 5070Ti | 1.20 h | 0.14 | 1.8e+16 | [card](https://huggingface.co/BananaMind/BananaMind-2.1-Pico-Preview) |
| 13 | Supra-Mini-v5-8M | supralabs | 7.9M | 5.05 | 10 | 1x 5060Ti | 11.00 h (5B x 2 epochs) | 0.77 | 4.7e+17 | [card](https://huggingface.co/SupraLabs/Supra-Mini-v5-8M) |
| 14 | ettin-decoder-17m | jhuclsp | 16.9M | 5.03 | 2000 | 4x H100 | 144.00 h (4xH100 node, ~6 days pretrain phase (paper App. E)) | 576.00 | 2.0e+20 | [card](https://huggingface.co/jhu-clsp/ettin-decoder-17m) |
| 15 | Doge-20M | smalldoge | 13.1M | 4.81 | 4 | 1x 4090 | 14.00 h (14 RTX 4090 GPU-h, table in card) | 2.38 | 3.1e+17 | [card](https://huggingface.co/SmallDoge/Doge-20M) |
| 16 | Tenete-8M | harleyml | 8.0M | 4.67 | 0.577 | 1x 2060 | 2.33 h | 0.07 | 2.8e+16 | [card](https://huggingface.co/Harley-ml/Tenete-8M) |
| 17 | Dillionv2-1.3M | harleyml | 1.3M | 4.44 | 24 | 1x 2060 + 2x T4 | 35.00 h (35 h combined, GPU split unclear) | 5.95 | 1.9e+17 | [card](https://huggingface.co/Harley-ml/Dillionv2-1.3M) |
| 18 | Spark-5M-Base-v4 | lhtechai | 5.0M | 4.43 | 0.7 | 2x T4 | 4.00 h | 0.56 | 2.1e+16 | [card](https://huggingface.co/LH-Tech-AI/Spark-5M-Base-v4) |
| 19 | LowOnMind-5M | dedeprogames | 4.9M | 4.24 | 0.2 | 1x T4 | 0.45 h | 0.03 | 5.9e+15 | [card](https://huggingface.co/DedeProGames/LowOnMind-5M) |
| 20 | SupraGDN-5M | supralabs | 5.0M | 3.88 | 5 | 1x RTX4500SE | 2.50 h (~$2 on Runpod) | 0.50 | 1.5e+17 | [card](https://huggingface.co/SupraLabs/SupraGDN-5M) |
| 21 | Syn-2.6M | fromzero | 2.6M | 3.78 | 2.7 | 1x 2060 | 1.80 h | 0.05 | 4.2e+16 | [card](https://huggingface.co/fromziro/Syn-2.6M) |
| 22 | Er-Tiny-1.3M | fromzero | 1.3M | 3.77 | 34.8 | 2x 5070 | 38.00 h | 7.60 | 2.7e+17 | [card](https://huggingface.co/fromziro/Er-Tiny-1.3M) |
| 23 | Dillion-1.2M | harleyml | 1.3M | 3.71 | 9 | 1x 2060 | 60.00 h (~2.5 days) | 1.80 | 7.0e+16 | [card](https://huggingface.co/Harley-ml/Dillion-1.2M) |
| 24 | CinnabarLM-1.4M-Base | mihaipopa | 1.5M | 3.48 | 0.03 | 1x T4 | 0.45 h | 0.03 | 2.7e+14 | [card](https://huggingface.co/MihaiPopa-1/CinnabarLM-1.4M-Base) |
| 25 | CinnabarLM-4M-Base | mihaipopa | 4.2M | 3.47 | 0.08 | 1x T4 | 0.60 h | 0.04 | 2.0e+15 | [card](https://huggingface.co/MihaiPopa-1/CinnabarLM-4M-Base) |
| 26 | Supra-Mini-v6-1M | supralabs | 1.4M | 3.39 | 5 | 1x 5060Ti | 3.00 h | 0.21 | 4.2e+16 | [card](https://huggingface.co/SupraLabs/Supra-Mini-v6-1M) |
| 27 | Glint-1.3 | compactai | 1.0M | 3.35 | 100 | 1x 5090 | 30.00 h (card inconsistent: 100B tok at 138k tok/s is ~200 h, not 30 h) | 6.30 | 6.0e+17 | [card](https://huggingface.co/Glint-Research/Glint-1.3) |
| 28 | LowOnMind-1M | dedeprogames | 1.0M | 2.86 | 0.2 | 1x T4 | 0.18 h | 0.01 | 1.2e+15 | [card](https://huggingface.co/DedeProGames/LowOnMind-1M) |
| 29 | CinnabarLM-1.5M-Base | mihaipopa | 1.7M | 2.71 | 0.05 | 1x T4 | 0.47 h | 0.03 | 5.1e+14 | [card](https://huggingface.co/MihaiPopa-1/CinnabarLM-1.5M-Base) |
| 30 | Supra-Mini-v4-2M | supralabs | 2.6M | 2.47 | 6 | 1x 5060Ti | 3.00 h (3B x 2 epochs) | 0.21 | 9.4e+16 | [card](https://huggingface.co/SupraLabs/Supra-Mini-v4-2M) |
| 31 | LowOnMind-300k | dedeprogames | 0.3M | 1.58 | 0.2 | 1x T4 | 0.72 h | 0.05 | 3.6e+14 | [card](https://huggingface.co/DedeProGames/LowOnMind-300k) |

Extra sources for two rows: Ettin training hardware and time are from the paper (arXiv 2507.11412, appendix E: 4xH100 node, about 6 days for the 17M pretraining phase, roughly 2T tokens in total; the 576 H100-h covers pretraining only); the card gives no hardware. Pythia is not in this table because GPU-hours for 14M and 31M are not published (arXiv 2304.01373, appendix D, table 5, starts at 70M).

Row notes: Er-Large, Er-Medium and Er-Tiny: "Total training time" in the card is taken as wall-clock on the listed GPUs. Dillionv2: "35 hours combined" across an RTX 2060 and two T4s is taken as wall-clock with all three. Glint-1.3: the card says 100B tokens, 30 hours and 138k tokens/s on a 5090; those three figures disagree by about 7x (100B tokens at 138k tok/s is about 200 hours), so treat the row as unreliable.

## 5. Bracket statistics among disclosers

Peers only (we are excluded from the averages), unique runs.

| Quantity | n | Mean | Median | gibc-v2-lm |
|---|---|---|---|---|
| Training tokens | 91 | 66B (2T Ettin and 300B Pythia inflate this) | 13.5B | 20B (about 1.5x median; 52 peers used fewer tokens, 32 more, 7 at exactly 20B) |
| 6ND FLOPs | 91 | 8.7e18 | 5.2e17 | 6.0e18 (about 11x median; 9 of 91 peers used more) |
| H100-eq hours (L3 only) | 30 | 23.1 (4.1 without Ettin's 576) | 0.64 | 8.58 (about 13x median; 5 of 30 used more) |
| Tokens, L3 subset | 30 | 82B | 5.5B | 20B |

Reading: against the disclosing sub-50M bracket we are in the upper tail on compute, at roughly the 83rd percentile of the L3 group and the 90th percentile by FLOPs, though still well under the mean once the two huge outliers (Ettin, Pythia) are included. Disclosers are dominated by hobbyist runs of a few minutes to a few hours on consumer GPUs, mostly of models scoring 1.6 to 6 Int Index. We spent more than most and scored more than all of them.

Sensitivity of "5 of 30 used more H100-eq hours than we did": if every conversion factor is halved the count is 3; if doubled it is 9. The 6ND version ("3 of 30 used more FLOPs", namely Ettin, BananaMind-2.1-Unified, Er-Large-30M) does not depend on any hardware factor.

## 6. Hardware conversion factors (assumptions)

Factor = approximate dense BF16/FP16 tensor peak divided by H100 SXM (989 TFLOPS). These come from my recollection of public spec sheets and were not re-verified against vendor documents; treat every factor as good to about 2x.

| GPU | Factor | | GPU | Factor |
|---|---|---|---|---|
| H100 | 1.00 | | RTX 4090 | 0.17 |
| A100 | 0.32 | | RTX 5090 | 0.21 |
| RTX PRO 6000 Blackwell | 0.50 | | RTX 5070 Ti / 5070 | 0.12 / 0.10 |
| RTX Pro 4500 SE | 0.20 | | RTX 5060 Ti / 5060 | 0.07 / 0.06 |
| TPU v5e chip | 0.20 | | L4 / T4 | 0.12 / 0.07 |
| RTX 3060 / 2060 | 0.03 / 0.03 | | RTX 3080 Ti | 0.08 |

H100-eq hours = number of GPUs x wall-clock hours x factor. Peak ratios ignore utilisation. Our run achieves about 20% of H100 peak on 6ND; the L3 median is about 6%. Peers running at low utilisation therefore look more expensive in H100-eq hours than the compute they strictly needed, which flatters us. The 6ND FLOPs column is the unflattering, hardware-free cross-check, and the two nearest competitors still sit at or above our FLOPs there. Sanity flags: implied utilisation above 0.4 (DynamicMind-MoE, 0.43, because 6ND on total MoE parameters overstates active compute) and 0.22 for Syn-2.6M (RTX 2060 factor may be low). 6ND understates compute for looped or recurrent models (pulvis, Purrence, min-spark, Sol-Lite, Bryne), which reuse blocks.

Break-even factors for our two nearest competitors, final run versus final run: BananaMind-2.1-Unified stays above us on compute unless the RTX PRO 6000 factor is below 0.14 (we use 0.50). Er-Large-30M stays above us unless the RTX 5070 factor is below 0.037 (we use 0.10), or below 0.074 if "116 hours" meant total GPU-hours rather than wall-clock on two cards.

If our 5.23 H100-h sweep is included (13.81 total), Unified still uses more unless the PRO 6000 factor is below 0.23, and Er-Large still uses more. Stentor3-20M (13.1, TPU v5e-8, free tier) and BananaMind-2-Micro (13.3) drop below our total. Peers rarely report ablations or pilot runs, so compare final runs and state our sweep separately. Peers also run sweeps without counting them: the pulvis-v1 card mentions about 30 pilot runs of 0.5 to 3B tokens on one RTX 5080, unquantified in GPU-hours.

## 7. Top 5 overall (context only)

Ranked by recomputed Int Index from the saved snapshot. Note that void.1 is rank 10 overall on this snapshot (23.92), not top 5. The ranks are: cagliostro-v3.5 27.49, SmolLM2-135M 27.13, GPT-X3-Preview-2 27.03, cagliostro-v3 26.55, SmolLM-135M 25.74.

| Rank | Model | Params | Tokens | Hardware | Time / GPU-h | Cost | Source |
|---|---|---|---|---|---|---|---|
| 1 | cagliostro-v3.5 | 146M | 75.39B (75B pretrain from v3 plus 0.39B post-training) | 1x RTX 5090 (pretrain, per v3 card) then 1x RTX PRO 6000 Blackwell (post-train) | about 9 days plus 41 min = about 216 RTX 5090 h, about 46 H100-eq h (estimate) | not disclosed | [v3.5](https://huggingface.co/bench-labs/cagliostro-v3.5), [v3](https://huggingface.co/bench-labs/cagliostro-v3) |
| 2 | SmolLM2-135M | 135M | 2T | 64x H100 | GPU-hours not disclosed. Estimate 1.5k to 2.3k H100-h (6ND = 1.6e21 FLOPs at 20 to 30% of peak; 4.5k at 10%) | not disclosed | [card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [paper](https://arxiv.org/abs/2502.02737) (paper's 256-H100 statement is for the 1.7B model) |
| 3 | GPT-X3-Preview-2 | 149M | not verified | not verified | not verified | not verified | card returns 401 ([link](https://huggingface.co/AxiomicLabs/GPT-X3-150M)); sibling GPT-X2.5-135M reports 75B tokens, 1x RTX 3080 Ti, about 800 h (about 60 H100-eq h estimate), [card](https://huggingface.co/AxiomicLabs/GPT-X2.5-135M) |
| 4 | cagliostro-v3 | 146M | 75B | 1x RTX 5090, 90 to 103k tok/s | about 9 days (about 216 h, about 45 H100-eq h estimate) | not disclosed | [card](https://huggingface.co/bench-labs/cagliostro-v3) |
| 5 | SmolLM-135M | 135M | 600B | 64x H100 | not disclosed. Estimate 0.45k to 0.7k H100-h (6ND = 4.9e20 at 20 to 30%) | not disclosed | [card](https://huggingface.co/HuggingFaceTB/SmolLM-135M) |
| 10 | void.1 | 90M | about 30 to 35B BPE tokens (card says about 120B bytes) | not disclosed | not disclosed | not disclosed | [card](https://huggingface.co/dotlabs/void.1) |

Other near-top entries: Haidass1.5-143M (rank 7) reports about 400B tokens on Huawei Ascend 910B NPUs, no hours; BananaMind-2-Pro (rank 8) reports 100B tokens and a "derived" 529.7 h at 52,438 tok/s with no hardware named; GPT-X2.5-135M (rank 6) as above.

Perspective: at 146M parameters cagliostro-v3.5 scores 27.49 for roughly 46 H100-eq hours (estimate); at 49.8M we score 16.13 for 8.58. Different sizes make these incomparable; neither is "more efficient".

## 8. Which efficiency claims survive

Ranked, best first.

1. **Compute-strict, n=30, qualified.** "Among sub-50M models that report their training compute, gibc-v2-lm has the highest Int Index (16.13, next 13.40). The two nearest reporters, BananaMind-2.1-Unified (est. 30 H100-eq h, 8.0e18 FLOPs) and Er-Large-30M (est. 23 H100-eq h, 6.5e18 FLOPs), used more compute than our 8.58 H100-h, 6.0e18 FLOPs." Hardware-free version: equal or lower 6ND FLOPs than both. Er-Large's FLOPs are only 8% above ours, so do not lean on that gap; the score gap (11.71 vs 16.13) is the point.
2. **Token-matched.** "Highest Int Index among sub-50M models that disclose a token count (n=91 peers). At about 20B tokens, every other entry scores lower: Blaze (20.97B) 15.45, Veyra2-Apricot (19.99B) 13.39, Supra2-Medium (20B) 12.09." The 0.68 gap to Blaze is probably inside evaluation noise; say "at or above" rather than "beats Blaze".
3. **Supporting only: residual from a fit.** Regressing Int Index on log10(params) and log10(tokens) across the 91 token disclosers (R2 0.55), gibc-v2-lm has the largest positive residual (+5.9), ahead of Blaze (+5.3) and BananaMind-2-Medium (+4.9). Regressing on log10 of 6ND alone (R2 0.44) we are also first (+7.2 vs Blaze +6.5). Regressing on log10 H100-eq hours across the L3 group gives R2 of 0.29 and we are the leverage point, so do not quote it. Report rank, not a z-score.
4. **Cost.** Only SupraGDN-5M (about $2, 5B tokens, 5M params, Int Index 3.88) and Stentor3-20M ($0) report any cost. We can state our own figure and nothing comparative.

One-liner: "Among sub-50M models that report their training compute, gibc-v2-lm has the highest Intelligence Index (16.13, on 8.58 H100-hours and 20B tokens); overall it is second under 50M, behind one model that does not disclose its compute."

## 9. What we must not claim

- "Most compute-efficient" or "most efficient sub-50M model", in any form. The top-ranked model (50M-pro) and two of the top six disclose nothing, so the sentence would be untestable.
- Int Index per H100-hour, ARC-Easy per H100-hour, or score per dollar. Int Index is floored at chance, so tiny runs win the ratio: we rank 22 of 31 L3 models at 1.9 points per H100-eq hour, led by LowOnMind-1M at 223 (11 minutes on a T4). The ratio rewards doing very little.
- "Below average", "low", or "small" compute. We are above the disclosed median by about 13x in H100-eq hours and about 11x in FLOPs.
- "Cheapest" or "lowest cost". Costs are not reported by peers.
- "Beats SmolLM2 / SmolLM / cagliostro per unit compute". Those are 135M to 146M models; SmolLM2 does not report GPU-hours; sizes differ.
- Any comparison with 50M-pro, ZeroS-Linear-50M or Linear-50M-exp on compute.
- A statistical-significance claim for the residual fits (z around 4 on the H100-eq fit is not meaningful with R2 0.29 and us as the leverage point).
- A compute comparison that quietly drops the sweep. If the 8.58 headline is quoted, say "final run" and give 13.81 total next to it when space allows.
- "Verified". Our numbers are from our own logs; every peer number here is read from a card.

## 10. Caveats

- Self-reported, unverified, often inconsistent (Glint-1.3). Stages and fine-tuning are handled inconsistently across cards.
- The L3 group is a biased sample: mostly hobbyist runs scoring 1.6 to 6, plus a handful above 8. Only 5 L3 peers exceed Int Index 8.
- Conversion factors are good to about 2x and ignore utilisation; use FLOPs when hardware independence matters.
- 6ND uses leaderboard parameter counts and ignores attention FLOPs, embedding handling and MoE or looped reuse.
- Cards could change after this snapshot.

## Draft request to Axiomic

Subject: Optional training-compute columns and an efficiency view

Hi Axiomic team, thank you for running the leaderboard; it has been very useful for sub-50M work.

Could you add optional compute disclosure per model: training tokens, GPU type and count, GPU-hours, and optionally cost? Most entries do not state this today. In the sub-50M group about four in five give a token count, but only about one in four give enough to derive GPU-hours.

An efficiency view would then be possible: Int Index against log training compute (6 x params x tokens, as your GPT-X2.5 card uses), with each model's residual from the fit, like your size fit. GPU-hours could serve as a second axis where reported.

Since values are self-reported, a "reported / verified" flag would keep the comparison honest.

We would disclose ours for gibc-v2-lm.

Thanks,
[name]
