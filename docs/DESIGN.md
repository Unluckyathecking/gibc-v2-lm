# Design notes

Why each part of the model and pipeline is there, and what we considered and dropped. The numbers come from the 1B-token sweep in the README unless stated otherwise.

## Constraints that shaped everything

- **50,000,000 trainable parameters, embeddings and head included.** At a 32k vocabulary the tied embedding is 16.8M, a third of the budget, so every other choice is judged per parameter.
- **Scored on HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103 perplexity.** The data mix and the anneal target these directly. WikiText-103 is built from Wikipedia, so contamination had to be dealt with before training, not after.
- **One H100 at a time on Modal, paid per second.** Throughput and a cheap ablation loop mattered more than peak scale. A 1B-token arm costs about 0.45 H100-hours, a 20B main run about 8.

## Model

| Component | Why |
|---|---|
| 14 layers x 512 wide, MLP 3x | A conventional deep-and-narrow shape for this budget. With the embedding fixed at 16.8M, 14 layers of 2.36M each fills the cap with 177,772 to spare. |
| Tied embedding | An untied 32k x 512 head would cost another 16.8M, a third of the budget, and would force a much smaller transformer. |
| 32k byte-level BPE, trained on our own mix | No pretrained tokenizer is allowed. 32k beat 16k in the sweep (WikiText ppl 57.2 vs 58.3, ARC-E 46.5 vs 44.2), even though the 16k arm spends the saved 8.4M on nothing. The byte-level alphabet means no unknown tokens. |
| Grouped-query attention, 4 KV heads | Halves Wk and Wv (262k saved per layer), and the savings go to the MLP. We did not ablate it; it is the standard trade at small scale. |
| RoPE, QK-norm, ReLU², zero-init output projections, logit softcap 15 | The modded-nanogpt baseline: well tested at this scale and stable under Muon's large learning rates. We did not re-ablate these. |
| U-net skip connections (7 scalars) | From modded-nanogpt. They give later layers direct access to early representations for almost no parameters. |
| Value residual (13 scalars) | Largest single gain in the sweep: WikiText ppl 57.2 -> 54.0, val bpb 1.1020 -> 1.0931, benchmarks flat. It costs 13 parameters and about 5% throughput. |

## Training

| Component | Why |
|---|---|
| Muon on block matrices, AdamW on the rest | AdamW alone was far worse at 1B tokens (WikiText ppl 89.0 vs 57.2, val bpb 1.22 vs 1.10). Muon's orthogonalised updates are the main reason small-model speedruns converge fast, and the embedding and scalars are not matrices Muon is meant for. |
| Muon lr 0.02, wd 0.01 | Doubling the lr to 0.04 was within noise, so we kept the lower, safer value for a run 20x longer. |
| Warmup-stable-decay, 35% linear decay | The long constant phase lets the two main runs use the same recipe for their first 65% and differ only in the decay. The decay window is also where the anneal mix goes. |
| Quality anneal during decay | At 1B it moved ARC-Easy up 1.5 and PIQA down 2.7. Our guess is that educational text helps ARC's science questions, while the everyday physical knowledge PIQA probes is more common in general web text. The mild anneal keeps about 35% general web text to test whether ARC gains can be kept without the PIQA loss. |
| 5% decontaminated Wikipedia in the anneal | Encyclopedic prose is close to WikiText-103's distribution, so it should help the perplexity metric without leaking the test set (see below). This was not tested at 1B. |
| Snapshot averaging | The last four decay-phase snapshots (250 steps apart; the last one is the final weights) are averaged as a cheap alternative weight set. Both final and averaged weights are evaluated, and the README states which one is reported. |

## What we rejected, and why

**Value embeddings.** Recent modded-nanogpt records add learned per-token embeddings to the attention values. Each table is at least 32,768 x 256 = 8.4M parameters at our vocabulary and KV width, and the records use several. We have 177,772 parameters of headroom. Fitting even one table would mean cutting about four transformer layers, and value residual gets a similar "early information reaches late layers" effect for 13 parameters.

**Exclusive Self Attention, NorMuon, cautious weight decay.** Each was a small, real-looking gain on WikiText (about 1.2 ppl) with benchmarks flat, so within noise at one seed. We did not stack several unconfirmed changes into an 8-hour run. NorMuon at modded-nanogpt's record hyperparameters (wd 1.2) was clearly worse at 1B tokens.

**Layer sharing (alt1: 8 wider blocks each applied twice).** Same quality as Config A within noise, at 25% lower throughput. It would pay off only if we were compute-rich and parameter-poor, and we are closer to the opposite.

**DCLM-heavier main mix.** 40% DCLM instead of 30% was within noise on everything, so we kept the more educational mix.

**Paid or restricted data.** The rules allow public datasets, and we wanted every source to be reproducible by anyone with a Hugging Face account under a permissive or share-alike licence. All five training sources are ODC-By, CC-BY or CC-BY-SA/GFDL.

**Synthetic or LLM-generated corpora** (for example textbook-style generated data). The rules forbid distilling from a larger model. Training on another model's generated text is close enough to distillation that we avoided it entirely, and it keeps the claim "trained from scratch on human-written text" simple and checkable.

**Raw Wikipedia, or any web data unfiltered.** WikiText-103's validation and test articles are Wikipedia articles, so including Wikipedia as-is would put the test set in the training data. The web sources contain copies too: before filtering, a 250M-token sample of our shards contained 4.96% of WikiText-103 test 13-grams, from 6 documents out of 182,375 that reproduce Wikipedia text. We drop every document in every source, training and validation, that shares any 13-gram with WikiText-103 validation or test. We also drop Wikipedia articles by title match, so a WikiText article is removed from the 2023 dump even where its text has since changed enough to share no 13-gram. Filtering by 13-gram over normalised words, rather than tokens, makes it independent of the tokenizer and of WikiText's spaced punctuation.

**Filtering the multiple-choice benchmarks from training data.** We measure 13-gram overlap with HellaSwag, ARC-Easy, PIQA and WinoGrande items but do not drop documents for it. We chose to report the overlap rather than act on it; it is in the README so readers can judge.
