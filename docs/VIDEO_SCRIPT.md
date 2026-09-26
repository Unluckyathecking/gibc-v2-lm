# Demo video outline (target 3-4 minutes, hard limits 2-5)

Screen recording with voice-over. Times are cumulative targets. Anything in `{{...}}` is filled in once the main runs finish.

## 0:00-0:20 What this is

- On screen: README title and the headline results table.
- Say: a 49.8M-parameter language model trained from scratch on 20B tokens of openly licensed text, with our own tokenizer, on one H100 in about 6.5 hours. It scores {{HEADLINE_ONE_LINE}} on HellaSwag, ARC-Easy, PIQA and WinoGrande, and has WikiText-103 perplexity {{MAIN_LMWT_WPPL}}.

## 0:20-1:05 Architecture and the parameter budget

- On screen: the architecture table, then run `uv run python scripts/count_params.py` live.
- Say: 14 layers, width 512, grouped-query attention, RoPE, QK-norm, ReLU², a tied 32k embedding, U-net skips and a value residual.
- Point at the count: 49,822,228, which is 177,772 under the cap. The tied embedding is a third of the budget; it is one tensor, counted once here, and 66.6M if counted twice. Say this plainly.
- Say: value residual was the biggest win in the sweep. It cost 13 parameters and cut WikiText perplexity from 57.2 to 54.0.

## 1:05-1:45 How we chose: the 1B sweep

- On screen: `results/sweep_1B/val_bpb.png`, then the ablation table in the README.
- Say: 11 arms of 1B tokens each, about 25 minutes per arm on an H100, 4.8 GPU-hours in total.
- Three findings: Muon beats AdamW by a wide margin (WikiText ppl 57 vs 89). Value residual helps. The quality anneal trades PIQA for ARC-Easy, so we run two strengths.
- Say it honestly: with one seed, most benchmark differences under 2 points are noise, so we leaned on perplexity.

## 1:45-2:15 Data and decontamination

- On screen: the data table with licences, then the decontamination section.
- Say: FineWeb-Edu, DCLM, FineMath, FinePDFs and Wikipedia, all public and permissively or share-alike licensed; no synthetic data and no pretrained weights.
- Say: WikiText-103 comes from Wikipedia and is mirrored on the web. Before filtering, 4.96% of its test 13-grams were in a training sample. We drop every document sharing any 13-gram with WikiText validation or test, and Wikipedia articles by title: {{DECONTAM_DROPPED_NGRAM_TOTAL}} documents in total.

## 2:15-2:50 Main-run curves and final numbers

- On screen: main-run training loss and val bpb curves ({{MAIN_CURVES_PNG}}), with the point where the anneal starts at 13B tokens marked.
- Say: both runs share the first 65% of training and differ only in the anneal mix.
- On screen: headline table. Read the submitted model's numbers: {{MAIN_SUBMITTED_RUN}}, HellaSwag {{...}}, ARC-Easy {{...}}, PIQA {{...}}, WinoGrande {{...}}, WikiText {{...}}.
- Say: evaluation uses lm-evaluation-harness 0.4.13, zero-shot, on the full sets. The command is `modal run modal_eval.py::main --run ...`.

## 2:50-3:40 Live generation

- On screen: terminal on a laptop, then run
  `uv run python scripts/demo.py --ckpt runs/final.pt --tok runs/tokenizer.json -i`
- Prompts, prepared in advance and picked to show range. Keep the good ones and the bad ones:
  - "The capital of France is"
  - "Photosynthesis is the process by which"
  - "To boil an egg, you"
  - "The derivative of x squared is"
- Say: it runs locally on a laptop, streaming. Point out one fluent continuation and one factual or reasoning error, because a 50M model makes both.

## 3:40-4:00 Close

- On screen: the AI use disclosure section.
- Say: the code, experiments and docs were written by Claude Code agents under my direction; I set goals, made the calls, reviewed the work and paid for compute. Everything needed to reproduce the result is in the README.
