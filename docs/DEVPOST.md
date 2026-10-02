# Devpost submission copy

Everything to paste into the Devpost form for GIBC V2, Track 01. The numbers match README.md. Requirements below were checked against https://gibc-v2.devpost.com/rules on 2026-09-26.

## 1. Name and tagline

**Project name:** gibc-v2-lm

**Tagline:** A 49.8M-parameter language model trained from scratch on 20B tokens of open web, maths, PDF and Wikipedia text, on one H100 in about 8 hours.

## 2. About the project

Paste in this order: the Headline results block, then the links, then the sections below.

```markdown
## Headline results

**No training on any benchmark.** HellaSwag, ARC, PIQA and WinoGrande data (train, validation or test) never entered training, and there is no fine-tuning stage of any kind. Every score below is zero-shot on the full evaluation set, lm-evaluation-harness 0.4.13.

**Submitted model:** `main_vres_anneal`, 49,822,228 trainable parameters, trained from scratch on 20B tokens, one NVIDIA H100, 8.58 GPU-hours.

| Benchmark | Score |
|---|---|
| HellaSwag (acc_norm) | **31.87** |
| ARC-Easy (acc) | **55.18** |
| PIQA (acc) | **61.92** |
| WinoGrande (acc) | 51.62 |
| WikiText-103 test word perplexity (lm-eval-style windows) | **39.42** |
| WikiText-103 test word perplexity (stride-512 sliding window) | 35.21 |

**Reference:** Pythia-70M (300B tokens) reports HellaSwag 26.6, ARC-Easy 36.9, PIQA 60.0.

**Methods:** Muon + AdamW, warmup-stable-decay, quality anneal on curated data for the last 35%, ResFormer value residual, grouped-query attention, RoPE, QK-norm, ReLU², U-net skips, logit softcap, tied 32k BPE embedding. Every choice tested first in a 12-arm, 1B-token ablation sweep. Training data decontaminated against WikiText-103 (4,969 documents dropped); benchmark overlap measured and reported.

**Code and full write-up:** https://github.com/Unluckyathecking/gibc-v2-lm
**Demo video:** https://youtu.be/rbA6RGoDeYU
```

### Inspiration

Track 01 caps the model at 50 million trainable parameters, embeddings and output head included. With a 32k vocabulary the embedding alone costs 16.8M, a third of the budget. We wanted to see how much a single rented H100 and the modded-nanogpt speedrun techniques could get out of the remaining two thirds, and to report every number plainly, including the ones that argue against our choices.

### What it does

gibc-v2-lm is a 14-layer decoder-only transformer, width 512, with 49,822,228 trainable parameters, under the 50,000,000 cap. It is trained from random initialisation on 20B tokens with our own 32k byte-level BPE tokenizer, pretraining only: no fine-tuning stage and no benchmark data of any kind in training. All scores are zero-shot.

The submitted model is main_vres_anneal (final weights). Zero-shot on the full sets it scores HellaSwag 31.87 (acc_norm), ARC-Easy 55.18 (acc), PIQA 61.92 (acc), WinoGrande 51.62 and WikiText-103 test word perplexity 39.42 with lm-eval-style windows (35.21 with a stride-512 sliding window). The second main run, main_vres_mild, which uses a milder anneal, scores 32.15, 53.24, 61.97, 51.46 and 38.76. We submit the anneal run for its 1.9-point ARC-Easy lead at equal PIQA and WinoGrande. A third 20B run, main_alt2_vres_anneal, is a vocabulary-size ablation: the same recipe with a 16k tokenizer and 41,433,620 trainable parameters. It scores 31.61, 53.75, 61.43, 50.67 and 42.46. For scale, Pythia-70M reports HellaSwag 26.6, ARC-Easy 36.9 and PIQA 60.0 after 300B tokens, with a different recipe and data.

### How we built it

The recipe is Muon on the block matrices and AdamW on the embedding and scalars, with QK-norm, ReLU², U-net skips and a logit softcap from modded-nanogpt, plus a ResFormer value residual. Training uses a warmup-stable-decay schedule; for the last 35% the loader switches to a quality anneal mix of high-scoring FineWeb-Edu, FineMath and 5% decontaminated Wikipedia. The two main runs differ only in how hard the anneal leans on curated data. The submitted run took 8.58 H100-hours; the 1B-token sweep that chose its recipe took 5.23.

Before committing to a 20B run we ran a 12-run, 1B-token ablation sweep: a baseline plus 11 single-change arms, each about half an H100-hour.

The data is all public: FineWeb-Edu, DCLM-baseline, FineMath, FinePDFs and Wikipedia, under ODC-By, CC-BY-4.0 and CC-BY-SA/GFDL. No intentionally synthetic or LLM-generated dataset, no teacher model, no distillation data, no pretrained weights and no pretrained tokenizer; the web-derived corpora may incidentally contain AI-generated pages, which no public crawl can rule out. Every document sharing a 13-gram with WikiText-103 validation or test is dropped, 4,969 in total, plus 16 Wikipedia articles by title. We measured but did not filter the multiple-choice benchmarks: 3 HellaSwag, 1 ARC-Easy, 2 PIQA and 0 WinoGrande items overlap a 300M-token sample of the training data.

Evaluation uses lm-evaluation-harness 0.4.13 and our own WikiText-103 script, which follows lm-eval's normalisation. Everything runs on Modal.

This project was built with Claude Code agents (Claude Opus 5.5 and Claude Fable 5.1), which wrote the code and docs under the direction and review of the human participant, who can explain every component.

### Challenges we ran into

WikiText-103 is made of Wikipedia articles, and the web copies them. Before filtering, a 250M-token sample of our shards contained 4.96% of WikiText-103 test 13-grams, from 6 web documents. The cap was the other constraint: the tied embedding alone takes a third of the 50M budget. One seed at 1B tokens gives standard errors of 0.45 to 1.4 benchmark points, so most decisions rest on perplexity and bits per byte. Near the end, both main runs were stopped by our Modal spend limit and resumed from checkpoints with model, optimizer, RNG and data cursors restored; the README describes this, including 218 training-log lines lost from one run when two trainers briefly ran at once.

### Accomplishments that we're proud of

Every change in the main runs was tested first. A 20B-token 16k-vocabulary ablation confirmed the sweep's choice of 32k, which leads by 0.3 to 1.4 benchmark points and 7% WikiText perplexity. Value residual cut 1B WikiText perplexity from 57.2 to 54.0 for 13 parameters, and Muon beat AdamW-only 57.2 to 89.0. The whole pipeline reproduces from public data with documented commands, and the README reports the unflattering results too.

### What we learned

At this size the embedding decides the architecture. Perplexity is a far steadier signal than the benchmarks. The quality anneal traded PIQA for ARC-Easy at 1B, which we could not resolve, so both strengths became main runs. The Wikipedia anneal arm finished after the main runs started: ARC-Easy 48.74 and WikiText perplexity 54.70, against 47.98 and 58.51 without Wikipedia. At 20B tokens the trade-off mostly went away: the strong anneal kept its ARC-Easy gain and the PIQA drop seen at 1B did not appear (61.92 vs 61.97). Averaging the last four checkpoints gave nothing, because after a decay to zero they are nearly identical.

### What's next

Multiple seeds for the close sweep results, a stacked run with Exclusive Self Attention and NorMuon, which each gave about 1.2 perplexity at 1B, and longer context than 1,024 tokens.

## 3. Built With

One tag per line.

Technologies

```
python
pytorch
torch.optim.muon
hugging-face-tokenizers
hugging-face-transformers
huggingface-hub
lm-evaluation-harness
pyarrow
numpy
matplotlib
modal
nvidia-h100
nvidia-l4
uv
ffmpeg
```

AI tools

```
claude-code
claude-opus-5.5
claude-fable-5.1
```

Datasets

```
fineweb-edu
dclm-baseline
finemath
finepdfs
wikipedia
wikitext-103
hellaswag
arc-easy
piqa
winogrande
```

`transformers`, `huggingface_hub`, PyArrow, NumPy and Matplotlib are listed because the README's Built With section and the video pipeline use them, and the rules ask for every technology. The four benchmarks are listed as evaluation datasets.

## 4. Screenshots

Upload at least 3. Captions are one line each.

| # | File | Caption |
|---|---|---|
| 1 | `results/main/val_bpb.png` | Validation bits per byte for the two 20B-token main runs. |
| 2 | `results/main/train_loss.png` | Training loss for the two main runs; the anneal mix starts at step 24,795. |
| 3 | `results/sweep_1B/val_bpb.png` | Validation bits per byte for the 1B-token ablation sweep, all arms overlaid. |
| 4 | `video/slides/05_sweep.png` | Ablation results: value residual gives the largest WikiText gain, AdamW-only is far behind Muon. |
| 5 | `video/slides/02_architecture.png` | Architecture and parameter count: 49,822,228 trainable parameters, 177,772 under the cap. |

There is no standalone image of the full ablation table; slide 05 shows the key rows. If a screenshot of the raw count printout is wanted, capture the terminal output of `uv run python scripts/count_params.py`. Items 1 and 2 are the rendered main-run curves.

## 5. Submission checklist

| Requirement (rules) | Where it is satisfied | Status |
|---|---|---|
| Project description | Form: "About the project", section 2 above | Ready |
| Public source repo with README: setup, prerequisites, run instructions | https://github.com/Unluckyathecking/gibc-v2-lm, README "Reproduction" | **Repo is currently private. Make it public before submitting.** |
| Demo video, 2-5 min, YouTube/Vimeo/Youku, unlisted OK, English audio or subtitles | `video/gibc_demo.mp4` (4 min 25 s, English narration), `video/captions.srt`; form: video link https://youtu.be/rbA6RGoDeYU | Rendered with final numbers; upload unlisted, add the SRT as subtitles, paste link |
| Built With: every technology, library, API, dataset, hardware, AI tools | Form: Built With, section 3 above | Ready |
| Team: all members by real full name, each with a Devpost account added to the submission | Form: team members | Add Mohammed Alibhai (and any teammates). Under-18 entrants need parental or guardian permission. |
| At least 3 screenshots | Form: image gallery, section 4 above | Ready (5 images) |
| Track selection | Form: Track 01, TECH (foundational LLM development) | Select on submit |
| Track 01: at most 50,000,000 trainable params incl. embeddings and head; count script and config in repo | README "Parameter count", `scripts/count_params.py`, `gibc/configs.py` | Done: 49,822,228 trainable parameters, printed by `scripts/count_params.py`; embedding and head are one tied tensor, included in the count |
| Track 01: trained from scratch, no pretrained weights, fine-tuning or distillation | README intro, "Data", "AI use disclosure" | Done |
| Track 01: hardware, total training time, approximate compute in README | README "Hardware, time and compute" | Done: 8.58 H100-hours for the submitted model plus 5.23 for the 12-arm sweep, run history |
| Track 01: HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-eval, WikiText-103 perplexity; numbers and eval script in README | README "Headline results", "Evaluation protocol", `gibc/evaluate.py`, `modal_eval.py` | Done |
| AI tools disclosed in Built With and AI-assisted parts noted in README | Built With (section 3), README "AI use disclosure" | Done |
| Commits pushed after the deadline may be disregarded | Push final README and results before the deadline | Pending |

Judging criteria for Track 01, as the rules word them: Perplexity & Accuracy; Reasoning Performance; Training Efficiency; Innovation; Documentation & Demo. Ties go to the first-listed criterion.

**Deadline:** 1 October 2026, 23:45 UTC+8 = 15:45 UTC = 16:45 BST (UK).

**Before pressing submit:** make the GitHub repo public, confirm no placeholder is left in README.md, `video/script.json` or this form, and confirm nothing is left unpushed.
