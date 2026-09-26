# Demo video outline (target 3-4 minutes, hard limits 2-5)

## Rendering the video

The video is rendered automatically; the narration and slides live in [`video/script.json`](../video/script.json), which is the source of truth (the outline below is the original plan it was written from).

```bash
uv run modal volume get gibc-runs <submitted_run>/final.pt runs/final.pt
uv run modal volume get gibc-data tok/tok32k/tokenizer.json runs/tokenizer.json
uv run python scripts/make_video.py --ckpt runs/final.pt --tok runs/tokenizer.json
```

This samples the demo prompts from the checkpoint, synthesises the narration with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (a local neural TTS, Apache-2.0, run on the CPU), renders one 1920x1080 slide per scene plus a typewriter animation of the demo output, and writes `video/gibc_demo.mp4` (H.264 + AAC 48 kHz, loudness-normalised to -16 LUFS), `video/captions.srt`, `video/slides/*.png` and `video/demo_output.json` (raw samples). It takes about a minute on an Apple laptop (`--reuse-demo` skips re-sampling the checkpoint) and needs `ffmpeg` and `espeak-ng` on the PATH (`brew install ffmpeg espeak-ng`). Kokoro is in the `dev` dependency group, so `uv sync` installs it; its weights (~330 MB) download from Hugging Face on first use, with no account or key.

The voice is `voice` in `script.json` (default `bf_emma`, British female; `speed` 1.0). Override it with `--voice`: British `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`, or American `af_heart`, `am_michael` (the first letter picks the accent; full list on the model card). To judge a voice quickly, `--sample --voice bm_george` renders only the first scene to `video/voice_sample.mp4` in about 10 seconds. `--tts say` falls back to macOS `say`, using `say_voice` and `rate` from `script.json` (or `--voice Daniel`).

To finish the video once the main runs are evaluated, edit only the `values` block at the top of `video/script.json` (every `[FINAL NUMBERS]` entry: run name, benchmark scores, WikiText perplexity, val bits/byte, GPU hours, weights variant, a one-sentence remark and the demo checkpoint label) and rerun the command with the final checkpoint. The script lists any placeholder still left when it finishes. `pronounce` maps written terms to how the voice should say them; captions keep the written form.

## Original outline

Screen recording with voice-over. Times are cumulative targets. Anything in `{{...}}` is filled in once the main runs finish.

## 0:00-0:20 What this is

- On screen: README title and the headline results table.
- Say: a 49.8M-parameter language model trained from scratch on 20B tokens of openly licensed text, with our own tokenizer, on one H100 in about 8 hours. It scores {{HEADLINE_ONE_LINE}} on HellaSwag, ARC-Easy, PIQA and WinoGrande, and has WikiText-103 perplexity {{MAIN_LMWT_WPPL}}.

## 0:20-1:05 Architecture and the parameter budget

- On screen: the architecture table, then run `uv run python scripts/count_params.py` live.
- Say: 14 layers, width 512, grouped-query attention, RoPE, QK-norm, ReLU², a tied 32k embedding, U-net skips and a value residual.
- Point at the count: 49,822,228, which is 177,772 under the cap. The tied embedding is a third of the budget; it is one tensor, counted once here, and 66.6M if counted twice. Say this plainly.
- Say: value residual was the biggest win in the sweep. It cost 13 parameters and cut WikiText perplexity from 57.2 to 54.0.

## 1:05-1:45 How we chose: the 1B sweep

- On screen: `results/sweep_1B/val_bpb.png`, then the ablation table in the README.
- Say: 12 arms of 1B tokens each, about 25 minutes per arm on an H100, 5.2 GPU-hours in total.
- Three findings: Muon beats AdamW by a wide margin (WikiText ppl 57 vs 89). Value residual helps. The quality anneal trades PIQA for ARC-Easy, so we run two strengths.
- Say it honestly: with one seed, most benchmark differences under 2 points are noise, so we leaned on perplexity.

## 1:45-2:15 Data and decontamination

- On screen: the data table with licences, then the decontamination section.
- Say: FineWeb-Edu, DCLM, FineMath, FinePDFs and Wikipedia, all public and permissively or share-alike licensed; no synthetic data and no pretrained weights.
- Say: WikiText-103 comes from Wikipedia and is mirrored on the web. Before filtering, 4.96% of its test 13-grams were in a training sample. We drop every document sharing any 13-gram with WikiText validation or test, and Wikipedia articles by title: 4,969 documents by 13-gram and 16 by title.

## 2:15-2:50 Main-run curves and final numbers

- On screen: main-run training loss and val bpb curves ({{MAIN_CURVES_PNG}}), with the point where the anneal starts at 13B tokens marked.
- Say: both runs use the same recipe for the first 65% of training and differ in the anneal mix.
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
