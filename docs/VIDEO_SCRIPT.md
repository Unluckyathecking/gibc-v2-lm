# Demo video outline (target 3-4 minutes, hard limits 2-5)

## Rendering the video

The video is rendered automatically; the narration and slides live in [`video/script.json`](../video/script.json), which is the source of truth (the outline below is the original plan it was written from).

```bash
uv run modal volume get gibc-runs main_vres_anneal/final.pt runs/final.pt
uv run modal volume get gibc-data tok/tok32k/tokenizer.json runs/tokenizer.json
uv run python scripts/make_video.py --ckpt runs/final.pt --tok runs/tokenizer.json
```

This samples the demo prompts from the checkpoint, synthesises the narration with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (a local neural TTS, Apache-2.0, run on the CPU), renders one 1920x1080 slide per scene plus a typewriter animation of the demo output, and writes `video/gibc_demo.mp4` (H.264 + AAC 48 kHz, loudness-normalised to -16 LUFS), `video/captions.srt`, `video/slides/*.png` and `video/demo_output.json` (raw samples). It takes about a minute on an Apple laptop (`--reuse-demo` skips re-sampling the checkpoint) and needs `ffmpeg` and `espeak-ng` on the PATH (`brew install ffmpeg espeak-ng`). Kokoro is in the `dev` dependency group, so `uv sync` installs it; its weights (~330 MB) download from Hugging Face on first use, with no account or key.

The voice is `voice` in `script.json` (default `bf_emma`, British female; `speed` 1.0). Override it with `--voice`: British `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`, or American `af_heart`, `am_michael` (the first letter picks the accent; full list on the model card). To judge a voice quickly, `--sample --voice bm_george` renders only the first scene to `video/voice_sample.mp4` in about 10 seconds. `--tts say` falls back to macOS `say`, using `say_voice` and `rate` from `script.json` (or `--voice Daniel`).

`--tts openrouter --model <id> --voice <name> [--style "<prompt>"]` narrates through an OpenRouter TTS model instead, one request per scene (MP3, decoded with ffmpeg), reading `OPENROUTER_API_KEY` from the environment; the SDK retries 429 and 5xx responses and the script prints each request's cost. `--style` reaches Google Gemini TTS (as `speech_metadata.style`) and OpenAI TTS (as `instructions`) models and is ignored for others. With `--sample` the raw MP3 is kept next to the clip. Scene-1 comparison clips and the command that renders them are in `video/voice_samples/`.

The `values` block at the top of `video/script.json` holds the final numbers for the submitted model, `main_vres_anneal` (final weights): run name, benchmark scores, WikiText perplexity, val bits/byte, GPU hours, weights variant, a one-sentence remark and the demo checkpoint label. To change them, edit that block and rerun the command. The script lists any `[FINAL` placeholder still left when it finishes. `pronounce` maps written terms to how the voice should say them; captions keep the written form.

## Recording your own voiceover

To narrate the video yourself instead of using a synthetic voice, record one take per scene with the teleprompter, then render with `--tts voiceover`. Budget about 30 minutes for the eight scenes.

```bash
uv run python scripts/teleprompter.py        # opens http://127.0.0.1:8741/ in Chrome; Ctrl+C when done
uv run python scripts/make_video.py --voiceover-check
uv run python scripts/make_video.py --tts voiceover --reuse-demo --ckpt runs/final.pt --tok runs/tokenizer.json
```

The teleprompter shows each scene's slide with its narration underneath and highlights the words at a target pace (150 words per minute by default), so you can see how fast to speak. Each take starts after a 3-2-1 countdown. The footer shows the scene's target length and your length; the side panel lists which scenes are recorded. When you allow the microphone, the level meter in the top bar should reach about -20 to -10 dB while you speak and never show CLIPPING.

| Key | Action |
|---|---|
| Space | start a take (after the countdown) / stop it and save |
| R | retake: discard what is running and start again |
| P | play back this scene's saved take |
| Enter or → | next scene (while recording: stop, save, then move on) |
| ← | previous scene |
| - / + | slower / faster pace (or use the slider) |
| M | switch between record mode and rehearse mode (a run-through with nothing recorded) |
| D | summary: every scene's length against its target and the estimated video length |

Each take is saved as `video/voiceover/scene_NN.webm` (or `.m4a`/`.ogg`, depending on the browser) with a `scene_NN.json` holding its length; a retake overwrites it. `--voiceover-check` prints every take's length against its target and the resulting video length, which must be 150-300 s, and lists any missing scene. `--tts voiceover` converts each take with ffmpeg, trims the silence at both ends (gently: it keeps 0.2 s before and 0.3 s after your speech), loudness-normalises it to -16 LUFS like the other voices, and builds the video exactly as the other backends do, each slide lasting as long as its take plus the usual padding. It stops with a list of missing scenes if any take is absent. Captions stay one cue per sentence, spread across each take in proportion to sentence length; there is no forced alignment, so a cue can run slightly ahead of or behind your voice if you pause unevenly.

Use Chrome. Its MediaRecorder is the most reliable here; Safari and Firefox should work too, but are less tested. Chrome asks for microphone permission the first time; if you blocked it, click the icon at the right of the address bar, allow the microphone, and reload. The page must come from the teleprompter's own address (`http://127.0.0.1:...`), not from opening the HTML file directly, because browsers only allow the microphone on localhost or https. To use a USB microphone, plug it in before starting and pick it in Chrome's site settings or in System Settings > Sound > Input.

Tips for a clean recording:

- Record in a quiet room with soft furnishings (a bedroom beats a kitchen), with fans, notifications and the laptop's own fan noise kept as low as you can.
- Sit about 20 cm from the mic and keep the same position and distance for every scene, so the scenes sound alike when cut together. With the built-in mic, that means the same seat and the same laptop position throughout.
- Record scene 1 twice: treat the first take as a warm-up, then press R and do it properly.
- Rehearse a scene once in rehearse mode (M) if it has numbers or names you might trip over.
- Leave a breath of silence after the countdown and at the end before pressing Space; the trim removes it, along with the click of the Space key.
- Play each take back with P before moving on; retakes are cheap.

## Original outline

Screen recording with voice-over. Times are cumulative targets. The final numbers are filled in below.

## 0:00-0:20 What this is

- On screen: README title and the headline results table.
- Say: a 49.8M-parameter language model trained from scratch on 20B tokens of openly licensed text, with our own tokenizer, on one H100 in about 8 hours. It scores 31.9, 55.2, 61.9 and 51.6 on HellaSwag, ARC-Easy, PIQA and WinoGrande, and has WikiText-103 perplexity 39.4 (lm-eval-style windows).

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

- On screen: main-run training loss and val bpb curves (`results/main/train_loss.png`, `results/main/val_bpb.png`), with the point where the anneal starts at 13B tokens marked.
- Say: both runs use the same recipe for the first 65% of training and differ in the anneal mix.
- On screen: headline table. Read the submitted model's numbers: main_vres_anneal, HellaSwag 31.87, ARC-Easy 55.18, PIQA 61.92, WinoGrande 51.62, WikiText 39.42.
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
