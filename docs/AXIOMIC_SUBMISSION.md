# Axiomic Labs Open SLM Leaderboard submission

Target: Hugging Face Space `AxiomicLabs/Open_SLM_Leaderboard` (edit `index.html`, the `ORGS` and `MODELS` constants under "CENTRALIZED DATA"). Not yet opened.

Model: https://huggingface.co/Unluckyexe/gibc-v2-lm (public; run `main_vres_anneal`, final weights, 49,822,228 parameters, trained from scratch). Code and logs: https://github.com/Unluckyathecking/gibc-v2-lm.

## Which metric the board uses

`acc_norm` for HellaSwag, ARC-Easy, ARC-Challenge and PIQA. Check against rows already on the board and their lm-eval 0-shot numbers: GPT-2 is listed as ARC-E 39.35 / HS 31.26 / PIQA 62.08 / ARC-C 22.35, where lm-eval gives acc_norm 39.5 / 31.1 / 62.5 / 22.7 and acc 43.8 / 28.9 / 62.9 / 19.0; OPT-125M is listed as 40.28 / 31.31 / 62.24 / 22.70 against acc_norm 40.1 / 31.3 / 62.0 / 22.9. ARC and HellaSwag rule out `acc`; PIQA is closer to `acc_norm`. ArithMark-3 is the official script's primary metric (`acc_norm`, which equals `acc` for our model); ArithMark-2 is its script's only metric (`acc`).

## Lines to add

Append to `ORGS`:

```js
    unluckyexe:  { name: 'Unluckyexe', chartColor: 'rgba(16, 185, 129, 0.70)', chartBorder: '#10b981', url: 'https://huggingface.co/Unluckyexe' },
```

Append to the bottom of `MODELS` (the file says "ADD NEW MODELS TO BOTTOM OF LIST"):

```js
    { name: 'gibc-v2-lm', org: 'unluckyexe', params: 49822228, paramsDisplay: '49.8M', arc: 47.85, hellaswag: 31.87, piqa: 60.28, arcChall: 26.11, arithmark3: 40.20, arithmark2: 36.80, links: { card: 'https://huggingface.co/Unluckyexe/gibc-v2-lm' } },
```

With the page's formula this gives an Intelligence Index of 16.13: (HS 9.16 + ARC 15.97 + PIQA 20.56 + 0.65 x AM3 20.27) / 3.65.

## PR description

> **Add gibc-v2-lm (49.8M, Unluckyexe)**
>
> gibc-v2-lm is a 49,822,228-parameter base model pretrained from scratch (random init, own 32k BPE tokenizer, no teacher or distillation) on 20B tokens of FineWeb-Edu, DCLM, FineMath, FinePDFs and Wikipedia in 8.58 H100-hours. It was our entry to GIBC V2 Track 01 (50M cap). The weights on the Hub are the final training checkpoint; `final.pt` is uploaded next to the converted `model.safetensors`.
>
> Weights: https://huggingface.co/Unluckyexe/gibc-v2-lm
> Code, training logs, eval JSON: https://github.com/Unluckyathecking/gibc-v2-lm
>
> | Benchmark (0-shot, full set) | Score | Std err |
> |---|---|---|
> | HellaSwag acc_norm | 31.87 | 0.47 |
> | ARC-Easy acc_norm | 47.85 | 1.03 |
> | ARC-Challenge acc_norm | 26.11 | 1.28 |
> | PIQA acc_norm | 60.28 | 1.14 |
> | ArithMark-3 (primary, acc_norm) | 40.20 | 1.55 |
> | ArithMark-2 acc | 36.80 | 0.96 |
>
> Commands (the model needs `trust_remote_code=True`; it is a custom GPT with GQA, QK-norm, U-net skips and a value residual):
>
> ```
> lm_eval --model hf --model_args pretrained=Unluckyexe/gibc-v2-lm,trust_remote_code=True,add_bos_token=True,mixed_precision_dtype=bfloat16 \
>     --tasks hellaswag,arc_easy,arc_challenge,piqa --num_fewshot 0 --batch_size 32
> python bencharithmark-3.py --model Unluckyexe/gibc-v2-lm        # AxiomicLabs/ArithMark-3.0 @ 6f6e59d, unmodified, defaults
> python benchmark_arithmark-2.0.py --model Unluckyexe/gibc-v2-lm # AxiomicLabs/ArithMark-2.0 @ 5bd986c, unmodified, defaults
> ```
>
> lm-evaluation-harness 0.4.13, transformers 5.17, torch 2.11, one L4. The lm-eval numbers above use bf16 autocast and an explicit BOS, which is how the model was trained and how the GitHub repo evaluates it; a plain fp32 run without the two extra model_args gives HellaSwag 31.89, ARC-E 47.85, ARC-C 26.19, PIQA 60.45 (the tokenizer adds BOS by default). ArithMark std errors are binomial. None of the four lm-eval benchmarks was used in training; ArithMark was not filtered.

## Discussion-post fallback

> **Submission: gibc-v2-lm (49.8M, from scratch)**
>
> Hi, I'd like to add https://huggingface.co/Unluckyexe/gibc-v2-lm, a 49.8M-parameter model pretrained from scratch on 20B tokens (final checkpoint, open weights, code at https://github.com/Unluckyathecking/gibc-v2-lm). Zero-shot results, lm-eval 0.4.13 and the official ArithMark scripts: HellaSwag 31.87, ARC-Easy 47.85, ARC-Challenge 26.11, PIQA 60.28 (all acc_norm), ArithMark-3 40.20, ArithMark-2 36.80. It loads with `trust_remote_code=True`. Suggested row:
>
> `{ name: 'gibc-v2-lm', org: 'unluckyexe', params: 49822228, paramsDisplay: '49.8M', arc: 47.85, hellaswag: 31.87, piqa: 60.28, arcChall: 26.11, arithmark3: 40.20, arithmark2: 36.80, links: { card: 'https://huggingface.co/Unluckyexe/gibc-v2-lm' } },`
>
> plus an `ORGS` entry `unluckyexe` (https://huggingface.co/Unluckyexe). Happy to rerun anything. Thanks!

## How these numbers were produced

- `uv run modal run modal_eval.py::main --run main_vres_anneal --variants final` -> `results/main/main_vres_anneal_eval_final.json` (adds ARC-Challenge; the other tasks reproduce the earlier JSON exactly).
- `uv run modal run modal_axiomic.py::arithmark` -> `results/main/main_vres_anneal_arithmark.json` (rebuilds the HF release folder from `final.pt`, then runs the two official scripts on it unmodified).
- `uv run modal run modal_axiomic.py::lm_eval_release` -> `results/main/main_vres_anneal_lmeval_release.json` (lm-eval CLI on the release folder, plain and with our settings).
- `uv run modal run modal_axiomic.py::upload` pushed the release to the Hub.

Rank at submission time (board snapshot of 2026-10-08, 123 other models under 50M): Intelligence Index 2nd (50M-pro 17.93), ARC-Easy 1st, ArithMark-3 2nd (Tokle-3M 40.80), HellaSwag 5th, ARC-Challenge 6th, ArithMark-2 7th of 64, PIQA 9th.
