---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - small-language-model
  - from-scratch
  - pretrained
  - base-model
datasets:
  - HuggingFaceFW/fineweb-edu
  - mlfoundations/dclm-baseline-1.0-parquet
  - HuggingFaceTB/finemath
  - HuggingFaceFW/finepdfs
  - wikimedia/wikipedia
model-index:
  - name: gibc-v2-lm
    results:
      - task: {type: text-generation}
        dataset: {name: HellaSwag, type: Rowan/hellaswag}
        metrics: [{type: acc_norm, value: 31.87, name: acc_norm (0-shot)}]
      - task: {type: text-generation}
        dataset: {name: ARC-Easy, type: allenai/ai2_arc, config: ARC-Easy}
        metrics: [{type: acc_norm, value: 47.85, name: acc_norm (0-shot)}]
      - task: {type: text-generation}
        dataset: {name: ARC-Challenge, type: allenai/ai2_arc, config: ARC-Challenge}
        metrics: [{type: acc_norm, value: 26.11, name: acc_norm (0-shot)}]
      - task: {type: text-generation}
        dataset: {name: PIQA, type: ybisk/piqa}
        metrics: [{type: acc_norm, value: 60.28, name: acc_norm (0-shot)}]
      - task: {type: text-generation}
        dataset: {name: ArithMark-3.0, type: AxiomicLabs/ArithMark-3.0}
        metrics: [{type: acc_norm, value: 40.20, name: acc_norm (0-shot)}]
---

# gibc-v2-lm

A 49.8M-parameter base language model, **pretrained from scratch** (random initialisation, own tokenizer, no teacher, no distillation) on 20B tokens of openly licensed English text in 8.58 H100-hours. It was built for the [Global Innovation Build Challenge V2](https://gibc-v2.devpost.com/), Track 01 (foundational LLM development, 50M-parameter cap). Code, training logs, ablations and every eval JSON: **https://github.com/Unluckyathecking/gibc-v2-lm**.

This is a base model: it continues text and has no instruction or chat tuning.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Unluckyexe/gibc-v2-lm"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids  # <|bos|> is prepended
with torch.no_grad():
    for _ in range(20):
        nxt = model(ids).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
print(tok.decode(ids[0], skip_special_tokens=True))
```

The model has no KV cache, so `generate()` is not wired up; the loop above (or `scripts/demo.py` in the GitHub repo) is the intended way to sample. Context is 1,024 tokens. Every training document starts with `<|bos|>` (id 0), and the tokenizer prepends it by default; scores are best with it.

## Files

| File | What it is |
|---|---|
| `model.safetensors` | final weights (fp32), loaded by `modeling_gibc.py` |
| `final.pt` | the raw training checkpoint the safetensors was converted from (`{"model": state_dict, "model_cfg": ...}`, load with `torch.load(..., weights_only=True)`) |
| `configuration_gibc.py`, `modeling_gibc.py` | self-contained model code, a copy of `gibc/model.py` with identical maths; logits match the training code exactly |
| `tokenizer.json` | the model's own 32,768-entry byte-level BPE tokenizer |

## Model

| | |
|---|---|
| Parameters | 49,822,228 (the tied embedding/head counted once; 33,045,012 non-embedding) |
| Layers | 14, pre-norm RMSNorm, no biases |
| Width | d_model 512, MLP 1,536 with ReLU² |
| Attention | 8 query heads, 4 KV heads (GQA), head dim 64, RoPE (base 10,000), QK-norm |
| Extras | U-net skip connections (layer i to layer 13 - i), ResFormer value residual, logit softcap 15 |
| Embedding | 32,768 x 512, tied input embedding and output head |
| Context | 1,024 tokens |

## Training

- **From scratch.** Random initialisation; tokenizer trained by us on a 3 GB sample of the training mix. No pretrained weights, teacher model or synthetic/LLM-generated dataset was used.
- **Tokens.** 20B (38,146 steps x 524,288 tokens).
- **Optimiser.** Muon (lr 0.02, wd 0.01) on block matrices, AdamW (lr 3e-3) on embeddings, norms and scalars; warmup-stable-decay schedule (1% warmup, linear decay to 0 over the last 35%).
- **Data, main phase (13B tokens).** FineWeb-Edu 60%, DCLM-baseline 30%, FineMath 5%, FinePDFs 5%.
- **Data, decay phase (7B tokens).** FineWeb-Edu int_score>=4 60%, FineMath 22.5%, FineWeb-Edu 7.5%, DCLM 5%, English Wikipedia 5%.
- **Compute.** 8.58 H100-hours on one NVIDIA H100 80GB (plus a 5.23 H100-hour, 12-arm 1B-token ablation sweep that chose the recipe).
- **Decontamination.** Every training and validation document sharing a 13-gram with WikiText-103 validation or test was dropped (4,969 documents), plus 16 Wikipedia articles matching WikiText titles. HellaSwag, ARC, PIQA and WinoGrande data never entered training; their measured 13-gram overlap with a 300M-token training sample is 3, 1, 2 and 0 items respectively. ArithMark is synthetic and was not filtered; we did not measure its overlap.

Run name in the repository: `main_vres_anneal`, final weights.

## Evaluation

All zero-shot, full sets, percent, with standard errors.

| Benchmark | Metric | Score |
|---|---|---|
| HellaSwag | acc_norm | 31.87 ± 0.47 |
| ARC-Easy | acc_norm / acc | 47.85 ± 1.03 / 55.18 ± 1.02 |
| ARC-Challenge | acc_norm / acc | 26.11 ± 1.28 / 22.10 ± 1.21 |
| PIQA | acc_norm / acc | 60.28 ± 1.14 / 61.92 ± 1.13 |
| WinoGrande | acc | 51.62 ± 1.40 |
| ArithMark-3.0 | acc_norm (primary) / acc | 40.20 ± 1.55 / 40.20 |
| ArithMark-2.0 | acc | 36.80 ± 0.96 |
| WikiText-103 test | word perplexity (lm-eval-style windows) | 39.42 |

Benchmarks use lm-evaluation-harness 0.4.13 with `add_bos_token=True` and bf16 autocast, matching the GitHub repo's pipeline; a plain fp32 `lm_eval --model hf` run on this repo gives the same numbers to within 0.25 points (HellaSwag 31.89, ARC-E 47.85, ARC-C 26.19, PIQA 60.45 acc_norm). ArithMark uses the unmodified scripts from the AxiomicLabs dataset repos (bf16, batch size as shipped). ArithMark standard errors are binomial, sqrt(p(1-p)/n). Reproduce on this repo:

```bash
lm_eval --model hf --model_args pretrained=Unluckyexe/gibc-v2-lm,trust_remote_code=True \
    --tasks hellaswag,arc_easy,arc_challenge,piqa --num_fewshot 0 --batch_size 32
python bencharithmark-3.py --model Unluckyexe/gibc-v2-lm          # from AxiomicLabs/ArithMark-3.0
python benchmark_arithmark-2.0.py --model Unluckyexe/gibc-v2-lm   # from AxiomicLabs/ArithMark-2.0
```

## Limitations

A 50M-parameter base model: it produces fluent but often wrong text, has no safety tuning, and reflects the biases of filtered web data. English only. Not for production use.

## Licence

Weights and code: Apache-2.0. Training data licences (ODC-By, CC-BY-4.0, CC-BY-SA/GFDL) are listed in the GitHub README.

## Disclosure

The code was written by AI coding agents under human direction; see the GitHub README.
