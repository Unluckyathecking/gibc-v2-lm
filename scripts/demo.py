"""Generate text from a trained checkpoint on a laptop (MPS or CPU), for the demo video.

Usage:
  modal volume get gibc-runs main_A/final.pt runs/main_A/final.pt
  modal volume get gibc-data tok/tok32k/tokenizer.json runs/tokenizer.json
  uv run python scripts/demo.py --ckpt runs/main_A/final.pt --tok runs/tokenizer.json \\
      --prompt "The capital of France is" --prompt "Photosynthesis is"
  uv run python scripts/demo.py --ckpt runs/main_A/final.pt --tok runs/tokenizer.json -i
"""
import argparse
import time

import torch
from tokenizers import Tokenizer

from gibc.configs import BOS_ID, EOS_ID
from gibc.hf_wrap import load_weights
from gibc.model import GPT


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ckpt: str, device: str) -> GPT:
    state, mc = load_weights(ckpt)
    model = GPT(mc)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def next_token(model: GPT, ids: list[int], temperature: float, top_k: int) -> int:
    """Sample one token from the last position; temperature 0 means greedy."""
    x = torch.tensor([ids[-model.cfg.ctx:]], device=next(model.parameters()).device)
    logits = model(x)[0, -1]
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return int(torch.multinomial(torch.softmax(logits, -1), 1))


def generate(model: GPT, tok: Tokenizer, prompt: str, max_new: int, temperature: float,
             top_k: int) -> str:
    """Stream the continuation to stdout; stop at BOS/EOS (end of document) or max_new."""
    ids = tok.encode(prompt).ids  # post-processor prepends BOS, as in training
    new, shown = [], ""
    print(prompt, end="", flush=True)
    for _ in range(max_new):
        t = next_token(model, ids + new, temperature, top_k)
        if t in (BOS_ID, EOS_ID):
            break
        new.append(t)
        text = tok.decode(new)
        print(text[len(shown):], end="", flush=True)  # decode whole tail: multi-byte safe
        shown = text
    print()
    return shown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="final.pt or snap/step_N.pt")
    ap.add_argument("--tok", required=True, help="tokenizer.json")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=pick_device())
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok = load_model(args.ckpt, args.device), Tokenizer.from_file(args.tok)
    sample = (args.max_new, args.temperature, args.top_k)
    for prompt in args.prompt:
        t0 = time.time()
        generate(model, tok, prompt, *sample)
        print(f"[{time.time() - t0:.1f}s]\n")
    while args.interactive:
        try:
            prompt = input(">>> ")
        except EOFError:
            break
        generate(model, tok, prompt, *sample)


if __name__ == "__main__":
    main()
