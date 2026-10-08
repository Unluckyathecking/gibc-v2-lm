"""Build the Hugging Face release folder from a train.py weights file.

  uv run python scripts/build_hf_release.py --weights final.pt --tokenizer tokenizer.json --out hf_release

Writes config.json (with auto_map), model.safetensors, tokenizer files and a copy of the .pt
into --out, next to the committed configuration_gibc.py / modeling_gibc.py / README.md, then
reloads the folder with AutoModelForCausalLM and checks its logits match gibc.model.GPT.
"""
import argparse
import json
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FILES = ("configuration_gibc.py", "modeling_gibc.py")
SPECIAL = {"bos_token": "<|bos|>", "eos_token": "<|eos|>", "pad_token": "<|pad|>"}


def hf_config_kwargs(mc: dict) -> dict:
    assert mc["repeats"] == 1 and not mc["xsa"], "release code supports repeats=1, xsa=False only"
    return dict(vocab_size=mc["vocab"], hidden_size=mc["d_model"], num_hidden_layers=mc["n_unique_blocks"],
                num_attention_heads=mc["n_head"], num_key_value_heads=mc["n_kv_head"],
                intermediate_size=mc["mlp_hidden"], max_position_embeddings=mc["ctx"],
                logit_softcap=mc["softcap"], rope_theta=mc["rope_base"], value_residual=mc["value_residual"])


def write_tokenizer(tok_json: str, out: str):
    shutil.copy(tok_json, os.path.join(out, "tokenizer.json"))
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024,
           "add_bos_token": True, "clean_up_tokenization_spaces": False, **SPECIAL}
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "special_tokens_map.json"), "w") as f:
        json.dump(SPECIAL, f, indent=2)


def build(weights: str, tok_json: str, out: str, copy_pt: bool = True) -> None:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    # The release's own modeling code (as trust_remote_code loads it), not gibc's.
    GibcConfig = get_class_from_dynamic_module("configuration_gibc.GibcConfig", out)
    GibcForCausalLM = get_class_from_dynamic_module("modeling_gibc.GibcForCausalLM", out)

    obj = torch.load(weights, map_location="cpu", weights_only=True)
    cfg = GibcConfig(**hf_config_kwargs(obj["model_cfg"]))
    cfg.auto_map = {"AutoConfig": "configuration_gibc.GibcConfig",
                    "AutoModelForCausalLM": "modeling_gibc.GibcForCausalLM"}
    cfg.architectures = ["GibcForCausalLM"]
    model = GibcForCausalLM(cfg)
    model.gpt.load_state_dict(obj["model"], strict=True)
    model.save_pretrained(out, safe_serialization=True)
    write_tokenizer(tok_json, out)
    if copy_pt:
        shutil.copy(weights, os.path.join(out, "final.pt"))
    check(obj, out)


@torch.no_grad()
def check(obj: dict, out: str) -> None:
    """Reload through AutoModelForCausalLM and compare logits with gibc.model.GPT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, ROOT)
    from gibc.configs import ModelConfig
    from gibc.model import GPT

    ref = GPT(ModelConfig(**obj["model_cfg"])).eval()
    ref.load_state_dict(obj["model"])
    hf = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    assert ids[0, 0].item() == 0, "tokenizer should prepend <|bos|>"
    x = torch.cat([ids, torch.randint(3, obj["model_cfg"]["vocab"], (1, 200))], dim=1)
    diff = (hf(x).logits - ref(x)).abs().max().item()
    n = sum(p.numel() for p in hf.parameters())
    print(f"params {n:,}  max |logit diff| vs gibc.model.GPT: {diff:.2e}")
    assert diff < 1e-4, diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "hf_release"))
    ap.add_argument("--no-pt", action="store_true", help="skip copying the .pt next to the safetensors")
    a = ap.parse_args()
    build(a.weights, a.tokenizer, a.out, copy_pt=not a.no_pt)
