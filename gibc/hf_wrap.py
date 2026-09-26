"""Hugging Face wrapper around gibc.model.GPT so lm-eval's HFLM can drive it.

The wrapper adds no computation: logits are exactly GPT(idx) (fp32, softcapped).
"""
from dataclasses import asdict

import torch
from tokenizers import Tokenizer
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutput

from gibc.configs import ModelConfig
from gibc.model import GPT


class GibcConfig(PretrainedConfig):
    model_type = "gibc"

    def __init__(self, model_cfg: dict | None = None, **kwargs):
        self.model_cfg = model_cfg or {}  # asdict(ModelConfig); empty only for HF's default-config diffing
        self.max_position_embeddings = self.model_cfg.get("ctx", 1024)
        self.vocab_size = self.model_cfg.get("vocab")
        kwargs.setdefault("tie_word_embeddings", False)  # GPT reuses wte.weight itself; nothing for HF to tie
        super().__init__(**kwargs)


class GibcForCausalLM(PreTrainedModel):
    config_class = GibcConfig
    base_model_prefix = "gpt"
    _no_split_modules = ["Block"]

    def __init__(self, config: GibcConfig):
        super().__init__(config)
        self.gpt = GPT(ModelConfig(**config.model_cfg))
        self.post_init()

    def _init_weights(self, module):
        pass  # GPT initialises itself, and real weights always come from a checkpoint

    def get_input_embeddings(self):
        return self.gpt.wte

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs) -> CausalLMOutput:
        # attention_mask is ignored: attention is causal and lm-eval right-pads loglikelihood
        # batches, so padding never reaches a scored position. RoPE tables stop at ctx.
        if input_ids.size(1) > self.gpt.cfg.ctx:
            raise ValueError(f"sequence length {input_ids.size(1)} > ctx {self.gpt.cfg.ctx}")
        return CausalLMOutput(logits=self.gpt(input_ids))


def load_tokenizer(tok_path: str) -> PreTrainedTokenizerFast:
    """HF tokenizer whose post-processor prepends <|bos|> (add_special_tokens=True)."""
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(tok_path),
        bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
    )


def load_weights(path: str) -> tuple[dict, ModelConfig]:
    """(state_dict, ModelConfig) from a train.py weights file: final.pt or snap/step_N.pt."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["model"], ModelConfig(**obj["model_cfg"])


def load_hf(state_dict_or_path, model_cfg: ModelConfig, tok_path: str, device: str = "cpu"):
    """-> (GibcForCausalLM in eval mode on `device`, tokenizer)."""
    state = state_dict_or_path
    if not isinstance(state, dict):
        state, _ = load_weights(state)
    model = GibcForCausalLM(GibcConfig(model_cfg=asdict(model_cfg)))
    model.gpt.load_state_dict(state)
    return model.to(device).eval(), load_tokenizer(tok_path)
