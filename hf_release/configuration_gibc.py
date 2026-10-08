"""Config for gibc-v2-lm (Unluckyexe/gibc-v2-lm), loaded with trust_remote_code=True."""
from transformers import PretrainedConfig


class GibcConfig(PretrainedConfig):
    model_type = "gibc"

    def __init__(self, vocab_size=32768, hidden_size=512, num_hidden_layers=14,
                 num_attention_heads=8, num_key_value_heads=4, intermediate_size=1536,
                 max_position_embeddings=1024, logit_softcap=15.0, rope_theta=10_000.0,
                 value_residual=True, rms_norm_eps=1e-6, bos_token_id=0, eos_token_id=1,
                 pad_token_id=2, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.logit_softcap = logit_softcap
        self.rope_theta = rope_theta
        self.value_residual = value_residual
        self.rms_norm_eps = rms_norm_eps
        # The output head is wte.weight used directly, so there is nothing for HF to tie.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id, **kwargs)
