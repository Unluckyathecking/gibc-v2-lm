"""gibc-v2-lm: 14-layer GPT (GQA, RoPE, QK-norm, ReLU^2 MLP, tied embedding, U-net skips,
ResFormer value residual, logit softcap). Self-contained copy of gibc/model.py from
https://github.com/Unluckyathecking/gibc-v2-lm, with identical parameter names and maths.

Training documents always start with <|bos|> (id 0); prepend it for best results.
attention_mask is ignored: attention is causal, so right padding never reaches a real token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_gibc import GibcConfig


def rope_tables(ctx, head_dim, base, device):
    """fp32 cos/sin [ctx, head_dim // 2], computed on CPU as in training, then moved."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(ctx, dtype=torch.float32), inv_freq)
    return angles.cos().to(device), angles.sin().to(device)


def apply_rope(x, cos, sin):
    """x: [B, T, H, hd]; rotate the two halves of the head dim."""
    T = x.size(1)
    cos, sin = cos[:T, None, :].to(x.dtype), sin[:T, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GibcConfig):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim, self.eps = cfg.hidden_size // cfg.num_attention_heads, cfg.rms_norm_eps
        kv_dim = self.n_kv_head * self.head_dim
        self.wq = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, v0=None, vres_lam=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v_raw = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        v = v_raw
        if vres_lam is not None:
            lam = vres_lam.type_as(v)
            v = lam * v + (1 - lam) * v0
        q = apply_rope(F.rms_norm(q, (self.head_dim,), eps=self.eps), cos, sin)
        k = apply_rope(F.rms_norm(k, (self.head_dim,), eps=self.eps), cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1)), v_raw


class MLP(nn.Module):
    def __init__(self, cfg: GibcConfig):
        super().__init__()
        self.w_up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w_down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w_down(F.relu(self.w_up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: GibcConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, v0=None, vres_lam=None):
        a, v = self.attn(self.norm1(x), cos, sin, v0, vres_lam)
        x = x + a
        return x + self.mlp(self.norm2(x)), v


class GPT(nn.Module):
    def __init__(self, cfg: GibcConfig):
        super().__init__()
        self.cfg, n = cfg, cfg.num_hidden_layers
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)  # input embedding and output head
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(n))
        self.skip_w = nn.Parameter(torch.ones(n // 2))
        if cfg.value_residual:
            self.vres_lambda = nn.Parameter(torch.full((n - 1,), 0.5))
        self.norm_f = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, idx):
        cfg, n = self.cfg, self.cfg.num_hidden_layers
        cos, sin = rope_tables(cfg.max_position_embeddings, cfg.hidden_size // cfg.num_attention_heads,
                               cfg.rope_theta, idx.device)
        x, n_skip = self.wte(idx), self.skip_w.numel()
        stack, v0 = [], None
        for layer in range(n):
            if layer >= n - n_skip:  # decoder half: add the output of layer n-1-layer
                x = x + self.skip_w[layer - (n - n_skip)] * stack.pop()
            lam = self.vres_lambda[layer - 1] if cfg.value_residual and layer > 0 else None
            x, v = self.blocks[layer](x, cos, sin, v0, lam)
            if layer == 0:
                v0 = v
            if layer < n_skip:
                stack.append(x)
        logits = self.norm_f(x) @ self.wte.weight.T
        return cfg.logit_softcap * torch.tanh(logits.float() / cfg.logit_softcap)


class GibcForCausalLM(PreTrainedModel):
    config_class = GibcConfig
    base_model_prefix = "gpt"
    _no_split_modules = ["Block"]

    def __init__(self, config: GibcConfig):
        super().__init__(config)
        self.gpt = GPT(config)
        self.post_init()

    def _init_weights(self, module):
        pass  # weights always come from the checkpoint

    def get_input_embeddings(self):
        return self.gpt.wte

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {input_ids.size(1)} > "
                             f"{self.config.max_position_embeddings}")
        logits = self.gpt(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)
