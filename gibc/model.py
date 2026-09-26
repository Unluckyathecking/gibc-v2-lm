"""GPT with GQA, RoPE, QK-norm, relu^2 MLP, tied embedding, weight sharing, U-net skips.

Effective layer order with repeats=2 is b0,b0,b1,b1,... Skip scalars are indexed by
effective depth: layer i's output is added to layer n_eff-1-i's input.

Opt-in (both off by default; off leaves the state_dict and outputs unchanged):
- cfg.xsa: Exclusive Self Attention (arXiv 2603.09078) with the learnable gate of
  modded-nanogpt record #82 (PR #264): per effective layer and query head,
  y <- y - tanh(a) * (y . v_hat) v_hat, v_hat = v / ||v|| of the token's own value from
  the KV head that query head reads. a is zero-init, so init is the baseline.
- cfg.value_residual: ResFormer value residual (arXiv 2410.17897). Effective layer 0's
  V (after Wv; V has no RoPE) is kept; each later effective layer l attends with
  lam_l * v_l + (1 - lam_l) * v_0, lam_l init 0.5. Layer 0 is unchanged: n_eff - 1 scalars.
Both sets of scalars live on GPT (not in blocks), 1-D, so optim.split_params sends them to AdamW.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gibc.configs import ModelConfig

# Tied head: the input token's own logit at init is ~ d_model * EMBED_STD (the final
# RMSNorm rescales the embedding to unit rms, then dots it with itself). 0.005 keeps
# it <= ~3 for d<=576, so initial loss ~ ln(V); Adam grows the embedding quickly.
EMBED_STD = 0.005
LINEAR_STD = 0.02
NORM_EPS = 1e-6  # explicit: the bf16 default eps (7.8e-3) would swamp small activations
SKIP_INIT = 1.0  # as in modded-nanogpt; zero-init Wo/W_down already make init a no-op stack
XSA_EPS = 1e-4   # F.normalize eps on v, as in modded-nanogpt record #82
VRES_INIT = 0.5  # value-residual mix: equal weight on own V and layer-0 V


def rope_tables(ctx: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [ctx, head_dim // 2] in fp32."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(ctx, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, hd]; rotate the two halves of the head dim."""
    T = x.size(1)
    cos = cos[:T, None, :].to(x.dtype)
    sin = sin[:T, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        kv_dim = cfg.n_kv_head * cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, v0=None, vres_lam=None, xsa_alpha=None):
        """Returns (output, this layer's raw V [B, T, n_kv_head, hd]).

        v0/vres_lam: value residual (v0 is layer 0's raw V, vres_lam a 0-D scalar).
        xsa_alpha: [n_head] XSA gate logits for this effective layer."""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v_raw = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        v = v_raw
        if vres_lam is not None:
            lam = vres_lam.type_as(v)
            v = lam * v + (1 - lam) * v0
        # QK-norm without learnable weight, then RoPE.
        q = apply_rope(F.rms_norm(q, (self.head_dim,), eps=NORM_EPS), cos, sin)
        k = apply_rope(F.rms_norm(k, (self.head_dim,), eps=NORM_EPS), cos, sin)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v, is_causal=True, enable_gqa=True,
        )
        if xsa_alpha is not None:
            y = self._exclude_self(y, v, xsa_alpha)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1)), v_raw

    def _exclude_self(self, y, v, alpha):
        """y: [B, H, T, hd]; v: [B, H_kv, T, hd]. Query head h reads KV head h // group
        (the enable_gqa layout). Dot and normalisation in fp32."""
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        vn = F.normalize(v.float(), dim=-1, eps=XSA_EPS)
        yf = y.float()
        proj = (yf * vn).sum(dim=-1, keepdim=True)
        gate = torch.tanh(alpha.float()).view(1, -1, 1, 1)
        return (yf - gate * proj * vn).type_as(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.relu(self.w_up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=NORM_EPS)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=NORM_EPS)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, v0=None, vres_lam=None, xsa_alpha=None):
        """Returns (x, this layer's raw V) — the V is only used by value residual."""
        a, v = self.attn(self.norm1(x), cos, sin, v0, vres_lam, xsa_alpha)
        x = x + a
        return x + self.mlp(self.norm2(x)), v


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)  # input embedding and output head
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_unique_blocks))
        self.skip_w = nn.Parameter(torch.full((cfg.n_eff // 2,), SKIP_INIT))
        if cfg.xsa:  # flat [n_eff * n_head] so it stays 1-D; row l is effective layer l
            self.xsa_alpha = nn.Parameter(torch.zeros(cfg.n_eff * cfg.n_head))
        if cfg.value_residual:  # entry l-1 is effective layer l (layer 0 has none)
            self.vres_lambda = nn.Parameter(torch.full((cfg.n_eff - 1,), VRES_INIT))
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=NORM_EPS)
        cos, sin = rope_tables(cfg.ctx, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.wte.weight, std=EMBED_STD)
        for block in self.blocks:
            for lin in (block.attn.wq, block.attn.wk, block.attn.wv, block.mlp.w_up):
                nn.init.normal_(lin.weight, std=LINEAR_STD)
            nn.init.zeros_(block.attn.wo.weight)
            nn.init.zeros_(block.mlp.w_down.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        n_eff, n_skip = self.cfg.n_eff, self.skip_w.numel()
        stack, v0 = [], None
        xsa = self.xsa_alpha.view(n_eff, -1) if self.cfg.xsa else None
        for layer in range(n_eff):
            if layer >= n_eff - n_skip:  # decoder half: pair with layer n_eff-1-layer
                x = x + self.skip_w[layer - (n_eff - n_skip)] * stack.pop()
            lam = self.vres_lambda[layer - 1] if self.cfg.value_residual and layer > 0 else None
            x, v = self.blocks[layer // self.cfg.repeats](
                x, self.rope_cos, self.rope_sin, v0, lam, None if xsa is None else xsa[layer])
            if layer == 0:
                v0 = v
            if layer < n_skip:  # encoder half
                stack.append(x)
        logits = self.norm_f(x) @ self.wte.weight.T
        logits = self.cfg.softcap * torch.tanh(logits.float() / self.cfg.softcap)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


def count_params(cfg: ModelConfig) -> dict[str, int]:
    """Unique parameter counts; "twice" counts the tied embedding as input + head."""
    with torch.device("meta"):
        model = GPT(cfg)
    once = sum(p.numel() for p in model.parameters())
    embedding = model.wte.weight.numel()
    return {"once": once, "twice": once + embedding,
            "embedding": embedding, "non_embedding": once - embedding}
