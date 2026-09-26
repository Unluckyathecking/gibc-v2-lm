"""Shared contracts: model/train configs, data sources, mixes, token ids, volume paths.

Every other module imports from here. Change a value here, not a copy of it elsewhere.
"""
from dataclasses import dataclass, field

# --- Modal volumes -----------------------------------------------------------
DATA_VOL, DATA_DIR = "gibc-data", "/data"   # tokenizers, shards, manifests
RUNS_VOL, RUNS_DIR = "gibc-runs", "/runs"   # checkpoints, logs, evals


def tokenizer_path(tok: str) -> str:
    return f"{DATA_DIR}/tok/{tok}/tokenizer.json"


def shard_dir(tok: str) -> str:
    return f"{DATA_DIR}/shards/{tok}"


def manifest_path(tok: str) -> str:
    return f"{shard_dir(tok)}/manifest.json"


def run_dir(run: str) -> str:
    return f"{RUNS_DIR}/{run}"


# --- Tokenizer ---------------------------------------------------------------
TOKENIZERS = {"tok32k": 32768, "tok16k": 16384}
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]  # ids are their list positions
BOS_ID, EOS_ID, PAD_ID = 0, 1, 2
# Token stream layout: [BOS] doc [BOS] doc ... BOS is the only document delimiter.

# --- Shard format (modded-nanogpt): 256 x int32 header, then uint16 tokens ----
SHARD_MAGIC, SHARD_VERSION, SHARD_HEADER_INTS = 20240520, 1, 256
SHARD_TOKENS = 100_000_000  # target tokens per shard file


# --- Data sources ------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    repo: str                     # HF dataset repo id
    pattern: str                  # parquet glob inside the repo (B2 verifies against the Hub)
    text_col: str = "text"
    min_int_score: int | None = None  # keep rows with int_score >= this


SOURCES = {
    "fwedu": Source("HuggingFaceFW/fineweb-edu", "sample/100BT/*.parquet"),
    "fwedu_hq": Source("HuggingFaceFW/fineweb-edu", "sample/100BT/*.parquet", min_int_score=4),
    "dclm": Source("mlfoundations/dclm-baseline-1.0-parquet", "filtered/**/*.parquet"),
    "finemath": Source("HuggingFaceTB/finemath", "finemath-4plus/*.parquet"),
    "finepdfs": Source("HuggingFaceFW/finepdfs", "data/eng_Latn/train/*.parquet"),
}
# fwedu and fwedu_hq must be built from disjoint parquet files, and val docs from
# files used by neither, so no document appears in two splits.

# Tokens to produce per source (train split), sized for a 15B-token main run:
# main phase 65% of tokens, decay 35% of which 87.5% is MIX_ANNEAL. Val: VAL_TOKENS_PER_SOURCE each.
TOKEN_TARGETS = {
    "tok32k": {"fwedu": 7_200_000_000, "dclm": 3_600_000_000, "finemath": 2_000_000_000,
               "finepdfs": 600_000_000, "fwedu_hq": 4_000_000_000},
    "tok16k": {"fwedu": 720_000_000, "dclm": 360_000_000, "finemath": 60_000_000,
               "finepdfs": 60_000_000},
}
VAL_TOKENS_PER_SOURCE = 5_000_000  # same val documents for both tokenizers

MIX_MAIN = {"fwedu": 0.60, "dclm": 0.30, "finemath": 0.05, "finepdfs": 0.05}
# Quality anneal used during the LR decay phase (12.5% general web kept).
MIX_ANNEAL = {"fwedu_hq": 0.65, "finemath": 0.225, "fwedu": 0.075, "dclm": 0.05}
VAL_MIX = MIX_MAIN  # val bpb is reported on the main-mix val docs


# --- Model -------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    d_model: int
    n_unique_blocks: int
    repeats: int            # each unique block applied `repeats` times in a row: b0,b0,b1,b1,...
    n_head: int
    n_kv_head: int
    mlp_hidden: int
    ctx: int = 1024
    softcap: float = 15.0
    rope_base: float = 10_000.0
    tokenizer: str = "tok32k"
    xsa: bool = False             # Exclusive Self Attention: remove each token's own value from its output
    value_residual: bool = False  # blend each layer's V with layer 1's V via a learned scalar per layer

    @property
    def n_eff(self) -> int:
        return self.n_unique_blocks * self.repeats

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# Param rule: tied embedding; per block Wq, Wk, Wv, Wo, W_up, W_down (no biases) plus
# two RMSNorm weights; QK-norm has no weight; n_eff // 2 U-net skip scalars; final norm.
MODEL_CONFIGS = {
    "A": ModelConfig(vocab=32768, d_model=512, n_unique_blocks=14, repeats=1,
                     n_head=8, n_kv_head=4, mlp_hidden=1536),            # 49,822,215
    "alt1": ModelConfig(vocab=32768, d_model=576, n_unique_blocks=8, repeats=2,
                        n_head=9, n_kv_head=3, mlp_hidden=2304),         # 47,195,720
    "alt2": ModelConfig(vocab=16384, d_model=512, n_unique_blocks=14, repeats=1,
                        n_head=8, n_kv_head=4, mlp_hidden=1536,
                        tokenizer="tok16k"),                             # 41,433,607 once / 49,822,215 twice
}
# Variants of A used by sweep arms; identical param count except +n_eff scalars for value_residual.
MODEL_CONFIGS["A_xsa"] = ModelConfig(**{**MODEL_CONFIGS["A"].__dict__, "xsa": True})
MODEL_CONFIGS["A_vres"] = ModelConfig(**{**MODEL_CONFIGS["A"].__dict__, "value_residual": True})
PARAM_CAP = 50_000_000


# --- Training ----------------------------------------------------------------
TOKENS_PER_STEP = 64 * 8 * 1024  # micro_bsz * grad_accum * ctx = 524,288


@dataclass(frozen=True)
class TrainConfig:
    model: str                               # key into MODEL_CONFIGS
    total_tokens: int
    mix_main: dict = field(default_factory=lambda: dict(MIX_MAIN))
    mix_anneal: dict | None = None           # if set, loader switches to it when decay starts
    opt: str = "muon"                        # "muon" (Muon + AdamW), "normuon" (NorMuon + AdamW) or "adamw"
    cautious_wd: bool = False                # decay only coords where update and weight agree in sign
    muon_lr: float = 0.02                    # torch.optim.Muon, adjust_lr_fn="original"
    muon_wd: float = 0.01
    adam_lr: float = 3e-3                    # embedding, norms, scalars (and matrices if opt="adamw")
    adam_betas: tuple = (0.9, 0.95)
    adam_wd: float = 0.0
    warmup: float = 0.01                     # fraction of steps
    decay: float = 0.35                      # final fraction of steps, linear to 0
    micro_bsz: int = 64
    grad_accum: int = 8
    ckpt_minutes: int = 30
    snap_every: int = 250                    # model-only snapshots during decay, for averaging
    val_every: int = 250
    val_tokens: int | None = None            # None: every val shard (same text for all tokenizers)
    seed: int = 1337

    @property
    def steps(self) -> int:
        return self.total_tokens // (self.micro_bsz * self.grad_accum * MODEL_CONFIGS[self.model].ctx)


SWEEP_TOKENS = 1_000_000_000
RUNS = {
    "sweep_A": TrainConfig("A", SWEEP_TOKENS),
    "sweep_A_lr2": TrainConfig("A", SWEEP_TOKENS, muon_lr=0.04),
    # opt="adamw" applies muon_wd to the block matrices; 0.1 is the standard AdamW LM value.
    "sweep_A_adamw": TrainConfig("A", SWEEP_TOKENS, opt="adamw", adam_lr=2e-3, muon_wd=0.1),
    "sweep_alt2": TrainConfig("alt2", SWEEP_TOKENS),
    "sweep_alt1": TrainConfig("alt1", SWEEP_TOKENS),
    "sweep_A_anneal": TrainConfig("A", SWEEP_TOKENS, mix_anneal=dict(MIX_ANNEAL)),
    # Parameter-free additions, each tested alone against sweep_A.
    "sweep_A_xsa": TrainConfig("A_xsa", SWEEP_TOKENS),
    "sweep_A_vres": TrainConfig("A_vres", SWEEP_TOKENS),
    "sweep_A_normuon": TrainConfig("A", SWEEP_TOKENS, opt="normuon", cautious_wd=True),
    # Main runs are added after the sweep picks a winner and runner-up.
}
SWEEP_RUNS = [name for name in RUNS if name.startswith("sweep_")]

# --- Evaluation --------------------------------------------------------------
LM_EVAL_TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1", "test")
