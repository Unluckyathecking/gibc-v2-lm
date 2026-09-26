"""HF wrapper: identical logits to the raw GPT, and lm-eval sees BOS-prefixed requests."""
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from gibc.configs import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, ModelConfig
from gibc.hf_wrap import load_hf, load_weights
from gibc.model import GPT

TINY = ModelConfig(vocab=256, d_model=32, n_unique_blocks=2, repeats=1, n_head=4,
                   n_kv_head=2, mlp_hidden=64, ctx=64)
TEXT = ["the cat sat on the mat.", "hi there, how are you?", "a quick brown fox jumps.",
        "language models predict the next token."]


def make_tokenizer(path) -> str:
    """Tiny byte-level BPE with the project's specials and BOS post-processor."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(TEXT * 20, trainers.BpeTrainer(vocab_size=TINY.vocab,
                                                           special_tokens=SPECIAL_TOKENS))
    tok.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A", special_tokens=[("<|bos|>", BOS_ID)])
    assert tok.get_vocab_size() <= TINY.vocab
    tok.save(str(path))
    return str(path)


def make_trained(tmp_path, seed: int = 0):
    """(raw GPT with non-trivial weights, hf model, tokenizer)."""
    torch.manual_seed(seed)
    gpt = GPT(TINY)
    with torch.no_grad():
        for p in gpt.parameters():  # Wo/W_down init at zero; randomise so every layer matters
            p.add_(torch.randn_like(p) * 0.05)
    hf, tok = load_hf(gpt.state_dict(), TINY, make_tokenizer(tmp_path / "tok.json"))
    return gpt.eval(), hf, tok


def test_logits_match_raw_gpt(tmp_path):
    gpt, hf, _ = make_trained(tmp_path)
    x = torch.randint(0, TINY.vocab, (3, 17))
    with torch.no_grad():
        out = hf(x).logits
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, gpt(x), atol=1e-5, rtol=0)


def test_rejects_sequences_longer_than_ctx(tmp_path):
    _, hf, _ = make_trained(tmp_path)
    with pytest.raises(ValueError):
        hf(torch.zeros(1, TINY.ctx + 1, dtype=torch.long))


def test_load_from_train_weights_file(tmp_path):
    gpt, _, _ = make_trained(tmp_path)
    path = tmp_path / "final.pt"
    torch.save({"model": gpt.state_dict(), "model_cfg": asdict(TINY)}, path)
    state, mc = load_weights(str(path))
    assert mc == TINY
    hf, _ = load_hf(str(path), mc, make_tokenizer(tmp_path / "tok2.json"))
    x = torch.randint(0, TINY.vocab, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(hf(x).logits, gpt(x), atol=1e-5, rtol=0)


def test_tokenizer_specials(tmp_path):
    _, _, tok = make_trained(tmp_path)
    assert (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) == (BOS_ID, EOS_ID, PAD_ID)
    assert tok("hi")["input_ids"][0] == BOS_ID
    assert BOS_ID not in tok.encode("hi", add_special_tokens=False)


def test_lm_eval_requests_start_with_bos(tmp_path):
    from gibc.evaluate import make_hflm
    _, hf, tok = make_trained(tmp_path)
    lm = make_hflm(hf, tok, batch_size=4)
    assert lm.max_length == TINY.ctx
    assert lm.prefix_token_id == BOS_ID
    assert lm.tok_encode("hi")[0] == BOS_ID
    # loglikelihood requests: context gets BOS once, continuation gets none.
    ctx_ids, cont_ids = lm._encode_pair("the cat sat on", " the mat.")
    assert ctx_ids[0] == BOS_ID and ctx_ids.count(BOS_ID) == 1
    assert BOS_ID not in cont_ids
    assert ctx_ids + cont_ids == lm.tok_encode("the cat sat on the mat.")
