"""
Shared fixtures for both test stages.

Stage 1 (no SLM): uses _MockModel + proxy_tokenizer
Stage 2 (integration): same fixtures — no Gemma needed yet
"""

import pytest
import torch
from transformers import AutoTokenizer
from model.loader import ModelBundle

# GPT-2 is the lightest tokenizer available — ~5MB, no model download needed.
# It's only a stand-in for Gemma 4; the tests here care about whether the
# compilation pipeline runs, not about the specific vocabulary.
PROXY_TOKENIZER_ID = "gpt2"


class _MockModel:
    """
    Minimal model stub — no download, no VRAM.

    Only implements the three things ModelBundle.from_existing() actually
    calls: eval(), parameters(), and config.vocab_size.
    """

    def __init__(self, vocab_size: int):
        self.config = type("Config", (), {"vocab_size": vocab_size})()

    def eval(self):
        return self

    def parameters(self):
        # Generator so each call produces a fresh tensor.
        # next(model.parameters()).device → cpu
        yield torch.zeros(1)


@pytest.fixture(scope="session")
def proxy_tokenizer():
    """GPT-2 tokenizer. Downloaded once per test session."""
    tok = AutoTokenizer.from_pretrained(PROXY_TOKENIZER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="session")
def stub_bundle(proxy_tokenizer):
    """
    A real ModelBundle built from a mock model.
    Tests can use this wherever a ModelBundle is expected,
    without any LLM in memory.
    """
    mock = _MockModel(vocab_size=proxy_tokenizer.vocab_size)
    return ModelBundle.from_existing(mock, proxy_tokenizer)


@pytest.fixture
def minimal_ebnf(tmp_path):
    """
    A tiny valid EBNF file for testing compilation.
    Used so tests don't depend on triton.ebnf existing yet.
    """
    path = tmp_path / "test.ebnf"
    path.write_text('root ::= "@triton.jit"\n', encoding="utf-8")
    return path