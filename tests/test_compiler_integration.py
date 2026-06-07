"""
Stage 2 tests — ModelBundle integration, still no Gemma needed.

These tests verify that compiler.py and loader.py work together correctly.
They use the same stub_bundle from conftest.py — no LLM download required.

The last test (test_compiled_grammar_creates_matcher) is the most important:
it exercises the exact thing generator.py will do, so passing it means
the compiler output is ready to plug into generation.

Run with:
  pytest tests/test_compiler_integration.py -v
"""

import pytest
import torch
import xgrammar as xgr
from model.loader import ModelBundle
from grammar.compiler import load_compiled_grammar


# ── ModelBundle.from_existing() behaviour ───────────────────────────────────

class TestModelBundle:
    def test_wraps_tokenizer_by_reference(self, stub_bundle, proxy_tokenizer):
        """from_existing() should not copy the tokenizer — same object."""
        assert stub_bundle.tokenizer is proxy_tokenizer

    def test_vocab_size_comes_from_config(self, stub_bundle, proxy_tokenizer):
        """
        vocab_size must come from model.config, not len(tokenizer).
        For Gemma 4 these can differ due to special tokens — this test
        catches that mismatch early.
        """
        assert stub_bundle.vocab_size == proxy_tokenizer.vocab_size

    def test_pad_token_is_always_set(self, stub_bundle):
        """
        Generation silently produces bad output if pad_token is None.
        from_existing() must set it even if the tokenizer didn't ship with one.
        """
        assert stub_bundle.tokenizer.pad_token is not None
        assert stub_bundle.tokenizer.pad_token_id is not None

    def test_device_is_inferred(self, stub_bundle):
        """device is detected automatically from the model's parameters."""
        assert isinstance(stub_bundle.device, torch.device)


# ── Compiler + ModelBundle chain ─────────────────────────────────────────────

class TestCompilerWithBundle:
    def test_produces_compiled_grammar(self, stub_bundle, minimal_ebnf):
        """Full chain: ModelBundle → TokenizerInfo → CompiledGrammar."""
        compiled = load_compiled_grammar(stub_bundle, grammar_path=minimal_ebnf)
        assert isinstance(compiled, xgr.CompiledGrammar)

    def test_compiled_grammar_creates_matcher(self, stub_bundle, minimal_ebnf):
        """
        The compiled grammar can be used to create a GrammarMatcher.

        This is the handshake test between compiler.py and generator.py:
        generator.py will do exactly this at the start of every generation call.
        If this passes, the compiler output is ready for the generation pipeline.
        """
        compiled = load_compiled_grammar(stub_bundle, grammar_path=minimal_ebnf)
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher is not None

    def test_compiling_same_grammar_twice_is_consistent(self, stub_bundle, minimal_ebnf):
        """
        Two calls with the same inputs should both succeed.
        Catches any accidental stateful behaviour in the compiler.
        """
        compiled_a = load_compiled_grammar(stub_bundle, grammar_path=minimal_ebnf)
        compiled_b = load_compiled_grammar(stub_bundle, grammar_path=minimal_ebnf)
        assert isinstance(compiled_a, xgr.CompiledGrammar)
        assert isinstance(compiled_b, xgr.CompiledGrammar)