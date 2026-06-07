"""
Stage 1 tests — no SLM needed.

What's covered:
  - triton.ebnf exists and is readable
  - XGrammar can parse the EBNF without errors
  - load_compiled_grammar() produces a CompiledGrammar
  - Error handling works (missing file, bad EBNF)

Run with:
  pytest tests/test_grammar.py -v
"""

import pytest
import xgrammar as xgr
from grammar.compiler import load_compiled_grammar, GRAMMAR_PATH


# ── Grammar file ────────────────────────────────────────────────────────────

class TestEbnfFile:
    def test_file_exists(self):
        """triton.ebnf must exist — nothing works without it."""
        assert GRAMMAR_PATH.exists(), (
            f"\ngrammar/triton.ebnf not found at {GRAMMAR_PATH}.\n"
            "Create it before running these tests."
        )

    def test_file_is_not_empty(self):
        content = GRAMMAR_PATH.read_text(encoding="utf-8")
        assert content.strip(), "triton.ebnf is empty"

    def test_file_parses_with_xgrammar(self):
        """XGrammar can parse the EBNF without raising."""
        ebnf = GRAMMAR_PATH.read_text(encoding="utf-8")
        grammar = xgr.Grammar.from_ebnf(ebnf)
        assert grammar is not None


# ── Compilation pipeline (uses stub_bundle from conftest) ───────────────────

class TestCompiler:
    def test_returns_compiled_grammar(self, stub_bundle, minimal_ebnf):
        """Happy path: valid EBNF + valid bundle → CompiledGrammar."""
        compiled = load_compiled_grammar(stub_bundle, grammar_path=minimal_ebnf)
        assert isinstance(compiled, xgr.CompiledGrammar)

    def test_raises_on_missing_file(self, stub_bundle, tmp_path):
        """FileNotFoundError with a clear message if the file doesn't exist."""
        missing = tmp_path / "nonexistent.ebnf"
        with pytest.raises(FileNotFoundError, match="Grammar file not found"):
            load_compiled_grammar(stub_bundle, grammar_path=missing)

    def test_raises_on_malformed_ebnf(self, stub_bundle, tmp_path):
        """ValueError is raised if the EBNF content can't be parsed."""
        bad = tmp_path / "bad.ebnf"
        bad.write_text("this is not valid EBNF !!!@@@###", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_compiled_grammar(stub_bundle, grammar_path=bad)
            