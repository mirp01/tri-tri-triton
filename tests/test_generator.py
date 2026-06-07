"""
Tests for model/generator.py

No GPU or real model needed. model.generate() is mocked to return
predictable token IDs so we can test what generate() does with them.

The two most important tests are:
  - test_strips_prompt_tokens:   verifies decode() only gets new tokens
  - test_new_matcher_per_call:   verifies stateful matchers aren't reused

Run with:
  pytest tests/test_generator.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import LogitsProcessorList

from model.generator import generate
import config

# ── Test data ─────────────────────────────────────────────────────────────────

PROMPT_LEN    = 5
PROMPT_IDS    = [1, 2, 3, 4, 5]
NEW_TOKEN_IDS = [10, 11, 12]
DECODED_KERNEL = "@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):\n    pass"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_compiled_grammar():
    return MagicMock()


@pytest.fixture
def mock_bundle():
    """
    ModelBundle stub with a tokenizer and model that return
    predictable tensors — no download, no VRAM.
    """
    bundle = MagicMock()
    bundle.device = torch.device("cpu")
    bundle.tokenizer.pad_token_id = 0

    # tokenizer(prompt, ...).to(device) → inputs dict
    prompt_tensor = torch.tensor([PROMPT_IDS])       # shape (1, PROMPT_LEN)
    inputs_dict = {
        "input_ids":      prompt_tensor,
        "attention_mask": torch.ones(1, PROMPT_LEN, dtype=torch.long),
    }
    tokenizer_output = MagicMock()
    tokenizer_output.to.return_value = inputs_dict
    bundle.tokenizer.return_value = tokenizer_output

    # model.generate(**inputs, ...) → [prompt tokens + new tokens]
    all_token_ids = torch.tensor([PROMPT_IDS + NEW_TOKEN_IDS])  # shape (1, 8)
    bundle.model.generate.return_value = all_token_ids

    # tokenizer.decode(new_tokens, ...) → kernel string
    bundle.tokenizer.decode.return_value = DECODED_KERNEL

    return bundle


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGenerate:

    # Patch XGrammar components in every test — they're not the subject here.
    # GrammarMatcher and LogitsProcessor are tested implicitly via call assertions.
    _xgr_patches = [
        "model.generator.xgr.GrammarMatcher",
        "model.generator.xgr.contrib.hf.LogitsProcessor",
    ]

    def _run(self, mock_bundle, mock_grammar, **kwargs):
        """Helper: run generate() with both XGrammar classes patched."""
        with patch(self._xgr_patches[0]), patch(self._xgr_patches[1]):
            return generate("test prompt", mock_bundle, mock_grammar, **kwargs)

    # ── Core output ───────────────────────────────────────────────────────────

    def test_returns_string(self, mock_bundle, mock_compiled_grammar):
        assert isinstance(self._run(mock_bundle, mock_compiled_grammar), str)

    def test_returns_decoded_kernel(self, mock_bundle, mock_compiled_grammar):
        result = self._run(mock_bundle, mock_compiled_grammar)
        assert result == DECODED_KERNEL

    def test_strips_prompt_tokens(self, mock_bundle, mock_compiled_grammar):
        """
        The most important correctness test.

        output_ids contains [prompt + new tokens]. Only the new tokens
        should be passed to decode(). If the prompt leaks through,
        validation/pipeline.py receives the system prompt + kernel
        instead of just the kernel.
        """
        self._run(mock_bundle, mock_compiled_grammar)

        decoded_tensor = mock_bundle.tokenizer.decode.call_args.args[0]
        assert decoded_tensor.tolist() == NEW_TOKEN_IDS  # [10, 11, 12], not [1..5, 10..12]

    def test_skip_special_tokens(self, mock_bundle, mock_compiled_grammar):
        """EOS tokens must be stripped so they don't appear in the kernel string."""
        self._run(mock_bundle, mock_compiled_grammar)

        decode_kwargs = mock_bundle.tokenizer.decode.call_args.kwargs
        assert decode_kwargs.get("skip_special_tokens") is True

    # ── XGrammar wiring ───────────────────────────────────────────────────────

    def test_new_matcher_per_call(self, mock_bundle, mock_compiled_grammar):
        """
        GrammarMatcher is stateful — it advances through the grammar's
        pushdown automaton with each token. Reusing a matcher across calls
        would silently break constrained decoding from the second call onwards.
        """
        with patch(self._xgr_patches[0]) as MockMatcher, \
             patch(self._xgr_patches[1]):
            generate("prompt A", mock_bundle, mock_compiled_grammar)
            generate("prompt B", mock_bundle, mock_compiled_grammar)

        assert MockMatcher.call_count == 2

    def test_matcher_receives_compiled_grammar(self, mock_bundle, mock_compiled_grammar):
        """GrammarMatcher must be initialised with the CompiledGrammar object."""
        with patch(self._xgr_patches[0]) as MockMatcher, \
             patch(self._xgr_patches[1]):
            generate("test prompt", mock_bundle, mock_compiled_grammar)

        MockMatcher.assert_called_once_with(mock_compiled_grammar)

    def test_logits_processor_passed_to_generate(self, mock_bundle, mock_compiled_grammar):
        """
        If logits_processor isn't in the generate() call, XGrammar never runs
        and the output is completely unconstrained.
        """
        self._run(mock_bundle, mock_compiled_grammar)

        gen_kwargs = mock_bundle.model.generate.call_args.kwargs
        assert "logits_processor" in gen_kwargs
        assert isinstance(gen_kwargs["logits_processor"], LogitsProcessorList)

    def test_pad_token_id_passed_to_generate(self, mock_bundle, mock_compiled_grammar):
        """Missing pad_token_id causes silent generation errors with Gemma."""
        self._run(mock_bundle, mock_compiled_grammar)

        gen_kwargs = mock_bundle.model.generate.call_args.kwargs
        assert gen_kwargs["pad_token_id"] == mock_bundle.tokenizer.pad_token_id

    # ── Configuration ─────────────────────────────────────────────────────────

    def test_config_defaults_forwarded_to_generate(self, mock_bundle, mock_compiled_grammar):
        """All generation settings from config.py must reach model.generate()."""
        self._run(mock_bundle, mock_compiled_grammar)

        gen_kwargs = mock_bundle.model.generate.call_args.kwargs
        assert gen_kwargs["max_new_tokens"] == config.MAX_NEW_TOKENS
        assert gen_kwargs["temperature"]    == config.TEMPERATURE
        assert gen_kwargs["top_p"]          == config.TOP_P
        assert gen_kwargs["do_sample"]      == config.DO_SAMPLE

    def test_kwargs_override_config_defaults(self, mock_bundle, mock_compiled_grammar):
        """Callers can override config defaults — useful for greedy debug runs."""
        self._run(mock_bundle, mock_compiled_grammar,
                  temperature=0.0, do_sample=False, max_new_tokens=16)

        gen_kwargs = mock_bundle.model.generate.call_args.kwargs
        assert gen_kwargs["temperature"]    == 0.0
        assert gen_kwargs["do_sample"]      is False
        assert gen_kwargs["max_new_tokens"] == 16

    # ── Memory efficiency ─────────────────────────────────────────────────────

    def test_runs_under_no_grad(self, mock_bundle, mock_compiled_grammar):
        """
        torch.no_grad() must be active during generate().
        Without it, PyTorch builds the full computation graph and
        doubles VRAM usage — fatal for a 12B+ model.
        """
        grad_was_enabled = []

        def capture_grad_state(**kwargs):
            grad_was_enabled.append(torch.is_grad_enabled())
            input_ids = kwargs["input_ids"]
            return torch.cat(
                [input_ids[0], torch.tensor(NEW_TOKEN_IDS)]
            ).unsqueeze(0)

        mock_bundle.model.generate.side_effect = capture_grad_state

        with patch(self._xgr_patches[0]), patch(self._xgr_patches[1]):
            generate("test prompt", mock_bundle, mock_compiled_grammar)

        assert grad_was_enabled == [False]