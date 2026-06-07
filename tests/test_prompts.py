"""
Tests for prompts/templates.py

No model needed — just the proxy_tokenizer from conftest.py.
MINIMAL_CHAT_TEMPLATE is passed directly to format_prompt() so tests
don't depend on the proxy tokenizer having its own chat template.

Run with:
  pytest tests/test_prompts.py -v
"""

import pytest
from tests.conftest import MINIMAL_CHAT_TEMPLATE
from prompts.templates import format_prompt, SYSTEM_PROMPT, TASK_TEMPLATE

SAMPLE_TASK = "adds two vectors elementwise"


# ── Helper ───────────────────────────────────────────────────────────────────
# Wraps format_prompt with the test template so every test call is concise.

def fp(task, tokenizer):
    return format_prompt(task, tokenizer, chat_template=MINIMAL_CHAT_TEMPLATE)


# ── Template constants (no fixtures needed) ───────────────────────────────────

class TestConstants:
    def test_system_prompt_is_not_empty(self):
        assert SYSTEM_PROMPT.strip(), "SYSTEM_PROMPT is empty"

    def test_system_prompt_mentions_triton(self):
        assert "triton" in SYSTEM_PROMPT.lower()

    def test_task_template_has_task_slot(self):
        """{task} must be present so format_prompt can fill it."""
        assert "{task}" in TASK_TEMPLATE

    def test_task_template_fills_without_error(self):
        filled = TASK_TEMPLATE.format(task=SAMPLE_TASK)
        assert SAMPLE_TASK in filled

    def test_task_template_slot_is_replaced(self):
        filled = TASK_TEMPLATE.format(task=SAMPLE_TASK)
        assert "{task}" not in filled


# ── format_prompt() output ────────────────────────────────────────────────────

class TestFormatPrompt:
    def test_returns_string(self, proxy_tokenizer):
        assert isinstance(fp(SAMPLE_TASK, proxy_tokenizer), str)

    def test_output_is_not_empty(self, proxy_tokenizer):
        assert fp(SAMPLE_TASK, proxy_tokenizer).strip()

    def test_task_appears_in_output(self, proxy_tokenizer):
        assert SAMPLE_TASK in fp(SAMPLE_TASK, proxy_tokenizer)

    def test_task_slot_is_not_raw_in_output(self, proxy_tokenizer):
        """{task} must be substituted, never appear literally."""
        assert "{task}" not in fp(SAMPLE_TASK, proxy_tokenizer)

    def test_triton_jit_hint_is_present(self, proxy_tokenizer):
        assert "@triton.jit" in fp(SAMPLE_TASK, proxy_tokenizer)

    def test_import_hint_is_present(self, proxy_tokenizer):
        assert "import triton" in fp(SAMPLE_TASK, proxy_tokenizer)

    def test_generation_prompt_is_appended(self, proxy_tokenizer):
        """With our minimal template, 'model:' appears at the end —
        this is where generation will continue from."""
        assert "model:" in fp(SAMPLE_TASK, proxy_tokenizer)

    def test_different_tasks_differ(self, proxy_tokenizer):
        a = fp("adds two vectors elementwise", proxy_tokenizer)
        b = fp("multiplies a matrix by a scalar", proxy_tokenizer)
        assert a != b

    def test_system_context_is_present(self, proxy_tokenizer):
        result = fp(SAMPLE_TASK, proxy_tokenizer)
        assert "tl.load" in result or "tl.store" in result