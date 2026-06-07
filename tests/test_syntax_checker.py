"""
Tests for validation/syntax_checker.py

No GPU, no Triton install needed — ast.parse() is stdlib.

Organised into four groups:
  TestSyntaxResult     — the dataclass itself
  TestValidKernel      — a well-formed kernel passes all checks
  TestPythonSyntax     — ast.parse() failures
  TestTritonStructure  — structural checks (imports, @triton.jit)

Run with:
  pytest tests/test_syntax_checker.py -v
"""

import pytest
from validation.syntax_checker import check_syntax, SyntaxResult

# ── Reference kernels ─────────────────────────────────────────────────────────

VALID_KERNEL = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * N + tl.arange(0, N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
""".strip()


# ── SyntaxResult dataclass ────────────────────────────────────────────────────

class TestSyntaxResult:
    def test_valid_true_no_errors(self):
        r = SyntaxResult(valid=True)
        assert r.valid is True
        assert r.errors == []

    def test_valid_false_with_errors(self):
        r = SyntaxResult(valid=False, errors=["something missing"])
        assert r.valid is False
        assert len(r.errors) == 1

    def test_errors_default_to_empty_list(self):
        """Each instance gets its own list — no shared mutable default."""
        r1 = SyntaxResult(valid=True)
        r2 = SyntaxResult(valid=True)
        r1.errors.append("oops")
        assert r2.errors == []


# ── Valid kernel ──────────────────────────────────────────────────────────────

class TestValidKernel:
    def test_valid_kernel_passes(self):
        result = check_syntax(VALID_KERNEL)
        assert result.valid is True
        assert result.errors == []

    def test_returns_syntax_result(self):
        assert isinstance(check_syntax(VALID_KERNEL), SyntaxResult)


# ── Python syntax failures ────────────────────────────────────────────────────

class TestPythonSyntax:
    def test_unmatched_parenthesis(self):
        bad = VALID_KERNEL.replace("tl.arange(0, N)", "tl.arange(0, N")
        result = check_syntax(bad)
        assert result.valid is False
        assert any("SyntaxError" in e for e in result.errors)

    def test_invalid_indentation(self):
        bad = "import triton\nimport triton.language as tl\n@triton.jit\ndef k():\npass"
        result = check_syntax(bad)
        assert result.valid is False
        assert any("SyntaxError" in e for e in result.errors)

    def test_empty_string(self):
        """Empty output from the generator has no @triton.jit."""
        result = check_syntax("")
        assert result.valid is False

    def test_syntax_error_reports_line_number(self):
        """Line number must appear in the error — it speeds up debugging."""
        bad = "import triton\nimport triton.language as tl\ndef k(:\n    pass"
        result = check_syntax(bad)
        assert result.valid is False
        assert any("line" in e.lower() for e in result.errors)


# ── Triton structural checks ──────────────────────────────────────────────────

class TestTritonStructure:

    # @triton.jit ──────────────────────────────────────────────────────────────

    def test_missing_triton_jit(self):
        no_decorator = VALID_KERNEL.replace("@triton.jit\n", "")
        result = check_syntax(no_decorator)
        assert result.valid is False
        assert any("triton.jit" in e for e in result.errors)

    def test_wrong_decorator_not_accepted(self):
        """@torch.jit.script is not @triton.jit."""
        wrong = VALID_KERNEL.replace("@triton.jit", "@torch.jit.script")
        result = check_syntax(wrong)
        assert result.valid is False
        assert any("triton.jit" in e for e in result.errors)

    def test_plain_function_not_accepted(self):
        """A bare def with no decorator must not pass."""
        plain = "import triton\nimport triton.language as tl\ndef kernel():\n    pass"
        result = check_syntax(plain)
        assert result.valid is False
        assert any("triton.jit" in e for e in result.errors)

    # imports ──────────────────────────────────────────────────────────────────

    def test_missing_import_triton(self):
        no_import = VALID_KERNEL.replace("import triton\n", "")
        result = check_syntax(no_import)
        assert result.valid is False
        assert any("import triton" in e for e in result.errors)

    def test_missing_import_tl(self):
        no_tl = VALID_KERNEL.replace("import triton.language as tl\n", "")
        result = check_syntax(no_tl)
        assert result.valid is False
        assert any("triton.language" in e for e in result.errors)

    def test_tl_without_alias_not_accepted(self):
        """'import triton.language' without 'as tl' must fail."""
        wrong_alias = VALID_KERNEL.replace(
            "import triton.language as tl", "import triton.language"
        )
        result = check_syntax(wrong_alias)
        assert result.valid is False

    def test_multiple_missing_items_all_reported(self):
        """All structural errors are returned in one pass."""
        bare = "@triton.jit\ndef kernel():\n    pass"
        result = check_syntax(bare)
        assert result.valid is False
        # Both import errors should be present
        assert len(result.errors) >= 2
        messages = " ".join(result.errors)
        assert "import triton" in messages
        assert "triton.language" in messages