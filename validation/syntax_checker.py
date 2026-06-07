"""
validation/syntax_checker.py

Lightweight, zero-GPU syntax validation for generated Triton kernels.
Runs two checks in sequence:

  1. Python syntax   — ast.parse() catches malformed Python
  2. Triton structure — AST walk verifies @triton.jit + required imports

Both run without importing Triton itself, so they're fast and GPU-free.
Semantic validation (does the kernel actually compile and run?) lives in
semantic_checker.py once TritonBench is integrated.
"""

import ast
from dataclasses import dataclass, field


@dataclass
class SyntaxResult:
    """
    Result returned by check_syntax().

    Attributes:
        valid:  True only when all checks pass.
        errors: List of human-readable error messages.
                Empty when valid=True. May contain multiple entries
                when several structural problems are found at once.
    """
    valid:  bool
    errors: list[str] = field(default_factory=list)


def check_syntax(code: str) -> SyntaxResult:
    """
    Run all syntax checks on a generated Triton kernel string.

    Checks are ordered cheapest-first and fail-fast at the Python
    syntax level — if ast.parse() raises, the AST-based checks below
    can't run, so we return immediately.

    Args:
        code: Raw string output from model/generator.py.

    Returns:
        SyntaxResult(valid=True)  if all checks pass.
        SyntaxResult(valid=False, errors=[...]) otherwise.
    """

    # ── Step 1: Python syntax ──────────────────────────────────────────────
    # ast.parse() raises SyntaxError for invalid Python.
    # We return early here because the structural checks below
    # require a valid AST to walk.
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return SyntaxResult(
            valid=False,
            errors=[f"SyntaxError on line {e.lineno}: {e.msg}"],
        )

    # ── Step 2: Triton structural checks ──────────────────────────────────
    # Walk the AST and check for required Triton patterns.
    # All checks run even if the first one fails, so the caller
    # gets the full picture in one pass.
    errors: list[str] = []
    errors.extend(_check_triton_jit(tree))
    errors.extend(_check_imports(tree))

    return SyntaxResult(valid=len(errors) == 0, errors=errors)


# ── AST checks ────────────────────────────────────────────────────────────────

def _check_triton_jit(tree: ast.Module) -> list[str]:
    """
    Verify that at least one function is decorated with @triton.jit.

    This is the minimum structural requirement for a valid Triton kernel.
    Without it, Triton won't register the function as a JIT-compiled kernel
    and any call to it will silently run as plain Python.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if _is_triton_jit(decorator):
                    return []
    return ["No @triton.jit decorated function found"]


def _is_triton_jit(node: ast.expr) -> bool:
    """Return True if an AST decorator node represents @triton.jit."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
        and node.attr == "jit"
    )


def _check_imports(tree: ast.Module) -> list[str]:
    """
    Verify both required import lines are present:

        import triton
        import triton.language as tl

    These are checked separately so the error message is specific about
    which one is missing — 'import triton' and 'import triton.language as tl'
    are independent lines the model might omit independently.
    """
    has_triton = False
    has_tl     = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton" and alias.asname is None:
                    has_triton = True
                if alias.name == "triton.language" and alias.asname == "tl":
                    has_tl = True

    errors = []
    if not has_triton:
        errors.append("Missing 'import triton'")
    if not has_tl:
        errors.append("Missing 'import triton.language as tl'")
    return errors