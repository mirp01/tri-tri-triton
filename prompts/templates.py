from transformers import PreTrainedTokenizerBase
from typing import Optional

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert GPU programmer specializing in Triton kernels.
Triton is a Python DSL for writing high-performance GPU kernels using the \
@triton.jit decorator.

Rules you must follow:
- Start every kernel with @triton.jit
- Import triton and triton.language as tl at the top
- Use tl.program_id() to get the thread index
- Use tl.load() and tl.store() for memory access, always with a mask
- Use tl.constexpr for compile-time constants in the signature
- Output only the kernel code — no explanations, no markdown fences\
"""

# ── Task template ──────────────────────────────────────────────────────────────
TASK_TEMPLATE = """\
Write a Triton kernel that {task}.

Start with:
import triton
import triton.language as tl

@triton.jit\
"""


def format_prompt(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    chat_template: Optional[str] = None,
) -> str:
    """
    Build a full prompt string for a given task description.

    Args:
        task:          Plain-English description of what the kernel should do.
                       e.g. "adds two vectors elementwise"
        tokenizer:     The tokenizer from your ModelBundle.
        chat_template: Optional Jinja2 template string. When None (default),
                       the tokenizer's own template is used — which is the
                       right behaviour in production with Gemma 4.
                       Pass a template explicitly in tests so the proxy
                       tokenizer (GPT-2) works without having its own template.

    Returns:
        A formatted string ready to be tokenized and fed to model.generate().
    """
    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\n{TASK_TEMPLATE.format(task=task)}",
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,   # None → use tokenizer's own template
    )