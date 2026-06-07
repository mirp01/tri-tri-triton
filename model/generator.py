import torch
import xgrammar as xgr
from transformers import LogitsProcessorList

import config
from model.loader import ModelBundle


def generate(
    prompt: str,
    bundle: ModelBundle,
    compiled_grammar: xgr.CompiledGrammar,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    do_sample: bool = config.DO_SAMPLE,
) -> str:
    """
    Run constrained generation and return the generated Triton kernel.

    Each call creates a fresh GrammarMatcher — matchers are stateful and
    advance through the grammar's pushdown automaton as tokens are produced.
    The CompiledGrammar is expensive to build and must be reused across calls
    (build it once in pipeline.py via compiler.py).

    Args:
        prompt:          Formatted prompt string from prompts/templates.py.
        bundle:          ModelBundle holding Gemma 4 12B + tokenizer.
        compiled_grammar: CompiledGrammar from grammar/compiler.py.
        max_new_tokens:  Cap on generated tokens. Default from config.py.
        temperature:     Sampling temperature. Lower = more deterministic.
        top_p:           Nucleus sampling threshold.
        do_sample:       False = greedy decoding (useful for debugging).

    Returns:
        The raw generated string — just the new tokens, prompt stripped.
        This goes straight into validation/pipeline.py.
    """

    # ── 1. Tokenize prompt ────────────────────────────────────────────────────
    inputs = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(bundle.device)

    prompt_len = inputs["input_ids"].shape[1]

    # ── 2. Set up constrained decoding ────────────────────────────────────────
    # GrammarMatcher wraps the compiled grammar and tracks the current
    # grammar state. It's called by XGrammar's logits processor at every
    # decoding step to zero-out token IDs that would violate the grammar.
    matcher = xgr.GrammarMatcher(compiled_grammar)
    xgr_processor = xgr.contrib.hf.LogitsProcessor(matcher)

    # ── 3. Generate ───────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            logits_processor=LogitsProcessorList([xgr_processor]),
            pad_token_id=bundle.tokenizer.pad_token_id,
        )

    # ── 4. Decode new tokens only ─────────────────────────────────────────────
    # output_ids includes the prompt — slice it off before decoding.
    new_tokens = output_ids[0][prompt_len:]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # End-to-end sanity check — run from the project root:
    #   python -m model.generator
    #
    # Expects MODEL_ID in config.py to match your Colab setup.
    # On Gemma 4 12B this takes ~30s on an A100.
    from grammar.compiler import load_compiled_grammar
    from prompts.templates import format_prompt

    print(f"Loading {config.MODEL_ID}...")
    bundle = ModelBundle.load(config.MODEL_ID)

    print("Compiling grammar...")
    compiled = load_compiled_grammar(bundle)

    task = "adds two vectors elementwise"
    prompt = format_prompt(task, bundle.tokenizer)

    print(f"Generating kernel for: {task!r}\n")
    kernel = generate(prompt, bundle, compiled)
    print(kernel)