import xgrammar as xgr
from pathlib import Path

from model.loader import ModelBundle

# Default path: grammar/triton.ebnf, relative to this file
GRAMMAR_PATH = Path(__file__).parent / "triton.ebnf"


def load_compiled_grammar(
    bundle: ModelBundle,
    grammar_path: Path = GRAMMAR_PATH,
) -> xgr.CompiledGrammar:
    """
    Read triton.ebnf, parse it with XGrammar, and compile it
    against the tokenizer vocabulary in the given ModelBundle.

    Call this once at startup and pass the result around — never
    call it inside the generation loop.

    Args:
        bundle:       A ModelBundle from loader.py. Provides the tokenizer
                      and vocab_size that XGrammar needs to build token masks.
        grammar_path: Path to the .ebnf file. Defaults to grammar/triton.ebnf.

    Returns:
        A CompiledGrammar ready to be passed to GrammarMatcher in generator.py.

    Raises:
        FileNotFoundError: if triton.ebnf does not exist at grammar_path.
        ValueError:        if the EBNF string is malformed.
    """

    # --- Step 1: Read the raw EBNF text ---
    if not grammar_path.exists():
        raise FileNotFoundError(
            f"Grammar file not found: {grammar_path}\n"
            "Make sure triton.ebnf exists in the grammar/ folder."
        )

    ebnf_string = grammar_path.read_text(encoding="utf-8")

    # --- Step 2: Parse into an XGrammar Grammar object ---
    try:
        grammar = xgr.Grammar.from_ebnf(ebnf_string)
    except Exception as e:
        raise ValueError(
            f"Failed to parse triton.ebnf — check your EBNF syntax.\n"
            f"XGrammar error: {e}"
        ) from e

    # --- Step 3: Build TokenizerInfo ---
    # vocab_size comes from bundle.vocab_size (i.e. model.config.vocab_size),
    # not len(tokenizer) — special tokens can inflate the tokenizer count
    # and silently break the token masks.
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        bundle.tokenizer,
        vocab_size=bundle.vocab_size,
    )

    # --- Step 4: Compile ---
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)

    return compiled_grammar


if __name__ == "__main__":
    # Sanity check — run from the project root:
    #   python -m grammar.compiler
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import MODEL_ID

    bundle = ModelBundle.load(MODEL_ID)

    print("Compiling grammar (this may take a moment)...")
    try:
        compiled = load_compiled_grammar(bundle)
        print(f"Grammar compiled successfully — type: {type(compiled).__name__}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)