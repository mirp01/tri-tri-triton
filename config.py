from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────────
# Update MODEL_ID to match exactly what you used to load the model in Colab.
# This is used by ModelBundle.load() for fresh loads and by the __main__
# sanity checks in compiler.py and generator.py.
MODEL_ID = "google/gemma-4-E4B-it"

# ── Generation ─────────────────────────────────────────────────────────────────
# Lower temperature = more deterministic output.
# For code generation 0.2–0.4 is a good starting range; with constrained
# decoding the grammar already filters invalid tokens, so you don't need
# randomness to explore the space — you need the model to commit.
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.3
TOP_P          = 0.95
DO_SAMPLE      = True   # set False to use greedy decoding during debugging

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
GRAMMAR_PATH = PROJECT_ROOT / "grammar" / "triton.ebnf"

# ── Device ─────────────────────────────────────────────────────────────────────
# "cuda" for Colab / any GPU machine.
# "cpu"  for local runs without a GPU (slow, but useful for testing the
#         pipeline logic without running inference).
DEVICE = "cuda"