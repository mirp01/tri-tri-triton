from __future__ import annotations

import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ModelBundle:
    """
    Holds the model and tokenizer together so they're always
    passed as a single object — no risk of them getting out of sync.

    Two ways to create one:
      - ModelBundle.from_existing(model, tokenizer)  ← Colab: model already loaded
      - ModelBundle.load(model_id)                   ← fresh load from HuggingFace
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    vocab_size: int       # model.config.vocab_size, stored here so compiler.py
    device: torch.device  # doesn't have to reach into the model to get it

    def __repr__(self) -> str:
        dtype = next(self.model.parameters()).dtype
        return (
            f"ModelBundle("
            f"model={self.model.__class__.__name__}, "
            f"vocab_size={self.vocab_size}, "
            f"device={self.device}, "
            f"dtype={dtype})"
        )

    # ------------------------------------------------------------------
    # Primary path for your Colab setup
    # ------------------------------------------------------------------

    @classmethod
    def from_existing(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> ModelBundle:
        """
        Wrap a model + tokenizer that are already loaded in memory.

        Use this in your Colab notebook where Gemma 4 is already in VRAM:

            from model.loader import ModelBundle
            bundle = ModelBundle.from_existing(model, tokenizer)

        This does NOT reload or move the model — it just packages it
        alongside metadata that the rest of the pipeline needs.
        """
        model.eval()  # make sure it's in inference mode

        # Gemma's tokenizer often ships without a pad token set.
        # Generation will silently misbehave if this is missing.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=tokenizer,
            vocab_size=model.config.vocab_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Fallback: fresh load (useful outside Colab or for a clean session)
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> ModelBundle:
        """
        Load a model + tokenizer from HuggingFace from scratch.

        Only use this when the model isn't already in memory — for example
        when running outside Colab or starting a fresh session.

            from model.loader import ModelBundle
            from config import MODEL_ID
            bundle = ModelBundle.load(MODEL_ID)

        Args:
            model_id:   HuggingFace model ID, e.g. "google/gemma-4-2b-it".
            dtype:      Tensor dtype. bfloat16 is the right choice for Gemma
                        on modern GPUs — float16 can cause NaNs on some layers.
            device_map: "auto" lets HuggingFace distribute across available GPUs.
                        Use "cuda:0" to pin to a single GPU.
        """
        print(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(f"Loading model: {model_id} ({dtype}, device_map={device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model.eval()

        # When device_map="auto" the model may be spread across devices.
        # We grab the device of the first parameter as a representative.
        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=tokenizer,
            vocab_size=model.config.vocab_size,
            device=device,
        )