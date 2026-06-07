from __future__ import annotations

import torch
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedModel


@dataclass
class ModelBundle:
    """
    Holds the model and processor together so they're always passed as a
    single object.

    Gemma 4 uses AutoProcessor (not AutoTokenizer) — it bundles the chat
    template and handles text formatting. We store it as `tokenizer` so
    the rest of the pipeline (generator.py, templates.py) doesn't need
    to know the difference.
    """

    model:      PreTrainedModel
    tokenizer:  object          # AutoProcessor for Gemma 4
    vocab_size: int
    device:     torch.device

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
    # Primary path — model + processor already loaded in Colab
    # ------------------------------------------------------------------

    @classmethod
    def from_existing(
        cls,
        model:     PreTrainedModel,
        processor: object,
    ) -> ModelBundle:
        """
        Wrap a model + processor that are already in memory.

        In Colab where Gemma 4 is already loaded:

            bundle = ModelBundle.from_existing(model, processor)

        Works with both AutoProcessor (Gemma 4) and AutoTokenizer
        (any other model or the proxy tokenizer used in tests).
        """
        model.eval()

        # AutoProcessor wraps an underlying tokenizer.
        # Pad token must be set on that inner tokenizer, not on the
        # processor itself, or generation will error on batched inputs.
        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=processor,
            vocab_size=model.config.vocab_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Fallback — fresh load outside Colab
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_id:   str,
        dtype:      torch.dtype = torch.bfloat16,
        device_map: str         = "auto",
    ) -> ModelBundle:
        """
        Load model + processor from HuggingFace from scratch.
        Use this for fresh sessions or outside Colab.
        """
        print(f"Loading processor: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)

        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        print(f"Loading model: {model_id} ({dtype}, device_map={device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        model.eval()

        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=processor,
            vocab_size=model.config.vocab_size,
            device=device,
        )