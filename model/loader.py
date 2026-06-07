from __future__ import annotations

import torch
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedModel


@dataclass
class ModelBundle:
    model:      PreTrainedModel
    tokenizer:  object
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

    @classmethod
    def from_existing(
        cls,
        model:     PreTrainedModel,
        processor: object,
    ) -> ModelBundle:
        model.eval()

        # Get the underlying tokenizer from the processor
        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        # len(tok) gives the full vocab including special tokens —
        # this is what XGrammar needs to build token masks.
        # More reliable than model.config.vocab_size, which Gemma 4
        # stores in a nested text_config instead of at the top level.
        vocab_size = len(tok)

        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=processor,
            vocab_size=vocab_size,
            device=device,
        )

    @classmethod
    def load(
        cls,
        model_id:   str,
        dtype:      torch.dtype = torch.bfloat16,
        device_map: str         = "auto",
    ) -> ModelBundle:
        print(f"Loading processor: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)

        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        vocab_size = len(tok)

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
            vocab_size=vocab_size,
            device=device,
        )