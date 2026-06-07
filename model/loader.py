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

        tok = getattr(processor, "tokenizer", processor)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        device = next(model.parameters()).device

        return cls(
            model=model,
            tokenizer=processor,
            vocab_size=_get_vocab_size(model),
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
            vocab_size=_get_vocab_size(model),
            device=device,
        )


def _get_vocab_size(model: PreTrainedModel) -> int:
    """
    Safely read vocab_size from a model config.

    Standard models (GPT-2, Llama, etc.) store it at model.config.vocab_size.
    Gemma 4 and other multimodal models store it one level deeper at
    model.config.text_config.vocab_size.
    If neither exists, fall back to the embedding weight shape — this is
    always accurate because the embedding matrix IS the vocabulary.
    """
    if hasattr(model.config, "vocab_size"):
        return model.config.vocab_size

    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
        return text_cfg.vocab_size

    # Final fallback: embedding weight shape[0] == vocab_size by definition
    return model.get_input_embeddings().weight.shape[0]