"""
A wrapper class for watermark generator.
"""

from typing import Any, Literal

import torch
from transformers import LogitsProcessorList, PreTrainedTokenizer


def get_wm_logits_processor(type: str, **kwargs) -> LogitsProcessorList:
    wmp = None
    match type:
        case "KGW":
            wmp = KGWWMLogitsProcessor(**kwargs)
        case "SIR":
            return SIRWMLogitsProcessor(**kwargs)
        case "PRW":
            return PRWWMLogitsProcessor(**kwargs)
        case _:
            raise ValueError(f"Invalid type: {type}")
    return LogitsProcessorList([wmp])

def KGWWMLogitsProcessor(tokenizer: PreTrainedTokenizer, key: Any, gamma: float, delta: float, seeding_scheme: str, **kwargs):
    from KGW.extended_watermark_processor import WatermarkLogitsProcessor
    return WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            hash_key=key,
        )


def SIRWMLogitsProcessor(tokenizer: PreTrainedTokenizer, key: int, mode: Literal["window", "context"], window_size: int, gamma: float, delta: float, chunk_length: int, transform_model_path: str, embedding_model: str, device: str, **kwargs):
    from SIR.watermark import WatermarkContext, WatermarkLogitsProcessor, WatermarkWindow
    if mode == "window":
        watermark_model = WatermarkWindow(
            device=device,
            window_size=window_size,
            target_tokenizer=tokenizer,
            target_vocab_size=tokenizer.vocab_size,
            gamma=gamma,
            delta=delta,
            hash_key=key,
        )
    elif mode == "context":
        watermark_model = WatermarkContext(
            device=device,
            chunk_length=chunk_length,
            target_tokenizer=tokenizer,
            target_vocab_size=tokenizer.vocab_size,
            gamma=gamma,
            delta=delta,
            transform_model_path=transform_model_path,
            embedding_model=embedding_model,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return WatermarkLogitsProcessor(watermark_model)


def PRWWMLogitsProcessor(tokenizer: PreTrainedTokenizer, key: int, fraction: float, strength: float, **kwargs):
    from PRW.gptwm import GPTWatermarkLogitsWarper
    return GPTWatermarkLogitsWarper(
            fraction=fraction,
            strength=strength,
            vocab_size=tokenizer.vocab_size,
            watermark_key=int(key),
        )
