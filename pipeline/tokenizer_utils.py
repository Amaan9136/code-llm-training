from __future__ import annotations
from typing import Dict, List, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from core.logging import get_logger
logger = get_logger("tokenizer")
def load_tokenizer(model_name: str, config: Optional[dict] = None) -> PreTrainedTokenizer:
    cfg = config or {}
    model_cfg = cfg.get("model", {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
def tokenize_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    text_field: str = "text",
    num_proc: int = 4,
):
    def tokenize_fn(examples):
        result = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    return dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in dataset.column_names if c != "text"],
        desc="Tokenizing",
    )
def compute_token_stats(dataset, tokenizer: PreTrainedTokenizer, sample_size: int = 1000) -> dict:
    sample = dataset.select(range(min(sample_size, len(dataset))))
    lengths = [len(tokenizer.encode(ex["text"])) for ex in sample]
    return {
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
        "min": min(lengths),
        "total_tokens": sum(lengths),
        "samples": len(lengths),
    }