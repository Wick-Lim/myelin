"""Byte-level BPE tokenizer training/loading (HuggingFace tokenizers)."""

from __future__ import annotations


def train_bpe(files: list[str], vocab_size: int, out_path: str):
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    tok.train(
        files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    tok.save(out_path)
    return tok


def load(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)
