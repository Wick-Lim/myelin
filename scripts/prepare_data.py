"""Prepare Korean Wikipedia data: download -> train BPE -> tokenize to .bin.

Usage:
    python scripts/prepare_data.py --out data/kowiki --vocab-size 8192 \
        [--max-docs 200000]

Produces in --out:
    corpus.txt      raw text (one document per line-block, <|endoftext|> separated)
    tokenizer.json  byte-level BPE tokenizer
    meta.json       {"vocab_size": ..., "train_tokens": ..., "val_tokens": ...}
    train.bin / val.bin   uint16 token ids (nanoGPT-style memmaps)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def download_kowiki(out_txt: str, max_docs: int | None) -> None:
    from datasets import load_dataset

    print("loading wikimedia/wikipedia 20231101.ko ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
    n = len(ds) if max_docs is None else min(max_docs, len(ds))
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in range(n):
            text = ds[i]["text"].strip()
            if len(text) < 200:
                continue
            f.write(text)
            f.write("\n<|endoftext|>\n")
    print(f"wrote {n} docs to {out_txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument(
        "--corpus", type=str, default="",
        help="use an existing text file instead of downloading kowiki",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    corpus_txt = args.corpus or os.path.join(args.out, "corpus.txt")
    if not args.corpus and not os.path.exists(corpus_txt):
        download_kowiki(corpus_txt, args.max_docs)

    tok_path = os.path.join(args.out, "tokenizer.json")
    if not os.path.exists(tok_path):
        from myelin.tokenizer import train_bpe

        print(f"training BPE vocab={args.vocab_size} ...")
        train_bpe([corpus_txt], args.vocab_size, tok_path)

    from myelin.tokenizer import load

    tok = load(tok_path)
    assert tok.get_vocab_size() <= 65535, "uint16 storage requires vocab <= 65535"

    print("tokenizing ...")
    chunks: list[np.ndarray] = []
    with open(corpus_txt, encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= 10_000:
                chunks.append(
                    np.asarray(tok.encode("".join(buf)).ids, dtype=np.uint16)
                )
                buf = []
        if buf:
            chunks.append(
                np.asarray(tok.encode("".join(buf)).ids, dtype=np.uint16)
            )

    arr = np.concatenate(chunks)
    # floor the val split so evaluation batches (block_size+2 tokens minimum)
    # always fit even on tiny test corpora
    n_val = max(4096, int(len(arr) * args.val_frac))
    if n_val >= len(arr):
        raise SystemExit(f"corpus too small: {len(arr)} tokens")
    train, val = arr[:-n_val], arr[-n_val:]
    train.tofile(os.path.join(args.out, "train.bin"))
    val.tofile(os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "vocab_size": tok.get_vocab_size(),
                "train_tokens": len(train),
                "val_tokens": len(val),
            },
            f,
            indent=2,
        )
    print(f"train tokens: {len(train):,}  val tokens: {len(val):,}")


if __name__ == "__main__":
    main()
