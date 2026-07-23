"""Token stream data: memmapped corpora and a synthetic Markov corpus.

Real runs use nanoGPT-style ``train.bin`` / ``val.bin`` (uint16 token ids)
produced by ``scripts/prepare_data.py``. Tests and demos use
:func:`markov_corpus`, a random-but-learnable stream: a fixed sparse
transition table gives the model actual structure to fit, unlike i.i.d. noise.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor


def load_split(data_dir: str, split: str) -> np.ndarray:
    path = os.path.join(data_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def markov_corpus(
    vocab_size: int, length: int, seed: int = 0, branching: int = 4
) -> np.ndarray:
    """Random walk on a fixed sparse transition graph (uint16 tokens)."""
    rng = np.random.default_rng(seed)
    nxt = rng.integers(0, vocab_size, size=(vocab_size, branching))
    probs = rng.dirichlet(np.ones(branching) * 0.5, size=vocab_size)
    cdf = np.cumsum(probs, axis=1)
    out = np.empty(length, dtype=np.uint16)
    u = rng.random(length)
    tok = int(rng.integers(0, vocab_size))
    for i in range(length):
        out[i] = tok
        j = int(np.searchsorted(cdf[tok], u[i]))
        tok = int(nxt[tok, min(j, branching - 1)])
    return out


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample random contiguous windows; returns (x, y) with y shifted by one."""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(
            f"split has {len(data)} tokens; need at least block_size + 2 = "
            f"{block_size + 2}"
        )
    ix = torch.randint(0, n, (batch_size,), generator=generator)
    xs = np.stack([np.asarray(data[i : i + block_size]) for i in ix.tolist()])
    ys = np.stack(
        [np.asarray(data[i + 1 : i + 1 + block_size]) for i in ix.tolist()]
    )
    x = torch.from_numpy(xs.astype(np.int64))
    y = torch.from_numpy(ys.astype(np.int64))
    return x, y
