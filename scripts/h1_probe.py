"""H1 probe: does the connectivity score predict per-channel quantization
sensitivity?

For a trained checkpoint, ALL channels are first reset to a uniform reference
bit-width; then, per sampled channel, that channel alone is dropped to min_bits
and the val-loss increase is measured, and correlated (Spearman) against the
connectivity score. H1 predicts a positive rank correlation.

The uniform reference matters: probing at the checkpoint's own (waterfilled)
allocation would be circular — bits were set as ~log4(score), so the drop
distance and hence the measured damage would grow with the score even under
the null hypothesis. Constant drop distance removes that artifact. Prefer
probing checkpoints whose training allocation was also score-independent
(uniform/random runs) for the cleanest reading.

Usage:
    python scripts/h1_probe.py runs/matrix/b4.0_uniform_s1337 \
        [--channels 128] [--eval-iters 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from myelin.config import TrainConfig
from myelin.data import get_batch, load_split, markov_corpus
from myelin.model import MiniGPT
from myelin.signals import FlowSignal, ProductSignal, QuantIndex, StructuralSignal


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


@torch.no_grad()
def val_loss(model, data, cfg, iters: int) -> float:
    g = torch.Generator().manual_seed(cfg.seed + 3)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg.batch_size, cfg.model.block_size, g)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--eval-iters", type=int, default=8)
    args = ap.parse_args()

    cfg = TrainConfig.from_json_file(os.path.join(args.run_dir, "config.json"))
    ckpt = torch.load(
        os.path.join(args.run_dir, "ckpt.pt"), map_location="cpu",
        weights_only=False,
    )
    model = MiniGPT(cfg.model)
    # strict=False: tolerate checkpoints from older code revisions that lack
    # newer tracking buffers (they are irrelevant to the probe)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    index = QuantIndex(model)

    if cfg.synthetic:
        corpus = markov_corpus(cfg.synthetic_vocab, cfg.synthetic_length, cfg.seed)
        n_val = max(cfg.model.block_size + 2, len(corpus) // 10)
        data = corpus[-n_val:]
    else:
        data = load_split(cfg.data_dir, "val")

    signals = {
        "product": ProductSignal().scores(index).numpy(),
        "structural": StructuralSignal().scores(index).numpy(),
        "flow": FlowSignal().scores(index).numpy(),
    }
    scores = signals["product"]

    # uniform reference allocation: constant drop distance for every channel
    ref_bits = round(cfg.alloc.avg_bits)
    saved = [m.bits.clone() for m in index.layers]
    for m in index.layers:
        m.bits.fill_(ref_bits)
    base = val_loss(model, data, cfg, args.eval_iters)
    print(f"base val loss @ uniform {ref_bits}-bit reference: {base:.4f}")

    rng = np.random.default_rng(0)
    sample = rng.choice(index.total_channels, size=min(args.channels, index.total_channels), replace=False)

    # map flat channel -> (layer, local idx)
    bounds = np.cumsum([0] + index.sizes)
    sens = []
    for ci in sample:
        li = int(np.searchsorted(bounds, ci, side="right") - 1)
        local = int(ci - bounds[li])
        layer = index.layers[li]
        layer.bits[local] = cfg.alloc.min_bits
        loss = val_loss(model, data, cfg, args.eval_iters)
        layer.bits[local] = ref_bits
        sens.append(loss - base)

    for m, b in zip(index.layers, saved):
        m.bits.copy_(b)

    sens = np.array(sens)
    result = {
        "n_channels": len(sample),
        "spearman_rho": spearman(scores[sample], sens),
        "spearman_by_signal": {
            name: spearman(sc[sample], sens) for name, sc in signals.items()
        },
        "mean_sensitivity": float(sens.mean()),
        "frac_positive": float((sens > 0).mean()),
    }
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.run_dir, "h1_probe.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
