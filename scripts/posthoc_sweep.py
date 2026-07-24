"""Post-hoc budget-dial sweep: allocation value when adaptation is frozen.

Stage 2 showed that DURING training, uniform allocation wins — QAT adapts
channels to whatever precision they get, making sensitivity endogenous. This
script tests the deploy-time story instead: take a uniform-trained checkpoint,
re-allocate bits post-hoc (no retraining — nested planes make this free) at a
range of budgets, and compare orderings. Here sensitivity is exogenous, which
is where PTQ mixed precision classically wins.

Usage:
    python scripts/posthoc_sweep.py [--seeds 1337 1338 1339] [--eval-iters 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from myelin.allocator import waterfill
from myelin.config import TrainConfig
from myelin.data import get_batch, load_split
from myelin.model import MiniGPT
from myelin.signals import (
    ProductSignal,
    QuantIndex,
    RandomFixedSignal,
    UniformSignal,
)

BUDGETS = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
ORDERINGS = ["connectivity", "random", "uniform"]


@torch.no_grad()
def val_loss(model, data, cfg, iters):
    g = torch.Generator().manual_seed(cfg.seed + 3)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg.batch_size, cfg.model.block_size, g)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 1338, 1339])
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--out", default="runs/posthoc_sweep.json")
    args = ap.parse_args()

    results = []
    for seed in args.seeds:
        run = f"runs/stage1/b3.0_uniform_s{seed}"
        cfg = TrainConfig.from_json_file(os.path.join(run, "config.json"))
        ckpt = torch.load(
            os.path.join(run, "ckpt.pt"), map_location="cpu", weights_only=False
        )
        model = MiniGPT(cfg.model)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        index = QuantIndex(model)
        data = load_split(cfg.data_dir, "val")
        n = index.total_channels

        scores = {
            "connectivity": ProductSignal().scores(index),
            "random": RandomFixedSignal(seed + 2).scores(index),
            "uniform": UniformSignal().scores(index),
        }
        for budget in BUDGETS:
            for ordering in ORDERINGS:
                bits = waterfill(
                    scores[ordering].clamp_min(1e-12), round(budget * n), 2, 8
                )
                index.set_flat_bits(bits)
                loss = val_loss(model, data, cfg, args.eval_iters)
                results.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "ordering": ordering,
                        "val_loss": loss,
                    }
                )
                print(f"seed {seed} budget {budget:>4} {ordering:>12}: {loss:.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

    # paired summary: ordering vs uniform per budget
    print("\n=== paired diffs vs uniform ordering (negative = ordering wins) ===")
    by = {(r["seed"], r["budget"], r["ordering"]): r["val_loss"] for r in results}
    for budget in BUDGETS:
        row = []
        for ordering in ["connectivity", "random"]:
            diffs = [
                by[(s, budget, ordering)] - by[(s, budget, "uniform")]
                for s in args.seeds
            ]
            d = np.array(diffs)
            sign = "WIN" if (d < 0).all() else ("LOSE" if (d > 0).all() else "mixed")
            row.append(f"{ordering}: {d.mean():+.4f} [{sign}]")
        print(f"budget {budget:>4}  " + "  ".join(row))


if __name__ == "__main__":
    main()
