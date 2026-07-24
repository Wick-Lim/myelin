"""Stage-2 verdict: does informed spread beat uniform?

Merges stage-1 (gamma=1 arms + uniform/random anchors) with stage-2 gamma
ablation runs and prints the paired comparisons that decide the ordering
question. All runs share seeds/data/eval batches, so per-seed pairing is exact.

Usage:
    python scripts/stage2_verdict.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ARMS = {
    "conn-g1": "runs/stage1/b3.0_connectivity_s{seed}",
    "fisher-g1": "runs/stage1/b3.0_fisher_s{seed}",
    "uniform": "runs/stage1/b3.0_uniform_s{seed}",
    "random": "runs/stage1/b3.0_random_s{seed}",
    "kquant": "runs/stage1/b3.0_kquant_s{seed}",
    "fp32": "runs/stage1/b3.0_fp32_s{seed}",
    "conn-g2": "runs/stage2_g2/b3.0_connectivity_s{seed}",
    "fisher-g2": "runs/stage2_g2/b3.0_fisher_s{seed}",
    "conn-g3": "runs/stage2_g3/b3.0_connectivity_s{seed}",
}
SEEDS = [1337, 1338, 1339]


def load() -> dict:
    out = {}
    for arm, pat in ARMS.items():
        for seed in SEEDS:
            path = os.path.join(pat.format(seed=seed), "summary.json")
            if os.path.exists(path):
                with open(path) as f:
                    out[(arm, seed)] = json.load(f)
    return out


def main() -> None:
    cells = load()
    print(f"{'arm':>10} {'n':>2} {'val_loss':>20}  {'mean_bits':>9}")
    for arm in ARMS:
        vals = [cells[(arm, s)]["val_loss"] for s in SEEDS if (arm, s) in cells]
        if not vals:
            continue
        v = np.array(vals)
        mb = np.mean(
            [cells[(arm, s)].get("mean_bits", 0) for s in SEEDS if (arm, s) in cells]
        )
        std = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{arm:>10} {len(v):>2} {v.mean():>11.4f} +/- {std:.4f}  {mb:>9.2f}")

    pairings = [
        ("핵심: 정보 있는 spread가 uniform을 이기는가", "conn-g2", "uniform"),
        ("spread 강화의 한계", "conn-g3", "uniform"),
        ("gamma 자체의 효과", "conn-g2", "conn-g1"),
        ("신호 상한 대조", "fisher-g2", "uniform"),
        ("fisher도 spread로 이득?", "fisher-g2", "fisher-g1"),
        ("정보 있는 vs 없는 spread", "conn-g2", "random"),
    ]
    for label, a, b in pairings:
        shared = [s for s in SEEDS if (a, s) in cells and (b, s) in cells]
        if not shared:
            print(f"\n[{label}] {a} vs {b}: 데이터 없음")
            continue
        diffs = np.array(
            [cells[(a, s)]["val_loss"] - cells[(b, s)]["val_loss"] for s in shared]
        )
        sign = "WIN" if (diffs < 0).all() else ("LOSE" if (diffs > 0).all() else "mixed")
        print(
            f"\n[{label}] {a} - {b}: mean {diffs.mean():+.4f} [{sign}] "
            f"(per-seed: {', '.join(f'{d:+.4f}' for d in diffs)})"
        )


if __name__ == "__main__":
    main()
