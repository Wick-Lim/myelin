"""Aggregate matrix results: per-(budget, strategy) mean +/- std and paired
per-seed comparisons against the random baseline (the H2 test).

Usage:
    python scripts/analyze.py runs/matrix
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_summaries(root: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/matrix"
    summaries = load_summaries(root)
    if not summaries:
        print(f"no summaries under {root}")
        return

    by_cell: dict[tuple, dict[int, float]] = defaultdict(dict)
    steps_of: dict[tuple, dict[int, int]] = defaultdict(dict)
    for s in summaries:
        key = (s["avg_bits_budget"], s["strategy"])
        by_cell[key][s["seed"]] = s["val_loss"]
        steps_of[key][s["seed"]] = s.get("steps", -1)

    print(f"{'budget':>7} {'strategy':>14} {'n':>2} {'val_loss':>18}")
    for (budget, strategy), seeds in sorted(by_cell.items()):
        v = np.array(list(seeds.values()))
        print(
            f"{budget:>7} {strategy:>14} {len(v):>2} "
            f"{v.mean():>10.4f} +/- {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
        )

    # paired per-seed comparisons at equal budget
    pairings = [
        ("H2", "connectivity", "random"),
        ("H3", "connectivity", "kquant"),
        ("signal", "connectivity", "fisher"),
    ]
    for label, a, b in pairings:
        header_printed = False
        for budget in sorted({bu for bu, _ in by_cell}):
            ra = by_cell.get((budget, a), {})
            rb = by_cell.get((budget, b), {})
            shared = sorted(set(ra) & set(rb))
            if not shared:
                continue
            if not header_printed:
                print(f"\n{label} paired diffs ({a} - {b}), negative = {a} wins:")
                header_printed = True
            # a paired diff is only valid if both runs trained identically
            mismatched = [
                s for s in shared
                if steps_of[(budget, a)][s] != steps_of[(budget, b)][s]
            ]
            if mismatched:
                print(
                    f"  budget {budget}: WARNING — seeds {mismatched} have "
                    f"mismatched step counts between {a} and {b}; excluded"
                )
                shared = [s for s in shared if s not in mismatched]
                if not shared:
                    continue
            diffs = np.array([ra[s] - rb[s] for s in shared])
            print(
                f"  budget {budget}: mean diff {diffs.mean():+.4f} "
                f"(per-seed: {', '.join(f'{d:+.4f}' for d in diffs)})"
            )


if __name__ == "__main__":
    main()
