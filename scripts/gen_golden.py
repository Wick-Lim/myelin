"""Generate golden test vectors for the Rust kernel (kernel/tests/golden.json).

The Rust side must reproduce `quantize_rows` bit-exactly and match an f64
reference dot within tolerance. Regenerate whenever the format spec changes;
the same vectors are the seed corpus for the Phase 3 RTL testbench.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from myelin.bitplane import quantize_rows


def make_case(name: str, w: torch.Tensor, bits: list[int], seed: int) -> dict:
    rows, cols = w.shape
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(cols, generator=g, dtype=torch.float32) * 2 - 1).float()
    bits_t = torch.tensor(bits, dtype=torch.long)
    q = quantize_rows(w.float(), bits_t)
    y = (q.double() @ x.double()).tolist()
    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "bits": bits,
        "weights": w.float().flatten().tolist(),
        "x": x.tolist(),
        "expected_q": q.flatten().tolist(),
        "expected_y": y,
    }


def main() -> None:
    torch.manual_seed(1234)
    cases = []

    cases.append(
        make_case("random_4x8", torch.randn(4, 8), [1, 3, 5, 8], seed=1)
    )
    cases.append(
        make_case(
            "random_6x100_nonmult64",
            torch.randn(6, 100) * 2.5,
            [2, 4, 6, 8, 1, 7],
            seed=2,
        )
    )
    cases.append(
        make_case(
            "random_8x128",
            torch.randn(8, 128),
            [(r % 8) + 1 for r in range(8)],
            seed=3,
        )
    )
    cases.append(
        make_case("cols_65_word_padding", torch.randn(3, 65), [4, 8, 2], seed=4)
    )
    cases.append(make_case("single_col", torch.randn(5, 1), [1, 2, 4, 6, 8], seed=5))

    w = torch.randn(4, 32)
    w[1] = 0.0  # all-zero row
    w[2] = -w[2].abs()  # all-negative row
    cases.append(make_case("zero_and_negative_rows", w, [4, 4, 4, 4], seed=6))

    w = torch.randn(4, 16)
    w[0] *= 1e-8
    w[1] *= 1e8
    w[2, 0] = -0.0  # negative zero: sign(0) := +1 convention
    w[3, :] = w[3, 0]  # constant row: every value at the absmax boundary
    cases.append(make_case("extreme_magnitudes", w, [3, 3, 5, 5], seed=7))

    # values exactly on quantizer grid boundaries (worst case for tie handling)
    k = 4
    grid = torch.arange(-8, 9, dtype=torch.float32) / 8.0  # multiples of 2^-3
    w = grid.repeat(2, 1)
    w[1] *= 0.7307  # off-grid copy for contrast
    cases.append(make_case("grid_boundaries", w, [k, k], seed=8))

    # ulp ladders around every grid boundary — the round-half-even collapse
    # window where a residual-form decomposition diverges from the closed form
    for k in (3, 8):
        delta = 2.0 ** (1 - k)
        g = torch.arange(-2 ** (k - 1), 2 ** (k - 1) + 1, dtype=torch.float32) * delta
        vals = [g]
        lo = torch.full_like(g, -2.0)
        hi = torch.full_like(g, 2.0)
        below, above = g.clone(), g.clone()
        for _ in range(16):
            below = torch.nextafter(below, lo)
            above = torch.nextafter(above, hi)
            vals.extend([below.clone(), above.clone()])
        row = torch.cat(vals + [torch.tensor([1.0])]).clamp(-1.0, 1.0)
        cases.append(
            make_case(f"boundary_ulp_ladders_k{k}", row.unsqueeze(0), [k], seed=100 + k)
        )

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kernel",
        "tests",
        "golden.json",
    )
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f)
    n = sum(c["rows"] * c["cols"] for c in cases)
    print(f"wrote {len(cases)} cases ({n} weights) to {out_path}")


if __name__ == "__main__":
    main()
