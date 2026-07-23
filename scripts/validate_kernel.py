"""Cross-validate the Rust kernel against the Python golden model.

Runs a randomized fuzz (pack -> dequantize bit-exact vs quantize_rows; matvec
vs f64 reference) and, if a run directory is given, validates the kernel on a
real trained checkpoint's quantized layers.

Usage:
    python scripts/validate_kernel.py [run_dir] [--fuzz 200]

Requires the extension built via:
    cd kernel && maturin develop --release --features python
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from myelin.bitplane import quantize_rows

try:
    import myelin_kernel
except ImportError:
    raise SystemExit(
        "myelin_kernel not built - run: cd kernel && maturin develop --release --features python"
    )


def check_matrix(w: torch.Tensor, bits: torch.Tensor, label: str) -> None:
    rows, cols = w.shape
    packed = myelin_kernel.pack_matrix(
        w.flatten().tolist(), rows, cols, bits.tolist()
    )

    q_py = quantize_rows(w, bits)
    q_rs = torch.tensor(packed.dequantize(), dtype=torch.float32).view(rows, cols)
    if not torch.equal(q_py, q_rs):
        bad = (q_py != q_rs).sum().item()
        raise SystemExit(f"[{label}] dequantize mismatch on {bad}/{rows*cols} weights")

    g = torch.Generator().manual_seed(rows * 31 + cols)
    x = torch.rand(cols, generator=g) * 2 - 1
    y_rs = np.array(packed.matvec(x.tolist()))
    y_ref = (q_py.double() @ x.double()).numpy()
    tol = 1e-5 * np.maximum(np.abs(y_ref), 1.0)
    if not (np.abs(y_rs - y_ref) <= tol).all():
        worst = np.abs(y_rs - y_ref).max()
        raise SystemExit(f"[{label}] matvec mismatch, worst abs err {worst}")


def fuzz(n: int) -> None:
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        rows = int(torch.randint(1, 24, (1,), generator=g))
        cols = int(torch.randint(1, 200, (1,), generator=g))
        scale = 10.0 ** float(torch.randint(-8, 9, (1,), generator=g))
        w = (torch.randn(rows, cols, generator=g) * scale).float()
        if i % 7 == 0 and rows > 1:
            w[0] = 0.0
        bits = torch.randint(1, 9, (rows,), generator=g)
        check_matrix(w, bits, f"fuzz#{i} {rows}x{cols}")
    print(f"fuzz OK: {n} random matrices bit-exact + matvec within tolerance")


def validate_checkpoint(run_dir: str) -> None:
    from myelin.config import TrainConfig
    from myelin.model import MiniGPT
    from myelin.signals import QuantIndex

    cfg = TrainConfig.from_json_file(os.path.join(run_dir, "config.json"))
    ckpt = torch.load(
        os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False
    )
    model = MiniGPT(cfg.model)
    model.load_state_dict(ckpt["model"], strict=False)
    index = QuantIndex(model)
    total_bytes = 0
    for name, m in index.entries:
        check_matrix(m.weight.detach().float(), m.bits, name)
        packed = myelin_kernel.pack_matrix(
            m.weight.detach().flatten().tolist(),
            m.out_features,
            m.in_features,
            m.bits.tolist(),
        )
        total_bytes += packed.plane_bytes()
    print(
        f"checkpoint OK: {len(index.entries)} layers bit-exact; "
        f"packed plane storage {total_bytes/1e6:.3f} MB "
        f"(fp32 would be {sum(m.weight.numel() for m in index.layers)*4/1e6:.3f} MB)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="")
    ap.add_argument("--fuzz", type=int, default=200)
    args = ap.parse_args()
    fuzz(args.fuzz)
    if args.run_dir:
        validate_checkpoint(args.run_dir)


if __name__ == "__main__":
    main()
