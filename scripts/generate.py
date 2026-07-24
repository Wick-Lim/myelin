"""Generate Korean text from a trained Myelin checkpoint.

Runs the quantized forward (the same path training optimized). With
--check-kernel, every BitplaneLinear output for the first token is
cross-checked against the Rust bit-plane kernel (packed storage + bit-serial
matvec), demonstrating the train = deploy contract on a live model.

Usage:
    python scripts/generate.py runs/stage1/b3.0_connectivity_s1337 \
        --prompt "한국의 역사는" --tokens 120 [--check-kernel]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from myelin.config import TrainConfig
from myelin.model import MiniGPT
from myelin.signals import QuantIndex


def check_kernel(model: MiniGPT, index: QuantIndex, x: torch.Tensor) -> None:
    import myelin_kernel

    from myelin.bitplane import quantize_rows

    total_bytes = 0
    worst = 0.0
    hooks = []
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = (inp[0].detach(), out.detach())
        return hook

    for name, m in index.entries:
        hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    for name, m in index.entries:
        packed = myelin_kernel.pack_matrix(
            m.weight.detach().flatten().tolist(),
            m.out_features, m.in_features, m.bits.tolist(),
        )
        total_bytes += packed.plane_bytes()
        q_rs = torch.tensor(packed.dequantize()).view(m.out_features, m.in_features)
        q_py = quantize_rows(m.weight.detach(), m.bits)
        assert torch.equal(q_rs, q_py), f"{name}: dequantize mismatch"
        inp, out = captured[name]
        v = inp.reshape(-1, m.in_features)[-1]  # last position activation
        y_rs = np.array(packed.matvec(v.tolist()))
        y_py = out.reshape(-1, m.out_features)[-1].numpy()
        err = float(np.abs(y_rs - y_py).max())
        worst = max(worst, err)
    fp32_bytes = sum(m.weight.numel() for m in index.layers) * 4
    print(
        f"[kernel check] {len(index.entries)} layers: weights bit-exact, "
        f"matvec worst abs err {worst:.2e} on live activations\n"
        f"[kernel check] packed planes {total_bytes/1e6:.3f} MB vs fp32 "
        f"{fp32_bytes/1e6:.3f} MB ({fp32_bytes/total_bytes:.1f}x smaller)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="한국의 역사는")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-kernel", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig.from_json_file(os.path.join(args.run_dir, "config.json"))
    ckpt = torch.load(
        os.path.join(args.run_dir, "ckpt.pt"), map_location="cpu", weights_only=False
    )
    model = MiniGPT(cfg.model)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    from myelin.tokenizer import load

    tok = load(os.path.join(cfg.data_dir, "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long)

    if args.check_kernel:
        check_kernel(model, QuantIndex(model), x)

    g = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for _ in range(args.tokens):
            logits, _ = model(x[:, -cfg.model.block_size :])
            logits = logits[0, -1] / args.temperature
            if args.top_k > 0:
                v, _ = torch.topk(logits, args.top_k)
                logits[logits < v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
            x = torch.cat([x, nxt.view(1, 1)], dim=1)

    print("\n=== generated ===")
    print(tok.decode(x[0].tolist()))


if __name__ == "__main__":
    main()
