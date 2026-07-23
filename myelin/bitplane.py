"""Nested bit-plane quantization.

Format spec
-----------
A weight row ``w`` (one output channel) is stored as a per-row scale
``s = max|w|`` plus a sequence of sign planes ``b_1 .. b_K``, ``b_i in {-1,+1}``.
The k-bit reconstruction is

    q_k(w) = s * sum_{i=1..k} b_i * 2^{-i}

Plane ``i`` halves the residual left by planes ``1..i-1`` (residual half-step),
so the planes are prefix-nested: reading one more plane refines the value and
never invalidates the planes already stored. ``q_k`` is exactly the mid-rise
uniform quantizer with step ``2^{1-k}`` on [-1, 1]: its levels are the odd
multiples of ``2^{-k}``.

Two implementations are provided:

* :func:`plane_decompose` / :func:`plane_reconstruct` — the literal plane-by-plane
  form. Slow, but it *is* the storage format; serves as the golden model for the
  Phase 2 Rust kernel and Phase 3 RTL.
* :func:`quantize_unit` — closed-form mid-rise quantizer, mathematically equal
  to the plane reconstruction. Used in training for speed.

The scale must not depend on the bit-width (only on the row values), otherwise
nesting breaks — this is why we use absmax rather than a per-bit-width min-max
fit.
"""

from __future__ import annotations

import torch
from torch import Tensor


def plane_decompose(x: Tensor, num_planes: int) -> Tensor:
    """Decompose values in [-1, 1] into sign planes.

    Args:
        x: tensor of any shape with values in [-1, 1].
        num_planes: number of planes K to emit.

    Returns:
        Tensor of shape (K, *x.shape) with entries in {-1, +1}.
        Sign convention: ties (residual exactly 0) resolve to +1.
    """
    planes = []
    r = x.clone()
    for i in range(1, num_planes + 1):
        b = torch.where(r >= 0, 1.0, -1.0).to(x.dtype)
        planes.append(b)
        r = r - b * (2.0 ** -i)
    return torch.stack(planes, dim=0)


def plane_reconstruct(planes: Tensor, bits: Tensor | int) -> Tensor:
    """Reconstruct from sign planes, reading only the first ``bits`` planes.

    Args:
        planes: (K, R, C) tensor of {-1, +1} planes (rows R = channels).
        bits: int, or (R,) integer tensor — per-row prefix length, 1 <= bits <= K.

    Returns:
        (R, C) reconstruction.
    """
    K = planes.shape[0]
    weights = torch.pow(
        2.0, -torch.arange(1, K + 1, dtype=planes.dtype, device=planes.device)
    ).view(K, *([1] * (planes.dim() - 1)))
    if isinstance(bits, int):
        return (planes[:bits] * weights[:bits]).sum(dim=0)
    mask = (
        torch.arange(K, device=planes.device).view(K, 1, 1)
        < bits.view(1, -1, 1)
    ).to(planes.dtype)
    return (planes * mask * weights).sum(dim=0)


def quantize_unit(x: Tensor, bits: Tensor) -> Tensor:
    """Mid-rise quantizer on [-1, 1] with per-row bit-width.

    Mathematically identical to ``plane_reconstruct(plane_decompose(x, k), k)``
    for k = bits[row] (same tie convention: exact grid boundaries round up).

    Args:
        x: (R, C) values in [-1, 1].
        bits: (R,) integer tensor, >= 1.

    Returns:
        (R, C) quantized values in (-1, 1).
    """
    delta = torch.pow(2.0, 1.0 - bits.to(x.dtype)).unsqueeze(-1)  # (R, 1)
    q = delta * (torch.floor(x / delta) + 0.5)
    hi = 1.0 - 0.5 * delta
    return torch.clamp(q, -hi, hi)


def quantize_rows(w: Tensor, bits: Tensor, eps: float = 1e-12) -> Tensor:
    """Quantize each row of ``w`` to ``bits[row]`` bit-planes (no gradient tricks).

    Scale is per-row absmax of the *current* values; it depends only on the row,
    not on the bit-width, preserving plane nesting.
    """
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    return quantize_unit(w / scale, bits) * scale


def fake_quantize_rows(w: Tensor, bits: Tensor, eps: float = 1e-12) -> Tensor:
    """Straight-through fake quantization: forward = quantized, backward = identity.

    Written as ``q + (w - w.detach())`` — the parenthesized term is exactly +0.0
    elementwise, so the forward value is bit-exactly ``quantize_rows(w, bits)``
    (the ``w + (q - w.detach())`` form double-rounds and drifts up to 1 ulp,
    which would break the bit-exact golden-model contract with Phase 2 kernels).
    """
    with torch.no_grad():
        q = quantize_rows(w.detach(), bits, eps)
    return q + (w - w.detach())
