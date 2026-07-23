"""Connectivity signals: per-channel scores that drive bit allocation.

All signals return one non-negative score per quantized output channel, as a
single flat tensor aligned with :class:`QuantIndex`. Scores are compared across
layers by the allocator, so each signal normalizes per layer to mean 1 — this
makes the product signal dimensionless and roughly comparable across layers
(raw weight/activation scales differ wildly between layers; see DESIGN.md §3).

The allocator turns score ratios into bit differences logarithmically
(a 4x score ratio ~ +1 bit), so scores only need to be meaningful up to
monotone ratio, not absolute magnitude.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from myelin.layers import BitplaneLinear

_EPS = 1e-12


class QuantIndex:
    """Deterministic registry of all quantized layers in a model.

    Provides the flat channel indexing shared by signals and the allocator.
    """

    def __init__(self, model: nn.Module):
        self.entries: list[tuple[str, BitplaneLinear]] = [
            (name, m)
            for name, m in model.named_modules()
            if isinstance(m, BitplaneLinear)
        ]
        self.sizes = [m.out_features for _, m in self.entries]
        self.total_channels = sum(self.sizes)

    @property
    def layers(self) -> list[BitplaneLinear]:
        return [m for _, m in self.entries]

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.entries]

    def flat_bits(self) -> Tensor:
        return torch.cat([m.bits for _, m in self.entries])

    @torch.no_grad()
    def set_flat_bits(self, flat: Tensor) -> None:
        assert flat.numel() == self.total_channels
        off = 0
        for _, m in self.entries:
            m.bits.copy_(flat[off : off + m.out_features])
            off += m.out_features

    def split(self, flat: Tensor) -> list[Tensor]:
        return list(torch.split(flat, self.sizes))


def _norm(x: Tensor) -> Tensor:
    return x / x.mean().clamp_min(_EPS)


class Signal:
    """Base class. Subclasses return a flat (total_channels,) score tensor."""

    #: whether scores can change over training (False => allocation is static
    #: after the initial waterfill; the allocator may skip reallocation cycles)
    dynamic = True
    #: whether the trainer must feed per-step weight gradients via observe_grad
    needs_grad = False

    def scores(self, index: QuantIndex) -> Tensor:
        raise NotImplementedError


class StructuralSignal(Signal):
    """Per-layer-normalized shadow-weight row norms."""

    def scores(self, index: QuantIndex) -> Tensor:
        return torch.cat([_norm(m.structural()) for m in index.layers])


class FlowSignal(Signal):
    """Per-layer-normalized activation-magnitude EMA."""

    def scores(self, index: QuantIndex) -> Tensor:
        return torch.cat([_norm(m.act_ema.clamp_min(0)) for m in index.layers])


class ProductSignal(Signal):
    """(normalized weight norm) x (normalized activation flow). Myelin default."""

    def scores(self, index: QuantIndex) -> Tensor:
        parts = []
        for m in index.layers:
            parts.append(_norm(m.structural()) * _norm(m.act_ema.clamp_min(0)))
        return torch.cat(parts)


class FisherSignal(Signal):
    """Diagonal-Fisher sensitivity proxy — the strong scientific control.

    Loss damage from quantization noise on row c is ~ sum_j E[g_cj^2] * E[dw^2],
    with noise scale dw ~ s_c * 2^{-bits}. Score = EMA(sum_j g_cj^2) * s_c^2,
    the HAWQ/FIT-family quantity at per-channel granularity. If connectivity
    cannot match this signal, the connectivity hypothesis loses (see
    docs/RESEARCH.md section 2).
    """

    needs_grad = True

    def scores(self, index: QuantIndex) -> Tensor:
        parts = []
        for m in index.layers:
            s = m.grad_sq_ema.clamp_min(0) * m.row_scale().pow(2)
            parts.append(_norm(s))
        return torch.cat(parts)


class RandomFixedSignal(Signal):
    """Random ranking drawn once; stable thereafter. The H2 control group:
    identical mechanics and budget, only the ordering is uninformed."""

    dynamic = False

    def __init__(self, seed: int):
        self.seed = seed
        self._cache: Tensor | None = None

    def scores(self, index: QuantIndex) -> Tensor:
        if self._cache is None or self._cache.numel() != index.total_channels:
            g = torch.Generator().manual_seed(self.seed)
            # Log-uniform over ~[4^-2, 4^2] so random bit assignments spread
            # across the full range instead of clustering at the mean.
            u = torch.rand(index.total_channels, generator=g)
            self._cache = torch.pow(4.0, (u - 0.5) * 4.0)
        return self._cache


class RandomChurnSignal(Signal):
    """Random ranking re-drawn every call: measures the cost of churn itself."""

    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    def scores(self, index: QuantIndex) -> Tensor:
        u = torch.rand(index.total_channels, generator=self.g)
        return torch.pow(4.0, (u - 0.5) * 4.0)


class KQuantSignal(Signal):
    """llama.cpp Q4_K_M-style hand-tuned heuristic (H3 baseline).

    Faithful to llama-quant.cpp @ 4310aa4 (docs/RESEARCH.md section 3):
    attn_v and ffn_down are promoted by +2 bits (Q4_K -> Q6_K) only on layers
    selected by ``use_more_bits`` — first 1/8, last 1/8, and every third layer
    of the middle band (integer arithmetic, as upstream). Score = 4^offset so
    the allocator's log4 waterfill reproduces the intended integer offsets.
    """

    dynamic = False

    PROMOTED_ROLES = ("attn_v", "mlp_down")
    PROMOTE_OFFSET = 2

    @staticmethod
    def use_more_bits(i_layer: int, n_layer: int) -> bool:
        return (
            i_layer < n_layer // 8
            or i_layer >= 7 * n_layer // 8
            or (i_layer - n_layer // 8) % 3 == 2
        )

    def scores(self, index: QuantIndex) -> Tensor:
        n_layer = max((m.layer_idx for m in index.layers), default=-1) + 1
        parts = []
        for m in index.layers:
            off = 0
            if (
                m.role in self.PROMOTED_ROLES
                and m.layer_idx >= 0
                and self.use_more_bits(m.layer_idx, n_layer)
            ):
                off = self.PROMOTE_OFFSET
            parts.append(torch.full((m.out_features,), 4.0 ** off))
        return torch.cat(parts)


class UniformSignal(Signal):
    """Constant scores: waterfill spreads the budget as evenly as possible."""

    dynamic = False

    def scores(self, index: QuantIndex) -> Tensor:
        return torch.ones(index.total_channels)


def make_signal(name: str, seed: int = 0, **kwargs) -> Signal:
    name = name.lower()
    if name in ("connectivity", "product"):
        return ProductSignal()
    if name == "structural":
        return StructuralSignal()
    if name == "flow":
        return FlowSignal()
    if name == "fisher":
        return FisherSignal()
    if name in ("random", "random_fixed"):
        return RandomFixedSignal(seed)
    if name == "random_churn":
        return RandomChurnSignal(seed)
    if name == "kquant":
        return KQuantSignal()
    if name == "uniform":
        return UniformSignal()
    raise ValueError(f"unknown signal: {name!r}")
