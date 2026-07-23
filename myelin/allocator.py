"""Zero-sum bit budget allocation.

Bit assignment is classic rate–distortion waterfilling: quantizing a channel
with score ``s`` at ``b`` bits costs roughly ``s * 4^{-b}`` (squared error
shrinks 4x per plane), so the marginal benefit of channel ``c``'s next plane is
``score_c * 4^{-bits_c}``. Greedily granting planes to the highest marginal
benefit yields the budget-constrained optimum for this cost model, and gives
the intuitive law: bits grow with log4(score) — a 4x more "connected" channel
earns one more plane.

Reallocation moves planes from over-allocated channels (current > target) to
under-allocated ones, capped per cycle by a cosine-decayed fraction (RigL-style
explore-then-freeze) and gated by a deadband so near-tied channels don't swap
planes every cycle. The total plane count is invariant (zero-sum).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from myelin.signals import QuantIndex, Signal


@dataclass
class AllocatorConfig:
    avg_bits: float = 4.0
    min_bits: int = 2
    max_bits: int = 8
    #: steps of uniform-min_bits training to gather EMA signal before the
    #: initial allocation
    signal_warmup_steps: int = 500
    #: steps between reallocation cycles (after the initial allocation)
    period: int = 500
    #: initial per-cycle move cap, as a fraction of total channels
    k_start: float = 0.2
    #: fraction of total training steps after which allocation freezes
    #: (RigL evidence favors early freeze, ~0.75)
    realloc_end_frac: float = 0.75
    #: receiver score must exceed donor score by this relative margin
    deadband: float = 0.25
    #: channels that received a plane may not donate for this many cycles
    #: (ITOP reliable-exploration condition; initial allocation doesn't count)
    promote_cooldown_cycles: int = 1
    #: contrast exponent applied to scores before waterfilling: the bit offset
    #: becomes gamma * log4(score). 1.0 = the plain s*4^-b cost model; raise it
    #: when the signal's within-layer ratios are too narrow to clear the 4x
    #: threshold for +-1 bit and allocation degenerates to uniform.
    score_gamma: float = 1.0


def waterfill(
    scores: Tensor, total_planes: int, min_bits: int, max_bits: int
) -> Tensor:
    """Optimal bit vector for the ``score * 4^{-bits}`` cost model.

    Deterministic. Ties in marginal benefit resolve by a fixed seeded
    permutation, NOT by channel index: flat index order equals network depth,
    so index tie-breaking would hand every fractional remainder to the earliest
    layers — silently depth-biasing exact-tie signals (uniform, kquant) in a
    way seed-averaging cannot wash out.

    Args:
        scores: (N,) non-negative scores.
        total_planes: exact total sum(bits) to allocate.
        min_bits / max_bits: per-channel clamp.

    Returns:
        (N,) long tensor with sum == total_planes.
    """
    n = scores.numel()
    if not (min_bits * n <= total_planes <= max_bits * n):
        raise ValueError(
            f"budget {total_planes} infeasible for {n} channels in "
            f"[{min_bits}, {max_bits}]"
        )
    s = scores.double().clamp_min(1e-30)
    bits = torch.full((n,), min_bits, dtype=torch.long)
    remaining = total_planes - min_bits * n
    if remaining == 0:
        return bits
    tie = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    # heap of (-marginal_benefit, tie_rank, index); benefit = s * 4^-bits
    heap = [(-(s[i].item() * 4.0 ** -min_bits), tie[i], i) for i in range(n)]
    heapq.heapify(heap)
    while remaining > 0:
        neg_b, t, i = heapq.heappop(heap)
        bits[i] += 1
        remaining -= 1
        if bits[i] < max_bits:
            heapq.heappush(
                heap, (-(s[i].item() * 4.0 ** -bits[i].item()), t, i)
            )
    return bits


class BitAllocator:
    """Owns the global plane budget across all quantized layers."""

    def __init__(
        self,
        index: QuantIndex,
        cfg: AllocatorConfig,
        signal: Signal,
        total_steps: int,
    ):
        self.index = index
        self.cfg = cfg
        self.signal = signal
        self.total_steps = total_steps
        self.n = index.total_channels
        self.total_planes = round(cfg.avg_bits * self.n)
        if not (cfg.min_bits * self.n <= self.total_planes <= cfg.max_bits * self.n):
            raise ValueError("avg_bits outside [min_bits, max_bits]")
        if max(1, cfg.signal_warmup_steps) > total_steps:
            raise ValueError(
                f"signal_warmup_steps {cfg.signal_warmup_steps} > total_steps "
                f"{total_steps}: the bit budget would never be allocated"
            )
        self.allocated = False
        self._init_step = -1
        # step at which each channel last received a plane (realloc only)
        self._last_promoted = torch.full((self.n,), -(10 ** 9), dtype=torch.long)

    # ------------------------------------------------------------------ #

    def move_fraction(self, step: int) -> float:
        """Cosine-decayed per-cycle move cap (fraction of channels)."""
        t_end = self.cfg.realloc_end_frac * self.total_steps
        if t_end <= 0 or step >= t_end:
            return 0.0
        return 0.5 * self.cfg.k_start * (1.0 + math.cos(math.pi * step / t_end))

    def maybe_update(self, step: int) -> dict | None:
        """Call once per training step (with the 1-based completed step count).

        Returns an event dict when an (re)allocation happened, else None.
        """
        if not self.allocated:
            if step >= max(1, self.cfg.signal_warmup_steps):
                self._init_step = step
                return self.initial_allocate(step)
            return None
        if not self.signal.dynamic:
            return None
        if (step - self._init_step) % self.cfg.period != 0 or step == self._init_step:
            return None
        if self.move_fraction(step) <= 0.0:
            return None
        return self.reallocate(step)

    def _scores(self) -> Tensor:
        s = self.signal.scores(self.index).clamp_min(1e-12)
        if self.cfg.score_gamma != 1.0:
            s = s.pow(self.cfg.score_gamma)
        return s

    def initial_allocate(self, step: int) -> dict:
        scores = self._scores()
        bits = waterfill(
            scores, self.total_planes, self.cfg.min_bits, self.cfg.max_bits
        )
        self.index.set_flat_bits(bits)
        self.allocated = True
        return self._event(step, "init", moves=int((bits != self.cfg.min_bits).sum()))

    def reallocate(self, step: int) -> dict:
        cfg = self.cfg
        scores = self._scores()
        cur = self.index.flat_bits().clone()
        assert int(cur.sum()) == self.total_planes, "zero-sum invariant broken"
        target = waterfill(scores, self.total_planes, cfg.min_bits, cfg.max_bits)

        surplus = (cur - target).clamp_min(0)
        deficit = (target - cur).clamp_min(0)
        cooldown = cfg.promote_cooldown_cycles * cfg.period
        may_donate = (step - self._last_promoted) > cooldown
        donors = torch.nonzero((surplus > 0) & may_donate, as_tuple=False).flatten()
        recvs = torch.nonzero(deficit > 0, as_tuple=False).flatten()
        # donors: lowest score first; receivers: highest score first
        donors = donors[torch.argsort(scores[donors], stable=True)]
        recvs = recvs[torch.argsort(scores[recvs], stable=True, descending=True)]

        max_moves = math.ceil(self.move_fraction(step) * self.n)
        moves = 0
        di = 0
        d_left = surplus[donors].tolist() if donors.numel() else []
        stop = False
        for r in recvs.tolist():
            if stop or moves >= max_moves or di >= len(d_left):
                break
            need = int(deficit[r])
            while need > 0 and moves < max_moves and di < len(d_left):
                d = int(donors[di])
                # Deadband: the receiver must clearly out-score the donor.
                # Donors are sorted ascending and receivers descending, so if
                # the check fails for the current (best) pair it fails for all
                # remaining pairs -> stop entirely.
                if scores[r] < (1.0 + cfg.deadband) * scores[d]:
                    stop = True
                    break
                cur[d] -= 1
                cur[r] += 1
                self._last_promoted[r] = step
                d_left[di] -= 1
                need -= 1
                moves += 1
                if d_left[di] == 0:
                    di += 1

        assert int(cur.sum()) == self.total_planes, "zero-sum invariant broken"
        assert int(cur.min()) >= cfg.min_bits and int(cur.max()) <= cfg.max_bits
        self.index.set_flat_bits(cur)
        return self._event(step, "realloc", moves=moves)

    # ------------------------------------------------------------------ #

    def _event(self, step: int, kind: str, moves: int) -> dict:
        bits = self.index.flat_bits()
        per_layer = []
        for (name, m), b in zip(self.index.entries, self.index.split(bits)):
            hist = torch.bincount(b, minlength=self.cfg.max_bits + 1).tolist()
            per_layer.append(
                {
                    "layer": name,
                    "role": m.role,
                    "mean_bits": float(b.float().mean()),
                    "hist": hist,
                    "bits": b.tolist(),
                }
            )
        return {
            "step": step,
            "kind": kind,
            "moves": moves,
            "mean_bits": float(bits.float().mean()),
            "move_fraction": self.move_fraction(step),
            "layers": per_layer,
        }
