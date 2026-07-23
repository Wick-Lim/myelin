import torch

from myelin.allocator import AllocatorConfig, BitAllocator, waterfill
from myelin.layers import BitplaneLinear
from myelin.signals import QuantIndex, Signal


class StubSignal(Signal):
    def __init__(self, scores, dynamic=True):
        self._scores = scores
        self.dynamic = dynamic

    def scores(self, index):
        return self._scores


class TinyModel(torch.nn.Module):
    def __init__(self, sizes=(8, 8)):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            BitplaneLinear(4, s, role="attn_q") for s in sizes
        )


def test_waterfill_budget_and_bounds():
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(100, generator=g) + 0.01
    bits = waterfill(scores, total_planes=400, min_bits=2, max_bits=8)
    assert int(bits.sum()) == 400
    assert bits.min() >= 2 and bits.max() <= 8


def test_waterfill_log4_proportionality():
    # score ratio 16 => +2 bits at the optimum of sum(score * 4^-bits)
    bits = waterfill(torch.tensor([1.0, 16.0]), 6, min_bits=1, max_bits=8)
    assert bits.tolist() == [2, 4]


def test_waterfill_uniform_scores_spread_evenly():
    bits = waterfill(torch.ones(10), 45, min_bits=2, max_bits=8)
    assert int(bits.sum()) == 45
    assert bits.max() - bits.min() <= 1


def test_waterfill_ties_are_not_depth_biased():
    """Exact-tie remainders must not all land on the lowest flat indices
    (= earliest layers), which would silently depth-bias the static baselines."""
    n = 400
    bits = waterfill(torch.ones(n), int(4.5 * n), min_bits=2, max_bits=8)
    extras_front = int((bits[: n // 2] - 4).clamp_min(0).sum())
    extras_total = int((bits - 4).clamp_min(0).sum())
    assert extras_total == n // 2
    # index-order tie-breaking would give extras_front == 200
    assert 60 <= extras_front <= 140


def test_waterfill_monotone_in_score():
    scores = torch.tensor([0.1, 1.0, 10.0, 100.0])
    bits = waterfill(scores, 16, min_bits=1, max_bits=8)
    assert (bits[1:] >= bits[:-1]).all()


def test_waterfill_infeasible_budget_raises():
    import pytest

    with pytest.raises(ValueError):
        waterfill(torch.ones(4), 3, min_bits=2, max_bits=8)


def test_warmup_longer_than_training_raises():
    import pytest

    model = TinyModel()
    index = QuantIndex(model)
    cfg = AllocatorConfig(avg_bits=4.0, signal_warmup_steps=500)
    with pytest.raises(ValueError, match="signal_warmup_steps"):
        BitAllocator(index, cfg, StubSignal(torch.ones(16)), total_steps=400)


def _make_alloc(scores, dynamic=True, **cfg_kwargs):
    model = TinyModel()
    index = QuantIndex(model)
    defaults = dict(
        avg_bits=4.0, min_bits=2, max_bits=8,
        signal_warmup_steps=10, period=10, k_start=1.0,
        realloc_end_frac=1.0, deadband=0.25,
    )
    defaults.update(cfg_kwargs)
    cfg = AllocatorConfig(**defaults)
    alloc = BitAllocator(index, cfg, StubSignal(scores, dynamic), total_steps=100)
    return model, index, alloc


def test_initial_allocation_at_warmup_and_zero_sum():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores)
    assert alloc.maybe_update(5) is None
    event = alloc.maybe_update(10)
    assert event is not None and event["kind"] == "init"
    assert int(index.flat_bits().sum()) == alloc.total_planes


def test_reallocation_zero_sum_and_bounds():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores)
    alloc.maybe_update(10)
    # shift the scores so the target moves
    alloc.signal._scores = torch.flip(scores, dims=(0,)) * 3
    event = alloc.maybe_update(20)
    assert event is not None and event["kind"] == "realloc"
    bits = index.flat_bits()
    assert int(bits.sum()) == alloc.total_planes
    assert bits.min() >= 2 and bits.max() <= 8
    assert event["moves"] > 0


def test_deadband_blocks_near_ties():
    """The waterfill target genuinely shifts (real donors/receivers exist),
    but the score gap (1.1x) is inside the 1.25x deadband: zero moves."""
    scores = torch.ones(16)
    model, index, alloc = _make_alloc(scores, avg_bits=4.25)
    alloc.maybe_update(10)
    before = index.flat_bits().clone()

    shifted = torch.ones(16)
    shifted[12:] = 1.1  # target moves the 4 remainder planes to channels 12-15
    alloc.signal._scores = shifted
    target = waterfill(shifted, alloc.total_planes, 2, 8)
    assert not torch.equal(target, before), "test setup: target must shift"

    event = alloc.maybe_update(20)
    assert event is None or event["moves"] == 0
    assert torch.equal(index.flat_bits(), before)


def test_deadband_allows_clear_winners():
    scores = torch.ones(16)
    model, index, alloc = _make_alloc(scores, avg_bits=4.25)
    alloc.maybe_update(10)
    shifted = torch.ones(16)
    shifted[12:] = 2.0  # 2.0 > 1.25 * 1.0: moves must happen
    alloc.signal._scores = shifted
    event = alloc.maybe_update(20)
    assert event is not None and event["moves"] > 0


def test_static_signal_skips_reallocation():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores, dynamic=False)
    assert alloc.maybe_update(10)["kind"] == "init"
    assert alloc.maybe_update(20) is None


def test_freeze_after_end_frac():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores, realloc_end_frac=0.5)
    alloc.maybe_update(10)
    alloc.signal._scores = torch.flip(scores, dims=(0,))
    # step 60 >= 0.5 * 100 => frozen
    assert alloc.maybe_update(60) is None


def test_move_cap_limits_churn():
    n = 16
    scores = torch.linspace(1, 100, n)
    model, index, alloc = _make_alloc(scores, k_start=0.1)
    alloc.maybe_update(10)
    alloc.signal._scores = torch.flip(scores, dims=(0,))
    event = alloc.maybe_update(20)
    if event is not None:
        # ceil(k(20) * 16) with k(20) < k_start=0.1 => at most 2 moves
        assert event["moves"] <= 2


def test_promote_cooldown_blocks_immediate_demotion():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores, promote_cooldown_cycles=10)
    alloc.maybe_update(10)
    # cycle 1: promote some channels
    alloc.signal._scores = torch.flip(scores, dims=(0,)) * 3
    e1 = alloc.maybe_update(20)
    assert e1 is not None and e1["moves"] > 0
    promoted = set(
        torch.nonzero(alloc._last_promoted == 20).flatten().tolist()
    )
    assert promoted
    bits_after = index.flat_bits().clone()
    # cycle 2: flip scores back — promoted channels are now over target but
    # must not donate within the cooldown window
    alloc.signal._scores = scores
    alloc.maybe_update(30)
    for c in promoted:
        assert index.flat_bits()[c] >= bits_after[c]


def test_score_gamma_amplifies_contrast():
    # ratio 3 stays under the 4x threshold at gamma=1 (near-uniform bits)
    # but becomes 9x at gamma=2, which clears it decisively
    scores = torch.tensor([1.0] * 8 + [3.0] * 8)
    _, index1, alloc1 = _make_alloc(scores, score_gamma=1.0)
    alloc1.maybe_update(10)
    spread1 = int(index1.flat_bits().max() - index1.flat_bits().min())

    _, index2, alloc2 = _make_alloc(scores, score_gamma=2.0)
    alloc2.maybe_update(10)
    bits2 = index2.flat_bits().float()
    assert bits2[8:].mean() - bits2[:8].mean() >= 1.0
    assert spread1 <= 1


def test_warmup_zero_allocates_at_step_one():
    scores = torch.rand(16) + 0.01
    model, index, alloc = _make_alloc(scores, signal_warmup_steps=0)
    event = alloc.maybe_update(1)
    assert event is not None and event["kind"] == "init"
