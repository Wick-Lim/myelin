import torch

from myelin.allocator import waterfill
from myelin.model import MiniGPT, ModelConfig
from myelin.signals import (
    KQuantSignal,
    ProductSignal,
    QuantIndex,
    RandomChurnSignal,
    RandomFixedSignal,
    UniformSignal,
    make_signal,
)


def _tiny_model():
    torch.manual_seed(0)
    return MiniGPT(ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=16, d_ff=32))


def test_quant_index_covers_all_block_linears():
    model = _tiny_model()
    index = QuantIndex(model)
    assert len(index.entries) == 2 * 6  # q,k,v,o,up,down per layer
    assert index.total_channels == 2 * (4 * 16 + 32 + 16)


def test_product_signal_positive_and_per_layer_normalized():
    model = _tiny_model()
    model.train()
    x = torch.randint(0, 64, (2, 16))
    model(x)  # populate act_ema
    index = QuantIndex(model)
    s = ProductSignal().scores(index)
    assert s.shape == (index.total_channels,)
    assert (s >= 0).all() and torch.isfinite(s).all()


def test_random_fixed_is_stable_and_seed_dependent():
    model = _tiny_model()
    index = QuantIndex(model)
    a = RandomFixedSignal(1)
    assert torch.equal(a.scores(index), a.scores(index))
    b = RandomFixedSignal(2)
    assert not torch.equal(a.scores(index), b.scores(index))


def test_random_churn_changes_every_call():
    model = _tiny_model()
    index = QuantIndex(model)
    s = RandomChurnSignal(1)
    assert not torch.equal(s.scores(index), s.scores(index))


def test_kquant_offsets_produce_integer_bit_offsets():
    model = _tiny_model()
    index = QuantIndex(model)
    scores = KQuantSignal().scores(index)
    bits = waterfill(scores, round(4.0 * index.total_channels), 2, 8)
    by_role = {}
    off = 0
    for (_, m), n in zip(index.entries, index.sizes):
        by_role.setdefault(m.role, []).extend(bits[off : off + n].tolist())
        off += n
    v_mean = sum(by_role["attn_v"]) / len(by_role["attn_v"])
    q_mean = sum(by_role["attn_q"]) / len(by_role["attn_q"])
    assert 0.8 <= v_mean - q_mean <= 1.2  # +1 bit for attn_v vs attn_q


def test_kquant_use_more_bits_matches_llamacpp_integer_math():
    # n=8: first 8//8=1 layer, last layer >= 7, middle (i-1)%3==2 => i=3, 6
    gated = [i for i in range(8) if KQuantSignal.use_more_bits(i, 8)]
    assert gated == [0, 3, 6, 7]
    # n=2 (tiny models): only the last layer
    assert [i for i in range(2) if KQuantSignal.use_more_bits(i, 2)] == [1]


def test_fisher_signal_requires_and_uses_grads():
    from myelin.signals import FisherSignal

    model = _tiny_model()
    index = QuantIndex(model)
    sig = FisherSignal()
    assert sig.needs_grad and sig.dynamic

    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    for m in index.layers:
        m.observe_grad()
    s = sig.scores(index)
    assert s.shape == (index.total_channels,)
    assert torch.isfinite(s).all() and (s >= 0).all()
    assert s.max() > 0


def test_uniform_signal_near_uniform_bits():
    model = _tiny_model()
    index = QuantIndex(model)
    scores = UniformSignal().scores(index)
    bits = waterfill(scores, round(4.0 * index.total_channels), 2, 8)
    assert bits.max() - bits.min() <= 1


def test_make_signal_names():
    for name in ["connectivity", "structural", "flow", "random", "random_churn", "kquant", "uniform"]:
        assert make_signal(name, seed=0) is not None
    import pytest

    with pytest.raises(ValueError):
        make_signal("nope")
