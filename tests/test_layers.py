import io

import torch

from myelin.layers import BitplaneLinear


def test_forward_shape_and_quantization_applied():
    torch.manual_seed(0)
    layer = BitplaneLinear(16, 8, init_bits=2)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    layer.quant_enabled = False
    y_fp = layer(x)
    assert not torch.allclose(y, y_fp), "2-bit quantization should change outputs"


def test_bits_travel_with_state_dict():
    layer = BitplaneLinear(16, 8)
    layer.bits.copy_(torch.tensor([2, 3, 4, 5, 6, 7, 8, 2]))
    buf = io.BytesIO()
    torch.save(layer.state_dict(), buf)
    buf.seek(0)
    fresh = BitplaneLinear(16, 8)
    fresh.load_state_dict(torch.load(buf, weights_only=True))
    assert fresh.bits.tolist() == [2, 3, 4, 5, 6, 7, 8, 2]


def test_act_ema_updates_only_in_training():
    torch.manual_seed(0)
    layer = BitplaneLinear(16, 8)
    x = torch.randn(4, 16)

    layer.eval()
    layer(x)
    assert layer.act_observations.item() == 0
    assert torch.all(layer.act_ema == 0)

    layer.train()
    layer(x)
    assert layer.act_observations.item() == 1
    assert torch.all(layer.act_ema >= 0)
    first = layer.act_ema.clone()
    layer(torch.randn(4, 16))
    assert layer.act_observations.item() == 2
    assert not torch.allclose(layer.act_ema, first)


def test_act_ema_has_no_grad_side_effects():
    layer = BitplaneLinear(16, 8)
    layer.train()
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert not layer.act_ema.requires_grad


def test_structural_is_shadow_row_norm():
    layer = BitplaneLinear(16, 8)
    expected = layer.weight.detach().norm(dim=1)
    assert torch.allclose(layer.structural(), expected)


def test_3d_input_tracking():
    layer = BitplaneLinear(16, 8)
    layer.train()
    y = layer(torch.randn(2, 5, 16))
    assert y.shape == (2, 5, 8)
    assert layer.act_observations.item() == 1
