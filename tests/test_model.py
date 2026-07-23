import torch

from myelin.model import MiniGPT, ModelConfig


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=16, d_ff=32)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_backward():
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    x = torch.randint(0, 64, (3, 16))
    y = torch.randint(0, 64, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 64)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n


def test_lm_head_is_tied():
    model = MiniGPT(_cfg())
    with torch.no_grad():
        x = torch.randint(0, 64, (1, 4))
        logits1, _ = model(x)
        model.tok_emb.weight.mul_(2.0)
        logits2, _ = model(x)
    assert not torch.allclose(logits1, logits2)


def test_quant_layers_and_roles():
    model = MiniGPT(_cfg())
    layers = model.quant_layers()
    assert len(layers) == 2 * 6
    roles = {m.role for m in layers}
    assert roles == {"attn_q", "attn_k", "attn_v", "attn_o", "mlp_up", "mlp_down"}


def test_reference_config_param_count():
    model = MiniGPT(ModelConfig())  # defaults: d192 L4 vocab 8192
    n = model.num_params()
    assert 3_000_000 < n < 4_000_000, n


def test_weight_traffic_scales_with_bits():
    model = MiniGPT(_cfg())
    t2 = model.weight_traffic_bytes()
    for m in model.quant_layers():
        m.bits.fill_(4)
    assert abs(model.weight_traffic_bytes() - 2 * t2) < 1e-6


def test_quant_toggle_changes_output():
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    model.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits_q, _ = model(x)
        model.set_quant_enabled(False)
        logits_fp, _ = model(x)
    assert not torch.allclose(logits_q, logits_fp)
