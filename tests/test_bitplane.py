import torch

from myelin.bitplane import (
    fake_quantize_rows,
    plane_decompose,
    plane_reconstruct,
    quantize_rows,
    quantize_unit,
)

MAX_BITS = 8


def _random_x(n_rows, n_cols, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_rows, n_cols, generator=g) * 2 - 1


def _boundary_x(k, ulps=16):
    """All k-bit grid boundaries in [-1, 1] plus ladders of `ulps` values
    immediately below and above each — the exact window where a residual-form
    decomposition diverges from the closed form."""
    delta = 2.0 ** (1 - k)
    grid = torch.arange(-2 ** (k - 1), 2 ** (k - 1) + 1, dtype=torch.float32) * delta
    xs = [grid]
    lo, hi = torch.full_like(grid, -2.0), torch.full_like(grid, 2.0)
    below, above = grid.clone(), grid.clone()
    for _ in range(ulps):
        below = torch.nextafter(below, lo)
        above = torch.nextafter(above, hi)
        xs.extend([below.clone(), above.clone()])
    x = torch.cat(xs)
    return x.clamp(-1.0, 1.0).unsqueeze(0)


def test_closed_form_equals_plane_reconstruction_bit_exact():
    for k in range(1, MAX_BITS + 1):
        x = _random_x(16, 512, seed=k)
        planes = plane_decompose(x, k)
        ref = plane_reconstruct(planes, k)
        bits = torch.full((x.shape[0],), k, dtype=torch.long)
        got = quantize_unit(x, bits)
        assert torch.equal(got, ref), f"mismatch at k={k}"


def test_closed_form_equals_planes_at_grid_boundaries():
    # regression for the round-half-even collapse: the residual-form
    # decomposition disagreed with the closed form in ulp-windows below
    # boundaries; the running-sum form must agree bit-exactly everywhere
    for k in range(1, MAX_BITS + 1):
        x = _boundary_x(k)
        planes = plane_decompose(x, k)
        ref = plane_reconstruct(planes, k)
        got = quantize_unit(x, torch.tensor([k]))
        assert torch.equal(got, ref), f"boundary mismatch at k={k}"


def test_reviewer_counterexample_0x3e7fffff():
    # u = 0.25 - 2^-26: residual-form decomposition yielded 0.375; the
    # closed form (and the fixed decomposition) yield 0.125
    u = torch.tensor([0x3E7FFFFF], dtype=torch.int32).view(torch.float32)
    x = torch.stack([u, torch.tensor([1.0])], dim=1)  # (1, 2) row
    q = quantize_rows(x, torch.tensor([3]))
    assert torch.equal(q, torch.tensor([[0.125, 0.875]]))
    planes = plane_decompose(x, 3)
    assert torch.equal(plane_reconstruct(planes, 3), torch.tensor([[0.125, 0.875]]))


def test_per_row_bits_in_plane_reconstruction():
    x = _random_x(8, 256, seed=42)
    planes = plane_decompose(x, MAX_BITS)
    bits = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    got = quantize_unit(x, bits)
    ref = plane_reconstruct(planes, bits)
    assert torch.equal(got, ref)


def test_nesting_one_more_plane_refines_by_half_step():
    x = torch.rand(4, 1000) * 2 - 1
    for k in range(1, MAX_BITS):
        b_k = torch.full((4,), k, dtype=torch.long)
        b_k1 = torch.full((4,), k + 1, dtype=torch.long)
        diff = quantize_unit(x, b_k1) - quantize_unit(x, b_k)
        assert torch.allclose(diff.abs(), torch.full_like(diff, 2.0 ** -(k + 1)), atol=1e-6)


def test_error_bound():
    x = torch.rand(4, 10_000) * 2 - 1
    for k in range(1, MAX_BITS + 1):
        bits = torch.full((4,), k, dtype=torch.long)
        err = (quantize_unit(x, bits) - x).abs().max()
        assert err <= 2.0 ** -k + 1e-6, f"error bound violated at k={k}"


def test_endpoints_clamp():
    x = torch.tensor([[-1.0, 1.0]])
    for k in range(1, MAX_BITS + 1):
        bits = torch.tensor([k])
        q = quantize_unit(x, bits)
        expected = 1.0 - 2.0 ** -k
        assert torch.allclose(q, torch.tensor([[-expected, expected]]), atol=1e-7)


def test_plane_values_are_signs():
    x = torch.rand(4, 100) * 2 - 1
    planes = plane_decompose(x, MAX_BITS)
    assert set(planes.unique().tolist()) <= {-1.0, 1.0}


def test_quantize_rows_scale_and_zero_row():
    w = torch.randn(3, 64)
    w[2] = 0.0
    bits = torch.tensor([2, 8, 4])
    q = quantize_rows(w, bits)
    assert torch.isfinite(q).all()
    # per-row max error <= scale * 2^-bits
    scale = w.abs().amax(dim=1)
    err = (q - w).abs().amax(dim=1)
    assert (err <= scale * 2.0 ** -bits.float() + 1e-6).all()


def test_fake_quant_ste_gradient_is_identity():
    w = torch.randn(5, 16, requires_grad=True)
    bits = torch.tensor([2, 3, 4, 5, 6])
    y = fake_quantize_rows(w, bits)
    y.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_fake_quant_forward_is_bit_exactly_quantize_rows():
    # bit-exact, not just allclose: this is the golden-model contract —
    # the trained forward must be the exact value a bit-plane kernel computes
    g = torch.Generator().manual_seed(3)
    w = torch.randn(64, 128, generator=g) * torch.logspace(-8, 8, 64, base=10).unsqueeze(1)
    bits = torch.randint(1, 9, (64,), generator=g)
    wq = w.clone().requires_grad_(True)
    assert torch.equal(fake_quantize_rows(wq, bits).detach(), quantize_rows(w, bits))


def test_more_bits_never_increase_row_error():
    w = torch.randn(1, 512)
    prev = None
    for k in range(1, MAX_BITS + 1):
        err = (quantize_rows(w, torch.tensor([k])) - w).norm()
        if prev is not None:
            assert err <= prev + 1e-6
        prev = err
