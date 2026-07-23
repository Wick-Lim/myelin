"""Quantized linear layer with per-output-channel bit-width and connectivity tracking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from myelin.bitplane import fake_quantize_rows


class BitplaneLinear(nn.Module):
    """Linear layer whose weight rows are fake-quantized to per-row bit-widths.

    The fp32 ``weight`` parameter is the shadow weight; ``bits`` (a buffer, so it
    travels with ``state_dict``) holds the current per-output-channel plane count.
    During training the layer also maintains ``act_ema``, an EMA of the mean |output|
    per channel — the "activation flow" half of the connectivity signal.

    Structural connectivity (row norms) is intentionally computed from the shadow
    weight, not the quantized one, to break the self-fulfilling loop where extra
    bits -> less noise -> looks more important -> more bits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init_bits: int = 2,
        act_momentum: float = 0.99,
        role: str = "",
        layer_idx: int = -1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.role = role
        self.layer_idx = layer_idx
        self.act_momentum = act_momentum
        self.quant_enabled = True
        self.track_activations = True

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "bits", torch.full((out_features,), init_bits, dtype=torch.long)
        )
        self.register_buffer("act_ema", torch.zeros(out_features))
        self.register_buffer("act_observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("grad_sq_ema", torch.zeros(out_features))
        self.register_buffer("grad_observations", torch.zeros((), dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        if self.quant_enabled:
            w = fake_quantize_rows(self.weight, self.bits)
        else:
            w = self.weight
        y = F.linear(x, w, self.bias)
        if self.training and self.track_activations:
            self._observe(y)
        return y

    @torch.no_grad()
    def _observe(self, y: Tensor) -> None:
        a = y.detach().reshape(-1, self.out_features).abs().mean(dim=0)
        if self.act_observations.item() == 0:
            self.act_ema.copy_(a)
        else:
            self.act_ema.lerp_(a, 1.0 - self.act_momentum)
        self.act_observations += 1

    @torch.no_grad()
    def observe_grad(self) -> None:
        """Accumulate per-channel squared weight gradients (Fisher proxy).

        Call after backward (and after clipping, so the EMA reflects the
        gradients actually applied). No-op when there is no grad.
        """
        if self.weight.grad is None:
            return
        g2 = self.weight.grad.pow(2).sum(dim=1)
        if self.grad_observations.item() == 0:
            self.grad_sq_ema.copy_(g2)
        else:
            self.grad_sq_ema.lerp_(g2, 1.0 - self.act_momentum)
        self.grad_observations += 1

    @torch.no_grad()
    def structural(self) -> Tensor:
        """Per-channel L2 norm of shadow weight rows."""
        return self.weight.detach().norm(dim=1)

    @torch.no_grad()
    def row_scale(self) -> Tensor:
        """Per-channel absmax — the actual quantization step scale."""
        return self.weight.detach().abs().amax(dim=1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, role={self.role!r}"
        )
