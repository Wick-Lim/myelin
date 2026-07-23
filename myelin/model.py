"""Mini GPT with bit-plane-quantized block linears.

All Linear layers inside transformer blocks (q/k/v/o, mlp up/down) are
:class:`BitplaneLinear` with per-output-channel bit-widths; embeddings, the
tied LM head, and LayerNorms stay fp32 (see DESIGN.md §5). Separate q/k/v
projections (not fused) so every quantized tensor has a single role — this is
what lets the k-quant baseline and per-role analyses address tensors cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from myelin.layers import BitplaneLinear


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 6
    d_model: int = 192
    d_ff: int = 768
    dropout: float = 0.0
    min_bits: int = 2
    max_bits: int = 8

    def __post_init__(self):
        assert self.d_model % self.n_head == 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout
        kw = dict(bias=False, init_bits=cfg.min_bits, layer_idx=layer_idx)
        self.q = BitplaneLinear(cfg.d_model, cfg.d_model, role="attn_q", **kw)
        self.k = BitplaneLinear(cfg.d_model, cfg.d_model, role="attn_k", **kw)
        self.v = BitplaneLinear(cfg.d_model, cfg.d_model, role="attn_v", **kw)
        self.o = BitplaneLinear(cfg.d_model, cfg.d_model, role="attn_o", **kw)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q = self.q(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        kw = dict(bias=False, init_bits=cfg.min_bits, layer_idx=layer_idx)
        self.up = BitplaneLinear(cfg.d_model, cfg.d_ff, role="mlp_up", **kw)
        self.down = BitplaneLinear(cfg.d_ff, cfg.d_model, role="mlp_down", **kw)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, layer_idx)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg, i) for i in range(cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # LM head is tied to tok_emb (applied functionally in forward).

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # GPT-2 style: scale residual-output projections by 1/sqrt(2L)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.o.weight.mul_(scale)
                block.mlp.down.weight.mul_(scale)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    # ------------------------------------------------------------------ #

    def quant_layers(self) -> list[BitplaneLinear]:
        return [m for m in self.modules() if isinstance(m, BitplaneLinear)]

    def set_quant_enabled(self, enabled: bool) -> None:
        for m in self.quant_layers():
            m.quant_enabled = enabled

    def set_tracking(self, enabled: bool) -> None:
        for m in self.quant_layers():
            m.track_activations = enabled

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def weight_traffic_bytes(self) -> float:
        """Bytes of quantized block-weight storage under the current allocation
        (the memory a bit-plane kernel would stream per forward pass)."""
        total_bits = 0
        for m in self.quant_layers():
            total_bits += int(m.bits.sum()) * m.in_features
        return total_bits / 8.0
