"""Myelin: connectivity-aware dynamic precision allocation for transformer training."""

from myelin.bitplane import (
    plane_decompose,
    plane_reconstruct,
    quantize_unit,
    quantize_rows,
    fake_quantize_rows,
)
from myelin.layers import BitplaneLinear
from myelin.allocator import AllocatorConfig, BitAllocator, waterfill
from myelin.model import ModelConfig, MiniGPT
from myelin.signals import make_signal, QuantIndex

__all__ = [
    "plane_decompose",
    "plane_reconstruct",
    "quantize_unit",
    "quantize_rows",
    "fake_quantize_rows",
    "BitplaneLinear",
    "AllocatorConfig",
    "BitAllocator",
    "waterfill",
    "ModelConfig",
    "MiniGPT",
    "make_signal",
    "QuantIndex",
]
