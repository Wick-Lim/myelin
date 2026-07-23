//! myelin-kernel — Phase 2: the bit-plane storage format made real.
//!
//! Phase 1 (Python) trains with fake quantization; this crate provides the
//! actual packed representation and a bit-serial GEMV whose cost is linear in
//! allocated bits. The golden-model contract (`tests/golden.rs`, vectors from
//! `scripts/gen_golden.py`) pins pack->dequantize to the Python
//! `quantize_rows` output bit-exactly; the same vectors later serve as the
//! Phase 3 RTL testbench inputs.

pub mod bitplane;
pub mod gemv;

pub use bitplane::{dequantize, pack, PackedMatrix, MAX_BITS};
pub use gemv::{matvec, matvec_dequant};

#[cfg(feature = "python")]
mod py;
