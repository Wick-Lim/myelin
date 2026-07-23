//! Bit-serial matrix-vector product over packed planes.
//!
//! Math: with unit weights u_j = sum_i b_ij 2^-i (b in {-1,+1}),
//!
//!   y_r = s_r * sum_i 2^-i * d_i,   d_i = sum_j b_ij x_j = 2*S_i - T
//!
//! where S_i sums x_j over set bits of plane i and T = sum_j x_j once per
//! input. Per-row cost is proportional to the row's plane count — compute and
//! memory both scale with allocated bits, which is the property Phase 3
//! silicon inherits.
//!
//! This is the scalar reference kernel: correct layout, correct traffic,
//! no SIMD. The optimization path (AVX2 pshufb 4-bit LUT a la T-MAC,
//! plane-blocked tiling) replaces the inner loop without touching the format.

use crate::bitplane::{PackedMatrix, STEP};

/// y = M x. `x.len()` must equal `m.cols`.
///
/// Dispatches to the AVX2 sign-mask kernel when the CPU supports it,
/// otherwise falls back to the scalar reference path. Both compute the same
/// mathematical sum; lane-parallel accumulation may differ from the scalar
/// path by normal fp reassociation (well inside the 1e-5 contract tolerance).
pub fn matvec(m: &PackedMatrix, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), m.cols, "x length {} != cols {}", x.len(), m.cols);
    #[cfg(target_arch = "x86_64")]
    if avx2::available() {
        return avx2::matvec(m, x);
    }
    matvec_scalar(m, x)
}

/// Scalar reference path (also the portable fallback).
pub fn matvec_scalar(m: &PackedMatrix, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), m.cols, "x length {} != cols {}", x.len(), m.cols);
    let total: f64 = x.iter().map(|&v| v as f64).sum();
    let wpp = m.words_per_plane;
    let mut y = Vec::with_capacity(m.rows);

    for r in 0..m.rows {
        let base = m.row_offsets[r];
        let mut acc = 0f64;
        for i in 0..m.bits[r] as usize {
            let plane = &m.planes[base + i * wpp..base + (i + 1) * wpp];
            let mut set_sum = 0f64;
            for (w, &word) in plane.iter().enumerate() {
                let mut bits = word;
                let col_base = w * 64;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    set_sum += x[col_base + tz] as f64;
                    bits &= bits - 1;
                }
            }
            acc += STEP[i + 1] as f64 * (2.0 * set_sum - total);
        }
        y.push((m.scales[r] as f64 * acc) as f32);
    }
    y
}

/// AVX2 kernel: per plane, expand each sign byte (8 columns) into a 256-bit
/// sign mask via a 256-entry LUT (8 KB, cache-resident), apply signs with a
/// single XOR against 8 input lanes, and accumulate. Cost per plane is
/// cols/8 xor+add regardless of bit pattern — unlike the scalar path whose
/// cost tracks popcount. Zero-padding of x beyond `cols` is harmless: XOR
/// only flips the sign of 0.0, and +-0.0 is additive identity.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    use crate::bitplane::{PackedMatrix, STEP};

    /// sign masks per byte value: bit set (=+1) -> 0, bit clear (=-1) -> sign flip
    static SIGN_LUT: OnceLock<Box<[[i32; 8]; 256]>> = OnceLock::new();

    fn sign_lut() -> &'static [[i32; 8]; 256] {
        SIGN_LUT.get_or_init(|| {
            let mut t = Box::new([[0i32; 8]; 256]);
            for (b, entry) in t.iter_mut().enumerate() {
                for (k, lane) in entry.iter_mut().enumerate() {
                    *lane = if b >> k & 1 == 1 { 0 } else { i32::MIN };
                }
            }
            t
        })
    }

    pub fn available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    pub fn matvec(m: &PackedMatrix, x: &[f32]) -> Vec<f32> {
        let mut xp = vec![0f32; m.words_per_plane * 64];
        xp[..m.cols].copy_from_slice(x);
        // SAFETY: gated on runtime avx2 detection in `available()`
        unsafe { matvec_inner(m, &xp) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn matvec_inner(m: &PackedMatrix, xp: &[f32]) -> Vec<f32> {
        let lut = sign_lut();
        let wpp = m.words_per_plane;
        let mut y = Vec::with_capacity(m.rows);
        for r in 0..m.rows {
            let base = m.row_offsets[r];
            let mut acc_row = 0f64;
            for i in 0..m.bits[r] as usize {
                let plane = &m.planes[base + i * wpp..base + (i + 1) * wpp];
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                for (w, &word) in plane.iter().enumerate() {
                    let col = w * 64;
                    let mut k = 0usize;
                    while k < 8 {
                        let s0 = _mm256_loadu_si256(
                            lut[(word >> (8 * k)) as usize & 0xFF].as_ptr() as *const __m256i,
                        );
                        let x0 = _mm256_loadu_ps(xp.as_ptr().add(col + 8 * k));
                        acc0 = _mm256_add_ps(acc0, _mm256_xor_ps(x0, _mm256_castsi256_ps(s0)));
                        let s1 = _mm256_loadu_si256(
                            lut[(word >> (8 * (k + 1))) as usize & 0xFF].as_ptr() as *const __m256i,
                        );
                        let x1 = _mm256_loadu_ps(xp.as_ptr().add(col + 8 * (k + 1)));
                        acc1 = _mm256_add_ps(acc1, _mm256_xor_ps(x1, _mm256_castsi256_ps(s1)));
                        k += 2;
                    }
                }
                let d = hsum_f64(acc0) + hsum_f64(acc1);
                acc_row += STEP[i + 1] as f64 * d;
            }
            y.push((m.scales[r] as f64 * acc_row) as f32);
        }
        y
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_f64(v: __m256) -> f64 {
        let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
        let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        let s = _mm256_add_pd(lo, hi);
        let mut out = [0f64; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), s);
        out[0] + out[1] + out[2] + out[3]
    }
}

/// Reference path: dequantize the row, then a plain f64 dot product.
/// Exists to cross-check `matvec` and as the baseline in benchmarks.
pub fn matvec_dequant(m: &PackedMatrix, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), m.cols);
    let q = crate::bitplane::dequantize(m);
    (0..m.rows)
        .map(|r| {
            q[r * m.cols..(r + 1) * m.cols]
                .iter()
                .zip(x)
                .map(|(&w, &v)| w as f64 * v as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitplane::pack;

    fn lcg_f32(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }

    #[test]
    fn bitserial_matches_dequant_reference() {
        let mut s = 42u64;
        for &(rows, cols) in &[(8usize, 64usize), (16, 100), (5, 1), (3, 65), (7, 128)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| lcg_f32(&mut s) * 2.0).collect();
            let bits: Vec<u8> = (0..rows).map(|r| (r % 8) as u8 + 1).collect();
            let x: Vec<f32> = (0..cols).map(|_| lcg_f32(&mut s)).collect();
            let m = pack(&w, rows, cols, &bits).unwrap();
            let b = matvec_dequant(&m, &x);
            for (label, a) in [("dispatch", matvec(&m, &x)), ("scalar", matvec_scalar(&m, &x))] {
                for (i, (&va, &vb)) in a.iter().zip(&b).enumerate() {
                    let tol = 1e-5 * vb.abs().max(1.0);
                    assert!(
                        (va - vb).abs() <= tol,
                        "({rows}x{cols}) {label} row {i}: {va} vs dequant {vb}"
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_and_scalar_agree_on_awkward_shapes() {
        // shapes chosen to stress word padding (cols % 64 != 0) and tiny sizes
        let mut s = 5u64;
        for &(rows, cols) in &[
            (1usize, 1usize),
            (2, 7),
            (3, 63),
            (3, 64),
            (3, 65),
            (4, 100),
            (2, 191),
            (1, 1024),
        ] {
            let w: Vec<f32> = (0..rows * cols).map(|_| lcg_f32(&mut s) * 3.0).collect();
            let bits: Vec<u8> = (0..rows).map(|r| (r % 8) as u8 + 1).collect();
            let x: Vec<f32> = (0..cols).map(|_| lcg_f32(&mut s)).collect();
            let m = pack(&w, rows, cols, &bits).unwrap();
            let a = matvec(&m, &x);
            let b = matvec_scalar(&m, &x);
            for (i, (&va, &vb)) in a.iter().zip(&b).enumerate() {
                let tol = 1e-5 * vb.abs().max(1.0);
                assert!(
                    (va - vb).abs() <= tol,
                    "({rows}x{cols}) row {i}: dispatch {va} vs scalar {vb}"
                );
            }
        }
    }

    #[test]
    fn single_plane_is_sign_matvec() {
        // 1-bit rows: y = s * 0.5 * sum(sign(w_j) * x_j)
        let w = vec![1.0f32, -2.0, 3.0, -4.0];
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let m = pack(&w, 1, 4, &[1]).unwrap();
        let y = matvec(&m, &x);
        // scale 4.0, signs (+,-,+,-) -> 4.0 * 0.5 * (1-1+1-1) = 0
        assert!((y[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "x length")]
    fn wrong_x_length_panics() {
        let m = pack(&[1.0f32; 8], 2, 4, &[2, 2]).unwrap();
        matvec(&m, &[1.0; 5]);
    }
}
