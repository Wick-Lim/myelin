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
pub fn matvec(m: &PackedMatrix, x: &[f32]) -> Vec<f32> {
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
            let a = matvec(&m, &x);
            let b = matvec_dequant(&m, &x);
            for (i, (&va, &vb)) in a.iter().zip(&b).enumerate() {
                let tol = 1e-5 * vb.abs().max(1.0);
                assert!(
                    (va - vb).abs() <= tol,
                    "({rows}x{cols}) row {i}: bitserial {va} vs dequant {vb}"
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
