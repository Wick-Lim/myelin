//! Rough single-thread throughput probe: bit-serial vs dequant+dot, and the
//! traffic-linearity of mean bits. Run: `cargo run --release --example bench`
//!
//! Not a rigorous benchmark — it exists to watch the crossover story from
//! DESIGN.md (bit-serial wins when average bits are low) take shape.

use std::time::Instant;

use myelin_kernel::{matvec, matvec_dequant, matvec_scalar, pack};

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
}

fn main() {
    let rows = 768;
    let cols = 768;
    let iters = 200;
    let mut s = 9u64;
    let w: Vec<f32> = (0..rows * cols).map(|_| lcg_f32(&mut s) * 2.0).collect();
    let x: Vec<f32> = (0..cols).map(|_| lcg_f32(&mut s)).collect();

    println!("{rows}x{cols}, {iters} iters, single thread");
    for avg_bits in [2u8, 3, 4, 6, 8] {
        let bits = vec![avg_bits; rows];
        let m = pack(&w, rows, cols, &bits).unwrap();

        let t0 = Instant::now();
        let mut sink = 0f32;
        for _ in 0..iters {
            sink += matvec(&m, &x)[0];
        }
        let dt_simd = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            sink += matvec_scalar(&m, &x)[0];
        }
        let dt_scalar = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            sink += matvec_dequant(&m, &x)[0];
        }
        let dt_dequant = t0.elapsed().as_secs_f64() / iters as f64;

        let mb = m.plane_bytes() as f64 / 1e6;
        println!(
            "bits {avg_bits}: planes {mb:.3} MB | simd {:.3} ms ({:.2} GB/s eff) | scalar {:.3} ms | dequant+dot {:.3} ms | sink {sink:.1}",
            dt_simd * 1e3,
            mb / 1e3 / dt_simd,
            dt_scalar * 1e3,
            dt_dequant * 1e3,
        );
    }
}
