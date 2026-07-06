//! Matrix-vector kernels over Q8_0 weights with Q8_0-quantized activations
//! (the llama.cpp `q8_0 x q8_0` scheme): integer dot products per 32-block,
//! scaled by the product of the two f16 block scales.

use crate::cpu;
use crate::q8::{BlockQ8_0, QK8_0};

/// y[r] = dot(w[r, :], x) for row-major Q8_0 `w` (rows x cols) and
/// Q8_0-quantized activation `x` (cols values).
pub fn matvec_q8(y: &mut [f32], w: &[BlockQ8_0], x: &[BlockQ8_0], rows: usize, cols: usize) {
    let bpr = cols / QK8_0;
    assert_eq!(y.len(), rows);
    assert_eq!(x.len(), bpr);
    assert!(w.len() >= rows * bpr);
    matvec_q8_range(y, w, x, 0, rows, bpr);
}

/// Row-range variant used to split work across cores.
pub fn matvec_q8_range(
    y: &mut [f32],
    w: &[BlockQ8_0],
    x: &[BlockQ8_0],
    row0: usize,
    row1: usize,
    blocks_per_row: usize,
) {
    if cpu::fast_path() {
        for r in row0..row1 {
            let row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            // SAFETY: fast_path() verified AVX2+FMA and enabled YMM state.
            y[r - row0] = unsafe { dot_q8_avx2(row, x) };
        }
    } else {
        for r in row0..row1 {
            let row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            y[r - row0] = dot_q8_scalar(row, x);
        }
    }
}

pub fn dot_q8_scalar(w: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
    let mut sum = 0f32;
    for (bw, bx) in w.iter().zip(x.iter()) {
        let mut acc = 0i32;
        for i in 0..QK8_0 {
            acc += bw.qs[i] as i32 * bx.qs[i] as i32;
        }
        sum += acc as f32 * bw.scale() * bx.scale();
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_avx2(w: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    for (bw, bx) in w.iter().zip(x.iter()) {
        let d = _mm256_set1_ps(bw.scale() * bx.scale());
        let qw = _mm256_loadu_si256(bw.qs.as_ptr() as *const __m256i);
        let qx = _mm256_loadu_si256(bx.qs.as_ptr() as *const __m256i);
        // maddubs needs one unsigned operand: use |w| and transfer w's sign
        // onto x. Pair sums stay within i16 (2 * 127*127 < 32768).
        let abs_w = _mm256_sign_epi8(qw, qw);
        let sgn_x = _mm256_sign_epi8(qx, qw);
        let p16 = _mm256_maddubs_epi16(abs_w, sgn_x);
        let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc);
    }
    // Horizontal sum.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}
