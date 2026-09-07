//! Prism GGUF PQ2_0 (g128): 128 two-bit weights with one f16 scale.
//! Codes 0/1/2/3 map to -1/0/+1/+2 times the scale. Dots use four Q8_0
//! activation blocks per weight block; weights always stay packed.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::q8::{BlockQ8_0, QK8_0};

pub const PQK2_0: usize = 128;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockPQ2_0 {
    pub d: u16,
    pub qs: [u8; PQK2_0 / 4],
}

const _: () = assert!(core::mem::size_of::<BlockPQ2_0>() == 34);

pub fn cast_blocks(bytes: &[u8]) -> &[BlockPQ2_0] {
    assert_eq!(bytes.len() % core::mem::size_of::<BlockPQ2_0>(), 0);
    // SAFETY: packed, alignment 1, and every byte pattern is valid.
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr().cast(),
            bytes.len() / core::mem::size_of::<BlockPQ2_0>(),
        )
    }
}

/// Q8_0 activations for the PQ2_0 kernels. Match ggml's SIMD quantizer:
/// quantize against the original f32 maximum, then store an f16 scale.
/// The older Q8-weight path intentionally keeps its existing numerics.
pub fn quantize_activations(src: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(src.len(), out.len() * QK8_0);
    for (block, chunk) in out.iter_mut().zip(src.as_chunks::<QK8_0>().0) {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        block.d = f32_to_f16(amax / 127.0);
        let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
        for (q, &v) in block.qs.iter_mut().zip(chunk) {
            *q = libm::rintf(v * inv) as i8;
        }
    }
}

pub fn dequantize(blocks: &[BlockPQ2_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * PQK2_0);
    for (block, row) in blocks.iter().zip(out.as_chunks_mut::<PQK2_0>().0) {
        let d = f16_to_f32(block.d);
        for (i, value) in row.iter_mut().enumerate() {
            let q = (block.qs[i / 4] >> (2 * (i % 4))) & 3;
            *value = (q as i32 - 1) as f32 * d;
        }
    }
}

/// Reference accumulation order matches ggml_vec_dot_pq2_0_q8_0_generic.
pub fn dot_scalar(w: &[BlockPQ2_0], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), w.len() * 4);
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let mut acc = 0i32;
            for (i, &q) in bx.qs.iter().enumerate() {
                let code = (bw.qs[k * 8 + i / 4] >> (2 * (i % 4))) & 3;
                acc += (code as i32 - 1) * q as i32;
            }
            sub += bx.scale() * acc as f32;
        }
        sum += f16_to_f32(bw.d) * sub;
    }
    sum
}

/// # Safety
/// AVX2 must be available and YMM state enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_avx2(w: &[BlockPQ2_0], x: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), w.len() * 4);
    let mask = _mm_set1_epi8(3);
    let ones = _mm256_set1_epi8(1);
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let packed = _mm_loadl_epi64(bw.qs.as_ptr().add(k * 8).cast());
            let q0 = _mm_and_si128(packed, mask);
            let q1 = _mm_and_si128(_mm_srli_epi16(packed, 2), mask);
            let q2 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let q3 = _mm_and_si128(_mm_srli_epi16(packed, 6), mask);
            let q01 = _mm_unpacklo_epi8(q0, q1);
            let q23 = _mm_unpacklo_epi8(q2, q3);
            let codes =
                _mm256_set_m128i(_mm_unpackhi_epi16(q01, q23), _mm_unpacklo_epi16(q01, q23));
            let q = _mm256_loadu_si256(bx.qs.as_ptr().cast());
            // sum(code * activation) - sum(activation). Widened pair
            // sums also handle reserved code 3 and activation -128.
            let pairs = _mm256_sub_epi16(
                _mm256_maddubs_epi16(codes, q),
                _mm256_maddubs_epi16(ones, q),
            );
            let acc = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
            let half = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256(acc, 1),
            );
            let half = _mm_hadd_epi32(half, half);
            let total = _mm_cvtsi128_si32(_mm_hadd_epi32(half, half));
            sub += bx.scale() * total as f32;
        }
        sum += f16_to_f32(bw.d) * sub;
    }
    sum
}

/// # Safety
/// NEON must be available (baseline on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_neon(w: &[BlockPQ2_0], x: &[BlockQ8_0]) -> f32 {
    use core::arch::aarch64::*;
    assert_eq!(x.len(), w.len() * 4);
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let mut acc = vdupq_n_s16(0);
            for b in 0..4 {
                let packed = bw.qs[k * 8 + b * 2] as u16 | ((bw.qs[k * 8 + b * 2 + 1] as u16) << 8);
                let codes: [i8; 8] = core::array::from_fn(|i| ((packed >> (2 * i)) & 3) as i8 - 1);
                acc = vmlal_s8(
                    acc,
                    vld1_s8(bx.qs.as_ptr().add(b * 8)),
                    vld1_s8(codes.as_ptr()),
                );
            }
            sub += bx.scale() * vaddlvq_s16(acc) as f32;
        }
        sum += f16_to_f32(bw.d) * sub;
    }
    sum
}

fn dot_kernel() -> unsafe fn(&[BlockPQ2_0], &[BlockQ8_0]) -> f32 {
    if crate::cpu::fast_path() {
        #[cfg(target_arch = "x86_64")]
        return dot_avx2;
        #[cfg(target_arch = "aarch64")]
        return dot_neon;
    }
    dot_scalar
}

struct SendPtr(*mut f32);
// SAFETY: used only for disjoint worker-owned output rows; POOL.run joins.
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

pub fn matvec(y: &mut [f32], w: &[BlockPQ2_0], x: &[BlockQ8_0], rows: usize, cols: usize) {
    matmul(y, w, x, rows, cols, 1);
}

/// Row-major output [batch][rows], activations [batch][cols/32]. Reuses
/// each packed weight row across the batch with the same decode dot.
pub fn matmul(
    y: &mut [f32],
    w: &[BlockPQ2_0],
    xs: &[BlockQ8_0],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    assert_eq!(cols % PQK2_0, 0);
    assert_eq!(y.len(), batch * rows);
    let bpr = cols / PQK2_0;
    let xpr = cols / QK8_0;
    assert_eq!(w.len(), rows * bpr);
    assert_eq!(xs.len(), batch * xpr);
    let dot = dot_kernel();
    let yp = SendPtr(y.as_mut_ptr());
    crate::parallel::POOL.run(rows, &|r0, r1| {
        for r in r0..r1 {
            let row = &w[r * bpr..(r + 1) * bpr];
            for b in 0..batch {
                // SAFETY: CPU checked above; each worker owns disjoint
                // rows, and each (batch, row) is written exactly once.
                unsafe {
                    yp.get()
                        .add(b * rows + r)
                        .write(dot(row, &xs[b * xpr..(b + 1) * xpr]));
                }
            }
        }
    });
}
