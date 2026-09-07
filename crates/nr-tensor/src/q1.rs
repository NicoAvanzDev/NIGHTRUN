//! GGUF Q1_0: 128 binary weights, stored LSB-first, with one f16 scale.
//! A zero bit is -scale and a one bit is +scale. Dots use four Q8_0
//! activation blocks per weight block; weights always stay packed.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::q8::{BlockQ8_0, QK8_0};

pub const QK1_0: usize = 128;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ1_0 {
    pub d: u16,
    pub qs: [u8; QK1_0 / 8],
}

const _: () = assert!(core::mem::size_of::<BlockQ1_0>() == 18);

pub fn cast_blocks(bytes: &[u8]) -> &[BlockQ1_0] {
    assert_eq!(bytes.len() % core::mem::size_of::<BlockQ1_0>(), 0);
    // SAFETY: packed, alignment 1, and every byte pattern is valid.
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr().cast(),
            bytes.len() / core::mem::size_of::<BlockQ1_0>(),
        )
    }
}

/// Q8_0 activations for the Q1_0 kernels. Match ggml's SIMD quantizer:
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

pub fn dequantize(blocks: &[BlockQ1_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK1_0);
    for (block, row) in blocks.iter().zip(out.as_chunks_mut::<QK1_0>().0) {
        let d = f16_to_f32(block.d);
        for (i, value) in row.iter_mut().enumerate() {
            *value = if block.qs[i / 8] & (1 << (i % 8)) != 0 {
                d
            } else {
                -d
            };
        }
    }
}

/// Reference accumulation order matches ggml_vec_dot_q1_0_q8_0_generic.
pub fn dot_scalar(w: &[BlockQ1_0], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len(), w.len() * 4);
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let mut acc = 0i32;
            for (i, &q) in bx.qs.iter().enumerate() {
                // Widen before negating: -(-128) must remain +128.
                let q = q as i32;
                acc += if bw.qs[k * 4 + i / 8] & (1 << (i % 8)) != 0 {
                    q
                } else {
                    -q
                };
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
pub unsafe fn dot_avx2(w: &[BlockQ1_0], x: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), w.len() * 4);
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let ones = _mm256_set1_epi8(1);
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let packed = u32::from_le_bytes(bw.qs[k * 4..k * 4 + 4].try_into().unwrap());
            let bits = _mm256_shuffle_epi8(_mm256_set1_epi32(packed as i32), shuf);
            let positive = _mm256_cmpeq_epi8(_mm256_and_si256(bits, masks), masks);
            let twice = _mm256_and_si256(positive, _mm256_set1_epi8(2));
            let q = _mm256_loadu_si256(bx.qs.as_ptr().cast());
            // 2*sum(positive activations) - sum(all activations).
            // i16 pair sums avoid signed-byte negation overflow at -128.
            let pairs = _mm256_sub_epi16(
                _mm256_maddubs_epi16(twice, q),
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
pub unsafe fn dot_neon(w: &[BlockQ1_0], x: &[BlockQ8_0]) -> f32 {
    use core::arch::aarch64::*;
    assert_eq!(x.len(), w.len() * 4);
    let masks = [1u8, 2, 4, 8, 16, 32, 64, 128];
    let masks = vld1_u8(masks.as_ptr());
    let mut sum = 0.0;
    for (bw, xs) in w.iter().zip(x.as_chunks::<4>().0) {
        let mut sub = 0.0;
        for (k, bx) in xs.iter().enumerate() {
            let mut acc = vdupq_n_s16(0);
            for b in 0..4 {
                let positive = vtst_u8(vdup_n_u8(bw.qs[k * 4 + b]), masks);
                let signs = vsub_s8(
                    vreinterpret_s8_u8(vand_u8(positive, vdup_n_u8(2))),
                    vdup_n_s8(1),
                );
                acc = vmlal_s8(acc, vld1_s8(bx.qs.as_ptr().add(b * 8)), signs);
            }
            sub += bx.scale() * vaddlvq_s16(acc) as f32;
        }
        sum += f16_to_f32(bw.d) * sub;
    }
    sum
}

fn dot_kernel() -> unsafe fn(&[BlockQ1_0], &[BlockQ8_0]) -> f32 {
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

pub fn matvec(y: &mut [f32], w: &[BlockQ1_0], x: &[BlockQ8_0], rows: usize, cols: usize) {
    matmul(y, w, x, rows, cols, 1);
}

/// Row-major output [batch][rows], activations [batch][cols/32]. Reuses
/// each packed weight row across the batch with the same decode dot.
pub fn matmul(
    y: &mut [f32],
    w: &[BlockQ1_0],
    xs: &[BlockQ8_0],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    assert_eq!(cols % QK1_0, 0);
    assert_eq!(y.len(), batch * rows);
    let bpr = cols / QK1_0;
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
