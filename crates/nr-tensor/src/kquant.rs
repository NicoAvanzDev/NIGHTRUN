//! K-quant formats (GGUF-compatible): Q4_K and Q6_K weights over 256-value
//! super-blocks, with Q8_K quantized activations. Byte layouts are identical
//! to ggml's `block_q4_K` / `block_q6_K` / `block_q8_K`, so converted
//! tensors are viewed in place and activation quantization matches
//! llama.cpp numerics.

use crate::f16::{f16_to_f32, f32_to_f16};

// Per-arch SIMD backend alias (see kernels.rs).
#[cfg(target_arch = "x86_64")]
mod simd {
    pub use super::{
        dot_q4k_avx2 as dot_q4k, dot_q4k_avx2_x4 as dot_q4k_x4, dot_q6k_avx2 as dot_q6k,
        dot_q6k_avx2_x4 as dot_q6k_x4,
    };
}
#[cfg(target_arch = "aarch64")]
use crate::neon::kquant as simd;

pub const QK_K: usize = 256;

/// 256 weights: two f16 super-scales + 8 six-bit (scale, min) pairs +
/// 128 bytes of 4-bit quants. `w = d*sc*q - dmin*m`.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,    // f16 bits: super-block scale of scales
    pub dmin: u16, // f16 bits: super-block scale of mins
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

/// 256 weights: 4-bit low + 2-bit high quant planes, 16 signed 8-bit
/// scales (one per 16 values), one f16 super-scale. `w = d*sc*(q-32)`.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: u16, // f16 bits
}

/// Activation-side format: 256 int8 values, f32 scale, and per-16 sums
/// (used for the `-dmin*m` correction term in Q4_K dots).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(core::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

pub fn cast_q4k(bytes: &[u8]) -> &[BlockQ4K] {
    assert_eq!(bytes.len() % core::mem::size_of::<BlockQ4K>(), 0);
    // SAFETY: repr(C, packed), valid for any byte pattern.
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr() as *const BlockQ4K,
            bytes.len() / core::mem::size_of::<BlockQ4K>(),
        )
    }
}

pub fn cast_q6k(bytes: &[u8]) -> &[BlockQ6K] {
    assert_eq!(bytes.len() % core::mem::size_of::<BlockQ6K>(), 0);
    // SAFETY: repr(C, packed), valid for any byte pattern.
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr() as *const BlockQ6K,
            bytes.len() / core::mem::size_of::<BlockQ6K>(),
        )
    }
}

/// Unpack the j-th (0..8) 6-bit (scale, min) pair of a Q4_K block
/// (ggml's `get_scale_min_k4`).
#[inline]
pub fn q4k_scale_min(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize one Q4_K block into 256 f32 values.
pub fn dequant_q4k(b: &BlockQ4K, out: &mut [f32]) {
    assert_eq!(out.len(), QK_K);
    let d = f16_to_f32(b.d);
    let dmin = f16_to_f32(b.dmin);
    let scales = b.scales;
    let mut is = 0;
    let mut y = 0;
    for chunk in 0..4 {
        let q = &b.qs[chunk * 32..chunk * 32 + 32];
        let (sc1, m1) = q4k_scale_min(&scales, is);
        let (sc2, m2) = q4k_scale_min(&scales, is + 1);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
        for &qq in q {
            out[y] = d1 * (qq & 0x0F) as f32 - min1;
            y += 1;
        }
        for &qq in q {
            out[y] = d2 * (qq >> 4) as f32 - min2;
            y += 1;
        }
        is += 2;
    }
}

/// Dequantize one Q6_K block into 256 f32 values.
pub fn dequant_q6k(b: &BlockQ6K, out: &mut [f32]) {
    assert_eq!(out.len(), QK_K);
    let d = f16_to_f32(b.d);
    let scales = b.scales;
    for half in 0..2 {
        let ql = &b.ql[half * 64..half * 64 + 64];
        let qh = &b.qh[half * 32..half * 32 + 32];
        let sc = &scales[half * 8..half * 8 + 8];
        let y = &mut out[half * 128..half * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0F) | (((qh[l]) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * sc[is] as f32 * q1 as f32;
            y[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
            y[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
            y[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
        }
    }
}

#[inline]
fn nearest_int(v: f32) -> i32 {
    // Round-to-nearest, ties away from zero (matches ggml's nearest_int
    // for the value ranges seen here).
    if v >= 0.0 { (v + 0.5) as i32 } else { (v - 0.5) as i32 }
}

/// Quantize f32 activations (length divisible by 256) into Q8_K blocks,
/// numerically matching llama.cpp's `quantize_row_q8_K_ref`.
pub fn quantize_q8k(src: &[f32], out: &mut [BlockQ8K]) {
    assert_eq!(src.len(), out.len() * QK_K);
    for (block, chunk) in out.iter_mut().zip(src.chunks_exact(QK_K)) {
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in chunk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs = [0; QK_K];
            block.bsums = [0; QK_K / 16];
            continue;
        }
        let iscale = -128.0 / max;
        for (q, &v) in block.qs.iter_mut().zip(chunk) {
            *q = nearest_int(iscale * v).min(127) as i8;
        }
        for (j, sums) in block.bsums.iter_mut().enumerate() {
            let mut s = 0i32;
            for k in 0..16 {
                s += block.qs[j * 16 + k] as i32;
            }
            *sums = s as i16;
        }
        block.d = 1.0 / iscale;
    }
}

// ---- Scalar reference dot products ----------------------------------------

pub fn dot_q4k_scalar(w: &[BlockQ4K], x: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (bw, bx) in w.iter().zip(x.iter()) {
        let d_all = f16_to_f32(bw.d) * bx.d;
        let dmin_all = f16_to_f32(bw.dmin) * bx.d;
        let scales = bw.scales;
        let mut sum_q = 0f32;
        let mut sum_m = 0i32;
        for j in 0..8 {
            let (sc, m) = q4k_scale_min(&scales, j);
            // 32 quants of sub-block j.
            let mut sub = 0i32;
            let chunk = j / 2;
            let hi = j % 2 == 1;
            let q = &bw.qs[chunk * 32..chunk * 32 + 32];
            let q8 = &bx.qs[j * 32..j * 32 + 32];
            for k in 0..32 {
                let qv = if hi { q[k] >> 4 } else { q[k] & 0x0F } as i32;
                sub += qv * q8[k] as i32;
            }
            sum_q += sc as f32 * sub as f32;
            sum_m += m as i32 * (bx.bsums[2 * j] as i32 + bx.bsums[2 * j + 1] as i32);
        }
        sumf += d_all * sum_q - dmin_all * sum_m as f32;
    }
    sumf
}

pub fn dot_q6k_scalar(w: &[BlockQ6K], x: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (bw, bx) in w.iter().zip(x.iter()) {
        let d_all = f16_to_f32(bw.d) * bx.d;
        let scales = bw.scales;
        let mut sum = 0f32;
        for half in 0..2 {
            let ql = &bw.ql[half * 64..half * 64 + 64];
            let qh = &bw.qh[half * 32..half * 32 + 32];
            let sc = &scales[half * 8..half * 8 + 8];
            let q8 = &bx.qs[half * 128..half * 128 + 128];
            // 8 groups of 16 within this half.
            let mut group_sums = [0i32; 8];
            for l in 0..32 {
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                group_sums[l / 16] += q1 * q8[l] as i32;
                group_sums[2 + l / 16] += q2 * q8[l + 32] as i32;
                group_sums[4 + l / 16] += q3 * q8[l + 64] as i32;
                group_sums[6 + l / 16] += q4 * q8[l + 96] as i32;
            }
            for g in 0..8 {
                sum += sc[g] as f32 * group_sums[g] as f32;
            }
        }
        sumf += d_all * sum;
    }
    sumf
}

// ---- AVX2 kernels ----------------------------------------------------------

/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled
/// (see [`crate::cpu::fast_path`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4k_avx2(w: &[BlockQ4K], x: &[BlockQ8K]) -> f32 {
    use core::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut acc_min = 0f32;

    for (bw, bx) in w.iter().zip(x.iter()) {
        let d = f16_to_f32(bw.d) * bx.d;
        let dmin = f16_to_f32(bw.dmin) * bx.d;
        let scales_raw = bw.scales;

        let mut sc = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            let (s, m) = q4k_scale_min(&scales_raw, j);
            sc[j] = s;
            mins[j] = m;
        }
        let mut min_sum = 0i32;
        for j in 0..8 {
            min_sum += mins[j] as i32 * (bx.bsums[2 * j] as i32 + bx.bsums[2 * j + 1] as i32);
        }
        acc_min += dmin * min_sum as f32;

        let qs_ptr = bw.qs.as_ptr();
        let q8_ptr = bx.qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            // 32 packed bytes -> 64 quants: low nibbles are sub-block 2j,
            // high nibbles sub-block 2j+1.
            let q4bits = _mm256_loadu_si256(qs_ptr.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            let q8l = _mm256_loadu_si256(q8_ptr.add(j * 64) as *const __m256i);
            let q8h = _mm256_loadu_si256(q8_ptr.add(j * 64 + 32) as *const __m256i);
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * j] as i16), p16l);
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * j + 1] as i16), p16h);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum256(acc) - acc_min
}

/// 4-wide Q4_K dot: one pass over the weight row serves four activation
/// vectors, sharing weight loads and scale unpacking. Each lane accumulates
/// its blocks in the same order as [`dot_q4k_avx2`], so per-token results
/// are bit-identical to the 1-wide kernel.
///
/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4k_avx2_x4(w: &[BlockQ4K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut acc_min = [0f32; 4];

    for (bi, bw) in w.iter().enumerate() {
        let d_w = f16_to_f32(bw.d);
        let dmin_w = f16_to_f32(bw.dmin);
        let scales_raw = bw.scales;
        let mut sc = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            let (s, m) = q4k_scale_min(&scales_raw, j);
            sc[j] = s;
            mins[j] = m;
        }
        for lane in 0..4 {
            let bx = &xs[lane][bi];
            let mut min_sum = 0i32;
            for j in 0..8 {
                min_sum += mins[j] as i32 * (bx.bsums[2 * j] as i32 + bx.bsums[2 * j + 1] as i32);
            }
            acc_min[lane] += dmin_w * bx.d * min_sum as f32;
        }

        let qs_ptr = bw.qs.as_ptr();
        let mut sumi = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let q4bits = _mm256_loadu_si256(qs_ptr.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            let scale_l = _mm256_set1_epi16(sc[2 * j] as i16);
            let scale_h = _mm256_set1_epi16(sc[2 * j + 1] as i16);
            for lane in 0..4 {
                let q8_ptr = xs[lane][bi].qs.as_ptr();
                let q8l = _mm256_loadu_si256(q8_ptr.add(j * 64) as *const __m256i);
                let q8h = _mm256_loadu_si256(q8_ptr.add(j * 64 + 32) as *const __m256i);
                let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l);
                let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi[lane] = _mm256_add_epi32(sumi[lane], _mm256_add_epi32(p16l, p16h));
            }
        }
        for lane in 0..4 {
            let d = _mm256_set1_ps(d_w * xs[lane][bi].d);
            acc[lane] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi[lane]), acc[lane]);
        }
    }
    [
        hsum256(acc[0]) - acc_min[0],
        hsum256(acc[1]) - acc_min[1],
        hsum256(acc[2]) - acc_min[2],
        hsum256(acc[3]) - acc_min[3],
    ]
}

/// 4-wide Q6_K dot (see [`dot_q4k_avx2_x4`]); bit-identical per lane to
/// [`dot_q6k_avx2`].
///
/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6k_avx2_x4(w: &[BlockQ6K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let m32s = _mm256_set1_epi8(32);
    let mut acc = [_mm256_setzero_ps(); 4];

    for (bi, bw) in w.iter().enumerate() {
        let d_w = f16_to_f32(bw.d);
        let scales = bw.scales;
        let mut sumi = [_mm256_setzero_si256(); 4];

        for half in 0..2 {
            let ql_ptr = bw.ql.as_ptr().add(half * 64);
            let qh_ptr = bw.qh.as_ptr().add(half * 32);
            let sc = &scales[half * 8..half * 8 + 8];

            let scale = |g: usize| -> __m256i {
                _mm256_set_m128i(_mm_set1_epi16(sc[g + 1] as i16), _mm_set1_epi16(sc[g] as i16))
            };

            let q4bits1 = _mm256_loadu_si256(ql_ptr as *const __m256i);
            let q4bits2 = _mm256_loadu_si256(ql_ptr.add(32) as *const __m256i);
            let qhbits = _mm256_loadu_si256(qh_ptr as *const __m256i);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(qhbits, m2), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 2), m2), 4);
            let q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 4), m2), 4);
            let q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 6), m2), 4);

            let q6 = [
                _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0),
                _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3),
            ];

            for (n, &q6n) in q6.iter().enumerate() {
                let sc_vec = scale(2 * n);
                for lane in 0..4 {
                    let q8_ptr = xs[lane][bi].qs.as_ptr().add(half * 128);
                    let q8 = _mm256_loadu_si256(q8_ptr.add(n * 32) as *const __m256i);
                    let corr = _mm256_maddubs_epi16(m32s, q8);
                    let mut p16 = _mm256_maddubs_epi16(q6n, q8);
                    p16 = _mm256_sub_epi16(p16, corr);
                    p16 = _mm256_madd_epi16(sc_vec, p16);
                    sumi[lane] = _mm256_add_epi32(sumi[lane], p16);
                }
            }
        }
        for lane in 0..4 {
            let d = _mm256_set1_ps(d_w * xs[lane][bi].d);
            acc[lane] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi[lane]), acc[lane]);
        }
    }
    [hsum256(acc[0]), hsum256(acc[1]), hsum256(acc[2]), hsum256(acc[3])]
}

/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6k_avx2(w: &[BlockQ6K], x: &[BlockQ8K]) -> f32 {
    use core::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let m32s = _mm256_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();

    for (bw, bx) in w.iter().zip(x.iter()) {
        let d = f16_to_f32(bw.d) * bx.d;
        let scales = bw.scales;
        let mut sumi = _mm256_setzero_si256();

        for half in 0..2 {
            let ql_ptr = bw.ql.as_ptr().add(half * 64);
            let qh_ptr = bw.qh.as_ptr().add(half * 32);
            let q8_ptr = bx.qs.as_ptr().add(half * 128);
            let sc = &scales[half * 8..half * 8 + 8];

            // i16-lane scale vectors matching maddubs group layout:
            // low 128-bit lane = bytes 0..16 (group g), high = 16..32 (g+1).
            let scale = |g: usize| -> __m256i {
                _mm256_set_m128i(_mm_set1_epi16(sc[g + 1] as i16), _mm_set1_epi16(sc[g] as i16))
            };

            let q4bits1 = _mm256_loadu_si256(ql_ptr as *const __m256i);
            let q4bits2 = _mm256_loadu_si256(ql_ptr.add(32) as *const __m256i);
            let qhbits = _mm256_loadu_si256(qh_ptr as *const __m256i);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(qhbits, m2), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 2), m2), 4);
            let q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 4), m2), 4);
            let q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhbits, 6), m2), 4);

            let q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            let q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            let q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            let q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            for (n, q6) in [q6_0, q6_1, q6_2, q6_3].into_iter().enumerate() {
                let q8 = _mm256_loadu_si256(q8_ptr.add(n * 32) as *const __m256i);
                // (q6 - 32) * q8 == q6*q8 - 32*q8, split so maddubs sees
                // unsigned q6.
                let corr = _mm256_maddubs_epi16(m32s, q8);
                let mut p16 = _mm256_maddubs_epi16(q6, q8);
                p16 = _mm256_sub_epi16(p16, corr);
                p16 = _mm256_madd_epi16(scale(2 * n), p16);
                sumi = _mm256_add_epi32(sumi, p16);
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum256(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

// ---- Matvec entry points (row-parallel via the worker pool) ----------------

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

macro_rules! kquant_matvec {
    ($name:ident, $block:ty, $scalar:ident, $avx2:path) => {
        /// y[r] = dot(w[r, :], x) with rows split across the worker pool.
        pub fn $name(y: &mut [f32], w: &[$block], x: &[BlockQ8K], rows: usize, cols: usize) {
            let bpr = cols / QK_K;
            assert_eq!(y.len(), rows);
            assert_eq!(x.len(), bpr);
            assert!(w.len() >= rows * bpr);
            let fast = crate::cpu::fast_path();
            let yp = SendPtr(y.as_mut_ptr());
            crate::parallel::POOL.run(rows, &|r0, r1| {
                // SAFETY: ranges are disjoint per worker.
                let out = unsafe { core::slice::from_raw_parts_mut(yp.get().add(r0), r1 - r0) };
                for r in r0..r1 {
                    let row = &w[r * bpr..(r + 1) * bpr];
                    out[r - r0] = if fast {
                        // SAFETY: fast_path() verified AVX2+FMA+YMM.
                        unsafe { $avx2(row, x) }
                    } else {
                        $scalar(row, x)
                    };
                }
            });
        }
    };
}

kquant_matvec!(matvec_q4k, BlockQ4K, dot_q4k_scalar, simd::dot_q4k);
kquant_matvec!(matvec_q6k, BlockQ6K, dot_q6k_scalar, simd::dot_q6k);

macro_rules! kquant_matmul {
    ($name:ident, $block:ty, $scalar:ident, $avx2:path, $avx2_x4:path) => {
        /// Batched matvec: y[b][r] = dot(w[r, :], xs[b]); bit-identical to
        /// per-token matvec, with the weight row reused across the batch.
        pub fn $name(
            y: &mut [f32],
            w: &[$block],
            xs: &[BlockQ8K],
            rows: usize,
            cols: usize,
            batch: usize,
        ) {
            let bpr = cols / QK_K;
            assert_eq!(y.len(), batch * rows);
            assert_eq!(xs.len(), batch * bpr);
            assert!(w.len() >= rows * bpr);
            let fast = crate::cpu::fast_path();
            let yp = SendPtr(y.as_mut_ptr());
            crate::parallel::POOL.run(rows, &|r0, r1| {
                for r in r0..r1 {
                    let row = &w[r * bpr..(r + 1) * bpr];
                    let mut b = 0;
                    // 4-wide tiles share weight loads + scale unpacking.
                    while fast && b + 4 <= batch {
                        let vs = unsafe {
                            // SAFETY: fast_path() verified AVX2+FMA+YMM.
                            $avx2_x4(row, [
                                &xs[b * bpr..(b + 1) * bpr],
                                &xs[(b + 1) * bpr..(b + 2) * bpr],
                                &xs[(b + 2) * bpr..(b + 3) * bpr],
                                &xs[(b + 3) * bpr..(b + 4) * bpr],
                            ])
                        };
                        for (i, v) in vs.into_iter().enumerate() {
                            // SAFETY: rows disjoint per worker; each (b, r) once.
                            unsafe { yp.get().add((b + i) * rows + r).write(v) };
                        }
                        b += 4;
                    }
                    while b < batch {
                        let x = &xs[b * bpr..(b + 1) * bpr];
                        let v = if fast {
                            // SAFETY: fast_path() verified AVX2+FMA+YMM.
                            unsafe { $avx2(row, x) }
                        } else {
                            $scalar(row, x)
                        };
                        // SAFETY: rows disjoint per worker; each (b, r) once.
                        unsafe { yp.get().add(b * rows + r).write(v) };
                        b += 1;
                    }
                }
            });
        }
    };
}

kquant_matmul!(matmul_q4k, BlockQ4K, dot_q4k_scalar, simd::dot_q4k, simd::dot_q4k_x4);
kquant_matmul!(matmul_q6k, BlockQ6K, dot_q6k_scalar, simd::dot_q6k, simd::dot_q6k_x4);

// ---- Test-support quantizers (host reference; converter uses GGUF bytes) --

/// Build a Q4_K block from 256 f32 values (simplified reference quantizer:
/// per-32 min/max affine, 6-bit scale/min against super-scales). Good
/// enough to generate valid, adversarial test blocks; not used in the
/// model pipeline (weights come pre-quantized from GGUF).
pub fn quantize_q4k_ref(src: &[f32]) -> BlockQ4K {
    assert_eq!(src.len(), QK_K);
    let mut sub_scale = [0f32; 8];
    let mut sub_min = [0f32; 8];
    for j in 0..8 {
        let s = &src[j * 32..j * 32 + 32];
        let max = s.iter().fold(f32::MIN, |m, &v| m.max(v));
        let min = s.iter().fold(f32::MAX, |m, &v| m.min(v));
        sub_min[j] = (-min).max(0.0);
        sub_scale[j] = ((max + sub_min[j]) / 15.0).max(0.0);
    }
    let max_scale = sub_scale.iter().fold(0f32, |m, &v| m.max(v));
    let max_min = sub_min.iter().fold(0f32, |m, &v| m.max(v));
    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let df = f16_to_f32(f32_to_f16(d));
    let dminf = f16_to_f32(f32_to_f16(dmin));

    let mut scales6 = [0u8; 8];
    let mut mins6 = [0u8; 8];
    for j in 0..8 {
        scales6[j] = if df > 0.0 { nearest_int(sub_scale[j] / df).clamp(0, 63) as u8 } else { 0 };
        mins6[j] = if dminf > 0.0 { nearest_int(sub_min[j] / dminf).clamp(0, 63) as u8 } else { 0 };
    }
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = (scales6[j] & 63) | ((scales6[j + 4] >> 4) << 6);
        scales[j + 4] = (mins6[j] & 63) | ((mins6[j + 4] >> 4) << 6);
        scales[j + 8] = (scales6[j + 4] & 0x0F) | ((mins6[j + 4] & 0x0F) << 4);
    }

    let mut qs = [0u8; 128];
    for j in 0..8 {
        let sc = df * scales6[j] as f32;
        let m = dminf * mins6[j] as f32;
        let s = &src[j * 32..j * 32 + 32];
        let chunk = j / 2;
        let hi = j % 2 == 1;
        for k in 0..32 {
            let q = if sc > 0.0 { nearest_int((s[k] + m) / sc).clamp(0, 15) as u8 } else { 0 };
            if hi {
                qs[chunk * 32 + k] |= q << 4;
            } else {
                qs[chunk * 32 + k] |= q;
            }
        }
    }
    BlockQ4K { d: f32_to_f16(d), dmin: f32_to_f16(dmin), scales, qs }
}

/// Build a Q6_K block from 256 f32 values (simplified reference).
pub fn quantize_q6k_ref(src: &[f32]) -> BlockQ6K {
    assert_eq!(src.len(), QK_K);
    // Per-16 scales against one super-scale.
    let mut group_amax = [0f32; 16];
    for g in 0..16 {
        group_amax[g] = src[g * 16..g * 16 + 16].iter().fold(0f32, |m, &v| m.max(v.abs()));
    }
    let amax = group_amax.iter().fold(0f32, |m, &v| m.max(v));
    let d = amax / (127.0 * 31.0);
    let df = f16_to_f32(f32_to_f16(d));
    let mut scales = [0i8; 16];
    for g in 0..16 {
        scales[g] = if df > 0.0 {
            nearest_int(group_amax[g] / (df * 31.0)).clamp(-128, 127) as i8
        } else {
            0
        };
    }

    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for half in 0..2 {
        for l in 0..32 {
            let mut set = |slot: usize, src_idx: usize, qh_shift: u8, ql_idx: usize, hi: bool| {
                let g = (half * 128 + slot * 32 + l) / 16;
                let sc = df * scales[g] as f32;
                let v = src[half * 128 + slot * 32 + l];
                let q = if sc != 0.0 { (nearest_int(v / sc) + 32).clamp(0, 63) as u8 } else { 32 };
                let (lo, hi_bits) = (q & 0x0F, (q >> 4) & 3);
                let qli = half * 64 + ql_idx;
                if hi {
                    ql[qli] |= lo << 4;
                } else {
                    ql[qli] |= lo;
                }
                qh[half * 32 + l] |= hi_bits << qh_shift;
                let _ = src_idx;
            };
            set(0, 0, 0, l, false); // q1: ql[l] low,  qh bits 0-1
            set(1, 0, 2, l + 32, false); // q2: ql[l+32] low, qh bits 2-3
            set(2, 0, 4, l, true); // q3: ql[l] high, qh bits 4-5
            set(3, 0, 6, l + 32, true); // q4: ql[l+32] high, qh bits 6-7
        }
    }
    BlockQ6K { ql, qh, scales, d: f32_to_f16(d) }
}
