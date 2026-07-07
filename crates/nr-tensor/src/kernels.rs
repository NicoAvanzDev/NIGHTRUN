//! Matrix-vector kernels over Q8_0 weights with Q8_0-quantized activations
//! (the llama.cpp `q8_0 x q8_0` scheme): integer dot products per 32-block,
//! scaled by the product of the two f16 block scales.

use crate::cpu;
use crate::q8::{BlockQ8_0, QK8_0};

// Architecture-specific SIMD backends behind one alias; call sites below
// are cfg-free. The active backend is only entered when
// `cpu::fast_path()` / `cpu::fast_f16()` says its features are present.
#[cfg(target_arch = "x86_64")]
mod simd {
    pub use super::{
        axpy_f16_f16c as axpy_f16, dot_f16_f16c as dot_f16, dot_q8_avx2 as dot_q8,
        dot_q8_avx2_x4 as dot_q8_x4,
    };
}
#[cfg(target_arch = "aarch64")]
use crate::neon::kernels as simd;

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// Method (not field) access so closures capture the whole Sync
    /// wrapper rather than the raw pointer.
    fn get(&self) -> *mut f32 {
        self.0
    }
}

/// y[r] = dot(w[r, :], x) for row-major Q8_0 `w` (rows x cols) and
/// Q8_0-quantized activation `x` (cols values). Rows are split across the
/// worker pool when one is active.
pub fn matvec_q8(y: &mut [f32], w: &[BlockQ8_0], x: &[BlockQ8_0], rows: usize, cols: usize) {
    let bpr = cols / QK8_0;
    assert_eq!(y.len(), rows);
    assert_eq!(x.len(), bpr);
    assert!(w.len() >= rows * bpr);
    let yp = SendPtr(y.as_mut_ptr());
    crate::parallel::POOL.run(rows, &|r0, r1| {
        // SAFETY: ranges are disjoint, so each worker writes its own slice.
        let out = unsafe { core::slice::from_raw_parts_mut(yp.get().add(r0), r1 - r0) };
        matvec_q8_range(out, w, x, r0, r1, bpr);
    });
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
            // SAFETY: fast_path() verified the backend's features.
            y[r - row0] = unsafe { simd::dot_q8(row, x) };
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

/// Batched matvec ("matmul"): y[b][r] = dot(w[r, :], xs[b]) for a batch of
/// quantized activation vectors. Same dots in the same block order as
/// `matvec_q8` — results are bit-identical to running matvec per token —
/// but the weight row is reused from cache across the batch, cutting DRAM
/// traffic by ~batch-fold (the whole point of batched prefill).
///
/// `y` is `[batch][rows]` contiguous; `xs` is `[batch][cols/32]` blocks.
pub fn matmul_q8(
    y: &mut [f32],
    w: &[BlockQ8_0],
    xs: &[BlockQ8_0],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    let bpr = cols / QK8_0;
    assert_eq!(y.len(), batch * rows);
    assert_eq!(xs.len(), batch * bpr);
    assert!(w.len() >= rows * bpr);
    let fast = cpu::fast_path();
    let yp = SendPtr(y.as_mut_ptr());
    crate::parallel::POOL.run(rows, &|r0, r1| {
        for r in r0..r1 {
            let row = &w[r * bpr..(r + 1) * bpr];
            let mut b = 0;
            // 4-wide tiles share weight loads across the batch.
            while fast && b + 4 <= batch {
                let vs = unsafe {
                    // SAFETY: fast_path() verified the backend's features.
                    simd::dot_q8_x4(
                        row,
                        [
                            &xs[b * bpr..(b + 1) * bpr],
                            &xs[(b + 1) * bpr..(b + 2) * bpr],
                            &xs[(b + 2) * bpr..(b + 3) * bpr],
                            &xs[(b + 3) * bpr..(b + 4) * bpr],
                        ],
                    )
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
                    // SAFETY: fast_path() verified the backend's features.
                    unsafe { simd::dot_q8(row, x) }
                } else {
                    dot_q8_scalar(row, x)
                };
                // SAFETY: rows are disjoint per worker; each (b, r) written once.
                unsafe { yp.get().add(b * rows + r).write(v) };
                b += 1;
            }
        }
    });
}

/// dot(k, q) where `k` holds f16 bits (KV cache) and `q` is f32.
pub fn dot_f16(k: &[u16], q: &[f32]) -> f32 {
    if cpu::fast_f16() {
        // SAFETY: fast_f16() verified the backend's features.
        unsafe { simd::dot_f16(k, q) }
    } else {
        let mut s = 0f32;
        for (&kb, &qv) in k.iter().zip(q) {
            s += crate::f16::f16_to_f32(kb) * qv;
        }
        s
    }
}

/// out += a * v where `v` holds f16 bits (KV cache values row).
pub fn axpy_f16(out: &mut [f32], a: f32, v: &[u16]) {
    if cpu::fast_f16() {
        // SAFETY: fast_f16() verified the backend's features.
        unsafe { simd::axpy_f16(out, a, v) };
    } else {
        for (o, &vb) in out.iter_mut().zip(v) {
            *o += a * crate::f16::f16_to_f32(vb);
        }
    }
}

/// # Safety
/// Caller must ensure AVX2+FMA+F16C are supported and YMM state is
/// enabled (see [`cpu::fast_f16`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16_f16c(k: &[u16], q: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = k.len().min(q.len());
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let kh = _mm_loadu_si128(k.as_ptr().add(i * 8) as *const __m128i);
        let kf = _mm256_cvtph_ps(kh);
        let qf = _mm256_loadu_ps(q.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(kf, qf, acc);
    }
    let mut sum = {
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    };
    for i in chunks * 8..n {
        sum += crate::f16::f16_to_f32(k[i]) * q[i];
    }
    sum
}

/// # Safety
/// Caller must ensure AVX2+FMA+F16C are supported and YMM state is
/// enabled (see [`cpu::fast_f16`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn axpy_f16_f16c(out: &mut [f32], a: f32, v: &[u16]) {
    use core::arch::x86_64::*;
    let n = out.len().min(v.len());
    let av = _mm256_set1_ps(a);
    let chunks = n / 8;
    for i in 0..chunks {
        let vh = _mm_loadu_si128(v.as_ptr().add(i * 8) as *const __m128i);
        let vf = _mm256_cvtph_ps(vh);
        let of = _mm256_loadu_ps(out.as_ptr().add(i * 8));
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_fmadd_ps(av, vf, of));
    }
    for i in chunks * 8..n {
        out[i] += a * crate::f16::f16_to_f32(v[i]);
    }
}

/// 4-wide Q8_0 dot: shares weight loads across four activation vectors;
/// bit-identical per lane to [`dot_q8_avx2`].
///
/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_avx2_x4(w: &[BlockQ8_0], xs: [&[BlockQ8_0]; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;

    let mut acc = [_mm256_setzero_ps(); 4];
    for (bi, bw) in w.iter().enumerate() {
        let qw = _mm256_loadu_si256(bw.qs.as_ptr() as *const __m256i);
        let abs_w = _mm256_sign_epi8(qw, qw);
        let dw = bw.scale();
        for lane in 0..4 {
            let bx = &xs[lane][bi];
            let d = _mm256_set1_ps(dw * bx.scale());
            let qx = _mm256_loadu_si256(bx.qs.as_ptr() as *const __m256i);
            let sgn_x = _mm256_sign_epi8(qx, qw);
            let p16 = _mm256_maddubs_epi16(abs_w, sgn_x);
            let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
            acc[lane] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc[lane]);
        }
    }
    let hsum = |v: __m256| -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    };
    [hsum(acc[0]), hsum(acc[1]), hsum(acc[2]), hsum(acc[3])]
}

/// # Safety
/// Caller must ensure AVX2+FMA are supported and YMM state is enabled
/// (see [`cpu::fast_path`]).
#[cfg(target_arch = "x86_64")]
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
