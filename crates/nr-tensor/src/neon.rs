//! aarch64 NEON kernels: baseline ARMv8.0 with a runtime-dispatched
//! FEAT_DotProd (sdot) fast path — one instruction per 16 int8 products
//! instead of four. Integer sums are exact either way, so both paths are
//! interchangeable bit-for-bit.
//!
//! Design rule: NEON computes the *integer* block dots (which are exact),
//! and every f32 scale/accumulate step then replicates the scalar
//! kernels' operation order exactly — so each dot here is bit-identical
//! to its `*_scalar` counterpart, and the tests assert `==`, not
//! tolerances. (The f16 helpers are the exception: their summation order
//! is vectorized, so they're tolerance-tested like x86 F16C.)

use core::arch::aarch64::*;

/// `acc.4s += sdot(a.16b, b.16b)` — FEAT_DotProd, one instruction for 16
/// i8 products. Emitted as a raw encoding with pinned registers: the
/// intrinsic is unstable, and using the mnemonic would require enabling
/// `dotprod` at build time, which must not leak into the baseline
/// fallback path. Callers gate on the runtime FEAT_DotProd probe.
#[inline(always)]
unsafe fn sdot_acc(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    core::arch::asm!(
        ".inst 0x4e829420", // sdot v0.4s, v1.16b, v2.16b
        inout("v0") r, in("v1") a, in("v2") b,
        options(pure, nomem, nostack)
    );
    r
}

/// Exact dot of 32 i8 pairs. `SDOT` selects FEAT_DotProd (one instruction
/// per 16 bytes) vs the baseline widening-multiply path; integer sums are
/// exact either way, so results are identical.
#[inline]
unsafe fn dot_i8x32<const SDOT: bool>(
    w0: int8x16_t,
    w1: int8x16_t,
    x0: int8x16_t,
    x1: int8x16_t,
) -> i32 {
    let mut acc = vdupq_n_s32(0);
    if SDOT {
        acc = sdot_acc(acc, w0, x0);
        acc = sdot_acc(acc, w1, x1);
    } else {
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
    }
    vaddvq_s32(acc)
}

/// Exact dot of 16 i8 pairs (see [`dot_i8x32`]).
#[inline]
unsafe fn dot_i8x16<const SDOT: bool>(w: int8x16_t, x: int8x16_t) -> i32 {
    let mut acc = vdupq_n_s32(0);
    if SDOT {
        acc = sdot_acc(acc, w, x);
    } else {
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w), vget_low_s8(x)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w), vget_high_s8(x)));
    }
    vaddvq_s32(acc)
}

pub mod kernels {
    use super::*;
    use crate::q8::BlockQ8_0;

    /// Per-block integer dot in NEON + the scalar kernel's exact f32
    /// accumulation: bit-identical to `dot_q8_scalar`. Dispatches once on
    /// FEAT_DotProd (sdot) vs the baseline multiply path.
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q8(w: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
        if crate::cpu::features().dotprod {
            dot_q8_impl::<true>(w, x)
        } else {
            dot_q8_impl::<false>(w, x)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q8_impl<const SDOT: bool>(w: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
        let mut sum = 0f32;
        for (bw, bx) in w.iter().zip(x.iter()) {
            let wp = bw.qs.as_ptr() as *const i8;
            let xp = bx.qs.as_ptr() as *const i8;
            let acc = dot_i8x32::<SDOT>(
                vld1q_s8(wp),
                vld1q_s8(wp.add(16)),
                vld1q_s8(xp),
                vld1q_s8(xp.add(16)),
            );
            sum += acc as f32 * bw.scale() * bx.scale();
        }
        sum
    }

    /// 4-wide variant sharing the weight loads; per lane bit-identical to
    /// [`dot_q8`] (same block order, own accumulator).
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q8_x4(w: &[BlockQ8_0], xs: [&[BlockQ8_0]; 4]) -> [f32; 4] {
        if crate::cpu::features().dotprod {
            dot_q8_x4_impl::<true>(w, xs)
        } else {
            dot_q8_x4_impl::<false>(w, xs)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q8_x4_impl<const SDOT: bool>(
        w: &[BlockQ8_0],
        xs: [&[BlockQ8_0]; 4],
    ) -> [f32; 4] {
        let mut sum = [0f32; 4];
        for (bi, bw) in w.iter().enumerate() {
            let wp = bw.qs.as_ptr() as *const i8;
            let w0 = vld1q_s8(wp);
            let w1 = vld1q_s8(wp.add(16));
            let dw = bw.scale();
            for lane in 0..4 {
                let bx = &xs[lane][bi];
                let xp = bx.qs.as_ptr() as *const i8;
                let acc = dot_i8x32::<SDOT>(w0, w1, vld1q_s8(xp), vld1q_s8(xp.add(16)));
                sum[lane] += acc as f32 * dw * bx.scale();
            }
        }
        sum
    }

    /// Convert 8 f16 values (bits) to two f32x4 (FCVTL is ARMv8.0
    /// baseline; the f16 vector *types* are unstable in core::arch, hence
    /// the one-instruction asm).
    #[inline]
    unsafe fn f16x8_to_f32(p: *const u16) -> (float32x4_t, float32x4_t) {
        let lo: float32x4_t;
        let hi: float32x4_t;
        core::arch::asm!(
            "ldr {tmp:q}, [{p}]",
            "fcvtl {lo:v}.4s, {tmp:v}.4h",
            "fcvtl2 {hi:v}.4s, {tmp:v}.8h",
            p = in(reg) p,
            tmp = out(vreg) _,
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            options(nostack, readonly)
        );
        (lo, hi)
    }

    /// dot(k, q) with f16-bit `k` (KV cache row). Vectorized summation
    /// order (tolerance-tested against scalar, like x86 F16C).
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_f16(k: &[u16], q: &[f32]) -> f32 {
        let n = k.len().min(q.len());
        let chunks = n / 8;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let (klo, khi) = f16x8_to_f32(k.as_ptr().add(i * 8));
            acc0 = vfmaq_f32(acc0, klo, vld1q_f32(q.as_ptr().add(i * 8)));
            acc1 = vfmaq_f32(acc1, khi, vld1q_f32(q.as_ptr().add(i * 8 + 4)));
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for i in chunks * 8..n {
            sum += crate::f16::f16_to_f32(k[i]) * q[i];
        }
        sum
    }

    /// out += a * v with f16-bit `v` (KV cache values row).
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn axpy_f16(out: &mut [f32], a: f32, v: &[u16]) {
        let n = out.len().min(v.len());
        let chunks = n / 8;
        let av = vdupq_n_f32(a);
        for i in 0..chunks {
            let (vlo, vhi) = f16x8_to_f32(v.as_ptr().add(i * 8));
            let p = out.as_mut_ptr().add(i * 8);
            vst1q_f32(p, vfmaq_f32(vld1q_f32(p), av, vlo));
            vst1q_f32(p.add(4), vfmaq_f32(vld1q_f32(p.add(4)), av, vhi));
        }
        for i in chunks * 8..n {
            out[i] += a * crate::f16::f16_to_f32(v[i]);
        }
    }
}

pub mod kquant {
    use super::*;
    use crate::f16::f16_to_f32;
    use crate::kquant::{q4k_scale_min, BlockQ4K, BlockQ6K, BlockQ8K};

    /// Unpack sub-block `j`'s 32 4-bit quants of a Q4_K block into two
    /// i8x16 vectors (values 0..15).
    #[inline]
    unsafe fn q4k_sub(qs: *const u8, j: usize) -> (int8x16_t, int8x16_t) {
        let chunk = qs.add((j / 2) * 32);
        let lo = vld1q_u8(chunk);
        let hi = vld1q_u8(chunk.add(16));
        let m4 = vdupq_n_u8(0x0F);
        if j % 2 == 0 {
            (
                vreinterpretq_s8_u8(vandq_u8(lo, m4)),
                vreinterpretq_s8_u8(vandq_u8(hi, m4)),
            )
        } else {
            (
                vreinterpretq_s8_u8(vshrq_n_u8(lo, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(hi, 4)),
            )
        }
    }

    /// Bit-identical to `dot_q4k_scalar` (exact integer sub-dots + the
    /// same f32 assembly).
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q4k(w: &[BlockQ4K], x: &[BlockQ8K]) -> f32 {
        if crate::cpu::features().dotprod {
            dot_q4k_impl::<true>(w, x)
        } else {
            dot_q4k_impl::<false>(w, x)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q4k_impl<const SDOT: bool>(w: &[BlockQ4K], x: &[BlockQ8K]) -> f32 {
        let mut sumf = 0f32;
        for (bw, bx) in w.iter().zip(x.iter()) {
            let d_all = f16_to_f32(bw.d) * bx.d;
            let dmin_all = f16_to_f32(bw.dmin) * bx.d;
            let scales = bw.scales;
            let mut sum_q = 0f32;
            let mut sum_m = 0i32;
            for j in 0..8 {
                let (sc, m) = q4k_scale_min(&scales, j);
                let (w0, w1) = q4k_sub(bw.qs.as_ptr(), j);
                let xp = bx.qs.as_ptr().add(j * 32);
                let sub = dot_i8x32::<SDOT>(w0, w1, vld1q_s8(xp), vld1q_s8(xp.add(16)));
                sum_q += sc as f32 * sub as f32;
                sum_m += m as i32 * (bx.bsums[2 * j] as i32 + bx.bsums[2 * j + 1] as i32);
            }
            sumf += d_all * sum_q - dmin_all * sum_m as f32;
        }
        sumf
    }

    /// 4-wide Q4_K dot sharing weight unpacking; per lane bit-identical
    /// to [`dot_q4k`].
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q4k_x4(w: &[BlockQ4K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
        if crate::cpu::features().dotprod {
            dot_q4k_x4_impl::<true>(w, xs)
        } else {
            dot_q4k_x4_impl::<false>(w, xs)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q4k_x4_impl<const SDOT: bool>(
        w: &[BlockQ4K],
        xs: [&[BlockQ8K]; 4],
    ) -> [f32; 4] {
        let mut sumf = [0f32; 4];
        for (bi, bw) in w.iter().enumerate() {
            let d_w = f16_to_f32(bw.d);
            let dmin_w = f16_to_f32(bw.dmin);
            let scales = bw.scales;
            let mut sum_q = [0f32; 4];
            let mut sum_m = [0i32; 4];
            for j in 0..8 {
                let (sc, m) = q4k_scale_min(&scales, j);
                let (w0, w1) = q4k_sub(bw.qs.as_ptr(), j);
                for lane in 0..4 {
                    let bx = &xs[lane][bi];
                    let xp = bx.qs.as_ptr().add(j * 32);
                    let sub = dot_i8x32::<SDOT>(w0, w1, vld1q_s8(xp), vld1q_s8(xp.add(16)));
                    sum_q[lane] += sc as f32 * sub as f32;
                    sum_m[lane] +=
                        m as i32 * (bx.bsums[2 * j] as i32 + bx.bsums[2 * j + 1] as i32);
                }
            }
            for lane in 0..4 {
                let bx_d = xs[lane][bi].d;
                sumf[lane] += (d_w * bx_d) * sum_q[lane] - (dmin_w * bx_d) * sum_m[lane] as f32;
            }
        }
        sumf
    }

    /// Unpack one 32-value Q6_K plane: `(nibble | msbits<<4) - 32` for
    /// lanes 0..16 (a) and 16..32 (b).
    #[inline]
    unsafe fn q6k_plane(
        ql: int8x16_t,
        ql2: int8x16_t,
        qh: int8x16_t,
        qh2: int8x16_t,
        shift: i32,
        high_nibble: bool,
    ) -> (int8x16_t, int8x16_t) {
        let m4 = vdupq_n_u8(0x0F);
        let m2 = vdupq_n_u8(3);
        let off = vdupq_n_s8(32);
        let sel = |v: int8x16_t| -> uint8x16_t {
            let u = vreinterpretq_u8_s8(v);
            if high_nibble {
                vshrq_n_u8(u, 4)
            } else {
                vandq_u8(u, m4)
            }
        };
        let hi_bits = |v: int8x16_t| -> uint8x16_t {
            let u = vreinterpretq_u8_s8(v);
            let sh = match shift {
                0 => u,
                2 => vshrq_n_u8(u, 2),
                4 => vshrq_n_u8(u, 4),
                _ => vshrq_n_u8(u, 6),
            };
            vandq_u8(sh, m2)
        };
        let a = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(sel(ql), vshlq_n_u8(hi_bits(qh), 4))),
            off,
        );
        let b = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(sel(ql2), vshlq_n_u8(hi_bits(qh2), 4))),
            off,
        );
        (a, b)
    }

    /// Bit-identical to `dot_q6k_scalar`.
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q6k(w: &[BlockQ6K], x: &[BlockQ8K]) -> f32 {
        if crate::cpu::features().dotprod {
            dot_q6k_impl::<true>(w, x)
        } else {
            dot_q6k_impl::<false>(w, x)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q6k_impl<const SDOT: bool>(w: &[BlockQ6K], x: &[BlockQ8K]) -> f32 {
        let mut sumf = 0f32;
        for (bw, bx) in w.iter().zip(x.iter()) {
            let d_all = f16_to_f32(bw.d) * bx.d;
            let scales = bw.scales;
            let mut sum = 0f32;
            for half in 0..2 {
                let ql = bw.ql.as_ptr().add(half * 64);
                let qh = bw.qh.as_ptr().add(half * 32);
                let sc = &scales[half * 8..half * 8 + 8];
                let q8 = bx.qs.as_ptr().add(half * 128);

                let ql_a = vld1q_s8(ql as *const i8);
                let ql_b = vld1q_s8(ql.add(16) as *const i8);
                let ql_c = vld1q_s8(ql.add(32) as *const i8);
                let ql_d = vld1q_s8(ql.add(48) as *const i8);
                let qh_a = vld1q_s8(qh as *const i8);
                let qh_b = vld1q_s8(qh.add(16) as *const i8);

                // Plane n holds quants n*32..n*32+32; scalar's
                // group_sums[2n + l/16] maps to (a = lanes 0..16, b = 16..32),
                // and its per-group f32 accumulation order is sc[0]*g0,
                // sc[1]*g1, ... — replicated exactly below.
                let planes = [
                    q6k_plane(ql_a, ql_b, qh_a, qh_b, 0, false),
                    q6k_plane(ql_c, ql_d, qh_a, qh_b, 2, false),
                    q6k_plane(ql_a, ql_b, qh_a, qh_b, 4, true),
                    q6k_plane(ql_c, ql_d, qh_a, qh_b, 6, true),
                ];
                let mut group_sums = [0i32; 8];
                for (n, (qa, qb)) in planes.into_iter().enumerate() {
                    let xa = vld1q_s8(q8.add(n * 32) as *const i8);
                    let xb = vld1q_s8(q8.add(n * 32 + 16) as *const i8);
                    group_sums[2 * n] = dot_i8x16::<SDOT>(qa, xa);
                    group_sums[2 * n + 1] = dot_i8x16::<SDOT>(qb, xb);
                }
                for g in 0..8 {
                    sum += sc[g] as f32 * group_sums[g] as f32;
                }
            }
            sumf += d_all * sum;
        }
        sumf
    }

    /// 4-wide Q6_K dot sharing plane unpacking; per lane bit-identical to
    /// [`dot_q6k`].
    ///
    /// # Safety
    /// NEON is baseline on aarch64; no extra requirements.
    pub unsafe fn dot_q6k_x4(w: &[BlockQ6K], xs: [&[BlockQ8K]; 4]) -> [f32; 4] {
        if crate::cpu::features().dotprod {
            dot_q6k_x4_impl::<true>(w, xs)
        } else {
            dot_q6k_x4_impl::<false>(w, xs)
        }
    }

    #[doc(hidden)]
    pub unsafe fn dot_q6k_x4_impl<const SDOT: bool>(
        w: &[BlockQ6K],
        xs: [&[BlockQ8K]; 4],
    ) -> [f32; 4] {
        let mut sumf = [0f32; 4];
        for (bi, bw) in w.iter().enumerate() {
            let d_w = f16_to_f32(bw.d);
            let scales = bw.scales;
            let mut sum = [0f32; 4];
            for half in 0..2 {
                let ql = bw.ql.as_ptr().add(half * 64);
                let qh = bw.qh.as_ptr().add(half * 32);
                let sc = &scales[half * 8..half * 8 + 8];

                let ql_a = vld1q_s8(ql as *const i8);
                let ql_b = vld1q_s8(ql.add(16) as *const i8);
                let ql_c = vld1q_s8(ql.add(32) as *const i8);
                let ql_d = vld1q_s8(ql.add(48) as *const i8);
                let qh_a = vld1q_s8(qh as *const i8);
                let qh_b = vld1q_s8(qh.add(16) as *const i8);

                let planes = [
                    q6k_plane(ql_a, ql_b, qh_a, qh_b, 0, false),
                    q6k_plane(ql_c, ql_d, qh_a, qh_b, 2, false),
                    q6k_plane(ql_a, ql_b, qh_a, qh_b, 4, true),
                    q6k_plane(ql_c, ql_d, qh_a, qh_b, 6, true),
                ];
                for lane in 0..4 {
                    let q8 = xs[lane][bi].qs.as_ptr().add(half * 128);
                    let mut group_sums = [0i32; 8];
                    for (n, (qa, qb)) in planes.iter().enumerate() {
                        let xa = vld1q_s8(q8.add(n * 32) as *const i8);
                        let xb = vld1q_s8(q8.add(n * 32 + 16) as *const i8);
                        group_sums[2 * n] = dot_i8x16::<SDOT>(*qa, xa);
                        group_sums[2 * n + 1] = dot_i8x16::<SDOT>(*qb, xb);
                    }
                    for g in 0..8 {
                        sum[lane] += sc[g] as f32 * group_sums[g] as f32;
                    }
                }
            }
            for lane in 0..4 {
                sumf[lane] += (d_w * xs[lane][bi].d) * sum[lane];
            }
        }
        sumf
    }
}
