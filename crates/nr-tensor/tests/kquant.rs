use nr_tensor::f16::f16_to_f32;
use nr_tensor::kquant::{
    self, dequant_q4k, dequant_q6k, dot_q4k_scalar, dot_q6k_scalar, quantize_q4k_ref,
    quantize_q6k_ref, quantize_q8k, BlockQ8K, QK_K,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }

    /// Adversarial vector: huge dynamic range, negatives, zeros, spikes.
    fn adversarial(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match self.next() % 7 {
                0 => 0.0,
                1 => 100.0 * self.f32(),
                2 => 1e-4 * self.f32(),
                3 => -50.0 + self.f32(),
                _ => self.f32() * (1 + i % 13) as f32,
            })
            .collect()
    }
}

fn q8k(v: &[f32]) -> Vec<BlockQ8K> {
    let mut out = vec![
        BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16] };
        v.len() / QK_K
    ];
    quantize_q8k(v, &mut out);
    out
}

#[test]
fn q8k_quantization_properties() {
    let mut rng = Rng(11);
    for src in [rng.vec(512), rng.adversarial(512)] {
        let blocks = q8k(&src);
        for (b, chunk) in blocks.iter().zip(src.chunks_exact(QK_K)) {
            // bsums must be exact group sums.
            for g in 0..16 {
                let s: i32 = b.qs[g * 16..g * 16 + 16].iter().map(|&q| q as i32).sum();
                assert_eq!(s, b.bsums[g] as i32);
            }
            // Reconstruction error bounded by half a step.
            for (q, &v) in b.qs.iter().zip(chunk) {
                let back = *q as f32 * b.d;
                assert!((back - v).abs() <= b.d.abs() * 0.5 + 1e-6, "{back} vs {v} (d={})", b.d);
            }
        }
    }
}

#[test]
fn q4k_dequant_roundtrip() {
    let mut rng = Rng(21);
    for src in [rng.vec(QK_K), rng.adversarial(QK_K)] {
        let block = quantize_q4k_ref(&src);
        let mut back = vec![0f32; QK_K];
        dequant_q4k(&block, &mut back);
        // Affine 4-bit per 32 values: error <= scale/2 (+ 6-bit scale error).
        for j in 0..8 {
            let (sc, _) = kquant::q4k_scale_min(&block.scales, j);
            let step = f16_to_f32(block.d) * sc as f32;
            for k in 0..32 {
                let i = j * 32 + k;
                let tol = step * 0.75 + 2e-2 * src[i].abs().max(1.0) * 0.0 + step.max(1e-3);
                assert!(
                    (back[i] - src[i]).abs() <= tol,
                    "sub {j} elem {k}: {} vs {} (step {step})",
                    back[i],
                    src[i]
                );
            }
        }
    }
}

/// Bit-packing exactness: dequantizing must reproduce exactly the values
/// the quantizer intended (q recomputed from the block's own d/scales).
/// This pins ql/qh plane packing and scale indexing with zero tolerance.
#[test]
fn q6k_dequant_roundtrip() {
    let mut rng = Rng(31);
    for src in [rng.vec(QK_K), rng.adversarial(QK_K)] {
        let block = quantize_q6k_ref(&src);
        let d = f16_to_f32(block.d);
        let mut back = vec![0f32; QK_K];
        dequant_q6k(&block, &mut back);
        for i in 0..QK_K {
            let g = i / 16;
            let sc = d * block.scales[g] as f32;
            let q = if sc != 0.0 {
                let r = src[i] / sc;
                let r = if r >= 0.0 { (r + 0.5) as i32 } else { (r - 0.5) as i32 };
                (r + 32).clamp(0, 63) - 32
            } else {
                0
            };
            let expect = sc * q as f32;
            assert!(
                (back[i] - expect).abs() <= expect.abs() * 1e-6 + 1e-7,
                "elem {i}: {} vs {expect} (q={q}, sc={sc})",
                back[i]
            );
        }
    }
}

/// The scalar dots must match the "dequantize everything and dot in f32"
/// formulation almost exactly (same math, different association).
#[test]
fn q4k_dot_matches_dequant_reference() {
    let mut rng = Rng(41);
    for blocks in [1usize, 3, 10] {
        let wsrc = rng.adversarial(blocks * QK_K);
        let xsrc = rng.vec(blocks * QK_K);
        let w: Vec<_> = wsrc.chunks_exact(QK_K).map(quantize_q4k_ref).collect();
        let x = q8k(&xsrc);

        let got = dot_q4k_scalar(&w, &x);

        let mut expect = 0f64;
        let mut wd = vec![0f32; QK_K];
        for (bw, bx) in w.iter().zip(x.iter()) {
            dequant_q4k(bw, &mut wd);
            for i in 0..QK_K {
                expect += wd[i] as f64 * (bx.qs[i] as f32 * bx.d) as f64;
            }
        }
        let tol = 1e-3 * expect.abs().max(1.0);
        assert!(
            (got as f64 - expect).abs() <= tol,
            "blocks={blocks}: got {got} expect {expect}"
        );
    }
}

#[test]
fn avx2_matches_scalar_q4k() {
    assert!(nr_tensor::cpu::fast_path(), "test host must have AVX2+FMA");
    let mut rng = Rng(61);
    for blocks in [1usize, 5, 16] {
        let w: Vec<_> = rng.adversarial(blocks * QK_K).chunks_exact(QK_K).map(quantize_q4k_ref).collect();
        let x = q8k(&rng.vec(blocks * QK_K));
        let s = dot_q4k_scalar(&w, &x);
        let v = unsafe { kquant::dot_q4k_avx2(&w, &x) };
        assert!(
            (s - v).abs() <= 1e-4 * s.abs().max(1.0),
            "blocks={blocks}: scalar={s} avx2={v}"
        );
    }
}

#[test]
fn avx2_matches_scalar_q6k() {
    assert!(nr_tensor::cpu::fast_path(), "test host must have AVX2+FMA");
    let mut rng = Rng(71);
    for blocks in [1usize, 5, 16] {
        let w: Vec<_> = rng.adversarial(blocks * QK_K).chunks_exact(QK_K).map(quantize_q6k_ref).collect();
        let x = q8k(&rng.vec(blocks * QK_K));
        let s = dot_q6k_scalar(&w, &x);
        let v = unsafe { kquant::dot_q6k_avx2(&w, &x) };
        assert!(
            (s - v).abs() <= 1e-4 * s.abs().max(1.0),
            "blocks={blocks}: scalar={s} avx2={v}"
        );
    }
}

/// Saturated metadata: max 6-bit scales/mins, extreme quants.
#[test]
fn avx2_matches_scalar_saturated_blocks() {
    assert!(nr_tensor::cpu::fast_path());
    // Q4_K block with all scales/mins at 63 and all quants at 15.
    let q4 = kquant::BlockQ4K {
        d: nr_tensor::f32_to_f16(0.9),
        dmin: nr_tensor::f32_to_f16(1.7),
        scales: [0xFF; 12],
        qs: [0xFF; 128],
    };
    // Q6_K block with extreme signed scales and full bit planes.
    let q6 = kquant::BlockQ6K {
        ql: [0xFF; 128],
        qh: [0xFF; 64],
        scales: [-128i8; 16],
        d: nr_tensor::f32_to_f16(1.3),
    };
    let mut rng = Rng(81);
    let x = q8k(&rng.adversarial(QK_K));
    let (s4, v4) = (dot_q4k_scalar(&[q4], &x), unsafe { kquant::dot_q4k_avx2(&[q4], &x) });
    assert!((s4 - v4).abs() <= 1e-4 * s4.abs().max(1.0), "q4k: {s4} vs {v4}");
    let (s6, v6) = (dot_q6k_scalar(&[q6], &x), unsafe { kquant::dot_q6k_avx2(&[q6], &x) });
    assert!((s6 - v6).abs() <= 1e-4 * s6.abs().max(1.0), "q6k: {s6} vs {v6}");
}

#[test]
fn matvec_kquant_matches_dot() {
    let mut rng = Rng(91);
    let (rows, cols) = (48, 512);
    let w4: Vec<_> = rng.vec(rows * cols).chunks_exact(QK_K).map(quantize_q4k_ref).collect();
    let x = q8k(&rng.vec(cols));
    let mut y = vec![0f32; rows];
    kquant::matvec_q4k(&mut y, &w4, &x, rows, cols);
    let bpr = cols / QK_K;
    for r in 0..rows {
        let expect = dot_q4k_scalar(&w4[r * bpr..(r + 1) * bpr], &x);
        assert!((y[r] - expect).abs() <= 1e-4 * expect.abs().max(1.0), "row {r}");
    }
}

#[test]
fn q6k_dot_matches_dequant_reference() {
    let mut rng = Rng(51);
    for blocks in [1usize, 3, 10] {
        let wsrc = rng.adversarial(blocks * QK_K);
        let xsrc = rng.vec(blocks * QK_K);
        let w: Vec<_> = wsrc.chunks_exact(QK_K).map(quantize_q6k_ref).collect();
        let x = q8k(&xsrc);

        let got = dot_q6k_scalar(&w, &x);

        let mut expect = 0f64;
        let mut wd = vec![0f32; QK_K];
        for (bw, bx) in w.iter().zip(x.iter()) {
            dequant_q6k(bw, &mut wd);
            for i in 0..QK_K {
                expect += wd[i] as f64 * (bx.qs[i] as f32 * bx.d) as f64;
            }
        }
        let tol = 1e-3 * expect.abs().max(1.0);
        assert!(
            (got as f64 - expect).abs() <= tol,
            "blocks={blocks}: got {got} expect {expect}"
        );
    }
}
