use nr_tensor::kernels::{dot_q8_scalar, matvec_q8};
#[cfg(target_arch = "x86_64")]
use nr_tensor::kernels::dot_q8_avx2;
use nr_tensor::q8::{self, BlockQ8_0, QK8_0};
use nr_tensor::rope::{self, Llama3Scaling};
use nr_tensor::vec;
use nr_tensor::{cpu, f16_to_f32, f32_to_f16};

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
        // Uniform in [-1, 1).
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

fn quantized(v: &[f32]) -> Vec<BlockQ8_0> {
    let mut out = vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; v.len() / QK8_0];
    q8::quantize(v, &mut out);
    out
}

#[test]
fn f16_known_values() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x3800), 0.5);
    assert_eq!(f16_to_f32(0x7bff), 65504.0); // f16 max
    assert_eq!(f16_to_f32(0x0001), 5.9604645e-8); // smallest subnormal
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert!(f16_to_f32(0x7c00).is_infinite());
    assert!(f16_to_f32(0x7e00).is_nan());

    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f32_to_f16(-2.0), 0xc000);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(1e30), 0x7c00); // overflow -> inf
}

#[test]
fn f16_roundtrip_normals() {
    // Every normal, finite f16 must roundtrip bit-exactly.
    for bits in 0x0400..0x7c00u16 {
        for sign in [0u16, 0x8000] {
            let b = bits | sign;
            assert_eq!(f32_to_f16(f16_to_f32(b)), b, "bits {b:#06x}");
        }
    }
}

#[test]
fn quantize_error_bound() {
    let mut rng = Rng(7);
    let x = rng.vec(1024);
    let blocks = quantized(&x);
    let mut back = vec![0f32; 1024];
    q8::dequantize(&blocks, &mut back);
    for (i, (&a, &b)) in x.iter().zip(&back).enumerate() {
        let scale = blocks[i / QK8_0].scale();
        assert!((a - b).abs() <= scale * 0.5 + 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar() {
    assert!(cpu::fast_path(), "test host must have AVX2+FMA");
    let mut rng = Rng(42);
    for blocks in [1usize, 3, 64, 65] {
        let w = quantized(&rng.vec(blocks * QK8_0));
        let x = quantized(&rng.vec(blocks * QK8_0));
        let s = dot_q8_scalar(&w, &x);
        let v = unsafe { dot_q8_avx2(&w, &x) };
        let tol = 1e-5 * (blocks as f32).max(1.0) + 1e-6;
        assert!(
            (s - v).abs() <= tol * s.abs().max(1.0),
            "blocks={blocks}: scalar={s} avx2={v}"
        );
    }
}

#[test]
fn matvec_matches_f32_reference() {
    let mut rng = Rng(1234);
    let (rows, cols) = (48, 128);
    let wf = rng.vec(rows * cols);
    let xf = rng.vec(cols);
    let w = quantized(&wf);
    let x = quantized(&xf);

    let mut y = vec![0f32; rows];
    matvec_q8(&mut y, &w, &x, rows, cols);

    // Reference: dequantized dot in f32 (so only kernel error, not quant
    // error, is measured).
    let mut wd = vec![0f32; rows * cols];
    q8::dequantize(&w, &mut wd);
    let mut xd = vec![0f32; cols];
    q8::dequantize(&x, &mut xd);
    for r in 0..rows {
        let expect = vec::dot(&wd[r * cols..(r + 1) * cols], &xd);
        assert!(
            (y[r] - expect).abs() <= 1e-4 * expect.abs().max(1.0),
            "row {r}: {} vs {expect}",
            y[r]
        );
    }
}

#[test]
fn rmsnorm_reference() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w = [0.5f32, 1.0, 1.5, 2.0];
    let mut out = [0f32; 4];
    vec::rmsnorm(&mut out, &x, &w, 1e-5);
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let scale = 1.0 / (ms + 1e-5).sqrt();
    for i in 0..4 {
        let expect = x[i] * scale * w[i];
        assert!((out[i] - expect).abs() < 1e-6);
    }
}

#[test]
fn softmax_reference() {
    let mut x = [0.0f32, 1.0, 2.0];
    vec::softmax(&mut x);
    let e = [1.0f32, 1.0f32.exp(), 2.0f32.exp()];
    let sum: f32 = e.iter().sum();
    for i in 0..3 {
        assert!((x[i] - e[i] / sum).abs() < 1e-6, "{:?}", x);
    }
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn swiglu_reference() {
    let mut gate = [1.0f32, -1.0];
    let up = [2.0f32, 3.0];
    vec::swiglu(&mut gate, &up);
    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let silu_m1 = -1.0 / (1.0 + 1.0f32.exp());
    assert!((gate[0] - silu1 * 2.0).abs() < 1e-6);
    assert!((gate[1] - silu_m1 * 3.0).abs() < 1e-6);
}

#[test]
fn rope_identity_at_pos0() {
    let mut freqs = vec![0f32; 32];
    rope::rope_freqs(&mut freqs, 64, 500000.0, None);
    let mut rng = Rng(5);
    let orig = rng.vec(64);
    let mut x = orig.clone();
    rope::apply(&mut x, 64, &freqs, 0, rope::RopeStyle::Adjacent);
    assert_eq!(x, orig);
}

#[test]
fn rope_rotation_matches_trig() {
    let head_dim = 8;
    let mut freqs = vec![0f32; head_dim / 2];
    rope::rope_freqs(&mut freqs, head_dim, 10000.0, None);
    assert!((freqs[0] - 1.0).abs() < 1e-6);

    let orig: Vec<f32> = (1..=head_dim).map(|v| v as f32).collect();
    let mut x = orig.clone();
    let pos = 3usize;
    rope::apply(&mut x, head_dim, &freqs, pos, rope::RopeStyle::Adjacent);
    for i in 0..head_dim / 2 {
        let angle = pos as f32 * freqs[i];
        let (a, b) = (orig[2 * i], orig[2 * i + 1]);
        let er = a * angle.cos() - b * angle.sin();
        let ei = a * angle.sin() + b * angle.cos();
        assert!((x[2 * i] - er).abs() < 1e-4);
        assert!((x[2 * i + 1] - ei).abs() < 1e-4);
    }
}

#[test]
fn rope_llama3_scaling_behaviour() {
    let head_dim = 64;
    let scaling = Llama3Scaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context: 8192.0,
    };
    let mut plain = vec![0f32; head_dim / 2];
    let mut scaled = vec![0f32; head_dim / 2];
    rope::rope_freqs(&mut plain, head_dim, 500000.0, None);
    rope::rope_freqs(&mut scaled, head_dim, 500000.0, Some(scaling));
    // High-frequency (short wavelength) pairs are untouched.
    assert_eq!(plain[0], scaled[0]);
    // Lowest-frequency pair is divided by the full factor.
    let last = head_dim / 2 - 1;
    let wavelen = 2.0 * std::f32::consts::PI / plain[last];
    assert!(wavelen > 8192.0, "sanity: lowest freq is in the scaled regime");
    assert!((scaled[last] - plain[last] / 32.0).abs() < plain[last] * 1e-5);
    // Everything is monotonically decreasing.
    for w in scaled.windows(2) {
        assert!(w[1] < w[0]);
    }
}

#[test]
fn argmax_works() {
    assert_eq!(vec::argmax(&[0.1, 4.0, -2.0, 4.0]), 1);
}

/// Not a correctness test: prints matvec throughput. Run with
/// `cargo test -p nr-tensor --release -- --ignored --nocapture bench`.
#[test]
#[ignore]
fn bench_matvec() {
    let mut rng = Rng(99);
    let (rows, cols) = (2048usize, 2048usize);
    let w = quantized(&rng.vec(rows * cols));
    let x = quantized(&rng.vec(cols));
    let mut y = vec![0f32; rows];

    let reps = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        matvec_q8(&mut y, &w, &x, rows, cols);
        std::hint::black_box(&y);
    }
    let dt = t0.elapsed();
    let bytes = rows * cols + rows * cols / 16; // ~1.06 B/weight in Q8_0
    let gbs = bytes as f64 * reps as f64 / dt.as_secs_f64() / 1e9;
    let gflops = 2.0 * (rows * cols) as f64 * reps as f64 / dt.as_secs_f64() / 1e9;
    println!(
        "matvec {rows}x{cols} (fast_path={}): {:.2} us/iter, {gbs:.1} GB/s, {gflops:.1} GFLOP/s",
        cpu::fast_path(),
        dt.as_micros() as f64 / reps as f64
    );

    // Scalar comparison.
    let t0 = std::time::Instant::now();
    let scalar_reps = 20;
    for _ in 0..scalar_reps {
        for r in 0..rows {
            y[r] = dot_q8_scalar(&w[r * cols / 32..(r + 1) * cols / 32], &x);
        }
        std::hint::black_box(&y);
    }
    let dts = t0.elapsed();
    println!(
        "scalar: {:.2} us/iter, speedup {:.1}x",
        dts.as_micros() as f64 / scalar_reps as f64,
        dts.as_secs_f64() / scalar_reps as f64 / (dt.as_secs_f64() / reps as f64)
    );
}

#[test]
fn matmul_q8_bit_equals_matvec() {
    let mut rng = Rng(202);
    let (rows, cols, batch) = (48, 256, 4);
    let w = quantized(&rng.vec(rows * cols));
    let xs = quantized(&rng.vec(batch * cols));
    let bpr = cols / QK8_0;

    let mut ym = vec![0f32; batch * rows];
    nr_tensor::kernels::matmul_q8(&mut ym, &w, &xs, rows, cols, batch);
    for b in 0..batch {
        let mut yv = vec![0f32; rows];
        matvec_q8(&mut yv, &w, &xs[b * bpr..(b + 1) * bpr], rows, cols);
        assert_eq!(&ym[b * rows..(b + 1) * rows], &yv[..], "batch {b}");
    }
}
