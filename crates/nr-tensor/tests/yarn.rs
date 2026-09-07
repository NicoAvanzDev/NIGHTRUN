use nr_tensor::rope::{self, RopeStyle, YarnScaling};

#[test]
fn bonsai_yarn_frequency_and_magnitude_reference() {
    let mut freqs = [0.0; 64];
    let magnitude = rope::yarn_freqs(
        &mut freqs,
        128,
        1_000_000.0,
        YarnScaling {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context: 16384.0,
        },
    );
    // Independent double-precision evaluation of ggml's correction ramp:
    // floor(corr(32)) = 20, ceil(corr(1)) = 37.
    for (i, expected) in [
        (0, 1.0f32),
        (16, 0.031_622_775),
        (20, 0.013_335_214),
        (24, 0.004_631_046),
        (28, 0.0015344183),
        (32, 0.0004705882),
        (40, 0.000044456985),
        (63, 0.00000031023444),
    ] {
        assert!((freqs[i] - expected).abs() < expected * 1e-6, "pair {i}");
    }
    assert!((magnitude - 1.1386294).abs() < 1e-7);
    // Even position zero needs the YaRN amplitude correction on Q and K.
    let mut q = [1.0; 256];
    rope::apply_scaled(&mut q, 128, &freqs, 0, RopeStyle::Neox, magnitude);
    assert_eq!(q, [magnitude; 256]);
}

#[test]
fn factor_one_is_ordinary_rope() {
    let mut yarn = [0.0; 64];
    let magnitude = rope::yarn_freqs(
        &mut yarn,
        128,
        1_000_000.0,
        YarnScaling {
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context: 16384.0,
        },
    );
    let mut base = [0.0; 64];
    rope::rope_freqs(&mut base, 128, 1_000_000.0, None);
    assert_eq!(magnitude, 1.0);
    for (a, b) in yarn.iter().zip(base) {
        assert!((a - b).abs() < 1e-7);
    }
}
