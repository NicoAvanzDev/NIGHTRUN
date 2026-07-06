//! Rotary position embeddings, adjacent-pair ("normal") convention as used
//! by GGUF-converted llama weights, with Llama-3 frequency scaling.

#[derive(Clone, Copy, Debug)]
pub struct Llama3Scaling {
    pub factor: f32,               // 32.0 for Llama 3.2
    pub low_freq_factor: f32,      // 1.0
    pub high_freq_factor: f32,     // 4.0
    pub original_context: f32,     // 8192.0
}

/// Per-pair base frequencies for one head: freqs[i] applies to elements
/// (2i, 2i+1). Applies Llama-3 wavelength-dependent scaling when given.
pub fn rope_freqs(out: &mut [f32], head_dim: usize, theta: f32, scaling: Option<Llama3Scaling>) {
    let half = head_dim / 2;
    assert_eq!(out.len(), half);
    for (i, slot) in out.iter_mut().enumerate() {
        let exponent = -2.0 * i as f32 / head_dim as f32;
        let mut freq = libm::powf(theta, exponent);
        if let Some(s) = scaling {
            let wavelen = 2.0 * core::f32::consts::PI / freq;
            let low_wavelen = s.original_context / s.low_freq_factor;
            let high_wavelen = s.original_context / s.high_freq_factor;
            if wavelen > low_wavelen {
                freq /= s.factor;
            } else if wavelen > high_wavelen {
                let smooth = (s.original_context / wavelen - s.low_freq_factor)
                    / (s.high_freq_factor - s.low_freq_factor);
                freq = (1.0 - smooth) * freq / s.factor + smooth * freq;
            }
        }
        *slot = freq;
    }
}

/// Rotate `x` (one or more heads laid out contiguously) in place for
/// position `pos`. `freqs` holds head_dim/2 base frequencies.
pub fn apply(x: &mut [f32], head_dim: usize, freqs: &[f32], pos: usize) {
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..head_dim / 2 {
            let angle = pos as f32 * freqs[i];
            let (sin, cos) = libm::sincosf(angle);
            let a = head[2 * i];
            let b = head[2 * i + 1];
            head[2 * i] = a * cos - b * sin;
            head[2 * i + 1] = a * sin + b * cos;
        }
    }
}
