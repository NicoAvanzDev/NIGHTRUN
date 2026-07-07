//! Rotary position embeddings, adjacent-pair ("normal") convention as used
//! by GGUF-converted llama weights, with Llama-3 frequency scaling.

#[derive(Clone, Copy, Debug)]
pub struct Llama3Scaling {
    pub factor: f32,           // 32.0 for Llama 3.2
    pub low_freq_factor: f32,  // 1.0
    pub high_freq_factor: f32, // 4.0
    pub original_context: f32, // 8192.0
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

/// Pairing convention for the rotation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Adjacent pairs (x[2i], x[2i+1]) — llama-family GGUFs (converter
    /// permutes Q/K weights into this layout).
    Adjacent,
    /// Half-split pairs (x[i], x[i+d/2]) — NEOX convention, used by Qwen3.
    Neox,
}

/// Rotate `x` (one or more heads laid out contiguously) in place for
/// position `pos`. `freqs` holds head_dim/2 base frequencies.
#[allow(clippy::needless_range_loop)] // i indexes freqs and paired lanes
pub fn apply(x: &mut [f32], head_dim: usize, freqs: &[f32], pos: usize, style: RopeStyle) {
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let angle = pos as f32 * freqs[i];
            let (sin, cos) = libm::sincosf(angle);
            let (ia, ib) = match style {
                RopeStyle::Adjacent => (2 * i, 2 * i + 1),
                RopeStyle::Neox => (i, i + half),
            };
            let a = head[ia];
            let b = head[ib];
            head[ia] = a * cos - b * sin;
            head[ib] = a * sin + b * cos;
        }
    }
}
