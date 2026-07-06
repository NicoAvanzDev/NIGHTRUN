//! Llama forward pass over .nrm weights: RMSNorm -> GQA attention with
//! rotary embeddings and an f16 KV cache -> SwiGLU MLP, Q8_0 quantized
//! matvecs throughout. Single token per step; all buffers preallocated.

use nr_tensor::kernels::{axpy_f16, dot_f16, matvec_q8};
use nr_tensor::q8::{self, BlockQ8_0, QK8_0};
use nr_tensor::{f32_to_f16, rope, vec as tvec};

use crate::format::{Model, ParseError, TensorKind, TensorView};

pub struct LayerWeights<'m> {
    pub attn_norm: &'m [f32],
    pub wq: &'m [BlockQ8_0],
    pub wk: &'m [BlockQ8_0],
    pub wv: &'m [BlockQ8_0],
    pub wo: &'m [BlockQ8_0],
    pub ffn_norm: &'m [f32],
    pub w_gate: &'m [BlockQ8_0],
    pub w_up: &'m [BlockQ8_0],
    pub w_down: &'m [BlockQ8_0],
}

pub struct Weights<'m> {
    pub embed: &'m [BlockQ8_0],
    pub layers: alloc::vec::Vec<LayerWeights<'m>>,
    pub out_norm: &'m [f32],
    /// Classifier; equals `embed` for tied models.
    pub output: &'m [BlockQ8_0],
}

/// Dimensions captured as plain usizes.
#[derive(Clone, Copy)]
pub struct Dims {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
}

pub struct InferCtx<'m> {
    pub dims: Dims,
    pub weights: Weights<'m>,
    norm_eps: f32,
    freqs: &'static mut [f32],
    // Scratch (all preallocated; generation never allocates).
    x: &'static mut [f32],
    xb: &'static mut [f32],
    xb2: &'static mut [f32],
    q: &'static mut [f32],
    k: &'static mut [f32],
    v: &'static mut [f32],
    att: &'static mut [f32],
    gate: &'static mut [f32],
    up: &'static mut [f32],
    logits: &'static mut [f32],
    xq: &'static mut [BlockQ8_0],
    fq: &'static mut [BlockQ8_0],
    /// f16 bits, [n_layers][ctx][kv_dim].
    key_cache: &'static mut [u16],
    val_cache: &'static mut [u16],
    pub pos: usize,
}

/// Allocation callback: (bytes, align) -> pointer valid for 'static.
pub type AllocFn<'f> = &'f mut dyn FnMut(usize, usize) -> *mut u8;

fn take<T>(alloc: AllocFn, n: usize) -> &'static mut [T] {
    let bytes = n * core::mem::size_of::<T>();
    let ptr = alloc(bytes, core::mem::align_of::<T>().max(64));
    assert!(!ptr.is_null(), "inference arena exhausted");
    // SAFETY: the callback contract hands us exclusive, zeroed, aligned
    // memory that lives for the session.
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, n) }
}

impl<'m> InferCtx<'m> {
    /// Bytes of scratch + KV needed for a given context length (for
    /// sizing arenas before construction).
    pub fn required_bytes(model: &Model, ctx: usize) -> usize {
        let m = &model.meta;
        let (dim, ffn, vocab) = (m.dim as usize, m.ffn_dim as usize, m.vocab as usize);
        let kv_dim = (m.n_kv_heads * m.head_dim) as usize;
        let f32s = dim * 4 + kv_dim * 2 + (m.n_heads as usize) * ctx + ffn * 2 + vocab
            + (m.head_dim as usize) / 2;
        let q8s = (dim / QK8_0 + ffn / QK8_0) * core::mem::size_of::<BlockQ8_0>();
        let kv = 2 * (m.n_layers as usize) * ctx * kv_dim * 2;
        f32s * 4 + q8s + kv + 4096 // slack for alignment
    }

    pub fn new(model: &'m Model<'m>, ctx: usize, alloc: AllocFn) -> Result<InferCtx<'m>, ParseError> {
        let m = &model.meta;
        let dims = Dims {
            dim: m.dim as usize,
            n_layers: m.n_layers as usize,
            n_heads: m.n_heads as usize,
            n_kv_heads: m.n_kv_heads as usize,
            head_dim: m.head_dim as usize,
            kv_dim: (m.n_kv_heads * m.head_dim) as usize,
            ffn_dim: m.ffn_dim as usize,
            vocab: m.vocab as usize,
            ctx,
        };

        let f32_view = |v: TensorView<'m>| v.f32();
        let q8_view = |v: TensorView<'m>| v.q8();

        let embed = q8_view(model.tensor(TensorKind::TokEmbed, 0)?);
        let mut layers = alloc::vec::Vec::with_capacity(dims.n_layers);
        for l in 0..dims.n_layers {
            layers.push(LayerWeights {
                attn_norm: f32_view(model.tensor(TensorKind::AttnNorm, l)?),
                wq: q8_view(model.tensor(TensorKind::AttnQ, l)?),
                wk: q8_view(model.tensor(TensorKind::AttnK, l)?),
                wv: q8_view(model.tensor(TensorKind::AttnV, l)?),
                wo: q8_view(model.tensor(TensorKind::AttnO, l)?),
                ffn_norm: f32_view(model.tensor(TensorKind::FfnNorm, l)?),
                w_gate: q8_view(model.tensor(TensorKind::FfnGate, l)?),
                w_up: q8_view(model.tensor(TensorKind::FfnUp, l)?),
                w_down: q8_view(model.tensor(TensorKind::FfnDown, l)?),
            });
        }
        let out_norm = f32_view(model.tensor(TensorKind::OutputNorm, 0)?);
        let output = if model.meta.flags & crate::format::FLAG_TIED_EMBEDDINGS != 0 {
            embed
        } else {
            q8_view(model.tensor(TensorKind::Output, 0)?)
        };
        let weights = Weights { embed, layers, out_norm, output };

        // Rope frequencies: base theta spectrum, then either the GGUF's
        // precomputed frequency factors (divisors) or the header's Llama-3
        // scaling parameters.
        let freqs: &'static mut [f32] = take(alloc, dims.head_dim / 2);
        let scaling = (m.rope_factor > 0.0).then_some(rope::Llama3Scaling {
            factor: m.rope_factor,
            low_freq_factor: m.rope_low,
            high_freq_factor: m.rope_high,
            original_context: m.rope_orig_ctx,
        });
        if model.has_tensor(TensorKind::RopeFreqs) {
            rope::rope_freqs(freqs, dims.head_dim, m.rope_theta, None);
            let factors = model.tensor(TensorKind::RopeFreqs, 0)?.f32();
            for (f, &d) in freqs.iter_mut().zip(factors) {
                *f /= d;
            }
        } else {
            rope::rope_freqs(freqs, dims.head_dim, m.rope_theta, scaling);
        }

        Ok(InferCtx {
            norm_eps: m.norm_eps,
            freqs,
            x: take(alloc, dims.dim),
            xb: take(alloc, dims.dim),
            xb2: take(alloc, dims.dim),
            q: take(alloc, dims.dim),
            k: take(alloc, dims.kv_dim),
            v: take(alloc, dims.kv_dim),
            att: take(alloc, dims.n_heads * ctx),
            gate: take(alloc, dims.ffn_dim),
            up: take(alloc, dims.ffn_dim),
            logits: take(alloc, dims.vocab),
            xq: take(alloc, dims.dim / QK8_0),
            fq: take(alloc, dims.ffn_dim / QK8_0),
            key_cache: take(alloc, dims.n_layers * ctx * dims.kv_dim),
            val_cache: take(alloc, dims.n_layers * ctx * dims.kv_dim),
            dims,
            weights,
            pos: 0,
        })
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Logits from the most recent `forward` call.
    pub fn logits(&self) -> &[f32] {
        self.logits
    }

    /// Remaining context slots.
    pub fn remaining(&self) -> usize {
        self.dims.ctx - self.pos
    }

    /// Run one token through the model at the current position, advancing
    /// it. Returns the logits over the vocabulary.
    pub fn forward(&mut self, token: u32) -> &[f32] {
        let d = self.dims;
        let pos = self.pos;
        assert!(pos < d.ctx, "context window exhausted");

        // Token embedding (dequantized Q8_0 row).
        let bpr = d.dim / QK8_0;
        let row = &self.weights.embed[token as usize * bpr..(token as usize + 1) * bpr];
        q8::dequantize(row, self.x);

        for l in 0..d.n_layers {
            let w = &self.weights.layers[l];

            // Attention block.
            tvec::rmsnorm(self.xb, self.x, w.attn_norm, self.norm_eps);
            q8::quantize(self.xb, self.xq);
            matvec_q8(self.q, w.wq, self.xq, d.dim, d.dim);
            matvec_q8(self.k, w.wk, self.xq, d.kv_dim, d.dim);
            matvec_q8(self.v, w.wv, self.xq, d.kv_dim, d.dim);
            rope::apply(self.q, d.head_dim, self.freqs, pos);
            rope::apply(self.k, d.head_dim, self.freqs, pos);

            // Append K/V to the cache as f16.
            let cache_row = (l * d.ctx + pos) * d.kv_dim;
            for i in 0..d.kv_dim {
                self.key_cache[cache_row + i] = f32_to_f16(self.k[i]);
                self.val_cache[cache_row + i] = f32_to_f16(self.v[i]);
            }

            // Multi-head attention against the cache (GQA: kv head shared
            // by n_heads / n_kv_heads query heads).
            let gqa = d.n_heads / d.n_kv_heads;
            let scale = 1.0 / libm::sqrtf(d.head_dim as f32);
            for h in 0..d.n_heads {
                let qh = &self.q[h * d.head_dim..(h + 1) * d.head_dim];
                let kvh = (h / gqa) * d.head_dim;
                let att = &mut self.att[h * d.ctx..h * d.ctx + pos + 1];
                for (t, a) in att.iter_mut().enumerate() {
                    let krow = (l * d.ctx + t) * d.kv_dim + kvh;
                    *a = dot_f16(&self.key_cache[krow..krow + d.head_dim], qh) * scale;
                }
                tvec::softmax(att);
                let out = &mut self.xb[h * d.head_dim..(h + 1) * d.head_dim];
                out.fill(0.0);
                for (t, &a) in att.iter().enumerate() {
                    let vrow = (l * d.ctx + t) * d.kv_dim + kvh;
                    axpy_f16(out, a, &self.val_cache[vrow..vrow + d.head_dim]);
                }
            }

            q8::quantize(self.xb, self.xq);
            matvec_q8(self.xb2, w.wo, self.xq, d.dim, d.dim);
            tvec::add_assign(self.x, self.xb2);

            // MLP block (SwiGLU).
            tvec::rmsnorm(self.xb, self.x, w.ffn_norm, self.norm_eps);
            q8::quantize(self.xb, self.xq);
            matvec_q8(self.gate, w.w_gate, self.xq, d.ffn_dim, d.dim);
            matvec_q8(self.up, w.w_up, self.xq, d.ffn_dim, d.dim);
            tvec::swiglu(self.gate, self.up);
            q8::quantize(self.gate, self.fq);
            matvec_q8(self.xb2, w.w_down, self.fq, d.dim, d.ffn_dim);
            tvec::add_assign(self.x, self.xb2);
        }

        // Final norm + classifier.
        tvec::rmsnorm(self.xb, self.x, self.weights.out_norm, self.norm_eps);
        q8::quantize(self.xb, self.xq);
        matvec_q8(self.logits, self.weights.output, self.xq, d.vocab, d.dim);

        self.pos += 1;
        self.logits
    }
}
