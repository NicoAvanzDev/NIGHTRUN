//! Llama-family forward pass over .nrm weights: RMSNorm -> GQA attention
//! with rotary embeddings and an f16 KV cache -> SwiGLU MLP, quantized
//! matvecs throughout. Supports the Llama-3 recipe and the Qwen3 variant
//! (per-head Q/K RMSNorm, attention width != hidden width).
//!
//! Weight matrices are dtype-tagged (Q8_0 / Q4_K / Q6_K); dispatch happens
//! once per matvec call, never inside row loops. Activations are quantized
//! to the matching format (Q8_0 or Q8_K). Single token per step; all
//! buffers preallocated — generation never allocates.

use nr_tensor::kernels::{axpy_f16, dot_f16, matvec_q8};
use nr_tensor::kquant::{self, BlockQ4K, BlockQ6K, BlockQ8K, QK_K};
use nr_tensor::q8::{self, BlockQ8_0, QK8_0};
use nr_tensor::{f32_to_f16, rope, vec as tvec};

use crate::format::{Arch, Model, ParseError, TensorDtype, TensorKind, TensorView};

/// A dtype-tagged weight matrix (rows x cols), viewed in place.
#[derive(Clone, Copy)]
pub enum QMat<'m> {
    Q8(&'m [BlockQ8_0]),
    Q4K(&'m [BlockQ4K]),
    Q6K(&'m [BlockQ6K]),
}

impl<'m> QMat<'m> {
    fn from_view(v: TensorView<'m>) -> Result<QMat<'m>, ParseError> {
        Ok(match v.dtype {
            TensorDtype::Q8_0 => QMat::Q8(v.q8()),
            TensorDtype::Q4K => QMat::Q4K(v.q4k()),
            TensorDtype::Q6K => QMat::Q6K(v.q6k()),
            TensorDtype::F32 => return Err(ParseError::BadTable),
        })
    }

    /// Dequantize row `r` (cols values) into `out`.
    fn dequant_row(&self, r: usize, cols: usize, out: &mut [f32]) {
        match self {
            QMat::Q8(blocks) => {
                let bpr = cols / QK8_0;
                q8::dequantize(&blocks[r * bpr..(r + 1) * bpr], out);
            }
            QMat::Q4K(blocks) => {
                let bpr = cols / QK_K;
                for (i, b) in blocks[r * bpr..(r + 1) * bpr].iter().enumerate() {
                    kquant::dequant_q4k(b, &mut out[i * QK_K..(i + 1) * QK_K]);
                }
            }
            QMat::Q6K(blocks) => {
                let bpr = cols / QK_K;
                for (i, b) in blocks[r * bpr..(r + 1) * bpr].iter().enumerate() {
                    kquant::dequant_q6k(b, &mut out[i * QK_K..(i + 1) * QK_K]);
                }
            }
        }
    }
}

/// Dimensions captured as plain usizes.
#[derive(Clone, Copy)]
pub struct Dims {
    pub dim: usize,
    /// n_heads * head_dim; equals `dim` for Llama, differs for Qwen3.
    pub att_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
}

pub struct LayerWeights<'m> {
    pub attn_norm: &'m [f32],
    /// Per-head RMSNorm weights (head_dim each); Qwen3 only.
    pub q_norm: Option<&'m [f32]>,
    pub k_norm: Option<&'m [f32]>,
    pub wq: QMat<'m>,
    pub wk: QMat<'m>,
    pub wv: QMat<'m>,
    pub wo: QMat<'m>,
    pub ffn_norm: &'m [f32],
    pub w_gate: QMat<'m>,
    pub w_up: QMat<'m>,
    pub w_down: QMat<'m>,
}

pub struct Weights<'m> {
    pub embed: QMat<'m>,
    pub layers: alloc::vec::Vec<LayerWeights<'m>>,
    pub out_norm: &'m [f32],
    /// Classifier; equals `embed` for tied models.
    pub output: QMat<'m>,
}

/// Scratch activation-quantization buffers; a matvec input is quantized
/// into the format its weight matrix needs.
struct Acts {
    q8: &'static mut [BlockQ8_0],
    q8k: &'static mut [BlockQ8K],
}

impl Acts {
    /// Quantize `x` and run `w.matvec` into y (rows = y.len(), cols = x.len()).
    fn matvec(&mut self, y: &mut [f32], w: QMat, x: &[f32]) {
        let cols = x.len();
        let rows = y.len();
        match w {
            QMat::Q8(blocks) => {
                let xq = &mut self.q8[..cols / QK8_0];
                q8::quantize(x, xq);
                matvec_q8(y, blocks, xq, rows, cols);
            }
            QMat::Q4K(blocks) => {
                let xk = &mut self.q8k[..cols / QK_K];
                kquant::quantize_q8k(x, xk);
                kquant::matvec_q4k(y, blocks, xk, rows, cols);
            }
            QMat::Q6K(blocks) => {
                let xk = &mut self.q8k[..cols / QK_K];
                kquant::quantize_q8k(x, xk);
                kquant::matvec_q6k(y, blocks, xk, rows, cols);
            }
        }
    }

    /// Run several matvecs that share one input vector, quantizing each
    /// needed format exactly once.
    fn matvec_group(&mut self, jobs: &mut [(&mut [f32], QMat)], x: &[f32]) {
        let cols = x.len();
        let mut q8_ready = false;
        let mut q8k_ready = false;
        for (y, w) in jobs.iter_mut() {
            match w {
                QMat::Q8(blocks) => {
                    if !q8_ready {
                        q8::quantize(x, &mut self.q8[..cols / QK8_0]);
                        q8_ready = true;
                    }
                    matvec_q8(y, blocks, &self.q8[..cols / QK8_0], y.len(), cols);
                }
                QMat::Q4K(blocks) => {
                    if !q8k_ready {
                        kquant::quantize_q8k(x, &mut self.q8k[..cols / QK_K]);
                        q8k_ready = true;
                    }
                    kquant::matvec_q4k(y, blocks, &self.q8k[..cols / QK_K], y.len(), cols);
                }
                QMat::Q6K(blocks) => {
                    if !q8k_ready {
                        kquant::quantize_q8k(x, &mut self.q8k[..cols / QK_K]);
                        q8k_ready = true;
                    }
                    kquant::matvec_q6k(y, blocks, &self.q8k[..cols / QK_K], y.len(), cols);
                }
            }
        }
    }
}

pub struct InferCtx<'m> {
    pub dims: Dims,
    pub weights: Weights<'m>,
    norm_eps: f32,
    rope_style: rope::RopeStyle,
    freqs: &'static mut [f32],
    // Scratch (all preallocated; generation never allocates).
    x: &'static mut [f32],
    xb: &'static mut [f32],
    xb2: &'static mut [f32],
    q: &'static mut [f32],
    attn_out: &'static mut [f32],
    k: &'static mut [f32],
    v: &'static mut [f32],
    att: &'static mut [f32],
    gate: &'static mut [f32],
    up: &'static mut [f32],
    logits: &'static mut [f32],
    acts: Acts,
    /// f16 bits, [n_layers][ctx][kv_dim].
    key_cache: &'static mut [u16],
    val_cache: &'static mut [u16],
    pub pos: usize,
}

/// Allocation callback: (bytes, align) -> pointer valid for 'static.
pub type AllocFn<'f> = &'f mut dyn FnMut(usize, usize) -> *mut u8;

fn take<T>(alloc: AllocFn, n: usize) -> &'static mut [T] {
    let bytes = (n * core::mem::size_of::<T>()).max(1);
    let ptr = alloc(bytes, core::mem::align_of::<T>().max(64));
    assert!(!ptr.is_null(), "inference arena exhausted");
    // SAFETY: the callback contract hands us exclusive, zeroed, aligned
    // memory that lives for the session.
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, n) }
}

fn dims_of(model: &Model, ctx: usize) -> Dims {
    let m = &model.meta;
    Dims {
        dim: m.dim as usize,
        att_dim: (m.n_heads * m.head_dim) as usize,
        n_layers: m.n_layers as usize,
        n_heads: m.n_heads as usize,
        n_kv_heads: m.n_kv_heads as usize,
        head_dim: m.head_dim as usize,
        kv_dim: (m.n_kv_heads * m.head_dim) as usize,
        ffn_dim: m.ffn_dim as usize,
        vocab: m.vocab as usize,
        ctx,
    }
}

impl<'m> InferCtx<'m> {
    /// Bytes of scratch + KV needed for a given context length (for
    /// sizing arenas before construction).
    pub fn required_bytes(model: &Model, ctx: usize) -> usize {
        let d = dims_of(model, ctx);
        let maxd = d.dim.max(d.att_dim).max(d.ffn_dim);
        let f32s = d.dim * 3 + d.att_dim * 2 + d.kv_dim * 2 + d.n_heads * ctx
            + d.ffn_dim * 2 + d.vocab + d.head_dim / 2;
        let acts = maxd / QK8_0 * core::mem::size_of::<BlockQ8_0>()
            + maxd / QK_K * core::mem::size_of::<BlockQ8K>();
        let kv = 2 * d.n_layers * ctx * d.kv_dim * 2;
        f32s * 4 + acts + kv + 64 * 32 // alignment slack
    }

    pub fn new(model: &'m Model<'m>, ctx: usize, alloc: AllocFn) -> Result<InferCtx<'m>, ParseError> {
        let m = &model.meta;
        let dims = dims_of(model, ctx);

        // Shape-checked tensor accessors.
        let f32_tensor = |kind: TensorKind, layer: usize, len: usize| -> Result<&'m [f32], ParseError> {
            let v = model.tensor(kind, layer)?;
            let s = v.f32();
            if s.len() != len {
                return Err(ParseError::MissingTensor("bad f32 tensor shape"));
            }
            Ok(s)
        };
        let mat = |kind: TensorKind, layer: usize, rows: usize, cols: usize| -> Result<QMat<'m>, ParseError> {
            let v = model.tensor(kind, layer)?;
            if v.rows != rows || v.cols != cols {
                return Err(ParseError::MissingTensor("bad matrix shape"));
            }
            QMat::from_view(v)
        };

        let embed = mat(TensorKind::TokEmbed, 0, dims.vocab, dims.dim)?;
        let is_qwen = m.arch == Arch::Qwen3;
        let mut layers = alloc::vec::Vec::with_capacity(dims.n_layers);
        for l in 0..dims.n_layers {
            layers.push(LayerWeights {
                attn_norm: f32_tensor(TensorKind::AttnNorm, l, dims.dim)?,
                q_norm: if is_qwen {
                    Some(f32_tensor(TensorKind::AttnQNorm, l, dims.head_dim)?)
                } else {
                    None
                },
                k_norm: if is_qwen {
                    Some(f32_tensor(TensorKind::AttnKNorm, l, dims.head_dim)?)
                } else {
                    None
                },
                wq: mat(TensorKind::AttnQ, l, dims.att_dim, dims.dim)?,
                wk: mat(TensorKind::AttnK, l, dims.kv_dim, dims.dim)?,
                wv: mat(TensorKind::AttnV, l, dims.kv_dim, dims.dim)?,
                wo: mat(TensorKind::AttnO, l, dims.dim, dims.att_dim)?,
                ffn_norm: f32_tensor(TensorKind::FfnNorm, l, dims.dim)?,
                w_gate: mat(TensorKind::FfnGate, l, dims.ffn_dim, dims.dim)?,
                w_up: mat(TensorKind::FfnUp, l, dims.ffn_dim, dims.dim)?,
                w_down: mat(TensorKind::FfnDown, l, dims.dim, dims.ffn_dim)?,
            });
        }
        let out_norm = f32_tensor(TensorKind::OutputNorm, 0, dims.dim)?;
        let output = if m.flags & crate::format::FLAG_TIED_EMBEDDINGS != 0 {
            embed
        } else {
            mat(TensorKind::Output, 0, dims.vocab, dims.dim)?
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

        let maxd = dims.dim.max(dims.att_dim).max(dims.ffn_dim);
        Ok(InferCtx {
            norm_eps: m.norm_eps,
            rope_style: match m.arch {
                Arch::Llama3 => rope::RopeStyle::Adjacent,
                Arch::Qwen3 => rope::RopeStyle::Neox,
            },
            freqs,
            x: take(alloc, dims.dim),
            xb: take(alloc, dims.dim),
            xb2: take(alloc, dims.dim),
            q: take(alloc, dims.att_dim),
            attn_out: take(alloc, dims.att_dim),
            k: take(alloc, dims.kv_dim),
            v: take(alloc, dims.kv_dim),
            att: take(alloc, dims.n_heads * ctx),
            gate: take(alloc, dims.ffn_dim),
            up: take(alloc, dims.ffn_dim),
            logits: take(alloc, dims.vocab),
            acts: Acts { q8: take(alloc, maxd / QK8_0), q8k: take(alloc, maxd / QK_K) },
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

        // Token embedding (dequantized row).
        self.weights.embed.dequant_row(token as usize, d.dim, self.x);

        for l in 0..d.n_layers {
            let w = &self.weights.layers[l];

            // Attention block.
            tvec::rmsnorm(self.xb, self.x, w.attn_norm, self.norm_eps);
            self.acts.matvec_group(
                &mut [(self.q, w.wq), (self.k, w.wk), (self.v, w.wv)],
                self.xb,
            );

            // Qwen3: per-head RMSNorm on Q and K, before RoPE.
            if let (Some(qn), Some(kn)) = (w.q_norm, w.k_norm) {
                for head in self.q.chunks_exact_mut(d.head_dim) {
                    tvec::rmsnorm_inplace(head, qn, self.norm_eps);
                }
                for head in self.k.chunks_exact_mut(d.head_dim) {
                    tvec::rmsnorm_inplace(head, kn, self.norm_eps);
                }
            }

            rope::apply(self.q, d.head_dim, self.freqs, pos, self.rope_style);
            rope::apply(self.k, d.head_dim, self.freqs, pos, self.rope_style);

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
                let out = &mut self.attn_out[h * d.head_dim..(h + 1) * d.head_dim];
                out.fill(0.0);
                for (t, &a) in att.iter().enumerate() {
                    let vrow = (l * d.ctx + t) * d.kv_dim + kvh;
                    axpy_f16(out, a, &self.val_cache[vrow..vrow + d.head_dim]);
                }
            }

            self.acts.matvec(self.xb2, w.wo, self.attn_out);
            tvec::add_assign(self.x, self.xb2);

            // MLP block (SwiGLU).
            tvec::rmsnorm(self.xb, self.x, w.ffn_norm, self.norm_eps);
            self.acts.matvec_group(&mut [(self.gate, w.w_gate), (self.up, w.w_up)], self.xb);
            tvec::swiglu(self.gate, self.up);
            self.acts.matvec(self.xb2, w.w_down, self.gate);
            tvec::add_assign(self.x, self.xb2);
        }

        // Final norm + classifier.
        tvec::rmsnorm(self.xb, self.x, self.weights.out_norm, self.norm_eps);
        self.acts.matvec(self.logits, self.weights.output, self.xb);

        self.pos += 1;
        self.logits
    }
}
