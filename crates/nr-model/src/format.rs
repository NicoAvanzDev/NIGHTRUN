//! .nrm binary format: parsing and in-place tensor views.
//!
//! Layout (all little-endian):
//!   [0..)    header (fixed size, see FIELD OFFSETS below)
//!   tok_off  tokenizer blob (see nr-token)
//!   tab_off  tensor table: N * 32-byte entries
//!   data_off tensor data, each tensor 64-byte aligned
//!
//! `meta_crc` covers header (with the crc fields zeroed) + tokenizer blob +
//! tensor table. `data_crc` covers the data section.

use alloc::vec::Vec;

pub const MAGIC: [u8; 4] = *b"NRUN";
// v4 added YaRN; v5 added PQ2_0; v6 retires dtype 4. Layout stays fixed.
pub const VERSION: u32 = 6;

pub fn supported_version(version: u32) -> bool {
    matches!(version, 3..=VERSION)
}
pub const HEADER_SIZE: usize = 192;
pub const NAME_LEN: usize = 48;
pub const ENTRY_SIZE: usize = 32;
pub const DATA_ALIGN: usize = 64;

pub const FLAG_TIED_EMBEDDINGS: u32 = 1;
pub const FLAG_ROPE_YARN: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Arch {
    Llama3 = 1,
    Qwen3 = 2,
    Granite = 3,
}

impl Arch {
    pub fn from_u32(v: u32) -> Option<Arch> {
        match v {
            1 => Some(Arch::Llama3),
            2 => Some(Arch::Qwen3),
            3 => Some(Arch::Granite),
            _ => None,
        }
    }

    /// Chat display label for the assistant.
    pub fn assistant_label(self) -> &'static str {
        match self {
            Arch::Llama3 => "llama",
            Arch::Qwen3 => "qwen",
            Arch::Granite => "granite",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TensorKind {
    TokEmbed = 1,
    OutputNorm = 2,
    Output = 3,
    RopeFreqs = 4,
    AttnNorm = 10,
    AttnQ = 11,
    AttnK = 12,
    AttnV = 13,
    AttnO = 14,
    FfnNorm = 15,
    FfnGate = 16,
    FfnUp = 17,
    FfnDown = 18,
    AttnQNorm = 19,
    AttnKNorm = 20,
}

impl TensorKind {
    pub fn from_u16(v: u16) -> Option<TensorKind> {
        use TensorKind::*;
        Some(match v {
            1 => TokEmbed,
            2 => OutputNorm,
            3 => Output,
            4 => RopeFreqs,
            10 => AttnNorm,
            11 => AttnQ,
            12 => AttnK,
            13 => AttnV,
            14 => AttnO,
            15 => FfnNorm,
            16 => FfnGate,
            17 => FfnUp,
            18 => FfnDown,
            19 => AttnQNorm,
            20 => AttnKNorm,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TensorDtype {
    F32 = 0,
    Q8_0 = 1,
    Q4K = 2,
    Q6K = 3,
    // ID 4 is retired; do not reuse it for another encoding.
    PQ2_0 = 5,
}

impl TensorDtype {
    /// Bytes required for `n` elements (n must divide the block size).
    /// Checked: absurd element counts return None instead of wrapping.
    pub fn byte_size(self, n: u64) -> Option<u64> {
        match self {
            TensorDtype::F32 => n.checked_mul(4),
            TensorDtype::PQ2_0 => n
                .is_multiple_of(128)
                .then_some(n / 128)
                .and_then(|b| b.checked_mul(34)),
            TensorDtype::Q8_0 => n
                .is_multiple_of(32)
                .then_some(n / 32)
                .and_then(|b| b.checked_mul(34)),
            TensorDtype::Q4K => n
                .is_multiple_of(256)
                .then_some(n / 256)
                .and_then(|b| b.checked_mul(144)),
            TensorDtype::Q6K => n
                .is_multiple_of(256)
                .then_some(n / 256)
                .and_then(|b| b.checked_mul(210)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub arch: Arch,
    pub dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    pub vocab: u32,
    pub ctx_train: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// RoPE scaling; factor == 0 means none. FLAG_ROPE_YARN selects
    /// YaRN, with rope_low/high holding beta_fast/slow instead.
    pub rope_factor: f32,
    pub rope_low: f32,
    pub rope_high: f32,
    pub rope_orig_ctx: f32,
    /// YaRN magnitude multiplier (v4 offset 164; neutral 1.0 for v3).
    pub rope_attn_factor: f32,
    pub flags: u32,
    /// muP-style scalars (Granite); neutral values elsewhere.
    /// embed_scale: multiplies the token embedding (neutral 1.0).
    pub embed_scale: f32,
    /// Attention score scale; 0.0 means "use 1/sqrt(head_dim)".
    pub attn_scale: f32,
    /// Residual branch multiplier (neutral 1.0).
    pub residual_scale: f32,
    /// Logits are divided by this (neutral 1.0).
    pub logit_scale: f32,
    /// Display name, NUL-padded UTF-8.
    pub name: [u8; NAME_LEN],
}

impl Meta {
    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(NAME_LEN);
        core::str::from_utf8(&self.name[..end]).unwrap_or("model")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TensorEntry {
    pub kind: u16,
    pub layer: u16,
    pub dtype: TensorDtype,
    /// Offset from the start of the data section.
    pub offset: u64,
    pub size: u64,
    pub rows: u32,
    pub cols: u32,
}

#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: TensorDtype,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn f32(&self) -> &'a [f32] {
        assert_eq!(self.dtype, TensorDtype::F32);
        // Belt over the parse-time alignment gate: the reinterpretation
        // below is UB on a misaligned pointer.
        assert_eq!(
            self.bytes.as_ptr() as usize % core::mem::align_of::<f32>(),
            0
        );
        // SAFETY: length is a multiple of 4 by construction (byte_size),
        // alignment asserted above; f32 accepts any bit pattern.
        unsafe {
            core::slice::from_raw_parts(self.bytes.as_ptr() as *const f32, self.bytes.len() / 4)
        }
    }

    pub fn pq2(&self) -> &'a [nr_tensor::pq2::BlockPQ2_0] {
        assert_eq!(self.dtype, TensorDtype::PQ2_0);
        nr_tensor::pq2::cast_blocks(self.bytes)
    }

    pub fn q8(&self) -> &'a [nr_tensor::BlockQ8_0] {
        assert_eq!(self.dtype, TensorDtype::Q8_0);
        nr_tensor::q8::cast_blocks(self.bytes)
    }

    pub fn q4k(&self) -> &'a [nr_tensor::kquant::BlockQ4K] {
        assert_eq!(self.dtype, TensorDtype::Q4K);
        nr_tensor::kquant::cast_q4k(self.bytes)
    }

    pub fn q6k(&self) -> &'a [nr_tensor::kquant::BlockQ6K] {
        assert_eq!(self.dtype, TensorDtype::Q6K);
        nr_tensor::kquant::cast_q6k(self.bytes)
    }
}

#[derive(Debug)]
pub enum ParseError {
    TooShort,
    BadMagic,
    BadVersion(u32),
    MetaCrc { expect: u32, got: u32 },
    BadTable,
    MissingTensor(&'static str),
}

pub struct Model<'a> {
    pub meta: Meta,
    entries: Vec<TensorEntry>,
    pub tokenizer_blob: &'a [u8],
    data: &'a [u8],
    pub data_crc: u32,
    blob: &'a [u8],
}

struct Cursor<'a>(&'a [u8], usize);

impl<'a> Cursor<'a> {
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes(self.0[self.1..self.1 + 2].try_into().unwrap());
        self.1 += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        self.1 += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.0[self.1..self.1 + 8].try_into().unwrap());
        self.1 += 8;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
}

impl<'a> Model<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Model<'a>, ParseError> {
        if blob.len() < HEADER_SIZE {
            return Err(ParseError::TooShort);
        }
        if blob[0..4] != MAGIC {
            return Err(ParseError::BadMagic);
        }
        let mut c = Cursor(blob, 4);
        let version = c.u32();
        if !supported_version(version) {
            return Err(ParseError::BadVersion(version));
        }
        let arch = Arch::from_u32(c.u32()).ok_or(ParseError::BadTable)?;
        let mut meta = Meta {
            arch,
            dim: c.u32(),
            n_layers: c.u32(),
            n_heads: c.u32(),
            n_kv_heads: c.u32(),
            head_dim: c.u32(),
            ffn_dim: c.u32(),
            vocab: c.u32(),
            ctx_train: c.u32(),
            rope_theta: c.f32(),
            norm_eps: c.f32(),
            rope_factor: c.f32(),
            rope_low: c.f32(),
            rope_high: c.f32(),
            rope_orig_ctx: c.f32(),
            rope_attn_factor: 1.0,
            flags: c.u32(),
            embed_scale: c.f32(),
            attn_scale: c.f32(),
            residual_scale: c.f32(),
            logit_scale: c.f32(),
            name: {
                let mut n = [0u8; NAME_LEN];
                n.copy_from_slice(&blob[c.1..c.1 + NAME_LEN]);
                c.1 += NAME_LEN;
                n
            },
        };
        let tok_off = c.u64() as usize;
        let tok_size = c.u64() as usize;
        let table_off = c.u64() as usize;
        let tensor_count = c.u32() as usize;
        let rope_attn_factor = c.f32();
        if version >= 4 {
            meta.rope_attn_factor = rope_attn_factor;
        }
        let data_off = c.u64() as usize;
        let data_size = c.u64() as usize;
        let data_crc = c.u32();
        let meta_crc = c.u32();
        debug_assert_eq!(c.1, HEADER_SIZE);
        // Scalar coherence: reject NaN/inf/nonsensical values outright.
        if !meta.embed_scale.is_finite()
            || !meta.attn_scale.is_finite()
            || meta.attn_scale < 0.0
            || !meta.residual_scale.is_finite()
            || meta.residual_scale <= 0.0
            || !meta.logit_scale.is_finite()
            || meta.logit_scale <= 0.0
        {
            return Err(ParseError::BadTable);
        }

        if meta.flags & FLAG_ROPE_YARN != 0
            && (version < 4
                || meta.arch != Arch::Qwen3
                || !meta.rope_factor.is_finite()
                || meta.rope_factor < 1.0
                || !meta.rope_low.is_finite()
                || !meta.rope_high.is_finite()
                || meta.rope_high <= 0.0
                || meta.rope_low <= meta.rope_high
                || !meta.rope_orig_ctx.is_finite()
                || meta.rope_orig_ctx <= 0.0
                || !meta.rope_theta.is_finite()
                || meta.rope_theta <= 1.0
                || !meta.rope_attn_factor.is_finite()
                || meta.rope_attn_factor <= 0.0)
        {
            return Err(ParseError::BadTable);
        }

        // All region bounds use checked arithmetic: a crafted header must
        // produce a clean ParseError, never a wrapped sum that panics (or
        // worse) at slice time.
        let region_end = |off: usize, len: usize| off.checked_add(len);
        let table_len = tensor_count
            .checked_mul(ENTRY_SIZE)
            .ok_or(ParseError::BadTable)?;
        let table_end = region_end(table_off, table_len).ok_or(ParseError::TooShort)?;
        let data_end = region_end(data_off, data_size).ok_or(ParseError::TooShort)?;
        let tok_end = region_end(tok_off, tok_size).ok_or(ParseError::TooShort)?;
        if blob.len() < table_end || blob.len() < data_end || blob.len() < tok_end {
            return Err(ParseError::TooShort);
        }

        // Verify metadata CRC: header (crc fields zeroed) + tokenizer + table.
        let mut crc = crate::crc32::Crc32::new();
        crc.update(&blob[..HEADER_SIZE - 8]);
        crc.update(&[0u8; 8]);
        crc.update(&blob[tok_off..tok_end]);
        crc.update(&blob[table_off..table_end]);
        let got = crc.finish();
        if got != meta_crc {
            return Err(ParseError::MetaCrc {
                expect: meta_crc,
                got,
            });
        }

        let mut entries = Vec::with_capacity(tensor_count);
        let mut tc = Cursor(blob, table_off);
        for _ in 0..tensor_count {
            let kind = tc.u16();
            let layer = tc.u16();
            let dtype = match tc.u16() {
                0 => TensorDtype::F32,
                1 => TensorDtype::Q8_0,
                2 => TensorDtype::Q4K,
                3 => TensorDtype::Q6K,
                5 if version >= 5 => TensorDtype::PQ2_0,
                _ => return Err(ParseError::BadTable),
            };
            let _pad = tc.u16();
            let offset = tc.u64();
            let size = tc.u64();
            let rows = tc.u32();
            let cols = tc.u32();
            // Checked end + within the data section (no wrapping sums).
            let end = offset.checked_add(size).ok_or(ParseError::BadTable)?;
            if end > data_size as u64 {
                return Err(ParseError::BadTable);
            }
            // The converter 64-byte-aligns every tensor; TensorView::f32
            // relies on it for the &[f32] reinterpretation, so a
            // misaligned offset is a hard reject (UB guard, not style).
            if offset % DATA_ALIGN as u64 != 0 {
                return Err(ParseError::BadTable);
            }
            // Size must agree exactly with dtype block math (catches
            // malformed payloads and non-block-aligned dimensions).
            let n = (rows as u64)
                .checked_mul(cols as u64)
                .ok_or(ParseError::BadTable)?;
            if dtype.byte_size(n) != Some(size)
                || (dtype == TensorDtype::PQ2_0 && !cols.is_multiple_of(128))
            {
                return Err(ParseError::BadTable);
            }
            entries.push(TensorEntry {
                kind,
                layer,
                dtype,
                offset,
                size,
                rows,
                cols,
            });
        }

        Ok(Model {
            meta,
            entries,
            tokenizer_blob: &blob[tok_off..tok_end],
            data: &blob[data_off..data_end],
            data_crc,
            blob,
        })
    }

    /// Verify the tensor-data checksum, reporting progress via callback
    /// (bytes_done, bytes_total).
    pub fn verify_data(&self, mut progress: impl FnMut(usize, usize)) -> bool {
        let mut crc = crate::crc32::Crc32::new();
        let chunk = 32 * 1024 * 1024;
        let total = self.data.len();
        let mut done = 0;
        while done < total {
            let end = (done + chunk).min(total);
            crc.update(&self.data[done..end]);
            done = end;
            progress(done, total);
        }
        crc.finish() == self.data_crc
    }

    pub fn tensor(&self, kind: TensorKind, layer: usize) -> Result<TensorView<'a>, ParseError> {
        self.entries
            .iter()
            .find(|e| e.kind == kind as u16 && e.layer as usize == layer)
            .map(|e| TensorView {
                dtype: e.dtype,
                rows: e.rows as usize,
                cols: e.cols as usize,
                bytes: &self.data[e.offset as usize..(e.offset + e.size) as usize],
            })
            .ok_or(ParseError::MissingTensor(kind_name(kind)))
    }

    pub fn has_tensor(&self, kind: TensorKind) -> bool {
        self.entries.iter().any(|e| e.kind == kind as u16)
    }

    pub fn total_size(&self) -> usize {
        self.blob.len()
    }
}

fn kind_name(kind: TensorKind) -> &'static str {
    use TensorKind::*;
    match kind {
        TokEmbed => "token_embd",
        OutputNorm => "output_norm",
        Output => "output",
        RopeFreqs => "rope_freqs",
        AttnNorm => "attn_norm",
        AttnQ => "attn_q",
        AttnK => "attn_k",
        AttnV => "attn_v",
        AttnO => "attn_output",
        FfnNorm => "ffn_norm",
        FfnGate => "ffn_gate",
        FfnUp => "ffn_up",
        FfnDown => "ffn_down",
        AttnQNorm => "attn_q_norm",
        AttnKNorm => "attn_k_norm",
    }
}
