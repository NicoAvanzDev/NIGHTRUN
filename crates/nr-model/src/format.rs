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
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 172;
pub const NAME_LEN: usize = 48;
pub const ENTRY_SIZE: usize = 32;
pub const DATA_ALIGN: usize = 64;

pub const FLAG_TIED_EMBEDDINGS: u32 = 1;

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
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TensorDtype {
    F32 = 0,
    Q8_0 = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
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
    /// Llama-3 rope scaling; factor == 0 means none.
    pub rope_factor: f32,
    pub rope_low: f32,
    pub rope_high: f32,
    pub rope_orig_ctx: f32,
    pub flags: u32,
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
        // SAFETY: converter emits 64-byte-aligned little-endian f32 data.
        unsafe {
            core::slice::from_raw_parts(self.bytes.as_ptr() as *const f32, self.bytes.len() / 4)
        }
    }

    pub fn q8(&self) -> &'a [nr_tensor::BlockQ8_0] {
        assert_eq!(self.dtype, TensorDtype::Q8_0);
        nr_tensor::q8::cast_blocks(self.bytes)
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
        if version != VERSION {
            return Err(ParseError::BadVersion(version));
        }
        let meta = Meta {
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
            flags: c.u32(),
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
        let _pad = c.u32();
        let data_off = c.u64() as usize;
        let data_size = c.u64() as usize;
        let data_crc = c.u32();
        let meta_crc = c.u32();
        debug_assert_eq!(c.1, HEADER_SIZE);

        let table_end = table_off + tensor_count * ENTRY_SIZE;
        if blob.len() < table_end || blob.len() < data_off + data_size || blob.len() < tok_off + tok_size
        {
            return Err(ParseError::TooShort);
        }

        // Verify metadata CRC: header (crc fields zeroed) + tokenizer + table.
        let mut crc = crate::crc32::Crc32::new();
        crc.update(&blob[..HEADER_SIZE - 8]);
        crc.update(&[0u8; 8]);
        crc.update(&blob[tok_off..tok_off + tok_size]);
        crc.update(&blob[table_off..table_end]);
        let got = crc.finish();
        if got != meta_crc {
            return Err(ParseError::MetaCrc { expect: meta_crc, got });
        }

        let mut entries = Vec::with_capacity(tensor_count);
        let mut tc = Cursor(blob, table_off);
        for _ in 0..tensor_count {
            let kind = tc.u16();
            let layer = tc.u16();
            let dtype = match tc.u16() {
                0 => TensorDtype::F32,
                1 => TensorDtype::Q8_0,
                _ => return Err(ParseError::BadTable),
            };
            let _pad = tc.u16();
            let offset = tc.u64();
            let size = tc.u64();
            let rows = tc.u32();
            let cols = tc.u32();
            if offset as usize + size as usize > data_size {
                return Err(ParseError::BadTable);
            }
            entries.push(TensorEntry { kind, layer, dtype, offset, size, rows, cols });
        }

        Ok(Model {
            meta,
            entries,
            tokenizer_blob: &blob[tok_off..tok_off + tok_size],
            data: &blob[data_off..data_off + data_size],
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
    }
}
