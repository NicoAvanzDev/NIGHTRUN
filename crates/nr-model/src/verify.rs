//! Streaming .nrm verification: checksums are updated while chunks stream
//! from disk into RAM, so integrity checking costs no second pass over the
//! model after loading.
//!
//! The format has two section CRCs — meta (header with its two crc fields
//! zeroed + tokenizer blob + tensor table) and data (the data section) —
//! so the verifier routes each incoming byte range to the right hasher.
//! Chunks must arrive in file order (offset strictly increasing), which is
//! how the loader reads.

use crate::crc32::Crc32;
use crate::format::{supported_version, HEADER_SIZE, MAGIC};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// Header magic/version wrong (not a supported .nrm).
    BadHeader,
    /// Header/tokenizer/tensor-table checksum mismatch.
    MetaCrc,
    /// Tensor data checksum mismatch.
    DataCrc,
}

pub struct StreamingVerifier {
    meta: Crc32,
    data: Crc32,
    /// Meta region: [0, meta_end) in file order, with bytes
    /// [crc_field_off, crc_field_off+8) replaced by zeros.
    meta_end: usize,
    data_off: usize,
    data_end: usize,
    expect_meta: u32,
    expect_data: u32,
}

const CRC_FIELDS_OFF: usize = HEADER_SIZE - 8;

impl StreamingVerifier {
    /// Build from the first bytes of the file (at least HEADER_SIZE).
    pub fn new(header: &[u8]) -> Result<StreamingVerifier, VerifyError> {
        if header.len() < HEADER_SIZE || header[0..4] != MAGIC {
            return Err(VerifyError::BadHeader);
        }
        let u32at = |off: usize| u32::from_le_bytes(header[off..off + 4].try_into().unwrap());
        let u64at = |off: usize| u64::from_le_bytes(header[off..off + 8].try_into().unwrap());
        if !supported_version(u32at(4)) {
            return Err(VerifyError::BadHeader);
        }
        // Offsets per format v3-v6 layout (see nr-model::format).
        let tok_off = u64at(136) as usize;
        let tok_size = u64at(144) as usize;
        let table_off = u64at(152) as usize;
        let tensor_count = u32at(160) as usize;
        let data_off = u64at(168) as usize;
        let data_size = u64at(176) as usize;
        let data_crc = u32at(184);
        let meta_crc = u32at(188);

        // The meta region is contiguous in practice (header | tok | table),
        // which streaming relies on; validate that shape.
        let table_end = tensor_count
            .checked_mul(crate::format::ENTRY_SIZE)
            .and_then(|len| table_off.checked_add(len))
            .ok_or(VerifyError::BadHeader)?;
        let tok_end = tok_off
            .checked_add(tok_size)
            .ok_or(VerifyError::BadHeader)?;
        if tok_off != HEADER_SIZE || table_off != tok_end || table_end > data_off {
            return Err(VerifyError::BadHeader);
        }

        Ok(StreamingVerifier {
            meta: Crc32::new(),
            data: Crc32::new(),
            meta_end: table_end,
            data_off,
            data_end: data_off + data_size,
            expect_meta: meta_crc,
            expect_data: data_crc,
        })
    }

    /// Feed the chunk that starts at `offset` in the file. Chunks must be
    /// fed in order without gaps or overlaps.
    pub fn feed(&mut self, offset: usize, chunk: &[u8]) {
        let end = offset + chunk.len();

        // Meta region, with the two crc fields hashed as zeros.
        let m0 = offset.min(self.meta_end);
        let m1 = end.min(self.meta_end);
        if m0 < m1 {
            let seg = &chunk[m0 - offset..m1 - offset];
            let zeros_start = CRC_FIELDS_OFF.max(m0);
            let zeros_end = (CRC_FIELDS_OFF + 8).min(m1);
            if zeros_start < zeros_end {
                self.meta.update(&seg[..zeros_start - m0]);
                self.meta.update(&[0u8; 8][..zeros_end - zeros_start]);
                self.meta.update(&seg[zeros_end - m0..]);
            } else {
                self.meta.update(seg);
            }
        }

        // Data region.
        let d0 = offset.max(self.data_off).min(self.data_end);
        let d1 = end.max(self.data_off).min(self.data_end);
        if d0 < d1 {
            self.data.update(&chunk[d0 - offset..d1 - offset]);
        }
    }

    pub fn finish(self) -> Result<(), VerifyError> {
        if self.meta.finish() != self.expect_meta {
            return Err(VerifyError::MetaCrc);
        }
        if self.data.finish() != self.expect_data {
            return Err(VerifyError::DataCrc);
        }
        Ok(())
    }
}
