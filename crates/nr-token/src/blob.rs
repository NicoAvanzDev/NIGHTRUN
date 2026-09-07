//! Tokenizer blob layout (little-endian), embedded in .nrm:
//!
//!   magic "NRTK", u32 version = 3 (v2 remains readable)
//!   u32 template (1 = llama3 header-style, 2 = chatml, 3 = granite, 4 = bonsai)
//!   u32 vocab_count, u32 merge_count, u32 pool_size
//!   u32 bos, eos, eot, start_header, end_header, end_of_text
//!   [u32; 256]  byte -> token id
//!   vocab_count * { u32 pool_off, u16 len, u16 flags }   token table
//!   merge_count * { u32 left, u32 right, u32 result, u32 rank }
//!       (sorted by (left, right) for binary search)
//!   pool bytes (raw token bytes)

use alloc::vec::Vec;

pub const MAGIC: [u8; 4] = *b"NRTK";
pub const VERSION: u32 = 3;

pub const FLAG_CONTROL: u16 = 1;

/// Chat template family baked into the blob by the converter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    /// Llama-3 instruct: <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|>
    Llama3,
    /// ChatML (Qwen): <|im_start|>role\n ... <|im_end|>\n  (no BOS)
    ChatMl,
    /// Bonsai ChatML with an empty thinking block in the generation header.
    /// Unused ChatML slots bos/end_header hold </think>/<think> respectively.
    Bonsai,
    /// Granite: <|start_of_role|>role<|end_of_role|>content<|end_of_text|>\n (no BOS)
    Granite,
}

#[derive(Clone, Copy, Debug)]
pub struct Specials {
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub start_header: u32,
    pub end_header: u32,
    pub end_of_text: u32,
}

pub struct Tokenizer<'a> {
    pub specials: Specials,
    pub template: Template,
    vocab_count: u32,
    byte_ids: &'a [u8], // 256 * u32
    table: &'a [u8],    // vocab_count * 8
    merges: &'a [u8],   // merge_count * 16
    merge_count: usize,
    pool: &'a [u8],
}

#[derive(Debug)]
pub enum Error {
    TooShort,
    BadMagic,
    BadVersion,
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

impl<'a> Tokenizer<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Tokenizer<'a>, Error> {
        if blob.len() < 8 + 12 + 24 + 1024 {
            return Err(Error::TooShort);
        }
        if blob[0..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        if !matches!(u32le(blob, 4), 2 | VERSION) {
            return Err(Error::BadVersion);
        }
        let template = match u32le(blob, 8) {
            1 => Template::Llama3,
            2 => Template::ChatMl,
            3 => Template::Granite,
            4 if u32le(blob, 4) >= 3 => Template::Bonsai,
            _ => return Err(Error::BadVersion),
        };
        let vocab_count = u32le(blob, 12);
        let merge_count = u32le(blob, 16) as usize;
        let pool_size = u32le(blob, 20) as usize;
        let specials = Specials {
            bos: u32le(blob, 24),
            eos: u32le(blob, 28),
            eot: u32le(blob, 32),
            start_header: u32le(blob, 36),
            end_header: u32le(blob, 40),
            end_of_text: u32le(blob, 44),
        };
        let byte_ids_off = 48usize;
        let table_off = byte_ids_off + 256 * 4;
        // All section sizes are u32-derived, so these sums cannot wrap on
        // the 64-bit targets NightRun supports; still compute checked so
        // the invariant is explicit rather than environmental.
        let merges_off = table_off
            .checked_add(
                (vocab_count as usize)
                    .checked_mul(8)
                    .ok_or(Error::TooShort)?,
            )
            .ok_or(Error::TooShort)?;
        let pool_off = merges_off
            .checked_add(merge_count.checked_mul(16).ok_or(Error::TooShort)?)
            .ok_or(Error::TooShort)?;
        let pool_end = pool_off.checked_add(pool_size).ok_or(Error::TooShort)?;
        if blob.len() < pool_end {
            return Err(Error::TooShort);
        }

        // Validate the whole vocab table up front: every token's pool
        // range must lie inside the pool. Without this, a malformed blob
        // panics at first use (token_bytes slice) instead of failing here.
        let table = &blob[table_off..merges_off];
        for id in 0..vocab_count as usize {
            let off = id * 8;
            let p = u32le(table, off) as usize;
            let len = u16::from_le_bytes(table[off + 4..off + 6].try_into().unwrap()) as usize;
            if p + len > pool_size {
                return Err(Error::TooShort);
            }
        }
        // Special-token ids must be real vocab entries — they are fed
        // straight into the model as token ids.
        let specials_ok = [
            specials.bos,
            specials.eos,
            specials.eot,
            specials.start_header,
            specials.end_header,
            specials.end_of_text,
        ]
        .iter()
        .all(|&t| t < vocab_count);
        if !specials_ok {
            return Err(Error::TooShort);
        }

        Ok(Tokenizer {
            specials,
            template,
            vocab_count,
            byte_ids: &blob[byte_ids_off..table_off],
            table: &blob[table_off..merges_off],
            merges: &blob[merges_off..pool_off],
            merge_count,
            pool: &blob[pool_off..pool_off + pool_size],
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab_count as usize
    }

    #[inline]
    fn byte_token(&self, b: u8) -> u32 {
        u32le(self.byte_ids, b as usize * 4)
    }

    /// Raw bytes of a token (empty for out-of-range ids).
    pub fn token_bytes(&self, id: u32) -> &'a [u8] {
        if id >= self.vocab_count {
            return &[];
        }
        let off = id as usize * 8;
        let pool_off = u32le(self.table, off) as usize;
        let len = u16::from_le_bytes(self.table[off + 4..off + 6].try_into().unwrap()) as usize;
        &self.pool[pool_off..pool_off + len]
    }

    pub fn is_control(&self, id: u32) -> bool {
        if id >= self.vocab_count {
            return true;
        }
        let off = id as usize * 8;
        let flags = u16::from_le_bytes(self.table[off + 6..off + 8].try_into().unwrap());
        flags & FLAG_CONTROL != 0
    }

    pub fn is_stop(&self, id: u32) -> bool {
        id == self.specials.eos || id == self.specials.eot || id == self.specials.end_of_text
    }

    /// Look up a merge (left, right) -> (result, rank).
    #[inline]
    fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let key = ((left as u64) << 32) | right as u64;
        let mut lo = 0usize;
        let mut hi = self.merge_count;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let off = mid * 16;
            let l = u32le(self.merges, off);
            let r = u32le(self.merges, off + 4);
            let k = ((l as u64) << 32) | r as u64;
            match k.cmp(&key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    return Some((u32le(self.merges, off + 8), u32le(self.merges, off + 12)));
                }
            }
        }
        None
    }

    /// BPE-encode one pretokenized chunk (raw bytes) into `out`.
    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = chunk.iter().map(|&b| self.byte_token(b)).collect();
        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32, u32)> = None; // (index, result, rank)
            for i in 0..ids.len().saturating_sub(1) {
                if let Some((result, rank)) = self.merge(ids[i], ids[i + 1]) {
                    if best.map(|(_, _, br)| rank < br).unwrap_or(true) {
                        best = Some((i, result, rank));
                    }
                }
            }
            match best {
                Some((i, result, _)) => {
                    ids[i] = result;
                    ids.remove(i + 1);
                }
                None => break,
            }
        }
        out.extend_from_slice(&ids);
    }

    /// Encode plain text (no special tokens are ever produced).
    pub fn encode_text(&self, text: &str, out: &mut Vec<u32>) {
        let style = match self.template {
            // Granite's "dbrx" pretokenizer is the cl100k pattern — the
            // same regex as Llama 3 (fixture-verified).
            Template::Llama3 | Template::Granite => crate::pretok::Style::Llama3,
            Template::ChatMl | Template::Bonsai => crate::pretok::Style::Qwen2,
        };
        for chunk in crate::pretok::chunks(text, style) {
            self.encode_chunk(chunk.as_bytes(), out);
        }
    }
}
