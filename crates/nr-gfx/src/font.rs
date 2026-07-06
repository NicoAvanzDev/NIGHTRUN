//! PSF1/PSF2 bitmap font parsing (as shipped by the Spleen font, BSD
//! licensed). The full unicode table is parsed, so everything the font
//! covers (ASCII, Latin-1/Extended-A incl. Polish, punctuation, box
//! drawing, ...) renders; codepoints without a glyph (e.g. emoji) return
//! `None` and are skipped by the renderer.

use alloc::vec::Vec;

const PSF1_MAGIC: [u8; 2] = [0x36, 0x04];
const PSF2_MAGIC: [u8; 4] = [0x72, 0xb5, 0x4a, 0x86];
const PSF2_HAS_UNICODE_TABLE: u32 = 0x1;

pub struct PsfFont<'a> {
    pub width: usize,
    pub height: usize,
    bytes_per_row: usize,
    bytes_per_glyph: usize,
    num_glyphs: usize,
    glyphs: &'a [u8],
    /// (codepoint, glyph index), sorted by codepoint.
    map: Vec<(u32, u16)>,
}

impl<'a> PsfFont<'a> {
    pub fn parse(data: &'a [u8]) -> Option<PsfFont<'a>> {
        if data.len() >= 4 && data[0..2] == PSF1_MAGIC {
            return Self::parse_psf1(data);
        }
        if data.len() < 32 || data[0..4] != PSF2_MAGIC {
            return None;
        }
        let u32le = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };
        let header_size = u32le(8) as usize;
        let flags = u32le(12);
        let num_glyphs = u32le(16) as usize;
        let bytes_per_glyph = u32le(20) as usize;
        let height = u32le(24) as usize;
        let width = u32le(28) as usize;

        let glyphs_end = header_size + num_glyphs * bytes_per_glyph;
        let glyphs = data.get(header_size..glyphs_end)?;

        let map = if flags & PSF2_HAS_UNICODE_TABLE != 0 {
            Self::parse_unicode_table_psf2(&data[glyphs_end..], num_glyphs)
        } else {
            identity_map(num_glyphs)
        };

        Some(PsfFont {
            width,
            height,
            bytes_per_row: bytes_per_glyph / height,
            bytes_per_glyph,
            num_glyphs,
            glyphs,
            map,
        })
    }

    /// PSF1: fixed 8px width, 256/512 glyphs, u16le unicode table.
    fn parse_psf1(data: &'a [u8]) -> Option<PsfFont<'a>> {
        let mode = data[2];
        let charsize = data[3] as usize;
        let num_glyphs = if mode & 0x01 != 0 { 512 } else { 256 };
        let glyphs_end = 4 + num_glyphs * charsize;
        let glyphs = data.get(4..glyphs_end)?;

        let map = if mode & 0x06 != 0 {
            let table = &data[glyphs_end..];
            let mut map: Vec<(u32, u16)> = Vec::new();
            let mut glyph = 0usize;
            let mut in_seq = false;
            for pair in table.chunks_exact(2) {
                if glyph >= num_glyphs {
                    break;
                }
                match u16::from_le_bytes([pair[0], pair[1]]) {
                    0xFFFF => {
                        glyph += 1;
                        in_seq = false;
                    }
                    0xFFFE => in_seq = true,
                    cp if !in_seq => map.push((cp as u32, glyph as u16)),
                    _ => {}
                }
            }
            finish_map(map)
        } else {
            identity_map(num_glyphs)
        };

        Some(PsfFont {
            width: 8,
            height: charsize,
            bytes_per_row: 1,
            bytes_per_glyph: charsize,
            num_glyphs,
            glyphs,
            map,
        })
    }

    /// PSF2 unicode table: per glyph, UTF-8 encoded codepoints terminated
    /// by 0xFF (0xFE starts combining sequences, skipped).
    fn parse_unicode_table_psf2(table: &[u8], num_glyphs: usize) -> Vec<(u32, u16)> {
        let mut map: Vec<(u32, u16)> = Vec::new();
        let mut glyph = 0usize;
        let mut i = 0usize;
        let mut in_seq = false;
        while i < table.len() && glyph < num_glyphs {
            let b = table[i];
            match b {
                0xFF => {
                    glyph += 1;
                    in_seq = false;
                    i += 1;
                }
                0xFE => {
                    in_seq = true;
                    i += 1;
                }
                _ => {
                    let len = match b {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    if !in_seq {
                        if let Some(cp) = decode_utf8(&table[i..(i + len).min(table.len())]) {
                            map.push((cp, glyph as u16));
                        }
                    }
                    i += len;
                }
            }
        }
        finish_map(map)
    }

    /// Glyph bitmap rows for a character; `None` when the font has no
    /// glyph for it (caller should skip, leaving a gap).
    pub fn glyph(&self, ch: char) -> Option<&'a [u8]> {
        let idx = self
            .map
            .binary_search_by_key(&(ch as u32), |e| e.0)
            .ok()
            .map(|i| self.map[i].1 as usize)?;
        if idx >= self.num_glyphs {
            return None;
        }
        Some(&self.glyphs[idx * self.bytes_per_glyph..(idx + 1) * self.bytes_per_glyph])
    }

    /// Whether pixel (x, y) of `glyph` is set.
    #[inline]
    pub fn pixel(&self, glyph: &[u8], x: usize, y: usize) -> bool {
        let byte = glyph[y * self.bytes_per_row + x / 8];
        byte & (0x80 >> (x % 8)) != 0
    }
}

fn identity_map(num_glyphs: usize) -> Vec<(u32, u16)> {
    (0x20..0x80.min(num_glyphs))
        .map(|g| (g as u32, g as u16))
        .collect()
}

/// Sort by codepoint, keeping the first glyph registered for a codepoint.
fn finish_map(mut map: Vec<(u32, u16)>) -> Vec<(u32, u16)> {
    map.sort_by_key(|&(cp, idx)| (cp, idx));
    map.dedup_by_key(|e| e.0);
    map
}

fn decode_utf8(bytes: &[u8]) -> Option<u32> {
    let b0 = *bytes.first()? as u32;
    Some(match bytes.len() {
        1 => b0,
        2 => ((b0 & 0x1F) << 6) | (*bytes.get(1)? as u32 & 0x3F),
        3 => ((b0 & 0x0F) << 12) | ((*bytes.get(1)? as u32 & 0x3F) << 6) | (*bytes.get(2)? as u32 & 0x3F),
        4 => {
            ((b0 & 0x07) << 18)
                | ((*bytes.get(1)? as u32 & 0x3F) << 12)
                | ((*bytes.get(2)? as u32 & 0x3F) << 6)
                | (*bytes.get(3)? as u32 & 0x3F)
        }
        _ => return None,
    })
}
