//! PSF2 bitmap font parsing (as shipped by the Spleen font, BSD licensed).

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
    /// Maps ASCII 0x20..0x7F to glyph index.
    ascii: [u16; 0x60],
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

        let mut ascii = [0u16; 0x60];
        if flags & PSF2_HAS_UNICODE_TABLE != 0 {
            Self::parse_unicode_table(&data[glyphs_end..], num_glyphs, &mut ascii);
        } else {
            for (i, slot) in ascii.iter_mut().enumerate() {
                let g = 0x20 + i;
                if g < num_glyphs {
                    *slot = g as u16;
                }
            }
        }

        Some(PsfFont {
            width,
            height,
            bytes_per_row: bytes_per_glyph / height,
            bytes_per_glyph,
            num_glyphs,
            glyphs,
            ascii,
        })
    }

    /// PSF1: fixed 8px width, 256/512 glyphs, u16le unicode table.
    fn parse_psf1(data: &'a [u8]) -> Option<PsfFont<'a>> {
        let mode = data[2];
        let charsize = data[3] as usize;
        let num_glyphs = if mode & 0x01 != 0 { 512 } else { 256 };
        let glyphs_end = 4 + num_glyphs * charsize;
        let glyphs = data.get(4..glyphs_end)?;

        let mut ascii = [0u16; 0x60];
        if mode & 0x06 != 0 {
            let table = &data[glyphs_end..];
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
                    cp if !in_seq && (0x20..0x80).contains(&cp) => {
                        let slot = &mut ascii[cp as usize - 0x20];
                        if *slot == 0 || cp == 0x20 {
                            *slot = glyph as u16;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            for (i, slot) in ascii.iter_mut().enumerate() {
                *slot = (0x20 + i) as u16;
            }
        }

        Some(PsfFont {
            width: 8,
            height: charsize,
            bytes_per_row: 1,
            bytes_per_glyph: charsize,
            num_glyphs,
            glyphs,
            ascii,
        })
    }

    /// The unicode table lists, per glyph, UTF-8 encoded codepoints
    /// terminated by 0xFF (0xFE starts combining sequences, skipped).
    fn parse_unicode_table(table: &[u8], num_glyphs: usize, ascii: &mut [u16; 0x60]) {
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
                    // Decode one UTF-8 codepoint.
                    let len = match b {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    if !in_seq && len == 1 && (0x20..0x80).contains(&(b as usize)) {
                        let slot = &mut ascii[b as usize - 0x20];
                        if *slot == 0 || b == 0x20 {
                            *slot = glyph as u16;
                        }
                    }
                    i += len;
                }
            }
        }
    }

    /// Glyph bitmap rows for a character (falls back to '?').
    pub fn glyph(&self, ch: char) -> &'a [u8] {
        let code = if (' '..='\u{7e}').contains(&ch) { ch as usize } else { '?' as usize };
        let idx = self.ascii[code - 0x20] as usize;
        let idx = if idx < self.num_glyphs { idx } else { 0 };
        &self.glyphs[idx * self.bytes_per_glyph..(idx + 1) * self.bytes_per_glyph]
    }

    /// Whether pixel (x, y) of `glyph` is set.
    #[inline]
    pub fn pixel(&self, glyph: &[u8], x: usize, y: usize) -> bool {
        let byte = glyph[y * self.bytes_per_row + x / 8];
        byte & (0x80 >> (x % 8)) != 0
    }
}
