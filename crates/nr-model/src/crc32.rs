//! CRC-32 (IEEE, reflected 0xEDB88320), table-driven.

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Crc32 {
        Crc32(0xffff_ffff)
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        for &b in data {
            c = TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }

    pub fn finish(&self) -> u32 {
        self.0 ^ 0xffff_ffff
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

pub fn checksum(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}
