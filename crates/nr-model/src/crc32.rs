//! CRC-32 (IEEE, reflected 0xEDB88320), slicing-by-8: processes 8 bytes
//! per step (~4-6x the classic byte-at-a-time table loop), so checksums
//! can run inline with disk reads without becoming the bottleneck.
//! Identical results to the classic algorithm (tested).

const fn make_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            t[s][i] = t[0][(t[s - 1][i] & 0xff) as usize] ^ (t[s - 1][i] >> 8);
            i += 1;
        }
        s += 1;
    }
    t
}

static TABLES: [[u32; 256]; 8] = make_tables();

pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Crc32 {
        Crc32(0xffff_ffff)
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        let mut chunks = data.chunks_exact(8);
        for w in &mut chunks {
            let lo = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) ^ c;
            c = TABLES[7][(lo & 0xff) as usize]
                ^ TABLES[6][((lo >> 8) & 0xff) as usize]
                ^ TABLES[5][((lo >> 16) & 0xff) as usize]
                ^ TABLES[4][(lo >> 24) as usize]
                ^ TABLES[3][w[4] as usize]
                ^ TABLES[2][w[5] as usize]
                ^ TABLES[1][w[6] as usize]
                ^ TABLES[0][w[7] as usize];
        }
        for &b in chunks.remainder() {
            c = TABLES[0][((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
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
