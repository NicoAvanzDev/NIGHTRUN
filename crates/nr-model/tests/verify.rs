//! Streaming verifier: valid files pass; a flipped byte in any section
//! fails with the right error. Also pins the slicing-by-8 CRC against
//! known vectors and the classic byte-at-a-time algorithm.

use nr_model::verify::{StreamingVerifier, VerifyError};

#[test]
fn crc32_known_vectors() {
    // Canonical IEEE CRC-32 check value.
    assert_eq!(nr_model::crc32::checksum(b"123456789"), 0xCBF4_3926);
    assert_eq!(nr_model::crc32::checksum(b""), 0);
    assert_eq!(nr_model::crc32::checksum(b"a"), 0xE8B7_BE43);
}

#[test]
fn crc32_matches_classic_algorithm() {
    // Classic byte-at-a-time reference, independent of the slicing tables.
    fn classic(data: &[u8]) -> u32 {
        let mut c = 0xffff_ffffu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        c ^ 0xffff_ffff
    }
    let mut x = 0x1234_5678u64;
    let data: Vec<u8> = (0..100_003)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();
    // Odd lengths exercise the remainder path.
    for len in [0, 1, 7, 8, 9, 63, 100_003] {
        assert_eq!(nr_model::crc32::checksum(&data[..len]), classic(&data[..len]), "len {len}");
    }
}

fn feed_all(bytes: &[u8], chunk: usize) -> Result<(), VerifyError> {
    let mut v = StreamingVerifier::new(&bytes[..nr_model::format::HEADER_SIZE.max(chunk).min(bytes.len())])?;
    let mut off = 0;
    while off < bytes.len() {
        let end = (off + chunk).min(bytes.len());
        v.feed(off, &bytes[off..end]);
        off = end;
    }
    v.finish()
}

#[test]
fn streaming_verifier_on_real_model() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let Ok(mut bytes) = std::fs::read(format!("{root}/models/model.nrm")) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };

    // Valid file passes, at several chunk sizes (incl. non-power-of-two).
    for chunk in [4096, 1_000_000, 16 * 1024 * 1024] {
        assert_eq!(feed_all(&bytes, chunk), Ok(()), "chunk {chunk}");
    }

    // Flip one byte in the data section -> DataCrc.
    let data_off = u64::from_le_bytes(bytes[168..176].try_into().unwrap()) as usize;
    bytes[data_off + 12345] ^= 0x01;
    assert_eq!(feed_all(&bytes, 1_000_000), Err(VerifyError::DataCrc));
    bytes[data_off + 12345] ^= 0x01;

    // Flip one byte in the tokenizer blob -> MetaCrc.
    bytes[nr_model::format::HEADER_SIZE + 100] ^= 0x01;
    assert_eq!(feed_all(&bytes, 1_000_000), Err(VerifyError::MetaCrc));
    bytes[nr_model::format::HEADER_SIZE + 100] ^= 0x01;

    // Flip a header dim byte -> MetaCrc (crc fields themselves are zeroed
    // in the hash, so corrupting them also fails, via mismatch).
    bytes[16] ^= 0x01;
    assert_eq!(feed_all(&bytes, 1_000_000), Err(VerifyError::MetaCrc));
    bytes[16] ^= 0x01;

    // Sanity: still valid after undo.
    assert_eq!(feed_all(&bytes, 1_000_000), Ok(()));
}
