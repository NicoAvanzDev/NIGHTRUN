//! Adversarial tokenizer-blob parsing: malformed blobs must fail at
//! parse time with a clean Error — never panic at first token lookup.

use nr_token::Tokenizer;

/// Minimal valid blob: 2-token vocab ("a", "b"), no merges, all specials
/// pointing at token 0.
fn tiny_blob() -> Vec<u8> {
    let vocab: u32 = 2;
    let merges: u32 = 0;
    let pool = b"ab";
    let byte_ids_off = 48usize;
    let table_off = byte_ids_off + 256 * 4;
    let pool_off = table_off + vocab as usize * 8;

    let mut b = vec![0u8; pool_off + pool.len()];
    b[0..4].copy_from_slice(b"NRTK");
    b[4..8].copy_from_slice(&2u32.to_le_bytes()); // version
    b[8..12].copy_from_slice(&1u32.to_le_bytes()); // template = Llama3
    b[12..16].copy_from_slice(&vocab.to_le_bytes());
    b[16..20].copy_from_slice(&merges.to_le_bytes());
    b[20..24].copy_from_slice(&(pool.len() as u32).to_le_bytes());
    // specials at 24..48: all zeros = token 0 (valid).
    // byte_ids: 256 * u32 zeros (map every byte to token 0) — fine.
    // vocab table: token0 -> pool[0..1], token1 -> pool[1..2].
    let t = table_off;
    b[t..t + 4].copy_from_slice(&0u32.to_le_bytes());
    b[t + 4..t + 6].copy_from_slice(&1u16.to_le_bytes());
    b[t + 8..t + 12].copy_from_slice(&1u32.to_le_bytes());
    b[t + 12..t + 14].copy_from_slice(&1u16.to_le_bytes());
    b[pool_off..].copy_from_slice(pool);
    b
}

#[test]
fn valid_tiny_blob_parses() {
    let b = tiny_blob();
    let t = Tokenizer::parse(&b).expect("valid blob");
    assert_eq!(t.vocab_len(), 2);
    assert_eq!(t.token_bytes(0), b"a");
    assert_eq!(t.token_bytes(1), b"b");
}

#[test]
fn truncated_blob_rejects() {
    let b = tiny_blob();
    assert!(Tokenizer::parse(&b[..b.len() - 1]).is_err());
    assert!(Tokenizer::parse(&b[..100]).is_err());
    assert!(Tokenizer::parse(&[]).is_err());
}

#[test]
fn token_pool_range_out_of_bounds_rejects() {
    let mut b = tiny_blob();
    // token 1's pool offset points past the pool end.
    let table_off = 48 + 256 * 4;
    b[table_off + 8..table_off + 12].copy_from_slice(&1000u32.to_le_bytes());
    assert!(
        Tokenizer::parse(&b).is_err(),
        "OOB pool range must fail parse, not token_bytes"
    );
}

#[test]
fn special_token_out_of_vocab_rejects() {
    let mut b = tiny_blob();
    // bos = 7 with vocab_count 2: fed directly into the model as a token
    // id, so it must be rejected here.
    b[24..28].copy_from_slice(&7u32.to_le_bytes());
    assert!(Tokenizer::parse(&b).is_err());
}

#[test]
fn huge_counts_reject_cleanly() {
    let mut b = tiny_blob();
    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes()); // vocab_count
    assert!(Tokenizer::parse(&b).is_err());
    let mut b = tiny_blob();
    b[16..20].copy_from_slice(&u32::MAX.to_le_bytes()); // merge_count
    assert!(Tokenizer::parse(&b).is_err());
}

#[test]
fn flush_utf8_handles_split_sequences() {
    // "ż" = 0xC5 0xBC split across two tokens; emoji = 4 bytes split 1+3.
    let mut out = String::new();
    let mut pending = vec![0xC5];
    nr_token::flush_utf8(&mut pending, &mut out);
    assert_eq!(out, "");
    assert_eq!(pending, [0xC5]);
    pending.push(0xBC);
    nr_token::flush_utf8(&mut pending, &mut out);
    assert_eq!(out, "ż");
    assert!(pending.is_empty());

    let emoji = "🌆".as_bytes();
    let mut pending = emoji[..1].to_vec();
    nr_token::flush_utf8(&mut pending, &mut out);
    pending.extend_from_slice(&emoji[1..]);
    nr_token::flush_utf8(&mut pending, &mut out);
    assert_eq!(out, "ż🌆");

    // Complete text followed by an incomplete tail: text flushes, tail stays.
    let mut pending = b"ok".to_vec();
    pending.push(0xE2); // first byte of a 3-byte sequence
    let mut out = String::new();
    nr_token::flush_utf8(&mut pending, &mut out);
    assert_eq!(out, "ok");
    assert_eq!(pending, [0xE2]);

    // Invalid byte cannot wedge the stream.
    let mut pending = vec![0xFF, b'h', b'i'];
    let mut out = String::new();
    nr_token::flush_utf8(&mut pending, &mut out);
    assert_eq!(out, "hi");
    assert!(pending.is_empty());
}
