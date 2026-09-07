//! Adversarial .nrm parsing: every malformed input must produce a clean
//! ParseError — never a panic, never a wrapped bound, never a misaligned
//! reinterpretation. Builds a minimal valid file, then mutates it.

use nr_model::format::{Model, ParseError, DATA_ALIGN, ENTRY_SIZE, HEADER_SIZE};

/// Build the smallest valid v3 .nrm: empty-ish tokenizer blob, one F32
/// tensor (8x8), correct section CRCs.
fn tiny_nrm() -> Vec<u8> {
    let tok_blob = vec![0xEEu8; 16];
    let tensor_count = 1usize;

    let tok_off = HEADER_SIZE;
    let tok_size = tok_blob.len();
    let table_off = tok_off + tok_size;
    let table_len = tensor_count * ENTRY_SIZE;
    // Data section 64-byte aligned.
    let data_off = (table_off + table_len).next_multiple_of(DATA_ALIGN);
    let rows = 8u32;
    let cols = 8u32;
    let tsize = (rows as u64 * cols as u64 * 4) as usize;
    let data_size = tsize;

    let mut b = vec![0u8; data_off + data_size];
    b[0..4].copy_from_slice(b"NRUN");
    b[4..8].copy_from_slice(&3u32.to_le_bytes()); // version
    b[8..12].copy_from_slice(&1u32.to_le_bytes()); // arch = Llama3
                                                   // dims (dim..ctx_train) — values only need to be present.
    for (i, v) in [8u32, 1, 1, 1, 8, 8, 16, 128].iter().enumerate() {
        b[12 + i * 4..16 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    // rope/norm f32s at 44..68 (zeros are fine), flags 68..72 = 0.
    // muP scalars 72..88: embed 1.0, attn 0.0, residual 1.0, logit 1.0.
    b[72..76].copy_from_slice(&1.0f32.to_le_bytes());
    b[76..80].copy_from_slice(&0.0f32.to_le_bytes());
    b[80..84].copy_from_slice(&1.0f32.to_le_bytes());
    b[84..88].copy_from_slice(&1.0f32.to_le_bytes());
    b[88..92].copy_from_slice(b"test"); // name
    b[136..144].copy_from_slice(&(tok_off as u64).to_le_bytes());
    b[144..152].copy_from_slice(&(tok_size as u64).to_le_bytes());
    b[152..160].copy_from_slice(&(table_off as u64).to_le_bytes());
    b[160..164].copy_from_slice(&(tensor_count as u32).to_le_bytes());
    b[168..176].copy_from_slice(&(data_off as u64).to_le_bytes());
    b[176..184].copy_from_slice(&(data_size as u64).to_le_bytes());

    b[tok_off..tok_off + tok_size].copy_from_slice(&tok_blob);

    // Tensor entry: kind=TokEmbed(1), layer=0, dtype=F32(0), offset=0.
    let t = table_off;
    b[t..t + 2].copy_from_slice(&1u16.to_le_bytes());
    b[t + 4..t + 6].copy_from_slice(&0u16.to_le_bytes()); // dtype F32
    b[t + 8..t + 16].copy_from_slice(&0u64.to_le_bytes()); // offset
    b[t + 16..t + 24].copy_from_slice(&(tsize as u64).to_le_bytes());
    b[t + 24..t + 28].copy_from_slice(&rows.to_le_bytes());
    b[t + 28..t + 32].copy_from_slice(&cols.to_le_bytes());

    // CRCs.
    let data_crc = nr_model::crc32::checksum(&b[data_off..data_off + data_size]);
    b[184..188].copy_from_slice(&data_crc.to_le_bytes());
    let mut crc = nr_model::crc32::Crc32::new();
    crc.update(&b[..HEADER_SIZE - 8]);
    crc.update(&[0u8; 8]);
    crc.update(&b[tok_off..tok_off + tok_size]);
    crc.update(&b[table_off..table_off + table_len]);
    let meta_crc = crc.finish();
    b[188..192].copy_from_slice(&meta_crc.to_le_bytes());
    b
}

/// Re-seal meta_crc after a header/table mutation, so tests hit the check
/// they intend instead of failing early at the CRC gate.
fn reseal(b: &mut [u8]) {
    let tok_off = u64::from_le_bytes(b[136..144].try_into().unwrap()) as usize;
    let tok_size = u64::from_le_bytes(b[144..152].try_into().unwrap()) as usize;
    let table_off = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(b[160..164].try_into().unwrap()) as usize;
    let mut crc = nr_model::crc32::Crc32::new();
    crc.update(&b[..HEADER_SIZE - 8]);
    crc.update(&[0u8; 8]);
    // Regions may be nonsense in adversarial cases; clamp for sealing.
    let t0 = tok_off.min(b.len());
    let t1 = tok_off.saturating_add(tok_size).min(b.len());
    crc.update(&b[t0..t1.max(t0)]);
    let e0 = table_off.min(b.len());
    let e1 = table_off
        .saturating_add(count.saturating_mul(ENTRY_SIZE))
        .min(b.len());
    crc.update(&b[e0..e1.max(e0)]);
    let meta = crc.finish();
    b[188..192].copy_from_slice(&meta.to_le_bytes());
}

fn tiny_pq2_nrm() -> Vec<u8> {
    let mut b = tiny_nrm();
    b[4..8].copy_from_slice(&5u32.to_le_bytes());
    let tab = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    b[tab + 4..tab + 6].copy_from_slice(&5u16.to_le_bytes());
    b[tab + 16..tab + 24].copy_from_slice(&68u64.to_le_bytes());
    b[tab + 24..tab + 28].copy_from_slice(&2u32.to_le_bytes());
    b[tab + 28..tab + 32].copy_from_slice(&128u32.to_le_bytes());
    reseal(&mut b);
    b
}

#[test]
fn pq2_layout_and_legacy_versions() {
    use nr_model::format::{TensorDtype, TensorKind};
    let b = tiny_pq2_nrm();
    let model = Model::parse(&b).unwrap();
    let t = model.tensor(TensorKind::TokEmbed, 0).unwrap();
    assert_eq!(t.dtype, TensorDtype::PQ2_0);
    assert_eq!(t.pq2().len(), 2);
    assert_eq!(TensorDtype::PQ2_0.byte_size(128), Some(34));
    assert_eq!(TensorDtype::PQ2_0.byte_size(127), None);
    assert_eq!(TensorDtype::PQ2_0.byte_size(u64::MAX), None);
    assert!(nr_model::verify::StreamingVerifier::new(&b).is_ok());
    assert!(nr_model::verify::StreamingVerifier::new(&tiny_nrm()).is_ok());
    let mut legacy = b.clone();
    legacy[4..8].copy_from_slice(&4u32.to_le_bytes());
    reseal(&mut legacy);
    assert!(matches!(Model::parse(&legacy), Err(ParseError::BadTable)));
    legacy[4..8].copy_from_slice(&3u32.to_le_bytes());
    reseal(&mut legacy);
    assert!(matches!(Model::parse(&legacy), Err(ParseError::BadTable)));
}

#[test]
fn rejects_pq2_partial_rows_and_wrong_sizes() {
    for partial_rows in [true, false] {
        let mut b = tiny_pq2_nrm();
        let tab = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
        if partial_rows {
            // Total elements still align; each individual row does not.
            b[tab + 24..tab + 28].copy_from_slice(&4u32.to_le_bytes());
            b[tab + 28..tab + 32].copy_from_slice(&64u32.to_le_bytes());
        } else {
            b[tab + 16..tab + 24].copy_from_slice(&67u64.to_le_bytes());
        }
        reseal(&mut b);
        assert!(matches!(Model::parse(&b), Err(ParseError::BadTable)));
    }
}

#[test]
fn retired_dtype_is_rejected() {
    // The old ID must never be interpreted as ternary weights, even in
    // an otherwise valid file with matching tensor sizes and checksums.
    for version in [3, 4, 5, nr_model::format::VERSION] {
        let mut b = tiny_nrm();
        b[4..8].copy_from_slice(&version.to_le_bytes());
        let tab = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
        b[tab + 4..tab + 6].copy_from_slice(&4u16.to_le_bytes());
        b[tab + 16..tab + 24].copy_from_slice(&36u64.to_le_bytes());
        b[tab + 24..tab + 28].copy_from_slice(&2u32.to_le_bytes());
        b[tab + 28..tab + 32].copy_from_slice(&128u32.to_le_bytes());
        reseal(&mut b);
        assert!(matches!(Model::parse(&b), Err(ParseError::BadTable)));
    }
}

#[test]
fn validates_yarn_parameters() {
    let mut b = tiny_nrm();
    b[4..8].copy_from_slice(&4u32.to_le_bytes());
    b[8..12].copy_from_slice(&2u32.to_le_bytes()); // Qwen3
    b[68..72].copy_from_slice(&nr_model::format::FLAG_ROPE_YARN.to_le_bytes());
    for (off, v) in [
        (44, 1_000_000.0f32),
        (52, 4.0),
        (56, 32.0),
        (60, 1.0),
        (64, 16384.0),
        (164, 1.0),
    ] {
        b[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    reseal(&mut b);
    assert!(Model::parse(&b).is_ok());
    for off in [44, 52, 56, 60, 64, 164] {
        for bad in [0.0f32, f32::NAN, f32::INFINITY, -1.0] {
            let mut mutated = b.clone();
            mutated[off..off + 4].copy_from_slice(&bad.to_le_bytes());
            reseal(&mut mutated);
            assert!(
                matches!(Model::parse(&mutated), Err(ParseError::BadTable)),
                "offset {off}: {bad}"
            );
        }
    }
}

fn expect_err(b: &[u8], what: &str) {
    match Model::parse(b) {
        Err(_) => {}
        Ok(_) => panic!("{what}: parser accepted malformed input"),
    }
}

#[test]
fn valid_tiny_file_parses() {
    let b = tiny_nrm();
    let m = Model::parse(&b).expect("tiny valid file");
    assert_eq!(m.meta.dim, 8);
    let t = m.tensor(nr_model::TensorKind::TokEmbed, 0).expect("tensor");
    assert_eq!((t.rows, t.cols), (8, 8));
    assert_eq!(t.f32().len(), 64);
}

#[test]
fn truncations_reject_cleanly() {
    let b = tiny_nrm();
    for cut in [0, 3, 4, 100, HEADER_SIZE - 1, HEADER_SIZE, b.len() - 1] {
        expect_err(&b[..cut], &format!("truncated to {cut}"));
    }
}

#[test]
fn bad_magic_and_version() {
    let mut b = tiny_nrm();
    b[0] = b'X';
    expect_err(&b, "magic");
    let mut b = tiny_nrm();
    b[4..8].copy_from_slice(&99u32.to_le_bytes());
    expect_err(&b, "version");
}

#[test]
fn wrapping_region_offsets_reject() {
    // tok_off near u64::MAX: tok_off + tok_size would wrap.
    let mut b = tiny_nrm();
    b[136..144].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
    reseal(&mut b);
    expect_err(&b, "wrapping tok_off");

    // data region wrap.
    let mut b = tiny_nrm();
    b[168..176].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
    reseal(&mut b);
    expect_err(&b, "wrapping data_off");

    // table_off wrap + huge count (mul overflow).
    let mut b = tiny_nrm();
    b[152..160].copy_from_slice(&(u64::MAX - 31).to_le_bytes());
    b[160..164].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut b);
    expect_err(&b, "wrapping table");
}

#[test]
fn entry_offset_size_wrap_rejects() {
    let mut b = tiny_nrm();
    let table_off = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    b[table_off + 8..table_off + 16].copy_from_slice(&(u64::MAX - 63).to_le_bytes());
    reseal(&mut b);
    expect_err(&b, "entry offset+size wrap");
}

#[test]
fn misaligned_entry_offset_rejects() {
    let mut b = tiny_nrm();
    let table_off = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    // Offset 4 is inside the data section but breaks the 64-byte contract
    // (and would misalign nothing here, but the contract is absolute).
    b[table_off + 8..table_off + 16].copy_from_slice(&4u64.to_le_bytes());
    // keep size/dims consistent so only alignment trips
    reseal(&mut b);
    expect_err(&b, "misaligned tensor offset");
}

#[test]
fn dim_overflow_rejects() {
    let mut b = tiny_nrm();
    let table_off = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    b[table_off + 24..table_off + 28].copy_from_slice(&u32::MAX.to_le_bytes());
    b[table_off + 28..table_off + 32].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut b);
    expect_err(&b, "rows*cols overflow");
}

#[test]
fn size_dtype_mismatch_rejects() {
    let mut b = tiny_nrm();
    let table_off = u64::from_le_bytes(b[152..160].try_into().unwrap()) as usize;
    b[table_off + 16..table_off + 24].copy_from_slice(&8u64.to_le_bytes()); // wrong size
    reseal(&mut b);
    expect_err(&b, "size/dtype mismatch");
}

#[test]
fn meta_crc_flip_rejects() {
    let mut b = tiny_nrm();
    b[100] ^= 0x40; // inside name field, covered by meta crc
    match Model::parse(&b) {
        Err(ParseError::MetaCrc { .. }) => {}
        other => panic!("expected MetaCrc, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn nonsense_scalars_reject() {
    let mut b = tiny_nrm();
    b[80..84].copy_from_slice(&f32::NAN.to_le_bytes()); // residual_scale
    reseal(&mut b);
    expect_err(&b, "NaN residual scale");
}
