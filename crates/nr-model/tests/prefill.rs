//! Batched prefill must be bit-identical to sequential decode: same final
//! logits AND an identical KV cache (proven by comparing the next decoded
//! step). Runs against all catalog model architectures when present.

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn check_model(file: &str) {
    let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let Ok(blob) = std::fs::read(format!("{root}/models/{file}")) else {
        eprintln!("SKIP: models/{file} not present");
        return;
    };
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("tokenizer");

    let mut alloc = |bytes: usize, align: usize| -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).unwrap();
        unsafe { std::alloc::alloc_zeroed(layout) }
    };

    // A prompt long enough to cross one chunk boundary matters less than
    // covering the batch path; ~70 tokens also exercises MAX_BATCH=64
    // chunking via prefill().
    let mut ids: Vec<u32> = Vec::new();
    tok.encode_conversation_start(
        Some("You are a helpful assistant. Answer concisely and clearly whenever possible."),
        &mut ids,
    );
    tok.encode_message(
        nr_token::template::ROLE_USER,
        "Explain, in about three sentences, why the sky appears blue during the day and red at \
         sunset. Then list four cities you would recommend visiting in autumn, with one short \
         reason for each, and finish with a haiku about neon lights reflecting on wet asphalt \
         after the rain has stopped falling over the quiet streets.",
        &mut ids,
    );
    tok.encode_header(nr_token::template::ROLE_ASSISTANT, &mut ids);
    assert!(
        ids.len() > nr_model::infer::MAX_BATCH,
        "prompt must span chunks ({})",
        ids.len()
    );

    // Sequential reference.
    let mut seq = nr_model::InferCtx::new(&model, 256, &mut alloc).expect("ctx");
    let mut seq_logits: Vec<f32> = Vec::new();
    for &id in &ids {
        seq_logits = seq.forward(id).to_vec();
    }
    let seq_next = nr_tensor::vec::argmax(&seq_logits) as u32;
    let seq_logits2 = seq.forward(seq_next).to_vec();

    // Batched prefill.
    let mut bat = nr_model::InferCtx::new(&model, 256, &mut alloc).expect("ctx");
    let bat_logits = bat.prefill(&ids).to_vec();
    assert_eq!(bat.pos, ids.len());
    assert_eq!(
        seq_logits, bat_logits,
        "{file}: prefill logits differ from sequential"
    );

    // The KV cache must be identical too: the next decode step must match
    // bit-for-bit.
    let bat_logits2 = bat.forward(seq_next).to_vec();
    assert_eq!(
        seq_logits2, bat_logits2,
        "{file}: post-prefill decode differs"
    );
}

#[test]
fn prefill_bit_identity_llama() {
    check_model("model.nrm");
}

#[test]
fn prefill_bit_identity_qwen() {
    check_model("qwen3-4b-q4km.nrm");
}

#[test]
fn prefill_bit_identity_granite() {
    check_model("granite-4.1-3b-q4km.nrm");
}

/// Spec'd odd lengths around batch boundaries: every chunking shape must
/// stay bit-identical to sequential decode (llama artifact; lengths
/// beyond the prompt reuse wrapped ids — token values are irrelevant to
/// the chunking math being pinned).
#[test]
fn prefill_odd_lengths_bit_identity() {
    let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let Ok(blob) = std::fs::read(format!("{root}/models/model.nrm")) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("tokenizer");

    let mut alloc = |bytes: usize, align: usize| -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).unwrap();
        unsafe { std::alloc::alloc_zeroed(layout) }
    };
    let mut seq = nr_model::InferCtx::new(&model, 640, &mut alloc).expect("ctx");
    let mut bat = nr_model::InferCtx::new(&model, 640, &mut alloc).expect("ctx");

    let mut ids: Vec<u32> = Vec::new();
    tok.encode_text(
        "The neon city hummed with quiet electricity as rain fell over empty streets, \
         reflections of magenta signs scattering across wet asphalt while distant trains \
         carried night workers home through tunnels of sodium light.",
        &mut ids,
    );
    while ids.len() < 513 {
        let extend: Vec<u32> = ids.clone();
        ids.extend(extend);
    }

    for &len in &[
        1usize, 7, 15, 17, 31, 33, 63, 65, 127, 128, 129, 255, 257, 511, 512, 513,
    ] {
        let prompt = &ids[..len];
        seq.reset();
        let mut seq_logits: Vec<f32> = Vec::new();
        for &id in prompt {
            seq_logits = seq.forward(id).to_vec();
        }
        bat.reset();
        let bat_logits = bat.prefill(prompt).to_vec();
        assert_eq!(seq_logits, bat_logits, "len {len}: logits diverge");
        assert_eq!(seq.pos, bat.pos, "len {len}: position diverges");
    }
}

#[test]
fn prefill_bit_identity_ternary_bonsai() {
    check_model("ternary-bonsai-8b-pq2.nrm");
}
