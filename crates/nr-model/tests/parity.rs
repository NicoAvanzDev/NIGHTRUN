//! End-to-end forward-pass parity: greedy continuations must match
//! llama.cpp (verified against llama-completion --temp 0 on the same
//! GGUF; see docs/architecture.md). Requires models/model.nrm; skips
//! when absent.

fn load_model_blob() -> Option<Vec<u8>> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    std::fs::read(format!("{root}/models/model.nrm")).ok()
}

fn greedy(prompt: &str, n: usize) -> Option<String> {
    let blob = load_model_blob()?;
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("tokenizer");

    let mut alloc = |bytes: usize, align: usize| -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).unwrap();
        unsafe { std::alloc::alloc_zeroed(layout) }
    };
    let mut ctx = nr_model::InferCtx::new(&model, 512, &mut alloc).expect("ctx");

    let mut ids = vec![tok.specials.bos];
    tok.encode_text(prompt, &mut ids);
    let mut logits: &[f32] = &[];
    for &id in &ids {
        logits = ctx.forward(id);
    }
    let mut out = Vec::new();
    for _ in 0..n {
        let next = nr_tensor::vec::argmax(logits) as u32;
        if tok.is_stop(next) {
            break;
        }
        out.extend_from_slice(tok.token_bytes(next));
        logits = ctx.forward(next);
    }
    Some(String::from_utf8_lossy(&out).into_owned())
}

#[test]
fn greedy_matches_llama_cpp_short() {
    let Some(text) = greedy("The capital of France is", 12) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };
    assert_eq!(text, " Paris. The Eiffel Tower is located in Paris.");
}

#[test]
fn greedy_matches_llama_cpp_long() {
    let Some(text) = greedy("Once upon a time, in a city of neon lights,", 24) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };
    assert_eq!(
        text,
        " where the skyscrapers pierced the sky and the streets hummed with the rhythm of the city, there lived a young"
    );
}
