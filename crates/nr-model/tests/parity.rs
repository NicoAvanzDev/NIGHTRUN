//! End-to-end forward-pass parity: greedy continuations must match
//! llama.cpp (verified against llama-completion --temp 0 on the same
//! GGUF; see docs/architecture.md). Requires models/model.nrm; skips
//! when absent.

/// Each test loads a 1.3-2.4 GB model; running them concurrently OOMs the
/// host, so every test body holds this lock.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn load_model_blob(file: &str) -> Option<Vec<u8>> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    std::fs::read(format!("{root}/models/{file}")).ok()
}

fn greedy_in(model_file: &str, prompt: &str, n: usize, chat: bool) -> Option<String> {
    let blob = load_model_blob(model_file)?;
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("tokenizer");

    let mut alloc = |bytes: usize, align: usize| -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).unwrap();
        unsafe { std::alloc::alloc_zeroed(layout) }
    };
    let mut ctx = nr_model::InferCtx::new(&model, 512, &mut alloc).expect("ctx");

    let mut ids = Vec::new();
    if chat {
        tok.encode_conversation_start(None, &mut ids);
        tok.encode_message(nr_token::template::ROLE_USER, prompt, &mut ids);
        tok.encode_header(nr_token::template::ROLE_ASSISTANT, &mut ids);
    } else {
        // Raw completion; BOS only for families that use one.
        if tok.template == nr_token::blob::Template::Llama3 {
            ids.push(tok.specials.bos);
        }
        tok.encode_text(prompt, &mut ids);
    }
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

fn greedy(prompt: &str, n: usize) -> Option<String> {
    greedy_in("model.nrm", prompt, n, false)
}

#[test]
fn greedy_matches_llama_cpp_short() {
    let _guard = serial();
    let Some(text) = greedy("The capital of France is", 12) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };
    assert_eq!(text, " Paris. The Eiffel Tower is located in Paris.");
}

#[test]
fn greedy_matches_llama_cpp_long() {
    let _guard = serial();
    let Some(text) = greedy("Once upon a time, in a city of neon lights,", 24) else {
        eprintln!("SKIP: models/model.nrm not present");
        return;
    };
    assert_eq!(
        text,
        " where the skyscrapers pierced the sky and the streets hummed with the rhythm of the city, there lived a young"
    );
}

// ---- Qwen3-4B-Instruct-2507 Q4_K_M pins (verified against llama-completion
// --temp 0 -c 4096 on the identical GGUF) --------------------------------

#[test]
fn qwen_greedy_matches_llama_cpp_short() {
    let _guard = serial();
    let Some(text) = greedy_in("qwen3-4b-q4km.nrm", "The capital of France is", 12, false) else {
        eprintln!("SKIP: models/qwen3-4b-q4km.nrm not present");
        return;
    };
    assert_eq!(text, " Paris. The capital of Germany is Berlin. The capital of");
}

#[test]
fn qwen_greedy_matches_llama_cpp_long() {
    let _guard = serial();
    let Some(text) = greedy_in(
        "qwen3-4b-q4km.nrm",
        "Once upon a time, in a city of neon lights,",
        24,
        false,
    ) else {
        eprintln!("SKIP: models/qwen3-4b-q4km.nrm not present");
        return;
    };
    assert_eq!(
        text,
        " there lived a young man named Leo. Leo was a quiet boy with a sharp mind and a deep love for stories."
    );
}

/// Chat-templated greedy reply must match llama.cpp chat mode. Because the
/// model is tied (no output.weight tensor exists in the artifact), this
/// also proves the logits projection uses the same tensor source as the
/// reference implementation: the token embedding matrix.
#[test]
fn qwen_chat_greedy_matches_llama_cpp() {
    let _guard = serial();
    let Some(text) = greedy_in("qwen3-4b-q4km.nrm", "What is the capital of France?", 20, true) else {
        eprintln!("SKIP: models/qwen3-4b-q4km.nrm not present");
        return;
    };
    assert_eq!(text, "The capital of France is Paris.");
}

/// The spec's tied-embedding audit: the artifact must have no dedicated
/// output head, and the runtime must be using the (Q6_K) embedding matrix
/// as classifier.
#[test]
fn qwen_output_head_is_tied_embedding() {
    let _guard = serial();
    let Some(blob) = load_model_blob("qwen3-4b-q4km.nrm") else {
        eprintln!("SKIP: models/qwen3-4b-q4km.nrm not present");
        return;
    };
    let model = nr_model::Model::parse(&blob).expect("parse model");
    assert_eq!(model.meta.arch, nr_model::format::Arch::Qwen3);
    assert!(model.meta.flags & nr_model::format::FLAG_TIED_EMBEDDINGS != 0);
    assert!(!model.has_tensor(nr_model::TensorKind::Output));
    let embed = model.tensor(nr_model::TensorKind::TokEmbed, 0).unwrap();
    assert_eq!(embed.dtype, nr_model::TensorDtype::Q6K);
    assert_eq!(embed.rows, 151936);
    assert_eq!(embed.cols, 2560);
}
