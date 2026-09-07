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
    greedy_tokens_in(model_file, prompt, n, chat).map(|(_, text)| text)
}

fn greedy_tokens_in(
    model_file: &str,
    prompt: &str,
    n: usize,
    chat: bool,
) -> Option<(Vec<u32>, String)> {
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
    let mut logits: &[f32] = ctx.prefill(&ids);
    let mut out = Vec::new();
    let mut generated = Vec::new();
    for _ in 0..n {
        let next = nr_tensor::vec::argmax(logits) as u32;
        if tok.is_stop(next) {
            break;
        }
        generated.push(next);
        out.extend_from_slice(tok.token_bytes(next));
        logits = ctx.forward(next);
    }
    Some((generated, String::from_utf8_lossy(&out).into_owned()))
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
    assert_eq!(
        text,
        " Paris. The capital of Germany is Berlin. The capital of"
    );
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
    let Some(text) = greedy_in(
        "qwen3-4b-q4km.nrm",
        "What is the capital of France?",
        20,
        true,
    ) else {
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

// ---- Granite 4.1 3B Q4_K_M pins (verified against llama-completion
// --temp 0 -c 4096 on the identical GGUF). Granite's /10 logit scaling
// compresses greedy gaps, so some prompts flip on near-ties (top-2 gap
// < ~0.4) between numerically-equivalent implementations; the pins below
// matched llama.cpp token-for-token. See docs/architecture.md.

#[test]
fn granite_greedy_matches_llama_cpp_capital() {
    let _guard = serial();
    let Some(text) = greedy_in(
        "granite-4.1-3b-q4km.nrm",
        "The capital of France is",
        12,
        false,
    ) else {
        eprintln!("SKIP: models/granite-4.1-3b-q4km.nrm not present");
        return;
    };
    assert_eq!(
        text,
        " Paris. It is located in the northern part of the country"
    );
}

#[test]
fn granite_greedy_matches_llama_cpp_colors() {
    let _guard = serial();
    let Some(text) = greedy_in(
        "granite-4.1-3b-q4km.nrm",
        "The three primary colors are",
        16,
        false,
    ) else {
        eprintln!("SKIP: models/granite-4.1-3b-q4km.nrm not present");
        return;
    };
    assert_eq!(
        text,
        " red, blue, and yellow. These colors are fundamental because they can be combined"
    );
}

#[test]
fn granite_chat_greedy_matches_llama_cpp() {
    let _guard = serial();
    let Some(text) = greedy_in(
        "granite-4.1-3b-q4km.nrm",
        "What is the capital of France?",
        20,
        true,
    ) else {
        eprintln!("SKIP: models/granite-4.1-3b-q4km.nrm not present");
        return;
    };
    assert_eq!(text, "The capital of France is Paris.");
}

/// Tied-head + scalar audit: the artifact has no output tensor, the Q6_K
/// embedding doubles as classifier, and the four muP scalars round-trip
/// through the .nrm header exactly.
#[test]
fn granite_metadata_audit() {
    let _guard = serial();
    let Some(blob) = load_model_blob("granite-4.1-3b-q4km.nrm") else {
        eprintln!("SKIP: models/granite-4.1-3b-q4km.nrm not present");
        return;
    };
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let m = &model.meta;
    assert_eq!(m.arch, nr_model::format::Arch::Granite);
    assert!(m.flags & nr_model::format::FLAG_TIED_EMBEDDINGS != 0);
    assert!(!model.has_tensor(nr_model::TensorKind::Output));
    assert!(!model.has_tensor(nr_model::TensorKind::AttnQNorm));
    assert_eq!(m.embed_scale, 12.0);
    assert_eq!(m.attn_scale, 0.015625);
    assert_eq!(m.residual_scale, 0.22);
    assert_eq!(m.logit_scale, 10.0);
    assert_eq!(m.head_dim, 64);
    let embed = model.tensor(nr_model::TensorKind::TokEmbed, 0).unwrap();
    assert_eq!(embed.dtype, nr_model::TensorDtype::Q6K);
    assert_eq!((embed.rows, embed.cols), (100352, 2560));
}

/// Neutral scalars must be exact no-ops (llama/qwen numerics unchanged).
#[test]
fn neutral_scalars_are_noops() {
    let _guard = serial();
    for file in ["model.nrm", "qwen3-4b-q4km.nrm"] {
        let Some(blob) = load_model_blob(file) else {
            eprintln!("SKIP: models/{file} not present");
            continue;
        };
        let m = nr_model::Model::parse(&blob).expect("parse").meta;
        assert_eq!(m.embed_scale, 1.0, "{file}");
        assert_eq!(m.attn_scale, 0.0, "{file}"); // 0 => 1/sqrt(head_dim)
        assert_eq!(m.residual_scale, 1.0, "{file}");
        assert_eq!(m.logit_scale, 1.0, "{file}");
        // saxpy with 1.0 and scale_inplace skipping are IEEE-exact no-ops.
        let mut y = [1.5f32, -2.25, 3.75];
        let x = [0.5f32, 0.25, -1.0];
        let mut y2 = y;
        nr_tensor::vec::saxpy(&mut y, 1.0, &x);
        nr_tensor::vec::add_assign(&mut y2, &x);
        assert_eq!(y, y2);
    }
}

// Bonsai-8B Q1_0: reference uses the GGUF's exact generation suffix,
// <|im_start|>assistant\n<think>\n\n</think>\n\n (thinking disabled).
#[test]
fn bonsai_chat_greedy_matches_llama_cpp() {
    let _guard = serial();
    let Some((ids, text)) = greedy_tokens_in(
        "bonsai-8b-q1.nrm",
        "What is the capital of France?",
        20,
        true,
    ) else {
        eprintln!("SKIP: models/bonsai-8b-q1.nrm not present");
        return;
    };
    assert_eq!(text, "The capital of France is Paris.");
    assert_eq!(ids, [785, 6722, 315, 9625, 374, 12095, 13]);
}

#[test]
fn bonsai_metadata_and_template_audit() {
    use nr_model::format::{Arch, TensorDtype, TensorKind, FLAG_ROPE_YARN, FLAG_TIED_EMBEDDINGS};
    let _guard = serial();
    let Some(blob) = load_model_blob("bonsai-8b-q1.nrm") else {
        eprintln!("SKIP: models/bonsai-8b-q1.nrm not present");
        return;
    };
    let model = nr_model::Model::parse(&blob).expect("parse");
    let m = &model.meta;
    assert_eq!(m.arch, Arch::Qwen3);
    assert_eq!(
        (m.dim, m.n_layers, m.n_heads, m.n_kv_heads, m.head_dim),
        (4096, 36, 32, 8, 128)
    );
    assert_eq!(m.flags & FLAG_TIED_EMBEDDINGS, 0);
    assert_ne!(m.flags & FLAG_ROPE_YARN, 0);
    assert_eq!(
        (m.rope_factor, m.rope_orig_ctx, m.rope_low, m.rope_high),
        (4.0, 16384.0, 32.0, 1.0)
    );
    assert_eq!(m.rope_attn_factor, 1.0);
    for kind in [TensorKind::TokEmbed, TensorKind::Output] {
        let t = model.tensor(kind, 0).unwrap();
        assert_eq!(t.dtype, TensorDtype::Q1_0);
        assert_eq!((t.rows, t.cols), (151669, 4096));
    }
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).unwrap();
    assert_eq!(tok.template, nr_token::blob::Template::Bonsai);
    let mut ids = Vec::new();
    tok.encode_conversation_start(None, &mut ids);
    assert!(ids.is_empty());
    tok.encode_message("user", "What is the capital of France?", &mut ids);
    tok.encode_header("assistant", &mut ids);
    // Token IDs independently verified with llama-completion --verbose-prompt.
    assert_eq!(
        ids,
        [
            151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
            151667, 271, 151668, 271
        ]
    );
    assert!(tok.is_stop(151645));
    ids.clear();
    tok.encode_text("<think></think>", &mut ids);
    assert!(
        !ids.contains(&151667) && !ids.contains(&151668),
        "user text must not inject control tokens"
    );
}

#[test]
fn bonsai_raw_greedy_matches_llama_cpp() {
    let _guard = serial();
    let Some((ids, text)) =
        greedy_tokens_in("bonsai-8b-q1.nrm", "The capital of France is", 12, false)
    else {
        eprintln!("SKIP: models/bonsai-8b-q1.nrm not present");
        return;
    };
    assert_eq!(text, " Paris. Paris is the capital of France. Paris is the");
    assert_eq!(
        ids,
        [12095, 13, 12095, 374, 279, 6722, 315, 9625, 13, 12095, 374, 279]
    );
}
