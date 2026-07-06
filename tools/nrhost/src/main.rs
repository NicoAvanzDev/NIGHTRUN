//! nrhost: run the NightRun inference engine on the host (same code path
//! as bare metal) for debugging, parity tests and benchmarks.
//!
//! Usage:
//!   nrhost <model.nrm> --prompt "..." [-n 64] [--temp 0.7] [--top-p 0.9]
//!          [--raw] [--ctx 4096] [--seed 1234]
//!
//! --raw skips the chat template (plain completion; used for parity tests).

use std::io::Write as _;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: nrhost <model.nrm> [opts]");
    let mut prompt = String::from("Why is the sky blue?");
    let mut n_tokens = 64usize;
    let mut temp = 0.7f32;
    let mut top_p = 0.9f32;
    let mut raw = false;
    let mut ctx = 4096usize;
    let mut seed = 0x5eed_cafe_u64;
    let mut threads = 1usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().unwrap().parse().unwrap(),
            "--prompt" => prompt = args.next().unwrap(),
            "-n" => n_tokens = args.next().unwrap().parse().unwrap(),
            "--temp" => temp = args.next().unwrap().parse().unwrap(),
            "--top-p" => top_p = args.next().unwrap().parse().unwrap(),
            "--ctx" => ctx = args.next().unwrap().parse().unwrap(),
            "--seed" => seed = args.next().unwrap().parse().unwrap(),
            "--raw" => raw = true,
            other => panic!("unknown arg {other}"),
        }
    }

    if threads > 1 {
        let workers = threads - 1;
        for _ in 0..workers {
            std::thread::spawn(|| nr_tensor::parallel::POOL.worker_loop());
        }
        while nr_tensor::parallel::POOL.ready_workers() < workers {
            std::thread::yield_now();
        }
        nr_tensor::parallel::POOL.activate(workers);
        eprintln!("{threads} threads ({workers} workers)");
    }

    eprintln!("loading {model_path} ...");
    let blob = std::fs::read(&model_path).expect("read model");
    let model = nr_model::Model::parse(&blob).expect("parse model");
    let tok = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("parse tokenizer");
    eprintln!(
        "model: {} ({} layers, dim {}, vocab {})",
        model.meta.name_str(),
        model.meta.n_layers,
        model.meta.dim,
        model.meta.vocab
    );

    let mut alloc = |bytes: usize, align: usize| -> *mut u8 {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).unwrap();
        // Leaked on purpose: mirrors the bare-metal arena lifetime.
        unsafe { std::alloc::alloc_zeroed(layout) }
    };
    let mut ictx = nr_model::InferCtx::new(&model, ctx, &mut alloc).expect("infer ctx");
    let mut sampler = nr_model::Sampler::new(temp, top_p, seed);

    // Build the token sequence.
    let mut ids: Vec<u32> = Vec::new();
    if raw {
        ids.push(tok.specials.bos);
        tok.encode_text(&prompt, &mut ids);
    } else {
        tok.encode_conversation_start(None, &mut ids);
        tok.encode_message(nr_token::template::ROLE_USER, &prompt, &mut ids);
        tok.encode_header(nr_token::template::ROLE_ASSISTANT, &mut ids);
    }
    eprintln!("prompt tokens: {ids:?}");

    // Prefill.
    let t0 = std::time::Instant::now();
    let mut logits: &[f32] = &[];
    for &id in &ids {
        logits = ictx.forward(id);
    }
    let prefill = t0.elapsed();
    eprintln!(
        "prefill: {} tokens in {:.2}s ({:.2} tok/s)",
        ids.len(),
        prefill.as_secs_f64(),
        ids.len() as f64 / prefill.as_secs_f64()
    );

    // Generate.
    let t0 = std::time::Instant::now();
    let mut generated = Vec::new();
    let mut out_bytes: Vec<u8> = Vec::new();
    for _ in 0..n_tokens {
        let next = sampler.sample(logits);
        if tok.is_stop(next) {
            break;
        }
        generated.push(next);
        out_bytes.extend_from_slice(tok.token_bytes(next));
        print!("{}", String::from_utf8_lossy(tok.token_bytes(next)));
        std::io::stdout().flush().unwrap();
        if ictx.remaining() == 0 {
            break;
        }
        logits = ictx.forward(next);
    }
    println!();
    let gen = t0.elapsed();
    eprintln!(
        "generated {} tokens in {:.2}s ({:.2} tok/s)",
        generated.len(),
        gen.as_secs_f64(),
        generated.len() as f64 / gen.as_secs_f64()
    );
    eprintln!("ids: {generated:?}");
}
