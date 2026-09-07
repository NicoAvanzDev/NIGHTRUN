//! nrconvert: GGUF (PQ2_0 / Q8_0 / Q4_K / Q6_K) -> NightRun .nrm converter.
//!
//! Usage: nrconvert <input.gguf> <output.nrm>

mod gguf;
mod tokenizer;

use nr_model::crc32::Crc32;
use nr_model::format::{self, TensorDtype, TensorKind};

use crate::gguf::{Gguf, TensorInfo, GGML_F32, GGML_PQ2_0, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};

fn kv_u32(g: &Gguf, key: &str) -> u32 {
    g.kv.get(key)
        .and_then(gguf::Value::as_u32)
        .unwrap_or_else(|| panic!("missing {key}"))
}

fn kv_f32(g: &Gguf, key: &str) -> f32 {
    g.kv.get(key)
        .and_then(gguf::Value::as_f32)
        .unwrap_or_else(|| panic!("missing {key}"))
}

/// Human label for the artifact's quantization recipe, from GGUF
/// general.file_type (llama.cpp ftype enum).
fn quant_label(g: &Gguf) -> &'static str {
    match g.kv.get("general.file_type").and_then(gguf::Value::as_u32) {
        Some(141) => "PQ2_0",
        Some(7) => "Q8_0",
        Some(14) => "Q4_K_S",
        Some(15) => "Q4_K_M",
        Some(17) => "Q5_K_M",
        Some(18) => "Q6_K",
        _ => "quantized",
    }
}

struct OutTensor<'a> {
    kind: TensorKind,
    layer: u16,
    dtype: TensorDtype,
    rows: u32,
    cols: u32,
    bytes: &'a [u8],
    offset: u64,
}

/// GGML dtype id -> human name for the formats we care about.
fn ggml_dtype_name(t: u32) -> String {
    match t {
        0 => "F32".into(),
        1 => "F16".into(),
        GGML_PQ2_0 => "PQ2_0".into(),
        8 => "Q8_0".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        other => format!("ggml_type_{other}"),
    }
}

/// Architecture-kind verdict: NightRun supports conventional dense
/// decoder-only transformers only.
fn arch_verdict(g: &Gguf, arch: &str) -> Result<(), String> {
    match arch {
        "llama" | "qwen3" | "granite" => {}
        "granitehybrid" | "granitemoehybrid" => {
            return Err(format!(
                "Unsupported Granite artifact:\n  detected architecture: {arch} (hybrid transformer + state-space)\n  NightRun Granite support targets the conventional dense transformer variant only.\n  Use a dense Granite GGUF (general.architecture == \"granite\"), e.g. ibm-granite/granite-4.1-3b-GGUF."
            ));
        }
        other => {
            return Err(format!(
                "unsupported architecture {other:?} (supported: llama, qwen3, granite)"
            ))
        }
    }
    // Belt and braces: reject SSM/MoE features even under a supported name.
    for key in [
        "ssm_conv_kernel",
        "ssm_state_size",
        "expert_count",
        "expert_used_count",
    ] {
        let k = format!("{arch}.{key}");
        if let Some(v) = g.kv.get(&k) {
            if v.as_u32().unwrap_or(0) != 0 {
                return Err(format!(
                    "unsupported architecture feature: {k} = {v:?} (SSM/MoE models are out of scope; use a conventional dense transformer artifact)"
                ));
            }
        }
    }
    if let Some(t) = g
        .tensors
        .iter()
        .find(|t| t.name.contains(".ssm_") || t.name.contains("ffn_gate_exps"))
    {
        return Err(format!(
            "unsupported tensor {:?}: SSM/MoE layers detected (dense transformers only)",
            t.name
        ));
    }
    Ok(())
}

/// Dump the per-tensor dtype table of a GGUF without converting.
fn inspect(input: &str) {
    let g = gguf::parse(input);
    let arch =
        g.kv.get("general.architecture")
            .and_then(gguf::Value::as_str)
            .unwrap_or("?");
    let name =
        g.kv.get("general.name")
            .and_then(gguf::Value::as_str)
            .unwrap_or("?");
    println!("arch={arch} name={name:?} tensors={}", g.tensors.len());
    match arch_verdict(&g, arch) {
        Ok(()) => println!("verdict: conventional dense transformer (supported)"),
        Err(e) => println!("verdict: REJECTED - {e}"),
    }
    for key in [
        "embedding_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "feed_forward_length",
        "rope.freq_base",
        "rope.scaling.type",
        "attention.layer_norm_rms_epsilon",
        "context_length",
        // Granite muP scalars.
        "embedding_scale",
        "attention.scale",
        "residual_scale",
        "logit_scale",
    ] {
        let k = format!("{arch}.{key}");
        if let Some(v) = g.kv.get(&k) {
            println!("  {k} = {v:?}");
        }
    }
    for key in [
        "general.file_type",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.add_bos_token",
    ] {
        if let Some(v) = g.kv.get(key) {
            println!("  {key} = {v:?}");
        }
    }
    // Aggregate dtype mix per tensor role (strip blk.N. prefixes).
    let mut by_role: std::collections::BTreeMap<(String, String), (u32, u64)> = Default::default();
    for t in &g.tensors {
        let role = match t.name.strip_prefix("blk.") {
            Some(rest) => rest
                .split_once('.')
                .map(|(_, r)| r.to_string())
                .unwrap_or(rest.into()),
            None => t.name.clone(),
        };
        let e = by_role
            .entry((role, ggml_dtype_name(t.dtype)))
            .or_insert((0, 0));
        e.0 += 1;
        e.1 += t.byte_size;
    }
    println!("{:<28} {:>6} {:>5} {:>9}", "role", "dtype", "count", "MB");
    for ((role, dtype), (count, bytes)) in &by_role {
        println!(
            "{role:<28} {dtype:>6} {count:>5} {:>9.1}",
            *bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let ["--inspect", input] = &args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        inspect(input);
        return;
    }
    let [input, output] = &args[..] else {
        eprintln!("usage: nrconvert <input.gguf> <output.nrm> | nrconvert --inspect <input.gguf>");
        std::process::exit(2);
    };

    println!("parsing {input} ...");
    let g = gguf::parse(input);
    let arch = g.kv["general.architecture"].as_str().unwrap();
    if let Err(e) = arch_verdict(&g, arch) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let (arch_id, family) = match arch {
        "llama" => (1u32, tokenizer::Family::Llama3),
        "qwen3" => (2u32, tokenizer::Family::Qwen3),
        "granite" => (3u32, tokenizer::Family::Granite),
        other => panic!("unsupported architecture {other}"),
    };

    let akey = |suffix: &str| format!("{arch}.{suffix}");
    let dim = kv_u32(&g, &akey("embedding_length"));
    let n_layers = kv_u32(&g, &akey("block_count"));
    let n_heads = kv_u32(&g, &akey("attention.head_count"));
    let n_kv_heads = kv_u32(&g, &akey("attention.head_count_kv"));
    let ffn_dim = kv_u32(&g, &akey("feed_forward_length"));
    let ctx_train = kv_u32(&g, &akey("context_length"));
    let rope_theta = kv_f32(&g, &akey("rope.freq_base"));
    let norm_eps = kv_f32(&g, &akey("attention.layer_norm_rms_epsilon"));
    // Qwen3 declares an explicit head size (attention width != hidden dim).
    let head_dim =
        g.kv.get(&akey("attention.key_length"))
            .and_then(gguf::Value::as_u32)
            .unwrap_or(dim / n_heads);
    let vocab = g.kv["tokenizer.ggml.tokens"].as_arr().unwrap().len() as u32;

    println!("building tokenizer blob ...");
    let tok = tokenizer::build(&g, family);
    println!("  vocab={} merges={}", tok.vocab, tok.merges);

    // Collect tensors in inference-friendly order.
    let find = |name: &str| -> Option<&TensorInfo> { g.tensors.iter().find(|t| t.name == name) };
    let expect = |name: &str| -> &TensorInfo {
        find(name).unwrap_or_else(|| panic!("gguf missing tensor {name}"))
    };

    let mut out: Vec<OutTensor> = Vec::new();
    let mut push = |t: &TensorInfo, kind: TensorKind, layer: u16| {
        let dtype = match t.dtype {
            GGML_F32 => TensorDtype::F32,
            GGML_PQ2_0 => TensorDtype::PQ2_0,
            GGML_Q8_0 => TensorDtype::Q8_0,
            GGML_Q4_K => TensorDtype::Q4K,
            GGML_Q6_K => TensorDtype::Q6K,
            other => panic!(
                "{}: unsupported dtype {other} (supported: F32, PQ2_0, Q8_0, Q4_K, Q6_K)",
                t.name
            ),
        };
        let cols = t.dims[0] as u32;
        let rows = t.dims.get(1).copied().unwrap_or(1) as u32;
        assert!(
            dtype != TensorDtype::PQ2_0 || cols.is_multiple_of(128),
            "{}: PQ2_0 row width must be divisible by 128",
            t.name
        );
        out.push(OutTensor {
            kind,
            layer,
            dtype,
            rows,
            cols,
            bytes: &g.data[t.file_offset as usize..(t.file_offset + t.byte_size) as usize],
            offset: 0,
        });
    };

    push(expect("token_embd.weight"), TensorKind::TokEmbed, 0);
    if let Some(t) = find("rope_freqs.weight") {
        push(t, TensorKind::RopeFreqs, 0);
    }
    for l in 0..n_layers as u16 {
        let n = |suffix: &str| format!("blk.{l}.{suffix}.weight");
        push(expect(&n("attn_norm")), TensorKind::AttnNorm, l);
        if arch_id == 2 {
            // Qwen3: per-head RMSNorm weights on Q and K (required).
            push(expect(&n("attn_q_norm")), TensorKind::AttnQNorm, l);
            push(expect(&n("attn_k_norm")), TensorKind::AttnKNorm, l);
        }
        push(expect(&n("attn_q")), TensorKind::AttnQ, l);
        push(expect(&n("attn_k")), TensorKind::AttnK, l);
        push(expect(&n("attn_v")), TensorKind::AttnV, l);
        push(expect(&n("attn_output")), TensorKind::AttnO, l);
        push(expect(&n("ffn_norm")), TensorKind::FfnNorm, l);
        push(expect(&n("ffn_gate")), TensorKind::FfnGate, l);
        push(expect(&n("ffn_up")), TensorKind::FfnUp, l);
        push(expect(&n("ffn_down")), TensorKind::FfnDown, l);
    }
    push(expect("output_norm.weight"), TensorKind::OutputNorm, 0);
    let tied = match find("output.weight") {
        Some(t) => {
            push(t, TensorKind::Output, 0);
            false
        }
        None => true,
    };

    // Llama-3 rope scaling params (present in HF config; GGUF carries the
    // precomputed rope_freqs tensor instead, which we prefer at runtime).
    let rope_factor =
        g.kv.get("llama.rope.scaling.factor")
            .and_then(gguf::Value::as_f32)
            .unwrap_or(0.0);

    // Bonsai's Qwen3 GGUF uses YaRN, including magnitude scaling.
    let yarn = arch_id == 2
        && g.kv
            .get(&akey("rope.scaling.type"))
            .and_then(gguf::Value::as_str)
            == Some("yarn");
    let optional_float = |key: &str, default| {
        g.kv.get(&akey(key))
            .and_then(gguf::Value::as_f32)
            .unwrap_or(default)
    };
    let (rope_factor, rope_low, rope_high, rope_orig_ctx, rope_attn_factor) = if yarn {
        (
            kv_f32(&g, &akey("rope.scaling.factor")),
            optional_float("rope.scaling.yarn_beta_fast", 32.0),
            optional_float("rope.scaling.yarn_beta_slow", 1.0),
            kv_u32(&g, &akey("rope.scaling.original_context_length")) as f32,
            optional_float("rope.scaling.attn_factor", 1.0),
        )
    } else {
        (rope_factor, 0.0, 0.0, 0.0, 1.0)
    };

    // Granite muP scalars: required semantics for granite (no invented
    // defaults); neutral values for other families. attn_scale 0.0 means
    // "use 1/sqrt(head_dim)" at runtime.
    let (embed_scale, attn_scale, residual_scale, logit_scale) = if arch_id == 3 {
        let req = |key: &str| -> f32 {
            g.kv.get(&akey(key))
                .and_then(gguf::Value::as_f32)
                .unwrap_or_else(|| panic!("granite artifact missing required scalar {}", akey(key)))
        };
        let scalars = (
            req("embedding_scale"),
            req("attention.scale"),
            req("residual_scale"),
            req("logit_scale"),
        );
        println!(
            "granite scalars: embed x{} attn x{} residual x{} logits /{}",
            scalars.0, scalars.1, scalars.2, scalars.3
        );
        scalars
    } else {
        (1.0, 0.0, 1.0, 1.0)
    };

    // Assign 64-byte-aligned data offsets.
    let mut cursor = 0u64;
    for t in out.iter_mut() {
        cursor = cursor.div_ceil(format::DATA_ALIGN as u64) * format::DATA_ALIGN as u64;
        t.offset = cursor;
        cursor += t.bytes.len() as u64;
    }
    let data_size = cursor;

    // Layout: header | tokenizer | table | data.
    let tok_off = format::HEADER_SIZE as u64;
    let table_off = tok_off + tok.blob.len() as u64;
    let table_size = out.len() as u64 * format::ENTRY_SIZE as u64;
    let data_off =
        (table_off + table_size).div_ceil(format::DATA_ALIGN as u64) * format::DATA_ALIGN as u64;

    // Serialize the tensor table.
    let mut table = Vec::with_capacity(table_size as usize);
    for t in &out {
        table.extend_from_slice(&(t.kind as u16).to_le_bytes());
        table.extend_from_slice(&t.layer.to_le_bytes());
        table.extend_from_slice(&(t.dtype as u16).to_le_bytes());
        table.extend_from_slice(&0u16.to_le_bytes());
        table.extend_from_slice(&t.offset.to_le_bytes());
        table.extend_from_slice(&(t.bytes.len() as u64).to_le_bytes());
        table.extend_from_slice(&t.rows.to_le_bytes());
        table.extend_from_slice(&t.cols.to_le_bytes());
    }

    // Data CRC.
    println!(
        "checksumming tensor data ({} MB) ...",
        data_size / (1024 * 1024)
    );
    let mut dcrc = Crc32::new();
    {
        let mut pos = 0u64;
        for t in &out {
            if t.offset > pos {
                dcrc.update(&vec![0u8; (t.offset - pos) as usize]);
            }
            dcrc.update(t.bytes);
            pos = t.offset + t.bytes.len() as u64;
        }
        if data_size > pos {
            dcrc.update(&vec![0u8; (data_size - pos) as usize]);
        }
    }
    let data_crc = dcrc.finish();

    // Serialize the header (see nr-model::format for the layout).
    let mut flags = if yarn { format::FLAG_ROPE_YARN } else { 0 };
    if tied {
        flags |= format::FLAG_TIED_EMBEDDINGS;
    }
    let mut header = Vec::with_capacity(format::HEADER_SIZE);
    header.extend_from_slice(&format::MAGIC);
    header.extend_from_slice(&format::VERSION.to_le_bytes());
    header.extend_from_slice(&arch_id.to_le_bytes());
    for v in [
        dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, vocab, ctx_train,
    ] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    for v in [
        rope_theta,
        norm_eps,
        rope_factor,
        rope_low,
        rope_high,
        rope_orig_ctx,
    ] {
        header.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    header.extend_from_slice(&flags.to_le_bytes());
    for v in [embed_scale, attn_scale, residual_scale, logit_scale] {
        header.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    // Ternary Bonsai omits general.name, but still provides its identity
    // in general.basename and general.size_label for the chat heading.
    let fallback_name = ["general.basename", "general.size_label"]
        .iter()
        .filter_map(|key| g.kv.get(*key).and_then(gguf::Value::as_str))
        .collect::<Vec<_>>()
        .join(" ");
    let base_name =
        g.kv.get("general.name")
            .and_then(gguf::Value::as_str)
            .unwrap_or(if fallback_name.is_empty() {
                "Unknown Model"
            } else {
                &fallback_name
            });
    let name = format!("{base_name} {}", quant_label(&g));
    let mut name_bytes = [0u8; format::NAME_LEN];
    let n = name.len().min(format::NAME_LEN);
    name_bytes[..n].copy_from_slice(&name.as_bytes()[..n]);
    header.extend_from_slice(&name_bytes);
    header.extend_from_slice(&tok_off.to_le_bytes());
    header.extend_from_slice(&(tok.blob.len() as u64).to_le_bytes());
    header.extend_from_slice(&table_off.to_le_bytes());
    header.extend_from_slice(&(out.len() as u32).to_le_bytes());
    header.extend_from_slice(&rope_attn_factor.to_le_bytes());
    header.extend_from_slice(&data_off.to_le_bytes());
    header.extend_from_slice(&data_size.to_le_bytes());
    header.extend_from_slice(&data_crc.to_le_bytes());

    // Metadata CRC: header with both crc fields zeroed + tokenizer + table
    // (matches Model::parse).
    let mut mcrc = Crc32::new();
    mcrc.update(&header[..format::HEADER_SIZE - 8]);
    mcrc.update(&[0u8; 8]);
    mcrc.update(&tok.blob);
    mcrc.update(&table);
    let meta_crc = mcrc.finish();
    header.extend_from_slice(&meta_crc.to_le_bytes());
    assert_eq!(header.len(), format::HEADER_SIZE);

    // Write the file.
    println!("writing {output} ...");
    let mut file = Vec::with_capacity((data_off + data_size) as usize);
    file.extend_from_slice(&header);
    file.extend_from_slice(&tok.blob);
    file.extend_from_slice(&table);
    file.resize(data_off as usize, 0);
    for t in &out {
        let abs = (data_off + t.offset) as usize;
        file.resize(abs, 0);
        file.extend_from_slice(t.bytes);
    }
    file.resize((data_off + data_size) as usize, 0);
    std::fs::write(output, &file).expect("write output");

    // Self-check: parse what we wrote.
    println!("verifying ...");
    let model = nr_model::Model::parse(&file).expect("self-parse failed");
    assert!(model.verify_data(|_, _| {}), "data crc self-check failed");
    let tk = nr_token::Tokenizer::parse(model.tokenizer_blob).expect("tokenizer self-parse");
    let mut ids = Vec::new();
    tk.encode_text("Hello, world!", &mut ids);
    println!(
        "ok ({name}): {} MB, {} tensors, dim={} layers={} heads={}/{} ffn={} vocab={} tied={} ctx={}",
        file.len() / (1024 * 1024),
        out.len(),
        model.meta.dim,
        model.meta.n_layers,
        model.meta.n_heads,
        model.meta.n_kv_heads,
        model.meta.ffn_dim,
        model.meta.vocab,
        tied,
        model.meta.ctx_train,
    );
    println!("tokenizer smoke: \"Hello, world!\" -> {ids:?}");
}
