//! nrconvert: GGUF (Q8_0) -> NightRun .nrm converter.
//!
//! Usage: nrconvert <input.gguf> <output.nrm>

mod gguf;
mod tokenizer;

use nr_model::crc32::Crc32;
use nr_model::format::{self, TensorDtype, TensorKind};

use crate::gguf::{Gguf, TensorInfo, GGML_F32, GGML_Q8_0};

fn kv_u32(g: &Gguf, key: &str) -> u32 {
    g.kv.get(key).and_then(gguf::Value::as_u32).unwrap_or_else(|| panic!("missing {key}"))
}

fn kv_f32(g: &Gguf, key: &str) -> f32 {
    g.kv.get(key).and_then(gguf::Value::as_f32).unwrap_or_else(|| panic!("missing {key}"))
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = &args[..] else {
        eprintln!("usage: nrconvert <input.gguf> <output.nrm>");
        std::process::exit(2);
    };

    println!("parsing {input} ...");
    let g = gguf::parse(input);
    let arch = g.kv["general.architecture"].as_str().unwrap();
    assert_eq!(arch, "llama", "expected a llama-architecture GGUF");

    let dim = kv_u32(&g, "llama.embedding_length");
    let n_layers = kv_u32(&g, "llama.block_count");
    let n_heads = kv_u32(&g, "llama.attention.head_count");
    let n_kv_heads = kv_u32(&g, "llama.attention.head_count_kv");
    let ffn_dim = kv_u32(&g, "llama.feed_forward_length");
    let ctx_train = kv_u32(&g, "llama.context_length");
    let rope_theta = kv_f32(&g, "llama.rope.freq_base");
    let norm_eps = kv_f32(&g, "llama.attention.layer_norm_rms_epsilon");
    let head_dim = dim / n_heads;
    let vocab = g.kv["tokenizer.ggml.tokens"].as_arr().unwrap().len() as u32;

    println!("building tokenizer blob ...");
    let tok = tokenizer::build(&g);
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
            GGML_Q8_0 => TensorDtype::Q8_0,
            other => panic!("{}: unsupported dtype {other} (convert a Q8_0 GGUF)", t.name),
        };
        let cols = t.dims[0] as u32;
        let rows = t.dims.get(1).copied().unwrap_or(1) as u32;
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
    let rope_factor = g.kv.get("llama.rope.scaling.factor").and_then(gguf::Value::as_f32).unwrap_or(0.0);

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
    let data_off = (table_off + table_size).div_ceil(format::DATA_ALIGN as u64) * format::DATA_ALIGN as u64;

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
    println!("checksumming tensor data ({} MB) ...", data_size / (1024 * 1024));
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
    let mut flags = 0u32;
    if tied {
        flags |= format::FLAG_TIED_EMBEDDINGS;
    }
    let mut header = Vec::with_capacity(format::HEADER_SIZE);
    header.extend_from_slice(&format::MAGIC);
    header.extend_from_slice(&format::VERSION.to_le_bytes());
    for v in [dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, vocab, ctx_train] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    for v in [rope_theta, norm_eps, rope_factor, 0.0, 0.0, 0.0] {
        header.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    header.extend_from_slice(&flags.to_le_bytes());
    let name = g
        .kv
        .get("general.name")
        .and_then(gguf::Value::as_str)
        .unwrap_or("Unknown Model");
    let mut name_bytes = [0u8; format::NAME_LEN];
    let n = name.len().min(format::NAME_LEN);
    name_bytes[..n].copy_from_slice(&name.as_bytes()[..n]);
    header.extend_from_slice(&name_bytes);
    header.extend_from_slice(&tok_off.to_le_bytes());
    header.extend_from_slice(&(tok.blob.len() as u64).to_le_bytes());
    header.extend_from_slice(&table_off.to_le_bytes());
    header.extend_from_slice(&(out.len() as u32).to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
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
        "ok: {} MB, {} tensors, dim={} layers={} heads={}/{} ffn={} vocab={} tied={} ctx={}",
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
