//! Build the NRTK tokenizer blob from GGUF tokenizer metadata.

use std::collections::HashMap;

use crate::gguf::{Gguf, Value};

/// GPT-2 byte-level encoding: maps the 256 bytes to printable unicode
/// codepoints. Returns codepoint -> byte.
fn unicode_to_byte() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!'..=b'~').map(u32::from).collect();
    bs.extend(0xa1..=0xac_u32);
    bs.extend(0xae..=0xff_u32);
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

// llama.cpp token types.
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

pub struct BuiltTokenizer {
    pub blob: Vec<u8>,
    pub vocab: usize,
    pub merges: usize,
}

/// Template/family selector, written into the blob header.
#[derive(Clone, Copy, PartialEq)]
pub enum Family {
    Llama3,
    Qwen3,
}

pub fn build(gguf: &Gguf, family: Family) -> BuiltTokenizer {
    let model = gguf.kv["tokenizer.ggml.model"].as_str().expect("tokenizer model");
    assert_eq!(model, "gpt2", "expected byte-level BPE (gpt2-style) tokenizer");

    let tokens = gguf.kv["tokenizer.ggml.tokens"].as_arr().expect("tokens");
    let types: Vec<i32> = gguf.kv["tokenizer.ggml.token_type"]
        .as_arr()
        .expect("token_type")
        .iter()
        .map(|v| match v {
            Value::I32(t) => *t,
            other => other.as_u32().map(|u| u as i32).unwrap_or(1),
        })
        .collect();
    let merges = gguf.kv["tokenizer.ggml.merges"].as_arr().expect("merges");
    let eos = gguf.kv["tokenizer.ggml.eos_token_id"].as_u32().expect("eos id");
    // Qwen has no BOS (its template never emits one); fall back to eos so
    // the slot holds a valid id either way.
    let bos = gguf
        .kv
        .get("tokenizer.ggml.bos_token_id")
        .and_then(Value::as_u32)
        .unwrap_or(eos);

    // Hard check: the GGUF's pretokenizer id must match the family we're
    // baking in, or runtime tokenization would silently diverge.
    if let Some(pre) = gguf.kv.get("tokenizer.ggml.pre").and_then(Value::as_str) {
        let expect = match family {
            Family::Llama3 => "llama-bpe",
            Family::Qwen3 => "qwen2",
        };
        assert_eq!(pre, expect, "pretokenizer mismatch: gguf says {pre:?}, family expects {expect:?}");
    }

    let u2b = unicode_to_byte();
    let decode = |s: &str, control: bool| -> Vec<u8> {
        if control {
            return s.as_bytes().to_vec();
        }
        let mut out = Vec::with_capacity(s.len());
        for ch in s.chars() {
            match u2b.get(&ch) {
                Some(&b) => out.push(b),
                None => out.extend_from_slice(ch.to_string().as_bytes()),
            }
        }
        out
    };

    // Decode all tokens to raw bytes.
    let mut raw_by_literal: HashMap<&str, u32> = HashMap::new();
    let mut decoded: Vec<Vec<u8>> = Vec::with_capacity(tokens.len());
    let mut flags: Vec<u16> = Vec::with_capacity(tokens.len());
    for (id, tok) in tokens.iter().enumerate() {
        let s = tok.as_str().expect("token string");
        let control = matches!(types.get(id), Some(&TYPE_CONTROL) | Some(&TYPE_USER_DEFINED));
        raw_by_literal.insert(s, id as u32);
        decoded.push(decode(s, control));
        flags.push(if control { nr_token::blob::FLAG_CONTROL } else { 0 });
    }

    // Reverse index over decoded bytes (first id wins; skip control tokens
    // so e.g. a literal "<|eot_id|>" typed by the user never encodes to the
    // control token).
    let mut by_bytes: HashMap<&[u8], u32> = HashMap::new();
    for (id, bytes) in decoded.iter().enumerate() {
        if flags[id] & nr_token::blob::FLAG_CONTROL == 0 {
            by_bytes.entry(bytes.as_slice()).or_insert(id as u32);
        }
    }

    // Byte -> token id table.
    let mut byte_ids = [0u32; 256];
    for b in 0..256usize {
        let single = [b as u8];
        byte_ids[b] = *by_bytes
            .get(single.as_slice())
            .unwrap_or_else(|| panic!("vocab has no token for byte {b:#04x}"));
    }

    // Resolve merges "A B" -> (left, right, result, rank).
    let mut resolved: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(merges.len());
    for (rank, m) in merges.iter().enumerate() {
        let s = m.as_str().expect("merge string");
        let (a, b) = s.split_once(' ').expect("merge pair");
        let ab = {
            let mut v = decode(a, false);
            v.extend_from_slice(&decode(b, false));
            v
        };
        let (Some(&l), Some(&r), Some(&res)) = (
            by_bytes.get(decode(a, false).as_slice()),
            by_bytes.get(decode(b, false).as_slice()),
            by_bytes.get(ab.as_slice()),
        ) else {
            // Merges over tokens that don't exist standalone can't ever
            // apply at runtime; skip them.
            continue;
        };
        resolved.push((l, r, res, rank as u32));
    }
    resolved.sort_by_key(|&(l, r, _, _)| ((l as u64) << 32) | r as u64);

    let lookup_literal = |s: &str| -> u32 {
        *raw_by_literal.get(s).unwrap_or_else(|| panic!("vocab missing {s}"))
    };
    // Generic special slots (see nr-token::blob): for chatml, <|im_start|>
    // fills start_header and <|im_end|> fills eot; end_header is unused.
    let (template_id, eot, start_header, end_header, end_of_text) = match family {
        Family::Llama3 => (
            1u32,
            lookup_literal("<|eot_id|>"),
            lookup_literal("<|start_header_id|>"),
            lookup_literal("<|end_header_id|>"),
            lookup_literal("<|end_of_text|>"),
        ),
        Family::Qwen3 => (
            2u32,
            lookup_literal("<|im_end|>"),
            lookup_literal("<|im_start|>"),
            lookup_literal("<|im_end|>"),
            lookup_literal("<|endoftext|>"),
        ),
    };

    // Serialize.
    let mut pool: Vec<u8> = Vec::new();
    let mut table: Vec<u8> = Vec::with_capacity(decoded.len() * 8);
    for (bytes, &flag) in decoded.iter().zip(flags.iter()) {
        assert!(bytes.len() <= u16::MAX as usize);
        table.extend_from_slice(&(pool.len() as u32).to_le_bytes());
        table.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        table.extend_from_slice(&flag.to_le_bytes());
        pool.extend_from_slice(bytes);
    }

    let mut blob = Vec::new();
    blob.extend_from_slice(&nr_token::blob::MAGIC);
    blob.extend_from_slice(&nr_token::blob::VERSION.to_le_bytes());
    blob.extend_from_slice(&template_id.to_le_bytes());
    blob.extend_from_slice(&(decoded.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(resolved.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(pool.len() as u32).to_le_bytes());
    for id in [bos, eos, eot, start_header, end_header, end_of_text] {
        blob.extend_from_slice(&id.to_le_bytes());
    }
    for id in byte_ids {
        blob.extend_from_slice(&id.to_le_bytes());
    }
    blob.extend_from_slice(&table);
    for (l, r, res, rank) in &resolved {
        blob.extend_from_slice(&l.to_le_bytes());
        blob.extend_from_slice(&r.to_le_bytes());
        blob.extend_from_slice(&res.to_le_bytes());
        blob.extend_from_slice(&rank.to_le_bytes());
    }
    blob.extend_from_slice(&pool);

    BuiltTokenizer { blob, vocab: decoded.len(), merges: resolved.len() }
}
