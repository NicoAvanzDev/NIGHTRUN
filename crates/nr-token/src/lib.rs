//! nr-token: byte-level BPE tokenizer for Llama 3 vocabularies.
//!
//! Parses the tokenizer blob embedded in .nrm files (see `blob` module for
//! the layout, produced by tools/nrconvert) and provides encoding with a
//! hand-rolled approximation of the Llama-3 pretokenizer regex, decoding to
//! raw bytes, and the Llama-3 instruct chat template.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod blob;
pub mod pretok;
pub mod template;

pub use blob::Tokenizer;
