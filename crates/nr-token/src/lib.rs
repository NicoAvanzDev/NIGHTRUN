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

/// Move complete UTF-8 prefixes of `pending` into `out`. Token boundaries
/// can split multi-byte characters; the incomplete tail stays in
/// `pending` until its continuation bytes arrive. Invalid bytes at the
/// front are dropped (one at a time) so a bad token cannot wedge the
/// stream forever.
pub fn flush_utf8(pending: &mut alloc::vec::Vec<u8>, out: &mut alloc::string::String) {
    loop {
        match core::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return;
            }
            Err(e) => {
                let ok = e.valid_up_to();
                if ok > 0 {
                    out.push_str(core::str::from_utf8(&pending[..ok]).unwrap());
                    pending.drain(..ok);
                } else if e.error_len().is_some() {
                    // Invalid leading byte: drop it and keep going.
                    pending.remove(0);
                } else {
                    // Incomplete sequence at the front: wait for more.
                    return;
                }
            }
        }
    }
}
