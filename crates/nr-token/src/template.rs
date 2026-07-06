//! Llama-3 instruct chat template:
//!
//! <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
//!   {system}<|eot_id|>
//! <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//! <|start_header_id|>assistant<|end_header_id|>\n\n

use alloc::vec::Vec;

use crate::blob::Tokenizer;

pub const ROLE_SYSTEM: &str = "system";
pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";

impl<'a> Tokenizer<'a> {
    /// `<|start_header_id|>{role}<|end_header_id|>\n\n`
    pub fn encode_header(&self, role: &str, out: &mut Vec<u32>) {
        out.push(self.specials.start_header);
        self.encode_text(role, out);
        out.push(self.specials.end_header);
        self.encode_text("\n\n", out);
    }

    /// One full `{header}{content}<|eot_id|>` message.
    pub fn encode_message(&self, role: &str, content: &str, out: &mut Vec<u32>) {
        self.encode_header(role, out);
        self.encode_text(content, out);
        out.push(self.specials.eot);
    }

    /// Start of a conversation: bos + optional system message.
    pub fn encode_conversation_start(&self, system: Option<&str>, out: &mut Vec<u32>) {
        out.push(self.specials.bos);
        if let Some(sys) = system {
            self.encode_message(ROLE_SYSTEM, sys, out);
        }
    }
}
