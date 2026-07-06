//! Chat templates.
//!
//! Llama-3 instruct:
//!   <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
//!     {system}<|eot_id|>
//!   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//!   <|start_header_id|>assistant<|end_header_id|>\n\n
//!
//! ChatML (Qwen3, no BOS):
//!   <|im_start|>system\n{system}<|im_end|>\n
//!   <|im_start|>user\n{user}<|im_end|>\n
//!   <|im_start|>assistant\n
//!
//! In the blob's generic special slots, chatml maps <|im_start|> to
//! `start_header` and <|im_end|> to `eot`/`eos`.

use alloc::vec::Vec;

use crate::blob::{Template, Tokenizer};

pub const ROLE_SYSTEM: &str = "system";
pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";

impl<'a> Tokenizer<'a> {
    /// The generation header for `role` (everything up to where the model
    /// starts producing content).
    pub fn encode_header(&self, role: &str, out: &mut Vec<u32>) {
        match self.template {
            Template::Llama3 => {
                out.push(self.specials.start_header);
                self.encode_text(role, out);
                out.push(self.specials.end_header);
                self.encode_text("\n\n", out);
            }
            Template::ChatMl => {
                out.push(self.specials.start_header); // <|im_start|>
                self.encode_text(role, out);
                self.encode_text("\n", out);
            }
        }
    }

    /// One full message: header + content + end-of-turn.
    pub fn encode_message(&self, role: &str, content: &str, out: &mut Vec<u32>) {
        self.encode_header(role, out);
        self.encode_text(content, out);
        out.push(self.specials.eot);
        if self.template == Template::ChatMl {
            self.encode_text("\n", out);
        }
    }

    /// Start of a conversation: BOS where the template uses one, plus the
    /// optional system message.
    pub fn encode_conversation_start(&self, system: Option<&str>, out: &mut Vec<u32>) {
        if self.template == Template::Llama3 {
            out.push(self.specials.bos);
        }
        if let Some(sys) = system {
            self.encode_message(ROLE_SYSTEM, sys, out);
        }
    }
}
