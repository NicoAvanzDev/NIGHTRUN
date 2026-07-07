//! nr-model: the NightRun model format (.nrm) and llama inference.
//!
//! The .nrm format is produced by tools/nrconvert from a GGUF file and is
//! designed for bare-metal loading: fixed little-endian header, flat tensor
//! table, 64-byte-aligned tensor data viewed in place (zero copies).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod crc32;
pub mod format;
pub mod infer;
pub mod sample;
pub mod verify;

pub use format::{Meta, Model, TensorDtype, TensorKind, TensorView};
pub use infer::InferCtx;
pub use sample::Sampler;
