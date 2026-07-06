//! nr-tensor: quantized CPU tensor kernels for NightRun.
//!
//! Everything here runs identically on bare metal (no_std) and on the host
//! (std, for unit tests and benchmarks). Kernels dispatch to AVX2+FMA when
//! the CPU supports it, with portable scalar fallbacks.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod cpu;
pub mod f16;
pub mod kernels;
pub mod parallel;
pub mod q8;
pub mod rope;
pub mod vec;

pub use f16::{f16_to_f32, f32_to_f16};
pub use q8::{BlockQ8_0, QK8_0};
