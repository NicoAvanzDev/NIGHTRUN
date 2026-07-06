//! nr-runtime: core runtime services (arena memory, timing).

#![no_std]

pub mod arena;
pub mod clock;

pub use arena::Arena;
pub use clock::Clock;
