//! nr-gfx: no_std framebuffer graphics for NightRun.
//!
//! Colors are `u32` in `0x00RRGGBB` form. A `Surface` is a RAM back buffer
//! that gets presented to the physical framebuffer once per frame.

#![no_std]

extern crate alloc;

pub mod color;
pub mod draw;
pub mod font;
pub mod surface;
pub mod theme;

pub use font::PsfFont;
pub use surface::{PixelLayout, Surface};
