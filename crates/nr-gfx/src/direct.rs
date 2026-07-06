//! Direct framebuffer drawing that bypasses the back buffer and heap —
//! used by the panic handler, where allocation may not be trustworthy.

use crate::font::PsfFont;
use crate::surface::PixelLayout;

#[derive(Clone, Copy)]
pub struct DirectFb {
    pub ptr: *mut u32,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub layout: PixelLayout,
}

unsafe impl Send for DirectFb {}
unsafe impl Sync for DirectFb {}

impl DirectFb {
    #[inline]
    fn encode(&self, color: u32) -> u32 {
        match self.layout {
            PixelLayout::Bgrx => color,
            PixelLayout::Rgbx => {
                (color & 0x0000ff00) | ((color & 0x00ff0000) >> 16) | ((color & 0x000000ff) << 16)
            }
        }
    }

    pub fn fill_rect(&self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        let c = self.encode(color);
        for row in y..(y + h).min(self.height) {
            for col in x..(x + w).min(self.width) {
                // SAFETY: bounds clamped to the mode's stride/height.
                unsafe { self.ptr.add(row * self.stride + col).write_volatile(c) };
            }
        }
    }

    pub fn text(&self, font: &PsfFont, x: usize, y: usize, s: &str, color: u32) {
        let c = self.encode(color);
        let mut pen = x;
        for ch in s.chars() {
            let Some(glyph) = font.glyph(ch).or_else(|| font.glyph('?')) else {
                pen += font.width;
                continue;
            };
            for gy in 0..font.height {
                for gx in 0..font.width {
                    if font.pixel(glyph, gx, gy) {
                        let (px, py) = (pen + gx, y + gy);
                        if px < self.width && py < self.height {
                            unsafe { self.ptr.add(py * self.stride + px).write_volatile(c) };
                        }
                    }
                }
            }
            pen += font.width;
        }
    }
}
