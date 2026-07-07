//! RAM back buffer and presentation to the physical framebuffer.

use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelLayout {
    /// Byte order B,G,R,X — a 0x00RRGGBB u32 can be stored directly (LE).
    Bgrx,
    /// Byte order R,G,B,X — channels must be swapped on present.
    Rgbx,
}

pub struct Surface {
    pub width: usize,
    pub height: usize,
    buf: Vec<u32>,
}

impl Surface {
    pub fn new(width: usize, height: usize) -> Surface {
        Surface {
            width,
            height,
            buf: vec![0; width * height],
        }
    }

    #[inline]
    pub fn rows(&self) -> &[u32] {
        &self.buf
    }

    pub fn clear(&mut self, color: u32) {
        self.buf.fill(color);
    }

    #[inline]
    pub fn put(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.buf[y as usize * self.width + x as usize] = color;
        }
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32) -> u32 {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.buf[y as usize * self.width + x as usize]
        } else {
            0
        }
    }

    /// Source-over blend with alpha 0..=255.
    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, color: u32, alpha: u32) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let idx = y as usize * self.width + x as usize;
        let dst = self.buf[idx];
        let a = alpha.min(255);
        let na = 255 - a;
        let blend_ch = |s: u32, d: u32| -> u32 { (s * a + d * na) / 255 };
        let r = blend_ch((color >> 16) & 0xff, (dst >> 16) & 0xff);
        let g = blend_ch((color >> 8) & 0xff, (dst >> 8) & 0xff);
        let b = blend_ch(color & 0xff, dst & 0xff);
        self.buf[idx] = (r << 16) | (g << 8) | b;
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x1 = ((x + w).max(0) as usize).min(self.width);
        let y1 = ((y + h).max(0) as usize).min(self.height);
        for row in y0..y1 {
            self.buf[row * self.width + x0..row * self.width + x1].fill(color);
        }
    }

    pub fn blend_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32, alpha: u32) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.blend(xx, yy, color, alpha);
            }
        }
    }

    /// Copy the back buffer to a physical framebuffer.
    ///
    /// # Safety
    /// `fb` must point to a mapped framebuffer of at least
    /// `stride * height` u32 pixels that remains valid for the call.
    pub unsafe fn present(&self, fb: *mut u32, stride: usize, layout: PixelLayout) {
        for y in 0..self.height {
            let src = &self.buf[y * self.width..(y + 1) * self.width];
            let dst = fb.add(y * stride);
            match layout {
                PixelLayout::Bgrx => {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), dst, self.width);
                }
                PixelLayout::Rgbx => {
                    for (x, &px) in src.iter().enumerate() {
                        let swapped = (px & 0x0000ff00)
                            | ((px & 0x00ff0000) >> 16)
                            | ((px & 0x000000ff) << 16);
                        dst.add(x).write_volatile(swapped);
                    }
                }
            }
        }
    }
}
