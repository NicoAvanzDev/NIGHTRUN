//! GOP framebuffer bring-up. Runs while Boot Services are alive, so the
//! framebuffer stays mapped for the whole session.

use nr_gfx::{PixelLayout, Surface};
use uefi::boot;
use uefi::proto::console::gop::{GraphicsOutput, PixelFormat};

pub struct Display {
    pub width: usize,
    pub height: usize,
    stride: usize,
    layout: PixelLayout,
    fb: *mut u32,
}

/// Resolutions we try, in order of preference (fast to blit, widely
/// supported); falls back to whatever mode the firmware is already in.
const PREFERRED: &[(usize, usize)] = &[(1280, 720), (1920, 1080), (1024, 768), (800, 600)];

pub fn init() -> Display {
    let handle = boot::get_handle_for_protocol::<GraphicsOutput>().expect("no GOP handle");
    let mut gop =
        boot::open_protocol_exclusive::<GraphicsOutput>(handle).expect("open GOP");

    let pick = PREFERRED.iter().find_map(|&(w, h)| {
        gop.modes().find(|m| {
            let info = m.info();
            info.resolution() == (w, h)
                && matches!(info.pixel_format(), PixelFormat::Bgr | PixelFormat::Rgb)
        })
    });
    if let Some(mode) = pick {
        gop.set_mode(&mode).expect("set GOP mode");
    }

    let info = gop.current_mode_info();
    let (width, height) = info.resolution();
    let layout = match info.pixel_format() {
        PixelFormat::Rgb => PixelLayout::Rgbx,
        _ => PixelLayout::Bgrx,
    };
    let mut fbuf = gop.frame_buffer();
    let fb = fbuf.as_mut_ptr() as *mut u32;
    crate::serial_println!(
        "[video] mode {}x{} stride={} format={:?}",
        width,
        height,
        info.stride(),
        info.pixel_format()
    );

    let display = Display { width, height, stride: info.stride(), layout, fb };
    // Keep the protocol open (exclusively) for the rest of the session so
    // nothing else redraws the screen.
    core::mem::forget(gop);
    display
}

impl Display {
    pub fn present(&self, surf: &Surface) {
        unsafe { surf.present(self.fb, self.stride, self.layout) }
    }

    pub fn direct(&self) -> nr_gfx::direct::DirectFb {
        nr_gfx::direct::DirectFb {
            ptr: self.fb,
            width: self.width,
            height: self.height,
            stride: self.stride,
            layout: self.layout,
        }
    }
}
