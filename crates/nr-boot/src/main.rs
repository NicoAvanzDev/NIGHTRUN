//! NightRun boot layer: UEFI entry, platform bring-up, panic screen.

#![no_std]
#![no_main]

extern crate alloc;

#[macro_use]
pub mod serial;
mod app;
mod input;
mod modelload;
mod smp;
mod video;

use core::fmt::Write as _;
use core::sync::atomic::{AtomicPtr, Ordering};

use uefi::prelude::*;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[entry]
fn main() -> Status {
    serial::init();
    serial_println!("[nightrun] v{} boot layer up", VERSION);
    uefi::helpers::init().expect("uefi helpers");
    // The firmware watchdog would reset the machine mid-chat; disable it.
    let _ = uefi::boot::set_watchdog_timer(0, 0x1_0000, None);
    enable_avx();

    let display = video::init();
    install_panic_fb(&display);

    app::run(display);
    Status::SUCCESS
}

/// Enable AVX (YMM state) via XCR0 so the AVX2/FMA/F16C kernels can run.
/// UEFI guarantees SSE only; we do the OSXSAVE dance ourselves.
fn enable_avx() {
    if !enable_avx_quiet() {
        serial_println!("[cpu] no AVX - scalar kernels");
        return;
    }
    let f = nr_tensor::cpu::features();
    serial_println!("[cpu] avx enabled; avx2={} fma={} f16c={}", f.avx2, f.fma, f.f16c);
}

/// AVX enable without serial output (also used by AP worker bring-up,
/// where two cores sharing the serial port would interleave garbage).
pub fn enable_avx_quiet() -> bool {
    use core::arch::x86_64::__cpuid;
    let leaf1 = __cpuid(1);
    let xsave = leaf1.ecx & (1 << 26) != 0;
    let avx = leaf1.ecx & (1 << 28) != 0;
    if !(xsave && avx) {
        return false;
    }
    // SAFETY: CPL0 under UEFI; setting CR4.OSXSAVE then XCR0 x87|SSE|AVX.
    unsafe {
        core::arch::asm!(
            "mov rax, cr4",
            "or rax, 1 << 18", // CR4.OSXSAVE
            "mov cr4, rax",
            out("rax") _,
            options(nostack)
        );
        core::arch::asm!(
            "xor ecx, ecx",
            "xgetbv",
            "or eax, 7", // x87 | SSE | AVX state
            "xsetbv",
            out("eax") _, out("ecx") _, out("edx") _,
            options(nostack)
        );
    }
    true
}

// ---- Panic screen ----------------------------------------------------------

static PANIC_FB: AtomicPtr<nr_gfx::direct::DirectFb> = AtomicPtr::new(core::ptr::null_mut());

fn install_panic_fb(display: &video::Display) {
    let fb = alloc::boxed::Box::leak(alloc::boxed::Box::new(display.direct()));
    PANIC_FB.store(fb, Ordering::Release);
}

/// Fixed-size formatting buffer usable during panic (no heap).
struct PanicBuf {
    buf: [u8; 512],
    len: usize,
}

impl core::fmt::Write for PanicBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let n = s.len().min(self.buf.len() - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial_println!("[panic] {}", info);

    let fb_ptr = PANIC_FB.load(Ordering::Acquire);
    if !fb_ptr.is_null() {
        // SAFETY: set once from a leaked box; framebuffer stays mapped.
        let fb = unsafe { &*fb_ptr };
        let mut msg = PanicBuf { buf: [0; 512], len: 0 };
        let _ = write!(msg, "{}", info);
        draw_panic_screen(fb, core::str::from_utf8(&msg.buf[..msg.len]).unwrap_or("panic"));
    }

    loop {
        unsafe { core::arch::asm!("hlt") };
    }
}

fn draw_panic_screen(fb: &nr_gfx::direct::DirectFb, msg: &str) {
    use nr_gfx::theme;
    static SMALL: &[u8] = include_bytes!("../../../assets/fonts/spleen-8x16.psfu");
    let Some(font) = nr_gfx::PsfFont::parse(SMALL) else { return };

    fb.fill_rect(0, 0, fb.width, fb.height, 0x12021c);
    let band_y = fb.height / 4;
    fb.fill_rect(0, band_y, fb.width, 4, theme::NEON_MAGENTA);
    fb.fill_rect(0, band_y + 90, fb.width, 4, theme::NEON_MAGENTA);
    fb.text(&font, 48, band_y + 28, "NIGHTRUN // SYSTEM FAULT", theme::NEON_MAGENTA);
    fb.text(&font, 48, band_y + 56, "the runtime hit an unrecoverable error - power cycle to restart", theme::TEXT_DIM);

    // Wrapped panic message.
    let cols = (fb.width - 96) / font.width;
    let mut y = band_y + 130;
    let bytes = msg.as_bytes();
    let mut i = 0;
    while i < bytes.len() && y < fb.height - 32 {
        let end = (i + cols).min(bytes.len());
        if let Ok(line) = core::str::from_utf8(&bytes[i..end]) {
            fb.text(&font, 48, y, line, theme::TEXT_PRIMARY);
        }
        i = end;
        y += font.height + 4;
    }
}
