//! NightRun boot layer: UEFI entry, platform bring-up, and (for now) the
//! M1 splash + keyboard echo loop.

#![no_std]
#![no_main]

extern crate alloc;

#[macro_use]
pub mod serial;
mod video;

use alloc::string::String;
use uefi::prelude::*;
use uefi::proto::console::text::Key;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[entry]
fn main() -> Status {
    serial::init();
    serial_println!("[nightrun] v{} boot layer up", VERSION);
    uefi::helpers::init().expect("uefi helpers");

    let display = video::init();
    let mut surf = nr_gfx::Surface::new(display.width, display.height);
    let fonts = nr_ui::Fonts::load();

    let footer = alloc::format!("NIGHTRUN v{}  //  TYPE TO ECHO  //  SERIAL COM1 115200", VERSION);
    nr_ui::splash::draw(&mut surf, &fonts, &footer);
    display.present(&surf);
    serial_println!("[nightrun] splash presented, entering input loop");

    echo_loop(&display, &mut surf, &fonts, &footer);

    Status::SUCCESS
}

/// M1 proof of input: typed characters echo onto the splash footer area.
fn echo_loop(display: &video::Display, surf: &mut nr_gfx::Surface, fonts: &nr_ui::Fonts, footer: &str) {
    let mut line = String::new();
    loop {
        let key = uefi::system::with_stdin(|stdin| stdin.read_key().ok().flatten());
        let Some(key) = key else {
            boot::stall(core::time::Duration::from_millis(10));
            continue;
        };
        match key {
            Key::Printable(c) => {
                let ch = char::from(c);
                match ch {
                    '\u{8}' => {
                        line.pop();
                    }
                    '\r' => line.clear(),
                    c if !c.is_control() => line.push(c),
                    _ => {}
                }
            }
            Key::Special(_) => continue,
        }
        serial_println!("[input] line: {:?}", line);

        // Redraw the echo band above the footer.
        let w = surf.width as i32;
        let h = surf.height as i32;
        let y = h - 64;
        surf.fill_rect(0, y, w, 28, nr_gfx::theme::BG_DEEP);
        let shown = alloc::format!("user: {}_", line);
        nr_gfx::draw::text(surf, &fonts.body, 24, y, &shown, nr_gfx::theme::NEON_CYAN, 1, 0);
        let _ = footer;
        display.present(surf);
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial_println!("[panic] {}", info);
    loop {
        unsafe { core::arch::asm!("hlt") };
    }
}
