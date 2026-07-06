//! M2 application shell: splash -> boot sequence -> chat loop.
//!
//! The chat responses are an explicitly-labelled shell demo until the
//! inference engine lands (Milestone 5).

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use nr_runtime::{Arena, Clock};
use nr_ui::chat::{Role, Stats, Turn};
use nr_ui::loading::LoadState;
use nr_ui::Fonts;
use uefi::boot;
use uefi::boot::{AllocateType, MemoryType};

use crate::input::{self, InputEvent};
use crate::video::Display;

const ARENA_MB: usize = 192;
const MODEL_NAME: &str = "SHELL DEMO (inference in M5)";

fn stall_us(us: u64) {
    boot::stall(Duration::from_micros(us));
}

pub struct Platform {
    pub display: Display,
    pub fonts: Fonts,
    pub clock: Clock,
    pub arena: Arena,
    pub ram_mb: u32,
}

pub fn run(display: Display) {
    let fonts = Fonts::load();
    let mut surf = nr_gfx::Surface::new(display.width, display.height);

    // Splash: shown until a key is pressed or ~2.5 s passes.
    let footer = alloc::format!("v{}  //  press any key", crate::VERSION);
    nr_ui::splash::draw(&mut surf, &fonts, &footer);
    display.present(&surf);
    serial_println!("[app] splash");
    for _ in 0..250 {
        if input::poll().is_some() {
            break;
        }
        stall_us(10_000);
    }

    let clock = Clock::calibrate(stall_us);
    let mut platform = boot_sequence(display, fonts, clock, &mut surf);
    serial_println!(
        "[app] boot sequence done: ram={} MB arena={} MB",
        platform.ram_mb,
        platform.arena.capacity() / (1024 * 1024)
    );

    chat_loop(&mut platform, &mut surf);
}

/// The staged boot/loading sequence. Memory scan and arena allocation are
/// real; the "model" stage is a placeholder pending M4/M5.
fn boot_sequence(display: Display, fonts: Fonts, clock: Clock, surf: &mut nr_gfx::Surface) -> Platform {
    let stages = [
        "initializing runtime",
        "scanning memory",
        "allocating resident arena",
        "preparing model stage (placeholder)",
        "starting chat interface",
    ];
    let mut frame = 0u32;
    let mut show = |current: usize, pm: u32, detail: &str, frame: &mut u32| {
        let st = LoadState { stages: &stages, current, progress_pm: pm, detail, frame: *frame };
        nr_ui::loading::draw(surf, &fonts, &st);
        display.present(surf);
        *frame += 1;
    };

    // Stage 0: runtime init (clock already calibrated).
    for pm in [200, 600, 1000] {
        show(0, pm, "calibrating TSC clock", &mut frame);
        stall_us(90_000);
    }

    // Stage 1: memory scan (real memory map walk).
    show(1, 300, "reading UEFI memory map", &mut frame);
    let ram_mb = conventional_ram_mb();
    let detail = alloc::format!("{} MB conventional RAM", ram_mb);
    show(1, 1000, &detail, &mut frame);
    stall_us(250_000);

    // Stage 2: arena allocation (real pages, touched).
    let pages = ARENA_MB * 1024 * 1024 / 4096;
    let base = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
        .expect("arena pages");
    // SAFETY: freshly allocated, exclusively owned.
    let mut arena = unsafe { Arena::new(base.as_ptr(), pages * 4096) };
    let chunk = 16 * 1024 * 1024;
    for (i, off) in (0..ARENA_MB * 1024 * 1024).step_by(chunk).enumerate() {
        // Touch the memory for real.
        unsafe { core::ptr::write_bytes(base.as_ptr().add(off), 0, chunk) };
        let pm = ((off + chunk) as u64 * 1000 / (ARENA_MB * 1024 * 1024) as u64) as u32;
        let detail = alloc::format!("{} / {} MB zeroed", (i + 1) * 16, ARENA_MB);
        show(2, pm, &detail, &mut frame);
    }
    let _ = arena.alloc_bytes(64, 64); // reserve a guard so `used` is non-zero

    // Stage 3: model placeholder.
    for pm in (0..=1000).step_by(125) {
        show(3, pm, "model loading arrives in milestone 4", &mut frame);
        stall_us(60_000);
    }

    // Stage 4: chat.
    show(4, 1000, "ready", &mut frame);
    stall_us(300_000);

    Platform { display, fonts, clock, arena, ram_mb }
}

fn conventional_ram_mb() -> u32 {
    use uefi::mem::memory_map::MemoryMap as _;
    match boot::memory_map(MemoryType::LOADER_DATA) {
        Ok(map) => {
            let pages: u64 = map
                .entries()
                .filter(|d| d.ty == MemoryType::CONVENTIONAL)
                .map(|d| d.page_count)
                .sum();
            (pages * 4096 / (1024 * 1024)) as u32
        }
        Err(_) => 0,
    }
}

fn chat_loop(p: &mut Platform, surf: &mut nr_gfx::Surface) {
    let mut turns: Vec<Turn> = Vec::new();
    turns.push(Turn {
        role: Role::System,
        text: String::from(
            "NightRun shell online. This build is the M2 runtime shell - responses below are canned demo text, not model output.",
        ),
    });
    let mut inputline = String::new();
    let mut frame = 0u32;
    let mut last_rate = 0u32;

    loop {
        let mut dirty = false;
        while let Some(ev) = input::poll() {
            dirty = true;
            match ev {
                InputEvent::Char(c) => inputline.push(c),
                InputEvent::Backspace => {
                    inputline.pop();
                }
                InputEvent::Enter => {
                    if !inputline.trim().is_empty() {
                        let prompt = core::mem::take(&mut inputline);
                        serial_println!("[chat] user: {}", prompt);
                        turns.push(Turn { role: Role::User, text: prompt });
                        last_rate = mock_generate(p, surf, &mut turns, frame);
                    } else {
                        inputline.clear();
                    }
                }
                _ => {}
            }
        }

        // Redraw at ~30 Hz for cursor blink; cheap enough and keeps the
        // screen owned by us.
        if dirty || frame % 8 == 0 {
            draw_chat(p, surf, &turns, &inputline, frame, last_rate, false);
        }
        frame = frame.wrapping_add(1);
        stall_us(16_000);
    }
}

fn draw_chat(
    p: &Platform,
    surf: &mut nr_gfx::Surface,
    turns: &[Turn],
    input: &str,
    frame: u32,
    rate_milli: u32,
    generating: bool,
) {
    let stats = Stats {
        model: MODEL_NAME,
        mem_used_mb: (p.arena.used() / (1024 * 1024)) as u32,
        mem_total_mb: p.ram_mb,
        tok_s_milli: rate_milli,
        generating,
    };
    let cursor_on = !generating && (frame / 16) % 2 == 0;
    nr_ui::chat::draw(surf, &p.fonts, turns, input, cursor_on, &stats);
    p.display.present(surf);
}

/// Stream a canned response word-by-word, measuring a real "rate" for the
/// status bar plumbing. Returns milli-tok/s.
fn mock_generate(p: &mut Platform, surf: &mut nr_gfx::Surface, turns: &mut Vec<Turn>, mut frame: u32) -> u32 {
    const REPLY: &str = "I am the NightRun runtime shell. The full Llama 3.2 inference engine \
        docks here in milestone 5; right now I demonstrate the boot path, framebuffer UI, \
        keyboard input, and streaming display you are looking at. Everything on screen is \
        rendered by our own bare-metal code - no operating system underneath.";

    turns.push(Turn { role: Role::Llama, text: String::new() });
    let t0 = p.clock.now();
    let mut words = 0u64;
    let word_count = REPLY.split(' ').count();
    for (i, word) in REPLY.split(' ').enumerate() {
        {
            let last = turns.last_mut().unwrap();
            if !last.text.is_empty() {
                last.text.push(' ');
            }
            last.text.push_str(word);
        }
        words += 1;
        let rate = p.clock.rate_milli(words, p.clock.now() - t0);
        let generating = i + 1 < word_count;
        draw_chat(p, surf, turns, "", frame, rate, generating);
        frame = frame.wrapping_add(1);
        stall_us(65_000);
    }
    let rate = p.clock.rate_milli(words, p.clock.now() - t0);
    serial_println!("[chat] llama reply streamed, {} words, {} milli-wps", words, rate);
    rate
}
