//! Application shell: splash -> boot sequence (real model load) -> chat.
//!
//! M4 status: the model is loaded, checksummed and parsed, and the real
//! tokenizer runs; chat replies demonstrate tokenization until the
//! inference engine lands (M5).

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use nr_model::Model;
use nr_runtime::{Arena, Clock};
use nr_token::Tokenizer;
use nr_ui::chat::{Role, Stats, Turn};
use nr_ui::loading::LoadState;
use nr_ui::Fonts;
use uefi::boot;
use uefi::boot::{AllocateType, MemoryType};

use crate::input::{self, InputEvent};
use crate::video::Display;

const ARENA_MB: usize = 256;

fn stall_us(us: u64) {
    boot::stall(Duration::from_micros(us));
}

pub struct Platform {
    pub display: Display,
    pub fonts: Fonts,
    pub clock: Clock,
    pub arena: Arena,
    pub ram_mb: u32,
    pub model: Model<'static>,
    pub tokenizer: Tokenizer<'static>,
    pub model_name: String,
}

pub fn run(display: Display) {
    let fonts = Fonts::load();
    let mut surf = nr_gfx::Surface::new(display.width, display.height);

    let footer = alloc::format!("v{}  //  press any key", crate::VERSION);
    nr_ui::splash::draw(&mut surf, &fonts, &footer);
    display.present(&surf);
    serial_println!("[app] splash");
    for _ in 0..200 {
        if input::poll().is_some() {
            break;
        }
        stall_us(10_000);
    }

    let clock = Clock::calibrate(stall_us);
    let mut platform = boot_sequence(display, fonts, clock, &mut surf);
    serial_println!(
        "[app] ready: model '{}' {} MB, ram {} MB",
        platform.model_name,
        platform.model.total_size() / (1024 * 1024),
        platform.ram_mb
    );

    chat_loop(&mut platform, &mut surf);
}

const STAGES: &[&str] = &[
    "initializing runtime",
    "scanning memory",
    "loading model into RAM",
    "verifying checksums",
    "allocating resident arena",
    "starting chat interface",
];

struct BootUi<'a> {
    display: &'a Display,
    surf: &'a mut nr_gfx::Surface,
    fonts: &'a Fonts,
    frame: u32,
}

impl BootUi<'_> {
    fn show(&mut self, current: usize, pm: u32, detail: &str) {
        let st = LoadState { stages: STAGES, current, progress_pm: pm, detail, frame: self.frame };
        nr_ui::loading::draw(self.surf, self.fonts, &st);
        self.display.present(self.surf);
        self.frame += 1;
    }

    fn fail(&mut self, message: &str) -> ! {
        serial_println!("[boot] FATAL: {}", message);
        let st = LoadState { stages: STAGES, current: usize::MAX, progress_pm: 0, detail: message, frame: self.frame };
        nr_ui::loading::draw(self.surf, self.fonts, &st);
        self.display.present(self.surf);
        loop {
            unsafe { core::arch::asm!("hlt") };
        }
    }
}

fn boot_sequence(display: Display, fonts: Fonts, clock: Clock, surf: &mut nr_gfx::Surface) -> Platform {
    let t_boot = clock.now();
    let mut ui = BootUi { display: &display, surf, fonts: &fonts, frame: 0 };

    // Stage 0: runtime init.
    ui.show(0, 500, "TSC clock calibrated");
    let simd = if nr_tensor_fast() { "AVX2+FMA kernels" } else { "scalar kernels (no AVX2)" };
    ui.show(0, 1000, simd);
    stall_us(120_000);

    // Stage 1: memory scan.
    let ram_mb = conventional_ram_mb();
    ui.show(1, 1000, &alloc::format!("{ram_mb} MB conventional RAM"));
    stall_us(120_000);

    // Stage 2: load model.nrm into RAM (chunked reads off the boot volume).
    let t0 = clock.now();
    let blob: &'static [u8] = {
        let ui = &mut ui;
        let clock = &clock;
        let mut cb = |done: usize, total: usize| {
            let pm = (done as u64 * 1000 / total.max(1) as u64) as u32;
            let mbs = {
                let ms = clock.ticks_to_ms(clock.now() - t0).max(1);
                done as u64 * 1000 / ms / (1024 * 1024)
            };
            ui.show(2, pm, &alloc::format!(
                "{} / {} MB  ({} MB/s)",
                done / (1024 * 1024),
                total / (1024 * 1024),
                mbs
            ));
        };
        match crate::modelload::load(&mut cb) {
            Ok(buf) => buf,
            Err(e) => ui.fail(&alloc::format!(
                "model.nrm load failed ({e:?}) - build the image with: cargo xtask image"
            )),
        }
    };
    let load_ms = clock.ticks_to_ms(clock.now() - t0);
    serial_println!("[boot] model loaded: {} bytes in {} ms", blob.len(), load_ms);

    // Stage 3: parse + verify.
    let model = match Model::parse(blob) {
        Ok(m) => m,
        Err(e) => ui.fail(&alloc::format!("model.nrm invalid: {e:?}")),
    };
    let t0 = clock.now();
    let ok = model.verify_data(|done, total| {
        let pm = (done as u64 * 1000 / total.max(1) as u64) as u32;
        ui.show(3, pm, &alloc::format!("CRC32 {} / {} MB", done / (1024 * 1024), total / (1024 * 1024)));
    });
    if !ok {
        ui.fail("tensor data checksum mismatch - rebuild model.nrm");
    }
    let verify_ms = clock.ticks_to_ms(clock.now() - t0);
    let tokenizer = match Tokenizer::parse(model.tokenizer_blob) {
        Ok(t) => t,
        Err(e) => ui.fail(&alloc::format!("tokenizer blob invalid: {e:?}")),
    };
    serial_println!(
        "[boot] verified in {} ms; vocab={} layers={}",
        verify_ms,
        tokenizer.vocab_len(),
        model.meta.n_layers
    );

    // Stage 4: arena.
    let pages = ARENA_MB * 1024 * 1024 / 4096;
    let base = match boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages) {
        Ok(b) => b,
        Err(_) => ui.fail("arena allocation failed - machine needs more RAM"),
    };
    // SAFETY: freshly allocated, exclusively owned.
    let mut arena = unsafe { Arena::new(base.as_ptr(), pages * 4096) };
    for off in (0..pages * 4096).step_by(16 * 1024 * 1024) {
        let len = (16 * 1024 * 1024).min(pages * 4096 - off);
        unsafe { core::ptr::write_bytes(base.as_ptr().add(off), 0, len) };
        ui.show(4, ((off + len) * 1000 / (pages * 4096)) as u32, &alloc::format!("{ARENA_MB} MB resident arena"));
    }
    let _ = arena.alloc_bytes(64, 64);

    // Stage 5: done.
    let boot_ms = clock.ticks_to_ms(clock.now() - t_boot);
    ui.show(5, 1000, &alloc::format!("boot sequence {}.{}s", boot_ms / 1000, boot_ms % 1000 / 100));
    stall_us(400_000);

    let model_name = alloc::format!("{} Q8_0", model.meta.name_str());
    Platform { display, fonts, clock, arena, ram_mb, model, tokenizer, model_name }
}

fn nr_tensor_fast() -> bool {
    nr_tensor::cpu::fast_path()
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
        text: alloc::format!(
            "{} loaded and verified in RAM. Tokenizer online ({} tokens). Generation lands in M5 - replies below show real tokenizer output.",
            p.model_name,
            p.tokenizer.vocab_len(),
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
                        turns.push(Turn { role: Role::User, text: prompt.clone() });
                        last_rate = tokenize_demo(p, &prompt, &mut turns);
                    } else {
                        inputline.clear();
                    }
                }
                _ => {}
            }
        }

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
        model: &p.model_name,
        mem_used_mb: ((p.model.total_size() + p.arena.used()) / (1024 * 1024)) as u32,
        mem_total_mb: p.ram_mb,
        tok_s_milli: rate_milli,
        generating,
    };
    let cursor_on = !generating && (frame / 16) % 2 == 0;
    nr_ui::chat::draw(surf, &p.fonts, turns, input, cursor_on, &stats);
    p.display.present(surf);
}

/// M4 placeholder reply: run the real tokenizer over the prompt and report
/// what the inference engine will see. Returns milli-tokens/sec (encode).
fn tokenize_demo(p: &mut Platform, prompt: &str, turns: &mut Vec<Turn>) -> u32 {
    let t0 = p.clock.now();
    let mut ids: Vec<u32> = Vec::new();
    p.tokenizer.encode_text(prompt, &mut ids);
    let dt = p.clock.now() - t0;
    let rate = p.clock.rate_milli(ids.len() as u64, dt);

    let mut text = alloc::format!("[tokenizer] {} tokens: ", ids.len());
    for (i, id) in ids.iter().take(24).enumerate() {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(&alloc::format!("{id}"));
    }
    if ids.len() > 24 {
        text.push_str(" ...");
    }
    text.push_str(" | decoded: ");
    for &id in ids.iter().take(24) {
        if let Ok(s) = core::str::from_utf8(p.tokenizer.token_bytes(id)) {
            text.push_str(&alloc::format!("[{s}]"));
        }
    }
    text.push_str(" | the inference engine plugs in here in M5.");
    turns.push(Turn { role: Role::Llama, text });
    serial_println!("[chat] tokenized {} tokens", ids.len());
    rate
}
