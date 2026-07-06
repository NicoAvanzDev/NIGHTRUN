//! Application shell: splash -> boot sequence (real model load) -> chat
//! with local Llama inference.

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

use nr_model::{InferCtx, Model, Sampler};
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
const CTX_LEN: usize = 4096;
const MAX_GEN_TOKENS: usize = 768;
const SYSTEM_PROMPT: &str = "You are NightRun, a helpful assistant running Llama 3.2 fully offline on bare-metal x86_64 hardware - no operating system underneath. Be concise and friendly.";

fn stall_us(us: u64) {
    boot::stall(Duration::from_micros(us));
}

pub struct Platform {
    pub display: Display,
    pub fonts: Fonts,
    pub clock: Clock,
    pub arena: Arena,
    pub ram_mb: u32,
    pub model: &'static Model<'static>,
    pub tokenizer: Tokenizer<'static>,
    pub model_name: String,
    pub infer: InferCtx<'static>,
    pub sampler: Sampler,
    pub cores: u32,
    pub pp_milli: u32,
}

pub fn run(display: Display) {
    let fonts = Fonts::load();
    let mut surf = nr_gfx::Surface::new(display.width, display.height);

    let footer = alloc::format!("v{}  //  press any key", crate::VERSION);
    nr_ui::splash::draw(&mut surf, &fonts, &footer);
    display.present(&surf);
    serial_println!("[app] splash");
    for _ in 0..350 {
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
    "preparing inference engine",
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

    // Stage 0: runtime init (SIMD + multi-core bring-up).
    ui.show(0, 300, "TSC clock calibrated");
    let simd = if nr_tensor_fast() { "AVX2+FMA kernels" } else { "scalar kernels (no AVX2)" };
    ui.show(0, 600, simd);
    let workers = crate::smp::start_workers();
    ui.show(0, 1000, &alloc::format!("{} cores online ({} inference workers)", workers + 1, workers));
    stall_us(150_000);

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

    // Stage 4: arena + inference context (KV cache, scratch buffers).
    let need = InferCtx::required_bytes(&model, CTX_LEN);
    assert!(need < ARENA_MB * 1024 * 1024, "arena too small for ctx");
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
        ui.show(4, ((off + len) * 500 / (pages * 4096)) as u32, &alloc::format!("{ARENA_MB} MB resident arena"));
    }

    // The model must outlive the InferCtx that borrows it; both live for
    // the whole session.
    let model: &'static Model<'static> = alloc::boxed::Box::leak(alloc::boxed::Box::new(model));
    let mut alloc_cb = |bytes: usize, align: usize| -> *mut u8 {
        arena
            .alloc_bytes(bytes, align)
            .map(|s| s.as_mut_ptr())
            .unwrap_or(core::ptr::null_mut())
    };
    ui.show(4, 750, &alloc::format!("KV cache + scratch ({} MB, ctx {})", need / (1024 * 1024), CTX_LEN));
    let infer = match InferCtx::new(model, CTX_LEN, &mut alloc_cb) {
        Ok(i) => i,
        Err(e) => ui.fail(&alloc::format!("inference init failed: {e:?}")),
    };
    let sampler = Sampler::new(0.7, 0.9, nr_runtime::clock::rdtsc());
    ui.show(4, 1000, "inference engine ready");

    // Stage 5: done.
    let boot_ms = clock.ticks_to_ms(clock.now() - t_boot);
    serial_println!("[boot] chat-ready in {} ms (after splash)", boot_ms);
    ui.show(5, 1000, &alloc::format!("boot sequence {}.{}s", boot_ms / 1000, boot_ms % 1000 / 100));
    stall_us(400_000);

    // The converter writes the full display name incl. quant label.
    let model_name = alloc::format!("{}", model.meta.name_str());
    Platform {
        display,
        fonts,
        clock,
        arena,
        ram_mb,
        model,
        tokenizer,
        model_name,
        infer,
        sampler,
        cores: workers as u32 + 1,
        pp_milli: 0,
    }
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
            "{} resident in RAM - fully local inference, no OS underneath. Type a prompt; ESC stops a generation; UP/DOWN scroll history.",
            p.model_name,
        ),
    });
    let mut inputline = String::new();
    let mut frame = 0u32;
    let mut last_rate = 0u32;
    let mut conversation_started = false;
    let mut scroll: usize = 0;
    let page = nr_ui::chat::page_lines(surf, &p.fonts);

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
                        scroll = 0; // jump back to live view for the reply
                        last_rate = generate(p, surf, &prompt, &mut turns, &mut conversation_started, frame);
                    } else {
                        inputline.clear();
                    }
                }
                InputEvent::Up => scroll = scroll.saturating_add(3),
                InputEvent::Down => scroll = scroll.saturating_sub(3),
                InputEvent::PageUp => scroll = scroll.saturating_add(page),
                InputEvent::PageDown => scroll = scroll.saturating_sub(page),
                _ => {}
            }
        }

        if dirty || frame % 8 == 0 {
            scroll = draw_chat(p, surf, &turns, &inputline, frame, last_rate, false, scroll);
        }
        frame = frame.wrapping_add(1);
        stall_us(16_000);
    }
}

/// Run one user turn through the model, streaming tokens to the screen.
/// Returns milli-tokens/sec over the generation phase.
fn generate(
    p: &mut Platform,
    surf: &mut nr_gfx::Surface,
    prompt: &str,
    turns: &mut Vec<Turn>,
    conversation_started: &mut bool,
    mut frame: u32,
) -> u32 {
    // Out of context? Start a fresh conversation.
    if p.infer.remaining() < MAX_GEN_TOKENS + 256 {
        p.infer.reset();
        *conversation_started = false;
        turns.push(Turn {
            role: Role::System,
            text: String::from("context window full - conversation reset"),
        });
    }

    // Build this turn's token sequence (Llama-3 instruct template).
    let mut ids: Vec<u32> = Vec::new();
    if !*conversation_started {
        p.tokenizer.encode_conversation_start(Some(SYSTEM_PROMPT), &mut ids);
        *conversation_started = true;
    }
    p.tokenizer.encode_message(nr_token::template::ROLE_USER, prompt, &mut ids);
    p.tokenizer.encode_header(nr_token::template::ROLE_ASSISTANT, &mut ids);

    turns.push(Turn { role: Role::Llama, text: String::new() });

    // Prefill. Redraw between tokens so the UI shows life.
    let t0 = p.clock.now();
    let mut logits_ready = false;
    for (i, &id) in ids.iter().enumerate() {
        p.infer.forward(id);
        logits_ready = true;
        if i % 4 == 0 {
            draw_chat(p, surf, turns, "", frame, 0, true, 0);
            frame = frame.wrapping_add(1);
        }
        if p.infer.remaining() == 0 {
            break;
        }
    }
    let prefill_ms = p.clock.ticks_to_ms(p.clock.now() - t0);
    p.pp_milli = (ids.len() as u64 * 1_000_000 / prefill_ms.max(1)) as u32;
    serial_println!(
        "[gen] prefill {} tokens in {} ms ({} tok/s)",
        ids.len(),
        prefill_ms,
        ids.len() as u64 * 1000 / prefill_ms.max(1)
    );
    let _ = logits_ready;

    // Generation loop.
    let t0 = p.clock.now();
    let mut produced = 0u64;
    let mut rate = 0u32;
    let mut utf8_pending: Vec<u8> = Vec::new();
    // The logits of the last prefilled token seed the first sample; re-run
    // sample/forward until a stop token, budget, or ESC.
    let mut next = {
        let logits = p.infer.logits();
        p.sampler.sample(logits)
    };
    while !p.tokenizer.is_stop(next) && produced < MAX_GEN_TOKENS as u64 && p.infer.remaining() > 0 {
        utf8_pending.extend_from_slice(p.tokenizer.token_bytes(next));
        flush_utf8(&mut utf8_pending, &mut turns.last_mut().unwrap().text);

        produced += 1;
        rate = p.clock.rate_milli(produced, p.clock.now() - t0);
        draw_chat(p, surf, turns, "", frame, rate, true, 0);
        frame = frame.wrapping_add(1);

        if matches!(input::poll(), Some(InputEvent::Escape)) {
            serial_println!("[gen] interrupted by user");
            break;
        }

        let logits = p.infer.forward(next);
        next = p.sampler.sample(logits);
    }
    // Keep the template consistent for the next turn.
    if p.tokenizer.is_stop(next) && p.infer.remaining() > 0 {
        p.infer.forward(p.tokenizer.specials.eot);
    }

    let gen_ms = p.clock.ticks_to_ms(p.clock.now() - t0);
    serial_println!(
        "[gen] {} tokens in {} ms ({} milli-tok/s)",
        produced,
        gen_ms,
        rate
    );
    if turns.last().map(|t| t.text.is_empty()).unwrap_or(false) {
        turns.last_mut().unwrap().text = String::from("(no output)");
    }
    rate
}

/// Move complete UTF-8 prefixes of `pending` into `out` (token boundaries
/// can split multi-byte characters).
fn flush_utf8(pending: &mut Vec<u8>, out: &mut String) {
    match core::str::from_utf8(pending) {
        Ok(s) => {
            out.push_str(s);
            pending.clear();
        }
        Err(e) => {
            let ok = e.valid_up_to();
            if ok > 0 {
                out.push_str(core::str::from_utf8(&pending[..ok]).unwrap());
                pending.drain(..ok);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_chat(
    p: &Platform,
    surf: &mut nr_gfx::Surface,
    turns: &[Turn],
    input: &str,
    frame: u32,
    rate_milli: u32,
    generating: bool,
    scroll: usize,
) -> usize {
    let stats = Stats {
        model: &p.model_name,
        mem_used_mb: ((p.model.total_size() + p.arena.used()) / (1024 * 1024)) as u32,
        mem_total_mb: p.ram_mb,
        tok_s_milli: rate_milli,
        pp_milli: p.pp_milli,
        ctx_used: p.infer.pos as u32,
        ctx_max: p.infer.dims.ctx as u32,
        cores: p.cores,
        generating,
    };
    let cursor_on = !generating && (frame / 16) % 2 == 0;
    let scroll = nr_ui::chat::draw(surf, &p.fonts, turns, input, cursor_on, &stats, scroll);
    p.display.present(surf);
    scroll
}

