# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NightRun is a bare-metal x86_64 LLM appliance: a single `no_std` Rust UEFI
application that boots from USB and runs Llama 3.2 1B Instruct with no OS.
It deliberately **stays in UEFI Boot Services** (for USB keyboard, disk
reads, and MP services) — do not add `ExitBootServices`, and never call
firmware services from AP worker code (`nr-tensor::parallel` workers are
atomics + compute only).

## Commands

```sh
cargo test                    # host test suite (default-members exclude nr-boot)
cargo test -p nr-token        # single crate; -- <name> for a single test
cargo xtask build             # build BOOTX64.EFI
cargo xtask image             # build nightrun.img (needs models/model.nrm)
cargo xtask run [--img] [--window] [--mem 4G] [--smp 8] \
    [--secs N] [--shot t:file.png] [--keys "t:text\n"]   # QEMU + OVMF
cargo xtask bench             # scripted QEMU run -> docs/benchmarks.md
cargo run --release -p nrconvert -- in.gguf models/model.nrm
cargo run --release -p nrhost -- models/model.nrm --prompt "..." [--raw] \
    [--temp 0] [--threads 8]  # same engine on the host (debugging)
```

- `nr-boot` builds **only** via xtask: it needs nightly + `-Zbuild-std` and
  the custom hard-float target `x86_64-nightrun-uefi.json` (the builtin
  UEFI target is soft-float and breaks both AVX intrinsics and f32 perf).
  Never `cargo build --workspace` / `cargo test --workspace` — nr-boot's
  panic handler collides with std on the host target.
- Tests that need the model (`nr-token` fixtures, `nr-model` parity) skip
  silently when `models/model.nrm` is absent; regenerate it with nrconvert
  before trusting a green run.
- QEMU testing of the model path needs `--img --mem 4G` (the model exceeds
  QEMU's virtual-FAT limit, so the ESP-directory dev mode boots without it).

## Architecture (read docs/architecture.md for the full picture)

- Crate layering: `nr-gfx`/`nr-ui` (framebuffer UI) and `nr-tensor`/
  `nr-token`/`nr-model` (engine) are platform-independent; `nr-boot` is the
  only crate that touches `uefi`. Engine crates are dual-target: `no_std`
  for the EFI build, `std` feature (default) for host tests and tools —
  keep new code `no_std`-clean (`libm` for float math, no `std::`).
- `tools/nrhost` runs the *identical* inference code path on Linux; debug
  engine issues there before reaching for QEMU. Serial (COM1) is the debug
  channel in QEMU: `target/serial.log`.
- `.nrm` format lives in `nr-model::format` (parser) and `tools/nrconvert`
  (writer); the tokenizer blob in `nr-token::blob` (parser) and
  `tools/nrconvert/src/tokenizer.rs` (writer). Change them in pairs and
  bump `VERSION`; `nr-token/tests/fixtures.rs` hardcodes header offsets.
- Inference correctness is pinned by greedy token-for-token parity with
  llama.cpp (`crates/nr-model/tests/parity.rs`). If you touch kernels,
  rope, or the forward pass and parity breaks, the code is wrong — not the
  fixture. Tokenizer fixtures (`tests/fixtures/tokenizer_cases.txt`) come
  from `scripts/gen_tokenizer_fixtures.py` + the official HF tokenizer.
- No allocations in the generation loop: model tensors are zero-copy views
  into the loaded blob; KV cache + scratch come from the boot-time arena
  (`InferCtx::required_bytes` sizes it — update it when adding buffers).
- Visual identity is centralized in `nr-gfx::theme`; screens must keep the
  synthwave look (verify with `--shot` screendumps, not by eye-balling
  code). The UI never claims fake capability: labels reflect what the
  build actually does.
