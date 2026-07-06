# NightRun architecture

## The load-bearing decision: UEFI Boot Services stay on

NightRun is one `no_std` Rust EFI binary (`BOOTX64.EFI`). It never calls
`ExitBootServices`. The firmware provides exactly four things at runtime:

| Firmware service | Used for |
|---|---|
| `GRAPHICS_OUTPUT_PROTOCOL` | obtaining the linear framebuffer (once) |
| `SIMPLE_TEXT_INPUT` | keyboard (works with USB keyboards via the firmware's own USB stack) |
| `SIMPLE_FILE_SYSTEM` | reading `model.nrm` off the boot volume (once) |
| `MP_SERVICES` + `AllocatePages`/`stall` | starting cores, memory, timing |

Everything else — rendering, fonts, memory management, tokenization,
tensor math, sampling, the UI — is NightRun's own code.

Why not exit boot services? After `ExitBootServices` a USB keyboard
requires a full XHCI host-controller driver; PS/2 emulation is unreliable
on modern firmware. Staying resident in Boot Services trades a purist
badge for something that actually boots and takes keystrokes on real
hardware. The application processors we start never touch firmware
services (pure compute + atomics), which keeps us within UEFI's rules.

## Boot flow

```
efi_main
├─ COM1 serial up (115200 8N1, port I/O)          [debug channel]
├─ disable firmware watchdog (or it reboots us after 5 min)
├─ enable AVX: CR4.OSXSAVE, XSETBV XCR0 |= x87|SSE|AVX
├─ GOP: pick 1280x720 BGRX (fallback chain), grab framebuffer pointer
├─ install panic screen (heap-free direct framebuffer drawing)
├─ splash (procedural synthwave scene + logo)
├─ boot sequence (loading screen with real progress):
│  ├─ TSC clock calibration against firmware stall
│  ├─ MP services: start all APs into the spin-worker pool (AVX per-core)
│  ├─ memory map scan (conventional RAM tally)
│  ├─ read \model.nrm in 16 MB chunks into AllocatePages memory
│  ├─ parse header, CRC32-verify metadata + 1.25 GB tensor data
│  └─ arena: 256 MB, hosts KV cache + all inference scratch
└─ chat loop (poll keys, template, prefill, sample/stream)
```

## Memory model

- **Model blob**: firmware pages (`LOADER_DATA`), loaded once, resident
  forever; tensors are viewed zero-copy (`&[BlockQ8_0]`/`&[f32]` straight
  into the blob, 64-byte aligned by the converter).
- **Arena** (256 MB bump allocator): KV cache (f16, 2 x 16 layers x 4096
  ctx x 512 = 128 MB), activation/scratch buffers, logits. Allocated during
  boot; **generation performs zero allocations**.
- **Heap** (UEFI pool via the `uefi` crate allocator): UI strings,
  scrollback, tokenizer output. Never touched inside the token loop proper.

## .nrm model format

Produced by `tools/nrconvert` from a GGUF (Q8_0). Little-endian, fixed
172-byte header: magic `NRUN`, version, dims (dim / layers / heads / kv
heads / head_dim / ffn / vocab / ctx), rope theta + Llama-3 scaling
params, flags (tied embeddings), display name, then offsets for the
tokenizer blob, tensor table (32-byte entries: kind, layer, dtype, offset,
size, rows, cols) and the 64-byte-aligned data section. CRC32 over
metadata and data. Parsing on bare metal is header reads + pointer
arithmetic — no GGUF parsing in the runtime.

Tensors stay in GGUF's Q8_0 block layout (32 x i8 + f16 scale = 34 bytes),
norms in f32. The GGUF `rope_freqs` tensor (Llama-3 frequency divisors) is
carried through and preferred at runtime.

## Tokenizer

Byte-level BPE (tiktoken-style, 128k vocab). The converter decodes GGUF's
byte-unicode token strings to raw bytes, resolves merge strings to id
pairs `(left, right) -> (result, rank)` sorted for binary search, and
emits a flat blob (token table + string pool + byte->id table + special
ids). The runtime pretokenizer is a hand-rolled implementation of the
Llama-3 split regex (contractions, letter runs with optional prefix,
1–3-digit numbers, punctuation, whitespace lookahead) using core's Unicode
tables. All 42 reference cases (incl. emoji, CJK, contractions, code)
match the official HF tokenizer exactly; exotic Unicode-category edge
cases may deviate — a documented limitation.

Chat uses the Llama-3 instruct template (`<|start_header_id|>` … 
`<|eot_id|>`) with a system prompt; the UI shows plain `user:` / `llama:`.

## Inference

Per token: embedding row dequant → 16 x [RMSNorm → QKV matvec → RoPE
(adjacent-pair, GGUF-permuted convention) → f16 KV append → GQA attention
(32 q heads / 8 kv heads) → output matvec → residual → RMSNorm → SwiGLU
MLP → residual] → final norm → tied-embedding classifier (128k logits).

- Weights Q8_0, activations quantized to Q8_0 per matvec; integer dot via
  AVX2 `sign/maddubs/madd` (llama.cpp's q8xq8 scheme), FMA accumulate.
- KV cache in f16; attention dot/axpy use F16C (`vcvtph2ps`).
- The builtin `x86_64-unknown-uefi` target is **soft-float** — unusable
  for this (breaks AVX intrinsics, software f32). NightRun builds against
  a custom hard-float target (`x86_64-nightrun-uefi.json`, `+sse,+sse2`)
  with `-Zbuild-std=core,alloc` on nightly.
- **Multi-core**: matvec rows fan out over a spin-worker pool
  (`nr-tensor::parallel`). APs are started once via MP services and never
  return; jobs are posted with atomics only. The same pool code drives std
  threads in `nrhost`.
- Sampling: greedy, or temperature + top-k(64) prefilter + top-p nucleus.

Correctness: AVX2 kernels are tested against scalar; f16 against known
bit patterns; the full forward pass is pinned to **token-for-token greedy
parity with llama.cpp** on the same GGUF (two regression tests).

## Performance notes (measured, QEMU/KVM, 8 cores)

See docs/benchmarks.md for current numbers. Q8_0 matvec is memory-bandwidth
bound: single-core ~12.5 GB/s ≈ 9 tok/s; 8 cores ≈ 22 tok/s (bandwidth
saturates well below core count — 12 threads is no better than 8 on the
dev machine). Prompt processing runs the same single-token path, so
pp ≈ tg; a batched GEMM prefill is the obvious next optimization.

## UI

`nr-gfx` renders into a RAM back buffer presented once per frame (row
`memcpy` for BGRX). Fonts are Spleen PSF1/PSF2 bitmaps (8x16 … 32x64).
The splash/loading backdrop (gradient sky, stars, striped sun, perspective
grid) is fully procedural; text supports shear, per-row gradients and a
blended glow. The panic handler draws a themed fault screen with direct,
heap-free framebuffer writes.

## Limitations

- Context capped at 4096 tokens (KV memory); the conversation auto-resets
  when full. No KV eviction/sliding window.
- Prefill is unbatched (see above).
- Pretokenizer is an approximation outside common Unicode classes.
- Q8_0 only (Q4_K would roughly halve memory and boost tok/s; the format
  has a dtype field reserved for it).
- Requires UEFI; no legacy BIOS path.
- Firmware keyboard repeat/rollover behaviour varies between vendors.
