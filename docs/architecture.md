# NightRun architecture

NightRun supports two model families as first-class citizens:
**Llama 3.2 1B Instruct (Q8_0)** and **Qwen3-4B-Instruct-2507 (Q4_K_M)**.
One model ships per image (`cargo xtask image --model <file.nrm>`).

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
│  ├─ parse header, CRC32-verify metadata + all tensor data
│  ├─ seal storage (further model reads from disk are a hard fault)
│  └─ arena sized from InferCtx::required_bytes (KV cache + scratch)
└─ chat loop (poll keys, template, prefill, sample/stream)
```

## Memory model

- **Model blob**: firmware pages (`LOADER_DATA`), loaded once, resident
  forever; tensors are viewed zero-copy (`&[BlockQ8_0]`/`&[f32]` straight
  into the blob, 64-byte aligned by the converter).
- **Arena** (bump allocator, sized per model at boot: ~140 MB for Llama
  1B, ~650 MB for Qwen3 4B): f16 KV cache, activation/scratch buffers,
  logits. Allocated during boot; **generation performs zero allocations**.
- **Heap** (UEFI pool via the `uefi` crate allocator): UI strings,
  scrollback, tokenizer output. Never touched inside the token loop proper.

## .nrm model format

Produced by `tools/nrconvert` from a GGUF. Little-endian, fixed
176-byte header: magic `NRUN`, version, dims (dim / layers / heads / kv
heads / head_dim / ffn / vocab / ctx), rope theta + Llama-3 scaling
params, flags (tied embeddings), display name, then offsets for the
tokenizer blob, tensor table (32-byte entries: kind, layer, dtype, offset,
size, rows, cols) and the 64-byte-aligned data section. CRC32 over
metadata and data. Parsing on bare metal is header reads + pointer
arithmetic — no GGUF parsing in the runtime.

Tensors stay in their GGUF block layouts: Q8_0 (32 x i8 + f16 scale =
34 B), Q4_K (256-value super-blocks, packed 6-bit scale/min pairs,
144 B) and Q6_K (4+2-bit planes, 16 signed scales, 210 B); norms in f32.
The header carries an `arch` field (llama3 / qwen3) and the parser
validates every tensor's byte size against its dtype's block math.

The audited dtype policy of the Qwen3-4B Q4_K_M artifact: everything
Q4_K except `attn_v` (18/36 layers), `ffn_down` (18/36 layers) and
`token_embd` in Q6_K; norms and the per-head Q/K norms F32. The model is
**tied** (no `output.weight`): the Q6_K embedding matrix doubles as the
classifier — validated by a dedicated test plus llama.cpp parity. The
GGUF `rope_freqs` tensor (Llama-3 frequency divisors) is carried through
and preferred at runtime when present; Qwen3 uses plain theta=5e6.

## Tokenizer

Byte-level BPE (tiktoken-style, 128k vocab). The converter decodes GGUF's
byte-unicode token strings to raw bytes, resolves merge strings to id
pairs `(left, right) -> (result, rank)` sorted for binary search, and
emits a flat blob (token table + string pool + byte->id table + special
ids). The runtime pretokenizer is a hand-rolled implementation of the
family split regexes (contractions, letter runs with optional prefix,
digit runs, punctuation, whitespace lookahead) using core's Unicode
tables. All 57 reference cases per family (emoji, CJK, Arabic, Polish,
code, JSON, URLs, special-token literals, ...) match the official HF
tokenizers exactly; exotic Unicode-category edge cases may deviate — a
documented limitation. Control tokens are never encoded from user text.

The blob's template field selects the chat format: Llama-3 headers
(`<|start_header_id|>` … `<|eot_id|>`, BOS-prefixed) or ChatML
(`<|im_start|>role\n` … `<|im_end|>\n`, no BOS — Qwen). The Qwen
pretokenizer differs from Llama-3's in exactly one rule (single `\p{N}`
instead of `{1,3}`). Both are fixture-tested against the official HF
tokenizers (57 cases each) and the ChatML path against
`apply_chat_template` exactly. The UI shows `user:` / `llama:` or
`user:` / `qwen:` by family.

## Inference

Per token: embedding row dequant → n_layers x [RMSNorm → QKV matvec →
(Qwen3: per-head RMSNorm on Q and K) → RoPE → f16 KV append → GQA
attention → output matvec → residual → RMSNorm → SwiGLU MLP → residual]
→ final norm → classifier.

Family differences handled by the same engine: attention width may
differ from hidden width (Qwen3: 32x128=4096 vs dim 2560 — dedicated
q/attention-out buffers), per-head Q/K RMSNorm before RoPE, and RoPE
pairing style — **adjacent pairs** for llama-family GGUFs (conversion
permutes Q/K weights) vs **NEOX half-split pairs** for Qwen3. Getting
the RoPE style wrong produces coherent-but-divergent output; parity
tests catch it.

- Weight matrices are dtype-tagged (`QMat`); dispatch happens once per
  matvec call. Activations are quantized to the matching format — Q8_0
  (32-blocks) or Q8_K (256-blocks with group sums, llama.cpp numerics) —
  once per activation vector even when several matrices consume it.
- Integer dots via AVX2: `sign/maddubs/madd` for q8xq8; the k-quant
  kernels follow ggml's maddubs structure with scalar 6-bit scale/min
  bookkeeping and bsums-based min correction (Q4_K) / bit-plane
  reassembly minus 32 (Q6_K).
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

Correctness: AVX2 kernels are tested against scalar (including
adversarial/saturated k-quant blocks); f16 against known bit patterns;
the full forward pass is pinned to **token-for-token greedy parity with
llama.cpp** on the same artifacts (six regression tests across both
models, including a chat-templated reply and the tied-head audit).

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

- Context capped at 4096 tokens (KV memory: 128 MB for Llama 1B, 604 MB
  for Qwen3 4B — the boot arena is sized from `InferCtx::required_bytes`);
  the conversation auto-resets when full. No KV eviction/sliding window.
- Prefill is unbatched (see above).
- Pretokenizer is an approximation outside common Unicode classes.
- Glyph coverage is ASCII: model output outside it (emoji, CJK) renders
  as `?` in the chat UI (the tokenizer handles it correctly).
- Requires UEFI; no legacy BIOS path.
- Firmware keyboard repeat/rollover behaviour varies between vendors.
