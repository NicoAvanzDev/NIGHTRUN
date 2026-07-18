# NightRun — Final Pre-Release Check

Date: 2026-07-14 · Branch: `release/final-preflight` · Baseline: `7c2ea5f` (master)

## 1. Scope

End-to-end release gate over runtime, kernels, formats, tokenizer, installer, images,
website, docs and benchmark tooling, building on the repository audit of 2026-07-08
(docs/QUALITY_AUDIT.md) rather than repeating it. Four release models validated; both
targets built; runtime behavior validated in QEMU.

## 2. Model validation table

| Model | Quant | Source (pinned) | GGUF sha256 | Inspect | .nrm conversion | Runtime load | Tokenizer/template | Inference smoke | RAM residency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | Q8_0 | bartowski @ 067b946c | matches manifest | dense llama, supported | byte-identical reconvert | QEMU boot OK | fixtures green | greedy deterministic ("1. Red 2. Blue 3. Yellow") | sealed-storage line verified | VALIDATED |
| Llama 3.2 3B Instruct | Q4_K_M | bartowski @ 5ab33fa9 | (see blocked note) | (from 2026-07-08 chain) | .nrm present, parses, loads | QEMU boot OK | family fixtures green | greedy deterministic | sealed-storage line verified | VALIDATED (GGUF re-fetch BLOCKED) |
| Granite 4.1 3B | Q4_K_M | ibm/bartowski @ ab470148 | matches manifest | dense granite, supported | byte-identical reconvert | QEMU boot OK | fixtures green | greedy deterministic | sealed-storage line verified | VALIDATED |
| Qwen3 4B Instruct 2507 | Q4_K_M | bartowski/Qwen @ a06e946b | matches manifest | qwen3, supported | byte-identical reconvert | QEMU boot OK | fixtures green | greedy deterministic | sealed-storage line verified | VALIDATED |

**Blocked note (Llama 3B GGUF re-fetch):** Hugging Face's CDN returned persistent
HTTP 403 (AccessDenied at the xet-bridge signed-URL hop) for both the pinned revision
and `main` during this pass, with no local HF token available. The shipped
`llama-3.2-3b-q4km.nrm` remains covered by its original verified chain: downloaded
2026-07-08 with sha256 matching the manifest pin, inspected ("conventional dense
transformer, supported", 28 layers, dim 3072, tied embeddings), converted and
validated then. What is missing is only an independent re-download today. To close:
re-run the pinned download when the CDN allows and confirm
`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.

Converter determinism: fresh conversions of all three locally-available GGUFs are
**byte-identical** to the shipped `.nrm` artifacts (sha256 compare).

## 3. Issues found and fixed in this pass

| Sev | Subsystem | Issue | Fix | Validation |
|---|---|---|---|---|
| P2 | xtask bench | `cargo xtask bench` overwrote docs/benchmarks.md wholesale, which would destroy the curated measurement log (Pi rows, llama.cpp head-to-head, decay notes) | bench() now appends a dated snapshot section, never clobbers | code inspection + build |
| P2 | tests | prefill bit-identity covered lengths 1..129; release spec demands coverage to 513 | lengths extended to 255/257/511/512/513 (ctx 640) | suite run (result below) |
| P3 | repo | fmt drift in backdrop.rs; duplicate `/build/` gitignore line | cargo fmt; dedupe | fmt --check clean |
| — | repo | `build/` log tracking (carried finding) | already fixed by Michal before this pass; verified 0 tracked files | git ls-files |

## 4. Performance results

Perf measurement collided with an actively-used host (desktop load 3.7-8.3 during the
window; an earlier slot was additionally polluted by the prefill-513 suite running
host-side). Numbers below are labeled accordingly; the documented reference numbers
live in docs/benchmarks.md (2026-07-07, quiet host).

**Clean-enough spot checks (QEMU q35, KVM, 8 cores, busy desktop):**

| Model | Runs | boot->chat | prefill | decode | Note |
|---|---|---|---|---|---|
| Llama 1B Q8_0 (4G) | 4 | 6.1-6.8 s | 37-53 tok/s | 14.5-18.5 tok/s | best run matches the documented 52-56 pp; boot ~1 s slower than the 5.6 s reference under load |
| Llama 3B Q4_K_M (5G) | 3* | 17.3-21.1 s | 14-16 tok/s | 6.9-8.2 tok/s | *contended window; treat as lower bound; first-ever QEMU numbers for this model |
| Granite 3B Q4_K_M (5G) | 3* | 10.7-11.2 s | 17-24 tok/s | 8.6-9.4 tok/s | *contended window; reference is 23-27 pp / ~14 tg |
| Qwen3 4B Q4_K_M (6G) | 3* | 14.0-14.3 s | 13-20 tok/s | ~7.5 tok/s | *contended window; reference is ~23 pp / 8.7-11.6 tg |

**Regression assessment: none.** No engine code changed since the 2026-07-08 audit
(whose quiet-host runs matched baselines at a fixed profile); the only post-audit code
change is the splash backdrop. Under identical load conditions the llama spot check
recovers to the documented range (pp 53 tok/s). The contended rows above are
measurement environment, not code. A quiet-host 3-run set per model remains the one
open perf item; command recorded below.

Rerun later (quiet host): `bash <scratchpad>/perf-only.sh out.log` or per model
`cargo xtask run --img --model <m> --mem <M> --smp 8 --secs 60 --keys "16:...\n"`
and read pp/tg/chat-ready from target/serial.log. `cargo xtask bench` (now
append-only) adds a dated snapshot to docs/benchmarks.md.

## 5. Checks run

| Check | Result |
|---|---|
| cargo fmt --check | clean |
| cargo clippy --workspace --exclude nr-boot --all-targets | 0 warnings |
| cargo test --release (host) | 68 passed / 0 failed |
| prefill bit-identity 1..513 | 1 passed (773 s) |
| EFI build x86_64 (nightly, custom target) | 0 errors |
| EFI build aarch64 (stable) | 0 errors |
| Installer suite | 47 passed / 0 failed |
| ShellCheck (installer + bootstrap + firmware script) | clean |
| Website QA (structure/links/fragments/balance/alt) | ALL OK, 309 KB |
| README/docs link check | no broken links |
| Secret scan | none |
| Large-file scan | max tracked 144 KB (demo GIF) |
| GGUF sha256 vs manifest (3 local artifacts) | all match |

## 5a. QEMU functional matrix (all four models, this pass)

Every model: boot to chat, storage-sealed serial line, multi-turn (two prompts), ESC
interrupt mid-generation, `/clear` (model resident), `/bye` (clean shutdown). Qwen
additionally handled a Polish prompt. One artifact in the 3B transcript ("user: y?")
is the capture script typing into the input while generation had it locked, i.e. the
lock working as designed, not a defect.

## 6. Skipped or blocked validation

- **Llama 3B GGUF re-download**: blocked by CDN 403 (above). Release impact: low; the
  artifact chain from 2026-07-08 stands. Re-run when HF access recovers.
- **Loopback flash e2e** (`sudo scripts/installer/tests/loopback.sh`): requires
  interactive sudo. Readback math is covered device-free (47-check suite).
- **llama.cpp side-by-side re-run**: no local llama.cpp binary; the documented host
  comparison in docs/benchmarks.md (2026-07-07, greedy, same GGUF) stands as the
  recorded reference. A fair fresh comparison needs that binary rebuilt.
- **Real-hardware boots**: x86 USB re-verify and Pi 5 sdot re-bench/thermal runs remain
  user-gated (hardware access). QEMU passes on both architectures.
- **C1-stepping Pi boards**: no hardware.
- **Quiet-host perf set (3 runs x 4 models) + a bench snapshot run**: the host was in
  active desktop use during this pass and the perf job was stopped externally;
  deferred with exact rerun commands above. Release impact: low (no code-change
  vector for a regression; spot checks recover to documented numbers).

## 7. Remaining risks

- Real-hardware x86 firmware quirks are only QEMU/OVMF-proven; boot reports wanted.
- Pi 5 numbers predate the sdot kernels (faster path shipped, unbenchmarked on-device).
- Contiguous-allocation RAM floor (~2.5x model size) may surprise on fragmented
  firmware memory maps; documented in README/architecture.
- Granite greedy near-ties (documented /10 logit-scale behavior) can flip single tokens
  between numerically-equivalent implementations.

## 8. Release recommendation

**READY WITH DOCUMENTED LIMITATIONS.**

All four release models validate end-to-end on the shipped artifacts: manifest-pinned
hashes verified (3 of 4 re-hashed this pass; the 4th blocked by a CDN denial with its
original verification chain intact), conversion proven byte-deterministic, greedy
inference deterministic, full QEMU interaction matrix green including the first Qwen
and Llama-3B boots. Static gates are uniformly green. The open items are environmental
(quiet-host perf set, loopback sudo run, real-hardware boots, 3B GGUF re-fetch), each
with a recorded rerun path, and none has a plausible code-defect vector.
