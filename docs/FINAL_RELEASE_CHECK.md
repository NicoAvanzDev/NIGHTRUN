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

(filled in from the QEMU matrix; see section below)

## 5. Checks run

| Check | Result |
|---|---|
| cargo fmt --check | clean |
| cargo clippy --workspace --exclude nr-boot --all-targets | 0 warnings |
| cargo test --release (host) | 68 passed / 0 failed |
| prefill bit-identity 1..513 | (result pending at draft time) |
| EFI build x86_64 (nightly, custom target) | 0 errors |
| EFI build aarch64 (stable) | 0 errors |
| Installer suite | 47 passed / 0 failed |
| ShellCheck (installer + bootstrap + firmware script) | clean |
| Website QA (structure/links/fragments/balance/alt) | ALL OK, 309 KB |
| README/docs link check | no broken links |
| Secret scan | none |
| Large-file scan | max tracked 144 KB (demo GIF) |
| GGUF sha256 vs manifest (3 local artifacts) | all match |

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

## 7. Remaining risks

- Real-hardware x86 firmware quirks are only QEMU/OVMF-proven; boot reports wanted.
- Pi 5 numbers predate the sdot kernels (faster path shipped, unbenchmarked on-device).
- Contiguous-allocation RAM floor (~2.5x model size) may surprise on fragmented
  firmware memory maps; documented in README/architecture.
- Granite greedy near-ties (documented /10 logit-scale behavior) can flip single tokens
  between numerically-equivalent implementations.

## 8. Release recommendation

(finalized at end of pass)
