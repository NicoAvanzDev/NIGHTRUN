# NightRun — Repository Quality Audit

Date: 2026-07-08 · Baseline commit: `204d11cd` (master) · Audit branch: `audit/repository-quality`

## 1. Scope

Every hand-written subsystem was inspected: Rust workspace (nr-gfx, nr-ui, nr-tensor,
nr-token, nr-model, nr-boot, nr-runtime), tools (nrconvert, nrhost, xtask), the installer
(`install.sh` + `scripts/installer/`), images (x86_64 + Pi 5 paths), website (`web/`),
documentation, fixtures, and repository hygiene. Generated artifacts, model weights and
vendored firmware sources were excluded except for hygiene checks.

The audit ran hypothesis-first: the maintainer-authored weak-spot list (untrusted-input
parsers, installer device resolution, interrupt paths, stale bench patterns) was verified
item by item; findings below record both confirmed bugs and disproven suspicions.

## 2. Issues found and fixed

| # | Sev | Subsystem | Issue | Fix | Regression test |
|---|-----|-----------|-------|-----|-----------------|
| 1 | P1 | nr-model/format.rs | Region/entry bounds used wrapping arithmetic; crafted headers passed validation and panicked at slice time | checked_add/mul on every untrusted sum; clean ParseError | tests/format_malformed.rs (10 cases) |
| 2 | P1 | nr-model/format.rs | Tensor offset alignment never validated; `TensorView::f32` reinterpretation is UB on a misaligned malformed file | parse-time 64-byte alignment reject + defensive assert at use | misaligned_entry_offset_rejects |
| 3 | P1 | nr-token/blob.rs | Vocab-table pool ranges and special-token ids unvalidated → OOB panic at first `token_bytes`; specials fed to model unchecked | full table validation + specials < vocab at parse | tests/blob_malformed.rs (5 cases) |
| 4 | P1 | installer media.sh | Protected-disk resolution stopped one PKNAME hop up — on LUKS/LVM/stacked roots the physical disk was never excluded (a USB-attached encrypted root could be offered for flashing) | full ancestor walk via `lsblk -snl` | live host-resolution tests in tests/run.sh |
| 5 | P2 | nr-model byte_size | `n * 4` could overflow u64 with absurd dims, defeating the size-consistency check | checked multiplies per dtype | dim_overflow_rejects |
| 6 | P2 | xtask bench | Serial patterns `"model loaded:"` / `"verified in"` went stale when streaming CRC reworded the line — benchmarks.md silently lost the load-time row | pattern updated to the current line, comment pins the format | validated against a live serial.log |
| 7 | P2 | nr-boot input | Prompt line grew unboundedly | 8192-char cap (beyond any context fit) | code-level (UI path) |
| 8 | P2 | nr-ui wrap() | cols==0 caused a nonterminating chunk loop on degenerate surfaces | early return | code-level guard |
| 9 | P2 | nr-boot flush_utf8 | An invalid leading byte wedged the pending buffer forever (subsequent valid text never rendered) | moved to nr-token (host-testable), drops invalid bytes one at a time | flush_utf8_handles_split_sequences |
| 10 | P3 | web | Dead `.toc` highlight JS (selector no longer exists); relative `og:image` (scrapers need absolute) | removed / absolute URL | site QA script |
| 11 | P3 | workspace | fmt drift; 21 clippy warnings; 2 missing `# Safety` docs | fmt applied; zero warnings (fixes + targeted justified allows); Safety docs written | `cargo fmt --check`, clippy clean |

**Disproven hypotheses (verified clean):**
- installer `${VAR:+-H "…"}` token expansion does **not** word-split — bash preserves the
  inner quoting; demonstrated argv integrity with a fake token.
- Ctrl+C during `dd`: the terminal delivers SIGINT to the foreground process group
  (dd included); the trap reports the media as incomplete and never claims success.
- Readback verification is fail-safe by construction: a failed read produces a
  partial-stream digest which cannot match the expected image hash.
- verify.rs streaming CRC: wrapped-bounds crafting degrades to a CRC mismatch (still
  hardened to checked arithmetic for clarity).
- No secrets in tracked files; no large blobs (max tracked file: 132 KB font); `*.img`,
  `models/`, `vendor/` all ignored; LICENSE (MIT) present; Spleen attribution intact.

## 3. Refactor regression ledger

| Change | Baseline checks | Post checks | Result |
|--------|----------------|-------------|--------|
| Parser hardening (format.rs/verify.rs/blob.rs) | 51 host tests green; llama+granite QEMU boots | 68 tests green (incl. 16 new adversarial); llama+granite QEMU boots; parity/prefill suites unchanged | no behavior change on valid inputs |
| flush_utf8 move to nr-token | (function previously untestable in nr-boot) | split-sequence + invalid-byte tests; EFI builds green | intentional improvement: invalid bytes now skipped instead of wedging |
| readback extraction (nr_readback_sha + nr_dd seam) | installer suite 42/42 | suite 47/47 incl. odd-size/corruption cases; ShellCheck clean | identical hashing behavior, now proven |
| media.sh ancestor walk | 42/42 (fixtures) | 47/47 + live host resolution (nvme root resolves) | strictly more disks protected |
| fmt + clippy zero-warning pass | full suite green | full suite green, both EFI targets build | formatting/lint only |
| Input cap, wrap guard, bench patterns | QEMU functional matrix | same matrix green post-change | no user-visible change |

## 4. Validation matrix

| Check | Config | Result |
|---|---|---|
| cargo test --release (host) | default members | 68 passed / 0 failed |
| cargo fmt --check / clippy --workspace --exclude nr-boot --all-targets | stable | clean / 0 warnings |
| nr-boot EFI build | x86_64-nightrun-uefi.json, nightly -Zbuild-std | 0 errors |
| nr-boot EFI build | aarch64-unknown-uefi, stable | 0 errors |
| NEON kernel suite | aarch64-unknown-linux-musl via qemu-user | 25 passed |
| Installer suite | fixtures + live read-only host checks | 47 passed / 0 failed |
| ShellCheck | install.sh, all libs, tests, bootstrap, firmware script | clean |
| Website QA | parse/h1/alt/links/fragments/tag-balance/size | all OK, 183 KB |
| QEMU functional matrix | P-llama (2 vCPU/2560M): boot→chat, prompt, ESC, /clear, /bye | all correct |
| Corrupt-image boot | bit-flipped copy, direct QEMU | clean FATAL (DataCrc), no "ready" |
| OOM boot | granite at 4096M | clean FATAL (OutOfMemory), actionable message |
| Dependency licenses | cargo metadata inventory | all MIT/Apache-compatible (1 MPL-2.0, no unknowns) |

## 5. Performance (constrained QEMU, fixed profiles)

Profiles: qemu-system-x86_64 q35, KVM, `-cpu max`, **2 vCPUs**, OVMF 4M, real disk image,
72/67-token prompt fixture, streaming decode, 3 runs, values = median (min–max).
Host: 16 threads / 15 GiB; ≥ 10 GiB host headroom maintained; no swap.

| Metric | P-llama baseline (2560M) | P-llama post-audit | P-granite baseline (5120M) | P-granite post |
|---|---|---|---|---|
| boot→chat | 5217 ms (4984–5254) | 4869 ms (4791–5534) | 8021 ms (8017–8026) | 8017 ms (1 run) |
| prefill | 27 tok/s (26–28) | 28 tok/s (27–29) | 16 tok/s (14–16) | 16 tok/s |
| decode | 11.55 tok/s (11.3–12.3) | 12.8 tok/s (12.8–13.1) | 7.9 tok/s (6.9–8.7) | 9.0 tok/s |

**No regression** — post-audit medians are equal or better at identical profiles (the
parser hardening adds O(tensor-count) checks at load only). These constrained numbers are
NOT the published benchmark numbers; `docs/benchmarks.md` remains the labeled 8-core
reference and was deliberately not regenerated under the audit's 4-vCPU cap.

Empirical note: granite (2.0 GB blob) needs a 5120M guest — the model requires one
contiguous UEFI page allocation, and a 4096–4608M map is too fragmented. Recorded as a
known limitation (scatter loading would be an architecture change, out of audit scope).

## 6. Test environment and resource constraints

```
No additional models were downloaded.
Performance tests ran in constrained QEMU only (2 vCPUs, minimal viable RAM), per the
  fixed profiles above, using locally available artifacts.
No physical hardware boot tests were run.
No physical USB or SD devices were flashed.
Runtime tests ran in QEMU only.
QEMU limits used: 2 vCPUs; 2560M (llama) / 5120M (granite) guests; >=10 GiB host headroom.
Local model artifacts used: models/model.nrm (Llama 1B Q8_0),
  models/granite-4.1-3b-q4km.nrm; (llama-3b/qwen .nrm present but not exercised in QEMU
  to bound audit runtime — same code paths as the two tested families' dtype/rope mix).
Model-backed tests skipped for missing artifacts: none (all four .nrm files present).
Remaining validation requiring real hardware: x86 USB boot, Pi 5 SD boot + sdot
  benchmarks + sustained thermal run (tracked pre-audit, unchanged).
```

## 7. Skipped validation

- **Loopback flash e2e** (`sudo scripts/installer/tests/loopback.sh`): needs interactive
  sudo, unavailable in this session. The readback math is now covered by device-free
  tests; the loop-device run remains one `sudo` away.
- **`cargo xtask bench` regeneration**: hardcodes 8 vCPUs — running it would breach the
  audit's 4-vCPU cap, and constrained numbers must not overwrite the labeled 8-core
  reference. Pattern fix validated against a live serial log. Run once unconstrained.
- **llama.cpp side-by-side**: no local llama.cpp binary; downloading one is out of
  constraint. The greedy token-parity fixtures pinned in `nr-model/tests/parity.rs`
  stand as the recorded reference.
- **Qwen/Llama-3B QEMU boots**: artifacts exist locally; skipped to bound audit runtime
  (Qwen needs a 6G+ guest). Host-side parity/template tests for both are green.
- **Real-hardware anything** (spec-mandated skip).

## 8. Remaining known risks

- Contiguous-allocation requirement makes minimum RAM ~2.5× blob size at the margin;
  fragmented firmware maps on real machines may need more headroom than QEMU suggests.
- The Pi 5 path is validated on one D0 board revision; C1 untested.
- Hand-built Granite hybrid rejection relies on nrconvert's architecture checks; new
  Granite releases may need new metadata handling.
- The installer's device model assumes lsblk/findmnt semantics of util-linux ≥ 2.37;
  exotic storage stacks (multipath, MD-on-dm) are covered by the ancestor walk but have
  no fixture.
- Parity near-ties on Granite (÷10 logit scale) remain sensitive to kernel changes;
  `--debug-gap` methodology documented in CLAUDE.md.

## 9. Release recommendation

**READY WITH DOCUMENTED LIMITATIONS.**

All P0/P1 findings are fixed with regression tests; the suites, builds, installer checks
and constrained-QEMU functional matrix are fully green; performance is unchanged at fixed
profiles. The limitations are environmental (real-hardware re-verification after this
branch merges, the loopback run, one unconstrained bench regeneration) rather than code
defects, and each has a concrete run-it-later command recorded above.
