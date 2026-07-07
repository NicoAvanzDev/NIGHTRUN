# x86_64 baseline for the Pi 5 port (feature/rpi5-support)

Reference point captured at branch creation (2026-07-07, HEAD = master @
fdde761 lineage). All later x86 measurements on this branch compare
against these numbers; a repeatable regression beyond ~2-3% blocks the
branch until explained.

## Test suite

`cargo test --release`: **51 tests, 0 failures** (kernels incl.
scalar/AVX2/matmul equality, k-quant adversarial blocks, tokenizer
fixtures x3 families, chat templates, llama.cpp greedy parity pins x3
models, prefill bit-identity x3 models, streaming-verifier corruption
tests, font coverage).

## Host engine (nrhost, 8 threads, --temp 0, batch 64 prefill)

| model | prefill (96-97 tok prompt) | decode (32 tok) |
|---|---|---|
| Llama 3.2 1B Q8_0 | 64.0 tok/s | 21.7 tok/s |
| Qwen3 4B Q4_K_M | 20.7 tok/s | 10.6 tok/s |
| Granite 4.1 3B Q4_K_M | 22.9 tok/s | 12.2 tok/s |

## QEMU (q35, KVM, -smp 8; measured during the P-milestones at this HEAD)

| model | boot-to-chat (after splash) | pp | tg |
|---|---|---|---|
| Llama 1B (-m 4G) | 5.6 s | 52 tok/s | ~20 tok/s |
| Qwen3 4B (-m 6G) | 11.5 s | — | ~10 tok/s |
| Granite 3B (-m 5G) | 9.5 s | 24-26.5 tok/s | 12.5-14 tok/s |

Environment: 12-thread AVX2 host, 15 GB RAM; QEMU numbers vary ±20% with
host memory pressure (documented in docs/benchmarks.md).
