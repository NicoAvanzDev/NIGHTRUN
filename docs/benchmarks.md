# NightRun benchmarks (measured)

Environment: QEMU q35, KVM, `-cpu max -smp 8`, OVMF; host: 12 hardware
threads (AVX2), 15 GB RAM. Date: 2026-07-07. Numbers are single scripted
runs; QEMU results vary ±20% with host memory pressure (the guest
competes with the host page cache) — ranges given where observed.

## Llama 3.2 1B Instruct Q8_0 (`-m 4G`)

```
[boot] model loaded: 1319604608 bytes in 3846 ms
[boot] verified in 4007 ms; vocab=128256 layers=16
[boot] chat-ready in 8390 ms (after splash)
[gen] prefill 72 tokens: 14-23 tok/s across runs
[gen] generation: 14.5-21.9 tok/s across runs
```

Host (nrhost, identical engine): 22.9 tok/s prefill, ~20 tok/s
generation at 8 threads — marginally faster than before the Qwen
dispatch refactor (dispatch-at-load costs nothing).

## Qwen3-4B-Instruct-2507 Q4_K_M (`-m 6G`)

```
[boot] model loaded: 2495954816 bytes in 12694 ms
[boot] verified in 6872 ms; vocab=151936 layers=36
[boot] storage sealed - model reads from disk are now forbidden
[boot] chat-ready in 20744 ms (after splash)
[gen] prefill 69 tokens in 6111 ms (11 tok/s)
[gen] generation 8.7-11.6 tok/s
```

Host (nrhost): 11.3 tok/s prefill, 10.6-11.9 tok/s generation at
8 threads; 4.0/4.4 tok/s single-thread. Memory: 2.38 GB model resident
+ ~650 MB arena (f16 KV cache at ctx 4096 = 604 MB, plus scratch).

## Granite 4.1 3B Q4_K_M (`-m 5G`)

```
[boot] model loaded: 2098994176 bytes in 9088 ms
[boot] verified in 5297 ms; vocab=100352 layers=40
[boot] storage sealed - model reads from disk are now forbidden
[boot] chat-ready in 15030 ms (after splash)
[gen] prefill 62 tokens in 4329 ms (14 tok/s), first-token latency ~1.7-4.3 s
[gen] generation 13.6-15.3 tok/s
```

Memory: 2.0 GB model resident + ~380 MB arena (f16 KV at ctx 4096 =
336 MB). Greedy parity vs llama.cpp on the identical GGUF: exact on the
pinned prompts (12-16 tokens + chat reply). Note: Granite divides logits
by 10, which compresses greedy top-2 gaps; some prompts flip single
tokens (measured gaps < 0.4 at logit scale ~31) between internally-
deterministic implementations, then re-converge — the same class of
divergence llama.cpp shows across its own backends.

## Head-to-head vs llama.cpp (host, same machine)

Same weights, same prompt, greedy, 8 threads, ctx 4096
(llama.cpp b1-cb295bf):

| | generation | prompt processing |
|---|---|---|
| llama.cpp, Llama 1B Q8_0 | 19.6 tok/s | 60-65 tok/s |
| NightRun, Llama 1B Q8_0 | 20.3 tok/s | 21.9 tok/s |
| llama.cpp, Qwen3 4B Q4_K_M | 10.7 tok/s | 31-32 tok/s |
| NightRun, Qwen3 4B Q4_K_M | 10.6-11.9 tok/s | 11.3 tok/s |
| NightRun, Granite 4.1 3B Q4_K_M | 13.6-15.3 tok/s (QEMU) | ~14 tok/s |

Generation is at parity for both models (memory-bandwidth-bound, same
integer dot schemes). Prompt processing remains ~3x slower in NightRun
because prefill runs the unbatched single-token path — the documented
next optimization. Benchmark llama.cpp with `-c 4096`: its default (the
model's full training context) allocates a KV cache far beyond an
8-16 GB machine and swaps.

Correctness note: every speed comparison above is backed by
token-for-token greedy parity tests on the identical artifacts
(`crates/nr-model/tests/parity.rs`). One unreproducible greedy
divergence was observed on the host during a memory-pressure benchmark
run (five subsequent runs plus the parity suite were identical);
bare-metal boots CRC-verify all tensor data at load, host-side nrhost
does not.
