# NightRun benchmarks (measured)

Environment: QEMU q35, KVM, `-cpu max -smp 8 -m 4G`, OVMF; host: 12 hardware threads.
Date: 2026-07-06

```
[smp] 8 processors (8 enabled), 7 APs
[smp] 7 workers active
[boot] model loaded: 1319604608 bytes in 4102 ms
[boot] verified in 4184 ms; vocab=128256 layers=16
[boot] chat-ready in 8850 ms (after splash)
[gen] prefill 63 tokens in 2672 ms (23 tok/s)
[gen] 74 tokens in 3473 ms (21887 milli-tok/s)
```

See docs/architecture.md for the performance discussion.

## Head-to-head vs llama.cpp (host, same machine)

Same Q8_0 weights, same prompt, greedy, 64 tokens, 8 threads, ctx 4096
(llama.cpp b1-cb295bf, `-t 8 -c 4096`; NightRun engine via nrhost):

| | generation | prompt processing |
|---|---|---|
| llama.cpp | 19.6 tok/s | 60–65 tok/s |
| NightRun  | 20.3 tok/s | 21.9 tok/s |
| NightRun, 1 thread | 7.2 tok/s | 7.1 tok/s |

Generation is at parity (both are memory-bandwidth-bound reading ~1.1 GB
of weights per token). Prompt processing is ~3x slower in NightRun because
prefill runs the single-token path instead of a batched GEMM — the
documented next optimization. Note: benchmark llama.cpp with `-c 4096`;
its default (the model's full 131k training context) allocates a ~4.3 GB
KV cache, which swaps on an 8 GB-class machine and produces nonsense
numbers.
