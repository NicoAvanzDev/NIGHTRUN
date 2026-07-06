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
