#### ckpt_fwd

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@65536x256` | 1 | 7117.6 | 7.118 | 76.9% | 106.62 | 113.9 | 0.02 | 12 | 2 | 92 | 40 | 133.20 | 306 | 65536×256 | latency |
| `main0@16x256` | 2 | 534.0 | 1.068 | 11.5% | 69.92 | 74.7 | 0.14 | 2 | 14 | 17 | 26 | 0.03 | 955636 | 16×256 | latency |
| `main0@1024x256` | 2 | 533.9 | 1.068 | 11.5% | 38.94 | 41.6 | 0.08 | 51 | 25 | 79 | 16 | 2.08 | 51894 | 1024×256 | latency |

kernel time 9.253 ms, traffic floor 0.347 ms (light 0.04); by class: latency 9.253 ms (100%)
