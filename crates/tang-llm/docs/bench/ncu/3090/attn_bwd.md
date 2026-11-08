#### attn_bwd

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@512x128` | 44 | 4346.3 | 191.235 | 69.0% | 149.24 | 159.4 | 0.04 | 24 | 4 | 512×128 | latency |
| `main0@512x256` | 45 | 1572.6 | 70.769 | 25.5% | 93.90 | 100.3 | 0.06 | 32 | 7 | 512×256 | latency |
| `main0@16384x256` | 221 | 51.0 | 11.265 | 4.1% | 30.92 | 33.0 | 0.65 | 23 | 78 | 16384×256 | traffic |
| `main0@32768x32` | 45 | 82.5 | 3.713 | 1.3% | 70.14 | 74.9 | 0.91 | 17 | 93 | 32768×32 | traffic |
| `main0@128x256` | 45 | 7.4 | 0.333 | 0.1% | 1.99 | 2.1 | 0.29 | 2 | 30 | 128×256 | latency |

kernel time 277.315 ms, traffic floor 22.299 ms (light 0.08); by class: latency 262.337 ms (95%), traffic 14.978 ms (5%)
