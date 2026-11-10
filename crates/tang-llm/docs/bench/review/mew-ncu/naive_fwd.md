#### naive_fwd

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@65536x256` | 5 | 433.5 | 2.168 | 52.1% | 120.98 | 129.2 | 0.30 | 37 | 69 | 83 | 16 | 133.20 | 3434 | 65536×256 | traffic |
| `main0@16x256` | 2 | 539.5 | 1.079 | 25.9% | 83.65 | 89.4 | 0.17 | 1 | 17 | 17 | 38 | 0.03 | 849992 | 16×256 | latency |
| `main0@1024x256` | 4 | 228.4 | 0.913 | 22.0% | 20.37 | 21.8 | 0.10 | 37 | 31 | 76 | 16 | 2.08 | 31437 | 1024×256 | latency |

kernel time 4.160 ms, traffic floor 0.912 ms (light 0.22); by class: traffic 2.168 ms (52%), latency 1.992 ms (48%)
