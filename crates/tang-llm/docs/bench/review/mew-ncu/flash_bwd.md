#### flash_bwd

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@64x128` | 1 | 511.5 | 0.511 | 63.6% | 8.56 | 9.1 | 0.02 | 13 | 2 | 8 | 255 | 0.78 | 455475 | 64×128 | latency |
| `main0@128x256` | 1 | 250.7 | 0.251 | 31.2% | 6.32 | 6.8 | 0.03 | 22 | 3 | 16 | 99 | 1.56 | 9781 | 128×256 | latency |
| `main0@1024x256` | 4 | 5.4 | 0.022 | 2.7% | 2.13 | 2.3 | 0.42 | 14 | 44 | 70 | 16 | 2.08 | 1145 | 1024×256 | latency |
| `main0@4096x256` | 1 | 17.4 | 0.017 | 2.2% | 5.77 | 6.2 | 0.35 | 75 | 36 | 89 | 27 | 8.33 | 204 | 4096×256 | compute |
| `main0@16x256` | 1 | 2.7 | 0.003 | 0.3% | 0.16 | 0.2 | 0.06 | 1 | 7 | 16 | 16 | 0.03 | 631 | 16×256 | launch |

kernel time 0.804 ms, traffic floor 0.031 ms (light 0.04); by class: latency 0.784 ms (97%), compute 0.017 ms (2%), launch 0.003 ms (0%)
