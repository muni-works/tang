#### matmul1024

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@64x256` | 1 | 83.1 | 0.083 | 100.0% | 12.53 | 13.4 | 0.16 | 31 | 17 | 16 | 178 | 0.78 | 2435 | 64×256 | latency |

kernel time 0.083 ms, traffic floor 0.013 ms (light 0.16); by class: latency 0.083 ms (100%)
