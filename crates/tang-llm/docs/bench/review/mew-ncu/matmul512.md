#### matmul512

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@96x128` | 1 | 59.9 | 0.060 | 100.0% | 8.94 | 9.6 | 0.16 | 18 | 16 | 8 | 168 | 1.17 | 7891 | 96×128 | latency |

kernel time 0.060 ms, traffic floor 0.010 ms (light 0.16); by class: latency 0.060 ms (100%)
