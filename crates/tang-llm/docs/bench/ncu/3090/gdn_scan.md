#### gdn_scan

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `main0@2048x256` | 406 | 301.2 | 122.303 | 60.7% | 182.89 | 195.4 | 0.65 | 53 | 66 | 2048×256 | traffic |
| `main0@128x128` | 203 | 389.4 | 79.046 | 39.2% | 215.23 | 229.9 | 0.59 | 42 | 61 | 128×128 | traffic |
| `main0@2x256` | 2 | 21.5 | 0.043 | 0.0% | 0.24 | 0.3 | 0.01 | 1 | 1 | 2×256 | latency |
| `main0@4x128` | 1 | 10.0 | 0.010 | 0.0% | 0.22 | 0.2 | 0.02 | 2 | 2 | 4×128 | latency |

kernel time 201.402 ms, traffic floor 126.011 ms (light 0.63); by class: traffic 201.349 ms (100%), latency 0.053 ms (0%)
