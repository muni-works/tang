#### H200 gdn_scan

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `main0@64x256` | 406 | 184.1 | 74.749 | 59.9% | – | – | – | 64×256 | ? |
| `main0@4x128` | 204 | 244.8 | 49.938 | 40.0% | 0.22 | 0.0 | 0.00 | 4×128 | latency |
| `main0@2x256` | 2 | 17.1 | 0.034 | 0.0% | 0.24 | 0.0 | 0.00 | 2×256 | latency |

kernels 124.721 ms over 612 launches; span 8503.169 ms, idle between kernels 8378.448 ms (99% of span); by class: ? 74.749 ms, latency 49.972 ms
