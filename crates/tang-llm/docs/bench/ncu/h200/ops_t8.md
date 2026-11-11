#### H200 ops_t8

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `dop01_gemv_q8_0` | 12 | 61.4 | 0.737 | 82.0% | 130.96 | 27.3 | 0.44 | 64×256 | latency |
| `dop00_route_topk` | 4 | 9.4 | 0.037 | 4.2% | 0.05 | 0.0 | 0.00 | 8×256 | latency |
| `cx_norm_q8` | 4 | 5.8 | 0.023 | 2.6% | 0.33 | 0.1 | 0.01 | 8×256 | latency |
| `cx_route` | 4 | 5.8 | 0.023 | 2.6% | 0.03 | 0.0 | 0.00 | 8×32 | latency |
| `cx_moe_add_norm` | 4 | 4.7 | 0.019 | 2.1% | 1.09 | 0.2 | 0.05 | 8×1024 | launch |
| `dop00_xq` | 12 | 1.4 | 0.017 | 1.9% | 0.12 | 0.0 | 0.02 | 64×256 | launch |
| `dop02_norm_ss` | 4 | 1.6 | 0.006 | 0.7% | 0.09 | 0.0 | 0.01 | 8×256 | launch |
| `dop01_norm_ss` | 4 | 1.6 | 0.006 | 0.7% | 0.07 | 0.0 | 0.01 | 8×256 | launch |
| `dop00_combine_norm` | 4 | 1.5 | 0.006 | 0.7% | 0.75 | 0.2 | 0.10 | 64×256 | launch |
| `dop03_combine_norm` | 4 | 1.4 | 0.006 | 0.6% | 0.08 | 0.0 | 0.01 | 64×256 | launch |
| `dop02_add_norm` | 4 | 1.4 | 0.006 | 0.6% | 0.08 | 0.0 | 0.01 | 64×256 | launch |
| `dop00_add_norm` | 4 | 1.1 | 0.004 | 0.5% | 0.14 | 0.0 | 0.03 | 64×256 | launch |
| `dop01_route_wts` | 4 | 1.0 | 0.004 | 0.5% | 0.01 | 0.0 | 0.00 | 1×256 | launch |
| `dop01_combine_norm` | 4 | 1.0 | 0.004 | 0.4% | 0.14 | 0.0 | 0.03 | 64×256 | launch |

kernels 0.899 ms over 72 launches; span 4507.319 ms, idle between kernels 4506.420 ms (100% of span); by class: latency 0.821 ms, launch 0.078 ms
