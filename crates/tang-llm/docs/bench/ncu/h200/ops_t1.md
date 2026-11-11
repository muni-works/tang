#### H200 ops_t1

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `dop00_gemv_q8_0` | 12 | 63.5 | 0.762 | 84.5% | 119.35 | 24.9 | 0.39 | 256×256 | latency |
| `dop00_route_topk` | 4 | 9.3 | 0.037 | 4.1% | 0.12 | 0.0 | 0.00 | 1×256 | latency |
| `cx_route` | 4 | 5.8 | 0.023 | 2.6% | 0.10 | 0.0 | 0.00 | 1×32 | latency |
| `cx_norm_q8` | 4 | 5.7 | 0.023 | 2.5% | 0.12 | 0.0 | 0.00 | 1×256 | latency |
| `cx_moe_add_norm` | 4 | 4.6 | 0.019 | 2.1% | 0.13 | 0.0 | 0.01 | 1×1024 | launch |
| `dop01_norm_ss` | 4 | 1.5 | 0.006 | 0.7% | 0.01 | 0.0 | 0.00 | 1×256 | launch |
| `dop02_norm_ss` | 4 | 1.5 | 0.006 | 0.7% | 0.04 | 0.0 | 0.01 | 1×256 | launch |
| `dop00_combine_norm` | 4 | 1.2 | 0.005 | 0.6% | 0.10 | 0.0 | 0.02 | 8×256 | launch |
| `dop03_combine_norm` | 4 | 1.2 | 0.005 | 0.5% | 0.02 | 0.0 | 0.00 | 8×256 | launch |
| `dop02_add_norm` | 4 | 1.1 | 0.005 | 0.5% | 0.02 | 0.0 | 0.00 | 8×256 | launch |
| `dop01_route_wts` | 4 | 1.0 | 0.004 | 0.4% | 0.04 | 0.0 | 0.01 | 1×256 | launch |
| `dop00_add_norm` | 4 | 0.9 | 0.004 | 0.4% | 0.04 | 0.0 | 0.01 | 8×256 | launch |
| `dop01_combine_norm` | 4 | 0.9 | 0.004 | 0.4% | 0.02 | 0.0 | 0.00 | 8×256 | launch |

kernels 0.901 ms over 60 launches; span 2851.256 ms, idle between kernels 2850.354 ms (100% of span); by class: latency 0.845 ms, launch 0.056 ms
