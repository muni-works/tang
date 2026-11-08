#### ops_t8

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dop01_gemv_q8_0` | 12 | 169.9 | 2.039 | 88.5% | 130.96 | 139.9 | 0.82 | 18 | 64 | 64×256 | traffic |
| `dop00_route_topk` | 4 | 11.5 | 0.046 | 2.0% | 0.05 | 0.1 | 0.00 | 3 | 0 | 8×256 | latency |
| `cx_route` | 4 | 11.2 | 0.045 | 1.9% | 0.03 | 0.0 | 0.00 | 1 | 0 | 8×32 | latency |
| `cx_norm_q8` | 4 | 8.8 | 0.035 | 1.5% | 0.33 | 0.4 | 0.04 | 1 | 4 | 8×256 | latency |
| `dop00_xq` | 12 | 2.4 | 0.029 | 1.2% | 0.12 | 0.1 | 0.06 | 6 | 6 | 64×256 | launch |
| `cx_moe_add_norm` | 4 | 6.8 | 0.027 | 1.2% | 1.09 | 1.2 | 0.17 | 4 | 18 | 8×1024 | latency |
| `dop00_combine_norm` | 4 | 3.3 | 0.013 | 0.6% | 0.75 | 0.8 | 0.25 | 6 | 25 | 64×256 | launch |
| `dop01_norm_ss` | 4 | 3.1 | 0.013 | 0.5% | 0.07 | 0.1 | 0.02 | 1 | 3 | 8×256 | launch |
| `dop02_norm_ss` | 4 | 3.1 | 0.012 | 0.5% | 0.09 | 0.1 | 0.03 | 1 | 3 | 8×256 | launch |
| `dop02_add_norm` | 4 | 2.5 | 0.010 | 0.4% | 0.08 | 0.1 | 0.04 | 3 | 4 | 64×256 | launch |
| `dop03_combine_norm` | 4 | 2.4 | 0.010 | 0.4% | 0.08 | 0.1 | 0.04 | 3 | 4 | 64×256 | launch |
| `dop01_combine_norm` | 4 | 2.2 | 0.009 | 0.4% | 0.14 | 0.2 | 0.07 | 3 | 7 | 64×256 | launch |
| `dop00_add_norm` | 4 | 2.2 | 0.009 | 0.4% | 0.14 | 0.1 | 0.07 | 3 | 7 | 64×256 | launch |
| `dop01_route_wts` | 4 | 2.1 | 0.008 | 0.4% | 0.01 | 0.0 | 0.00 | 0 | 0 | 1×256 | launch |

kernel time 2.304 ms, traffic floor 1.693 ms (light 0.73); by class: traffic 2.039 ms (88%), latency 0.153 ms (7%), launch 0.112 ms (5%)
