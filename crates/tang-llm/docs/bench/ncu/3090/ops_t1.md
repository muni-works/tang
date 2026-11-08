#### ops_t1

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dop00_gemv_q8_0` | 12 | 144.2 | 1.730 | 88.1% | 119.35 | 127.5 | 0.88 | 25 | 70 | 256×256 | traffic |
| `dop00_route_topk` | 4 | 11.8 | 0.047 | 2.4% | 0.12 | 0.1 | 0.01 | 0 | 1 | 1×256 | latency |
| `cx_route` | 4 | 11.2 | 0.045 | 2.3% | 0.10 | 0.1 | 0.01 | 0 | 1 | 1×32 | latency |
| `cx_norm_q8` | 4 | 8.9 | 0.036 | 1.8% | 0.12 | 0.1 | 0.01 | 0 | 1 | 1×256 | latency |
| `cx_moe_add_norm` | 4 | 6.7 | 0.027 | 1.4% | 0.13 | 0.1 | 0.02 | 1 | 2 | 1×1024 | latency |
| `dop02_norm_ss` | 4 | 3.2 | 0.013 | 0.6% | 0.04 | 0.0 | 0.01 | 0 | 1 | 1×256 | launch |
| `dop01_norm_ss` | 4 | 3.1 | 0.012 | 0.6% | 0.01 | 0.0 | 0.01 | 0 | 1 | 1×256 | launch |
| `dop00_combine_norm` | 4 | 2.6 | 0.010 | 0.5% | 0.10 | 0.1 | 0.04 | 1 | 4 | 8×256 | launch |
| `dop02_add_norm` | 4 | 2.4 | 0.010 | 0.5% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |
| `dop03_combine_norm` | 4 | 2.4 | 0.010 | 0.5% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |
| `dop01_route_wts` | 4 | 2.2 | 0.009 | 0.4% | 0.04 | 0.0 | 0.02 | 0 | 2 | 1×256 | launch |
| `dop00_add_norm` | 4 | 2.1 | 0.008 | 0.4% | 0.04 | 0.0 | 0.02 | 0 | 2 | 8×256 | launch |
| `dop01_combine_norm` | 4 | 2.0 | 0.008 | 0.4% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |

kernel time 1.964 ms, traffic floor 1.533 ms (light 0.78); by class: traffic 1.730 ms (88%), latency 0.154 ms (8%), launch 0.080 ms (4%)
