#### H200 moe_t48

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `dop08_ex_gate_up` | 3 | 182.4 | 0.547 | 44.3% | 417.43 | 87.0 | 0.48 | 7680×256 | latency |
| `dop10_ex_down` | 3 | 124.5 | 0.374 | 30.2% | 228.18 | 47.5 | 0.38 | 15360×256 | latency |
| `dop11_sh_gate` | 3 | 19.1 | 0.057 | 4.6% | 1.60 | 0.3 | 0.02 | 48×256 | latency |
| `dop12_sh_up` | 3 | 17.2 | 0.051 | 4.2% | 1.51 | 0.3 | 0.02 | 48×256 | latency |
| `dop06_sort_key` | 3 | 14.0 | 0.042 | 3.4% | 0.01 | 0.0 | 0.00 | 2×256 | latency |
| `dop07_sort_tok` | 3 | 12.1 | 0.036 | 2.9% | 0.01 | 0.0 | 0.00 | 2×256 | latency |
| `dop00_TensorId_13_` | 3 | 10.0 | 0.030 | 2.4% | 0.14 | 0.0 | 0.00 | 48×256 | latency |
| `dop14_sh_down` | 3 | 6.5 | 0.020 | 1.6% | 1.66 | 0.3 | 0.05 | 192×256 | latency |
| `dop05_TensorId_22_` | 3 | 5.5 | 0.017 | 1.3% | 0.01 | 0.0 | 0.00 | 2×256 | latency |
| `dop15_y` | 3 | 4.0 | 0.012 | 1.0% | 5.51 | 1.1 | 0.29 | 384×256 | launch |
| `dop09_ex_silu` | 3 | 2.8 | 0.008 | 0.7% | 2.63 | 0.5 | 0.20 | 960×256 | launch |
| `dop17_norm_ss` | 3 | 2.2 | 0.007 | 0.5% | 0.40 | 0.1 | 0.04 | 48×1024 | launch |
| `dop03_xq` | 3 | 2.0 | 0.006 | 0.5% | 0.56 | 0.1 | 0.06 | 384×256 | launch |
| `dop18_norm` | 3 | 1.9 | 0.006 | 0.5% | 0.61 | 0.1 | 0.07 | 384×256 | launch |
| `dop13_sh_silu` | 3 | 1.8 | 0.005 | 0.4% | 0.24 | 0.0 | 0.03 | 96×256 | launch |
| `dop16_resid` | 3 | 1.6 | 0.005 | 0.4% | 0.94 | 0.2 | 0.12 | 384×256 | launch |
| `dop19_xq_out` | 3 | 1.5 | 0.005 | 0.4% | 0.53 | 0.1 | 0.07 | 384×256 | launch |
| `dop01_TensorId_14_` | 3 | 1.0 | 0.003 | 0.2% | 0.01 | 0.0 | 0.00 | 2×256 | launch |
| `dop02_TensorId_15_` | 3 | 0.9 | 0.003 | 0.2% | 0.01 | 0.0 | 0.00 | 2×256 | launch |
| `dop04_TensorId_21_` | 3 | 0.9 | 0.003 | 0.2% | 0.01 | 0.0 | 0.00 | 2×256 | launch |

kernels 1.236 ms over 60 launches; span 1.331 ms, idle between kernels 0.096 ms (7% of span); by class: latency 1.174 ms, launch 0.062 ms
