#### H200 moe_t1

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `dop02_ex_gate_up` | 3 | 10.7 | 0.032 | 26.4% | 14.68 | 3.1 | 0.29 | 320×256 | latency |
| `dop00_TensorId_13_` | 3 | 9.4 | 0.028 | 23.1% | 0.03 | 0.0 | 0.00 | 1×256 | latency |
| `dop03_ex_down` | 3 | 5.4 | 0.016 | 13.3% | 9.33 | 1.9 | 0.36 | 160×256 | latency |
| `dop05_sh_down` | 3 | 3.5 | 0.010 | 8.5% | 1.34 | 0.3 | 0.08 | 32×256 | launch |
| `dop04_sh_gate_up` | 3 | 3.1 | 0.009 | 7.5% | 2.63 | 0.5 | 0.18 | 64×256 | launch |
| `dop06_y` | 3 | 2.0 | 0.006 | 4.9% | 0.10 | 0.0 | 0.01 | 8×256 | launch |
| `dop08_norm_ss` | 3 | 2.0 | 0.006 | 4.9% | 0.02 | 0.0 | 0.00 | 1×1024 | launch |
| `dop01_xq` | 3 | 1.2 | 0.004 | 3.1% | 0.02 | 0.0 | 0.00 | 8×256 | launch |
| `dop09_norm` | 3 | 1.2 | 0.004 | 3.0% | 0.02 | 0.0 | 0.00 | 8×256 | launch |
| `dop10_xq_out` | 3 | 1.2 | 0.003 | 2.8% | 0.02 | 0.0 | 0.00 | 8×256 | launch |
| `dop07_resid` | 3 | 1.0 | 0.003 | 2.4% | 0.02 | 0.0 | 0.00 | 8×256 | launch |

kernels 0.122 ms over 33 launches; span 0.165 ms, idle between kernels 0.043 ms (26% of span); by class: latency 0.076 ms, launch 0.045 ms
