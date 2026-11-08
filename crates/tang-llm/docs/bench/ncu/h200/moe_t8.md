#### H200 moe_t8

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `dop02_ex_gate_up` | 3 | 59.4 | 0.178 | 42.7% | 100.71 | 21.0 | 0.35 | 2560×256 | latency |
| `dop04_ex_down` | 3 | 23.6 | 0.071 | 17.0% | 70.39 | 14.7 | 0.62 | 1280×256 | traffic |
| `dop05_sh_gate` | 3 | 12.3 | 0.037 | 8.9% | 1.34 | 0.3 | 0.02 | 32×256 | latency |
| `dop06_sh_up` | 3 | 11.8 | 0.035 | 8.4% | 1.53 | 0.3 | 0.03 | 32×256 | latency |
| `dop00_TensorId_13_` | 3 | 9.5 | 0.028 | 6.8% | 0.06 | 0.0 | 0.00 | 8×256 | latency |
| `dop08_sh_down` | 3 | 8.5 | 0.025 | 6.1% | 1.41 | 0.3 | 0.03 | 32×256 | latency |
| `dop09_y` | 3 | 2.5 | 0.007 | 1.8% | 0.82 | 0.2 | 0.07 | 64×256 | launch |
| `dop03_ex_silu` | 3 | 2.4 | 0.007 | 1.7% | 0.39 | 0.1 | 0.03 | 160×256 | launch |
| `dop11_norm_ss` | 3 | 2.1 | 0.006 | 1.5% | 0.08 | 0.0 | 0.01 | 8×1024 | launch |
| `dop07_sh_silu` | 3 | 1.6 | 0.005 | 1.2% | 0.05 | 0.0 | 0.01 | 16×256 | launch |
| `dop01_xq` | 3 | 1.6 | 0.005 | 1.1% | 0.09 | 0.0 | 0.01 | 64×256 | launch |
| `dop10_resid` | 3 | 1.4 | 0.004 | 1.0% | 0.14 | 0.0 | 0.02 | 64×256 | launch |
| `dop12_norm` | 3 | 1.4 | 0.004 | 1.0% | 0.08 | 0.0 | 0.01 | 64×256 | launch |
| `dop13_xq_out` | 3 | 1.2 | 0.004 | 0.9% | 0.11 | 0.0 | 0.02 | 64×256 | launch |

kernels 0.417 ms over 42 launches; span 0.481 ms, idle between kernels 0.063 ms (13% of span); by class: latency 0.304 ms, traffic 0.071 ms, launch 0.042 ms
