#### moe_t1

| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dop02_ex_gate_up` | 3 | 22.1 | 0.066 | 29.1% | 14.68 | 15.7 | 0.71 | 8 | 73 | 320×256 | traffic |
| `dop03_ex_down` | 3 | 14.7 | 0.044 | 19.4% | 9.33 | 10.0 | 0.68 | 11 | 70 | 160×256 | traffic |
| `dop00_TensorId_13_` | 3 | 11.9 | 0.036 | 15.7% | 0.03 | 0.0 | 0.00 | 0 | 0 | 1×256 | latency |
| `dop04_sh_gate_up` | 3 | 5.8 | 0.017 | 7.7% | 2.63 | 2.8 | 0.48 | 4 | 50 | 64×256 | latency |
| `dop05_sh_down` | 3 | 5.5 | 0.017 | 7.3% | 1.34 | 1.4 | 0.26 | 4 | 27 | 32×256 | latency |
| `dop08_norm_ss` | 3 | 3.7 | 0.011 | 4.9% | 0.02 | 0.0 | 0.00 | 0 | 0 | 1×1024 | launch |
| `dop06_y` | 3 | 3.1 | 0.009 | 4.1% | 0.10 | 0.1 | 0.03 | 1 | 4 | 8×256 | launch |
| `dop09_norm` | 3 | 2.4 | 0.007 | 3.1% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |
| `dop01_xq` | 3 | 2.3 | 0.007 | 3.1% | 0.02 | 0.0 | 0.01 | 1 | 1 | 8×256 | launch |
| `dop10_xq_out` | 3 | 2.3 | 0.007 | 3.0% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |
| `dop07_resid` | 3 | 2.0 | 0.006 | 2.7% | 0.02 | 0.0 | 0.01 | 0 | 1 | 8×256 | launch |

kernel time 0.228 ms, traffic floor 0.090 ms (light 0.40); by class: traffic 0.110 ms (49%), latency 0.070 ms (31%), launch 0.048 ms (21%)
