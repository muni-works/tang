#### H200 decode step, der MoE, N=1 (wall 4.39 ms)

| group | ms | share of wall | floor MB | floor ms | light |
|---|---|---|---|---|---|
| moe | 1.297 | 30% | 1077 | 0.224 | 0.17 |
| dense | 1.151 | 26% | 2199 | 0.458 | 0.40 |
| gdn | 0.227 | 5% | 151 | 0.031 | 0.14 |
| attn | 0.355 | 8% | 34 | 0.007 | 0.02 |
| small | 1.107 | 25% | – | – | – |
| idle between kernels | 0.363 | 8% | | | |
| **step** | **4.390** | | 3460 | 0.721 | **0.16** |

830 launches; distinct experts a layer U(1) = 10.0

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `cx_gemvk2_t1` | 145 | 7.9 | 1.151 | 27.8% | – | – | – | 3088×256 | ? |
| `dop02_ex_gate_up` | 47 | 11.1 | 0.524 | 12.7% | – | – | – | 320×256 | ? |
| `dop00_TensorId_13_` | 47 | 9.4 | 0.443 | 10.7% | – | – | – | 1×256 | ? |
| `dop03_ex_down` | 47 | 6.2 | 0.293 | 7.1% | – | – | – | 160×256 | ? |
| `cx_attn_part` | 12 | 22.3 | 0.267 | 6.5% | – | – | – | 1×128 | ? |
| `fl_gdn_conv_step_seg` | 36 | 6.3 | 0.227 | 5.5% | – | – | – | 32×512 | ? |
| `cx_norm_q8` | 49 | 3.9 | 0.193 | 4.7% | – | – | – | 1×1024 | launch |
| `dop05_sh_down` | 47 | 3.8 | 0.177 | 4.3% | – | – | – | 32×256 | launch |
| `dop04_sh_gate_up` | 47 | 3.6 | 0.169 | 4.1% | – | – | – | 64×256 | launch |
| `dop06_resid` | 47 | 2.1 | 0.099 | 2.4% | – | – | – | 8×256 | launch |
| `dop07_norm_ss` | 47 | 2.1 | 0.099 | 2.4% | – | – | – | 1×1024 | launch |
| `cx_attn_merge` | 12 | 7.3 | 0.087 | 2.1% | – | – | – | 1×256 | ? |
| `dop08_norm` | 47 | 1.6 | 0.077 | 1.9% | – | – | – | 8×256 | launch |
| `fe_q8` | 48 | 1.3 | 0.065 | 1.6% | – | – | – | 128×32 | launch |
| `dop09_xq_out` | 47 | 1.3 | 0.062 | 1.5% | – | – | – | 64×256 | launch |
| `dop01_xq` | 47 | 1.3 | 0.061 | 1.5% | – | – | – | 8×256 | launch |
| `fl_gdn_conv_commit_seg` | 36 | 1.5 | 0.052 | 1.3% | – | – | – | 32×256 | launch |
| `cx_attn_prep` | 12 | 2.6 | 0.032 | 0.8% | – | – | – | 1×256 | launch |
| `cx_moe_down_d2f` | 1 | 16.8 | 0.017 | 0.4% | – | – | – | 128×256 | ? |
| `cx_moe_gu_d2` | 1 | 13.8 | 0.014 | 0.3% | – | – | – | 64×256 | ? |
| `fe_copy` | 4 | 2.5 | 0.010 | 0.2% | – | – | – | 1×256 | launch |
| `cx_route` | 1 | 6.0 | 0.006 | 0.1% | – | – | – | 1×32 | ? |
| `cx_moe_add_norm` | 1 | 5.8 | 0.006 | 0.1% | – | – | – | 1×1024 | ? |
| `fe_argmax1` | 1 | 3.6 | 0.004 | 0.1% | – | – | – | 64×1024 | launch |
| `fe_argmax2` | 1 | 1.5 | 0.002 | 0.0% | – | – | – | 1×32 | launch |

kernels 4.137 ms over 830 launches; span 4.500 ms, idle between kernels 0.363 ms (8% of span); by class: ? 3.035 ms, launch 1.101 ms
