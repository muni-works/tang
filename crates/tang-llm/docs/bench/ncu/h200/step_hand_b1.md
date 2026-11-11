#### H200 decode step, hand MoE, N=1 (wall 4.04 ms)

| group | ms | share of wall | floor MB | floor ms | light |
|---|---|---|---|---|---|
| moe | 1.629 | 40% | 1077 | 0.224 | 0.14 |
| dense | 1.246 | 31% | 2199 | 0.458 | 0.37 |
| gdn | 0.239 | 6% | 151 | 0.031 | 0.13 |
| attn | 0.359 | 9% | 34 | 0.007 | 0.02 |
| small | 0.621 | 15% | – | – | – |
| idle between kernels | 0.052 | 1% | | | |
| **step** | **4.040** | | 3460 | 0.721 | **0.18** |

548 launches; distinct experts a layer U(1) = 10.0

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `cx_gemvk2_t1` | 145 | 8.6 | 1.246 | 30.4% | – | – | – | 3088×256 | ? |
| `cx_moe_down_d2f` | 48 | 14.1 | 0.679 | 16.6% | – | – | – | 128×256 | ? |
| `cx_moe_gu_d2` | 48 | 13.8 | 0.664 | 16.2% | – | – | – | 64×256 | ? |
| `cx_route` | 48 | 6.0 | 0.287 | 7.0% | – | – | – | 1×32 | ? |
| `cx_attn_part` | 12 | 22.6 | 0.272 | 6.6% | – | – | – | 1×128 | ? |
| `cx_moe_add_norm` | 48 | 5.4 | 0.261 | 6.4% | – | – | – | 1×1024 | ? |
| `fl_gdn_conv_step_seg` | 36 | 6.6 | 0.239 | 5.8% | – | – | – | 32×512 | ? |
| `cx_norm_q8` | 49 | 4.0 | 0.194 | 4.7% | – | – | – | 1×1024 | launch |
| `cx_attn_merge` | 12 | 7.3 | 0.087 | 2.1% | – | – | – | 1×256 | ? |
| `fe_q8` | 48 | 1.4 | 0.065 | 1.6% | – | – | – | 128×32 | launch |
| `fl_gdn_conv_commit_seg` | 36 | 1.5 | 0.053 | 1.3% | – | – | – | 32×256 | launch |
| `cx_attn_prep` | 12 | 2.7 | 0.033 | 0.8% | – | – | – | 1×256 | launch |
| `fe_copy` | 4 | 2.4 | 0.010 | 0.2% | – | – | – | 1×256 | launch |
| `fe_argmax1` | 1 | 4.1 | 0.004 | 0.1% | – | – | – | 64×1024 | launch |
| `fe_argmax2` | 1 | 1.5 | 0.002 | 0.0% | – | – | – | 1×32 | launch |

kernels 4.095 ms over 548 launches; span 4.145 ms, idle between kernels 0.052 ms (1% of span); by class: ? 3.734 ms, launch 0.360 ms
