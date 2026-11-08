#### H200 decode step, hand MoE, N=48 (wall 29.67 ms)

| group | ms | share of wall | floor MB | floor ms | light |
|---|---|---|---|---|---|
| moe | 16.533 | 56% | 28873 | 6.015 | 0.36 |
| dense | 6.087 | 21% | 2199 | 0.458 | 0.08 |
| gdn | 3.280 | 11% | 7247 | 1.510 | 0.46 |
| attn | 2.570 | 9% | 1628 | 0.339 | 0.13 |
| small | 1.256 | 4% | – | – | – |
| idle between kernels | 0.055 | 0% | | | |
| **step** | **29.670** | | 39947 | 8.322 | **0.28** |

858 launches; distinct experts a layer U(48) = 313.3

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `cx_moe_gu_q` | 48 | 187.7 | 9.008 | 30.3% | – | – | – | 16×256 | ? |
| `cx_moe_down_q` | 48 | 144.3 | 6.925 | 23.3% | – | – | – | 16×256 | ? |
| `cx_mma_q8x` | 96 | 41.8 | 4.015 | 13.5% | – | – | – | 192×128 | ? |
| `fl_gdn_conv_step_seg` | 36 | 86.0 | 3.096 | 10.4% | – | – | – | 32×512 | ? |
| `cx_attn_part` | 12 | 200.3 | 2.403 | 8.1% | – | – | – | 48×128 | ? |
| `cx_gemvkb2_t32` | 84 | 15.7 | 1.317 | 4.4% | – | – | – | 16×256 | ? |
| `cx_gemvkb2_t16` | 84 | 8.5 | 0.718 | 2.4% | – | – | – | 16×256 | ? |
| `cx_moe_add_norm` | 48 | 6.6 | 0.319 | 1.1% | – | – | – | 48×1024 | ? |
| `cx_mma_q8x_red` | 95 | 3.3 | 0.315 | 1.1% | – | – | – | 96×128 | launch |
| `cx_silu_q` | 48 | 6.5 | 0.313 | 1.1% | – | – | – | 16×32 | ? |
| `cx_route` | 48 | 6.0 | 0.287 | 1.0% | – | – | – | 48×32 | ? |
| `fe_q8` | 48 | 4.9 | 0.233 | 0.8% | – | – | – | 128×32 | launch |
| `cx_norm_q8` | 49 | 4.2 | 0.207 | 0.7% | – | – | – | 48×1024 | launch |
| `fl_gdn_conv_commit_seg` | 36 | 5.1 | 0.184 | 0.6% | – | – | – | 32×256 | ? |
| `cx_attn_merge` | 12 | 13.9 | 0.167 | 0.6% | – | – | – | 48×256 | ? |
| `cx_moe_plan_w` | 48 | 2.5 | 0.119 | 0.4% | – | – | – | 1×1024 | launch |
| `cx_attn_prep` | 12 | 3.5 | 0.042 | 0.1% | – | – | – | 48×256 | launch |
| `fe_argmax1` | 1 | 36.8 | 0.037 | 0.1% | – | – | – | 64×1024 | ? |
| `fe_copy` | 4 | 4.8 | 0.019 | 0.1% | – | – | – | 1×256 | launch |
| `fe_argmax2` | 1 | 1.5 | 0.002 | 0.0% | – | – | – | 48×32 | launch |

kernels 29.726 ms over 858 launches; span 29.762 ms, idle between kernels 0.055 ms (0% of span); by class: ? 28.788 ms, launch 0.937 ms
