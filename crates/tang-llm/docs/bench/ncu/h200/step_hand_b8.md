#### H200 decode step, hand MoE, N=8 (wall 9.83 ms)

| group | ms | share of wall | floor MB | floor ms | light |
|---|---|---|---|---|---|
| moe | 5.402 | 55% | 7009 | 1.460 | 0.27 |
| dense | 2.546 | 26% | 2199 | 0.458 | 0.18 |
| gdn | 0.583 | 6% | 1208 | 0.252 | 0.43 |
| attn | 0.708 | 7% | 271 | 0.057 | 0.08 |
| small | 0.687 | 7% | – | – | – |
| idle between kernels | 0.034 | 0% | | | |
| **step** | **9.830** | | 10687 | 2.226 | **0.23** |

548 launches; distinct experts a layer U(8) = 74.7

| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |
|---|---|---|---|---|---|---|---|---|---|
| `cx_moe_down_d4f` | 48 | 56.7 | 2.724 | 27.4% | – | – | – | 64×256 | ? |
| `cx_gemvk2_t8` | 145 | 17.5 | 2.539 | 25.6% | – | – | – | 3088×256 | ? |
| `cx_moe_gu_d4` | 48 | 49.8 | 2.388 | 24.1% | – | – | – | 32×256 | ? |
| `cx_attn_part` | 12 | 51.2 | 0.615 | 6.2% | – | – | – | 8×128 | ? |
| `fl_gdn_conv_step_seg` | 36 | 16.2 | 0.583 | 5.9% | – | – | – | 32×512 | ? |
| `cx_moe_add_norm` | 48 | 6.2 | 0.300 | 3.0% | – | – | – | 8×1024 | ? |
| `cx_route` | 48 | 6.0 | 0.290 | 2.9% | – | – | – | 8×32 | ? |
| `cx_norm_q8` | 49 | 4.1 | 0.202 | 2.0% | – | – | – | 8×1024 | launch |
| `cx_attn_merge` | 12 | 7.8 | 0.093 | 0.9% | – | – | – | 8×256 | ? |
| `fe_q8` | 48 | 1.6 | 0.079 | 0.8% | – | – | – | 128×32 | launch |
| `fl_gdn_conv_commit_seg` | 36 | 1.7 | 0.060 | 0.6% | – | – | – | 32×256 | launch |
| `cx_attn_prep` | 12 | 2.8 | 0.034 | 0.3% | – | – | – | 8×256 | launch |
| `fe_copy` | 4 | 2.8 | 0.011 | 0.1% | – | – | – | 1×256 | launch |
| `fe_argmax1` | 1 | 7.4 | 0.007 | 0.1% | – | – | – | 64×1024 | ? |
| `fe_argmax2` | 1 | 1.5 | 0.002 | 0.0% | – | – | – | 8×32 | launch |

kernels 9.927 ms over 548 launches; span 9.949 ms, idle between kernels 0.034 ms (0% of span); by class: ? 9.539 ms, launch 0.388 ms
