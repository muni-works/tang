# GLM-5.3-Flash TCX: GPU validation plan (branch frog/glm5-tcx)

Script: `artifacts/glm5-tcx-pod.sh [stage ...]` (bash, `set -euo pipefail`, DONE / `FAIL <stage>`
trap into `$OUT/status`, summary in `$OUT/summary.md`). Ship the tree with the usual
`scripts/glm5-b300-bench.sh`-style md5-verified copy; builds run outside `~/tang-gpu.lock`, GPU runs
inside it.

| stage | what | pass bar | box | time |
|---|---|---|---|---|
| fetch | `hf download unsloth/GLM-5.3-Flash-GGUF --include '*UD-Q4_K_XL*'` with hf_transfer to `/root/glm53` (container disk, not /workspace) | 6 shards | any | ~15–35 min |
| build | `cargo build --release --features cuda` + test binaries | builds | any | ~20 min |
| kernels | `flash::glm5::moe` synthetic suite (tcx modes) + `real_layers_fast_matches_cpu` on layers 3,4,11,12,20,30,44 × T 1..64 | tcx rel l2 vs CPU ≤ 1e-3 (q8x2), routing sets = CPU, tcx-vs-g5f ≤ 2× fast-vs-naive; no NaN | **1× H200 ok** | ~30 min |
| gates | bench prompt `-n 128 / 512` greedy, plain vs `TCX=1 MIN_T=1` | first 8 ids = `785 1196 6801 264 6386 38038 911 279`; md5 7174cdd6 / 1453cf0e for plain; TCX md5 drift reported (tolerance path) | B300/B200 (or H200 with `TANG_GLM5_GPU_FFN_LAYERS=0-29`, ids only) | ~20 min |
| single | plain / tcx / MTP K=1 / MTP K=1+tcx at -n 128 and 512, ×2 | tcx within noise of plain at T=1 (MIN_T=2 keeps T=1 on g5f); report verify-window effect | B300/B200 | ~40 min |
| sweep | `--slots 1 4 8 16 32 64 --diverse`, plain vs tcx | aggregate tok/s up with B; slot 0 ids = oracle prefix | B300/B200 | ~45 min |
| prefill | ~2k / ~8k token prompts, plain vs tcx | report tok/s | B300/B200 | ~15 min |

Order on a 1× H200: `fetch build kernels` (+ `gates` with `GLM5_ENV`/BASE edited to add
`TANG_GLM5_GPU_FFN_LAYERS=0-29`). On a B300/B200: all stages.
Estimated GPU time: H200 ~1.5 h; B300 ~3 h (≈2 h if the H200 run already cleared the kernels).
If kernels fail: `TANG_GLM5_TCX_DOWN=0` isolates gate|up; `TANG_GLM5_MOE_ACT=q8` isolates the A2
MMAs; `TANG_GLM5_TCX_SGU=1` removes the K split.
