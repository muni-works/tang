# Porting the coder-next speed wins to GLM-5.3-Flash and DeepSeek v4.1 flash (2026-11-05)

Source of the wins: tang `frog/integrate-2026-11-03` @ 99f8c6f9 (+ 406c4a91), results in
`docs/plans/gates-2026-11-04.md` ("Round 2 integration") and `crates/tang-llm/docs/speed-log.md`.
Mac-only phase: everything below that says "measured" is a prior pod number; every new number is
an estimate until the GPU plan in `artifacts/GPU-VALIDATION.md` (branch `frog/glm5-tcx`) runs.

## The one fact that sets the order

All coder wins are **batch / prefill** wins. TCX left single-stream unchanged (B=64 38.9 → 35.2 ms
per step); wide prefill and the serve layer only matter with many rows. GLM today is a
single-stream engine (100–114 tok/s plain, 129.6 MTP K=1 on a B300) whose decode token is ~10 ms,
of which the `g5f_*` MoE family is ~7.4 ms against a ~0.45 ms bytes floor (NOTES "underfilled
grids for t = 1"). At T = 1–4 every routed expert has ~1 row (≈80 % distinct at a verify window),
so the MMA's 8-column n-tile is 1/8 full. **So for GLM: TCX/wide/serve gains arrive with the
batcher, not with single-stream.** DeepSeek is already batched (8× H100 EP, 147 tok/s greedy,
spec), so its gains land sooner, but its experts are fp4, not GGUF K-quants.

## GLM-5.3-Flash (`flash/glm5/`, branch `frog/glm5-forward` c2fb65af)

| coder win | verdict | coder side | GLM side | expected gain (from the last profile) |
|---|---|---|---|---|
| TCX GGUF-native expert GEMMs (`cx_moe_gu_q` / `cx_moe_down_q`, k-permuted 8-byte qs loads, mins as 2nd MMA, per-32 acts, K split 2/1) | **adapt — done on `frog/glm5-tcx`, off by default** | `coder_kernels.rs` `gg_load` / `gg_rows` / `moe_gg_body`, `cx_moe_plan_w`; launch `coder.rs` ~4329 (`var.tcx && in_batch`) | new `glm5/moe_tcx.cuh` `g5x_plan`, `g5x_gu_q[2]`, `g5x_down_q[2]`, `g5x_silu`; `moe_gpu.rs` `ffn_tcx` / `tcx_fits`, hooked in `ffn_fast` after routing | T=1: ~0 (coder: unchanged). Verify windows K=1–3: ±0, maybe −; measure. Batch B≥16 on the batcher: the g5f path re-reads each expert per row; TCX reads it once per 8 rows → MoE ms/step should scale ~B/8 instead of ~B once experts repeat; coder got −10 % step time at B=64 with an already-grouped baseline, GLM's baseline is not grouped so expect more (est. 1.5–2.5× MoE at B=32–64). |
| TCX tile repack (`flash/tcx.rs`) | skip | `tcx.rs`, `TANG_CODER_TCX_TILES` | — | off by default on coder (single-stream −5 %); not worth the upload-path churn |
| Wide router `cx_sgemm_wide2` (87 → 34 ms, bitwise) | adapt (later) | `coder_kernels.rs` `cx_sgemm_wide2[s]` + `Seg` | `moe_cuda.cuh` `g5f_route` / `g5f_router` (F32 router 256×6144, fused q8+select) | router is 27 µs/layer at T=1; matters only for prefill windows of 1k+ rows (est. a few % of prefill) |
| Staged GEMM token tiles + cp.async, z=1 expert grid, 2048-row windows | adapt (after batcher) | `coder_kernels.rs` `cx_moe_gemm_gu` / `_down`, `cx_mma_q8x`; `coder.rs` `in_wide` windows | GLM prefill = `resident.rs` windows of `TANG_GLM5_KDA_MAXT` rows (64) + `g5f_gemv` (`G5F_TT` 8 tokens per weight read) | GLM prefill is unmeasured at length; KDA and DSA prefill are their own work (GDN chunking doesn't apply). The expert half: the same 4k → 7.6k-class jump needs wide windows first |
| Serve layer (batch serve loop, urgent prefill share, wide slice 1024, disk save off thread, MTP-fill gating, D-Cut spec) | adapt | `server.rs` / `coder.rs` serve loop (`frog/coder-serve2`, `-serve3` @ 21e5a49f) | `glm5/serve_glue.rs` (`max_slots`, `window_batch`, `save/restore_running_at`), batcher (merged, `TANG_GLM5_SLOTS`) | coder C=64: 987 → 1,287 tok/s (+30 %). GLM: mostly glue reuse once its batch decode is fast; MTP gating applies (GLM MTP K=1 is 1.29×) |
| GDN kernels / fa2 | **does not apply** | GDN is Qwen3-Next only | GLM has KDA (34 layers) + NoPE-MLA/DSA (11 layers) | KDA has its own k5d path; MLA needs its own batched attend |
| Gate discipline (T≤8 bitwise, batch tolerance ≥99 % agree & mean |Δlogit| ≤ 0.2, KL vs jitter control) | **ports as-is** | `flash-batch-check`, `flash-wide-check` | GLM gates are oracle ids + md5 (7174cdd6 @128 / 1453cf0e @512) + KL ≤ 1e-3; TCX is tolerance-path so the md5 may drift late — gate TCX on first-8 = oracle + per-layer rel l2 ≤ 1e-3 + batch tolerance bars | — |

What was ported on `frog/glm5-tcx` (commit c15151ad):
- `g5x_rows` is coder's `gg_rows` with two adaptations: runtime K (GLM shapes are not compile-time
  defines) and **A2** — GLM's two-level activations (`g5f_q8` codes + residual codes,
  x = d (q + q2/254), the reason the K4 rewrite failed the KL bar was activation error, not order)
  as two more MMAs, keeping the q8x2 ~1e-5 band. `TANG_GLM5_MOE_ACT=q8` uses the one-level kernel.
- Pairs (t, slot) keep GLM's `t*S + j` row index, so h / hq / yp are the g5f layouts: the
  Q6_K-down layers keep the g5f split down on the TCX h (`down.ty ∉ {Q4_K,Q5_K}`), shared-expert
  non-K-quant rows take `g5f_dot_any` per row inside the same launch.
- Fixed grid `min(n, ne + 2 + n/8)` groups → graph-capturable.
- Flags: `TANG_GLM5_TCX=1` (default 0), `TANG_GLM5_TCX_MIN_T` (2), `TANG_GLM5_TCX_DOWN` (1),
  `TANG_GLM5_TCX_SGU` / `_SDN` (2 / 1).
- CPU tests `flash::glm5::moe::tcx_ref` emulate the m16n8k32 fragments lane by lane (Q4_K, Q5_K,
  m = 1..8, A2 on/off) against `gguf::dequantize` dots, plus the plan grouping. The GPU tests
  (`check2`, `real_layers_fast_matches_cpu`) gained `tcx-*` modes and a `tcx-vs-g5f` line.
- Not wired: wide router, staged prefill tiles (not cheap: GLM has no wide-window path yet).

GLM order of work:
1. GPU-validate TCX kernels per layer (1× H200 is enough, see below) and the ids gate.
2. Make the batcher's decode window the target: `--slots 8/32/64` sweep with TCX on; then raise
   `TANG_GLM5_KDA_MAXT` and batch the DSA attend (MLA work, not a coder port).
3. Serve layer: lift coder's batch loop / urgent prefill share / MTP-fill gating into
   `serve_glue.rs`.
4. Wide windows for prefill (2048 rows) + staged expert GEMM tiles + wide router.
5. Single-stream stays on the round-9 path; its next rungs are in NOTES (cheaper verify rows,
   auto K), not coder ports.

## DeepSeek v4.1 flash (`flash/deepseek/`)

State: 8× H100 EP, greedy 147 tok/s spec (wave 7b), prefill 460–475 tok/s; the WIP 35066e91
(shared expert spread across EP ranks) is reverted on `frog/integrate-launch-fixes` (57af86ec).
Verify-step profile (direct path, 32.7 ms): `layer.moe` 11.8 ms (router 2.0, route 0.8,
grouped_gu 3.5, grouped_down 1.7, reduce 2.3, shared 2.9), `layer.attn` 12.8, `layer.hc` 4.3.

| coder win | verdict | DeepSeek side | expected gain |
|---|---|---|---|
| TCX expert GEMMs | **adapt (new dequant)**: experts are fp4 (e2m1 + per-32 e8m0 scale), shared fp8. 2× e2m1 values are integers in [−12, 12], so the same int8 m16n8k32 MMA works on a nibble→int8 LUT (`__byte_perm`), scale = e8m0 · ½ in the epilogue, no mins MMA. Grouping already exists (`ds_moe_plan_tiles`, 8-row tiles) | `kernels.cu` grouped MoE (~1565, 1888–1970, 2145+): replace the per-tile dot with a `ds_tcx_rows` | grouped_gu+down 5.2 ms of 32.7 → est. 2.5–3.5 ms at the 5-row verify window (rows per expert per rank are small; the gain is the barrier/staging removal coder saw) ≈ −6–8 % step |
| Wide router | adapt | `ds` router (2.0 ms/verify = launch-heavy, 43 layers) | fuse with route (0.8): est. −1.5 ms/step |
| Staged prefill tiles / z=1 expert grid | adapt | prefill grouped MoE + GEMM tiles (wave 6) | prefill 460 → est. 700–900 tok/s (coder 1.9×; EP all-reduce caps it) |
| Serve layer | ports mostly as-is | `deepseek/serve.rs` | multi-user throughput; the urgent-prefill share matters most (prefill is slow) |
| MLA / compressed attention | does not apply | `layer.attn` 12.8 ms is the biggest line | its own work (attn3 lane) |
| Gate discipline | as-is | flash-batch-check/tcheck/spec-test already run for DS | — |

DeepSeek order: (1) fp4 TCX in the grouped MoE (biggest isolated kernel win, gateable per
kernel on the EP tests), (2) router+route fusion, (3) prefill tiles, (4) serve layer.
Needs ≥ 4 GPUs (272 GB of experts): no Mac/1-GPU end-to-end; kernel tests can run on one GPU with
`ep_two_ranks_one_device_match`.

## Single-GPU fit and the GPU ask

- GLM UD-Q4_K_XL = 186 GB. One B300 (288 GB) or B200 (192 GB, tight: weights + KV + scratch ≈ at
  the limit) runs the all-resident 100+ tok/s path. 1× H200 (141 GB) cannot.
- **2 GPUs today:** `TANG_GPUS=2` exists only on the per-call FFN path (`engine.rs` `GpuFfn`:
  layers [0, split) on GPU 0, the rest on GPU 1, activations through the host). The resident /
  whole-step-graph / MTP path that gives 100–130 tok/s is single-device only (NOTES: "single
  device only (`TANG_GPUS=2` keeps the per-call path)"). So 2× H200 runs GLM correctly but at
  per-call speed (tens of tok/s), useless for speed numbers.
- **Adding 2-GPU resident:** pipeline split (layers by device, one 6144-f32 residual × mHC
  streams copied peer-to-peer at the cut, one graph per device, MTP on the last device):
  ~3–5 days. Expert parallel like DeepSeek's `ep.rs` (experts `e % 2`, NCCL all-reduce of the
  partial y per layer, everything else replicated: ~2× 30 GB dense replicated fits): ~1–2 weeks,
  and it adds an all-reduce per layer to a 10 ms token (45 × ~15 µs ≈ 0.7 ms).
- **Smaller quant for one H200:** UD-Q2_K_XL / IQ2-class (~2.6–2.8 bpw, ~110–120 GB) would fit,
  but its experts are Q2_K/Q3_K/IQ — TCX only covers Q4_K/Q5_K, so it validates nothing of this
  port. Not recommended.
- **What 1× H200 *can* validate today:** the per-layer kernel gate (`real_layers_fast_matches_cpu`
  uploads one layer at a time; it was written for 24 GB cards) on the real UD-Q4_K_XL weights,
  and end-to-end ids with `TANG_GLM5_GPU_FFN_LAYERS=0-29` (the layers past 29 on CPU; used on
  the H200 before) — correctness, not speed.

Recommendation: **1× H200 now** for kernels + ids (~1.5 h: 35 min download, 20 min build,
30 min tests), then **1× B300 (or B200)** when stock returns for single-stream before/after,
B-sweep and prefill (~3 h incl. download/build). 2× H200 only pays if we first build the 2-GPU
resident split.
