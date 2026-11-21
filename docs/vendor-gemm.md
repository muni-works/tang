# The weight path and the MMA target, per machine

2026-11-12. Scoping only: nothing here is implemented and no kernel was run. Everything below is
derived from tang's source at origin/main `2b7ff9eb` plus vendor documentation; where only the GPU
can answer, the exact command is given and marked **[needs GPU]**.

**Two machines, and this doc does not blur them:**

| | **prod** | **bench** |
|---|---|---|
| pod | `q6pluwjkklulmt` (`frog-cloud-6000`) | H200 pods, `w2myvdc7bi8mke` and friends |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA H200 SXM |
| compute capability | **12.0 (`sm_120`)** — GB202, workstation Blackwell | 9.0 (`sm_90a`) — Hopper |
| SMs / memory / bandwidth | 188 / 96 GB GDDR7 / 1,597 GB/s | 132 / 141 GB HBM3e / 4.8 TB/s |
| dense 8-bit tensor rate | **~0.9–1.0 PFLOP/s** (int8 = fp8) | ~1.98 PFLOP/s (int8 = fp8) |
| price | $2.69/hr secure ($2.49 on this pod) | $5.29/hr |
| what it serves | coder-next Q4_K_XL, 131k ctx, batch 32, wide 4096 | benchmarks only |

The `43k tok/s` vLLM figure in `docs/vllm-tricks.md` is an **H200** number. Prod's own GEMM
ceiling is roughly **half** that (see §4), so the 43k baseline has never applied to the machine
that earns revenue, and neither does the trick ranked against it.

Note also: **the card's memory size is disputed, and command A settles it.** NVIDIA's spec sheet,
RunPod's GPU catalogue and tang's own bench logs on this exact GPU
(`bench/results/derived-race/rtx-pro-6000.txt`: "NVIDIA RTX PRO 6000 Blackwell Server Edition,
595.91.07, 97887 MiB") all say **96 GB**. The pod's own metadata reports
`"gpu": { "memory": 188 }` — which is the card's **SM count**, not its memory: `list-pods` returns
188 for `q6pluwjkklulmt` while the same GPU in `list-gpu-types` is `"memory": 96`. The handoff's
"188 GB" almost certainly came from that field. This does not change the recommendation below,
which rests on arithmetic rate rather than capacity, but it does change how much room an fp8
checkpoint would leave (§2.2a), so confirm it before quoting either number.

## 1. What MMA instructions the 6000 actually has

### 1.1 The instruction set

`sm_120` is **not** a Hopper part and **not** a datacenter-Blackwell part. It is its own family:

| instruction family | sm_90a (H200) | sm_100 (B200) | **sm_120 (prod)** |
|---|---|---|---|
| `mma.sync` m16n8k32 `.s8.s8.s32` (what tang runs) | yes | yes | **yes** |
| `mma.sync` m16n8k32 `.e4m3.e4m3.f32` (fp8) | yes (PTX 8.4) | yes | **yes** (PTX 8.4, needs sm_89+) |
| `mma.sync` `…k16.f32.e4m3` f16-accum | — | — | yes (PTX 8.7) |
| `mma.sync` `.kind::mxf8f6f4` / `.kind::mxf4nvf4` **block-scaled** | — | — | **yes, but requires `sm_120a`** |
| `wgmma.mma_async` (Hopper warpgroup) | **yes** | yes | **no** |
| `tcgen05.mma` + TMEM (datacenter Blackwell) | — | yes | **no** |
| thread-block clusters > 1 | yes | yes | no |
| shared memory per block | 227 KiB | 228 KiB | **99 KiB** |

The `wgmma` row is the one that decides this whole scoping. PTX ISA 9.5 lists `wgmma.mma_async`,
`wgmma.fence`, `wgmma.commit_group` and `wgmma.wait_group` as "Requires `sm_90a`", and
architecture-specific (`a`) targets are documented as not forward-compatible. Compiling Hopper
`wgmma` for `sm_120` is reported to fail with `ptxas` error *"Instruction 'wgmma.fence' not
supported on .target 'sm_120'"*. One secondary source (the `0xSero/blackwell-gpu-wiki` feature
matrix) claims `wgmma` works on SM12.0 at lower throughput; that contradicts the PTX ISA and the
reported `ptxas` failure, and I am going with the primary. **[needs GPU]** settles it in one
command, §1.3.

**Consequence for trick #3.** `docs/vllm-tricks.md` §3 estimates "2-3× on the GEMM share (wgmma
fp8 ~1000+ TFLOP/s vs int8 mma.sync tiles at ~100s)". The entire gain in that estimate is
**wgmma versus mma.sync**, not fp8 versus int8. On prod there is no wgmma, so there is nothing to
switch to.

### 1.2 The fp8 rate is the int8 rate on this chip

Microbenchmarks on GB202 (`zartbot`'s SM_120 reference, RTX PRO 5000, 110 SMs) put the dense
peaks at:

| format | measured TFLOP/s (110 SMs) | scaled to 188 SMs [A] |
|---|---|---|
| BF16 / FP16 | 271 | ~460 |
| **FP8** | **529** | **~900** |
| **INT8** | **526** | **~900** |
| MX-FP4 | 1,069 | ~1,830 |
| NV-FP4 | 1,012 | ~1,730 |

FP8 and INT8 are the same rate to within 0.6%, which the source attributes to both running the
same `QMMA.16832` 8-bit path. NVIDIA's headline "2 PFLOPS FP8" is the sparse figure; dense is
~1 PFLOP/s, consistent with the scaled measurement.

So on prod, **switching weights from int8 to fp8 buys 0% of arithmetic throughput**, and it makes
the weights *bigger*: Q4_K is 4.5 bits/weight, fp8 is 8 bits. The only thing fp8 buys is a cheaper
epilogue (no per-sub-block `d`/`dmin` correction) worth single-digit percent, against 1.8× the
weight bytes.

**The Blackwell-shaped version of trick #3 is MXFP4, not fp8**: same ~4.25 bits/weight as Q4_K,
and 2× the 8-bit rate. That needs `sm_120a` and a new tile (§3.3).

### 1.3 What the code says about arch dispatch — and three live bugs

The arch string is built in `crates/tang-gpu/src/cuda.rs:186`:

```rust
let suffix = if major == 9 { "a" } else { "" };
```

On prod that yields **`sm_120`**, not `sm_120a`. Every `sm_120a`-only instruction — including the
block-scaled `mma.sync` that would make MXFP4 reachable — is therefore unassemblable today. This
is the first thing to fix (§5 step 2), and it is a one-line change plus a retest.

The Hopper gates themselves are string-exact, which is why prod has not fallen over:
`crates/tang-gpu/src/program.rs:1847,1892` gate the `SM90_MMA_SHAPES` / `SM90_FLASH_SHAPES`
variants on `d.arch == "sm_90a"`, so on prod the tensor-IR tuner offers only the portable
`mma.sync` shapes. Correct behaviour, by luck rather than by design, but worth making explicit.

**Bug 1 — the fp8 lowering is not arch-gated at all.** In
`crates/tang-compile/src/tensor_cuda.rs:2102-2127` the `_ if fp8 =>` arm comes *first* in the
match and ignores the arch: any kernel reading fp8 storage gets `cuda_fp8_kernel_shaped`, and
`crates/tang-compile/src/tensor_cuda_fp8.rs:127` returns `None` unless `sh.wgmma` is set (all six
`SM90_FP8_SHAPES` carry `.wg()`), so the fallback chain ends in a panic. The only reason prod
never hits this is that coder-next sets no fp8 storage anywhere. **The existing fp8 code is
wgmma-only by construction and therefore unusable on prod.** "Turn on fp8 on the 6000" is not a
flag flip; it is a new tile.

**Bug 2 — `cuda_sm() >= 90` is true on prod.** `cuda_sm()` parses the numeric part
(`tensor_cuda.rs:35`, fed from `program.rs:1852-1858`), so on `sm_120` it returns 120.
`crates/tang-compile/src/tensor_cuda_attn_bwd.rs:569` then budgets **227 KB of shared memory per
block**, and `:645` enables the `d == 256` attention-backward path — on a part with a **99 KiB**
ceiling. Any d=256 attention backward would fail to launch or corrupt. Not on the prefill serving
path today (the coder engine sizes its own smem from the device attribute,
`crates/tang-llm/src/flash/coder.rs:880-894`, and caps at the device's opt-in value), but it is a
trap for the first person to run that backward on prod.

**Bug 3, cosmetic but real: prod's `TANG_CODER_ATTN=derived` is dead.** The pod env sets it.
`crates/tang-llm/src/flash/derived_attn.rs:272` compiles with `arch: Some("sm_90a")` and `:227`
rejects any variant whose source lacks the literal string `"wgmma"`, so the generated kernel is
Hopper wgmma PTX by construction. On prod the module load must fail (wgmma PTX is not
forward-compatible), and `crates/tang-llm/src/flash/coder.rs:3029` catches it and prints
`coder: no derived attention (…); FA2 instead`. So prod is silently on FA2, after paying the
startup cost of compiling kernels that cannot load. Worth confirming and then deleting the env
var — a free win in startup time and a removed source of confusion.

### 1.4 Commands for the manager (all read-only, no pod restarts)

```bash
# A. what the card actually is (also settles the 96 vs 188 GB question)
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm --format=csv

# B. does wgmma assemble for this target? expect a ptxas error on sm_120
printf '.version 8.7\n.target sm_120\n.address_size 64\n.visible .entry t() {\n  wgmma.fence.sync.aligned;\n  ret;\n}\n' > /tmp/wg.ptx
ptxas -arch=sm_120 /tmp/wg.ptx -o /tmp/wg.cubin   # expect: instruction 'wgmma.fence' not supported

# C. does fp8 mma.sync assemble? expect success — this is the tile prod could actually use
printf '.version 8.7\n.target sm_120\n.address_size 64\n.visible .entry t() {\n  .reg .b32 d<4>, a<4>, b<2>;\n  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {d0,d1,d2,d3};\n  ret;\n}\n' > /tmp/f8.ptx
ptxas -arch=sm_120 /tmp/f8.ptx -o /tmp/f8.cubin

# D. does the block-scaled (MXFP4) mma assemble as sm_120, and does it need sm_120a?
#    run the same file twice, once with each target — that difference is the whole §3.3 argument
sed -e 's/mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32/mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32/' /tmp/f8.ptx > /tmp/mx4.ptx
ptxas -arch=sm_120  /tmp/mx4.ptx -o /tmp/mx4a.cubin
ptxas -arch=sm_120a /tmp/mx4.ptx -o /tmp/mx4b.cubin

# E. is prod really falling back off derived attention?
#    in the serving log: "coder: derived attention: N window kernels" = it worked;
#    "coder: no derived attention (…) ; FA2 instead" = Bug 3 confirmed.
```

## 2. The weight path

### 2.1 What exists today

- **GGUF only, memory-mapped, decoded in-kernel.** `crates/tang-llm/src/gguf.rs` maps shards;
  `crates/tang-llm/src/flash/coder.rs:343` (`native_id`) maps GGUF block types to kernel `ty`
  ids; a weight stack is four scalars, `ExW { base, estride, ty, row_bytes }`
  (`crates/tang-llm/src/flash/coder_kernels.rs:463`). Kernels read Q4_K / Q5_K nibbles straight
  from global memory and apply `d`/`dmin`/6-bit sub-block scales in the epilogue. There is no
  dequantized copy of anything, anywhere.
- **safetensors exists but is unusable as a checkpoint loader.** `crates/tang-gpu/src/safetensors.rs`
  converts every tensor to `Vec<f32>` on the host (`:23-40`) — 4 bytes per weight, so coder-next
  would be ~320 GB in host RAM. It is a small-model convenience, not a serving path.
- **DeepSeek's native formats are GEMV-only.** `Format::Fp8B32` and `Format::Mxfp4` exist
  (`crates/tang-compile/src/tensor_precision.rs:75,80`, both documented "Input storage only"),
  but `crates/tang-compile/src/tensor_int8_sched.rs:84` refuses them for MMA and routes them to
  the Gemv stage. That is the measured GEMV signature in `docs/prefill-sol.md` §4 (~93 tok/s).
- **The vendor GEMM path is not on main.** `git merge-base --is-ancestor 3feefe09 origin/main`
  fails; `crates/tang-gpu/src/vendor.rs` exists only on `claude/ds-prefill`. It dequantizes the
  GGUF weight and the Q8x32 activation to f16 scratch and calls cuBLASLt — the +12.5% figure.

### 2.2 The three options, and what each is worth on prod

**(a) Add an fp8 checkpoint loader.** Needs: a loader that keeps E4M3 bytes plus the block-scale
layout without a host f32 round-trip; a `Format` for it; a **new `mma.sync` fp8 tile** (the
existing fp8 tile is wgmma-only, §1.3); and an fp8 per-token activation quant (the analogue of the
int8 `cx_rmsnorm_q8` / `cx_silu_q` pair at `coder_kernels.rs:109,544` — trick #12). Cost: the
largest of the three. Payoff on prod: **~0%** on rate (§1.2), minus 1.8× the weight bytes
(49.6 → ~80 GB, leaving 16 GB of a 96 GB card for a 131k-token KV pool). **Recommend against.**

**(b) A bf16 checkpoint.** 2 bytes/weight ≈ 160 GB. Does not fit. Dead.

**(c) Dequantize in flight.** Keep the 49.6 GB Q4_K resident and convert to the tile's operand
format as it is read. Two variants:
 - **Q4_K → f16 in the GEMM** (what `claude/ds-prefill`'s vendor path does): no new weight path
   at all, but it lands on cuBLASLt f16, whose dense rate on the 6000 is ~460 TFLOP/s — *half*
   the int8/fp8 rate. Wrong direction for prefill.
 - **Q4_K → MXFP4 at load** (offline repack, once): same bytes, 2× the rate, and it is the only
   format change that pays on this chip (§3.3). Qwen already publishes an `MXFP4_MOE` GGUF for
   this model, and `crates/tang-llm/docs/models.md:33` lists it. tang's `GgmlType` has no MXFP4
   variant (`crates/tang-llm/src/gguf.rs:38-67`), so it would need one — but the repack can also
   be done from the bf16/fp8 checkpoint, which is better than re-quantizing an already-4-bit file.

**Answer to the loader question: do not add an fp8 loader. If the format is to change at all on
prod, the target is MXFP4 and the loader is a repack, not a new checkpoint format.**

One more thing the loader question should not obscure: **at 4096-token windows the model is not
format-bound.** 49.6 GB read once per window at 1,597 GB/s is ~31 ms, and 6.5 GFLOP/token × 4096
tokens at ~500 TFLOP/s [A, half the 8-bit peak to allow for starvation] is ~53 ms — so prod's own
ceiling is roughly 77k tok/s, and the measured ~6k is 8% of it. **The 12× gap is chunk size,
expert grouping and tile efficiency, not number format.** Any format work before #1/#2 land is
unmeasurable.

## 3. The MMA target per arch, and what it implies

### 3.1 Hopper (bench, `sm_90a`)

`wgmma` is real, `SM90_MMA_SHAPES` and `SM90_FP8_SHAPES` are already written and measured
(`docs/bench/fp8-hopper.md`: E4M3 wgmma 731 TFLOP/s at 8192³ vs cuBLASLt FP8 1457; bf16 wgmma 430
vs cuBLAS bf16 707). Trick #3 as written — fp8 weights, wgmma, block-128 scales — is a coherent
plan **for this machine**, and the tile already exists. It is bench work: prod cannot run it.

### 3.2 Datacenter Blackwell (B200/B300, `sm_100a`/`sm_103a`)

`tcgen05` + TMEM, no wgmma. tang has nothing for it, and `crates/tang-moe/src/gpu.rs:115-145`
already documents the NVRTC-doesn't-know-`sm_103` retry-to-`sm_90` workaround. Out of scope for
prod; relevant only if the fleet moves to B200/B300, in which case the fp8 path would need a
third rewrite, not a port.

### 3.3 The 6000 (prod, `sm_120` / `sm_120a`)

`mma.sync` only. Two things follow.

**Trick #3 is not the lever for prod.** Not "less valuable" — zero on arithmetic rate, negative on
bytes, and it requires writing a tile that does not exist. If the goal is "the GEMM share gets
faster on the machine that earns revenue", #3 does not deliver it.

**The lever that does exist is block-scaled `mma.sync` (MXFP4).** Measured at 1,069 TFLOP/s vs
529 for FP8 on the same GB202 silicon — a real 2×, at the same ~4.25 bits/weight as the Q4_K we
ship. It needs `sm_120a`, which tang's arch string does not currently produce (§1.3), and a new
tile emitter. Caveat from the same measurements: MXFP4's 256-wide tile needs a large M to fill
(27.6% efficiency at M=128 vs FP8's 39.4%), and it only overtakes FP8 in absolute terms at
M ≳ 32–256. At prod's 4096-token windows each routed expert sees `4096 × 10 / 512 = 80` rows, so
**MXFP4 would be a loss at today's window size** and a ~1.3–2× win at 8k–32k. It is a multiplier
on #1, not a substitute for it.

### 3.4 Swap-AB (#11) on the 6000

**Already implemented, and it is the production wide-prefill expert layout.**
`crates/tang-llm/src/flash/coder_kernels.rs:1242-1249` states it outright: *"SwapAB: each warp
owns 16 rows of its group's expert (M), the group's <= 8 tokens are N, and K steps 32 weights …
a mma.sync m16n8k32 s8"*, and it is the default (`cx_moe_gemm_gu` / `cx_moe_gemm_down`, WGEMM on,
`crates/tang-llm/src/flash/coder.rs:252-254,292-294`).

The framing in `docs/vllm-tricks.md` #11 is Hopper-specific: on wgmma, `BLOCK_M < 64` wastes the
M slot of an m64nNk16 instruction, so swapping to put tokens in N recovers 1.2–1.5×. On prod there
is no wgmma; `mma.sync` is m16n8k32, whose N slot is only 8 wide, so putting the 16 expert rows in
M and 8 tokens in N is the *only* sane orientation. **#11 needs no work on the 6000.** What is
left is not the orientation but the fan-out: each 64-token block walks the weight rows once per
8-token channel (`nch = (cnt + 7) / 8`), so the real gain is bigger groups and bigger chunks —
again #1/#2.

## 4. GPU-hour estimate

Work that needs no GPU at all: everything in §2.2(c) weight repacking, the `sm_120a` arch string,
the `cuda_sm()` gate fix, and the `TANG_CODER_ATTN` deletion. NVRTC/ptxas checks (commands B–D)
run in seconds on the already-running pod.

| step | GPU? | GPU-hours | note |
|---|---|---|---|
| 0. arch + instruction verification (§1.4 A–E) | read-only | ~0 | minutes on a pod that is already up |
| 1. prod GEMM ceiling: `scripts/prefill_gemm_bench.cu` on the 6000 at M = 2k/8k/32k | yes | 1 | needs a quiet window; do **not** bench on prod while it serves |
| 1b. same bench, H200, same shapes | yes | 1 | the like-for-like column prod has never had |
| 2. MXFP4 row added to the bench (cublasLt block-scaled) | yes | 2–3 | fiddly API; without it step 1 only bounds fp8/int8 |
| 3. `sm_120a` arch string + `cuda_sm()` gate fix | no | 0 | one line each, plus a retest |
| 4. `mma.sync` MXFP4 tile emitter | yes | 8–16 | the real piece of work; NVRTC iteration needs a GPU |
| 5. weight repack to MXFP4 + loader support | no | 0 | CPU-side, can start immediately |
| 6. end-to-end prefill A/B on prod | yes | 2–4 | wide windows, correctness gate |
| **total** | | **14–27** | ≈ **$38–73** at $2.69/hr |

That is the cost of the MXFP4 route. The fp8 route costs about the same and returns nothing on
prod (§1.2), so it should be scoped as H200 bench work only.

## 5. Ordered steps

1. **Run §1.4 A–E on prod and paste the output.** Nothing below is worth starting until the arch
   is confirmed from the device rather than from a datasheet, and E tells us whether prod has been
   silently on FA2.
2. **Fix the arch string and the smem gate** (§1.3, Bugs 1–2): `sm_120a` for major ≥ 12, and make
   the `cuda_sm() >= 90` smem budget in `tensor_cuda_attn_bwd.rs:569` an actual capability test
   (99 KiB below sm_100). Delete `TANG_CODER_ATTN=derived` from the pod env. No GPU cost, removes
   three traps.
3. **Get prod's own GEMM ceiling** (§4 step 1) so #1/#2 have a denominator. Until this exists every
   "N× on prod" claim in the trick table is an H200 claim wearing a prod label.
4. **Land #1 and #2** — 8k+ windows and grouped expert GEMMs. These are 12× of headroom and they
   belong to the prefill-gap session, not this one. Format work waits behind them because MXFP4 is
   a multiplier on chunk size, not a substitute for it.
5. **Then, and only then, MXFP4** (§3.3, §4 steps 3–6): repack the weights, add the block-scaled
   tile, A/B it at 8k and 32k windows. Expect ~1.3× at 8k and up to ~2× at 32k.
6. **Do not build an fp8 weight path for prod.** The existing fp8 tile stays what it is: an H200
   benchmark asset, correctly gated to `sm_90a`. If someone wants it to *run* on prod, the honest
   description is "write a second, unrelated `mma.sync` fp8 tile" — and §1.2 says it would not be
   faster than the int8 tile already there.
