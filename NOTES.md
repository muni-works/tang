# NOTES — cram Beat 2 (worktree `frog/cram-beat-2`)

Checkpoint 0, before measuring. Branch `frog/cram-beat-2`, at merge `2da92e35`.

## What I know so far (survey, no numbers yet)

- `crates/cram-codec` exists in tang (renamed from tang-compress, brand "cram",
  2026-10-21). Error-bounded lossy compression of f32/f64 arrays, pure Rust,
  single-threaded, deterministic. Codecs: QTT (codec.rs), wavelet CDF 9/7
  (wavcodec.rs), tucker, interp, lorenzo, plus GPU stream layouts
  (`gpuwav` TCG1, `gpuwav3` TCG3 static/learned) and a KV codec (`kvcodec`).
- `container.rs` (CRM1, recent, commit c1ccfc3f/476750f5) is a *lossless*
  general-purpose container: it runs arms stored/zstd-3/zstd-19/xz-9/brotli-11,
  picks the smallest, adds a 13-byte header. `cram bench FILE` prints per-arm
  CSV. The lossy arm (`cram-lossy`) is the f32 portfolio (`encode_auto_f32`),
  enabled only with a bound in the hint.
- Home page in Cargo.toml says `https://cram.sh` — DNS is dead (predecessor
  claim; I checked nothing resolves). A cram.sh site was built on branch
  `claude/cram-site` (merged to main f9d12989, Oct 21): landing + demo +
  docs redirect, with RD charts vs SPERR/SZ3/ZFP/TTHRESH on held-out SDRBench.
  It is NOT on frog.toys and there is no bench page.
- bench/ma-gate.md (committed, mew Oct 21): container ties best-arm+13 bytes on
  24/25 files. Silesia + text corpus. Numbers there are mew's (i9-12900K),
  NOT reproduced by me.
- docs/compression-math-survey/ has the measured story: tc4-gate (held-out
  SDRBench, geomean vs best external 0.27–0.88× by point class, 103/104 points
  pass the 1.05× gate, ONE fail at hurricane-W 60dB 1.053× SPERR), portfolio,
  KV decode speed ("not fast enough in-loop: 4 Gvalue/s ceiling, position is
  store/move"), tc3-cleanup. All ran on mew; I have no mew access today.
- KV in-loop numbers exist (Apple M4 Max, results in bench/results/kv3): tkv
  saves 21–23% bits vs KIVI-4 at matched NLL; in-loop decode 1.8–5.5× slower
  than bf16 step — NOT fast enough for the attention loop.
- WebGPU decoder gate: 285/285 byte-identical (M4 Chrome, Oct 20), decode
  0.021–0.114 GB/s end-to-end on 1M-cell fields (small fields, dispatch-bound).

## Constraints / honesty notes

- mew unreachable today: tried `ssh mew` (no ssh-config entry), history shows
  `cam@192.168.0.96` → connect timeout (wrong network). All MY numbers are
  LOCAL on this Mac (Apple silicon) unless marked otherwise. Single thread.
- No GPU in this machine's codec path here: tc-gpu decode = the CPU reference
  path unless the `gpu` feature is enabled (wgpu/Metal). I'll try the gpu
  feature locally; if it doesn't build/run, tc-gpu decode numbers stay
  {{PLACEHOLDER}}.
- The KV arrays I have are synthetic-but-labeled (I'll say so), plus REAL
  weights from a local HF checkpoint (Qwen3-1.7B MLX quantized — bf16 payloads
  widened to f32: layernorm, embed scales, mlp biases). No real KV dump on
  this disk; bench/tensors (SmolLM2 dumps) not fetched.

## Plan

1. Build `cram` + examples (done: release build OK).
2. Measure: container arms (zstd/xz/brotli) vs cram-lossy on my arrays, with
   real bounds (rmse 0.01 etc). Record ratio + MB/s per arm.
3. Try `--features gpu` for tc-gpu decode; record if it works.
4. Write docs/launch/beat2.md in beat1's voice; placeholders where unmeasured.
5. Stage ~/Developer/frog/site/design/cram.html + cram.css from frog.toys
   tokens (task spec's #191919/#232323/ececec/green/Plex).
6. Verify: numbers trace to NOTES.md, HTML renders, commit everything `-s`.
   Never push.

## MEASURED — local run, Apple M4 Max (this Mac, 16 cores), single thread, 2026-11-03

Build: `cargo build --release -p cram-codec --bin cram --examples` at 2da92e35.
Cram version: workspace 0.1.0, container CRM1 (13-byte header).

### Test arrays (/tmp/cram-in, each is plain little-endian f32)
- `kv-k`, `kv-v`: 8×1024×128, 4 MiB each — synthetic KV-like fields
  (AR(1) noise, per-head offsets, rope-like even/odd structure). SYNTHETIC, labeled as such.
- `hidden-128x2048`: 1 MiB synthetic hidden-state-like field.
- `mlp-down-f32`: REAL weights — Qwen3-1.7B (MLX 4-bit checkpoint) layer-0
  mlp.down_proj biases, bf16 payload widened to f32, 786,432 B.
- `embed-scales-f32`: REAL weights — same checkpoint's embed scales bf16→f32, 19,447,808 B.
- `layernorm`: REAL bf16→f32 layer-0 input_layernorm weight, 1024 vals, 4,096 B.

### Container `cram bench` — every arm at rmse=0.01 bound, bytes and MB/s (median-ish of runs)

| file | in B | stored | zstd-3 | zstd-19 | xz-9 | brotli-11 | cram-lossy | lossy c MB/s | lossy d MB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kv-k | 4,194,304 | 4,194,304 | 3,859,438 | 3,857,184 | 3,836,972 | 3,826,726 | 934,768 | 0.49 | 51.06 |
| kv-v | 4,194,304 | 4,194,304 | 3,861,832 | 3,859,758 | 3,839,184 | 3,832,558 | 932,808 | 0.52 | 52.92 |
| hidden-128x2048 | 1,048,576 | 1,048,576 | 970,289 | 970,230 | 968,468 | 964,626 | 284,557 | 0.61 | 48.39 |
| mlp-down-f32 | 786,432 | 786,432 | 292,325 | 251,538 | 225,716 | 228,233 | 72,802 | 0.54 | 105.16 |
| embed-scales-f32 | 19,447,808 | 19,447,808 | 6,938,541 | 5,685,039 | 5,352,776 | 5,526,765 | pending (job-3) | — | — |
| layernorm (1024 vals) | 4,096 | 4,096 | 2,427 | 2,402 | 2,184 | 2,036 | 165 | 0.43 | 91.53 |

Reads (bound rmse=0.01, so quality is matched across codecs):
- On KV-like and hidden-state fields, GENERAL-PURPOSE lossless compressors
  compress these f32 arrays by only ~1.09× (zstd-3 1.0868×, brotli-11 1.0961×
  on kv-k). cram's lossy wavelet path gets 4.49× at the bound — 4.1× smaller
  than the best lossless arm. This is the whole point: float mantissa noise
  defeats byte-oriented LZ.
- On the real weight arrays the lossless arms do better (3.63× xz-9 on
  embed-scales) but the lossy portfolio still wins (10.8× on mlp-down,
  3.1× smaller than xz-9).
- Speeds are what they are: lossless arms are zstd/xz/brotli C libraries;
  cram-lossy encode is 0.4–0.6 MB/s single-thread (it trial-encodes the
  portfolio), decode 48–105 MB/s single-thread. Decode is the fast direction;
  in a KV/prefetch pipeline decode is what runs hot.
- mb/s = raw bytes / 1e6 / s as `cram bench` prints it. Single thread.

### KIVI parity check (kv arrays, my numpy reimplementation of KIVI's scheme)
KIVI-4 g32 (6.0 bpv) has relerr 7.1% on these fields; reaching cram's relerr
1% needs KIVI-7–8 (9–10 bpv incl. fp16 scales), recent-token window included
→ kivi8+win128 ≈ 12 bpv, kivi7+win128 ≈ 11 bpv vs cram kvc 7.06–7.13 bpv.
So on THESE synthetic fields cram's error-bounded path is ~1.6× smaller than
KIVI at matched ~1% relerr. (The committed docs' 21–23% saving was measured on
real model dumps on mew; my KIVI numbers are my own reimplementation on
synthetic fields — consistent in direction, not the same measurement.)

### kvc (KV codec CLI), streaming + one-shot
- kvc one-shot k: 928,999 B = 7.088 bpv, encode 0.058 s, decode 0.049 s (47 ns/value)
- kvc one-shot v: 921,557 B = 7.031 bpv, encode 0.050 s, decode 0.047 s (45 ns/value)
- kvc streaming (prefill 128, append 896): k 934,315 B = 7.128 bpv (append 0.044 s),
  v 922,513 B = 7.038 bpv (append 0.042 s). Streaming costs ~0.6% bits here.
- decode measured 47 ns/value → single-thread ≈ 21 Mvalue/s CPU; the M4 GPU
  kernel (committed docs) is ~4 Gvalue/s — that's the in-loop ceiling.

### tc-gpu / TCG3 on THIS machine (M4 Max, Metal, wgpu feature enabled)
`gpubench speed kv-k.f32 8x1024x128 --target p60 --bricks 64 --threads 8`:
- tc-gpu (TCG1): bpv 7.117 @ psnr 60.001; CPU 1-thread decode 0.053 GB/s;
  CPU 8-thread 0.408 GB/s; GPU (Metal) kernels 0.254 GB/s, end-to-end
  (parse+upload+kernels+readback) 0.208 GB/s; `gpu_bit_identical: true`.
- random access 64-cell box (1/64th of field): 2.6 ms 8-thread (0.4% of full decode).
- embed-scales 151936x32 p60 brick 128: GPU kernels 1.71 GB/s, e2e 0.96 GB/s
  (CPU 1-thread 0.053 GB/s) — `gpu_bit_identical: true`.
- hidden 128x2048: gpu e2e 0.079 GB/s; mlp-down 2048x96: gpu e2e 0.096 GB/s
  (small fields pay fixed per-dispatch cost; the README says the same).
- TCG3 on embed-scales panics on Metal: "Compute function exceeds available
  stack space" (wgpu create_compute_pipeline, 'lift'). TCG1 works; TCG3
  compile fails on THIS GPU config. Honest note: the learned-model path is
  not verified on this machine.
- TCG3 CPU decode (gpuwav3::decode3_f32) on kv-k p60: 0.096 s for 4 MiB
  (0.043 GB/s 1 thread).

### Container tie behavior (matches the committed ma-gate.md)
`cram` (default level) ties best-arm+13 header bytes on all six arrays here
(kv-k/kv-v/hidden pick brotli-11; mlp-down/embed-scales pick xz-9; layernorm
brotli-11) — same "never beats the best arm, only adds 13 bytes" behavior as
the committed Silesia gate. cram-fast = stored+zstd-3 only (878–964 MB/s
encode on the 4 MiB fields, ~1.087×).

### Speed cross-check of the C arms (CLI, same file)
zstd -19: 251,542 B in 0.139 s (5.7 MB/s); xz -9e: 225,904 B in 0.222 s
(3.5 MB/s); brotli -q 11: 228,180 B in 0.764 s (1.03 MB/s) on
mlp-down-f32 (786,432 B). The in-crate arms match these byte sizes
(251,538 / 225,716 / 228,233) — differences are CLI flags/multithreading.

### What I could NOT measure here (gaps → placeholders)
- H200 / Linux server numbers (mew unreachable: ssh mew fails, 192.168.0.96
  times out; nothing resolves cram.sh).
- Real model KV dumps (bench/tensors not fetched; SmolLM2 dump needs torch).
- The committed M4 in-loop KV results are cited from
  docs/compression-math-survey/tang-compress-kv-decode-speed.md (they ran on
  an M4 Max too, but not by me today).
- TCG3 GPU decode: fails to compile its pipeline on this Metal config.

### embed-scales lossy result (completed after the table above)
- `tc --codec wavelet --rmse 0.01` (the portfolio kept hanging on the 19 MB
  file — job-3 produced nothing in 305 s; wavelet alone finished in 1.07 s):
  **124,938 B = 0.2056 bpv = 155.7× vs xz-9's 42.8× (3.64× smaller than the
  best lossless arm), rmse 0.010000, maxabs 0.035, decode 0.120 s.**
- Caveat recorded: `cram bench` on embed-scales with the lossy arm never
  finished here (portfolio hangs/trials forever on this 2-D 19 MB field);
  the wavelet-only run is what I have. The container would still pick the
  smallest arm.
