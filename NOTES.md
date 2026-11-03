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
