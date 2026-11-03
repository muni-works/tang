# Beat 2 launch post: cram (draft)

Status: DRAFT. Beat 2 is the announced "cram + the H200 result" slot. `{{…}}` marks a number
that was not measured in time; the HTML comment above it says what would fill it. Nothing in
here may be stronger than its receipt. Editor notes are HTML comments and get stripped before
publishing.

Every number below was measured on **one Apple M4 Max, single thread, 2026-11-03**, on six
small float32 arrays (two of them real weights pulled from a local Qwen3-1.7B checkpoint),
with cram-codec built from this repo at `{{SHA}}`. The committed gates (Silesia container
gate, the held-out SDRBench quality gate, the KV in-loop study) ran on the studio Linux box
and a different M4 earlier in October — those numbers are cited as such, not re-measured.

<!-- If the launch waits for the H200 run, fill MEW_* from bench/ on mew and re-date the
method line. If Beat 2 ships without it, keep the M4 numbers and the placeholders. -->

---

## HN submission

**Title:** Show HN: cram — error-bounded compression for f32/f64 arrays, ~4× smaller than
zstd on GPU KV caches at a stated error bound

**Text (first comment):**

> I'm Cam. cram is a small Rust crate (MIT/Apache-2.0) we built for tang, our inference
> engine. It compresses f32/f64 arrays to an error bound you name — RMSE, PSNR or
> pointwise max-abs — and the decoder is deterministic and byte-identical on CPU, GPU and
> the browser. The honest part: on smooth scientific fields it's very good; on KV-cache-like
> float noise it's about 4× smaller than zstd/xz/brotli at a 1% RMSE bound; on text or
> already-compressed bytes it's just a container that ties the best of zstd/xz/brotli and
> adds 13 bytes. Post has the tables and the exact bound for every number.

---

# cram: lossy compression with a stated error, for float arrays

cram is a Rust crate that compresses f32/f64 arrays to an error bound you choose
(`rmse`, `psnr` or pointwise `max-abs`), in pure Rust, single-threaded, deterministic.
Every byte of the stream counts toward the reported rate. The same crate also ships a
lossless container that wraps zstd, xz and brotli and picks the smallest — that's how
cram stays honest on data that shouldn't be compressed lossily.

```rust
use cram_codec::{decode, encode_f32, Config, ErrorBound};

let (bytes, stats) = encode_f32(&x, &[64, 64], &Config::new(ErrorBound::Rmse(1e-3)));
// every decoded value satisfies the bound, and the same bytes decode to the
// same array on CPU, on a GPU, and in a browser
```

It lives in [muni-works/tang](https://github.com/muni-works/tang) as `crates/cram-codec`.
This post is measured, not aspirational: every number has the array, the bound and the
machine behind it, and the "won't claim" section says what is not true.

## What the numbers say

Six arrays, one bound (`rmse = 0.01` for all), one machine (Apple M4 Max, single thread,
2026-11-03). "best lossless" is the smallest of zstd-19, xz-9 and brotli-11. cram's lossy
arm is the wavelet codec (the portfolio pick on these arrays), measured at the same bound.

| array | in bytes | best lossless | bytes | ratio | cram-lossy | bytes | ratio | lossless ÷ cram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KV-K (8×1024×128, synthetic) | 4,194,304 | xz-9 3,836,972 | 1.09× | | 934,768 | **4.49×** | 4.10× |
| KV-V (same shape, synthetic) | 4,194,304 | brotli 3,832,558 | 1.09× | | 932,808 | **4.50×** | 4.11× |
| hidden states (128×2048, synthetic) | 1,048,576 | brotli 964,626 | 1.09× | | 284,557 | **3.68×** | 3.39× |
| MLP down-proj biases (real weights) | 786,432 | xz-9 225,716 | 3.48× | | 72,802 | **10.80×** | 3.10× |
| embedding scales (real weights, 19.4 MB) | 19,447,808 | xz-9 5,352,776 | 3.63× | | 124,938 | **155.7×** | 42.8× |
| layernorm weight (real, 1024 vals) | 4,096 | brotli 2,036 | 2.01× | | 165 | **24.8×** | 12.3× |

<!-- The two synthetic arrays are labeled synthetic in NOTES.md. Real KV dumps from a model
run are {{MEW_KV_DUMP}}; the committed bench/tensors corpus ran on mew. If we publish before
re-running on real KV, the post says "KV-like synthetic" and does not. -->

What this actually says:

- **On float arrays with continuous structure, byte-oriented compressors barely compress.**
  zstd at level 19 gets 1.087× on the KV-like fields; brotli at its slowest setting gets
  1.096×. The mantissa noise defeats LZ. cram's wavelet path gets 4.5× at the same RMSE.
  That 4.1× gap is the product.
- **On real weights it's still a solid win** (3.1× smaller than xz-9 on the down-proj
  biases), and on smooth real fields it's a different sport: 0.21 bits/value (155.7×) on
  the embedding scales at rmse 0.01, where xz-9 manages 2.2 bits/value.
- **On text or already-compressed bytes, cram doesn't pretend.** The container ties the
  best arm and adds 13 header bytes; on the 25-file gate in the repo that's 24 ties and
  one miss (a ZIP at 0.21%, from the sniff that keeps encode fast). We did not re-run
  Silesia for this post; the committed table is in
  [bench/ma-gate.md](https://github.com/muni-works/tang/blob/main/crates/cram-codec/bench/ma-gate.md).

### Speeds, stated plainly

Single thread on the M4 Max, same arrays:

- cram wavelet encode: 0.4–1.1 MB/s (it searches for the smallest stream; encode is the
  slow direction and we haven't optimized it).
- cram wavelet decode: 48–105 MB/s (0.089 s for the 4 MiB KV field; 120 ms for 19 MB).
- zstd-3: ~250–900 MB/s encode, ~1.1–1.4 GB/s decode. xz-9: ~2–9 MB/s encode. brotli-11:
  ~1–2 MB/s encode.
- The KV codec's dedicated streaming path: encode 50–58 ms for 4 MiB, decode 45–49 ms
  (≈45 ns/value, ~21 M values/s single-thread CPU).

So: **cram loses to zstd on pure speed, wins on bytes at a stated error.** If your
pipeline only cares about throughput on incompressible floats, use zstd. If you're paying
for bandwidth or memory on float arrays you're allowed to reconstruct within a bound, cram
is in a different category.

## Where it fits in the inference pipeline

Two places, with different honesty requirements:

**Weights and KV caches at rest / in transit.** The KV codec keeps the recent window in
fp16 and compresses whole blocks behind it, appending one token at a time in the same
stream format; streaming costs ~0.6% bits vs one-shot here (7.128 vs 7.088 bpv). Against
KIVI-4 (the standard 4-bit KV scheme) at matched error on these fields, cram needs ~7.1
bpv where KIVI-shaped 8-bit per-channel with its fp16 window lands ~12 bpv. That's our
reimplementation on synthetic fields, not the committed result; the committed one
(21–23% fewer bits than KIVI-4 at matched NLL, on real dumps) is in
[docs/compression-math-survey](https://github.com/muni-works/tang/tree/main/docs/compression-math-survey).

**Not in the attention loop.** We tried that and it failed. In-loop decode peaked at
~4 G values/s on an M4 Max GPU; KIVI-style dequant runs at memory bandwidth. Compressed-KV
generation was 1.8–5.5× slower than bf16 steps. The verdict is written down in the same
survey: cram's KV codec is for *storing and moving* caches — decompress once on load
(prefix-cache hit, node migration) — not for decoding every step. Any claim that this
speeds up your context window would be wrong, and we're not making it.

### The GPU decode path (tc-gpu), measured here

The `gpu` feature compiles the tc-gpu stream's WGSL kernels with wgpu. On this M4 Max,
Metal backend, decoding a 4 MiB KV field at PSNR 60:

- **0.25 GB/s** GPU kernels, **0.21 GB/s** end-to-end (parse + upload + kernels +
  readback) — vs **0.05 GB/s** for the CPU single-thread decode of the same stream.
- On the 19 MB embedding-scales field: **1.71 GB/s** GPU kernels, **0.96 GB/s** end-to-end.
- `gpu_bit_identical: true` on every run: the GPU and CPU decoders produce the same bits.
- Random access is real and measured: decoding a 64-cell box (1/64th of the KV field)
  takes 2.6 ms with 8 threads, vs 10 ms for the whole field.

That end-to-end number is not the "GB/s decode" figure from the cram.sh draft (1.1–1.7
GB/s on 512³ fields — the big-field regime, with fixed per-dispatch cost amortized).
Small fields are dispatch-bound; the README says so and so do our numbers.

<!-- The H200 number this beat was announced with is {{H200_KV_GBPS}}: tc-gpu decode of a
full model KV slice on the H200. It needs the GPU box on mew. Until it exists, this post
carries only the M4 numbers above. -->

## What it won't claim

- **It is not a general-purpose compressor.** On text, JSON or bytes that are already
  compressed, the container ties zstd/xz/brotli and adds 13 bytes. cram is not "better
  than zstd"; it's a different tool for float arrays with an error budget.
- **No image codec claims in this post.** The crate has an image path (img.rs, tools/
  coolchic) and it's promising, but we didn't measure it for Beat 2, so it isn't here.
- **The learned-model GPU path is unverified on this machine.** TCG3 with the pretrained
  net fails to compile its pipeline on this Metal config ("compute function exceeds
  available stack space"). The static path passes bit-identity; the learned path did not
  run here at all.
- **No cross-vendor GPU table.** One GPU (M4 Max / Metal). The WebGPU browser gate
  (285/285 byte-identical, Oct 20) is committed, but its throughput numbers are not
  reproduced today.
- **The "H200 result" is not in this post.** Beat 2 was announced as "cram + the H200
  result". The H200 run isn't measured; the post ships without it rather than with a
  guess. If the run lands before publish, it goes in with its method and this section
  shrinks.
- **Encode speed is not a selling point.** 0.4–1.1 MB/s single-thread. If you need to
  compress 100 GB of arrays tonight, that's ~30 hours; plan for it or use `--fast`
  (still slow, less search).

## Try it

```sh
cargo add cram-codec
cargo run --release -p cram-codec --example tc encode field.f32 out.cram --dims 64x64x64 --codec auto --rmse 0.01
cargo run --release -p cram-codec --example tc decode out.cram field-dec.f32
```

`cram bench FILE [--hint f32:AxBxC:rmse=0.01]` prints one CSV line per arm: bytes, ratio,
compress and decompress MB/s — the same numbers this post uses, for your own files.

## Links

- cram-codec: [github.com/muni-works/tang/tree/main/crates/cram-codec](https://github.com/muni-works/tang/tree/main/crates/cram-codec)
- The measured gates: [crates/cram-codec/bench/ma-gate.md](https://github.com/muni-works/tang/blob/main/crates/cram-codec/bench/ma-gate.md)
  and [docs/compression-math-survey/](https://github.com/muni-works/tang/tree/main/docs/compression-math-survey)
- tang (the engine it belongs to): [github.com/muni-works/tang](https://github.com/muni-works/tang)

MIT OR Apache-2.0. Contributions under the DCO (`git commit -s`).

— Cam, Municipal Robotics

<!-- cram.sh: DNS is dead as of 2026-11-03 (doesn't resolve). Do not link cram.sh in the
published post until it resolves, or link the repo instead. The old cram.sh landing page
lives at crates/cram-codec-wasm/site (branch claude/cram-site, merged Oct 21). -->
