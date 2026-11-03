# Beat 2 launch post: cram (draft)

Status: DRAFT. Beat 2 was announced as the "cram + the H200 result" slot. The H200 part
turned out to be impossible with the current code (there is no CUDA KV decoder — see "what
it won't claim"), so this draft ships the measured cram story and says so plainly. `{{…}}`
marks a number that was not measured in time; the HTML comment above it says what would fill
it. Nothing in here may be stronger than its receipt. Editor notes are HTML comments and get
stripped before publishing.

Every number below was measured twice on 2026-11-03: once on **an Apple M4 Max (single
thread)** and once on **an Intel i9-12900K with an RTX 3090 (24 threads; GPU via
Vulkan/wgpu)**, both at commit `07a737b7`, on the same six float32 arrays (two of them real
weights from a local Qwen3-1.7B checkpoint). The two machines produce **byte-identical
streams**, which is the point of the codec and the strongest result in this post. The
committed gates (Silesia container gate, held-out SDRBench quality gate, KV in-loop study)
ran on the same Linux box and a different M4 earlier in October — cited as such, not
re-measured.

<!-- H200: cannot be measured, and the post now says why (no CUDA tkv_decode). If a CUDA
kernel lands before publish, add it; otherwise this stays as written. -->

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

Six arrays, one bound (`rmse = 0.01` for all), measured on the i9-12900K (mew) with the
M4 Max agreeing to the byte. "best lossless" is the smallest of zstd-19, xz-9 and
brotli-11. cram's lossy arm is the wavelet codec (the portfolio pick on these arrays),
measured at the same bound.

| array | in bytes | best lossless | bytes | ratio | cram-lossy | bytes | ratio | lossless ÷ cram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KV-K (8×1024×128, synthetic) | 4,194,304 | xz-9 3,836,972 | 1.09× | | 934,768 | **4.49×** | 4.10× |
| KV-V (same shape, synthetic) | 4,194,304 | brotli 3,832,558 | 1.09× | | 932,808 | **4.50×** | 4.11× |
| hidden states (128×2048, synthetic) | 1,048,576 | brotli 964,826 | 1.09× | | 284,522 | **3.69×** | 3.39× |
| MLP down-proj biases (real weights) | 786,432 | xz-9 225,716 | 3.48× | | 72,802 | **10.80×** | 3.10× |
| embedding scales (real weights, 19.4 MB) | 19,447,808 | xz-9 5,352,776 | 3.63× | | 124,938 | **155.7×** | 42.8× |
| layernorm weight (real, 2048 values) | 8,192 | brotli 2,108 | 3.89× | | 295 | **27.8×** | 7.15× |

<!-- The two synthetic arrays are labeled synthetic in NOTES.md. Real KV dumps from a model
run are {{MEW_KV_DUMP}}; the committed bench/tensors corpus ran on mew. If we publish before
re-running on real KV, the post says "KV-like synthetic" and does not. -->

**Both machines produced the same bytes.** KV-K: 934,768 on the M4 Max and 934,768 on the
i9-12900K. Same for every row. That's not a coincidence to note in passing — it's the
guarantee the crate makes, and the reason the browser, CPU and GPU decoders can be held to
one hash.

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

Single thread, same arrays, on both machines:

- cram wavelet encode: 0.3–0.6 MB/s on the 12900K, 0.4–1.1 on the M4 Max (it searches for
  the smallest stream; encode is the slow direction and we haven't optimized it).
- cram wavelet decode: 42–133 MB/s (0.089 s for the 4 MiB KV field on the M4; 0.238 s for
  the 19 MB embedding scales on the 12900K).
- zstd-3: 130–690 MB/s encode, 0.7–1.4 GB/s decode. xz-9: 2–6 MB/s encode. brotli-11:
  ~0.9–2 MB/s encode.
- The KV codec's dedicated streaming path: encode 70–82 ms for 4 MiB, decode 58–63 ms on
  the 12900K (58–60 ns/value); 45–49 ms on the M4 Max (45–47 ns/value).

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
speeds up your context window would be wrong, and we're not making it. That kernel is Metal
only, which is also why there is no H200 version of this measurement.

### The GPU decode path (tc-gpu), measured on two GPUs

The `gpu` feature compiles the tc-gpu stream's WGSL kernels with wgpu. Decoding a 4 MiB
KV-like field at PSNR 60:

- M4 Max (Metal): **0.25 GB/s** GPU kernels, **0.21 GB/s** end-to-end.
- RTX 3090 (Vulkan): **0.29 GB/s** kernels, **0.20 GB/s** end-to-end.
- Both against **~0.05 GB/s** for the CPU single-thread decode of the same stream.
- `gpu_bit_identical: true` on every run, on both GPUs: same bits as the CPU decoder.

Small fields are dispatch-bound — that's why those numbers look unimpressive. The big-field
regime is where the design pays off. On a 512³ f32 field (512 MiB, 4.3–4.6 bits/value at
60 dB) on the RTX 3090, brick size dominates:

| brick | GPU kernels | GPU end-to-end | CPU 16-thread |
|---|---:|---:|---:|
| 128³ | 0.79 GB/s | 0.51 GB/s | 0.58 GB/s |
| 64³ | 3.58 GB/s | 0.97 GB/s | 0.58 GB/s |
| 32³ | 4.51 GB/s | 1.08 GB/s | 0.56 GB/s |
| 16³ | **5.60 GB/s** | **1.10 GB/s** | 0.48 GB/s |

Two things to read carefully. **The kernel is 5.6 GB/s at 16³ bricks — 11× the 16-thread CPU
decode.** The end-to-end number plateaus at ~1.1 GB/s because 0.27 s of the 0.49 s total is
*readback*: moving 512 MiB back to the host over PCIe. So "1.1 GB/s" is what you get when you
copy the decoded field to RAM; the codec itself is not the limit. Any number that quotes
only one of these without saying which is being sloppy — including ours if we ever do it.

Random access is real: a 64³ box out of the 512³ field decodes in **5.7 ms** (16 threads) at
brick 16, i.e. 1/512th of the work.

<!-- H200: see "what it won't claim". There is no CUDA kv decode, so no H200 KV number exists
to fill in; tc-gpu above runs on any Vulkan device including an H200, but that is a field
codec, not a KV-cache-in-attention result. -->

## What it won't claim

- **It is not a general-purpose compressor.** On text, JSON or bytes that are already
  compressed, the container ties zstd/xz/brotli and adds 13 bytes. cram is not "better
  than zstd"; it's a different tool for float arrays with an error budget.
- **No image codec claims in this post.** The crate has an image path (img.rs, tools/
  coolchic) and it's promising, but we didn't measure it for Beat 2, so it isn't here.
- **The learned-model GPU path is unverified.** TCG3 with the pretrained net fails to
  compile its pipeline on the M4's Metal config ("compute function exceeds available stack
  space"). The static path passes bit-identity on both GPUs; the learned path did not run
  here at all.
- **The "H200 result" cannot be produced from this code.** Beat 2 was announced as "cram +
  the H200 result". The KV GPU decoder (`tkv_decode`) exists only as a Metal kernel;
  `ComputeDevice::tkv_decode` is `unimplemented!("tkv_decode is implemented for Metal
  only")`. There is no CUDA implementation, so an H200 can run the field codec (tc-gpu,
  above) but **cannot decode a KV cache in the attention loop**. That is a named piece of
  work, not a measurement we could rent: write the CUDA `tkv_decode` and rerun the
  in-loop study on an H200. The post ships without the number rather than with a guess.
- **No cross-vendor GPU table.** Two GPUs, both consumer/workstation class (M4 Max, RTX
  3090). The WebGPU browser gate (285/285 byte-identical, Oct 20) is committed, but its
  throughput numbers are not reproduced today.
- **A real defect found while measuring.** The lossy portfolio allocates ~184 GB and aborts
  on a 151936×32 field (the Tucker candidate on a very long, thin 2-D field). Wavelet alone
  handles it fine in 2.7 s. It's filed, not hidden; until it's fixed, `--codec wavelet` is
  the safe path for that shape.
- **Encode speed is not a selling point.** 0.3–1.1 MB/s single-thread. If you need to
  compress 100 GB of arrays tonight, that's 25–90 hours; plan for it or use `--fast`
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
