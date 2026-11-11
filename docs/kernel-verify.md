# Kernel verification: every GPU kernel checked against its CPU oracle

`crates/tang-verify`. The goal is to make the GPU-bug class we hit on GLM-5.3 structurally
impossible. Every kernel gets checked against the CPU reference when it is written, and again
in any engine run with one env var.

## Why

On 2026-10-26 the GLM-5.3 GPU engine produced garbage tokens while every unit test passed.
Four bugs were behind it (crates/tang-llm/NOTES.md has the full story). None was subtle numerics.
Each was a structural error that a plausible-looking test could not see:

| # | bug | why the tests missed it |
|---|-----|-------------------------|
| 1 | KDA q/k/v GEMVs launched at `ptr + di` with `di` in **floats**; a `CUdeviceptr` counts **bytes**. k landed `di/4` floats into q, v `di/2`; half of k and all of v were never written (cc578187) | the dumps printed `q[0..16]`, the one region nothing overwrote; unwritten v read 0.0 from a fresh buffer, which looks plausible |
| 2 | one `MlaCuda` per device served all 11 DSA layers: layer 3's weights and caches everywhere (74192590) | the tiny fixture has one DSA layer |
| 3 | GPU mHC ran block 0's (`hc_attn`) weights for the FFN half too (324e7f6b) | every 3090 run put mHC on the CPU |
| 4 | teardown SIGSEGV: CUDA contexts dropped before their streams and modules | not numeric. The GPU test "passed" by skipping when no GPU was found |

Four ad-hoc tools found them: the shadow check, NaN poisoning, first-divergence attribution and
a config bisect (`kda_debug.rs`, `TANG_GLM5_KDA_DEBUG=1`). tang-verify is those tools turned
into a library that every kernel author uses.

## The method

1. **Shadow.** Run the CPU reference on the *same input* the device got (upstream divergence
   can't leak in). Start it from the *device's own state*: after each window, resync the shadow
   to the device. Then each window tests one step, not accumulated drift. Compare every
   intermediate in full, every token and every channel, never a 16-element prefix. Each
   comparison is a `Stat`: rel-L2, max abs and *where* (decoded through a `Layout`:
   `tok7 h46 d36`), cosine, exact-zero counts and NaN counts.
2. **Poison.** Before the kernel runs, fill its outputs with NaN. An element no kernel wrote
   then reads NaN, never a plausible 0.0. NaN propagates through every downstream op, and
   `Stat` counts it. In bug 1 the NaN counts matched the offset arithmetic exactly.
3. **Attribute.** Name the kernel, not just the first failing tensor. The old rule ("earliest
   failing stage whose inputs pass") blamed `attn_q` for bug 1, because q's block failed first.
   But the values in it were k's. The fixed rule ranks evidence:
   1. *provenance*: look for a failing tap's wrong values in every tap's CPU reference.
      Found in a sibling → that sibling's writer put them there. When both live in one device
      buffer (`Spec::stacked` / `Tap::within`), the geometry gives the writer's actual offset,
      and a ×4 ratio is diagnosed as "an ELEMENT offset added to a BYTE address";
   2. *unwritten*: NaN beyond the reference's (poison survived) is direct evidence against
      that writer. A finite-but-wrong tap whose sibling is unwritten is blamed on the sibling;
   3. *zeroed*: notably more exact zeros than the reference;
   4. only then pipeline order.
4. **Bisect** (engine level, by env): flip components between GPU and CPU
   (`TANG_GLM5_GPU_KDA/MHC/FFN/MLA=0`) and find the first config that diverges. Bug 3 was
   found this way (mHC-on-CPU matched the oracle, mHC-on-GPU didn't). Per-component shadows
   make the bisect unnecessary for anything that has an adapter.
5. **Never skip silently.** Under `TANG_REQUIRE_CUDA=1` a GPU test that can't run *fails*.
   That includes a build without the `cuda` feature. Without the env var it prints a loud
   banner.

## The four bugs under the harness

- **Bug 1 (KDA byte offsets)** is caught in one window, at write time *and* at run time.
  The `k5_gemv` kernel test (`kda_dev.rs`, `k5_gemv_stacked_qkv_matches_cpu`) launches
  q/k/v into a poisoned stacked buffer exactly as the engine does. With the bug planted it fails
  every shape, and the verdict names `k`'s writer with the ×4 diagnosis
  (`k5_gemv_element_offset_bug_is_named` pins this). On the real model, see the demo below.
- **Bug 2 (one MLA for all DSA layers)**: a per-layer shadow (unit = layer) runs layer 7's
  CPU MLA with layer 7's weights. The GPU ran layer 3's, so the q/kv projections of every DSA
  layer after the first fail while layer 3 passes. `VERDICT … l7: stage kv_a fails with its inputs
  passing` on exactly the layers after the first is the per-layer-state signature. Needs the
  MLA adapter (next to port; the API is the same as KDA's).
- **Bug 3 (mHC block index)**: unit = (layer, block). Every block-1 window fails and every
  block-0 window passes. In a kernel test, put the block in the case grid
  (`.dtypes(&[0, 1])`): the fixture-blind "only block 0 tested" can't happen.
- **Bug 4 (teardown)** is not numeric, so the harness doesn't detect it directly. What it does
  is remove the silent skip that hid it: `kernel_test!` runs the full alloc→launch→drop
  lifecycle in-process and fails under `TANG_REQUIRE_CUDA=1` instead of passing. Run GPU tests
  under `compute-sanitizer` on the pod to catch drop-order faults.

## Demo: a planted bug, caught in one run

`scripts/kernel-verify-demo.sh` (on a CUDA box with the GLM-5.3 GGUF; mew's 3090 fits KDA
with the FFNs on the CPU):

1. copies the tree and plants bug 1 (reverts the byte scaling of cc578187);
2. builds both binaries, runs the real model once with `TANG_GLM5_KDA_DEBUG=1`, planted and
   fixed;
3. prints the one-screen verdict. `--replay` reprints it from the saved logs.

Transcript (H200 pod, GLM-5.3-Flash UD-Q4_K_XL, real weights; `-n 1` keeps it to the 20-token prefill window, ~5 min):

```

  ━━ tang-verify: a planted GPU bug, caught in one run ━━

  1. The bug we planted (crates/tang-llm/src/flash/glm5/kda_dev.rs, the q/k/v projection launch)
    fixed: [(&self.wq, 0u64), (&self.wk, di as u64 * f), (&self.wv, 2 * di as u64 * f)] {
    planted: [(&self.wq, 0u64), (&self.wk, di as u64), (&self.wv, 2 * di as u64)] {
    A GPU pointer counts BYTES; the planted line adds a count of FLOATS (4 bytes each).
    So k is written 1/4 of the way, and v 1/2, into the q block. This exact bug shipped
    in the GLM-5.3 engine on Oct 26 and was found by this method (fix: cc578187).

  2. One run of the real model with the harness on (GLM-5.3, real weights, layer 0, first window: 20 tokens)
    Every output buffer is filled with NaN first, so memory no kernel wrote stays NaN.
      qkv.q  WRONG         0 of 163840 values never written
      qkv.k  WRONG     81920 of 163840 values never written
      qkv.v  WRONG    163840 of 163840 values never written
    Predicted by the offset arithmetic (20 tokens × 8192 floats per section):
      k: v's misplaced writes stop halfway through k → 20 × 8192/2 = 81920 unwritten  ✓
      v: nothing writes v's block at all             → 20 × 8192   = 163840 unwritten  ✓
    Harness verdict:
      `qkv.k` wrote into `qkv.q`: qkv.q@tok0 h16 d0 holds qkv.k@tok0 h0 d0
            (writer's base offset in `qkv_pre` is 2048, expected 8192:
             expected = 4 × actual: an ELEMENT offset added to a BYTE address (size_of::<f32>()))
          → suspect: k5_gemv attn_k (device)
    (the naive rule, "first failing stage", would have blamed attn_q: its block is the
     first to fail, but the values in it are k's)

  3. The same run with the fix
    SUMMARY: 34 windows checked, 0 failed; first: none (all within tol)
    ids: 785

  logs: /root/w-verify-demo/planted.log, /root/w-verify-demo/fixed.log
  DEMO_RC=0
```

## The API

```toml
tang-verify = { workspace = true }            # core: no deps, runs anywhere
# feature "cuda": the device side on tang-moe's DevBuf/Gpu
```

### A kernel test, in a few lines

```rust
use tang_verify::{Cases, Layout, Spec, Tap, Taps};

tang_verify::kernel_test!(gemv8_matches_cpu, {
    let Some(g) = tang_verify::device::gpu_or_skip("gemv8") else { return };
    let k = Kernels::new(&g).unwrap();
    Cases::new("gemv8")
        .shapes(&[(1, 4096, 4096), (20, 4096, 512), (64, 8192, 4096)])
        .dtypes(&[GgmlType::Q8_0, GgmlType::Q4K])
        .run(|&(t, rows, cols), &dt, p| {
            let (w, x) = make_inputs(t, rows, cols, dt);
            let y = p.output(t * rows)?;                    // NaN-poisoned
            k.gemv8(&w, &x, &y, t, rows, cols)?;
            p.read("y", &y, t * rows)?;                     // the device tap
            let mut cpu = Taps::new();
            cpu.put("y", host_gemv(&w, &x, t, rows, cols)); // the oracle
            let lay = Layout::new().dim("tok", 0).dim("row", rows);
            Ok((Spec::new("gemv8").tap(Tap::new("y", lay).what("gemv8")), cpu))
        });
});
```

The grid is shapes × dtypes, with poison on. Every element is compared, and the panic lists
every failing case with its verdict. Without the invoking crate's `cuda` feature the test
calls `skip_or_fail`. `Cases::check` returns the failures instead of panicking, which is how
a test asserts that a planted bug *is* caught.

### Engine-side shadow checks

```rust
// setup (once): None unless TANG_KERNEL_VERIFY=1|mykernel or the legacy switch
let v: Option<Verifier> = Verifier::from_env("mykernel", Some("TANG_GLM5_MY_DEBUG"));

// per checked window
if let Some(v) = v.as_mut().filter(|v| v.wants(layer)) {
    dev.probe = Some(Probe::new(v.cfg.poison)); // device wrapper: p.poison(&[..]), p.read(..)
    let y = dev.run(x)?;
    let gpu = dev.probe.take().unwrap().taps;
    let mut cpu = Taps::new();
    cpu_reference(x, &state_from_device, Some(&mut cpu)); // tang_verify::tap(&mut taps, name, || v.clone())
    v.check(layer, &format!("t{t}"), &spec, &gpu, &cpu);   // prints rows, VERDICT, dumps
}
```

Off, the `Option<Verifier>` is `None` and nothing runs. Hot paths that need a check without an
adapter use `tang_verify::on()` (a cached atomic load).

### Types

| type | what |
|------|------|
| `Stat`, `stat(gpu, cpu, tol)` | rel-L2, max abs + index (NaN wins it), norms, cosine, zero and NaN counts, bad elements; `bad(tol)` |
| `Layout` | named row-major dims (`dim("tok", 0)` = unbounded leading dim; `labeled("sec", &["q","k","v"])`); `locate(i)` |
| `Tap` | name, layout, `after(inputs)`, `what(writer)`, `within(buffer, base, len, stride)` |
| `Spec` | taps in pipeline order; `stacked(buffer, sections, layout, inputs)` |
| `Taps` | named host tensors (one side); `put`, `put_at` (chunked; gaps stay NaN) |
| `Probe` | device side: `poison(&[&DevBuf])`, `output(n)`, `read(name, buf, n)`, `read_at` |
| `Verifier` | the session: `wants`, `check → Report`, dumps, `SUMMARY` on drop |
| `Verdict`, `Cause` | `Misplaced{victim, at, from, actual_base, expected_base}`, `Unwritten`, `Zeroed`, `Wrong`, `NoCleanInput` |
| `Cases`, `kernel_test!` | the test grid |
| `skip_or_fail`, `device::gpu_or_skip` | the no-silent-skip rule |

### Switches

| env | effect |
|-----|--------|
| `TANG_KERNEL_VERIFY=1` / `=kda,mhc` | all kernels / these |
| `TANG_KERNEL_VERIFY_{TOL,LAYERS,WINDOWS,RESYNC,POISON,DIR,DUMPS}` | knobs (tol 1e-3, resync on, poison on, 3 dumps) |
| `{LEGACY}`, `{LEGACY}_{KNOB}` | an adapter's own switch, e.g. `TANG_GLM5_KDA_DEBUG=1`, `TANG_GLM5_KDA_DEBUG_LAYERS=0,1`; these win over the global knobs |
| `TANG_REQUIRE_CUDA=1` | GPU tests fail instead of skip |

Dumps: `DIR/w{W}_{unit}{U}_{tap}.{gpu,cpu}.f32` (raw LE f32) for the first window and the first
`DUMPS` failing windows. The global `TANG_KERNEL_VERIFY_DIR` gets a per-kernel subdirectory.

### Output

```
kdadbg w0 l0 t20 qkv.q       ok  rel 9.706e-7 max 7.153e-6 @tok7 h46 d36 |gpu| 8.7909e1 |cpu| 8.7909e1 cos 1.000000 zeros 0/0 nonfinite 0/0 bad 0/163840
kdadbg VERDICT w0 l0 t20: `qkv.k` wrote into `qkv.q`: …
kdadbg FIRST DIVERGENCE …                     (once: the first failing window)
kdadbg SUMMARY: 272 windows checked, 0 failed; first: none (all within tol)
```

## The KDA port (the proof the API fits)

`crates/tang-llm/src/flash/glm5/kda_debug.rs` is now an adapter:

- `spec()` declares the layer's 17 taps: `state.in`; q|k|v as sections of `qkv_pre`; g; beta;
  conv q|k|v as sections of `conv_out`; o; g2; y_norm; out; state; conv_state; conv_host. Each
  has its inputs and writer.
- `Shadow` keeps the per-layer CPU state (prime/resync).
- `kda_dev::DevLayer::window` fills a `Probe` (poisons ten scratch buffers, reads taps,
  including the per-chunk conv output).
- `kda::kda_layer_taps` fills `Taps`.

Every `TANG_GLM5_KDA_DEBUG*` env var and the `kdadbg` line format are unchanged.

Verified on the real model (mew, RTX 3090, KDA + DSA + mHC on the GPU, FFN on the CPU,
`TANG_GLM5_KDA_MAXT=64`, "Write a haiku about the sea." `-n 8 --temp 0`):
ids `785 1196 6801 264 6386 38038 911 279` (= the CPU oracle),
`kdadbg SUMMARY: 272 windows checked, 0 failed`, worst rel 2.8e-6 (34 layers × 8 windows).

## Adopting it for a new kernel (checklist)

1. Write the CPU reference first (it is the oracle; f64 accumulate where the host does).
2. `Spec`: one `Tap` per intermediate you can read back, in pipeline order, with `after` and
   `what`. Outputs sharing a buffer → `stacked`/`within`.
3. Kernel test with `kernel_test!` + `Cases`: shapes that cover the tails (t = 1, a non-multiple
   of the block, the max), every dtype, every variant index (block 0 *and* 1, every layer kind).
4. Engine adapter if the kernel runs in a model: `Option<Verifier>` + `Probe` + resync.
5. CI/pods run tests with `TANG_REQUIRE_CUDA=1`.
