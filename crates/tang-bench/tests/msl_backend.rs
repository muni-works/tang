//! The native Metal backend against the f64 interpreter (and against
//! `eval_emulated` for f16 schedules), for every emitter path: one thread
//! per point, workgroup split, tiles with register tiles, multi-dim
//! reductions with guarded/dilated windowed staging, epilogue fusion,
//! online-softmax attention, backward graphs, and fragment lowering.
//! Random inputs, every output element compared.
#![cfg(feature = "metal")]

use tang::Scalar;
use tang_bench::ir_msl::{Executable, MslDevice};
use tang_compile::tensor::{Arg, Form, Format, Monoid, Precision, TensorGraph, TensorId, View};

fn dev() -> Option<MslDevice> {
    let d = MslDevice::new();
    if d.is_none() {
        eprintln!("no Metal device, skipping");
    }
    d
}

fn rand(n: usize, rng: &mut tang_bench::Rng) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn f64s(v: &[Vec<f32>]) -> Vec<Vec<f64>> {
    v.iter()
        .map(|x| x.iter().map(|&y| y as f64).collect())
        .collect()
}

/// Run on Metal and compare every element of every output with `want`,
/// relative to each output's largest magnitude.
fn check(
    dev: &MslDevice,
    name: &str,
    g: &TensorGraph,
    outs: &[TensorId],
    inputs: &[Vec<f32>],
    want: &[Vec<f64>],
    tol: f64,
) -> Executable {
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let exe = Executable::new(dev, g.to_msl(outs), &refs);
    exe.run(dev);
    for (o, w) in outs.iter().zip(want) {
        let got = exe.read(*o);
        assert_eq!(got.len(), w.len(), "{name}: length");
        // an all-but-zero output (a masked gradient) is judged absolutely
        let scale = w.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(w).enumerate() {
            assert!(
                (*a as f64 - b).abs() <= tol * scale,
                "{name} out {}[{i}]: {a} vs {b} (scale {scale})",
                o.index()
            );
        }
    }
    exe
}

fn matmul_graph(m: usize, k: usize, n: usize) -> (TensorGraph, TensorId) {
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let c = g.matmul(a, b).unwrap();
    (g, c)
}

type Mapping = (&'static str, u32, (u32, u32, u32), (u32, u32));

const MAPPINGS: [Mapping; 6] = [
    ("point", 0, (0, 0, 0), (1, 1)),
    ("split", 64, (0, 0, 0), (1, 1)),
    ("tile", 0, (16, 16, 32), (1, 1)),
    ("tile 2x2", 0, (16, 16, 32), (2, 2)),
    ("tile 4x4", 0, (16, 16, 32), (4, 4)),
    ("rows", 0, (64, 1, 8), (1, 1)),
];

#[test]
fn matmul_every_mapping_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(1);
    for (m, k, n) in [(100, 100, 100), (128, 576, 1536), (70, 50, 90)] {
        let (g, c) = matmul_graph(m, k, n);
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[c]);
        for (name, split, tile, reg) in MAPPINGS {
            let mut gv = g.clone();
            gv.set_schedule(c, split, tile, reg);
            check(
                &dev,
                &format!("{m}x{k}x{n} {name}"),
                &gv,
                &[c],
                &inputs,
                &want,
                1e-5,
            );
        }
    }
}

/// f16 buffers are native `half`; f16 accumulators round every combine.
/// The interpreter's emulation predicts both to within an f16 ulp or two.
#[test]
fn f16_storage_and_accum_match_emulation() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(7);
    let (m, k, n) = (48, 96, 80);
    let inputs = vec![
        rand(m * k, &mut rng),
        rand(k * n, &mut rng),
        rand(n, &mut rng),
    ];
    let half = Precision::storage(Format::F16);
    for accum in [Format::F32, Format::F16] {
        for (name, split, tile, reg) in MAPPINGS {
            let mut g = TensorGraph::new();
            let ga = g.input(&[m, k]);
            let gb = g.input(&[k, n]);
            let gbias = g.input(&[1, n]);
            let y = g.matmul(ga, gb).unwrap();
            let bb = g.broadcast(gbias, &[m, n]).unwrap();
            let z = g.map(&[y, bb], |v| v[0] * v[0] + v[1]).unwrap();
            for t in [ga, gb, gbias, z] {
                g.set_precision(t, half);
            }
            g.set_precision(
                y,
                Precision {
                    storage: Format::F16,
                    accum,
                    operand: Format::F32,
                },
            );
            g.set_schedule(y, split, tile, reg);
            let emu = g.eval_emulated(&f64s(&inputs), &[y, z]);
            // the emulation sums in a fixed order; the GPU's split/tile
            // order differs, so allow a couple of f16 ulps of the result
            let tol = if accum == Format::F16 { 2e-2 } else { 2e-3 };
            check(
                &dev,
                &format!("f16 {accum:?} {name}"),
                &g,
                &[y, z],
                &inputs,
                &emu,
                tol,
            );
        }
    }
}

/// Direct conv `y[o,t] = Σ_{c,k} w[o,c,k]·x[c, t + d·k − p]`, guarded.
fn conv_graph(ci: usize, co: usize, t: usize, k: usize, dil: i64) -> (TensorGraph, TensorId) {
    let pad = dil * (k as i64 - 1) / 2;
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let w = g.input(&[co, ci, k]);
    let xv = View {
        forms: vec![
            Form::unit(4, 2),
            Form {
                coefs: vec![0, 1, 0, dil],
                offset: -pad,
            },
        ],
        guard: true,
        gather: None,
    };
    let y = g
        .comp(
            &[co, t],
            &[ci, k],
            vec![
                Arg { src: x, view: xv },
                Arg {
                    src: w,
                    view: View::pick(4, &[Some(0), Some(2), Some(3)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    (g, y)
}

#[test]
fn conv_guarded_dilated_and_backward_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(11);
    for &(ci, co, t, k, dil) in &[(5, 7, 37, 3, 1), (40, 33, 70, 7, 3), (19, 64, 130, 5, 2)] {
        let (g, y) = conv_graph(ci, co, t, k, dil);
        let inputs = vec![rand(ci * t, &mut rng), rand(co * ci * k, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[y]);
        for (name, split, tile, reg) in MAPPINGS {
            let mut gv = g.clone();
            gv.set_schedule(y, split, tile, reg);
            let name = format!("conv {ci}x{co}x{t} k{k} d{dil} {name}");
            check(&dev, &name, &gv, &[y], &inputs, &want, 1e-5);
        }
        // forward + backward, auto-scheduled, then every candidate per node
        let mut gb = g.clone();
        let grads = gb
            .backward(y, &[TensorId::from_index(0), TensorId::from_index(1)])
            .unwrap();
        let mut outs = vec![y];
        outs.extend(grads);
        let want = gb.eval(&f64s(&inputs), &outs);
        let (gb, outs) = gb.substitute_maps(&outs);
        let (mut gb, outs) = gb.eliminate_common(&outs);
        gb.schedule_auto(&outs);
        check(&dev, "conv fwd+bwd auto", &gb, &outs, &inputs, &want, 1e-4);
        for o in &outs {
            for (split, tile, reg) in gb.schedule_candidates(*o) {
                let mut gv = gb.clone();
                gv.set_schedule(*o, split, tile, reg);
                let name = format!("bwd node {} {split} {tile:?} {reg:?}", o.index());
                check(&dev, &name, &gv, &outs, &inputs, &want, 1e-4);
            }
        }
    }
}

/// A tiled conv with a fused epilogue (bias + snake), f32.
#[test]
fn conv_epilogue_fused_f32() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(5);
    let (ci, co, t, k, dil) = (32, 48, 100, 7, 3);
    let (mut g, y) = conv_graph(ci, co, t, k, dil);
    let bias = g.input(&[co, 1]);
    let bb = g.broadcast(bias, &[co, t]).unwrap();
    let z = g
        .map(&[y, bb], |v| {
            let s = v[0] + v[1];
            s + s.sin() * s.sin()
        })
        .unwrap();
    let inputs = vec![
        rand(ci * t, &mut rng),
        rand(co * ci * k, &mut rng),
        rand(co, &mut rng),
    ];
    let want = g.eval(&f64s(&inputs), &[z]);
    let (g, outs) = g.substitute_maps(&[z]);
    let mut g = g;
    g.schedule_auto(&outs);
    let z = outs[0];
    let exe = check(&dev, "conv epilogue", &g, &[z], &inputs, &want, 1e-5);
    assert_eq!(exe.program.kernels.len(), 1, "epilogue should fuse");
}

/// Online-softmax attention (one fused reduction) under every mapping
/// the WGSL path runs, plus the materialized naive form.
#[test]
fn attention_online_softmax_f32() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_gpu::{flash_attention, naive_attention};
    let mut rng = tang_bench::Rng::new(9);
    for (n, d) in [(64, 16), (130, 32)] {
        let inputs = vec![
            rand(n * d, &mut rng),
            rand(n * d, &mut rng),
            rand(n * d, &mut rng),
        ];
        let (g, y) = naive_attention(n, d, d);
        let want = g.eval(&f64s(&inputs), &[y]);
        check(&dev, "naive", &g, &[y], &inputs, &want, 1e-5);
        let (g, y) = flash_attention(n, d, d);
        check(&dev, "flash point", &g, &[y], &inputs, &want, 1e-5);
        let mut gs = g.clone();
        gs.schedule_split(&[y], 256, 64);
        check(&dev, "flash split", &gs, &[y], &inputs, &want, 1e-5);
        let mut gt = g.clone();
        gt.schedule_tile_2d(&[y], d as u32, 4, 32, 64);
        check(&dev, "flash tiled", &gt, &[y], &inputs, &want, 1e-5);
        let (ga, ya) = tang_bench::ir_gpu::flash_attention_auto(n, d, d);
        check(&dev, "flash auto", &ga, &[ya], &inputs, &want, 1e-5);
    }
}

/// The transformer block's backward graph on Metal vs the interpreter.
#[test]
fn block_backward_f32() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_block::{transformer_block, BlockDims};
    let dims = BlockDims {
        seq: 12,
        kv_len: 12,
        rope: true,
        bias: true,
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        ff_dim: 48,
    };
    let (mut g, out) = transformer_block(&dims);
    let mut rng = tang_bench::Rng::new(31);
    let n_inputs = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, tang_compile::tensor::Op::Input(_)))
        .count();
    let inputs: Vec<Vec<f32>> = (0..n_inputs)
        .map(|k| {
            let node = g
                .nodes()
                .iter()
                .find(|n| matches!(n.op, tang_compile::tensor::Op::Input(j) if j == k))
                .unwrap();
            let len: usize = node.shape.iter().map(|d| d.n()).product();
            (0..len).map(|_| (rng.f32() * 2.0 - 1.0) * 0.5).collect()
        })
        .collect();
    let (xi, wq) = (TensorId::from_index(0), TensorId::from_index(1));
    let grads = g.backward(out, &[xi, wq]).unwrap();
    let want = g.eval(&f64s(&inputs), &grads);
    let (g, outs) = g.substitute_maps(&grads);
    let (mut g, outs) = g.eliminate_common(&outs);
    g.schedule_auto(&outs);
    check(&dev, "block backward", &g, &outs, &inputs, &want, 1e-4);
}

// ---------------------------------------------------------------------------
// fragment lowering
// ---------------------------------------------------------------------------

fn f16_operands(g: &mut TensorGraph, ts: &[TensorId]) {
    for &t in ts {
        g.set_precision(t, Precision::storage(Format::F16));
    }
}

#[test]
fn fragment_matmul_ragged() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(21);
    for (m, k, n) in [
        (100, 100, 100),
        (128, 576, 1536),
        (64, 64, 64),
        (7, 300, 33),
    ] {
        let (mut g, c) = matmul_graph(m, k, n);
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        assert!(
            g.fragment_plan(c).is_none(),
            "f32 operands must not get fragments"
        );
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        assert!(g.fragment_plan(c).is_some());
        g.set_fragment(c, true);
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        let exe = check(
            &dev,
            &format!("frag {m}x{k}x{n}"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-5,
        );
        assert!(exe.program.kernels[0].fragment);
        // f16 output
        g.set_precision(c, Precision::storage(Format::F16));
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        check(
            &dev,
            &format!("frag {m}x{k}x{n} f16 out"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-3,
        );
    }
}

#[test]
fn fragment_batched_matmul() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(22);
    let (bt, m, k, n) = (3, 70, 40, 90);
    let mut g = TensorGraph::new();
    let a = g.input(&[bt, m, k]);
    let b = g.input(&[bt, k, n]);
    let c = g
        .comp(
            &[bt, m, n],
            &[k],
            vec![
                Arg {
                    src: a,
                    view: View::pick(4, &[Some(0), Some(1), Some(3)]),
                },
                Arg {
                    src: b,
                    view: View::pick(4, &[Some(0), Some(3), Some(2)]),
                },
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    f16_operands(&mut g, &[a, b]);
    g.set_fragment(c, true);
    let inputs = vec![rand(bt * m * k, &mut rng), rand(bt * k * n, &mut rng)];
    let emu = g.eval_emulated(&f64s(&inputs), &[c]);
    let exe = check(&dev, "batched frag", &g, &[c], &inputs, &emu, 1e-5);
    assert!(exe.program.kernels[0].fragment);
}

/// Conv as implicit GEMM: `w[o, (c,k)] · x[(c,k), t]`, the `x` operand
/// gathered through its guarded, dilated view (zero padding at staging),
/// with a fused bias epilogue.
#[test]
fn fragment_conv_implicit_gemm_with_epilogue() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(23);
    for &(ci, co, t, k, dil) in &[(5, 7, 37, 3, 1), (40, 33, 70, 7, 3), (64, 128, 200, 7, 1)] {
        let (mut g, y) = conv_graph(ci, co, t, k, dil);
        let bias = g.input(&[co, 1]);
        let bb = g.broadcast(bias, &[co, t]).unwrap();
        let z = g.map(&[y, bb], |v| v[0] + v[1]).unwrap();
        let inputs = vec![
            rand(ci * t, &mut rng),
            rand(co * ci * k, &mut rng),
            rand(co, &mut rng),
        ];
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        let emu = g.eval_emulated(&f64s(&inputs), &[z]);
        // unfused: the conv itself on fragments
        let mut gu = g.clone();
        gu.set_fragment(y, true);
        let exe = check(&dev, "frag conv", &gu, &[z], &inputs, &emu, 1e-5);
        assert!(exe.program.kernels.iter().any(|k| k.fragment));
        // fused: the map takes the conv as its epilogue core
        let mut gf = g.clone();
        gf.set_inline(y, true);
        gf.set_fragment(z, true);
        assert!(gf.fragment_plan(z).is_some(), "epilogue plan");
        let exe = check(&dev, "frag conv epilogue", &gf, &[z], &inputs, &emu, 1e-5);
        // the conv has no kernel of its own: it is the map's epilogue core
        assert!(exe.program.kernels.iter().all(|k| k.node != y));
        assert!(exe
            .program
            .kernels
            .iter()
            .any(|k| k.node == z && k.fragment));
    }
}

/// Transposed operands take the transposed device-load path (`a` stored
/// `[k, m]`, `b` stored `[n, k]`: q·kᵀ's layout), interior and ragged.
#[test]
fn fragment_transposed_operands() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(24);
    for (m, k, n) in [(128, 64, 256), (100, 72, 90)] {
        let mut g = TensorGraph::new();
        let at = g.input(&[k, m]);
        let bt = g.input(&[n, k]);
        let a = g.transpose(at, &[1, 0]).unwrap();
        let b = g.transpose(bt, &[1, 0]).unwrap();
        let c = g.matmul(a, b).unwrap();
        let inputs = vec![rand(m * k, &mut rng), rand(k * n, &mut rng)];
        let want = g.eval(&f64s(&inputs), &[c]);
        let (mut g, outs) = g.substitute_maps(&[c]);
        let c = outs[0];
        f16_operands(&mut g, &[TensorId::from_index(0), TensorId::from_index(1)]);
        g.set_fragment(c, true);
        assert!(g.fragment_plan(c).is_some());
        let emu = g.eval_emulated(&f64s(&inputs), &[c]);
        let exe = check(
            &dev,
            &format!("frag T {m}x{k}x{n}"),
            &g,
            &[c],
            &inputs,
            &emu,
            1e-5,
        );
        assert!(exe.program.kernels.iter().any(|k| k.fragment));
        let _ = want;
    }
}

/// The backend dimension: the same graph through WGSL and MSL, both
/// checked against the f64 reference.
#[test]
fn pick_backend_runs_both() {
    let Some(dev) = dev() else { return };
    let Ok(wgpu) = tang_gpu::GpuDevice::new_sync() else {
        return;
    };
    let mut cache = tang_gpu::KernelCache::new();
    let mut rng = tang_bench::Rng::new(25);
    let (mut g, c) = matmul_graph(96, 80, 112);
    g.schedule_auto(&[c]);
    let inputs = vec![rand(96 * 80, &mut rng), rand(80 * 112, &mut rng)];
    let want = g.eval(&f64s(&inputs), &[c]);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let (_, w, m) = tang_bench::ir_msl::pick_backend(
        &wgpu,
        &mut cache,
        &dev,
        &g,
        &[c],
        &refs,
        &want,
        tang_compile::tensor::ErrorBudget::rel(1e-5),
        10,
    );
    assert!(w.is_some() && m.is_some());
}

// ---------------------------------------------------------------------------
// windowed fragment staging
// ---------------------------------------------------------------------------

/// Strided, dilated conv `y[o,τ] = Σ_{c,k} w[o,c,k]·x[c, s·τ + d·k − p]`,
/// guarded, as one comp.
fn conv_strided(
    g: &mut TensorGraph,
    x: TensorId,
    w: TensorId,
    tout: usize,
    stride: i64,
    dil: i64,
    pad: i64,
) -> TensorId {
    let (ci, co, k) = (g.shape(x)[0].n(), g.shape(w)[0].n(), g.shape(w)[2].n());
    let xv = View {
        forms: vec![
            Form::unit(4, 2),
            Form {
                coefs: vec![0, stride, 0, dil],
                offset: -pad,
            },
        ],
        guard: true,
        gather: None,
    };
    g.comp(
        &[co, tout],
        &[ci, k],
        vec![
            Arg { src: x, view: xv },
            Arg {
                src: w,
                view: View::pick(4, &[Some(0), Some(2), Some(3)]),
            },
        ],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

fn windowed(exe: &Executable) -> bool {
    exe.program
        .kernels
        .iter()
        .any(|k| k.fragment && k.source.contains("windowed operand"))
}

/// The conv's `x` staged once per (channel chunk, time tile) and read by
/// every tap at a shifted offset: ragged time and channels, dilation,
/// stride (phase-de-interleaved window), padding past both edges.
#[test]
fn fragment_window_conv_variants() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(31);
    // (ci, co, t, k, stride, dil, pad)
    for &(ci, co, t, k, st, dil, pad) in &[
        (5, 7, 37, 3, 1, 1, 1),
        (40, 33, 70, 7, 1, 3, 9),
        (64, 128, 200, 7, 1, 1, 3),
        (24, 72, 131, 5, 2, 1, 2),
        (19, 20, 300, 4, 3, 2, 5),
        (32, 64, 129, 3, 1, 5, 0),
    ] {
        let tout = ((t as i64 + 2 * pad - dil * (k as i64 - 1) - 1) / st + 1) as usize;
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, k]);
        let y = conv_strided(&mut g, x, w, tout, st, dil, pad);
        f16_operands(&mut g, &[x, w]);
        g.set_fragment(y, true);
        let inputs = vec![rand(ci * t, &mut rng), rand(co * ci * k, &mut rng)];
        let emu = g.eval_emulated(&f64s(&inputs), &[y]);
        let name = format!("window conv {ci}->{co} t{t} k{k} s{st} d{dil} p{pad}");
        let exe = check(&dev, &name, &g, &[y], &inputs, &emu, 1e-5);
        assert!(windowed(&exe), "{name}: not windowed");
    }
}

/// Polyphase upsample `out[o, q, r] = Σ_{j,c} x[c, q − j]·w[o, c, r + s·j]`:
/// the phase `r` is a batch dim of the fragment tile, the tap `j` reads
/// the window backwards.
#[test]
fn fragment_window_upsample() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(32);
    for &(ci, co, t, s) in &[(16, 24, 50, 3), (40, 64, 130, 4)] {
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, 2 * s]);
        let f = |terms: &[(usize, i64)]| {
            let mut fm = Form::zero(5);
            for &(d, c) in terms {
                fm.coefs[d] = c;
            }
            fm
        };
        let xv = View {
            forms: vec![f(&[(4, 1)]), f(&[(1, 1), (3, -1)])],
            guard: true,
            gather: None,
        };
        let wv = View {
            forms: vec![f(&[(0, 1)]), f(&[(4, 1)]), f(&[(2, 1), (3, s as i64)])],
            guard: false,
            gather: None,
        };
        let y = g
            .comp(
                &[co, t, s],
                &[2, ci],
                vec![Arg { src: x, view: xv }, Arg { src: w, view: wv }],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        f16_operands(&mut g, &[x, w]);
        g.set_fragment(y, true);
        let inputs = vec![rand(ci * t, &mut rng), rand(co * ci * 2 * s, &mut rng)];
        let emu = g.eval_emulated(&f64s(&inputs), &[y]);
        let exe = check(&dev, "window upsample", &g, &[y], &inputs, &emu, 1e-5);
        assert!(windowed(&exe));
    }
}

/// Backward of `Σ y ⊙ gy` through the conv on fragments: dx is a windowed
/// transposed conv over gy (its weight read through an always-in-range
/// guarded view), dw a GEMM over t with `x` staged along t. Both carry an
/// unread third arg (the product rule's dropped factor).
#[test]
fn fragment_conv_backward() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(33);
    for &(ci, co, t, k, dil) in &[(24, 40, 77, 3, 2), (64, 64, 150, 7, 3)] {
        let pad = dil * (k as i64 - 1) / 2;
        let mut g = TensorGraph::new();
        let x = g.input(&[ci, t]);
        let w = g.input(&[co, ci, k]);
        let gy = g.input(&[co, t]);
        let y = conv_strided(&mut g, x, w, t, 1, dil, pad);
        let l = g.map(&[y, gy], |v| v[0] * v[1]).unwrap();
        let grads = g.backward(l, &[x, w]).unwrap();
        let outs = vec![y, grads[0], grads[1]];
        let (g, outs) = g.substitute_maps(&outs);
        let (mut g, outs) = g.eliminate_common(&outs);
        let inputs = vec![
            rand(ci * t, &mut rng),
            rand(co * ci * k, &mut rng),
            rand(co * t, &mut rng),
        ];
        let cands = tang_bench::ir_msl::fragment_candidates(&g, &outs);
        assert!(
            cands.contains(&outs[1]) && cands.contains(&outs[2]),
            "dx, dw eligible: {cands:?}"
        );
        for &c in &cands {
            for o in g.fragment_operands(c).unwrap() {
                g.set_precision(o, Precision::storage(Format::F16));
            }
            g.set_fragment(c, true);
        }
        let emu = g.eval_emulated(&f64s(&inputs), &outs);
        let exe = check(&dev, "frag conv backward", &g, &outs, &inputs, &emu, 1e-5);
        for o in &outs[1..] {
            assert!(exe
                .program
                .kernels
                .iter()
                .any(|k| k.node == *o && k.fragment));
        }
        assert!(windowed(&exe), "dx should be windowed");
    }
}

/// A matmul whose operand is an inlined map (ConvNeXt's second pointwise:
/// `w2 · gelu(h + b1)`): the map is evaluated while staging and its value
/// rounded to f16 in threadgroup memory. Ragged shapes, with the map on
/// either side; checked against f64 at f16-operand tolerance (the
/// emulator does not round inlined values).
#[test]
fn fragment_inlined_map_operand() {
    let Some(dev) = dev() else { return };
    let mut rng = tang_bench::Rng::new(34);
    for &(m, k, n, map_on_b) in &[
        (40, 96, 70, true),
        (128, 200, 64, true),
        (70, 33, 100, false),
    ] {
        let mut g = TensorGraph::new();
        let (ws, hs) = if map_on_b {
            ([m, k], [k, n])
        } else {
            ([k, n], [m, k])
        };
        let w = g.input(&ws);
        let h = g.input(&hs);
        let bias = g.input(&[1, 1]);
        let bb = g.broadcast(bias, &hs).unwrap();
        let act = g
            .map(&[h, bb], |v| (v[0] + v[1]) * (v[0] + v[1]).sin())
            .unwrap();
        let c = if map_on_b {
            g.matmul(w, act)
        } else {
            g.matmul(act, w)
        }
        .unwrap();
        g.set_inline(act, true);
        f16_operands(&mut g, &[w]);
        assert!(g.fragment_plan(c).is_some(), "inlined map operand plan");
        g.set_fragment(c, true);
        let inputs = vec![
            rand(g.shape(w).iter().map(|d| d.n()).product(), &mut rng),
            rand(hs.iter().product(), &mut rng),
            vec![0.25],
        ];
        // reference: f64 with the f16 weight
        let want = g.eval_emulated(&f64s(&inputs), &[c]);
        let exe = check(&dev, "frag inlined map", &g, &[c], &inputs, &want, 3e-3);
        assert!(exe
            .program
            .kernels
            .iter()
            .any(|kk| kk.node == c && kk.fragment));
        assert!(exe.program.kernels.iter().all(|kk| kk.node != act));
    }
}

/// Flash attention on simdgroup fragments, derived from the naive graph:
/// S = Q·Kᵀ and O += P·V on f16 fragments, the online softmax in
/// registers, the normalizer fused. Ragged rows/keys/head dims, causal
/// (with derived key-tile skipping) and not, multi-head and GQA. Judged
/// against the f16-storage emulation and the pure f64 reference.
#[test]
fn fragment_attention() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_gpu::flash_attention_heads_fragment;
    let mut rng = tang_bench::Rng::new(41);
    // (kv heads, query heads per kv head, n, d, dv, causal)
    for (hk, gq, n, d, dv, causal) in [
        (1, 1, 100, 64, 64, true),
        (1, 1, 100, 64, 64, false),
        (1, 1, 1000, 64, 64, true),
        (1, 1, 1000, 64, 64, false),
        (1, 1, 512, 64, 64, true),
        (1, 1, 512, 64, 64, false),
        (2, 3, 130, 64, 64, true),
        (2, 3, 130, 64, 64, false),
        (4, 1, 96, 32, 32, true),
        (1, 1, 77, 40, 24, true),
        (1, 2, 70, 48, 136, false),
        // full tiles: K/V fragments straight from device memory
        (2, 2, 256, 64, 64, true),
        (2, 2, 256, 64, 64, false),
        (1, 1, 128, 64, 128, false),
    ] {
        let name = format!("attn hk{hk} g{gq} n{n} d{d} dv{dv} causal {causal}");
        let (g, y) = flash_attention_heads_fragment(hk, gq, n, d, dv, causal);
        assert!(g.attention_plan(y).is_some(), "{name}: no plan");
        assert_eq!(
            g.attention_plan(y).unwrap().mask.is_some(),
            causal,
            "{name}: mask"
        );
        let inputs = vec![
            rand(hk * gq * n * d, &mut rng),
            rand(hk * n * d, &mut rng),
            rand(hk * n * dv, &mut rng),
        ];
        // the interpreter recomputes the score per value column (n²·d·dv):
        // check it on the small shapes, use a direct f64 attention above
        let f = |r: bool| attention_ref(&inputs, hk, gq, n, d, dv, causal, r);
        if n * n * d * dv * hk * gq <= 200 * 200 * 64 * 64 {
            assert_close(
                &g.eval_emulated(&f64s(&inputs), &[y])[0],
                &f(true),
                1e-5,
                &name,
            );
        }
        // P is rounded to f16 before P·V (the emulation keeps it exact)
        let exe = check(&dev, &name, &g, &[y], &inputs, &[f(true)], 4e-3);
        assert!(
            exe.program.kernels.len() == 1 && exe.program.kernels[0].fragment,
            "{name}"
        );
        check(
            &dev,
            &format!("{name} vs f64"),
            &g,
            &[y],
            &inputs,
            &[f(false)],
            8e-3,
        );
    }
}

/// f32 operands are not eligible: the fragment flag falls back.
#[test]
fn fragment_attention_needs_f16() {
    use tang_bench::ir_gpu::fused_attention_heads;
    let (mut g, y) = fused_attention_heads(1, 1, 64, 32, 32, true);
    g.set_fragment(y, true);
    assert!(g.attention_plan(y).is_none());
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, name: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "{name} ref[{i}]: {x} vs {y}");
    }
}

/// Direct f64 GQA attention (inputs optionally rounded to f16 first, as
/// the f16-storage emulation does).
#[allow(clippy::too_many_arguments)]
fn attention_ref(
    inputs: &[Vec<f32>],
    hk: usize,
    gq: usize,
    n: usize,
    d: usize,
    dv: usize,
    causal: bool,
    round: bool,
) -> Vec<f64> {
    let r = |x: f32| {
        if round {
            Format::F16.round(x as f64)
        } else {
            x as f64
        }
    };
    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    let scale = 1.0 / (d as f64).sqrt();
    let mut y = vec![0.0; hk * gq * n * dv];
    let mut s = vec![0.0; n];
    for h in 0..hk {
        for gi in 0..gq {
            for i in 0..n {
                let qo = ((h * gq + gi) * n + i) * d;
                let mut mx = f64::NEG_INFINITY;
                for (j, sj) in s.iter_mut().enumerate() {
                    let ko = (h * n + j) * d;
                    let mut acc = 0.0;
                    for c in 0..d {
                        acc += r(q[qo + c]) * scale * r(k[ko + c]);
                    }
                    if causal && j > i {
                        acc += -1e30;
                    }
                    *sj = acc;
                    mx = mx.max(acc);
                }
                let mut l = 0.0;
                let yo = ((h * gq + gi) * n + i) * dv;
                for (j, sj) in s.iter().enumerate() {
                    let p = (sj - mx).exp();
                    l += p;
                    for c in 0..dv {
                        y[yo + c] += p * r(v[(h * n + j) * dv + c]);
                    }
                }
                for c in 0..dv {
                    y[yo + c] /= l;
                }
            }
        }
    }
    y
}

/// `tune_precision` offers the attention fragment kernel as a candidate
/// and keeps it under a 1% error budget against f64.
#[test]
fn tune_precision_picks_attention_fragments() {
    let Some(dev) = dev() else { return };
    use tang_bench::ir_gpu::flash_attention_heads_auto;
    use tang_bench::ir_msl::{fragment_candidates, tune_precision};
    use tang_compile::tensor::ErrorBudget;
    let (hk, gq, n, d) = (2, 4, 1024, 64);
    let (g, y) = flash_attention_heads_auto(hk, gq, n, d, d, true);
    assert_eq!(fragment_candidates(&g, &[y]), vec![y]);
    let mut rng = tang_bench::Rng::new(43);
    let inputs = vec![
        rand(hk * gq * n * d, &mut rng),
        rand(hk * n * d, &mut rng),
        rand(hk * n * d, &mut rng),
    ];
    let want = vec![attention_ref(&inputs, hk, gq, n, d, d, true, false)];
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let r = tune_precision(&dev, g, &[y], &refs, &want, ErrorBudget::rel(1e-2), 10);
    for t in &r.trials {
        eprintln!(
            "{} err {:.2e} ms {:?} kept {}",
            t.what, t.error.max_rel, t.ms, t.kept
        );
    }
    assert!(r.fragments.contains(&y), "fragment candidate not kept");
    assert!(r.error.max_rel <= 1e-2);
}
