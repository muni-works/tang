//! Kokoro/TTS-shaped ops derived from the tensor IR, kernel-only on the
//! GPU (5 warmups, 50 launches, one sync — same method as `kernel_only`
//! and `scripts/torch_kokoro_ops.py`). Every op is checked against a
//! straightforward f64 CPU reference first.
//!
//! The ops need zero padding, which the IR expresses as guarded views
//! (`View::guard`): a read outside the source returns 0.
//!
//!   cargo run --release -p tang-bench --example kokoro_ops
use std::time::Instant;

use tang::Scalar;
use tang_bench::ir_gpu::{autotune_kernels, tune_precision, Executable};
use tang_compile::tensor::{
    Arg, ErrorBudget, ErrorStats, Form, Format, Monoid, Op, Precision, TensorGraph, TensorId, View,
};
use tang_expr::ExprId;
use tang_gpu::{GpuDevice, KernelCache};

const ITERS: usize = 50;

fn time_exe(device: &GpuDevice, cache: &mut KernelCache, exe: &Executable, out: TensorId) -> f64 {
    for _ in 0..5 {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    let t = Instant::now();
    for _ in 0..ITERS {
        exe.run(device, cache);
    }
    cache.flush(device);
    let _ = exe.read(device, out);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// max |got − want| / max |want|
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()))
        / scale
}

/// `offset + Σ coef·coord` over a `rank`-dim space.
fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::zero(rank);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f.offset = offset;
    f
}

fn view(forms: Vec<Form>, guard: bool) -> View {
    View { forms, guard }
}

fn arg(src: TensorId, view: View) -> Arg {
    Arg { src, view }
}

fn lit(v: f64) -> ExprId {
    ExprId::from_f64(v)
}

/// tanh-approximate GELU (matches `F.gelu(approximate="tanh")`)
fn gelu(x: ExprId) -> ExprId {
    let u = lit((2.0 / std::f64::consts::PI).sqrt()) * (x + lit(0.044715) * x * x * x);
    let th = lit(1.0) - lit(2.0) / ((lit(2.0) * u).exp() + lit(1.0));
    lit(0.5) * x * (lit(1.0) + th)
}

// ---------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------

/// `y[o,t] = Σ_{c,k} w[o,c,k] · x[c, t + d·k − p]`, zero padded ("same").
fn conv1d(g: &mut TensorGraph, x: TensorId, w: TensorId, dil: i64) -> TensorId {
    let (ci, t) = (g.shape(x)[0].n(), g.shape(x)[1].n());
    let (co, k) = (g.shape(w)[0].n(), g.shape(w)[2].n());
    let pad = dil * (k as i64 - 1) / 2;
    // space (o, t, c, k)
    let xv = view(
        vec![form(4, &[(2, 1)], 0), form(4, &[(1, 1), (3, dil)], -pad)],
        true,
    );
    let wv = View::pick(4, &[Some(0), Some(2), Some(3)]);
    g.comp(
        &[co, t],
        &[ci, k],
        vec![arg(x, xv), arg(w, wv)],
        Some(Monoid::sum()),
        |v| vec![v[0] * v[1]],
    )
    .unwrap()
}

/// The same conv as unfold (guarded map) + reshape + matmul: im2col.
fn conv1d_im2col(g: &mut TensorGraph, x: TensorId, w: TensorId, dil: i64) -> TensorId {
    let (ci, t) = (g.shape(x)[0].n(), g.shape(x)[1].n());
    let (co, k) = (g.shape(w)[0].n(), g.shape(w)[2].n());
    let pad = dil * (k as i64 - 1) / 2;
    // space (c, k, t)
    let xv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(2, 1), (1, dil)], -pad)],
        true,
    );
    let unfold = g
        .comp(&[ci, k, t], &[], vec![arg(x, xv)], None, |v| vec![v[0]])
        .unwrap();
    let u = g.reshape(unfold, &[ci * k, t]).unwrap();
    let wr = g.reshape(w, &[co, ci * k]).unwrap();
    g.matmul(wr, u).unwrap()
}

fn conv_graph(
    c: usize,
    t: usize,
    k: usize,
    dil: i64,
    im2col: bool,
) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let w = g.input(&[c, c, k]);
    let y = if im2col {
        conv1d_im2col(&mut g, x, w, dil)
    } else {
        conv1d(&mut g, x, w, dil)
    };
    (g, vec![y])
}

/// conv + bias, then snake `v + sin²(αv)/α` as its epilogue.
fn conv_snake_graph(
    c: usize,
    t: usize,
    k: usize,
    dil: i64,
    im2col: bool,
) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let w = g.input(&[c, c, k]);
    let b = g.input(&[c, 1]);
    let alpha = g.input(&[c, 1]);
    let y = if im2col {
        conv1d_im2col(&mut g, x, w, dil)
    } else {
        conv1d(&mut g, x, w, dil)
    };
    let bb = g.broadcast(b, &[c, t]).unwrap();
    let ab = g.broadcast(alpha, &[c, t]).unwrap();
    let out = g
        .map(&[y, bb, ab], |v| {
            let z = v[0] + v[1];
            let s = (v[2] * z).sin();
            z + s * s / v[2]
        })
        .unwrap();
    (g, vec![out])
}

/// Polyphase upsample: `out[o, q, r] = Σ_{j,c} x[c, q−j] · w[o, c, r + s·j]`,
/// read as `[co, t·s]`. Equals `conv_transpose1d(k = 2s, stride s, p = 0)`
/// cropped to `t·s` samples.
fn upsample_graph(ci: usize, co: usize, t: usize, s: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let w = g.input(&[co, ci, 2 * s]);
    // space (o, q, r, j, c)
    let xv = view(
        vec![form(5, &[(4, 1)], 0), form(5, &[(1, 1), (3, -1)], 0)],
        true,
    );
    let wv = view(
        vec![
            form(5, &[(0, 1)], 0),
            form(5, &[(4, 1)], 0),
            form(5, &[(2, 1), (3, s as i64)], 0),
        ],
        false,
    );
    let y = g
        .comp(
            &[co, t, s],
            &[2, ci],
            vec![arg(x, xv), arg(w, wv)],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let flat = g.reshape(y, &[co, t * s]).unwrap();
    (g, vec![flat])
}

/// Same upsample as unfold + one matmul per phase (`r`), weights
/// pre-arranged on the host as `wp[r, o, (c, j)]`, then a transpose to
/// `[o, q, r]`.
fn upsample_bmm_graph(ci: usize, co: usize, t: usize, s: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[ci, t]);
    let wp = g.input(&[s, co, ci * 2]);
    // unfold u[(c, j), q] = x[c, q − j]
    let uv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(2, 1), (1, -1)], 0)],
        true,
    );
    let u = g
        .comp(&[ci, 2, t], &[], vec![arg(x, uv)], None, |v| vec![v[0]])
        .unwrap();
    let u = g.reshape(u, &[ci * 2, t]).unwrap();
    // bmm: z[r, o, q] = Σ_m wp[r, o, m] · u[m, q]
    let z = g
        .comp(
            &[s, co, t],
            &[ci * 2],
            vec![
                arg(wp, View::pick(4, &[Some(0), Some(1), Some(3)])),
                arg(u, View::pick(4, &[Some(3), Some(2)])),
            ],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let y = g.transpose(z, &[1, 2, 0]).unwrap();
    let flat = g.reshape(y, &[co, t * s]).unwrap();
    (g, vec![flat])
}

/// iSTFT head (Kokoro): `spec[22, T]` = 11 log-magnitudes + 11 phase
/// logits; `mag = exp(a)`, `φ = sin(b)`; inverse real DFT (n_fft 20) as a
/// reduction over bins with a cos basis, Hann-windowed, then overlap-add
/// (hop 5, 4 taps) as a polyphase reduction with guarded frame reads,
/// divided by the window envelope and center-trimmed like `torch.istft`.
fn istft_graph(frames: usize) -> (TensorGraph, Vec<TensorId>) {
    let (n_fft, hop, bins) = (20usize, 5usize, 11usize);
    let taps = n_fft / hop;
    let mut g = TensorGraph::new();
    let spec = g.input(&[2 * bins, frames]);
    let win = g.input(&[n_fft]);
    let out_len = hop * (frames - 1);
    let env = g.input(&[out_len / hop, hop]); // 1 / Σ win² at each output sample
                                              // frame[t, n] = win[n]/N · Σ_k c_k · exp(a_k) · cos(sin(b_k) + 2πkn/N)
    let space = [frames, n_fft, bins];
    let ik = g.iota(&space, 2);
    let inn = g.iota(&space, 1);
    let id3 = View::identity(3, 3);
    let frame = g
        .comp(
            &[frames, n_fft],
            &[bins],
            vec![
                arg(spec, View::pick(3, &[Some(2), Some(0)])),
                arg(
                    spec,
                    view(
                        vec![form(3, &[(2, 1)], bins as i64), form(3, &[(0, 1)], 0)],
                        false,
                    ),
                ),
                arg(win, View::pick(3, &[Some(1)])),
                arg(ik, id3.clone()),
                arg(inn, id3),
            ],
            Some(Monoid::sum()),
            |v| {
                let (a, b, w, k, n) = (v[0], v[1], v[2], v[3], v[4]);
                // c_k = 1 at DC and Nyquist, 2 elsewhere
                let edge = k * (lit((bins - 1) as f64) - k); // 0 at both ends
                let ck = ExprId::select(edge, lit(2.0), lit(1.0));
                let theta = b.sin() + lit(2.0 * std::f64::consts::PI / n_fft as f64) * k * n;
                let cos = (theta + lit(std::f64::consts::FRAC_PI_2)).sin();
                vec![ck * a.exp() * cos * w * lit(1.0 / n_fft as f64)]
            },
        )
        .unwrap();
    // y[q, r] (output sample m = 5q + r + 10 after the center trim):
    //   Σ_j frame[q + 2 − j, r + 5j] · env[q, r]
    let q = out_len / hop;
    let fv = view(
        vec![
            form(3, &[(0, 1), (2, -1)], 2),
            form(3, &[(1, 1), (2, hop as i64)], 0),
        ],
        true,
    );
    let y = g
        .comp(
            &[q, hop],
            &[taps],
            vec![arg(frame, fv), arg(env, View::pick(3, &[Some(0), Some(1)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let flat = g.reshape(y, &[out_len]).unwrap();
    (g, vec![flat])
}

/// AdaIN: instance norm over T, then `(1 + γ)·x̂ + β` with `[γ; β] = W s + b`.
fn adain_graph(c: usize, t: usize, sd: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let s = g.input(&[sd, 1]);
    let w = g.input(&[2 * c, sd]);
    let b = g.input(&[2 * c, 1]);
    let h = g.matmul(w, s).unwrap();
    let h = g.map(&[h, b], |v| v[0] + v[1]).unwrap();
    let mean = g.reduce(x, 1, Monoid::sum()).unwrap();
    let mean = g.map(&[mean], |v| v[0] * lit(1.0 / t as f64)).unwrap();
    let mb = g.broadcast(mean, &[c, t]).unwrap();
    let dev = g.map(&[x, mb], |v| (v[0] - v[1]) * (v[0] - v[1])).unwrap();
    let var = g.reduce(dev, 1, Monoid::sum()).unwrap();
    let inv = g
        .map(&[var], |v| {
            lit(1.0) / (v[0] * lit(1.0 / t as f64) + lit(1e-5)).sqrt()
        })
        .unwrap();
    let ib = g.broadcast(inv, &[c, t]).unwrap();
    // γ = h[o], β = h[c + o], read through views
    let gamma = g
        .comp(
            &[c, t],
            &[],
            vec![arg(h, View::pick(2, &[Some(0), None]))],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let beta = g
        .comp(
            &[c, t],
            &[],
            vec![arg(
                h,
                view(vec![form(2, &[(0, 1)], c as i64), Form::zero(2)], false),
            )],
            None,
            |v| vec![v[0]],
        )
        .unwrap();
    let y = g
        .map(&[x, mb, ib, gamma, beta], |v| {
            (lit(1.0) + v[3]) * (v[0] - v[1]) * v[2] + v[4]
        })
        .unwrap();
    (g, vec![y])
}

struct ConvNext {
    g: TensorGraph,
    y: TensorId,
    params: Vec<TensorId>,
}

/// Vocos ConvNeXt block on `[C, T]`: depthwise conv k7 → LayerNorm over C →
/// pointwise C→F, GELU, F→C → + residual.
fn convnext_graph(c: usize, t: usize, f: usize) -> ConvNext {
    let k = 7;
    let mut g = TensorGraph::new();
    let x = g.input(&[c, t]);
    let dw = g.input(&[c, k]);
    let db = g.input(&[c, 1]);
    let lg = g.input(&[c, 1]);
    let lb = g.input(&[c, 1]);
    let w1 = g.input(&[f, c]);
    let b1 = g.input(&[f, 1]);
    let w2 = g.input(&[c, f]);
    let b2 = g.input(&[c, 1]);
    // space (c, t, k)
    let xv = view(
        vec![form(3, &[(0, 1)], 0), form(3, &[(1, 1), (2, 1)], -3)],
        true,
    );
    let d = g
        .comp(
            &[c, t],
            &[k],
            vec![arg(x, xv), arg(dw, View::pick(3, &[Some(0), Some(2)]))],
            Some(Monoid::sum()),
            |v| vec![v[0] * v[1]],
        )
        .unwrap();
    let dbb = g.broadcast(db, &[c, t]).unwrap();
    let d = g.map(&[d, dbb], |v| v[0] + v[1]).unwrap();
    let mean = g.reduce(d, 0, Monoid::sum()).unwrap();
    let mean = g.map(&[mean], |v| v[0] * lit(1.0 / c as f64)).unwrap();
    let mb = g.broadcast(mean, &[c, t]).unwrap();
    let sq = g.map(&[d, mb], |v| (v[0] - v[1]) * (v[0] - v[1])).unwrap();
    let var = g.reduce(sq, 0, Monoid::sum()).unwrap();
    let inv = g
        .map(&[var], |v| {
            lit(1.0) / (v[0] * lit(1.0 / c as f64) + lit(1e-6)).sqrt()
        })
        .unwrap();
    let ib = g.broadcast(inv, &[c, t]).unwrap();
    let lgb = g.broadcast(lg, &[c, t]).unwrap();
    let lbb = g.broadcast(lb, &[c, t]).unwrap();
    let n = g
        .map(&[d, mb, ib, lgb, lbb], |v| {
            (v[0] - v[1]) * v[2] * v[3] + v[4]
        })
        .unwrap();
    let h = g.matmul(w1, n).unwrap();
    let b1b = g.broadcast(b1, &[f, t]).unwrap();
    let h = g.map(&[h, b1b], |v| gelu(v[0] + v[1])).unwrap();
    let o = g.matmul(w2, h).unwrap();
    let b2b = g.broadcast(b2, &[c, t]).unwrap();
    let y = g.map(&[o, b2b, x], |v| v[0] + v[1] + v[2]).unwrap();
    ConvNext {
        g,
        y,
        params: vec![x, dw, db, lg, lb, w1, b1, w2, b2],
    }
}

// ---------------------------------------------------------------------------
// CPU references (f64)
// ---------------------------------------------------------------------------

fn conv_ref(x: &[f64], w: &[f64], c: usize, t: usize, k: usize, dil: i64) -> Vec<f64> {
    let pad = dil * (k as i64 - 1) / 2;
    let mut y = vec![0.0; c * t];
    for o in 0..c {
        for ci in 0..c {
            for kk in 0..k {
                let wv = w[(o * c + ci) * k + kk];
                let sh = dil * kk as i64 - pad;
                let row = &x[ci * t..(ci + 1) * t];
                let yr = &mut y[o * t..(o + 1) * t];
                for (tt, yv) in yr.iter_mut().enumerate() {
                    let s = tt as i64 + sh;
                    if (0..t as i64).contains(&s) {
                        *yv += wv * row[s as usize];
                    }
                }
            }
        }
    }
    y
}

/// Gradients of `Σ y ⊙ gy` for the "same" conv: `dx[c, τ] = Σ_{o,k}
/// w[o,c,k]·gy[o, τ − d·k + p]`, `dw[o,c,k] = Σ_t gy[o,t]·x[c, t + d·k − p]`.
fn conv_bwd_ref(
    x: &[f64],
    w: &[f64],
    gy: &[f64],
    c: usize,
    t: usize,
    k: usize,
    dil: i64,
) -> (Vec<f64>, Vec<f64>) {
    let pad = dil * (k as i64 - 1) / 2;
    let mut dx = vec![0.0; c * t];
    let mut dw = vec![0.0; c * c * k];
    for o in 0..c {
        let g = &gy[o * t..(o + 1) * t];
        for ci in 0..c {
            let row = &x[ci * t..(ci + 1) * t];
            for kk in 0..k {
                let wv = w[(o * c + ci) * k + kk];
                let sh = dil * kk as i64 - pad;
                let mut acc = 0.0;
                let dr = &mut dx[ci * t..(ci + 1) * t];
                for (tt, &gv) in g.iter().enumerate() {
                    let s = tt as i64 + sh;
                    if (0..t as i64).contains(&s) {
                        acc += gv * row[s as usize];
                        dr[s as usize] += wv * gv;
                    }
                }
                dw[(o * c + ci) * k + kk] = acc;
            }
        }
    }
    (dx, dw)
}

fn upsample_ref(x: &[f64], w: &[f64], ci: usize, co: usize, t: usize, s: usize) -> Vec<f64> {
    // conv_transpose1d(k = 2s, stride s, p = 0), cropped to t·s
    let k = 2 * s;
    let n = t * s;
    let mut y = vec![0.0; co * n];
    for o in 0..co {
        for c in 0..ci {
            for tt in 0..t {
                let xv = x[c * t + tt];
                for kk in 0..k {
                    let m = tt * s + kk;
                    if m < n {
                        y[o * n + m] += xv * w[(o * ci + c) * k + kk];
                    }
                }
            }
        }
    }
    y
}

fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// 1/Σ_t win²[m − hop·t] at each kept sample (after the n_fft/2 trim).
fn istft_env(frames: usize) -> Vec<f64> {
    let (n_fft, hop) = (20, 5);
    let w = hann(n_fft);
    let full = hop * (frames - 1) + n_fft;
    let mut env = vec![0.0; full];
    for t in 0..frames {
        for n in 0..n_fft {
            env[t * hop + n] += w[n] * w[n];
        }
    }
    let out_len = hop * (frames - 1);
    (0..out_len).map(|m| 1.0 / env[m + n_fft / 2]).collect()
}

fn istft_ref(spec: &[f64], frames: usize) -> Vec<f64> {
    let (n_fft, hop, bins) = (20usize, 5usize, 11usize);
    let w = hann(n_fft);
    let full = hop * (frames - 1) + n_fft;
    let mut y = vec![0.0; full];
    let mut env = vec![0.0; full];
    for t in 0..frames {
        for n in 0..n_fft {
            // irfft of X_k = mag·e^{iφ}
            let mut v = 0.0;
            for k in 0..bins {
                let mag = spec[k * frames + t].exp();
                let ph = spec[(bins + k) * frames + t].sin();
                let ck = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
                let th = 2.0 * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
                v += ck * mag * (ph + th).cos();
            }
            y[t * hop + n] += v / n_fft as f64 * w[n];
            env[t * hop + n] += w[n] * w[n];
        }
    }
    let out_len = hop * (frames - 1);
    (0..out_len)
        .map(|m| y[m + n_fft / 2] / env[m + n_fft / 2])
        .collect()
}

fn adain_ref(
    x: &[f64],
    s: &[f64],
    w: &[f64],
    b: &[f64],
    c: usize,
    t: usize,
    sd: usize,
) -> Vec<f64> {
    let h: Vec<f64> = (0..2 * c)
        .map(|o| b[o] + (0..sd).map(|i| w[o * sd + i] * s[i]).sum::<f64>())
        .collect();
    let mut y = vec![0.0; c * t];
    for ch in 0..c {
        let row = &x[ch * t..(ch + 1) * t];
        let mean = row.iter().sum::<f64>() / t as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for tt in 0..t {
            y[ch * t + tt] = (1.0 + h[ch]) * (row[tt] - mean) * inv + h[c + ch];
        }
    }
    y
}

fn gelu_ref(x: f64) -> f64 {
    let u = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

fn convnext_ref(p: &[Vec<f64>], c: usize, t: usize, f: usize) -> Vec<f64> {
    let (x, dw, db, lg, lb, w1, b1, w2, b2) = (
        &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7], &p[8],
    );
    let mut d = vec![0.0; c * t];
    for ch in 0..c {
        for tt in 0..t {
            let mut v = db[ch];
            for k in 0..7 {
                let s = tt as i64 + k as i64 - 3;
                if (0..t as i64).contains(&s) {
                    v += x[ch * t + s as usize] * dw[ch * 7 + k];
                }
            }
            d[ch * t + tt] = v;
        }
    }
    let mut n = vec![0.0; c * t];
    for tt in 0..t {
        let mean = (0..c).map(|ch| d[ch * t + tt]).sum::<f64>() / c as f64;
        let var = (0..c)
            .map(|ch| (d[ch * t + tt] - mean).powi(2))
            .sum::<f64>()
            / c as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for ch in 0..c {
            n[ch * t + tt] = (d[ch * t + tt] - mean) * inv * lg[ch] + lb[ch];
        }
    }
    let mut h = vec![0.0; f * t];
    for o in 0..f {
        for ch in 0..c {
            let wv = w1[o * c + ch];
            for tt in 0..t {
                h[o * t + tt] += wv * n[ch * t + tt];
            }
        }
        for tt in 0..t {
            h[o * t + tt] = gelu_ref(h[o * t + tt] + b1[o]);
        }
    }
    let mut y = vec![0.0; c * t];
    for o in 0..c {
        for i in 0..f {
            let wv = w2[o * f + i];
            for tt in 0..t {
                y[o * t + tt] += wv * h[i * t + tt];
            }
        }
        for tt in 0..t {
            y[o * t + tt] += b2[o] + x[o * t + tt];
        }
    }
    y
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/// The graph passes before scheduling. `merge_reduce_dims` runs before
/// `schedule_auto`'s multi-dim tiler: mergeable reduce dims become one,
/// the tiler handles what stays split (a conv's `(c, k)`).
/// `TANG_NO_MERGE=1` skips merge_siblings / relayout / merge_reduce_dims.
fn optimized(g: &TensorGraph, outs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = g.substitute_maps(outs);
    let (g, outs) = g.eliminate_common(&outs);
    if std::env::var_os("TANG_NO_MERGE").is_some() {
        return (g, outs);
    }
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    g.merge_reduce_dims(&outs)
}

struct Row {
    name: &'static str,
    best: f64,
    schedule: String,
}

/// Time `naive`, `auto` and per-kernel `autotune` for every algebra in
/// `algebras`, checking each against `want` (output 0).
fn bench(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &'static str,
    algebras: Vec<(&'static str, TensorGraph, Vec<TensorId>)>,
    inputs: &[&[f32]],
    want: &[f64],
    tol: f64,
) -> Row {
    let mut best = (f64::INFINITY, String::new());
    let mut line = format!("{name}:");
    for (alg, g, outs) in algebras {
        let (g, outs) = optimized(&g, &outs);
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let tuned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            autotune_kernels(device, cache, auto.clone(), &outs, inputs, 10).0
        }));
        let mut variants = vec![("naive", g), ("auto", auto)];
        match tuned {
            Ok(t) => variants.push(("autotune", t)),
            Err(_) => line.push_str(&format!("  [{alg}/autotune FAILED]")),
        }
        for (sched, g) in variants {
            let exe = Executable::new(device, g.to_wgsl(&outs), inputs);
            exe.run(device, cache);
            let got = exe.read(device, outs[0]);
            let err = rel_err(&got, want);
            let ms = time_exe(device, cache, &exe, outs[0]);
            let ok = if err <= tol { "" } else { " WRONG" };
            line.push_str(&format!(
                "\n    {alg:>8}/{sched:<8} {ms:8.3} ms  {} kernels  err {err:.1e}{ok}",
                exe.program.kernels.len()
            ));
            if err <= tol && ms < best.0 {
                best = (ms, format!("{alg}/{sched}"));
            }
        }
    }
    println!("{line}");
    Row {
        name,
        best: best.0,
        schedule: best.1,
    }
}

/// Forward + backward of `sum(y)` w.r.t. `wrt`, all schedules; checked on
/// a small instance against the f64 interpreter (`small`).
fn bench_train(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &'static str,
    algebras: Vec<(&'static str, TensorGraph, TensorId, Vec<TensorId>)>,
    inputs: &[&[f32]],
) -> Row {
    let mut best = (f64::INFINITY, String::new());
    let mut line = format!("{name}:");
    for (alg, mut g, y, wrt) in algebras {
        let grads = g.backward(y, &wrt).unwrap();
        let mut outs = vec![y];
        outs.extend(grads);
        let (g, outs) = optimized(&g, &outs);
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let mut variants = vec![("naive", g), ("auto", auto.clone())];
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            autotune_kernels(device, cache, auto.clone(), &outs, inputs, 10).0
        })) {
            Ok(t) => variants.push(("autotune", t)),
            Err(_) => line.push_str(&format!("  [{alg}/autotune FAILED]")),
        }
        for (sched, g) in variants {
            let exe = Executable::new(device, g.to_wgsl(&outs), inputs);
            let ms = time_exe(device, cache, &exe, outs[0]);
            line.push_str(&format!(
                "\n    {alg:>8}/{sched:<8} {ms:8.3} ms  {} kernels",
                exe.program.kernels.len()
            ));
            if ms < best.0 {
                best = (ms, format!("{alg}/{sched}"));
            }
        }
    }
    println!("{line}");
    Row {
        name,
        best: best.0,
        schedule: best.1,
    }
}

/// GPU forward+backward on a small instance vs the f64 interpreter.
fn check_train(
    device: &GpuDevice,
    cache: &mut KernelCache,
    name: &str,
    (g, y, wrt): (TensorGraph, TensorId, Vec<TensorId>),
    inputs: &[Vec<f32>],
) {
    let mut g = g;
    let grads = g.backward(y, &wrt).unwrap();
    let mut outs = vec![y];
    outs.extend(grads);
    let want = g.eval(&inputs.iter().map(|v| f64s(v)).collect::<Vec<_>>(), &outs);
    let (mut og, oouts) = optimized(&g, &outs);
    og.schedule_auto(&oouts);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let exe = Executable::new(device, og.to_wgsl(&oouts), &refs);
    exe.run(device, cache);
    let worst = oouts
        .iter()
        .zip(&want)
        .map(|(o, w)| rel_err(&exe.read(device, *o), w))
        .fold(0.0, f64::max);
    println!("  {name} fwd+bwd check (small, vs f64 interpreter): max err {worst:.1e}");
    assert!(worst < 1e-4, "{name} backward wrong");
}

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let only: Option<String> = std::env::args().nth(1);
    if only.as_deref() == Some("precision") {
        precision_main(&device, &mut cache, std::env::args().nth(2));
        return;
    }
    if only.as_deref() == Some("msl") {
        #[cfg(feature = "metal")]
        msl::main(std::env::args().nth(2));
        #[cfg(not(feature = "metal"))]
        eprintln!("build with --features metal");
        return;
    }
    let want_op = |n: &str| only.as_deref().is_none_or(|o| n.starts_with(o));
    println!(
        "{} — kernel-only, {ITERS} launches per sample\n",
        device.name
    );
    let mut rows: Vec<Row> = Vec::new();
    let tol = 1e-5;

    // (a) conv1d 512→512, k7, dilation 3, same padding, T=2000
    let (c, t, k, dil) = (512, 2000, 7, 3);
    let x = rand(c * t, 1, 1.0);
    let w = rand(c * c * k, 2, 0.05);
    if want_op("a") {
        let want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        let (gd, od) = conv_graph(c, t, k, dil, false);
        let (gi, oi) = conv_graph(c, t, k, dil, true);
        rows.push(bench(
            &device,
            &mut cache,
            "a conv1d 512→512 k7 d3 T2000",
            vec![("direct", gd, od), ("im2col", gi, oi)],
            &[&x, &w],
            &want,
            tol,
        ));
    }

    // (b) polyphase upsample 512→256, k20, s10, T=2000 → [256, 20000]
    let (ci, co, s) = (512, 256, 10);
    if want_op("b") {
        let xu = rand(ci * t, 3, 1.0);
        let wu = rand(co * ci * 2 * s, 4, 0.05);
        let want = upsample_ref(&f64s(&xu), &f64s(&wu), ci, co, t, s);
        // host-side weight arrangement for the bmm algebra: wp[r, o, (c, j)]
        let mut wp = vec![0.0f32; s * co * ci * 2];
        for r in 0..s {
            for o in 0..co {
                for cc in 0..ci {
                    for j in 0..2 {
                        wp[((r * co + o) * ci + cc) * 2 + j] =
                            wu[(o * ci + cc) * 2 * s + r + s * j];
                    }
                }
            }
        }
        let (gd, od) = upsample_graph(ci, co, t, s);
        let line = bench(
            &device,
            &mut cache,
            "b polyphase upsample 512→256 k20 s10",
            vec![("direct", gd, od)],
            &[&xu, &wu],
            &want,
            tol,
        );
        let (gb, ob) = upsample_bmm_graph(ci, co, t, s);
        let bmm = bench(
            &device,
            &mut cache,
            "b polyphase upsample (bmm algebra)",
            vec![("bmm", gb, ob)],
            &[&xu, &wp],
            &want,
            tol,
        );
        rows.push(if bmm.best < line.best {
            Row {
                name: line.name,
                ..bmm
            }
        } else {
            line
        });
    }

    // (c) iSTFT head: [22, T] → audio, n_fft 20, hop 5
    if want_op("c") {
        let frames = 4000;
        let spec = rand(22 * frames, 5, 1.0);
        let win: Vec<f32> = hann(20).iter().map(|&v| v as f32).collect();
        let env: Vec<f32> = istft_env(frames).iter().map(|&v| v as f32).collect();
        let want = istft_ref(&f64s(&spec), frames);
        let (g, o) = istft_graph(frames);
        rows.push(bench(
            &device,
            &mut cache,
            "c iSTFT n_fft20 hop5 T4000",
            vec![("dft+ola", g, o)],
            &[&spec, &win, &env],
            &want,
            tol,
        ));
    }

    // (d) conv (a) + bias + snake epilogue
    if want_op("d") {
        let b = rand(c, 6, 0.1);
        let alpha: Vec<f32> = rand(c, 7, 0.5).iter().map(|v| v + 1.0).collect();
        let mut want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        for ch in 0..c {
            for tt in 0..t {
                let z = want[ch * t + tt] + b[ch] as f64;
                let a = alpha[ch] as f64;
                want[ch * t + tt] = z + (a * z).sin().powi(2) / a;
            }
        }
        let (gd, od) = conv_snake_graph(c, t, k, dil, false);
        let (gi, oi) = conv_snake_graph(c, t, k, dil, true);
        // one kernel: the conv reduction recomputed under the snake map
        let (gf, of) = optimized(&gd, &od);
        let mut fused = gf.clone();
        let conv_node = fused.nodes().iter().position(
            |n| matches!(&n.op, tang_compile::tensor::Op::Comp(cc) if cc.monoid.is_some()),
        );
        fused.set_inline(TensorId::from_index(conv_node.unwrap()), true);
        let exe = Executable::new(&device, fused.to_wgsl(&of), &[&x, &w, &b, &alpha]);
        exe.run(&device, &mut cache);
        let err = rel_err(&exe.read(&device, of[0]), &want);
        let ms = time_exe(&device, &mut cache, &exe, of[0]);
        println!(
            "d fused into one kernel (conv inlined under snake): {ms:.3} ms  {} kernels  err {err:.1e}",
            exe.program.kernels.len()
        );
        let mut row = bench(
            &device,
            &mut cache,
            "d conv + bias + snake",
            vec![("direct", gd, od), ("im2col", gi, oi)],
            &[&x, &w, &b, &alpha],
            &want,
            tol,
        );
        if err <= tol && ms < row.best {
            row.best = ms;
            row.schedule = "direct/fused-1-kernel".into();
        }
        rows.push(row);
    }

    // (e) AdaIN, C=512, T=2000, style 128
    if want_op("e") {
        let sd = 128;
        let xa = rand(c * t, 8, 1.0);
        let st = rand(sd, 9, 1.0);
        let wa = rand(2 * c * sd, 10, 0.1);
        let ba = rand(2 * c, 11, 0.1);
        let want = adain_ref(&f64s(&xa), &f64s(&st), &f64s(&wa), &f64s(&ba), c, t, sd);
        let (g, o) = adain_graph(c, t, sd);
        rows.push(bench(
            &device,
            &mut cache,
            "e AdaIN C512 T2000 style128",
            vec![("adain", g, o)],
            &[&xa, &st, &wa, &ba],
            &want,
            tol,
        ));
    }

    // (f) ConvNeXt block, C=512, T=200, F=1536
    let (cf, tf, ff) = (512, 200, 1536);
    let cn_inputs: Vec<Vec<f32>> = vec![
        rand(cf * tf, 12, 1.0),
        rand(cf * 7, 13, 0.3),
        rand(cf, 14, 0.1),
        rand(cf, 15, 0.2).iter().map(|v| v + 1.0).collect(),
        rand(cf, 16, 0.1),
        rand(ff * cf, 17, 0.04),
        rand(ff, 18, 0.1),
        rand(cf * ff, 19, 0.02),
        rand(cf, 20, 0.1),
    ];
    let cn_refs: Vec<&[f32]> = cn_inputs.iter().map(|v| v.as_slice()).collect();
    if want_op("f") {
        let want = convnext_ref(
            &cn_inputs.iter().map(|v| f64s(v)).collect::<Vec<_>>(),
            cf,
            tf,
            ff,
        );
        let cn = convnext_graph(cf, tf, ff);
        rows.push(bench(
            &device,
            &mut cache,
            "f ConvNeXt block C512 T200 F1536",
            vec![("convnext", cn.g, vec![cn.y])],
            &cn_refs,
            &want,
            tol,
        ));
    }

    // forward + backward for (a) and (f)
    if want_op("a") || want_op("train") {
        let small = conv_graph(6, 20, 7, 3, false);
        let (sg, so) = small;
        let (ci_s, t_s) = (6, 20);
        let inputs = vec![rand(ci_s * t_s, 21, 1.0), rand(ci_s * ci_s * 7, 22, 0.3)];
        check_train(
            &device,
            &mut cache,
            "a",
            (
                sg,
                so[0],
                vec![TensorId::from_index(0), TensorId::from_index(1)],
            ),
            &inputs,
        );
        let (si, soi) = conv_graph(ci_s, t_s, 7, 3, true);
        check_train(
            &device,
            &mut cache,
            "a im2col",
            (
                si,
                soi[0],
                vec![TensorId::from_index(0), TensorId::from_index(1)],
            ),
            &inputs,
        );
        let xw = vec![TensorId::from_index(0), TensorId::from_index(1)];
        let (g, o) = conv_graph(c, t, k, dil, false);
        let (gi, oi) = conv_graph(c, t, k, dil, true);
        rows.push(bench_train(
            &device,
            &mut cache,
            "a fwd+bwd (dx, dw)",
            vec![("direct", g, o[0], xw.clone()), ("im2col", gi, oi[0], xw)],
            &[&x, &w],
        ));
    }
    if want_op("f") || want_op("train") {
        let (cs, ts, fs) = (8, 12, 16);
        let small = convnext_graph(cs, ts, fs);
        let sizes = [cs * ts, cs * 7, cs, cs, cs, fs * cs, fs, cs * fs, cs];
        let inputs: Vec<Vec<f32>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| rand(n, 30 + i as u64, 0.5))
            .collect();
        check_train(
            &device,
            &mut cache,
            "f",
            (small.g, small.y, small.params),
            &inputs,
        );
        let cn = convnext_graph(cf, tf, ff);
        rows.push(bench_train(
            &device,
            &mut cache,
            "f fwd+bwd (all params)",
            vec![("convnext", cn.g, cn.y, cn.params)],
            &cn_refs,
        ));
    }

    println!("\n| op | IR best (ms) | schedule |");
    println!("|---|---:|---|");
    for r in rows {
        println!("| {} | {:.3} | {} |", r.name, r.best, r.schedule);
    }
}

// ---------------------------------------------------------------------------
// precision schedules
// ---------------------------------------------------------------------------

/// One op for the precision search: the full-size graph with inputs and
/// its f64 reference, input names for the report, and a small instance
/// of the same builder for the interpreter's prediction.
struct PrecOp {
    name: &'static str,
    g: TensorGraph,
    outs: Vec<TensorId>,
    inputs: Vec<Vec<f32>>,
    want: Vec<f64>,
    names: Vec<&'static str>,
    small: (TensorGraph, Vec<TensorId>, Vec<Vec<f64>>),
}

fn matmul_graph(m: usize, k: usize, n: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let a = g.input(&[m, k]);
    let b = g.input(&[k, n]);
    let y = g.matmul(a, b).unwrap();
    (g, vec![y])
}

fn matmul_ref(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk];
            let row = &b[kk * n..(kk + 1) * n];
            for (yv, bv) in y[i * n..(i + 1) * n].iter_mut().zip(row) {
                *yv += av * bv;
            }
        }
    }
    y
}

fn small_inputs(g: &TensorGraph, seed: u64, scales: &[f32]) -> Vec<Vec<f64>> {
    let n_in = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input(_)))
        .count();
    (0..n_in)
        .map(|k| {
            let node = g
                .nodes()
                .iter()
                .find(|n| matches!(n.op, Op::Input(i) if i == k))
                .unwrap();
            let len = node.shape.iter().map(|d| d.n()).product();
            f64s(&rand(len, seed + k as u64, scales[k]))
        })
        .collect()
}

/// `name[shape]` of a node, for the assignment column.
fn describe(g: &TensorGraph, t: TensorId, names: &[&str], outs: &[TensorId]) -> String {
    let n = &g.nodes()[t.index()];
    let shape = n
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("×");
    match n.op {
        Op::Input(k) => format!("{}[{shape}]", names[k]),
        _ if outs.contains(&t) => format!("out[{shape}]"),
        _ => format!("n{}[{shape}]", t.index()),
    }
}

fn fmt_prec(p: Precision) -> String {
    let f = |f: Format| match f {
        Format::F32 => "f32",
        Format::F16 => "f16",
        Format::BF16 => "bf16",
        Format::E4M3(_) => "e4m3",
        Format::E5M2(_) => "e5m2",
        Format::Fp8Accum => "fp8acc",
    };
    if p.accum == Format::F32 {
        f(p.storage).to_string()
    } else {
        format!("{} acc {}", f(p.storage), f(p.accum))
    }
}

/// The small instance with the same precision assignment, error predicted
/// by the interpreter (`None` if the passes produced a different node
/// layout at the small size, so ids don't line up).
fn predict(
    op: &PrecOp,
    full: &TensorGraph,
    assignment: &[(TensorId, Precision)],
) -> Option<ErrorStats> {
    let (sg, souts, sin) = &op.small;
    let (mut sg, souts) = optimized(sg, souts);
    if sg.nodes().len() != full.nodes().len() {
        return None;
    }
    for &(t, p) in assignment {
        sg.set_inline(t, full.nodes()[t.index()].inline);
        sg.set_precision(t, p);
    }
    Some(sg.precision_error(sin, &souts))
}

/// The ops of the precision tables (`a`–`f` without `d`, plus `m`), full
/// size with their f64 references, and `d` when `with_d`.
fn prec_ops(only: Option<&str>, with_d: bool) -> Vec<PrecOp> {
    let want_op = |n: &str| only.is_none_or(|o| n.starts_with(o));
    let mut ops: Vec<PrecOp> = Vec::new();

    if want_op("a") {
        let (c, t, k, dil) = (512, 2000, 7, 3);
        let x = rand(c * t, 1, 1.0);
        let w = rand(c * c * k, 2, 0.05);
        let want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        let (g, outs) = conv_graph(c, t, k, dil, false);
        let (sg, so) = conv_graph(32, 256, k, dil, false);
        let si = small_inputs(&sg, 100, &[1.0, 0.05]);
        ops.push(PrecOp {
            name: "a conv1d 512→512 k7 d3 T2000",
            g,
            outs,
            inputs: vec![x, w],
            want,
            names: vec!["x", "w"],
            small: (sg, so, si),
        });
    }
    if want_op("b") {
        let (ci, co, t, s) = (512, 256, 2000, 10);
        let xu = rand(ci * t, 3, 1.0);
        let wu = rand(co * ci * 2 * s, 4, 0.05);
        let want = upsample_ref(&f64s(&xu), &f64s(&wu), ci, co, t, s);
        let (g, outs) = upsample_graph(ci, co, t, s);
        let (sg, so) = upsample_graph(32, 16, 64, s);
        let si = small_inputs(&sg, 110, &[1.0, 0.05]);
        ops.push(PrecOp {
            name: "b upsample 512→256 k20 s10",
            g,
            outs,
            inputs: vec![xu, wu],
            want,
            names: vec!["x", "w"],
            small: (sg, so, si),
        });
    }
    if want_op("c") {
        let frames = 4000;
        let spec = rand(22 * frames, 5, 1.0);
        let win: Vec<f32> = hann(20).iter().map(|&v| v as f32).collect();
        let env: Vec<f32> = istft_env(frames).iter().map(|&v| v as f32).collect();
        let want = istft_ref(&f64s(&spec), frames);
        let (g, outs) = istft_graph(frames);
        let sf = 200;
        let (sg, so) = istft_graph(sf);
        let si = vec![f64s(&rand(22 * sf, 120, 1.0)), hann(20), istft_env(sf)];
        ops.push(PrecOp {
            name: "c iSTFT n_fft20 hop5 T4000",
            g,
            outs,
            inputs: vec![spec, win, env],
            want,
            names: vec!["spec", "win", "env"],
            small: (sg, so, si),
        });
    }
    if want_op("e") {
        let (c, t, sd) = (512, 2000, 128);
        let xa = rand(c * t, 8, 1.0);
        let st = rand(sd, 9, 1.0);
        let wa = rand(2 * c * sd, 10, 0.1);
        let ba = rand(2 * c, 11, 0.1);
        let want = adain_ref(&f64s(&xa), &f64s(&st), &f64s(&wa), &f64s(&ba), c, t, sd);
        let (g, outs) = adain_graph(c, t, sd);
        // same T: the per-channel statistics are the long reductions
        let (sg, so) = adain_graph(8, t, 16);
        let si = small_inputs(&sg, 130, &[1.0, 1.0, 0.1, 0.1]);
        ops.push(PrecOp {
            name: "e AdaIN C512 T2000 style128",
            g,
            outs,
            inputs: vec![xa, st, wa, ba],
            want,
            names: vec!["x", "style", "W", "b"],
            small: (sg, so, si),
        });
    }
    if want_op("f") {
        let (cf, tf, ff) = (512, 200, 1536);
        let inputs: Vec<Vec<f32>> = vec![
            rand(cf * tf, 12, 1.0),
            rand(cf * 7, 13, 0.3),
            rand(cf, 14, 0.1),
            rand(cf, 15, 0.2).iter().map(|v| v + 1.0).collect(),
            rand(cf, 16, 0.1),
            rand(ff * cf, 17, 0.04),
            rand(ff, 18, 0.1),
            rand(cf * ff, 19, 0.02),
            rand(cf, 20, 0.1),
        ];
        let want = convnext_ref(
            &inputs.iter().map(|v| f64s(v)).collect::<Vec<_>>(),
            cf,
            tf,
            ff,
        );
        let cn = convnext_graph(cf, tf, ff);
        let sm = convnext_graph(32, 24, 96);
        let si = small_inputs(&sm.g, 140, &[1.0, 0.3, 0.1, 0.2, 0.1, 0.04, 0.1, 0.02, 0.1]);
        ops.push(PrecOp {
            name: "f ConvNeXt C512 T200 F1536",
            g: cn.g,
            outs: vec![cn.y],
            inputs,
            want,
            names: vec!["x", "dw", "db", "ln_g", "ln_b", "w1", "b1", "w2", "b2"],
            small: (sm.g, vec![sm.y], si),
        });
    }
    if want_op("m") {
        let n = 1024;
        let a = rand(n * n, 40, 1.0);
        let b = rand(n * n, 41, 1.0);
        let want = matmul_ref(&f64s(&a), &f64s(&b), n, n, n);
        let (g, outs) = matmul_graph(n, n, n);
        let (sg, so) = matmul_graph(64, n, 64);
        let si = small_inputs(&sg, 150, &[1.0, 1.0]);
        ops.push(PrecOp {
            name: "m matmul 1024³",
            g,
            outs,
            inputs: vec![a, b],
            want,
            names: vec!["a", "b"],
            small: (sg, so, si),
        });
    }

    if with_d && want_op("d") {
        let (c, t, k, dil) = (512, 2000, 7, 3);
        let x = rand(c * t, 1, 1.0);
        let w = rand(c * c * k, 2, 0.05);
        let b = rand(c, 6, 0.1);
        let alpha: Vec<f32> = rand(c, 7, 0.5).iter().map(|v| v + 1.0).collect();
        let mut want = conv_ref(&f64s(&x), &f64s(&w), c, t, k, dil);
        for ch in 0..c {
            for tt in 0..t {
                let z = want[ch * t + tt] + b[ch] as f64;
                let a = alpha[ch] as f64;
                want[ch * t + tt] = z + (a * z).sin().powi(2) / a;
            }
        }
        let (g, outs) = conv_snake_graph(c, t, k, dil, false);
        let (sg, so) = conv_snake_graph(32, 256, k, dil, false);
        let si = small_inputs(&sg, 160, &[1.0, 0.05, 0.1, 1.0]);
        ops.push(PrecOp {
            name: "d conv + bias + snake",
            g,
            outs,
            inputs: vec![x, w, b, alpha],
            want,
            names: vec!["x", "w", "b", "alpha"],
            small: (sg, so, si),
        });
    }
    ops
}

fn precision_main(device: &GpuDevice, cache: &mut KernelCache, only: Option<String>) {
    println!(
        "{} — precision schedules, kernel-only, {ITERS} launches\n",
        device.name
    );
    let ops = prec_ops(only.as_deref(), false);
    if std::env::var("PREC_DEBUG").is_ok() {
        for op in &ops {
            let (sg, souts, sin) = &op.small;
            let (mut g, outs) = optimized(sg, souts);
            g.schedule_auto(&outs);
            for (i, n) in g.nodes().iter().enumerate() {
                if let Op::Comp(c) = &n.op {
                    if c.monoid.is_some() {
                        let mut v = g.clone();
                        v.set_precision(
                            TensorId::from_index(i),
                            Precision {
                                storage: Format::F32,
                                accum: Format::F16,
                                operand: Format::F32,
                            },
                        );
                        let e = v.precision_error(sin, &outs);
                        println!(
                            "{} n{i} inline={} red={:?} out={:?} acc f16 → {:.1e}",
                            op.name, n.inline, c.red, n.shape, e.max_rel
                        );
                    }
                }
            }
        }
        return;
    }
    let mut table = Vec::new();
    let mut rejects = Vec::new();
    for op in &ops {
        let refs: Vec<&[f32]> = op.inputs.iter().map(|v| v.as_slice()).collect();
        let (g, outs) = optimized(&op.g, &op.outs);
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let tuned = autotune_kernels(device, cache, auto, &outs, &refs, 10).0;
        let want = vec![op.want.clone()];
        for rel in [1e-2, 1e-3] {
            let budget = ErrorBudget::rel(rel);
            let r = tune_precision(
                device,
                cache,
                tuned.clone(),
                &outs,
                &refs,
                &want,
                budget,
                ITERS,
            );
            let assign = if r.assignment.is_empty() {
                "all f32".to_string()
            } else {
                r.assignment
                    .iter()
                    .map(|&(t, p)| {
                        format!(
                            "{} {}",
                            describe(&r.graph, t, &op.names, &outs),
                            fmt_prec(p)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let pred = predict(op, &r.graph, &r.assignment)
                .map_or("n/a".into(), |e| format!("{:.1e}", e.max_rel));
            println!(
                "{} @ rel {rel:.0e}: f32 {:.3} ms (err {:.1e}) → {:.3} ms (err {:.1e}, predicted {pred})  [{assign}]",
                op.name, r.base_ms, r.base_error.max_rel, r.ms, r.error.max_rel
            );
            for t in &r.trials {
                let verdict = match (t.kept, t.ms, budget.admits(&t.error)) {
                    (true, _, _) => "kept".to_string(),
                    (false, Some(ms), _) => format!("no gain ({ms:.3} ms)"),
                    (false, None, true) => "within budget (accum: speed not modeled)".into(),
                    (false, None, false) => "REJECTED: over budget".into(),
                };
                println!(
                    "    {:<28} {:<12} err {:.1e}  {verdict}",
                    describe(&r.graph, t.node, &op.names, &outs),
                    fmt_prec(t.precision),
                    t.error.max_rel
                );
                if rel == 1e-3 && t.precision.accum == Format::F16 && !budget.admits(&t.error) {
                    rejects.push(format!(
                        "| {} | {} {} | {:.1e} |",
                        op.name,
                        describe(&r.graph, t.node, &op.names, &outs),
                        fmt_prec(t.precision),
                        t.error.max_rel
                    ));
                }
            }
            table.push(format!(
                "| {} | {rel:.0e} | {:.3} | {:.3} | {:.2}× | {assign} | {:.1e} | {pred} |",
                op.name,
                r.base_ms,
                r.ms,
                r.base_ms / r.ms,
                r.error.max_rel
            ));
        }
    }
    println!("\n| op | budget (rel) | f32 ms | chosen ms | speedup | chosen assignment | measured err vs f64 | interp-predicted err (small) |");
    println!("|---|---|---:|---:|---:|---|---:|---:|");
    for row in table {
        println!("{row}");
    }
    println!("\nf16 accumulators the 1e-3 budget rejects:\n\n| op | node | measured err |\n|---|---|---:|");
    for row in rejects {
        println!("{row}");
    }
}

// ---------------------------------------------------------------------------
// native Metal backend
// ---------------------------------------------------------------------------

/// `kokoro_ops msl [op]`: the same ops on the MSL lowering — f32 (naive,
/// auto, per-kernel autotune), f16 operands on simdgroup fragments
/// (forced, to show what they cost and buy), and the precision search
/// with the fragment dimension under 1e-2 / 1e-3 budgets. Fwd+bwd rows
/// for (a) and (f) in f32.
#[cfg(feature = "metal")]
mod msl {
    use super::*;
    use tang_bench::ir_msl::{self, Executable, MslDevice};

    fn time(dev: &MslDevice, exe: &Executable) -> f64 {
        (0..3)
            .map(|_| exe.time(dev, ITERS))
            .fold(f64::INFINITY, f64::min)
    }

    fn run(
        dev: &MslDevice,
        g: &TensorGraph,
        outs: &[TensorId],
        inputs: &[&[f32]],
        want: &[f64],
    ) -> (f64, f64, usize) {
        let prog = g.to_msl(outs);
        if !Executable::fits(dev, &prog) {
            // staging over the 32 KiB threadgroup limit: not runnable here
            return (f64::INFINITY, f64::INFINITY, prog.kernels.len());
        }
        let exe = Executable::new(dev, prog, inputs);
        exe.run(dev);
        let err = rel_err(&exe.read(outs[0]), want);
        (time(dev, &exe), err, exe.program.kernels.len())
    }

    pub fn main(only: Option<String>) {
        let dev = MslDevice::new().expect("metal");
        println!(
            "{} — MSL backend, kernel-only, {ITERS} launches per sample\n",
            dev.name
        );
        let ops = prec_ops(only.as_deref(), true);
        let mut table = Vec::new();
        for op in &ops {
            let refs: Vec<&[f32]> = op.inputs.iter().map(|v| v.as_slice()).collect();
            let (g, outs) = optimized(&op.g, &op.outs);
            let mut auto = g.clone();
            auto.schedule_auto(&outs);
            let tuned = ir_msl::autotune_kernels(&dev, auto.clone(), &outs, &refs, 10).0;
            let mut line = format!("{}:", op.name);
            let mut best = (f64::INFINITY, "", 0.0);
            for (sched, gv) in [("naive", &g), ("auto", &auto), ("autotune", &tuned)] {
                let (ms, err, nk) = run(&dev, gv, &outs, &refs, &op.want);
                line.push_str(&format!(
                    "\n    f32 {sched:<8} {ms:8.3} ms  {nk} kernels  err {err:.1e}"
                ));
                if err <= 1e-5 && ms < best.0 {
                    best = (ms, sched, err);
                }
            }
            // forced: every fragment-eligible contraction on f16 fragments
            let cands = ir_msl::fragment_candidates(&tuned, &outs);
            let forced = if cands.is_empty() {
                line.push_str("\n    f16 fragments: no eligible contraction");
                "n/a".to_string()
            } else {
                let mut gf = tuned.clone();
                for &t in &cands {
                    let srcs: Vec<TensorId> = match &gf.nodes()[t.index()].op {
                        Op::Comp(c) => {
                            let core = c
                                .args
                                .iter()
                                .find(|a| gf.nodes()[a.src.index()].inline && matches!(&gf.nodes()[a.src.index()].op, Op::Comp(r) if r.monoid.is_some()))
                                .map_or(t, |a| a.src);
                            match &gf.nodes()[core.index()].op {
                                Op::Comp(cc) => cc.args.iter().map(|a| gf.storage(a.src)).collect(),
                                _ => vec![],
                            }
                        }
                        _ => vec![],
                    };
                    for s in srcs {
                        gf.set_precision(s, Precision::storage(Format::F16));
                    }
                    gf.set_fragment(t, true);
                }
                let (ms, err, nk) = run(&dev, &gf, &outs, &refs, &op.want);
                line.push_str(&format!(
                    "\n    f16 fragments ({} nodes) {ms:8.3} ms  {nk} kernels  err {err:.1e}",
                    cands.len()
                ));
                format!("{ms:.3} ({err:.0e})")
            };
            {
                let gv = match best.1 {
                    "naive" => &g,
                    "auto" => &auto,
                    _ => &tuned,
                };
                let exe = Executable::new(&dev, gv.to_msl(&outs), &refs);
                let sep = (0..3)
                    .map(|_| exe.time_separate(&dev, ITERS))
                    .fold(f64::INFINITY, f64::min);
                line.push_str(&format!(
                    "\n    f32 {} with one command buffer per run: {sep:.3} ms",
                    best.1
                ));
            }
            let mut tuned_cols = Vec::new();
            for rel in [1e-2, 1e-3] {
                let r = ir_msl::tune_precision(
                    &dev,
                    tuned.clone(),
                    &outs,
                    &refs,
                    std::slice::from_ref(&op.want),
                    ErrorBudget::rel(rel),
                    ITERS,
                );
                let assign = if r.assignment.is_empty() && r.fragments.is_empty() {
                    "all f32".to_string()
                } else {
                    let mut parts: Vec<String> = r
                        .assignment
                        .iter()
                        .map(|&(t, p)| {
                            format!(
                                "{} {}",
                                describe(&r.graph, t, &op.names, &outs),
                                fmt_prec(p)
                            )
                        })
                        .collect();
                    parts.extend(r.fragments.iter().map(|&t| {
                        format!("{} on fragments", describe(&r.graph, t, &op.names, &outs))
                    }));
                    parts.join(", ")
                };
                line.push_str(&format!(
                    "\n    tune @ {rel:.0e}: {:.3} → {:.3} ms  err {:.1e}  [{assign}]",
                    r.base_ms, r.ms, r.error.max_rel
                ));
                for t in &r.trials {
                    line.push_str(&format!(
                        "\n        {:<24} err {:.1e}  {}",
                        t.what,
                        t.error.max_rel,
                        match (t.kept, t.ms) {
                            (true, Some(ms)) => format!("kept ({ms:.3} ms)"),
                            (false, Some(ms)) => format!("no gain ({ms:.3} ms)"),
                            _ => "over budget".into(),
                        }
                    ));
                }
                tuned_cols.push(format!("{:.3} [{assign}]", r.ms));
            }
            println!("{line}");
            table.push(format!(
                "| {} | {:.3} ({}) | {forced} | {} | {} |",
                op.name, best.0, best.1, tuned_cols[0], tuned_cols[1]
            ));
        }
        // forward + backward, f32
        let want_op = |n: &str| only.as_deref().is_none_or(|o| n.starts_with(o));
        let mut train = Vec::new();
        if want_op("a") || want_op("train") {
            let (c, t, k, dil) = (512, 2000, 7, 3);
            let x = rand(c * t, 1, 1.0);
            let w = rand(c * c * k, 2, 0.05);
            let xw = vec![TensorId::from_index(0), TensorId::from_index(1)];
            let (g, o) = conv_graph(c, t, k, dil, false);
            let (gi, oi) = conv_graph(c, t, k, dil, true);
            train.push(train_row(
                &dev,
                "a fwd+bwd (dx, dw)",
                vec![("direct", g, o[0], xw.clone()), ("im2col", gi, oi[0], xw)],
                &[&x, &w],
            ));
        }
        if want_op("a") || want_op("train") {
            train.push(train_dy_row(&dev));
        }
        if want_op("f") || want_op("train") {
            let (cf, tf, ff) = (512, 200, 1536);
            let inputs: Vec<Vec<f32>> = vec![
                rand(cf * tf, 12, 1.0),
                rand(cf * 7, 13, 0.3),
                rand(cf, 14, 0.1),
                rand(cf, 15, 0.2).iter().map(|v| v + 1.0).collect(),
                rand(cf, 16, 0.1),
                rand(ff * cf, 17, 0.04),
                rand(ff, 18, 0.1),
                rand(cf * ff, 19, 0.02),
                rand(cf, 20, 0.1),
            ];
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
            let cn = convnext_graph(cf, tf, ff);
            train.push(train_row(
                &dev,
                "f fwd+bwd (all params)",
                vec![("convnext", cn.g, cn.y, cn.params)],
                &refs,
            ));
        }
        println!("\n| op | MSL f32 best (ms) | MSL f16 fragments, forced (ms, err) | tuned @1e-2 (ms) | tuned @1e-3 (ms) |");
        println!("|---|---:|---:|---|---|");
        for r in table {
            println!("{r}");
        }
        for r in train {
            println!("{r}");
        }
    }

    /// Op (a) forward + backward with a given upstream gradient
    /// (`L = Σ y ⊙ gy`, so dx and dw are real contractions with dy; with
    /// `dy = 1` they collapse to sums). f32 autotuned, then
    /// `tune_precision` (fragments for y, dx, dw) against f64 references.
    fn train_dy_row(dev: &MslDevice) -> String {
        let (c, t, k, dil) = (512, 2000, 7, 3);
        let x = rand(c * t, 1, 1.0);
        let w = rand(c * c * k, 2, 0.05);
        let gy = rand(c * t, 3, 1.0);
        let (mut g, o) = conv_graph(c, t, k, dil, false);
        let gyi = g.input(&[c, t]);
        let l = g.map(&[o[0], gyi], |v| v[0] * v[1]).unwrap();
        let xw = [TensorId::from_index(0), TensorId::from_index(1)];
        let grads = g.backward(l, &xw).unwrap();
        let (g, outs) = optimized(&g, &[o[0], grads[0], grads[1]]);
        let (xd, wd, gd) = (f64s(&x), f64s(&w), f64s(&gy));
        let y = conv_ref(&xd, &wd, c, t, k, dil);
        let (dx, dw) = conv_bwd_ref(&xd, &wd, &gd, c, t, k, dil);
        let want = vec![y, dx, dw];
        let inputs: [&[f32]; 3] = [&x, &w, &gy];
        let mut auto = g.clone();
        auto.schedule_auto(&outs);
        let tuned = ir_msl::autotune_kernels(dev, auto, &outs, &inputs, 10).0;
        let exe = Executable::new(dev, tuned.to_msl(&outs), &inputs);
        let f32_ms = time(dev, &exe);
        let f32_err = ir_msl::gpu_error(dev, &exe, &outs, &want).max_rel;
        let mut line = format!(
            "a fwd+bwd, dy given (y, dx, dw):\n    f32 autotune {f32_ms:8.3} ms  {} kernels  err {f32_err:.1e}",
            exe.program.kernels.len()
        );
        let mut cols = Vec::new();
        for rel in [1e-2, 1e-3] {
            let r = ir_msl::tune_precision(
                dev,
                tuned.clone(),
                &outs,
                &inputs,
                &want,
                ErrorBudget::rel(rel),
                ITERS,
            );
            let frags = r.fragments.len();
            line.push_str(&format!(
                "\n    tune @ {rel:.0e}: {:.3} → {:.3} ms  err {:.1e}  {frags} nodes on fragments",
                r.base_ms, r.ms, r.error.max_rel
            ));
            for tr in &r.trials {
                line.push_str(&format!(
                    "\n        {:<24} err {:.1e}  {}",
                    tr.what,
                    tr.error.max_rel,
                    match (tr.kept, tr.ms) {
                        (true, Some(ms)) => format!("kept ({ms:.3} ms)"),
                        (false, Some(ms)) => format!("no gain ({ms:.3} ms)"),
                        _ => "over budget".into(),
                    }
                ));
            }
            cols.push(format!(
                "{:.3} ({frags} on fragments, err {:.0e})",
                r.ms, r.error.max_rel
            ));
        }
        println!("{line}");
        format!(
            "| a fwd+bwd, dy given | {f32_ms:.3} (autotune) | | {} | {} |",
            cols[0], cols[1]
        )
    }

    fn train_row(
        dev: &MslDevice,
        name: &str,
        algebras: Vec<(&'static str, TensorGraph, TensorId, Vec<TensorId>)>,
        inputs: &[&[f32]],
    ) -> String {
        let mut best = (f64::INFINITY, String::new());
        let mut line = format!("{name}:");
        for (alg, mut g, y, wrt) in algebras {
            let grads = g.backward(y, &wrt).unwrap();
            let mut outs = vec![y];
            outs.extend(grads);
            let (g, outs) = optimized(&g, &outs);
            let mut auto = g.clone();
            auto.schedule_auto(&outs);
            let tuned = ir_msl::autotune_kernels(dev, auto.clone(), &outs, inputs, 10).0;
            for (sched, gv) in [("auto", auto), ("autotune", tuned)] {
                let exe = Executable::new(dev, gv.to_msl(&outs), inputs);
                let ms = time(dev, &exe);
                line.push_str(&format!(
                    "\n    {alg:>8}/{sched:<8} {ms:8.3} ms  {} kernels",
                    exe.program.kernels.len()
                ));
                if ms < best.0 {
                    best = (ms, format!("{alg}/{sched}"));
                }
            }
        }
        println!("{line}");
        format!("| {name} | {:.3} ({}) | | | |", best.0, best.1)
    }
}
