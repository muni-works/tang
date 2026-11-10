//! A design space of sequence-mixing layers, filtered by what tang derives.
//!
//! Two families:
//!
//! - **attention**: `O = norm(score(s, i, j, u, w) masked)·v`, where `s` is the
//!   scaled q·k, `u = g[i]` and `w = g[j]` read a per-position input vector `g`
//!   (a learned, data-dependent gate: `s + u − w` with `g` the cumulative log
//!   forget gate is the Forgetting Transformer), and `norm` is one of
//!   [`Norm`]. The question asked of tang: does the derived program (forward,
//!   and forward + backward through q, k, v and g) avoid every `n×n` buffer,
//!   i.e. is it a one-pass flash-style kernel?
//! - **scan**: `S_t = α_t(I − [δ]β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ`, `o = Sᵀq`
//!   (`delta_scan` + `scan_readout`). The question: does
//!   `chunked_scan_plan` recognize it (chunked tensor-core forward, derived
//!   chunked backward)?
//!
//! Nothing here knows which variant is "flash attention" or "Mamba-2"; the
//! answer is whatever tang's pass pipeline produces.

use super::*;
use tang_compile::tensor::wgsl::WgslProgram;

/// How the masked scores become mixing weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// `exp(s − max) / Σ exp(s − max)`
    Softmax,
    /// `exp(s) / (1 + Σ exp(s))` (off-by-one / "quiet" softmax), stabilized
    /// with `max(rowmax, 0)`
    Softmax1,
    /// `σ(s)`, unnormalized (sigmoid attention)
    Sigmoid,
    /// `σ(s) / Σ σ(s)`
    SigmoidNorm,
    /// `relu(s)²`, unnormalized
    Relu2,
    /// `s`, unnormalized (masked linear attention, quadratic form)
    Identity,
}

impl Norm {
    pub const ALL: [Norm; 6] = [
        Norm::Softmax,
        Norm::Softmax1,
        Norm::Sigmoid,
        Norm::SigmoidNorm,
        Norm::Relu2,
        Norm::Identity,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Norm::Softmax => "softmax",
            Norm::Softmax1 => "softmax1",
            Norm::Sigmoid => "sigmoid",
            Norm::SigmoidNorm => "sigmoid_norm",
            Norm::Relu2 => "relu2",
            Norm::Identity => "identity",
        }
    }
}

/// Whether the variant reads the gate vector.
pub fn gated(v: &Variant) -> bool {
    uses(&v.score, 'u') || uses(&v.score, 'w')
}

/// The naive graph of one attention-family variant. Inputs q, k, v `[n,d]`,
/// then g `[n, 1]` when [`gated`], then dO `[n,d]` for the backward (whose outputs
/// are the gradients of `Σ O·dO` w.r.t. q, k, v and, with `gate_grad`, g).
pub fn design_graph(
    v: &Variant,
    norm: Norm,
    n: usize,
    d: usize,
    backward: bool,
    gate_grad: bool,
) -> Result<(TensorGraph, Vec<TensorId>), String> {
    let e = |r: Result<TensorId, tang_compile::tensor::ShapeError>| r.map_err(|e| format!("{e:?}"));
    let mut g = TensorGraph::new();
    let q = g.input(&[n, d]);
    let k = g.input(&[n, d]);
    let vv = g.input(&[n, d]);
    let gate = gated(v).then(|| g.input(&[n, 1]));
    let scale = 1.0 / (d as f64).sqrt();
    let kt = e(g.transpose(k, &[1, 0]))?;
    let raw = e(g.matmul(q, kt))?;
    let ii = g.iota(&[n, n], 0);
    let jj = g.iota(&[n, n], 1);
    let (nf, df) = (n as f64, d as f64);
    {
        let env = |c: char| match c {
            'n' => lit(nf),
            'd' => lit(df),
            _ => ExprId::ZERO,
        };
        tang_expr::trace(|| lower(&v.score, &env).map(|_| ())).1?;
        if let Some(m) = &v.mask {
            tang_expr::trace(|| lower(m, &env).map(|_| ())).1?;
        }
    }
    let mut args = vec![raw, ii, jj];
    if let Some(gt) = gate {
        let gu = e(g.broadcast(gt, &[n, n]))?; // g[i]
        let gr = e(g.transpose(gt, &[1, 0]))?;
        let gw = e(g.broadcast(gr, &[n, n]))?; // g[j]
        args.extend([gu, gw]);
    }
    let score_ast = v.score.clone();
    let mut fail = None;
    let s = e(g.map(&args, |x| {
        let env = |c: char| match c {
            's' => x[0] * lit(scale),
            'i' => x[1],
            'j' => x[2],
            'u' => x[3],
            'w' => x[4],
            'n' => lit(nf),
            _ => lit(df),
        };
        lower(&score_ast, &env).unwrap_or_else(|er| {
            fail = Some(er);
            ExprId::ZERO
        })
    }))?;
    if let Some(er) = fail {
        return Err(er);
    }
    // masked-out weight: −∞ for the exponential norms, 0 for the others
    let exp_norm = matches!(norm, Norm::Softmax | Norm::Softmax1);
    let s = match &v.mask {
        None => s,
        Some(m) => {
            let m = m.clone();
            e(g.map(&[s, ii, jj], |x| {
                let env = |c: char| match c {
                    'i' => x[1],
                    'j' => x[2],
                    'n' => lit(nf),
                    _ => lit(df),
                };
                let keep = lower(&m, &env).unwrap_or(ExprId::ONE);
                if exp_norm {
                    x[0] + sel(keep, ExprId::ZERO, lit(-1e30))
                } else {
                    // applied after the pointwise weight below
                    sel(keep, x[0], lit(f64::NAN))
                }
            }))?
        }
    };
    let y = if exp_norm {
        let mx = e(g.reduce(s, 1, Monoid::max()))?;
        let mx = if norm == Norm::Softmax1 {
            e(g.map(&[mx], |x| sel(x[0], x[0], ExprId::ZERO)))?
        } else {
            mx
        };
        let mb = e(g.broadcast(mx, &[n, n]))?;
        let p = e(g.map(&[s, mb], |x| e_exp(x[0] - x[1])))?;
        let l = e(g.reduce(p, 1, Monoid::sum()))?;
        let l = if norm == Norm::Softmax1 {
            e(g.map(&[l, mx], |x| x[0] + e_exp(-x[1])))?
        } else {
            l
        };
        let o = e(g.matmul(p, vv))?;
        let lb = e(g.broadcast(l, &[n, d]))?;
        e(g.map(&[o, lb], |x| x[0] / x[1]))?
    } else {
        // pointwise weight; a masked entry (NaN marker) is 0. The marker is
        // re-derived from the mask so no NaN ever flows.
        let p = match &v.mask {
            None => e(g.map(&[s], |x| weight(norm, x[0])))?,
            Some(m) => {
                let m = m.clone();
                // rebuild from the unmasked score: weight(s)·keep
                let _ = s;
                let s0 = e(g.map(&args, |x| {
                    let env = |c: char| match c {
                        's' => x[0] * lit(scale),
                        'i' => x[1],
                        'j' => x[2],
                        'u' => x[3],
                        'w' => x[4],
                        'n' => lit(nf),
                        _ => lit(df),
                    };
                    lower(&score_ast, &env).unwrap_or(ExprId::ZERO)
                }))?;
                e(g.map(&[s0, ii, jj], |x| {
                    let env = |c: char| match c {
                        'i' => x[1],
                        'j' => x[2],
                        'n' => lit(nf),
                        _ => lit(df),
                    };
                    let keep = lower(&m, &env).unwrap_or(ExprId::ONE);
                    sel(keep, weight(norm, x[0]), ExprId::ZERO)
                }))?
            }
        };
        let o = e(g.matmul(p, vv))?;
        if norm == Norm::SigmoidNorm {
            let l = e(g.reduce(p, 1, Monoid::sum()))?;
            let lb = e(g.broadcast(l, &[n, d]))?;
            e(g.map(&[o, lb], |x| x[0] / x[1]))?
        } else {
            o
        }
    };
    if !backward {
        return Ok((g, vec![y]));
    }
    let dout = g.input(&[n, d]);
    let prod = e(g.map(&[y, dout], |x| x[0] * x[1]))?;
    let r1 = e(g.reduce(prod, 1, Monoid::sum()))?;
    let loss = e(g.reduce(r1, 0, Monoid::sum()))?;
    let mut wrt = vec![q, k, vv];
    if gate_grad {
        wrt.extend(gate);
    }
    let grads = g.backward(loss, &wrt).map_err(|e| format!("{e:?}"))?;
    Ok((g, grads))
}

fn weight(norm: Norm, s: ExprId) -> ExprId {
    match norm {
        Norm::Sigmoid | Norm::SigmoidNorm => lit(1.0) / (lit(1.0) + e_exp(-s)),
        Norm::Relu2 => {
            let r = sel(s, s, ExprId::ZERO);
            r * r
        }
        _ => s,
    }
}

/// What tang made of one variant.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// `n×n` buffers the naive program writes
    pub naive_nxn: usize,
    /// `n×n` buffers in the derived program
    pub nxn: usize,
    pub kernels: usize,
}

fn nxn(p: &WgslProgram, n: usize) -> usize {
    p.buffers
        .iter()
        .filter(|b| b.input.is_none() && b.len >= n * n)
        .count()
}

/// Derive one variant through the same pipeline the benchmarks run
/// ([`derive_graph`]) and count its `n×n` buffers.
pub fn judge(
    score: &str,
    mask: &str,
    norm: Norm,
    n: usize,
    d: usize,
    backward: bool,
    gate_grad: bool,
) -> Result<Verdict, String> {
    let v = parse_variant(score, mask)?;
    let (g, outs) = design_graph(&v, norm, n, d, backward, gate_grad)?;
    let naive = g.to_wgsl(&outs);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (dg, douts) = derive_graph(&g, &outs);
        dg.to_wgsl(&douts)
    }))
    .map_err(|_| "the compiler panicked".to_string())?;
    Ok(Verdict {
        naive_nxn: nxn(&naive, n),
        nxn: nxn(&r, n),
        kernels: r.kernels.len(),
    })
}

/// The score functions of the attention family: (name, score expression).
/// `u`/`w` are the per-query / per-key gate values.
pub const SCORES: &[(&str, &str)] = &[
    ("plain", "s"),
    ("softcap", "30*tanh(s/30)"),
    ("alibi", "s - 0.0625*(i - j)"),
    ("forget", "s + u - w"),
    ("keybias", "s + w"),
    ("keyscale", "s*sigmoid(w)"),
    ("forget_softcap", "30*tanh(s/30) + u - w"),
    ("forget_alibi", "s + u - w - 0.0625*(i - j)"),
    ("logdist", "s - log(1 + i - j)"),
    ("sigmoid_bias", "s - log(n)"),
];

/// Masks: (name, mask expression).
pub const MASKS: &[(&str, &str)] = &[
    ("causal", "j <= i"),
    ("window128", "j <= i && i - j < 128"),
    ("prefix64", "j <= i || j < 64"),
];

/// The scan family: (name, alpha?, beta?, delta?). Every combination of the
/// three switches of `delta_scan`.
pub const SCANS: &[(&str, bool, bool, bool)] = &[
    ("linear_attn", false, false, false),
    ("retention_like", true, false, false),
    ("beta_linear", false, true, false),
    ("ssd_mamba2", true, true, false),
    ("deltanet_unit", false, false, true),
    ("gated_delta_unit", true, false, true),
    ("deltanet", false, true, true),
    ("gated_deltanet", true, true, true),
];

/// Whether `chunked_scan_plan` takes the scan variant (forward chunked
/// kernels and the derived chunked backward), at `[heads, t, n]`.
pub fn scan_lifts(alpha: bool, beta: bool, delta: bool, heads: usize, t: usize, n: usize) -> bool {
    let mut g = TensorGraph::new();
    let q = g.input(&[heads, t, n]);
    let k = g.input(&[heads, t, n]);
    let v = g.input(&[heads, t, n]);
    let a = alpha.then(|| g.input(&[heads, t]));
    let b = beta.then(|| g.input(&[heads, t]));
    let Ok(s) = g.delta_scan(k, v, a, b, 1, delta) else {
        return false;
    };
    let Ok(o) = g.scan_readout(s, q, 1.0 / (n as f64).sqrt()) else {
        return false;
    };
    g.chunked_scan_plan(o).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole space, printed as CSV (run with `--nocapture`).
    #[test]
    fn design_space_table() {
        let (n, d) = (256, 16);
        println!("family,name,mask,norm,naive_nxn,fwd_nxn,bwd_nxn,bwd_nxn_gate,fwd_kernels,bwd_kernels,lifts_qkv,lifts_all");
        let mut lifted = 0;
        let mut total = 0;
        for &(sn, score) in SCORES {
            for &(mn, mask) in MASKS {
                for norm in Norm::ALL {
                    total += 1;
                    let f = judge(score, mask, norm, n, d, false, false);
                    let b = judge(score, mask, norm, n, d, true, false);
                    let bg = judge(score, mask, norm, n, d, true, true);
                    match (f, b, bg) {
                        (Ok(f), Ok(b), Ok(bg)) => {
                            let ok = f.nxn == 0 && b.nxn == 0;
                            let all = ok && bg.nxn == 0;
                            lifted += all as usize;
                            println!(
                                "attn,{sn},{mn},{},{},{},{},{},{},{},{ok},{all}",
                                norm.name(),
                                f.naive_nxn,
                                f.nxn,
                                b.nxn,
                                bg.nxn,
                                f.kernels,
                                b.kernels
                            );
                        }
                        (f, b, _) => println!(
                            "attn,{sn},{mn},{},ERR {:?} {:?},,,,,,false,false",
                            norm.name(),
                            f.err(),
                            b.err()
                        ),
                    }
                }
            }
        }
        for &(name, a, b, dl) in SCANS {
            total += 1;
            let ok = scan_lifts(a, b, dl, 4, 256, 128);
            lifted += ok as usize;
            println!("scan,{name},causal,-,,,,,,,{ok},{ok}");
        }
        eprintln!("{lifted}/{total} variants lift");
        // the anchors: softmax attention and the gated delta rule must lift
        assert!(
            judge("s", "j <= i", Norm::Softmax, n, d, true, true)
                .unwrap()
                .nxn
                == 0
        );
        assert!(scan_lifts(true, true, true, 4, 256, 128));
    }
}
