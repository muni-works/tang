//! Fused delta-rule steps for the CUDA decode lowering.
//!
//! A derived step that appends `k > 1` rows runs a gated-delta scan from
//! the cached state (`DeltaForm::init`) and reads it twice: the readout
//! `o_t = scale·S_tᵀ·q_t` of every new row and the last state (the
//! updated cache). Lowered as written, the scan stores every `S_t` (and
//! its consumers store the dense `A_t`), `k·n·m` floats a head — 2 GB a
//! layer for coder-next at `k = 256`. [`TensorGraph::fuse_delta_steps`]
//! replaces the readout and the last-row pick by [`ScanKind::Delta`]
//! nodes reading the form's `k`, `v`, `α`, `β` and the initial state;
//! [`TensorGraph::cuda_delta_kernel`] runs each as one kernel with the
//! state in registers: a block per (head, column block), the `n` rows of
//! a column split over `parts` thread groups, two block reductions a step
//! (`kᵀS` for the delta term, `qᵀS` for the readout).

use std::collections::HashMap;
use std::fmt::Write;

use super::{
    fixed, DeltaStep, Form, Format, MonoidKind, Op, Scan, ScanKind, Semiring, TensorGraph,
    TensorId, View,
};

impl TensorGraph {
    /// The graph with every delta scan that starts from a state and is
    /// read only by readouts and its last row rewritten as fused
    /// [`ScanKind::Delta`] nodes (see the module docs); `None` when there
    /// is nothing to fuse.
    pub(super) fn fuse_delta_steps(&self, outputs: &[TensorId]) -> Option<TensorGraph> {
        let mut users: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.nodes.len() {
            for d in self.deps(i) {
                users.entry(d).or_default().push(i);
            }
        }
        let mut out: Option<TensorGraph> = None;
        for j in 0..self.nodes.len() {
            let Op::Scan(sc) = &self.nodes[j].op else {
                continue;
            };
            let ScanKind::Affine {
                form: Some(f),
                ring: Semiring::Real,
                bands: None,
            } = &sc.kind
            else {
                continue;
            };
            let Some(init) = f.init else { continue };
            if sc.exclusive || sc.reverse || outputs.iter().any(|o| o.n() == j) {
                continue;
            }
            let us = users.get(&j).cloned().unwrap_or_default();
            if us.is_empty() {
                continue;
            }
            let k = sc.inputs[f.k];
            let ks = fixed(self.shape(k));
            let p = ks.len() - 1;
            let n = ks[p];
            let m = *fixed(self.shape(sc.inputs[f.v])).last().unwrap();
            let mut base = vec![k, sc.inputs[f.v]];
            base.extend(f.alpha.map(|i| sc.inputs[i]));
            base.extend(f.beta.map(|i| sc.inputs[i]));
            base.push(sc.inputs[init]);
            let mut plan = Vec::new();
            for &u in &us {
                if let Some((scale, q)) = self.delta_readout(u, j, p) {
                    if fixed(self.shape(q)) != ks {
                        plan.clear();
                        break;
                    }
                    let mut ins = base.clone();
                    ins.push(q);
                    plan.push((u, ins, Some(scale)));
                } else if self.delta_last_row(u, j, p, sc.dim) {
                    plan.push((u, base.clone(), None));
                } else {
                    plan.clear();
                    break;
                }
            }
            if plan.is_empty() || Self::delta_threads(n, m).is_none() {
                continue;
            }
            // f32 buffers only
            let f32s = plan
                .iter()
                .flat_map(|(u, ins, _)| ins.iter().map(|t| self.storage(*t).n()).chain([*u]))
                .all(|i| self.nodes[i].precision.storage == Format::F32);
            if !f32s {
                continue;
            }
            let g = out.get_or_insert_with(|| self.clone());
            for (u, inputs, readout) in plan {
                g.nodes[u].op = Op::Scan(Scan {
                    inputs,
                    dim: sc.dim,
                    exclusive: false,
                    reverse: false,
                    kind: ScanKind::Delta(DeltaStep {
                        delta: f.delta,
                        alpha: f.alpha.is_some(),
                        beta: f.beta.is_some(),
                        readout,
                    }),
                });
                g.nodes[u].inline = false;
            }
        }
        out
    }

    /// `(scale, q)` when node `u` is `scan_readout(s, q, scale)` of the
    /// scan `s` with `p` point dims.
    fn delta_readout(&self, u: usize, s: usize, p: usize) -> Option<(f64, TensorId)> {
        let c = self.comp_at(u)?;
        if c.monoid.as_ref().map(|m| m.kind()) != Some(MonoidKind::Sum)
            || c.args.len() != 2
            || c.red.len() != 1
            || c.tuple != 0
            || c.args[0].src.n() != s
            || self.nodes[u].segment.is_some()
        {
            return None;
        }
        let pts: Vec<Option<usize>> = (0..p).map(Some).collect();
        let mut sp = pts.clone();
        sp.extend([Some(p + 1), Some(p)]);
        let mut qp = pts;
        qp.push(Some(p + 1));
        if c.args[0].view != View::pick(p + 2, &sp) || c.args[1].view != View::pick(p + 2, &qp) {
            return None;
        }
        let a = c.body.eval_many(&c.body_outs, &[1.0, 1.0]);
        let b = c.body.eval_many(&c.body_outs, &[2.0, 3.0]);
        let scale: f64 = a[0];
        ((b[0] - 6.0 * scale).abs() <= 1e-12 * scale.abs().max(1.0))
            .then_some((scale, c.args[1].src))
    }

    /// Node `u` is the last row of scan `s` along `dim` (the axis
    /// dropped), as `Deriver::last_row` builds it.
    fn delta_last_row(&self, u: usize, s: usize, p: usize, dim: usize) -> bool {
        let Some(c) = self.comp_at(u) else {
            return false;
        };
        if c.monoid.is_some() || !c.red.is_empty() || c.args.len() != 1 || c.args[0].src.n() != s {
            return false;
        }
        let tn = fixed(&self.nodes[s].shape)[dim] as i64;
        let r = p + 1;
        let forms: Vec<Form> = (0..p + 2)
            .map(|d| {
                if d == dim {
                    Form::constant(r, tn - 1)
                } else {
                    Form::unit(r, if d < dim { d } else { d - 1 })
                }
            })
            .collect();
        let want = View {
            forms,
            guard: false,
            gather: None,
        };
        if c.args[0].view != want {
            return false;
        }
        let x = c.body.eval_many(&c.body_outs, &[2.5]);
        x.len() == 1 && x[0] == 2.5
    }

    /// `(column block, row parts)`: a block is `bv · parts` threads, each
    /// holding `n / parts` rows of one column.
    fn delta_threads(n: usize, m: usize) -> Option<(usize, usize)> {
        let parts = [16, 8, 4, 2, 1]
            .into_iter()
            .find(|&x| n.is_multiple_of(x))?;
        let bv = [16, 8, 4, 2, 1]
            .into_iter()
            .find(|&x| m.is_multiple_of(x))?;
        Some((bv, parts))
    }

    /// The CUDA kernel of a [`ScanKind::Delta`] node: `(source, threads,
    /// blocks)`, parameters the kernel's `reads` then `out`.
    pub(super) fn cuda_delta_kernel(&self, root: usize, reads: &[usize]) -> (String, usize, usize) {
        let sc = self.scan_at(TensorId(root as u32)).unwrap();
        let ScanKind::Delta(d) = &sc.kind else {
            unreachable!()
        };
        let (ik, iv, ia, ib, ii, iq) = d.slots();
        let ks = fixed(self.shape(sc.inputs[ik]));
        let p = ks.len() - 1;
        let n = ks[p];
        let m = *fixed(self.shape(sc.inputs[iv])).last().unwrap();
        let tn = ks[sc.dim];
        let lo: usize = ks[sc.dim + 1..p].iter().product();
        let heads: usize = ks[..p].iter().product::<usize>() / tn;
        // many rows (a prefill chunk, `TANG_DELTA_CHUNK_MIN` = 64): the
        // chunked form
        if tn >= env_usize("TANG_DELTA_CHUNK_MIN", 64)
            && m.is_multiple_of(8)
            && n <= 256
            && std::env::var("TANG_DELTA_CHUNK").as_deref() != Ok("0")
        {
            return self.cuda_delta_chunked(root, reads);
        }
        let (bv, parts) = Self::delta_threads(n, m).unwrap();
        let (threads, rows, ncb) = (bv * parts, n / parts, m / bv);
        let arg = |i: usize| {
            let s = self.storage(sc.inputs[i]).n();
            format!("q{}", reads.iter().position(|&r| r == s).unwrap())
        };
        let params: Vec<String> = (0..reads.len())
            .map(|i| format!("const float* __restrict__ q{i}"))
            .collect();
        let (kk, vv, s0) = (arg(ik), arg(iv), arg(ii));
        let al = ia.map_or("1.0f".to_string(), |i| format!("{}[p]", arg(i)));
        let be = ib.map_or("1.0f".to_string(), |i| format!("{}[p]", arg(i)));
        // a column's `parts` row groups are adjacent lanes (a power of two
        // ≤ 16, so a group never crosses a warp): both reductions are
        // shuffles. Rows interleave (`part + parts·j`) so a group reads
        // consecutive shared words. The inputs of `TC` steps are staged in
        // shared memory at once: two barriers per `TC` steps.
        let tc = (4096 / n).clamp(1, 32);
        let iqs = iq.map(&arg);
        let mut src = String::new();
        writeln!(
            src,
            "// Auto-generated by tang-compile: fused delta step (decode)"
        )
        .unwrap();
        write!(
            src,
            r#"extern "C" __global__ void __launch_bounds__({threads}) main0({params}, float* __restrict__ out, unsigned count) {{
    __shared__ float sk[{tc}][{n}];
    __shared__ float sq[{tcq}][{nq}];
    __shared__ float sv[{tc}][{bv}];
    __shared__ float sab[{tc}][2];
    const int h = blockIdx.x / {ncb}, cb = blockIdx.x % {ncb};
    const int c = threadIdx.x / {parts}, part = threadIdx.x % {parts};
    const int col = cb * {bv} + c;
    const long long hi = h / {lo}, lo = h % {lo};
    float S[{rows}];
    for (int j = 0; j < {rows}; ++j) S[j] = {s0}[(long long)h * {nm} + (long long)(part + {parts} * j) * {m} + col];
    for (int t0 = 0; t0 < {tn}; t0 += {tc}) {{
        const int nt = min({tc}, {tn} - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * {n}; i += {threads}) {{
            const int tt = i / {n}, e = i % {n};
            const long long p = (hi * {tn} + t0 + tt) * {lo} + lo;
            sk[tt][e] = {kk}[p * {n} + e];
{qload}        }}
        for (int i = threadIdx.x; i < nt * {bv}; i += {threads}) {{
            const int tt = i / {bv}, e = i % {bv};
            const long long p = (hi * {tn} + t0 + tt) * {lo} + lo;
            sv[tt][e] = {vv}[p * {m} + cb * {bv} + e];
        }}
        for (int tt = threadIdx.x; tt < nt; tt += {threads}) {{
            const long long p = (hi * {tn} + t0 + tt) * {lo} + lo;
            sab[tt][0] = {al}; sab[tt][1] = {be};
        }}
        __syncthreads();
        for (int tt = 0; tt < nt; ++tt) {{
            const float al = sab[tt][0], be = sab[tt][1], vc = sv[tt][c];
"#,
            params = params.join(", "),
            nm = n * m,
            tcq = if iq.is_some() { tc } else { 1 },
            nq = if iq.is_some() { n } else { 1 },
            qload = match &iqs {
                Some(q) => format!("            sq[tt][e] = {q}[p * {n} + e];\n"),
                None => String::new(),
            },
        )
        .unwrap();
        // a sum over the column's row groups, in every lane of the group
        let group_sum = |v: &str| {
            let mut s = String::new();
            let mut off = parts / 2;
            while off > 0 {
                write!(s, " {v} += __shfl_xor_sync(0xffffffffu, {v}, {off});").unwrap();
                off /= 2;
            }
            s
        };
        if d.delta {
            writeln!(
                src,
                "            float ks = 0.0f;\n            for (int j = 0; j < {rows}; ++j) ks += sk[tt][part + {parts} * j] * S[j];\n           {}",
                group_sum("ks")
            )
            .unwrap();
        } else {
            writeln!(src, "            const float ks = 0.0f;").unwrap();
        }
        writeln!(
            src,
            "            for (int j = 0; j < {rows}; ++j) {{ const float kr = sk[tt][part + {parts} * j]; S[j] = al * (S[j] - be * kr * ks) + be * kr * vc; }}"
        )
        .unwrap();
        if let Some(scale) = d.readout {
            writeln!(
                src,
                "            float o = 0.0f;\n            for (int j = 0; j < {rows}; ++j) o += sq[tt][part + {parts} * j] * S[j];\n           {}\n            if (part == 0) out[((hi * {tn} + t0 + tt) * {lo} + lo) * {m} + col] = {scale:.9e}f * o;",
                group_sum("o")
            )
            .unwrap();
        }
        writeln!(src, "        }}\n    }}").unwrap();
        if d.readout.is_none() {
            writeln!(
                src,
                "    for (int j = 0; j < {rows}; ++j) out[(long long)h * {} + (long long)(part + {parts} * j) * {m} + col] = S[j];",
                n * m
            )
            .unwrap();
        }
        writeln!(src, "    (void)count;\n}}").unwrap();
        (src, threads, heads * ncb)
    }

    /// [`cuda_delta_kernel`](Self::cuda_delta_kernel) for many rows: a
    /// block per (head, `bv` state columns) walks the rows in chunks of
    /// `C = 64`. With `γ` the running sum of `log α` in a chunk and `S₀`
    /// the state entering it, the recurrence
    /// `S_t = α_t(S − β_t k_t(k_tᵀS)) + β_t k_t v_tᵀ` is
    /// `S_t = e^{γ_t}S₀ + Σ_{s≤t} e^{γ_t−γ_s} k_s u_sᵀ` where
    /// `(I + A)U = βV − (βe^γ ⊙ K)S₀`, `A_ts = β_t e^{γ_t−γ_s} k_t·k_s`
    /// (`s < t`). Then `o_t = scale·(e^{γ_t} q_t·S₀ + Σ_{s≤t}
    /// e^{γ_t−γ_s}(q_t·k_s) u_s)` and `S₀ ← e^{γ_C}S₀ + Σ_s e^{γ_C−γ_s}
    /// k_s u_sᵀ`. Every product (`K·Kᵀ`, `K·S₀`, `Q·Kᵀ`, `Q·S₀`, `P·U`,
    /// `Kᵀ·U`) runs on tf32 `wmma` (16 × 16 × 8, f32 accumulation); the
    /// triangular solve is one thread per column with `U`'s column in
    /// registers. Shared buffers are zero-padded to 16 (`n`, `bv` < 16 in
    /// the test models).
    fn cuda_delta_chunked(&self, root: usize, reads: &[usize]) -> (String, usize, usize) {
        let sc = self.scan_at(TensorId(root as u32)).unwrap();
        let ScanKind::Delta(d) = &sc.kind else {
            unreachable!()
        };
        let (ik, iv, ia, ib, ii, iq) = d.slots();
        let ks = fixed(self.shape(sc.inputs[ik]));
        let p = ks.len() - 1;
        let n = ks[p];
        let m = *fixed(self.shape(sc.inputs[iv])).last().unwrap();
        let tn = ks[sc.dim];
        let lo: usize = ks[sc.dim + 1..p].iter().product();
        let heads: usize = ks[..p].iter().product::<usize>() / tn;
        // state columns a block (`TANG_DELTA_BV`, 32): more blocks vs more
        // recomputed Gram products
        let bv = [env_usize("TANG_DELTA_BV", 32), 16, 8]
            .into_iter()
            .find(|b| m.is_multiple_of(*b))
            .unwrap();
        let ncb = m / bv;
        let cc = 64usize;
        // rows a chunk that take part (`TANG_DELTA_CHUNK_ROWS`, 64; the
        // tiles stay 64 wide, the rest zero)
        let cr = env_usize("TANG_DELTA_CHUNK_ROWS", 64).clamp(2, 64);
        let np = n.div_ceil(16) * 16;
        let bp = bv.div_ceil(16) * 16;
        let (lk, lm, lb) = (np + 4, cc + 4, bp + 4);
        let arg = |i: usize| {
            let s = self.storage(sc.inputs[i]).n();
            format!("q{}", reads.iter().position(|&r| r == s).unwrap())
        };
        let params: Vec<String> = (0..reads.len())
            .map(|i| format!("const float* __restrict__ q{i}"))
            .collect();
        let (kk, vv, s0) = (arg(ik), arg(iv), arg(ii));
        let al = ia.map_or("1.0f".to_string(), |i| format!("{}[p]", arg(i)));
        let be = ib.map_or("1.0f".to_string(), |i| format!("{}[p]", arg(i)));
        let ro = d.readout.is_some();
        let qf = if ro { cc * lk } else { 0 };
        let smem = 4 * (cc * lk + qf + cc * lm + np * lb + 3 * cc * lb + 3 * 64);
        let qload = match iq {
            Some(i) => format!(
                "            Qs[t * {lk} + x] = (t < nt && x < {n}) ? {}[p * {n} + x] : 0.f;\n",
                arg(i)
            ),
            None => String::new(),
        };
        // C[rows × cols] (+)= A·B over `kd` (multiples of 16, 16, 8), a
        // warp per 16 × 16 tile
        let gemm = |name: &str,
                    rows: usize,
                    cols: usize,
                    kd: usize,
                    a: &str,
                    la: usize,
                    a_col: bool,
                    b: &str,
                    lbb: usize,
                    b_col: bool,
                    c: &str,
                    lc: usize,
                    acc_in: bool| {
            let fa = if a_col { "FAc" } else { "FA" };
            let fb = if b_col { "FBc" } else { "FB" };
            let a_at = if a_col {
                format!("{a} + (k0) * {la} + r0")
            } else {
                format!("{a} + r0 * {la} + (k0)")
            };
            let b_at = if b_col {
                format!("{b} + c0 * {lbb} + (k0)")
            } else {
                format!("{b} + (k0) * {lbb} + c0")
            };
            format!(
                r#"        // {name}
        for (int tile = warp; tile < {nt_}; tile += 8) {{
            const int r0 = tile / {tc} * 16, c0 = tile % {tc} * 16;
            FC acc;
            {init}
            for (int k0 = 0; k0 < {kd}; k0 += 8) {{
                {fa} fa_; {fb} fb_;
                wmma::load_matrix_sync(fa_, {a_at}, {la});
                wmma::load_matrix_sync(fb_, {b_at}, {lbb});
                wmma::mma_sync(acc, fa_, fb_, acc);
            }}
            wmma::store_matrix_sync({c} + r0 * {lc} + c0, acc, {lc}, wmma::mem_row_major);
        }}
        __syncthreads();
"#,
                nt_ = (rows / 16) * (cols / 16),
                tc = cols / 16,
                init = if acc_in {
                    format!("wmma::load_matrix_sync(acc, {c} + r0 * {lc} + c0, {lc}, wmma::mem_row_major);")
                } else {
                    "wmma::fill_fragment(acc, 0.f);".to_string()
                },
            )
        };
        let delta = if d.delta {
            format!(
                "{}{}{}",
                gemm(
                    "A = K·Kᵀ",
                    cc,
                    cc,
                    np,
                    "Ks",
                    lk,
                    false,
                    "Ks",
                    lk,
                    true,
                    "Ms",
                    lm,
                    false
                ),
                format!(
                    r#"        for (int e = tid; e < {cc} * {cc}; e += 256) {{
            const int t = e / {cc}, s_ = e % {cc};
            float& a = Ms[t * {lm} + s_];
            a = (s_ < t && t < nt) ? a * bs[t] * expf(gs[t] - gs[s_]) : 0.f;
        }}
"#
                ),
                gemm(
                    "R = K·S₀",
                    cc,
                    bp,
                    np,
                    "Ks",
                    lk,
                    false,
                    "S0",
                    lb,
                    false,
                    "Rs",
                    lb,
                    false
                ),
            ) + &format!(
                r#"        for (int e = tid; e < {cc} * {bp}; e += 256) {{
            const int t = e / {bp}, c = e % {bp};
            Rs[t * {lb} + c] = t < nt ? bs[t] * (Vs[t * {lb} + c] - expf(gs[t]) * Rs[t * {lb} + c]) : 0.f;
        }}
        __syncthreads();
        // (I + A)U = R: a thread per column, U's column in registers
        if (tid < {bp}) {{
            float u[{cc}];
            #pragma unroll
            for (int t = 0; t < {cc}; ++t) {{
                float a = Rs[t * {lb} + tid];
                #pragma unroll
                for (int s_ = 0; s_ < t; ++s_) a -= Ms[t * {lm} + s_] * u[s_];
                u[t] = a;
            }}
            #pragma unroll
            for (int t = 0; t < {cc}; ++t) Us[t * {lb} + tid] = u[t];
        }}
        __syncthreads();
"#
            )
        } else {
            format!(
                "        for (int e = tid; e < {cc} * {bp}; e += 256) Us[(e / {bp}) * {lb} + e % {bp}] = bs[e / {bp}] * Vs[(e / {bp}) * {lb} + e % {bp}];\n        __syncthreads();\n"
            )
        };
        let readout = match d.readout {
            Some(scale) => format!(
                "{}{}{}{}{}",
                gemm("P = Q·Kᵀ", cc, cc, np, "Qs", lk, false, "Ks", lk, true, "Ms", lm, false),
                format!(
                    r#"        for (int e = tid; e < {cc} * {cc}; e += 256) {{
            const int t = e / {cc}, s_ = e % {cc};
            float& a = Ms[t * {lm} + s_];
            a = (s_ <= t && t < nt) ? a * expf(gs[t] - gs[s_]) : 0.f;
        }}
"#
                ),
                gemm("e^γ·Q·S₀", cc, bp, np, "Qs", lk, false, "S0", lb, false, "Rs", lb, false),
                format!(
                    "        for (int e = tid; e < {cc} * {bp}; e += 256) Rs[(e / {bp}) * {lb} + e % {bp}] *= expf(gs[e / {bp}]);\n        __syncthreads();\n"
                ),
                gemm("O = e^γ·Q·S₀ + P·U", cc, bp, cc, "Ms", lm, false, "Us", lb, false, "Rs", lb, true),
            ) + &format!(
                r#"        for (int e = tid; e < nt * {bv}; e += 256) {{
            const int t = e / {bv}, c = e % {bv};
            const long long p = (hi * {tn} + t0 + t) * {lo} + lo;
            out[p * {m} + cb * {bv} + c] = {scale:.9e}f * Rs[t * {lb} + c];
        }}
        __syncthreads();
"#
            ),
            None => String::new(),
        };
        let state = format!(
            r#"        // S₀ ← e^{{γ_C}} S₀ + Kᵀ·(e^{{γ_C−γ}} ⊙ U)
        {{
            const float gc = gs[nt - 1];
            for (int e = tid; e < {cc} * {bp}; e += 256) {{ const int t = e / {bp}; Us[t * {lb} + e % {bp}] *= t < nt ? expf(gc - gs[t]) : 0.f; }}
            for (int e = tid; e < {np} * {bp}; e += 256) S0[(e / {bp}) * {lb} + e % {bp}] *= expf(gc);
            __syncthreads();
        }}
{}"#,
            gemm(
                "S₀ += Kᵀ·U",
                np,
                bp,
                cc,
                "Ks",
                lk,
                true,
                "Us",
                lb,
                false,
                "S0",
                lb,
                true
            )
        );
        let fin = if ro {
            String::new()
        } else {
            format!(
                "    for (int e = tid; e < {n} * {bv}; e += 256) out[(long long)h * {nm} + (long long)(e / {bv}) * {m} + cb * {bv} + e % {bv}] = S0[(e / {bv}) * {lb} + e % {bv}];\n",
                nm = n * m
            )
        };
        let src = format!(
            r#"#define TANG_SMEM {smem}
// Auto-generated by tang-compile: fused delta step (decode), chunked tf32
#include <mma.h>
using namespace nvcuda;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> FA;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> FAc;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> FB;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> FBc;
typedef wmma::fragment<wmma::accumulator, 16, 16, 8, float> FC;
extern "C" __global__ void __launch_bounds__(256) main0({params}, float* __restrict__ out, unsigned count) {{
    extern __shared__ float4 tang_dsm4[];
    float* Ks = (float*)tang_dsm4;
    float* Qs = Ks + {cc} * {lk};
    float* Ms = Qs + {qf};
    float* S0 = Ms + {cc} * {lm};
    float* Us = S0 + {np} * {lb};
    float* Rs = Us + {cc} * {lb};
    float* Vs = Rs + {cc} * {lb};
    float* gs = Vs + {cc} * {lb};
    float* bs = gs + 64;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int h = blockIdx.x / {ncb}, cb = blockIdx.x % {ncb};
    const long long hi = h / {lo}, lo = h % {lo};
    for (int e = tid; e < {np} * {bp}; e += 256) {{
        const int x = e / {bp}, c = e % {bp};
        S0[x * {lb} + c] = (x < {n} && c < {bv}) ? {s0}[(long long)h * {nm} + (long long)x * {m} + cb * {bv} + c] : 0.f;
    }}
    for (int t0 = 0; t0 < {tn}; t0 += {cr}) {{
        const int nt = min({cr}, {tn} - t0);
        __syncthreads();
        for (int e = tid; e < {cc} * {np}; e += 256) {{
            const int t = e / {np}, x = e % {np};
            const long long p = (hi * {tn} + t0 + min(t, nt - 1)) * {lo} + lo;
            Ks[t * {lk} + x] = (t < nt && x < {n}) ? {kk}[p * {n} + x] : 0.f;
{qload}        }}
        for (int e = tid; e < {cc} * {bp}; e += 256) {{
            const int t = e / {bp}, c = e % {bp};
            const long long p = (hi * {tn} + t0 + min(t, nt - 1)) * {lo} + lo;
            Vs[t * {lb} + c] = (t < nt && c < {bv}) ? {vv}[p * {m} + cb * {bv} + c] : 0.f;
        }}
        // γ: the running sum of log α (one warp's scan of 64), β
        if (tid < 32) {{
            float l[2];
            for (int i = 0; i < 2; ++i) {{
                const int t = 2 * tid + i;
                const long long p = (hi * {tn} + t0 + min(t, nt - 1)) * {lo} + lo;
                // α may underflow to 0: log α ≥ −80 (e^{{−80}} is 0 in f32 already)
                l[i] = t < nt ? fmaxf(logf({al}), -80.f) : 0.f;
                bs[t] = t < nt ? {be} : 0.f;
            }}
            float x = l[0] + l[1];
            for (int o = 1; o < 32; o *= 2) {{ const float y = __shfl_up_sync(0xffffffffu, x, o); if (tid >= o) x += y; }}
            const float ex = x - (l[0] + l[1]);
            gs[2 * tid] = ex + l[0]; gs[2 * tid + 1] = x;
        }}
        __syncthreads();
{delta}{readout}{state}    }}
    __syncthreads();
{fin}    (void)count;
}}
"#,
            params = params.join(", "),
            nm = n * m,
        );
        (src, 256, heads * ncb)
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
