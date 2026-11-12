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
    /// block per (head, 16 (or 8) state columns) walks the rows in chunks of
    /// `C = 64`. With `γ` the running sum of `log α` in a chunk and `S₀`
    /// the state entering it, the recurrence
    /// `S_t = α_t(S − β_t k_t(k_tᵀS)) + β_t k_t v_tᵀ` is
    /// `S_t = e^{γ_t}S₀ + Σ_{s≤t} e^{γ_t−γ_s} k_s u_sᵀ` where
    /// `(I + A)U = βV − (βe^γ ⊙ K)S₀`, `A_ts = β_t e^{γ_t−γ_s} k_t·k_s`
    /// (`s < t`; forward substitution, the chunk's rows in order). Then
    /// `o_t = scale·(e^{γ_t} q_t·S₀ + Σ_{s≤t} e^{γ_t−γ_s}(q_t·k_s) u_s)` and
    /// `S₀ ← e^{γ_C}S₀ + Σ_s e^{γ_C−γ_s} k_s u_sᵀ`. The same sums as the
    /// step kernel, regrouped: matrix products over the chunk instead of
    /// `C` dependent steps.
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
        // rows a chunk (`TANG_DELTA_CHUNK_ROWS`, 64; 2..=64)
        let bv = if m.is_multiple_of(16) { 16usize } else { 8 };
        let cc = env_usize("TANG_DELTA_CHUNK_ROWS", 64).clamp(2, 64);
        let ncb = m / bv;
        let ld = n + 1;
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
        // shared floats: K, Q [C][n+1], M [C][C], S₀ [n][bv], U, R, V [C][bv], γ, β [C]
        let qf = if ro { cc * ld } else { 0 };
        // γ and β: 64 each (the scan writes a full warp's 64)
        let smem = 4 * (cc * ld + qf + cc * cc + n * bv + 3 * cc * bv + 128);
        let qload = match iq {
            Some(i) => format!("            Qs[t * {ld} + x] = {}[p * {n} + x];\n", arg(i)),
            None => String::new(),
        };
        let delta = if d.delta {
            format!(
                r#"        // A (strictly lower) and R = βV − (βe^γ ⊙ K)S₀
        gram(Ks, true);
        for (int e = tid; e < {cc} * {bv}; e += 256) {{
            const int t = e / {bv}, c = e % {bv};
            float a = 0.f;
            if (t < nt) {{ const float* kt = Ks + t * {ld}; for (int x = 0; x < {n}; ++x) a += kt[x] * S0[x * {bv} + c]; }}
            Rs[e] = bs[t] * (Vs[e] - expf(gs[t]) * a);
        }}
        __syncthreads();
        // forward substitution: U_t = R_t − Σ_{{s<t}} A_ts U_s, a column per
        // 16 lanes, U in registers (lane `part` holds rows part + 16·j): no
        // barrier in the loop
        {{
            const int c = col16 < {bv} ? col16 : 0;
            float u[4] = {{0.f, 0.f, 0.f, 0.f}};
            for (int t = 0; t < nt; ++t) {{
                float a = 0.f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {{ const int s_ = part + 16 * j; if (s_ < t) a += Ms[t * {cc} + s_] * u[j]; }}
                a += __shfl_xor_sync(0xffffffffu, a, 8); a += __shfl_xor_sync(0xffffffffu, a, 4);
                a += __shfl_xor_sync(0xffffffffu, a, 2); a += __shfl_xor_sync(0xffffffffu, a, 1);
                if (part == (t & 15)) u[t >> 4] = Rs[t * {bv} + c] - a;
            }}
            if (col16 < {bv}) {{
                #pragma unroll
                for (int j = 0; j < 4; ++j) {{ const int s_ = part + 16 * j; if (s_ < {cc}) Us[s_ * {bv} + c] = s_ < nt ? u[j] : 0.f; }}
            }}
        }}
        __syncthreads();
"#
            )
        } else {
            format!(
                r#"        for (int e = tid; e < {cc} * {bv}; e += 256) Us[e] = e / {bv} < nt ? bs[e / {bv}] * Vs[e] : 0.f;
        __syncthreads();
"#
            )
        };
        let readout = match d.readout {
            Some(scale) => format!(
                r#"        // P = tril(Q Kᵀ ⊙ e^{{γt−γs}}), then o = scale·(e^γ Q S₀ + P U)
        gram(Qs, false);
        __syncthreads();
        for (int e = tid; e < {cc} * {bv}; e += 256) {{
            const int t = e / {bv}, c = e % {bv};
            if (t < nt) {{
                float a = 0.f, b = 0.f;
                const float* qt = Qs + t * {ld};
                for (int x = 0; x < {n}; ++x) a += qt[x] * S0[x * {bv} + c];
                for (int s_ = 0; s_ <= t; ++s_) b += Ms[t * {cc} + s_] * Us[s_ * {bv} + c];
                const long long p = (hi * {tn} + t0 + t) * {lo} + lo;
                out[p * {m} + cb * {bv} + c] = {scale:.9e}f * (expf(gs[t]) * a + b);
            }}
        }}
        __syncthreads();
"#
            ),
            None => String::new(),
        };
        let fin = if ro {
            String::new()
        } else {
            format!(
                "    for (int e = tid; e < {n} * {bv}; e += 256) out[(long long)h * {nm} + (long long)(e / {bv}) * {m} + cb * {bv} + e % {bv}] = S0[e];\n",
                nm = n * m
            )
        };
        let src = format!(
            r#"#define TANG_SMEM {smem}
// Auto-generated by tang-compile: fused delta step (decode), chunked
extern "C" __global__ void __launch_bounds__(256) main0({params}, float* __restrict__ out, unsigned count) {{
    extern __shared__ float tang_dsm[];
    float* Ks = tang_dsm;
    float* Qs = Ks + {cc} * {ld};
    float* Ms = Qs + {qf};
    float* S0 = Ms + {cc} * {cc};
    float* Us = S0 + {n} * {bv};
    float* Rs = Us + {cc} * {bv};
    float* Vs = Rs + {cc} * {bv};
    float* gs = Vs + {cc} * {bv};
    float* bs = gs + 64;
    const int tid = threadIdx.x, col16 = tid / 16, part = tid % 16;
    // Ms = a C × C Gram of X's rows with K's, 4 × 4 a thread (rows ti + 16a,
    // columns si + 16b: a warp reads distinct banks of K), masked lower
    // (strict: × β_t, the delta system; else the readout's) and decayed
    auto gram = [&](const float* X, bool strict) {{
        const int ti = tid / 16, si = tid % 16;
        for (int t4 = 0; t4 < {cc}; t4 += 64) {{
        float acc[4][4] = {{}};
        #pragma unroll 4
        for (int x = 0; x < {n}; ++x) {{
            float xa[4], kb[4];
            #pragma unroll
            for (int a = 0; a < 4; ++a) xa[a] = X[(ti + 16 * a) % {cc} * {ld} + x];
            #pragma unroll
            for (int b = 0; b < 4; ++b) kb[b] = Ks[(si + 16 * b) % {cc} * {ld} + x];
            #pragma unroll
            for (int a = 0; a < 4; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] += xa[a] * kb[b];
        }}
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 4; ++b) {{
                const int t = ti + 16 * a, s_ = si + 16 * b;
                if (t >= {cc} || s_ >= {cc}) continue;
                const bool on = t < nt && (strict ? s_ < t : s_ <= t);
                Ms[t * {cc} + s_] = on ? acc[a][b] * (strict ? bs[t] : 1.f) * expf(gs[t] - gs[s_]) : 0.f;
            }}
        }}
    }};
    const int h = blockIdx.x / {ncb}, cb = blockIdx.x % {ncb};
    const long long hi = h / {lo}, lo = h % {lo};
    for (int e = tid; e < {n} * {bv}; e += 256) S0[e] = {s0}[(long long)h * {nm} + (long long)(e / {bv}) * {m} + cb * {bv} + e % {bv}];
    for (int t0 = 0; t0 < {tn}; t0 += {cc}) {{
        const int nt = min({cc}, {tn} - t0);
        __syncthreads();
        for (int e = tid; e < nt * {n}; e += 256) {{
            const int t = e / {n}, x = e % {n};
            const long long p = (hi * {tn} + t0 + t) * {lo} + lo;
            Ks[t * {ld} + x] = {kk}[p * {n} + x];
{qload}        }}
        for (int e = tid; e < {cc} * {bv}; e += 256) {{
            const int t = e / {bv};
            const long long p = (hi * {tn} + t0 + min(t, nt - 1)) * {lo} + lo;
            Vs[e] = t < nt ? {vv}[p * {m} + cb * {bv} + e % {bv}] : 0.f;
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
{delta}{readout}        // S₀ ← e^{{γ_C}} S₀ + Σ_s e^{{γ_C−γ_s}} k_s u_sᵀ
        {{
            const float gc = gs[nt - 1];
            // e^{{γ_C−γ_s}}·u_s into R's slots (R is not read again this chunk)
            __syncthreads();
            for (int e = tid; e < nt * {bv}; e += 256) Rs[e] = expf(gc - gs[e / {bv}]) * Us[e];
            __syncthreads();
            // n × bv outputs, (x, c) and (x + n/2, c) a thread pair per pass
            for (int e = tid; e < {n} * {bv}; e += 256) {{
                const int x = e / {bv}, c = e % {bv};
                float a = 0.f;
                for (int s_ = 0; s_ < nt; ++s_) a += Ks[s_ * {ld} + x] * Rs[s_ * {bv} + c];
                S0[e] = expf(gc) * S0[e] + a;
            }}
        }}
    }}
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
