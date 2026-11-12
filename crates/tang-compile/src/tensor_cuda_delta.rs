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

    /// The CUDA kernel of a [`ScanKind::Sort`] node: one block counting
    /// positions per expert in shared memory, an exclusive scan of the
    /// counts, then each position takes the next slot of its expert
    /// (atomics: the order within an expert varies, which no consumer
    /// sees). `(source, threads)`.
    pub(super) fn cuda_sort_kernel(
        &self,
        root: usize,
        experts: usize,
        reads: &[usize],
    ) -> (String, usize) {
        let sc = self.scan_at(TensorId(root as u32)).unwrap();
        let np = fixed(self.shape(sc.inputs[0]))[0];
        let s = self.storage(sc.inputs[0]).n();
        let q = format!("q{}", reads.iter().position(|&r| r == s).unwrap());
        let params: Vec<String> = (0..reads.len())
            .map(|i| format!("const float* __restrict__ q{i}"))
            .collect();
        let src = format!(
            r#"// Auto-generated by tang-compile: expert sort (counting sort)
extern "C" __global__ void __launch_bounds__(1024) main0({params}, float* __restrict__ out, unsigned count) {{
    __shared__ int cnt[{experts}];
    const int tid = threadIdx.x;
    for (int e = tid; e < {experts}; e += 1024) cnt[e] = 0;
    __syncthreads();
    for (int p = tid; p < {np}; p += 1024) atomicAdd(&cnt[min(max((int)rintf({q}[p]), 0), {experts} - 1)], 1);
    __syncthreads();
    if (tid == 0) {{
        int run = 0;
        for (int e = 0; e < {experts}; ++e) {{ const int c = cnt[e]; cnt[e] = run; run += c; }}
    }}
    __syncthreads();
    for (int p = tid; p < {np}; p += 1024) {{
        const int r = atomicAdd(&cnt[min(max((int)rintf({q}[p]), 0), {experts} - 1)], 1);
        out[p] = (float)r;
        out[{np} + r] = (float)p;
    }}
    (void)count;
}}
"#,
            params = params.join(", "),
        );
        (src, 1024)
    }
}
