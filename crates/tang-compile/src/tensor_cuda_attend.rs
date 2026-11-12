//! Fused causal attention rows for the CUDA decode lowering.
//!
//! A derived step that appends `k > 1` rows computes attention as written
//! in the prefill graph: the scores `s[t, g, r, j] = scale·q·K_j` over the
//! whole cache capacity, the masked maximum and sum over `j`, then
//! `Σ_j select(j ≤ pos + t, exp(s − m)·V_j / l, 0)`. Lowered as written
//! that materializes `k·heads·cap` scores (half a GB a layer for
//! coder-next at `k = 256`, 32k context) and reads them three times, all
//! over the full capacity. [`TensorGraph::fuse_attention_rows`] recognizes
//! the pattern and replaces the readout by one [`ScanKind::Attend`] node
//! over `(q, K, V, pos)`; [`TensorGraph::cuda_attend_kernel`] runs it as
//! one flash-attention kernel: online softmax, bf16 `mma.sync` m16n8k16
//! with f32 accumulation, each block reading keys only up to its last
//! row's position (read from the device-resident `pos`, so the kernel is
//! graph-capture safe).
//!
//! Blocks are `(kv head, 16·W rows)`; a row is a `(t, r)` pair (the `per`
//! query heads of one kv head share its K and V tiles). Each warp owns 16
//! rows. Per 64-key tile: K (bf16, `[key][dim]`) and Vᵀ (bf16,
//! `[dim][key]`) staged in shared memory, `S = Q·Kᵀ` on fragments, the
//! running max / sum updated, `P` re-packed from the accumulator layout
//! straight into A fragments for `O += P·V`.

use std::fmt::Write;

use super::{fixed, AttendStep, Format, MonoidKind, Op, Scan, ScanKind, TensorGraph, TensorId};

/// Keys a tile.
const BC: usize = 64;

/// The roles a probe assigns to an arg of the attention comps.
#[derive(Clone, Copy)]
enum Role {
    Score,
    Max,
    Sum,
    Value,
    Pos,
    Iota(usize),
}

impl TensorGraph {
    /// The graph with every derived-step attention readout rewritten as a
    /// fused [`ScanKind::Attend`] node (see the module docs); `None` when
    /// there is nothing to fuse. `TANG_FLASH=0` turns it off.
    pub(super) fn fuse_attention_rows(&self, outputs: &[TensorId]) -> Option<TensorGraph> {
        if std::env::var("TANG_FLASH").as_deref() == Ok("0") {
            return None;
        }
        let mut out: Option<TensorGraph> = None;
        for u in 0..self.nodes.len() {
            if outputs.iter().any(|o| o.n() == u) {
                continue;
            }
            let Some((ins, a)) = self.attention_rows(u) else {
                continue;
            };
            let g = out.get_or_insert_with(|| self.clone());
            for &t in &ins[..3] {
                let s = g.storage(t).n();
                g.nodes[s].inline = false;
            }
            g.nodes[u].op = Op::Scan(Scan {
                inputs: ins,
                dim: 0,
                exclusive: false,
                reverse: false,
                kind: ScanKind::Attend(a),
            });
            g.nodes[u].inline = false;
        }
        out
    }

    /// `([q, K, V, pos], step)` when node `u` is the readout of a derived
    /// step's causal attention (the coder graph's `attention()` as
    /// `derive_step` leaves it), checked numerically on the bodies.
    fn attention_rows(&self, u: usize) -> Option<(Vec<TensorId>, AttendStep)> {
        let c = self.comp_at(u)?;
        if c.monoid.as_ref().map(|m| m.kind()) != Some(MonoidKind::Sum)
            || c.red.len() != 1
            || c.tuple != 0
            || c.out.len() != 4
            || c.body_outs.len() != 1
        {
            return None;
        }
        let out = fixed(&c.out);
        let (k, kvh, per, hd) = (out[0], out[1], out[2], out[3]);
        let cap = fixed(&c.red)[0];
        if k * per < 16 || !hd.is_multiple_of(8) || hd > 256 || cap < 2 {
            return None;
        }
        if self.nodes[u].precision.storage != Format::F32 {
            return None;
        }
        let unit = |rank: usize, d: usize| super::Form::unit(rank, d);
        // the readout's args: space (t, g, r, c, j)
        let (mut sc, mut mx, mut sm, mut vv, mut pos) = (None, None, None, None, None);
        let mut roles = Vec::new();
        for a in &c.args {
            let p = a.src.n();
            let f = &a.view.forms;
            let role = match &self.nodes[p].op {
                Op::Iota(d) => Role::Iota(*d),
                _ if fixed(self.shape(a.src)).iter().product::<usize>() == 1 => {
                    pos = Some(a.src);
                    Role::Pos
                }
                Op::Comp(pc) => match pc.monoid.as_ref().map(|m| m.kind()) {
                    Some(MonoidKind::Max) if f[..] == [unit(5, 0), unit(5, 1), unit(5, 2)] => {
                        mx = Some(p);
                        Role::Max
                    }
                    Some(MonoidKind::Sum)
                        if f.len() == 3 && f[..] == [unit(5, 0), unit(5, 1), unit(5, 2)] =>
                    {
                        sm = Some(p);
                        Role::Sum
                    }
                    Some(MonoidKind::Sum)
                        if f[..] == [unit(5, 0), unit(5, 1), unit(5, 2), unit(5, 4)] =>
                    {
                        sc = Some(p);
                        Role::Score
                    }
                    _ => self.value_role(a, kvh, hd, cap, &mut vv)?,
                },
                _ => self.value_role(a, kvh, hd, cap, &mut vv)?,
            };
            if a.view.guard || a.view.gather.is_some() {
                return None;
            }
            roles.push(role);
        }
        let (sc, mx, sm, vv, pos) = (sc?, mx?, sm?, vv?, pos?);
        // the scores: Σ_c q[t, g·per + r, c]·K[j, g, c]·scale, space (t, g, r, j, c)
        let s = self.comp_at(sc)?;
        if s.args.len() != 2 || s.red.len() != 1 || fixed(&s.red)[0] != hd || s.tuple != 0 {
            return None;
        }
        let (q, kk) = (s.args[0].src, s.args[1].src);
        let mut gq = super::Form::zero(5);
        gq.coefs[1] = per as i64;
        gq.coefs[2] = 1;
        if s.args[0].view.forms[..] != [unit(5, 0), gq, unit(5, 4)]
            || s.args[1].view.forms[..] != [unit(5, 3), unit(5, 1), unit(5, 4)]
            || s.args
                .iter()
                .any(|a| a.view.guard || a.view.gather.is_some())
            || fixed(self.shape(q)) != [k, kvh * per, hd]
            || fixed(self.shape(kk)) != [cap, kvh, hd]
        {
            return None;
        }
        let scale: Vec<f64> = s.body.eval_many(&s.body_outs, &[1.0, 1.0]);
        let two: Vec<f64> = s.body.eval_many(&s.body_outs, &[2.0, 3.0]);
        if scale.len() != 1 || (two[0] - 6.0 * scale[0]).abs() > 1e-12 * scale[0].abs() {
            return None;
        }
        let scale = scale[0];
        // the readout, max and sum bodies probed against the causal softmax
        let vis = |t: f64, j: f64, p: f64| j <= t + p;
        let probes = [
            (0.0, 0.0, 0.0),
            (3.0, 2.0, 0.0),
            (2.0, 3.0, 0.0),
            (1.0, 5.0, 4.0),
            (1.0, 6.0, 4.0),
            (0.0, 9.0, 9.0),
            (0.0, 10.0, 9.0),
        ];
        let val = |roles: &[Role], args: &[super::Arg], pt: &[f64], x: [f64; 5]| -> Vec<f64> {
            // x: (s, m, l, v, pos)
            roles
                .iter()
                .zip(args)
                .map(|(r, a)| match r {
                    Role::Score => x[0],
                    Role::Max => x[1],
                    Role::Sum => x[2],
                    Role::Value => x[3],
                    Role::Pos => x[4],
                    Role::Iota(d) => {
                        let f = &a.view.forms[*d];
                        f.offset as f64
                            + f.coefs
                                .iter()
                                .zip(pt)
                                .map(|(c, p)| *c as f64 * p)
                                .sum::<f64>()
                    }
                })
                .collect()
        };
        for &(t, j, p) in &probes {
            let (sv, m, l, v) = (0.7, 0.2, 1.9, -1.3);
            let got = c.body.eval_many(
                &c.body_outs,
                &val(&roles, &c.args, &[t, 1.0, 1.0, 2.0, j], [sv, m, l, v, p]),
            )[0];
            let want = if vis(t, j, p) {
                (sv - m).exp() * v / l
            } else {
                0.0
            };
            if (got - want).abs() > 1e-9 {
                return None;
            }
        }
        // max and sum: space (t, g, r, j)
        for (node, is_max) in [(mx, true), (sm, false)] {
            let mc = self.comp_at(node)?;
            if mc.red.len() != 1 || fixed(&mc.red)[0] != cap || mc.body_outs.len() != 1 {
                return None;
            }
            let mut mroles = Vec::new();
            for a in &mc.args {
                let p = a.src.n();
                let r = match &self.nodes[p].op {
                    Op::Iota(d) => Role::Iota(*d),
                    _ if a.src == pos => Role::Pos,
                    _ if p == sc
                        && a.view.forms[..] == [unit(4, 0), unit(4, 1), unit(4, 2), unit(4, 3)] =>
                    {
                        Role::Score
                    }
                    _ if p == mx
                        && !is_max
                        && a.view.forms[..] == [unit(4, 0), unit(4, 1), unit(4, 2)] =>
                    {
                        Role::Max
                    }
                    _ => return None,
                };
                mroles.push(r);
            }
            for &(t, j, p) in &probes {
                let (sv, m) = (0.7, 0.2);
                let got = mc.body.eval_many(
                    &mc.body_outs,
                    &val(&mroles, &mc.args, &[t, 1.0, 1.0, j], [sv, m, 0.0, 0.0, p]),
                )[0];
                let ok = match (is_max, vis(t, j, p)) {
                    (true, true) => got == sv,
                    (true, false) => got < -1e20,
                    (false, true) => (got - (sv - m).exp()).abs() < 1e-9,
                    (false, false) => got == 0.0,
                };
                if !ok {
                    return None;
                }
            }
        }
        let f32s = [q, kk, vv, pos, TensorId(u as u32)]
            .iter()
            .all(|t| self.nodes[self.storage(*t).n()].precision.storage == Format::F32);
        if !f32s {
            return None;
        }
        Some((
            vec![q, kk, vv, pos],
            AttendStep {
                scale,
                kvh,
                per,
                hd,
            },
        ))
    }

    /// The value read: `V[j, g·hd + c]` of a `[cap, kvh·hd]` tensor.
    fn value_role(
        &self,
        a: &super::Arg,
        kvh: usize,
        hd: usize,
        cap: usize,
        vv: &mut Option<TensorId>,
    ) -> Option<Role> {
        let mut f = super::Form::zero(5);
        f.coefs[1] = hd as i64;
        f.coefs[3] = 1;
        if a.view.forms[..] != [super::Form::unit(5, 4), f]
            || fixed(self.shape(a.src)) != [cap, kvh * hd]
            || vv.is_some()
        {
            return None;
        }
        *vv = Some(a.src);
        Some(Role::Value)
    }

    /// Warps a block of [`ScanKind::Attend`] (16 rows each).
    fn attend_warps(rows: usize) -> usize {
        // enough blocks to fill the GPU at short chunks; 4 warps otherwise
        if rows >= 4096 {
            4
        } else {
            2
        }
    }

    /// The CUDA kernel of a [`ScanKind::Attend`] node: `(source, threads,
    /// blocks)`, parameters the kernel's `reads` then `out`.
    pub(super) fn cuda_attend_kernel(
        &self,
        root: usize,
        reads: &[usize],
    ) -> (String, usize, usize) {
        let sc = self.scan_at(TensorId(root as u32)).unwrap();
        let ScanKind::Attend(a) = &sc.kind else {
            unreachable!()
        };
        let (kvh, per, hd) = (a.kvh, a.per, a.hd);
        let k = fixed(self.shape(sc.inputs[0]))[0];
        let cap = fixed(self.shape(sc.inputs[1]))[0];
        let rows = k * per;
        let w = Self::attend_warps(rows * kvh);
        let br = 16 * w;
        let nrb = rows.div_ceil(br);
        let threads = 32 * w;
        // dims padded to the k-step (16); the pad reads as zero
        let hdp = hd.next_multiple_of(16);
        let (qst, kst, vst) = (hdp + 8, hdp + 8, BC + 8);
        let smem = 2 * (br * qst + BC * kst + hd * vst);
        let arg = |i: usize| {
            let s = self.storage(sc.inputs[i]).n();
            format!("q{}", reads.iter().position(|&r| r == s).unwrap())
        };
        let params: Vec<String> = (0..reads.len())
            .map(|i| format!("const float* __restrict__ q{i}"))
            .collect();
        let (qq, kk, vv, pp) = (arg(0), arg(1), arg(2), arg(3));
        let sl2 = a.scale * std::f64::consts::LOG2_E;
        let mut src = String::new();
        writeln!(src, "#define TANG_SMEM {smem}").unwrap();
        writeln!(
            src,
            "// Auto-generated by tang-compile: fused attention rows (flash, bf16 mma)"
        )
        .unwrap();
        src.push_str(PRELUDE);
        write!(
            src,
            r#"extern "C" __global__ void __launch_bounds__({threads}) main0({params}, float* __restrict__ out, unsigned count) {{
    extern __shared__ float4 tang_dsm[];
    unsigned short* Qs = (unsigned short*)tang_dsm;
    unsigned short* Ks = Qs + {br} * {qst};
    unsigned short* Vt = Ks + {bc} * {kst};
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tig = lane & 3;
    const int g = blockIdx.x / {nrb}, rb = blockIdx.x % {nrb};
    const int pos = (int)rintf({pp}[0]);
    const int r0 = rb * {br};
    // Q rows (t, r) of kv head g as bf16
    for (int i = tid; i < {br} * {hdp}; i += {threads}) {{
        const int rr = i / {hdp}, d = i % {hdp}, R = r0 + rr;
        float x = 0.0f;
        if (R < {rows} && d < {hd}) {{
            const int t = R / {per}, r = R % {per};
            x = {qq}[((long long)t * {nh} + g * {per} + r) * {hd} + d];
        }}
        Qs[rr * {qst} + d] = tang_bf(x);
    }}
    const int Ra = r0 + warp * 16 + gq, Rb = Ra + 8;
    const int lima = Ra < {rows} ? pos + Ra / {per} : -1;
    const int limb = Rb < {rows} ? pos + Rb / {per} : -1;
    const int tlast = min({k} - 1, (r0 + {br} - 1) / {per});
    const int jend = min({cap}, pos + tlast + 1);
    float O[{dt}][4];
    #pragma unroll
    for (int i = 0; i < {dt}; ++i) O[i][0] = O[i][1] = O[i][2] = O[i][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    for (int j0 = 0; j0 < jend; j0 += {bc}) {{
        __syncthreads();
        for (int i = tid; i < {bc} * {hdp}; i += {threads}) {{
            const int jj = i / {hdp}, d = i % {hdp}, j = j0 + jj;
            float kx = 0.0f, vx = 0.0f;
            if (j < jend && d < {hd}) {{
                kx = {kk}[((long long)j * {kvh} + g) * {hd} + d];
                vx = {vv}[((long long)j * {kvh} + g) * {hd} + d];
            }}
            Ks[jj * {kst} + d] = tang_bf(kx);
            if (d < {hd}) Vt[d * {vst} + jj] = tang_bf(vx);
        }}
        __syncthreads();
        float S[{nt}][4];
        #pragma unroll
        for (int n = 0; n < {nt}; ++n) S[n][0] = S[n][1] = S[n][2] = S[n][3] = 0.0f;
        for (int ks = 0; ks < {hdp}; ks += 16) {{
            unsigned af[4];
            const unsigned short* qa = Qs + (warp * 16 + gq) * {qst} + ks + tig * 2;
            af[0] = *(const unsigned*)qa;
            af[1] = *(const unsigned*)(qa + 8 * {qst});
            af[2] = *(const unsigned*)(qa + 8);
            af[3] = *(const unsigned*)(qa + 8 * {qst} + 8);
            #pragma unroll
            for (int n = 0; n < {nt}; ++n) {{
                const unsigned short* kb = Ks + (n * 8 + gq) * {kst} + ks + tig * 2;
                unsigned bf[2];
                bf[0] = *(const unsigned*)kb;
                bf[1] = *(const unsigned*)(kb + 8);
                tang_mma_bf16(S[n], af, bf);
            }}
        }}
        // masked, scaled to log2 units; running max
        float xa = -INFINITY, xb = -INFINITY;
        #pragma unroll
        for (int n = 0; n < {nt}; ++n) {{
            const int j = j0 + n * 8 + tig * 2;
            S[n][0] = j <= lima ? S[n][0] * {sl2:e}f : -INFINITY;
            S[n][1] = j + 1 <= lima ? S[n][1] * {sl2:e}f : -INFINITY;
            S[n][2] = j <= limb ? S[n][2] * {sl2:e}f : -INFINITY;
            S[n][3] = j + 1 <= limb ? S[n][3] * {sl2:e}f : -INFINITY;
            xa = fmaxf(xa, fmaxf(S[n][0], S[n][1]));
            xb = fmaxf(xb, fmaxf(S[n][2], S[n][3]));
        }}
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 1));
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 1));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 2));
        const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
        // rows with nothing visible yet keep zero (no inf − inf)
        const float ca = na == -INFINITY ? 1.0f : exp2f(ma - na);
        const float cb = nb == -INFINITY ? 1.0f : exp2f(mb - nb);
        const float sa = na == -INFINITY ? 0.0f : na, sb = nb == -INFINITY ? 0.0f : nb;
        ma = na; mb = nb;
        float pa = 0.0f, pb = 0.0f;
        #pragma unroll
        for (int n = 0; n < {nt}; ++n) {{
            S[n][0] = exp2f(S[n][0] - sa); S[n][1] = exp2f(S[n][1] - sa);
            S[n][2] = exp2f(S[n][2] - sb); S[n][3] = exp2f(S[n][3] - sb);
            pa += S[n][0] + S[n][1];
            pb += S[n][2] + S[n][3];
        }}
        la = la * ca + pa;
        lb = lb * cb + pb;
        #pragma unroll
        for (int i = 0; i < {dt}; ++i) {{ O[i][0] *= ca; O[i][1] *= ca; O[i][2] *= cb; O[i][3] *= cb; }}
        // O += P·V: P's accumulator pairs are its A fragments
        #pragma unroll
        for (int kk2 = 0; kk2 < {nt} / 2; ++kk2) {{
            unsigned af[4];
            af[0] = tang_pack_bf(S[2 * kk2][0], S[2 * kk2][1]);
            af[1] = tang_pack_bf(S[2 * kk2][2], S[2 * kk2][3]);
            af[2] = tang_pack_bf(S[2 * kk2 + 1][0], S[2 * kk2 + 1][1]);
            af[3] = tang_pack_bf(S[2 * kk2 + 1][2], S[2 * kk2 + 1][3]);
            #pragma unroll
            for (int i = 0; i < {dt}; ++i) {{
                const unsigned short* vb = Vt + (i * 8 + gq) * {vst} + kk2 * 16 + tig * 2;
                unsigned bf[2];
                bf[0] = *(const unsigned*)vb;
                bf[1] = *(const unsigned*)(vb + 8);
                tang_mma_bf16(O[i], af, bf);
            }}
        }}
    }}
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    const float ia = 1.0f / la, ib = 1.0f / lb;
    if (Ra < {rows}) {{
        float* o = out + ((long long)(Ra / {per}) * {kvh} + g) * {per} * {hd} + (long long)(Ra % {per}) * {hd};
        #pragma unroll
        for (int i = 0; i < {dt}; ++i) {{ o[i * 8 + tig * 2] = O[i][0] * ia; o[i * 8 + tig * 2 + 1] = O[i][1] * ia; }}
    }}
    if (Rb < {rows}) {{
        float* o = out + ((long long)(Rb / {per}) * {kvh} + g) * {per} * {hd} + (long long)(Rb % {per}) * {hd};
        #pragma unroll
        for (int i = 0; i < {dt}; ++i) {{ o[i * 8 + tig * 2] = O[i][2] * ib; o[i * 8 + tig * 2 + 1] = O[i][3] * ib; }}
    }}
    (void)count;
}}
"#,
            params = params.join(", "),
            bc = BC,
            nt = BC / 8,
            dt = hd / 8,
            nh = kvh * per,
        )
        .unwrap();
        (src, threads, kvh * nrb)
    }
}

/// bf16 conversion and the m16n8k16 bf16 `mma.sync` (with a shuffle
/// emulation for the CPU harness, `TANG_HOST_EMU`).
const PRELUDE: &str = r#"#ifdef TANG_HOST_EMU
static inline unsigned short tang_bf(float x) {
    unsigned u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
static inline unsigned tang_pack_bf(float lo, float hi) { return (unsigned)tang_bf(lo) | ((unsigned)tang_bf(hi) << 16); }
static inline float tang_bf_lo(unsigned v) { return __uint_as_float(v << 16); }
static inline float tang_bf_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }
static inline void tang_mma_bf16(float* c, const unsigned* a, const unsigned* b) {
    const int l = (int)(threadIdx.x & 31), gq = l >> 2, tg = l & 3;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t = 0; t < 4; t++) {
        const unsigned a0 = __shfl_sync(0xffffffffu, a[0], gq * 4 + t);
        const unsigned a1 = __shfl_sync(0xffffffffu, a[1], gq * 4 + t);
        const unsigned a2 = __shfl_sync(0xffffffffu, a[2], gq * 4 + t);
        const unsigned a3 = __shfl_sync(0xffffffffu, a[3], gq * 4 + t);
        for (int cc = 0; cc < 2; cc++) {
            const unsigned b0 = __shfl_sync(0xffffffffu, b[0], (2 * tg + cc) * 4 + t);
            const unsigned b1 = __shfl_sync(0xffffffffu, b[1], (2 * tg + cc) * 4 + t);
            acc[cc] += tang_bf_lo(a0) * tang_bf_lo(b0) + tang_bf_hi(a0) * tang_bf_hi(b0)
                + tang_bf_lo(a2) * tang_bf_lo(b1) + tang_bf_hi(a2) * tang_bf_hi(b1);
            acc[2 + cc] += tang_bf_lo(a1) * tang_bf_lo(b0) + tang_bf_hi(a1) * tang_bf_hi(b0)
                + tang_bf_lo(a3) * tang_bf_lo(b1) + tang_bf_hi(a3) * tang_bf_hi(b1);
        }
    }
    c[0] += acc[0]; c[1] += acc[1]; c[2] += acc[2]; c[3] += acc[3];
}
#else
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
static __device__ __forceinline__ unsigned short tang_bf(float x) {
    unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x)); return h;
}
static __device__ __forceinline__ unsigned tang_pack_bf(float lo, float hi) {
    unsigned d; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
}
static __device__ __forceinline__ void tang_mma_bf16(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif
"#;
