//! CUDA lowering of the [`IntMma`] schedule: a contraction
//! `out[p…, r] = Σ_k W[idx(p), r, k] · X[p…, k]` on int8 tensor cores
//! (`mma.sync.m16n8k32.s8`), `W` dequantized in registers from GGUF Q4_K /
//! Q5_K, `X` read as [`Format::Q8x32`] codes; and the writer of a Q8x32
//! buffer (any kernel whose node is stored Q8x32 quantizes on store).
//!
//! The kernel is generated from the comp's views: the point dims before the
//! last are flattened into "positions" `p`, the last point dim is the
//! weight row `r`, the one reduce dim is `k`. A block of 256 threads is
//! `8 / ksplit` 16-row tiles × `ksplit` warps splitting `k`, at one
//! position. Positions are cut into tiles of 8 (the MMA's N); a run of
//! consecutive positions in a tile with the same gather index shares the A
//! fragments and is done by the block at the run's first position (the
//! others exit). Sorting the positions by the index — ordinary IR comps —
//! makes the runs as long as the routing allows.
//!
//! Per 32-column step: one MMA of the 4-bit (5-bit) codes against the int8
//! activations, then `acc += dₓ·(d·scⱼ)·dot − (dmin·mⱼ)·(dₓ·Σc)` with the
//! activation's `dₓ` and `dₓ·Σc` stored beside its codes, so the GGUF mins
//! need no second MMA.

use std::fmt::Write;

use super::{Format, IntMma, Op, Stage, TensorGraph, TensorId, View};

/// Byte layout of a GGUF k-quant super-block of 256.
struct Blk {
    bytes: usize,
    /// Elements a block (256 for the k-quants, 32 for Q8_0).
    per: usize,
    /// Offset of the 4-bit codes.
    qs: usize,
    five: bool,
}

fn blk(f: Format) -> Option<Blk> {
    match f {
        Format::Q8_0 => Some(Blk {
            bytes: 34,
            per: 32,
            qs: 2,
            five: false,
        }),
        Format::Q4K => Some(Blk {
            bytes: 144,
            per: 256,
            qs: 16,
            five: false,
        }),
        Format::Q5K => Some(Blk {
            bytes: 176,
            per: 256,
            qs: 48,
            five: true,
        }),
        _ => None,
    }
}

/// DeepSeek's weight formats ([`Format::Fp8B32`], [`Format::Mxfp4`]): no
/// GGUF block, the Gemv stage only ([`gemv_kernel_ds`]).
fn ds_format(f: Format) -> bool {
    matches!(f, Format::Fp8B32 | Format::Mxfp4)
}

/// `Σ_d stride_d · (form_d(coords) + gather_d · idx)` as C, over source
/// `shape`, with iteration coordinates named `names` (empty name: that
/// coordinate is 0 here) and the gather index expression `idx`.
fn offset_expr(view: &View, shape: &[usize], names: &[String], idx: Option<&str>) -> String {
    let mut stride = 1i64;
    let mut terms = Vec::new();
    for d in (0..shape.len()).rev() {
        let f = &view.forms[d];
        let mut t = vec![format!("{}", f.offset)];
        for (j, &c) in f.coefs.iter().enumerate() {
            if c != 0 && !names[j].is_empty() {
                t.push(format!("{c} * (long long)({})", names[j]));
            }
        }
        if let (Some(g), Some(i)) = (&view.gather, idx) {
            if g.coefs[d] != 0 {
                t.push(format!("{} * (long long)({i})", g.coefs[d]));
            }
        }
        terms.push(format!("{stride}ll * ({})", t.join(" + ")));
        stride *= shape[d] as i64;
    }
    terms.join(" + ")
}

const HELPERS: &str = r#"
__device__ __forceinline__ float tang_h2f_bits(unsigned short h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ void tang_mma_s8(int* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(0), "r"(0), "r"(0), "r"(0));
}
"#;

/// Q8_0 unit loader: four 34-byte blocks (136 bytes, 8-aligned) as 17
/// aligned `uint2` loads, then each block's f16 scale and 8 code words.
pub(super) const Q8_0_UNIT: &str = r#"
__device__ __forceinline__ void tang_q8_0_unit(const unsigned char* p, float* d, unsigned (*q)[8]) {
    unsigned w[34];
    #pragma unroll
    for (int i = 0; i < 17; i++) { const uint2 v = __ldg((const uint2*)p + i); w[2 * i] = v.x; w[2 * i + 1] = v.y; }
    d[0] = tang_h2f_bits((unsigned short)(w[0] & 0xFFFFu));
    d[1] = tang_h2f_bits((unsigned short)(w[8] >> 16));
    d[2] = tang_h2f_bits((unsigned short)(w[17] & 0xFFFFu));
    d[3] = tang_h2f_bits((unsigned short)(w[25] >> 16));
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q[0][i] = __funnelshift_r(w[i], w[i + 1], 16);
        q[1][i] = w[9 + i];
        q[2][i] = __funnelshift_r(w[17 + i], w[18 + i], 16);
        q[3][i] = w[26 + i];
    }
}
"#;

/// The Q8x32 store of a one-thread-per-point kernel: 32 consecutive points
/// are one warp (the caller checks the point count is a multiple of 32).
pub(super) const Q8X32_STORE: &str = r#"
__device__ __forceinline__ void tang_q8x32_store(float* out, unsigned g, float v, unsigned n) {
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    ((signed char*)out)[g] = (signed char)q;
    int s = q;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((g & 31u) == 0u) ((float2*)((unsigned char*)out + n))[g >> 5] = make_float2(d, d * (float)s);
}
"#;

/// Rewrite a transpiled one-thread-per-point kernel to store its node as
/// Q8x32. `None` when the kernel is not of that shape.
pub(super) fn q8x32_writer(src: &str, points: usize, f32_scale: bool) -> Option<String> {
    if !points.is_multiple_of(32) {
        return None;
    }
    let store = "out[g * 1u + 0u] = v[0];";
    if src.matches(store).count() != 1 {
        return None;
    }
    let src = src.replace(store, "tang_q8x32_store(out, g, v[0], params.count);");
    let at = src.find("extern \"C\"")?;
    let helper = if f32_scale {
        // the scale alone, one f32 a 32
        Q8X32_STORE.replace(
            "    int s = q;\n    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);\n    if ((g & 31u) == 0u) ((float2*)((unsigned char*)out + n))[g >> 5] = make_float2(d, d * (float)s);",
            "    if ((g & 31u) == 0u) ((float*)((unsigned char*)out + n))[g >> 5] = d;",
        )
    } else {
        Q8X32_STORE.to_string()
    };
    Some(format!("{}{helper}{}", &src[..at], &src[at..]))
}

impl TensorGraph {
    /// The int8 MMA kernel for `root` (scheduled [`IntMma`]), its reads in
    /// `reads` binding order: `(source, threads a block, blocks)`.
    pub(super) fn cuda_int8_kernel(
        &self,
        root: usize,
        reads: &[usize],
        s: IntMma,
    ) -> Result<(String, usize, usize), String> {
        let c = self.comp_at(root).ok_or("not a comp")?;
        if c.tuple != 0 || c.width() != 1 || c.red.len() != 1 {
            return Err("needs one reduce dim and one output".into());
        }
        if c.monoid.as_ref().map(super::Monoid::kind) != Some(super::MonoidKind::Sum) {
            return Err("needs a sum".into());
        }
        let pts: Vec<usize> = c.points().iter().map(|d| d.n()).collect();
        let np = pts.len();
        if np < 2 {
            return Err("needs position dims and a row dim".into());
        }
        let (rd, kd) = (np - 1, np); // row and reduce coordinates
        let rows = pts[rd];
        let k = c.red[0].n();
        if !rows.is_multiple_of(16) || !k.is_multiple_of(256) {
            return Err(format!(
                "rows {rows} must be a multiple of 16, k {k} of 256"
            ));
        }
        let storage = |t: TensorId| self.nodes[t.n()].precision.storage;
        let is_w = |f: Format| blk(f).is_some() || ds_format(f);
        let wi = (0..c.args.len())
            .find(|&a| is_w(storage(c.args[a].src)))
            .ok_or("no Q4K / Q5K / Q8_0 / Fp8B32 / Mxfp4 operand")?;
        let xi = (0..c.args.len())
            .find(|&a| storage(c.args[a].src) == Format::Q8x32)
            .ok_or("no Q8x32 operand")?;
        if !matches!(self.nodes[c.args[wi].src.n()].op, Op::Input(_)) {
            return Err("the block-format weight must be a graph input".into());
        }
        let wfmt = storage(c.args[wi].src);
        if ds_format(wfmt) && s.stage != Stage::Gemv {
            return Err(format!("{wfmt:?} weights run only on the Gemv stage"));
        }
        // DS formats have no GGUF block; their kernel reads `wfmt`
        let b = blk(wfmt).unwrap_or(Blk {
            bytes: 0,
            per: 32,
            qs: 0,
            five: false,
        });
        // the body must be the product of the two operands
        let nv = c.args.len();
        let probe = |w: f64, x: f64| {
            let mut v = vec![0.0f64; nv];
            v[wi] = w;
            v[xi] = x;
            c.body.eval::<f64>(c.body_outs[0], &v)
        };
        if (probe(3.0, 2.0) - 6.0).abs() > 1e-12 || (probe(-1.5, 5.0) + 7.5).abs() > 1e-12 {
            return Err("body is not W · X".into());
        }
        let shape =
            |t: TensorId| -> Vec<usize> { self.nodes[t.n()].shape.iter().map(|d| d.n()).collect() };
        let (wa, xa) = (&c.args[wi], &c.args[xi]);
        let (ws, xs) = (shape(wa.src), shape(xa.src));
        let unit = |f: &super::Form, d: usize| f.as_unit() == Some(d);
        let free_of = |f: &super::Form, d: usize| f.coefs[d] == 0;
        // `d` with coefficient 1 plus position terms (not r or k)
        let on = |f: &super::Form, d: usize| {
            f.coefs[d] == 1 && (rd..=kd).all(|j| j == d || f.coefs[j] == 0)
        };
        // W: [.., rows, k] with k a unit form, r one plus position terms (a
        // block-diagonal weight's group picks its rows), nothing else on them
        let wl = ws.len();
        if wl < 2
            || !unit(&wa.view.forms[wl - 1], kd)
            || !on(&wa.view.forms[wl - 2], rd)
            || wa.view.forms[..wl - 2]
                .iter()
                .any(|f| !free_of(f, kd) || !free_of(f, rd))
            || ws[wl - 1] != k
            || ws[wl - 2] < rows
            || wa.view.guard
            || xa.view.guard
        {
            return Err("W must be read [.., r, k]".into());
        }
        if let Some(g) = &wa.view.gather {
            if g.coefs[wl - 1] != 0 || g.coefs[wl - 2] != 0 {
                return Err("W's gather must index its leading dims".into());
            }
        }
        // X: [.., k], k one plus position terms that keep a slice on a
        // 32-boundary (a group's slice of a wider row), no form on r
        let xl = xs.len();
        let xf = &xa.view.forms[xl - 1];
        if !on(xf, kd)
            || xf.offset % 32 != 0
            || xf
                .coefs
                .iter()
                .enumerate()
                .any(|(j, &q)| j != kd && q % 32 != 0)
            || xa.view.forms[..xl - 1].iter().any(|f| !free_of(f, kd))
            || xa.view.forms.iter().any(|f| !free_of(f, rd))
            || xs[xl - 1] % 32 != 0
        {
            return Err("X must be read [.., k] independent of r".into());
        }
        if let Some(g) = &xa.view.gather {
            if g.coefs[xl - 1] != 0 {
                return Err("X's gather must index its leading dims".into());
            }
        }
        let param = |a: usize| -> Result<String, String> {
            // a reshape reads its source's buffer
            let mut n = c.args[a].src.n();
            while let Op::Reshape(src) = self.nodes[n].op {
                n = src.n();
            }
            reads
                .iter()
                .position(|&r| r == n)
                .map(|i| format!("q{i}"))
                .ok_or(format!("arg {a} not bound"))
        };
        // positions: the point dims before r, row-major
        let npos: usize = pts[..rd].iter().product();
        let mut names: Vec<String> = (0..=kd).map(|_| String::new()).collect();
        let mut decode = String::new();
        let mut stride = 1usize;
        for d in (0..rd).rev() {
            writeln!(
                decode,
                "    const long long c{d}_P = ((long long)(P) / {stride}) % {};",
                pts[d]
            )
            .unwrap();
            names[d] = format!("c{d}_P");
            stride *= pts[d];
        }
        // index reads at position P (a macro-like lambda over P)
        let index_at = |a: usize| -> Result<String, String> {
            let ia = &c.args[a];
            if ia.view.is_gather() {
                return Err("an index read must be affine".into());
            }
            if ia
                .view
                .forms
                .iter()
                .any(|f| !free_of(f, kd) || !free_of(f, rd))
            {
                return Err("an index must depend on positions only".into());
            }
            let off = offset_expr(&ia.view, &shape(ia.src), &names, None);
            Ok(format!("(long long)rintf({}[{off}])", param(a)?))
        };
        let wkey = match &wa.view.gather {
            Some(g) => index_at(g.arg)?,
            None => "0ll".into(),
        };
        let xkey = match &xa.view.gather {
            Some(g) => Some(index_at(g.arg)?),
            None => None,
        };
        // W slice base in elements (r = k = 0) and X row base in elements (k = 0)
        let w_off = offset_expr(&wa.view, &ws, &names, Some("KEY"));
        let x_off = offset_expr(&xa.view, &xs, &names, xkey.as_ref().map(|_| "XKEY"));
        let x_total: usize = xs.iter().product();
        // X's writer inlined (its node marked inline): only the GEMV, which
        // recomputes and quantizes a position's row in shared memory
        let xn = xa.src.n();
        let fused = if self.nodes[xn].inline {
            if !unit(xf, kd) {
                return Err("an inlined Q8x32 producer needs X read [.., k]".into());
            }
            if s.stage != Stage::Gemv {
                return Err("an inlined Q8x32 producer needs the Gemv stage".into());
            }
            Some(self.gemv_prologue(xn, xa, &names, xkey.is_some(), reads, k)?)
        } else {
            None
        };
        if s.stage == Stage::Gemv {
            if ds_format(wfmt) && s.pos_tile > 1 {
                return Err("pos_tile > 1 is not lowered for DS formats".into());
            }
            if s.pos_tile > 1 && wa.view.gather.is_some() {
                return Err("pos_tile > 1 needs an ungathered weight".into());
            }
            let wp = param(wi)?;
            let xp = if fused.is_some() {
                String::new()
            } else {
                param(xi)?
            };
            return Ok(gemv_kernel(&GemvShape {
                reads: reads.len(),
                npos,
                rows,
                k,
                b,
                decode: &decode,
                wkey: &wkey,
                xkey: xkey.as_deref(),
                w_off: &w_off,
                x_off: &x_off,
                x_total,
                wp: &wp,
                xp: &xp,
                fused: fused.as_deref(),
                ptile: (s.pos_tile.max(1) as usize).min(npos),
                fmt: wfmt,
                w_total: ws.iter().product(),
            }));
        }
        let ksplit = s.ksplit as usize;
        if b.per == 32 {
            let nb = k / 32;
            if ![1, 2, 4, 8].contains(&ksplit) || nb % ksplit != 0 {
                return Err(format!("ksplit {ksplit} must divide 8 and k / 32"));
            }
            let tiles_per_block = 8 / ksplit;
            let row_blocks = rows.div_ceil(16 * tiles_per_block);
            let (wp, xp) = (param(wi)?, param(xi)?);
            let params: Vec<String> = (0..reads.len())
                .map(|i| format!("const float* __restrict__ q{i}"))
                .collect();
            let reduce = if ksplit > 1 {
                if !rows.is_multiple_of(16 * tiles_per_block) {
                    return Err("ksplit > 1 needs rows a multiple of a block's rows".into());
                }
                format!(
                    "    __shared__ float4 red_s[8][32];\n    red_s[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    __syncthreads();\n    if (ks != 0) return;\n    for (int j = 1; j < {ksplit}; j++) {{ const float4 v = red_s[warp + j][lane]; acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w; }}"
                )
            } else {
                String::new()
            };
            let nt = (s.pos_tile.max(1) as usize)
                .div_ceil(8)
                .min(npos.div_ceil(8));
            if wa.view.gather.is_none() && xkey.is_none() && (k / 128) % ksplit == 0 {
                // dense: a block takes `nt` position tiles, the A fragments loaded once
                let groups = npos.div_ceil(8 * nt);
                let mut src = String::new();
                src.push_str(HELPERS);
                write!(
                    src,
                    r#"__device__ __forceinline__ unsigned tang_ld32h(const unsigned char* p) {{
    const unsigned short* h = (const unsigned short*)p;
    return (unsigned)__ldg(h) | ((unsigned)__ldg(h + 1) << 16);
}}
__device__ __forceinline__ unsigned tang_s32h(const unsigned char* p) {{
    const unsigned short* h = (const unsigned short*)p;
    return (unsigned)h[0] | ((unsigned)h[1] << 16);
}}
extern "C" __global__ void __launch_bounds__(256) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    const int grp = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int ks = warp % {ksplit}, r0 = (rb * {tiles_per_block} + warp / {ksplit}) * 16;
    if (grp >= {groups} || r0 >= {rows}) return;
    const int pos0 = grp * {pg};
    const unsigned char* wbase;
    {{ const long long P = pos0;
{decode}      const long long KEY = {wkey}; (void)KEY;
      wbase = (const unsigned char*)({wp}) + ({w_off}) / 32 * 34; }}
    const unsigned char* wa = wbase + (size_t)(r0 + gq) * {row_bytes};
    const unsigned char* wb = wbase + (size_t)(r0 + gq + 8) * {row_bytes};
    auto xrow = [&](int j) -> long long {{
        const long long P = min(pos0 + j, {npos} - 1);
{decode}        return ({x_off});
    }};
    const signed char* xcodes = (const signed char*)({xp});
    const float2* xsc = (const float2*)((const unsigned char*)({xp}) + {x_total});
    const signed char* xbp[{nt}]; long long xa_[{nt}], xb_[{nt}];
    #pragma unroll
    for (int t = 0; t < {nt}; t++) {{
        xbp[t] = xcodes + xrow(8 * t + gq);
        xa_[t] = xrow(8 * t + 2 * tig) / 32; xb_[t] = xrow(8 * t + 2 * tig + 1) / 32;
    }}
    float acc[{nt}][4];
    #pragma unroll
    for (int t = 0; t < {nt}; t++) {{ acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f; }}
    // a warp's 16 rows × one unit (four 34-byte blocks, 136 bytes, 8-aligned)
    // staged through shared memory by coalesced 8-byte loads
    __shared__ __align__(8) unsigned char tang_ws[8][16 * 136];
    unsigned char* ws = tang_ws[warp];
    const unsigned char* wrow0 = wbase + (size_t)r0 * {row_bytes};
    for (int u = ks * {nu_per}; u < (ks + 1) * {nu_per}; u++) {{
        __syncwarp();
        #pragma unroll
        for (int i = lane; i < 272; i += 32) {{
            const int rr = i / 17, ww = i % 17;
            ((uint2*)(ws + rr * 136))[ww] = __ldg((const uint2*)(wrow0 + (size_t)rr * {row_bytes} + (size_t)u * 136) + ww);
        }}
        __syncwarp();
        #pragma unroll
        for (int b = 0; b < 4; b++) {{
            const int blk = 4 * u + b;
            const unsigned char* pa = ws + gq * 136 + b * 34;
            const unsigned char* pb = ws + (gq + 8) * 136 + b * 34;
            const float da = tang_h2f_bits(*(const unsigned short*)pa), db = tang_h2f_bits(*(const unsigned short*)pb);
            unsigned a[4];
            a[0] = tang_s32h(pa + 2 + tig * 4); a[2] = tang_s32h(pa + 18 + tig * 4);
            a[1] = tang_s32h(pb + 2 + tig * 4); a[3] = tang_s32h(pb + 18 + tig * 4);
            #pragma unroll
            for (int t = 0; t < {nt}; t++) {{
                const signed char* xs_ = xbp[t] + blk * 32;
                const unsigned b0 = __ldg((const unsigned*)(xs_ + tig * 4)), b1 = __ldg((const unsigned*)(xs_ + 16 + tig * 4));
                int d[4];
                tang_mma_s8(d, a, b0, b1);
                const float2 x0 = __ldg(xsc + xa_[t] + blk), x1 = __ldg(xsc + xb_[t] + blk);
                acc[t][0] = fmaf(da * x0.x, (float)d[0], acc[t][0]);
                acc[t][1] = fmaf(da * x1.x, (float)d[1], acc[t][1]);
                acc[t][2] = fmaf(db * x0.x, (float)d[2], acc[t][2]);
                acc[t][3] = fmaf(db * x1.x, (float)d[3], acc[t][3]);
            }}
        }}
    }}
{reduce_nt}
    #pragma unroll
    for (int t = 0; t < {nt}; t++) {{
        const int pa_ = pos0 + 8 * t + 2 * tig;
        const long long o0 = (long long)pa_ * {rows} + r0 + gq;
        if (pa_ < {npos}) {{ out[o0] = acc[t][0]; out[o0 + 8] = acc[t][2]; }}
        if (pa_ + 1 < {npos}) {{ out[o0 + {rows}] = acc[t][1]; out[o0 + {rows} + 8] = acc[t][3]; }}
    }}
}}
"#,
                    params = params.join(", "),
                    row_bytes = k / 32 * 34,
                    pg = 8 * nt,
                    nu_per = k / 128 / ksplit,
                    reduce_nt = if ksplit > 1 {
                        format!("    __shared__ float4 red_s[8][32];\n    #pragma unroll\n    for (int t = 0; t < {nt}; t++) {{\n        __syncthreads();\n        red_s[warp][lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);\n        __syncthreads();\n        if (ks == 0) for (int j = 1; j < {ksplit}; j++) {{ const float4 v = red_s[warp + j][lane]; acc[t][0] += v.x; acc[t][1] += v.y; acc[t][2] += v.z; acc[t][3] += v.w; }}\n    }}\n    if (ks != 0) return;")
                    } else {
                        String::new()
                    },
                )
                .unwrap();
                return Ok((src, 256, groups * row_blocks));
            }
            let mut src = String::new();
            src.push_str(HELPERS);
            write!(
                src,
                r#"__device__ __forceinline__ unsigned tang_ld32h(const unsigned char* p) {{
    const unsigned short* h = (const unsigned short*)p;
    return (unsigned)__ldg(h) | ((unsigned)__ldg(h + 1) << 16);
}}
extern "C" __global__ void __launch_bounds__(256) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    const int pos = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int ks = warp % {ksplit}, r0 = (rb * {tiles_per_block} + warp / {ksplit}) * 16;
    if (pos >= {npos}) return;
    long long key0, woff0;
    {{ const long long P = pos;
{decode}      key0 = {wkey}; const long long KEY = key0; woff0 = ({w_off}); }}
    if (pos % 8 != 0) {{
        const long long P = pos - 1;
{decode}        {{ const long long KEY = {wkey}; if (({w_off}) == woff0) return; }}
    }}
    int m = 1;
    while (m < 8 - pos % 8 && pos + m < {npos}) {{
        const long long P = pos + m;
{decode}        {{ const long long KEY = {wkey}; if (({w_off}) != woff0) break; }}
        m++;
    }}
    if (r0 >= {rows}) return;
    const long long KEY = key0;
    const unsigned char* wbase;
    {{ const long long P = pos;
{decode}      wbase = (const unsigned char*)({wp}) + ({w_off}) / 32 * 34; }}
    const unsigned char* wa = wbase + (size_t)min(r0 + gq, {rows} - 1) * {row_bytes};
    const unsigned char* wb = wbase + (size_t)min(r0 + gq + 8, {rows} - 1) * {row_bytes};
    auto xrow = [&](int j) -> long long {{
        const long long P = pos + (j < m ? j : 0);
{decode}{xkey_decl}        return ({x_off});
    }};
    const signed char* xcodes = (const signed char*)({xp});
    const float2* xsc = (const float2*)((const unsigned char*)({xp}) + {x_total});
    const signed char* xb = xcodes + xrow(gq);
    const long long xa_ = xrow(2 * tig) / 32, xb_ = xrow(2 * tig + 1) / 32;
    float acc[4] = {{0.f, 0.f, 0.f, 0.f}};
    #pragma unroll 4
    for (int blk = ks * {nb_per}; blk < (ks + 1) * {nb_per}; blk++) {{
        const unsigned char* pa = wa + blk * 34;
        const unsigned char* pb = wb + blk * 34;
        const float da = tang_h2f_bits(__ldg((const unsigned short*)pa)), db = tang_h2f_bits(__ldg((const unsigned short*)pb));
        unsigned a[4];
        a[0] = tang_ld32h(pa + 2 + tig * 4); a[2] = tang_ld32h(pa + 18 + tig * 4);
        a[1] = tang_ld32h(pb + 2 + tig * 4); a[3] = tang_ld32h(pb + 18 + tig * 4);
        const signed char* xs_ = xb + blk * 32;
        const unsigned b0 = __ldg((const unsigned*)(xs_ + tig * 4)), b1 = __ldg((const unsigned*)(xs_ + 16 + tig * 4));
        int d[4];
        tang_mma_s8(d, a, b0, b1);
        const float2 x0 = __ldg(xsc + xa_ + blk), x1 = __ldg(xsc + xb_ + blk);
        acc[0] = fmaf(da * x0.x, (float)d[0], acc[0]);
        acc[1] = fmaf(da * x1.x, (float)d[1], acc[1]);
        acc[2] = fmaf(db * x0.x, (float)d[2], acc[2]);
        acc[3] = fmaf(db * x1.x, (float)d[3], acc[3]);
    }}
{reduce}
    const long long o0 = (long long)(pos + 2 * tig) * {rows} + r0 + gq;
    const bool ra = r0 + gq < {rows}, rbv = r0 + gq + 8 < {rows};
    if (2 * tig < m) {{ if (ra) out[o0] = acc[0]; if (rbv) out[o0 + 8] = acc[2]; }}
    if (2 * tig + 1 < m) {{ if (ra) out[o0 + {rows}] = acc[1]; if (rbv) out[o0 + {rows} + 8] = acc[3]; }}
}}
"#,
                params = params.join(", "),
                row_bytes = k / 32 * 34,
                nb_per = nb / ksplit,
                xkey_decl = match &xkey {
                    Some(e) => format!("        const long long XKEY = {e};\n"),
                    None => String::new(),
                },
            )
            .unwrap();
            return Ok((src, 256, npos * row_blocks));
        }
        if ![1, 2, 4, 8].contains(&ksplit) || (k / 256) % ksplit != 0 {
            return Err(format!("ksplit {ksplit} must divide 8 and k / 256"));
        }
        let tiles_per_block = 8 / ksplit;
        let row_blocks = rows.div_ceil(16 * tiles_per_block);
        let nsb = k / 256;
        let frag = s.stage == Stage::Fragment;
        let (wp, xp) = (param(wi)?, param(xi)?);
        let params: Vec<String> = (0..reads.len())
            .map(|i| format!("const float* __restrict__ q{i}"))
            .collect();

        let mut src = String::new();
        src.push_str(HELPERS);
        writeln!(
            src,
            "extern \"C\" __global__ void __launch_bounds__(256) main0({}, float* __restrict__ out, unsigned count) {{",
            params.join(", ")
        )
        .unwrap();
        let bb = b.bytes;
        let qs = b.qs;
        write!(
            src,
            r#"    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tig = lane & 3;
    const int pos = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int ks = warp % {ksplit}, r0 = (rb * {tiles_per_block} + warp / {ksplit}) * 16;
    if (pos >= {npos}) return;
    // the run this position starts (or exit): consecutive positions reading the same weight
    // slice (the whole offset, not only the gather key: a position dim may move it too)
    long long key0, woff0;
    {{ const long long P = pos;
{decode}      key0 = {wkey}; const long long KEY = key0; woff0 = ({w_off}); }}
    if (pos % 8 != 0) {{
        const long long P = pos - 1;
{decode}        {{ const long long KEY = {wkey}; if (({w_off}) == woff0) return; }}
    }}
    int m = 1;
    while (m < 8 - pos % 8 && pos + m < {npos}) {{
        const long long P = pos + m;
{decode}        {{ const long long KEY = {wkey}; if (({w_off}) != woff0) break; }}
        m++;
    }}
    if (r0 >= {rows}) return;
    const long long KEY = key0;
    const unsigned char* wbase;
    {{ const long long P = pos;
{decode}      wbase = (const unsigned char*)({wp}) + ({w_off}) / 256 * {bb}; }}
    const unsigned char* wa = wbase + (size_t)(r0 + gq) * {row_bytes};
    const unsigned char* wb = wa + (size_t)8 * {row_bytes};
    // this lane's B column (position pos + gq) and its C columns (pos + 2 tig, + 1)
    auto xrow = [&](int j) -> long long {{
        const long long P = pos + (j < m ? j : 0);
{decode}{xkey_decl}        return ({x_off});
    }};
    const signed char* xcodes = (const signed char*)({xp});
    const float2* xsc = (const float2*)((const unsigned char*)({xp}) + {x_total});
    const signed char* xb = xcodes + xrow(gq);
    const long long xa_ = xrow(2 * tig) / 32, xb_ = xrow(2 * tig + 1) / 32;
    float acc[4] = {{0.f, 0.f, 0.f, 0.f}};
    for (int sb = ks * {nsb_per}; sb < (ks + 1) * {nsb_per}; sb++) {{
        const unsigned char* ba = wa + sb * {bb};
        const unsigned char* bq = wb + sb * {bb};
        const unsigned dwa = __ldg((const unsigned*)ba), dwb = __ldg((const unsigned*)bq);
        const float da = tang_h2f_bits((unsigned short)(dwa & 0xFFFFu)), dma = tang_h2f_bits((unsigned short)(dwa >> 16));
        const float db = tang_h2f_bits((unsigned short)(dwb & 0xFFFFu)), dmb = tang_h2f_bits((unsigned short)(dwb >> 16));
        unsigned sca[3], scb[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {{ sca[i] = __ldg((const unsigned*)(ba + 4 + 4 * i)); scb[i] = __ldg((const unsigned*)(bq + 4 + 4 * i)); }}
        // 6-bit scales and mins, one byte lane each: [s0..3], [s4..7], [m0..3], [m4..7]
        const unsigned sa03 = sca[0] & 0x3F3F3F3Fu, ma03 = sca[1] & 0x3F3F3F3Fu;
        const unsigned sa47 = (sca[2] & 0x0F0F0F0Fu) | (((sca[0] >> 6) & 0x03030303u) << 4);
        const unsigned ma47 = ((sca[2] >> 4) & 0x0F0F0F0Fu) | (((sca[1] >> 6) & 0x03030303u) << 4);
        const unsigned sb03 = scb[0] & 0x3F3F3F3Fu, mb03 = scb[1] & 0x3F3F3F3Fu;
        const unsigned sb47 = (scb[2] & 0x0F0F0F0Fu) | (((scb[0] >> 6) & 0x03030303u) << 4);
        const unsigned mb47 = ((scb[2] >> 4) & 0x0F0F0F0Fu) | (((scb[1] >> 6) & 0x03030303u) << 4);
{loads}
        #pragma unroll
        for (int s = 0; s < 8; s++) {{
            const int qp = s >> 1;
            unsigned a[4];
{afrag}
            const signed char* xs_ = xb + sb * 256 + s * 32;
{bfrag}
            int d[4];
            tang_mma_s8(d, a, b0, b1);
            const int sh = 8 * (s & 3);
            const float fsa = da * (float)(((s < 4 ? sa03 : sa47) >> sh) & 0xFFu);
            const float fma_ = dma * (float)(((s < 4 ? ma03 : ma47) >> sh) & 0xFFu);
            const float fsb = db * (float)(((s < 4 ? sb03 : sb47) >> sh) & 0xFFu);
            const float fmb = dmb * (float)(((s < 4 ? mb03 : mb47) >> sh) & 0xFFu);
            const int blk = sb * 8 + s;
            const float2 x0 = __ldg(xsc + xa_ + blk), x1 = __ldg(xsc + xb_ + blk);
            acc[0] = fmaf(fsa * x0.x, (float)d[0], fmaf(-fma_, x0.y, acc[0]));
            acc[1] = fmaf(fsa * x1.x, (float)d[1], fmaf(-fma_, x1.y, acc[1]));
            acc[2] = fmaf(fsb * x0.x, (float)d[2], fmaf(-fmb, x0.y, acc[2]));
            acc[3] = fmaf(fsb * x1.x, (float)d[3], fmaf(-fmb, x1.y, acc[3]));
        }}
    }}
{reduce}
    const long long o0 = (long long)(pos + 2 * tig) * {rows} + r0 + gq;
    if (2 * tig < m) {{ out[o0] = acc[0]; out[o0 + 8] = acc[2]; }}
    if (2 * tig + 1 < m) {{ out[o0 + {rows}] = acc[1]; out[o0 + {rows} + 8] = acc[3]; }}
}}
"#,
            row_bytes = k / 256 * bb,
            nsb_per = nsb / ksplit,
            xkey_decl = match &xkey {
                Some(e) => format!("        const long long XKEY = {e};\n"),
                None => String::new(),
            },
            loads = if frag {
                format!(
                    "        uint2 qa[4], qb[4];\n        #pragma unroll\n        for (int j = 0; j < 4; j++) {{ qa[j] = __ldg((const uint2*)(ba + {qs} + j * 32 + tig * 8)); qb[j] = __ldg((const uint2*)(bq + {qs} + j * 32 + tig * 8)); }}\n{}",
                    if b.five {
                        "        const uint2 ha = __ldg((const uint2*)(ba + 16 + tig * 8)), hb = __ldg((const uint2*)(bq + 16 + tig * 8));\n"
                    } else {
                        ""
                    }
                )
            } else {
                format!(
                    "        unsigned qa[4][2], qb[4][2];\n        #pragma unroll\n        for (int j = 0; j < 4; j++) {{ qa[j][0] = __ldg((const unsigned*)(ba + {qs} + j * 32 + tig * 4)); qa[j][1] = __ldg((const unsigned*)(ba + {qs} + j * 32 + 16 + tig * 4)); qb[j][0] = __ldg((const unsigned*)(bq + {qs} + j * 32 + tig * 4)); qb[j][1] = __ldg((const unsigned*)(bq + {qs} + j * 32 + 16 + tig * 4)); }}\n{}",
                    if b.five {
                        "        const unsigned ha[2] = {__ldg((const unsigned*)(ba + 16 + tig * 4)), __ldg((const unsigned*)(ba + 32 + tig * 4))};\n        const unsigned hb[2] = {__ldg((const unsigned*)(bq + 16 + tig * 4)), __ldg((const unsigned*)(bq + 32 + tig * 4))};\n"
                    } else {
                        ""
                    }
                )
            },
            afrag = {
                // register c: row (c & 1 ? b : a), k half (c >> 1)
                let word = |r: &str, h: usize| {
                    if frag {
                        format!("q{r}[qp].{}", if h == 0 { "x" } else { "y" })
                    } else {
                        format!("q{r}[qp][{h}]")
                    }
                };
                let hbit = |r: &str, h: usize| {
                    if frag {
                        format!("h{r}.{}", if h == 0 { "x" } else { "y" })
                    } else {
                        format!("h{r}[{h}]")
                    }
                };
                let mut a = String::new();
                for cc in 0..4 {
                    let (r, h) = (if cc & 1 == 1 { "b" } else { "a" }, cc >> 1);
                    let w = word(r, h);
                    write!(
                        a,
                        "            a[{cc}] = (s & 1) ? (({w} >> 4) & 0x0F0F0F0Fu) : ({w} & 0x0F0F0F0Fu);"
                    )
                    .unwrap();
                    if b.five {
                        write!(a, " a[{cc}] |= (({} >> s) & 0x01010101u) << 4;", hbit(r, h))
                            .unwrap();
                    }
                    a.push('\n');
                }
                a
            },
            bfrag = if frag {
                "            const uint2 bv = __ldg((const uint2*)(xs_ + tig * 8));\n            const unsigned b0 = bv.x, b1 = bv.y;"
                    .to_string()
            } else {
                "            const unsigned b0 = __ldg((const unsigned*)(xs_ + tig * 4)), b1 = __ldg((const unsigned*)(xs_ + 16 + tig * 4));"
                    .to_string()
            },
            reduce = if ksplit > 1 {
                format!(
                    "    __shared__ float4 red_s[8][32];\n    red_s[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    __syncthreads();\n    if (ks != 0) return;\n    for (int j = 1; j < {ksplit}; j++) {{ const float4 v = red_s[warp + j][lane]; acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w; }}"
                )
            } else {
                String::new()
            },
        )
        .unwrap();
        // a block where r0 >= rows returned before the barrier: only legal
        // when every warp of a block passes it together
        if ksplit > 1 && !rows.is_multiple_of(16 * tiles_per_block) {
            return Err("ksplit > 1 needs rows a multiple of a block's rows".into());
        }
        let _ = (wp, x_total);
        // stage each warp's 16 weight rows × one super-block through shared memory with
        // coalesced 16-byte loads (lanes otherwise read 4 bytes each, a row apart)
        let src = if std::env::var("TANG_INT8_NOSTAGE").is_ok() {
            src
        } else {
            stage_kquant_tiles(&src, b.bytes, k / 256 * b.bytes)
        };
        Ok((src, 256, npos * row_blocks))
    }

    /// The [`Stage::Gemv`] prologue computing the inlined producer `xn`'s
    /// row at position `P` (the pointwise comp X is, read through `xa`) and
    /// quantizing it Q8x32 into shared `tang_sx` / `tang_ss`, as the
    /// writer's [`Q8X32_STORE`] would. The decoded position names `names`
    /// and `XKEY` (when `has_key`) are in scope where it is pasted.
    fn gemv_prologue(
        &self,
        xn: usize,
        xa: &super::Arg,
        names: &[String],
        has_key: bool,
        reads: &[usize],
        k: usize,
    ) -> Result<String, String> {
        let pc = self
            .comp_at(xn)
            .ok_or("the inlined producer is not a comp")?;
        if !pc.red.is_empty() || pc.tuple != 0 || pc.width() != 1 {
            return Err("the inlined producer must be pointwise with one output".into());
        }
        let pl = pc.points().len();
        // producer coordinates: the leading ones from the position, the last k
        let mut pnames: Vec<String> = Vec::with_capacity(pl);
        let mut s = String::new();
        for (i, f) in xa.view.forms[..pl - 1].iter().enumerate() {
            let mut t = vec![format!("{}", f.offset)];
            for (j, &c) in f.coefs.iter().enumerate() {
                if c != 0 && !names[j].is_empty() {
                    t.push(format!("{c} * (long long)({})", names[j]));
                }
            }
            if let (Some(g), true) = (&xa.view.gather, has_key) {
                if g.coefs[i] != 0 {
                    t.push(format!("{} * XKEY", g.coefs[i]));
                }
            }
            writeln!(s, "      const long long tang_pc{i} = {};", t.join(" + ")).unwrap();
            pnames.push(format!("tang_pc{i}"));
        }
        pnames.push("tang_kk".into());
        let mut vals = Vec::new();
        let mut loads = String::new();
        for (ai, a) in pc.args.iter().enumerate() {
            let n = a.src.n();
            if a.view.is_gather() || a.view.guard {
                return Err("the inlined producer's reads must be affine".into());
            }
            let qi = reads
                .iter()
                .position(|&r| r == n)
                .ok_or("an inlined producer read is not a buffer")?;
            if self.nodes[n].precision.storage != Format::F32 {
                return Err("the inlined producer must read f32".into());
            }
            let shape: Vec<usize> = self.nodes[n].shape.iter().map(|d| d.n()).collect();
            let off = offset_expr(&a.view, &shape, &pnames, None);
            writeln!(loads, "          const float tang_a{ai} = q{qi}[{off}];").unwrap();
            vals.push(format!("tang_a{ai}"));
        }
        let mut body = String::new();
        let outs = super::tile::emit_scalar(&mut body, &pc.body, &pc.body_outs, "tang_e", &|v| {
            vals[v as usize].clone()
        });
        let mut b = String::new();
        for line in body.lines() {
            let line = super::cuda::float_literals(&line.replacen("let ", "const float ", 1));
            writeln!(b, "          {}", line.replace("select(", "tang_select(")).unwrap();
        }
        write!(
            s,
            r#"      // loads of every step first (independent), then the quantization
      float tang_v[{steps}];
      #pragma unroll
      for (int tang_i = 0; tang_i < {steps}; tang_i++) {{
          const int tang_kk = threadIdx.x + 256 * tang_i;
{loads}{b}          tang_v[tang_i] = {out};
      }}
      #pragma unroll
      for (int tang_i = 0; tang_i < {steps}; tang_i++) {{
          const int tang_kk = threadIdx.x + 256 * tang_i;
          const float v = tang_v[tang_i];
          float a = fabsf(v);
          for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
          const float d = a / 127.0f;
          const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
          tang_sx[tang_kk] = (signed char)q;
          int sq = q;
          for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
          if ((tang_kk & 31) == 0) tang_ss[tang_kk >> 5] = make_float2(d, d * (float)sq);
      }}
"#,
            out = outs[0],
            steps = k / 256,
        )
        .unwrap();
        Ok(s)
    }
}

/// What [`gemv_kernel`] needs from the validated contraction.
struct GemvShape<'a> {
    reads: usize,
    npos: usize,
    rows: usize,
    k: usize,
    b: Blk,
    decode: &'a str,
    wkey: &'a str,
    xkey: Option<&'a str>,
    w_off: &'a str,
    x_off: &'a str,
    x_total: usize,
    wp: &'a str,
    xp: &'a str,
    /// the inlined producer's prologue ([`TensorGraph::gemv_prologue`])
    fused: Option<&'a str>,
    /// positions a block (one weight read for all of them)
    ptile: usize,
    /// the weight's storage format
    fmt: Format,
    /// elements of the whole weight tensor (where a DS format's scales start)
    w_total: usize,
}

/// [`Stage::Gemv`]: one dot a `(position, row)` on `dp4a`. A lane takes a
/// half super-block (128 columns: sub-blocks `4h..4h + 4`, qs pairs `2h`
/// and `2h + 1`) at a time; `lanes` lanes share a row, `32 / lanes` rows a
/// warp, 8 warps a block, one position a block. Same scaling as the MMA
/// kernel: `acc += dₓ·(d·scⱼ)·dot − (dmin·mⱼ)·(dₓ·Σc)`.
/// Bytes-in-flight knobs of the GEMV lowerings: warps a block (`TANG_GEMV_WARPS`, default 8;
/// fewer = more, smaller blocks) and a fully unrolled unit loop with a compile-time trip count
/// (`TANG_GEMV_UNROLL=1`: every load of a lane issued before its math). A fused activation
/// prologue quantizes with the whole block, so it keeps 8 warps.
fn gemv_knobs(g: &GemvShape) -> (usize, bool) {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let warps = if g.fused.is_some() {
        8
    } else {
        env("TANG_GEMV_WARPS").unwrap_or(8).clamp(1, 32)
    };
    (warps, env("TANG_GEMV_UNROLL").unwrap_or(0) == 1)
}

/// The GEMV's loop over a lane's units: `u = u0, u0 + lanes, …`.
fn uloop(units: usize, lanes: usize, unroll: bool) -> String {
    if unroll && units.is_multiple_of(lanes) {
        format!(
            "        #pragma unroll\n        for (int ui = 0; ui < {}; ui++) {{\n            const int u = u0 + ui * {lanes};\n",
            units / lanes
        )
    } else {
        format!("        for (int u = u0; u < {units}; u += {lanes}) {{\n")
    }
}

fn gemv_kernel(g: &GemvShape) -> (String, usize, usize) {
    if ds_format(g.fmt) {
        return gemv_kernel_ds(g);
    }
    if g.b.per == 32 && g.ptile > 1 {
        return gemv_kernel_q8_0_tiled(g);
    }
    if g.b.per == 32 {
        return gemv_kernel_q8_0(g);
    }
    let units = g.k / 128;
    // half-super-blocks a lane (more loads in flight a lane), then lanes a row
    let upl: usize = std::env::var("TANG_GEMV_UPL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let lanes = units.div_ceil(upl.max(1)).next_power_of_two().clamp(1, 32);
    let (warps, unroll) = gemv_knobs(g);
    let rpw = 32 / lanes;
    let row_blocks = g.rows.div_ceil(warps * rpw);
    let (bb, qs) = (g.b.bytes, g.b.qs);
    let row_bytes = g.k / 256 * bb;
    let params: Vec<String> = (0..g.reads)
        .map(|i| format!("const float* __restrict__ q{i}"))
        .collect();
    let (five_load, five) = if g.b.five {
        (
            "                const uint4 h4[2] = {__ldg((const uint4*)(bp + 16)), __ldg((const uint4*)(bp + 32))};\n",
            "                    const unsigned qh = w & 4 ? (&h4[1].x)[w & 3] : (&h4[0].x)[w & 3];\n                    lo |= ((qh >> (2 * jp)) & 0x01010101u) << 4;\n                    hi |= ((qh >> (2 * jp + 1)) & 0x01010101u) << 4;\n",
        )
    } else {
        ("", "")
    };
    let xkey_decl = match g.xkey {
        Some(e) => format!("      const long long XKEY = {e};\n"),
        None => String::new(),
    };
    let xsrc = match g.fused {
        // the row in shared memory, quantized by the whole block
        Some(pro) => format!(
            "    __shared__ __align__(16) signed char tang_sx[{k}];\n    __shared__ float2 tang_ss[{kb}];\n    {{ const long long P = pos;\n{decode}{xkey_decl}{pro}    }}\n    __syncthreads();\n    (void)xr;\n    const signed char* xc = tang_sx;\n    const float2* xs = tang_ss;\n",
            k = g.k,
            kb = g.k / 32,
            decode = g.decode,
        ),
        None => format!(
            "    const signed char* xc = (const signed char*)({xp}) + xr;\n    const float2* xs = (const float2*)((const unsigned char*)({xp}) + {x_total}) + xr / 32;\n",
            xp = g.xp,
            x_total = g.x_total,
        ),
    };
    // shared rows are not read through the read-only cache
    let xld = if g.fused.is_some() { "*" } else { "__ldg" };
    let mut src = String::new();
    src.push_str(HELPERS);
    if g.fused.is_some() {
        src.push_str(
            "__device__ __forceinline__ float tang_select(float f, float t, bool c) { return c ? t : f; }\n",
        );
    }
    write!(
        src,
        r#"extern "C" __global__ void __launch_bounds__({threads}) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int r = (rb * {warps} + warp) * {rpw} + lane / {lanes}, u0 = lane % {lanes};
    if (pos >= {npos}) return;
    const bool live = r < {rows};
    long long KEY, xr;
    {{ const long long P = pos;
{decode}      KEY = {wkey};
{xkey_decl}      xr = ({x_off}); }}
    const unsigned char* wbase;
    {{ const long long P = pos;
{decode}      wbase = (const unsigned char*)({wp}) + ({w_off}) / 256 * {bb}; }}
    const unsigned char* wr = wbase + (size_t)(live ? r : 0) * {row_bytes};
{xsrc}    float acc = 0.f;
    if (live) {{
{uloop}            const int sb = u >> 1, h = u & 1;
            const unsigned char* bp = wr + sb * {bb};
            const unsigned dw = __ldg((const unsigned*)bp);
            const float d = tang_h2f_bits((unsigned short)(dw & 0xFFFFu)), dm = tang_h2f_bits((unsigned short)(dw >> 16));
            const unsigned s0 = __ldg((const unsigned*)(bp + 4)), s1 = __ldg((const unsigned*)(bp + 8)), s2 = __ldg((const unsigned*)(bp + 12));
            // 6-bit scales / mins of sub-blocks 4h..4h+3, one byte lane each
            const unsigned sc = h ? ((s2 & 0x0F0F0F0Fu) | (((s0 >> 6) & 0x03030303u) << 4)) : (s0 & 0x3F3F3F3Fu);
            const unsigned mn = h ? (((s2 >> 4) & 0x0F0F0F0Fu) | (((s1 >> 6) & 0x03030303u) << 4)) : (s1 & 0x3F3F3F3Fu);
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {{
                const int jp = 2 * h + jj;
                int dl = 0, dh = 0;
                // 16-byte loads: the pair's 32 code bytes, both sub-blocks' 32 activations
                const uint4* qv = (const uint4*)(bp + {qs} + jp * 32);
                const int4* xlv = (const int4*)(xc + sb * 256 + (2 * jp) * 32);
                const int4* xhv = (const int4*)(xc + sb * 256 + (2 * jp + 1) * 32);
                const uint4 q4[2] = {{__ldg(qv), __ldg(qv + 1)}};
                const int4 xl4[2] = {{{xld}(xlv), {xld}(xlv + 1)}}, xh4[2] = {{{xld}(xhv), {xld}(xhv + 1)}};
{five_load}                #pragma unroll
                for (int w = 0; w < 8; w++) {{
                    const unsigned q = w & 4 ? (&q4[1].x)[w & 3] : (&q4[0].x)[w & 3];
                    unsigned lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
{five}                    const int xl = w & 4 ? (&xl4[1].x)[w & 3] : (&xl4[0].x)[w & 3];
                    const int xh = w & 4 ? (&xh4[1].x)[w & 3] : (&xh4[0].x)[w & 3];
                    dl = __dp4a((int)lo, xl, dl);
                    dh = __dp4a((int)hi, xh, dh);
                }}
                const int il = 2 * jj, ih = 2 * jj + 1; // byte lanes of sub-blocks 2jp, 2jp+1
                const float2 xa = {xld}(xs + sb * 8 + 2 * jp), xb = {xld}(xs + sb * 8 + 2 * jp + 1);
                acc = fmaf(d * (float)((sc >> (8 * il)) & 0xFFu) * xa.x, (float)dl, fmaf(-dm * (float)((mn >> (8 * il)) & 0xFFu), xa.y, acc));
                acc = fmaf(d * (float)((sc >> (8 * ih)) & 0xFFu) * xb.x, (float)dh, fmaf(-dm * (float)((mn >> (8 * ih)) & 0xFFu), xb.y, acc));
            }}
        }}
    }}
    #pragma unroll
    for (int o = {lanes} / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (live && u0 == 0) out[(long long)pos * {rows} + r] = acc;
}}
"#,
        params = params.join(", "),
        threads = warps * 32,
        uloop = uloop(units, lanes, unroll),
        npos = g.npos,
        rows = g.rows,
        decode = g.decode,
        wkey = g.wkey,
        x_off = g.x_off,
        wp = g.wp,
        w_off = g.w_off,
    )
    .unwrap();
    (src, warps * 32, g.npos * row_blocks)
}

/// [`Stage::Gemv`] on GGUF Q8_0 weights: a lane takes 128 columns (four
/// 34-byte blocks) at a time, `dot(w codes, x codes) · d_w · d_x` a block on
/// `dp4a`. Blocks are 2-byte aligned, so codes are loaded as halfwords.
fn gemv_kernel_q8_0(g: &GemvShape) -> (String, usize, usize) {
    let units = g.k / 128;
    // units a lane (`TANG_GEMV_Q8_UPL`, default 1): more independent loads in flight a thread
    let upl: usize = std::env::var("TANG_GEMV_Q8_UPL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let lanes = units.div_ceil(upl.max(1)).next_power_of_two().clamp(1, 32);
    let (warps, unroll) = gemv_knobs(g);
    let rpw = 32 / lanes;
    let row_blocks = g.rows.div_ceil(warps * rpw);
    let row_bytes = g.k / 32 * 34;
    let params: Vec<String> = (0..g.reads)
        .map(|i| format!("const float* __restrict__ q{i}"))
        .collect();
    let xkey_decl = match g.xkey {
        Some(e) => format!("      const long long XKEY = {e};\n"),
        None => String::new(),
    };
    let xsrc = match g.fused {
        Some(pro) => format!(
            "    __shared__ __align__(16) signed char tang_sx[{k}];\n    __shared__ float2 tang_ss[{kb}];\n    {{ const long long P = pos;\n{decode}{xkey_decl}{pro}    }}\n    __syncthreads();\n    (void)xr;\n    const signed char* xc = tang_sx;\n    const float2* xs = tang_ss;\n",
            k = g.k,
            kb = g.k / 32,
            decode = g.decode,
        ),
        None => format!(
            "    const signed char* xc = (const signed char*)({xp}) + xr;\n    const float2* xs = (const float2*)((const unsigned char*)({xp}) + {x_total}) + xr / 32;\n",
            xp = g.xp,
            x_total = g.x_total,
        ),
    };
    let xld = if g.fused.is_some() { "*" } else { "__ldg" };
    let mut src = String::new();
    src.push_str(HELPERS);
    src.push_str(Q8_0_UNIT);
    if g.fused.is_some() {
        src.push_str(
            "__device__ __forceinline__ float tang_select(float f, float t, bool c) { return c ? t : f; }\n",
        );
    }
    write!(
        src,
        r#"extern "C" __global__ void __launch_bounds__({threads}) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int r = (rb * {warps} + warp) * {rpw} + lane / {lanes}, u0 = lane % {lanes};
    if (pos >= {npos}) return;
    const bool live = r < {rows};
    long long KEY, xr;
    {{ const long long P = pos;
{decode}      KEY = {wkey};
{xkey_decl}      xr = ({x_off}); }}
    const unsigned char* wbase;
    {{ const long long P = pos;
{decode}      wbase = (const unsigned char*)({wp}) + ({w_off}) / 32 * 34; }}
    const unsigned char* wr = wbase + (size_t)(live ? r : 0) * {row_bytes};
{xsrc}    float acc = 0.f;
    if (live) {{
{uloop}            float dd[4]; unsigned qq[4][8];
            tang_q8_0_unit(wr + (size_t)u * 136, dd, qq);
            #pragma unroll
            for (int b = 0; b < 4; b++) {{
                const int blk = 4 * u + b;
                const int4* xv = (const int4*)(xc + blk * 32);
                const int4 x4[2] = {{{xld}(xv), {xld}(xv + 1)}};
                int dot = 0;
                #pragma unroll
                for (int w = 0; w < 8; w++) {{
                    const int xw = w & 4 ? (&x4[1].x)[w & 3] : (&x4[0].x)[w & 3];
                    dot = __dp4a((int)qq[b][w], xw, dot);
                }}
                const float2 xd = {xld}(xs + blk);
                acc = fmaf(dd[b] * xd.x, (float)dot, acc);
            }}
        }}
    }}
    #pragma unroll
    for (int o = {lanes} / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (live && u0 == 0) out[(long long)pos * {rows} + r] = acc;
}}
"#,
        params = params.join(", "),
        threads = warps * 32,
        uloop = uloop(units, lanes, unroll),
        npos = g.npos,
        rows = g.rows,
        decode = g.decode,
        wkey = g.wkey,
        x_off = g.x_off,
        wp = g.wp,
        w_off = g.w_off,
    )
    .unwrap();
    (src, warps * 32, g.npos * row_blocks)
}

/// [`gemv_kernel_q8_0`] with `ptile` positions a block: each lane loads a
/// weight block once and dots it with every position's activations (the
/// decode window's rows share the weight traffic).
fn gemv_kernel_q8_0_tiled(g: &GemvShape) -> (String, usize, usize) {
    let pt = g.ptile;
    let units = g.k / 128;
    let lanes = units.next_power_of_two().clamp(1, 32);
    let rpw = 32 / lanes;
    let row_blocks = g.rows.div_ceil(8 * rpw);
    let row_bytes = g.k / 32 * 34;
    let ptiles = g.npos.div_ceil(pt);
    let (k, kb) = (g.k, g.k / 32);
    let params: Vec<String> = (0..g.reads)
        .map(|i| format!("const float* __restrict__ q{i}"))
        .collect();
    let xkey_decl = match g.xkey {
        Some(e) => format!("      const long long XKEY = {e};\n"),
        None => String::new(),
    };
    let xsrc = match g.fused {
        Some(pro) => {
            let pro = pro
                .replace("tang_sx[", &format!("(tang_sx + tang_j * {k})["))
                .replace("tang_ss[", &format!("(tang_ss + tang_j * {kb})["));
            format!(
                "    __shared__ __align__(16) signed char tang_sx[{pk}];\n    __shared__ float2 tang_ss[{pkb}];\n    for (int tang_j = 0; tang_j < np; tang_j++) {{ const long long P = pos0 + tang_j;\n{decode}{xkey_decl}{pro}    }}\n    __syncthreads();\n    const signed char* xcj[{pt}]; const float2* xsj[{pt}];\n    #pragma unroll\n    for (int j = 0; j < {pt}; j++) {{ xcj[j] = tang_sx + j * {k}; xsj[j] = tang_ss + j * {kb}; }}\n",
                pk = pt * k,
                pkb = pt * kb,
                decode = g.decode,
            )
        }
        None => format!(
            "    const signed char* xcj[{pt}]; const float2* xsj[{pt}];\n    #pragma unroll\n    for (int j = 0; j < {pt}; j++) {{\n        long long xr = 0;\n        if (j < np) {{ const long long P = pos0 + j;\n{decode}{xkey_decl}          xr = ({x_off}); }}\n        xcj[j] = (const signed char*)({xp}) + xr;\n        xsj[j] = (const float2*)((const unsigned char*)({xp}) + {x_total}) + xr / 32;\n    }}\n",
            decode = g.decode,
            x_off = g.x_off,
            xp = g.xp,
            x_total = g.x_total,
        ),
    };
    let xld = if g.fused.is_some() { "*" } else { "__ldg" };
    let mut src = String::new();
    src.push_str(HELPERS);
    src.push_str(Q8_0_UNIT);
    if g.fused.is_some() {
        src.push_str(
            "__device__ __forceinline__ float tang_select(float f, float t, bool c) { return c ? t : f; }\n",
        );
    }
    write!(
        src,
        r#"extern "C" __global__ void __launch_bounds__(256) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ptl = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int r = (rb * 8 + warp) * {rpw} + lane / {lanes}, u0 = lane % {lanes};
    if (ptl >= {ptiles}) return;
    const int pos0 = ptl * {pt}, np = min({pt}, {npos} - pos0);
    const bool live = r < {rows};
    const unsigned char* wr;
    {{ const long long P = pos0;
{decode}      const long long KEY = {wkey}; (void)KEY;
      wr = (const unsigned char*)({wp}) + ({w_off}) / 32 * 34 + (size_t)(live ? r : 0) * {row_bytes}; }}
{xsrc}    float acc[{pt}];
    #pragma unroll
    for (int j = 0; j < {pt}; j++) acc[j] = 0.f;
    if (live) {{
        for (int u = u0; u < {units}; u += {lanes}) {{
            float dd[4]; unsigned qq[4][8];
            tang_q8_0_unit(wr + (size_t)u * 136, dd, qq);
            #pragma unroll
            for (int b = 0; b < 4; b++) {{
                const int blk = 4 * u + b;
                const float d = dd[b];
                const unsigned* q = qq[b];
                #pragma unroll
                for (int j = 0; j < {pt}; j++) {{
                    if (j < np) {{
                        const int4* xv = (const int4*)(xcj[j] + blk * 32);
                        const int4 x4[2] = {{{xld}(xv), {xld}(xv + 1)}};
                        int dot = 0;
                        #pragma unroll
                        for (int w = 0; w < 8; w++) dot = __dp4a((int)q[w], w & 4 ? (&x4[1].x)[w & 3] : (&x4[0].x)[w & 3], dot);
                        const float2 xd = {xld}(xsj[j] + blk);
                        acc[j] = fmaf(d * xd.x, (float)dot, acc[j]);
                    }}
                }}
            }}
        }}
    }}
    #pragma unroll
    for (int j = 0; j < {pt}; j++) {{
        float a = acc[j];
        #pragma unroll
        for (int o = {lanes} / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (live && u0 == 0 && j < np) out[(long long)(pos0 + j) * {rows} + r] = a;
    }}
}}
"#,
        params = params.join(", "),
        npos = g.npos,
        rows = g.rows,
        decode = g.decode,
        wkey = g.wkey,
        wp = g.wp,
        w_off = g.w_off,
    )
    .unwrap();
    (src, 256, ptiles * row_blocks)
}

/// Rewrite the k-quant MMA kernel to read its A operands from a per-warp
/// shared tile (16 rows × one `bb`-byte super-block), filled by coalesced
/// 16-byte loads at the top of each super-block step.
fn stage_kquant_tiles(src: &str, bb: usize, row_bytes: usize) -> String {
    let head = "        const unsigned char* ba = wa + sb * ";
    let Some(at) = src.find(head) else {
        return src.to_string();
    };
    let line_end = at + src[at..].find('\n').unwrap() + 1;
    let line2_end = line_end + src[line_end..].find('\n').unwrap() + 1;
    let fill = format!(
        "        __syncwarp();\n        for (int i = lane; i < {n16}; i += 32) {{ const int rr = i / {per}, cc = i % {per}; ((uint4*)(tang_ws[warp] + rr * {bb}))[cc] = __ldg((const uint4*)(wa0 + (size_t)rr * {row_bytes} + (size_t)sb * {bb}) + cc); }}\n        __syncwarp();\n        const unsigned char* ba = tang_ws[warp] + gq * {bb};\n        const unsigned char* bq = tang_ws[warp] + (gq + 8) * {bb};\n",
        n16 = 16 * bb / 16,
        per = bb / 16,
    );
    let mut out = String::with_capacity(src.len() + 1024);
    out.push_str(&src[..at]);
    out.push_str(&fill);
    out.push_str(&src[line2_end..]);
    let out = out
        .replace("__ldg((const unsigned*)ba)", "(*(const unsigned*)ba)")
        .replace("__ldg((const unsigned*)bq)", "(*(const unsigned*)bq)")
        .replace("__ldg((const unsigned*)(ba", "(*(const unsigned*)(ba")
        .replace("__ldg((const unsigned*)(bq", "(*(const unsigned*)(bq")
        .replace("__ldg((const uint2*)(ba", "(*(const uint2*)(ba")
        .replace("__ldg((const uint2*)(bq", "(*(const uint2*)(bq");
    // the tile and the warp's first row
    let decl = "    const unsigned char* wa = wbase + ";
    match out.find(decl) {
        Some(d) => format!(
            "{}    __shared__ __align__(16) unsigned char tang_ws[8][{tile}];\n    const unsigned char* wa0 = wbase + (size_t)r0 * {row_bytes};\n{}",
            &out[..d],
            &out[d..],
            tile = 16 * bb,
        ),
        None => src.to_string(),
    }
}

/// Decoders of DeepSeek's weight bytes. `tang_fp4x4`: four e2m1 codes (the
/// low nibble of each byte of `n`) as four int8 lanes holding twice their
/// value (`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`: exact, so the dot of a block
/// is one integer `dp4a` chain). `tang_e4m3`: an OCP e4m3 byte's value
/// (subnormals exact, without f32 subnormals: no flush to zero).
/// `tang_e8m0`: a bare exponent's power of two.
const DS_HELPERS: &str = r#"
__device__ __forceinline__ int tang_fp4x4(unsigned n) {
    const unsigned x = n & 0x07070707u, t = x | (x >> 4);
    const unsigned s = (t & 0xFFu) | ((t >> 8) & 0xFF00u);
    const unsigned pos = __byte_perm(0x03020100u, 0x0C080604u, s);
    const unsigned neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, s);
    const unsigned m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int)((pos & ~m) | (neg & m));
}
__device__ __forceinline__ float tang_e4m3(unsigned b) {
    const unsigned v = b & 0x7Fu;
    const float f = v >= 8u ? __uint_as_float((v << 20) + (120u << 23)) : (float)v * 0.001953125f;
    return (b & 0x80u) ? -f : f;
}
__device__ __forceinline__ float tang_e8m0(unsigned s) {
    return __uint_as_float(s ? (s << 23) : 0x00400000u);
}
"#;

/// [`Stage::Gemv`] on DeepSeek's weights: [`Format::Mxfp4`] (routed
/// experts) or [`Format::Fp8B32`] (dense), dequantized in registers against
/// Q8x32 activations. A lane takes 128 columns (four 32-column scale blocks)
/// at a time, `lanes` lanes a row, as [`gemv_kernel_q8_0`].
///
/// - Mxfp4: 64 code bytes (four 16-byte loads) and the four e8m0 scales
///   (one 4-byte load) a unit; a block's 32 codes become int8 lanes of twice
///   their value ([`DS_HELPERS`]), re-interleaved to element order with
///   `__byte_perm`, `dot` on `dp4a`, then `acc += ½·2^(s−127)·dₓ·dot`.
/// - Fp8B32: 128 code bytes (eight 16-byte loads) and the scales of the
///   row's 32-row band (one 4-byte load); each code decoded to f32 and
///   multiplied by its int8 activation code, `acc += 2^(s−127)·dₓ·Σ`.
fn gemv_kernel_ds(g: &GemvShape) -> (String, usize, usize) {
    let fp4 = g.fmt == Format::Mxfp4;
    let units = g.k / 128;
    let upl: usize = std::env::var("TANG_GEMV_DS_UPL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let lanes = units.div_ceil(upl.max(1)).next_power_of_two().clamp(1, 32);
    let (warps, unroll) = gemv_knobs(g);
    let rpw = 32 / lanes;
    let row_blocks = g.rows.div_ceil(warps * rpw);
    let k = g.k;
    let params: Vec<String> = (0..g.reads)
        .map(|i| format!("const float* __restrict__ q{i}"))
        .collect();
    let xkey_decl = match g.xkey {
        Some(e) => format!("      const long long XKEY = {e};\n"),
        None => String::new(),
    };
    let xsrc = match g.fused {
        Some(pro) => format!(
            "    __shared__ __align__(16) signed char tang_sx[{k}];\n    __shared__ float2 tang_ss[{kb}];\n    {{ const long long P = pos;\n{decode}{xkey_decl}{pro}    }}\n    __syncthreads();\n    (void)xr;\n    const signed char* xc = tang_sx;\n    const float2* xs = tang_ss;\n",
            kb = k / 32,
            decode = g.decode,
        ),
        None => format!(
            "    const signed char* xc = (const signed char*)({xp}) + xr;\n    const float2* xs = (const float2*)((const unsigned char*)({xp}) + {x_total}) + xr / 32;\n",
            xp = g.xp,
            x_total = g.x_total,
        ),
    };
    let xld = if g.fused.is_some() { "*" } else { "__ldg" };
    // the row's codes and scales, from the slice's element offset `woff`
    let (wdecl, body) = if fp4 {
        (
            format!(
                "    const unsigned char* wc = (const unsigned char*)({wp}) + (woff + (long long)rr * {k}) / 2;\n    const unsigned char* wsc = (const unsigned char*)({wp}) + {codes} + (woff + (long long)rr * {k}) / 32;\n",
                wp = g.wp,
                codes = g.w_total / 2,
            ),
            format!(
                r#"            const uint4* cv = (const uint4*)(wc + (size_t)u * 64);
            const uint4 c4[4] = {{__ldg(cv), __ldg(cv + 1), __ldg(cv + 2), __ldg(cv + 3)}};
            const unsigned sc4 = __ldg((const unsigned*)(wsc + (size_t)u * 4));
            #pragma unroll
            for (int b = 0; b < 4; b++) {{
                const int blk = 4 * u + b;
                const int4* xv = (const int4*)(xc + blk * 32);
                const int4 x4[2] = {{{xld}(xv), {xld}(xv + 1)}};
                int dot = 0;
                #pragma unroll
                for (int w = 0; w < 4; w++) {{
                    const unsigned q = (&c4[b].x)[w];
                    const unsigned lo = (unsigned)tang_fp4x4(q), hi = (unsigned)tang_fp4x4(q >> 4);
                    const int x0 = w & 2 ? (&x4[1].x)[2 * (w & 1)] : (&x4[0].x)[2 * (w & 1)];
                    const int x1 = w & 2 ? (&x4[1].x)[2 * (w & 1) + 1] : (&x4[0].x)[2 * (w & 1) + 1];
                    dot = __dp4a((int)__byte_perm(lo, hi, 0x5140u), x0, dot);
                    dot = __dp4a((int)__byte_perm(lo, hi, 0x7362u), x1, dot);
                }}
                const float2 xd = {xld}(xs + blk);
                acc = fmaf(0.5f * tang_e8m0((sc4 >> (8 * b)) & 0xFFu) * xd.x, (float)dot, acc);
            }}
"#
            ),
        )
    } else {
        (
            format!(
                "    const long long grow = woff / {k} + rr;\n    const unsigned char* wc = (const unsigned char*)({wp}) + grow * {k};\n    const unsigned char* wsc = (const unsigned char*)({wp}) + {codes} + (grow / 32) * {kb};\n",
                wp = g.wp,
                codes = g.w_total,
                kb = k / 32,
            ),
            format!(
                r#"            const uint4* cv = (const uint4*)(wc + (size_t)u * 128);
            uint4 c8[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) c8[i] = __ldg(cv + i);
            const unsigned sc4 = __ldg((const unsigned*)(wsc + (size_t)u * 4));
            #pragma unroll
            for (int b = 0; b < 4; b++) {{
                const int blk = 4 * u + b;
                const int4* xv = (const int4*)(xc + blk * 32);
                const int4 x4[2] = {{{xld}(xv), {xld}(xv + 1)}};
                float dot = 0.f;
                #pragma unroll
                for (int w = 0; w < 8; w++) {{
                    const unsigned q = w & 4 ? (&c8[2 * b + 1].x)[w & 3] : (&c8[2 * b].x)[w & 3];
                    const int xw = w & 4 ? (&x4[1].x)[w & 3] : (&x4[0].x)[w & 3];
                    #pragma unroll
                    for (int e = 0; e < 4; e++)
                        dot = fmaf(tang_e4m3((q >> (8 * e)) & 0xFFu), (float)(signed char)(xw >> (8 * e)), dot);
                }}
                const float2 xd = {xld}(xs + blk);
                acc = fmaf(tang_e8m0((sc4 >> (8 * b)) & 0xFFu) * xd.x, dot, acc);
            }}
"#
            ),
        )
    };
    let mut src = String::new();
    src.push_str(HELPERS);
    src.push_str(DS_HELPERS);
    if g.fused.is_some() {
        src.push_str(
            "__device__ __forceinline__ float tang_select(float f, float t, bool c) { return c ? t : f; }\n",
        );
    }
    write!(
        src,
        r#"extern "C" __global__ void __launch_bounds__({threads}) main0({params}, float* __restrict__ out, unsigned count) {{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = blockIdx.x / {row_blocks}, rb = blockIdx.x % {row_blocks};
    const int r = (rb * {warps} + warp) * {rpw} + lane / {lanes}, u0 = lane % {lanes};
    if (pos >= {npos}) return;
    const bool live = r < {rows};
    const int rr = live ? r : 0;
    long long KEY, xr, woff;
    {{ const long long P = pos;
{decode}      KEY = {wkey};
{xkey_decl}      xr = ({x_off});
      woff = ({w_off}); }}
{wdecl}{xsrc}    float acc = 0.f;
    if (live) {{
{uloop}{body}        }}
    }}
    #pragma unroll
    for (int o = {lanes} / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (live && u0 == 0) out[(long long)pos * {rows} + r] = acc;
}}
"#,
        params = params.join(", "),
        threads = warps * 32,
        uloop = uloop(units, lanes, unroll),
        npos = g.npos,
        rows = g.rows,
        decode = g.decode,
        wkey = g.wkey,
        x_off = g.x_off,
        w_off = g.w_off,
    )
    .unwrap();
    (src, warps * 32, g.npos * row_blocks)
}
