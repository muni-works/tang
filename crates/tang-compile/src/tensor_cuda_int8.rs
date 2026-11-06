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
    /// Offset of the 4-bit codes.
    qs: usize,
    five: bool,
}

fn blk(f: Format) -> Option<Blk> {
    match f {
        Format::Q4K => Some(Blk {
            bytes: 144,
            qs: 16,
            five: false,
        }),
        Format::Q5K => Some(Blk {
            bytes: 176,
            qs: 48,
            five: true,
        }),
        _ => None,
    }
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
pub(super) fn q8x32_writer(src: &str, points: usize) -> Option<String> {
    if !points.is_multiple_of(32) {
        return None;
    }
    let store = "out[g * 1u + 0u] = v[0];";
    if src.matches(store).count() != 1 {
        return None;
    }
    let src = src.replace(store, "tang_q8x32_store(out, g, v[0], params.count);");
    let at = src.find("extern \"C\"")?;
    Some(format!("{}{Q8X32_STORE}{}", &src[..at], &src[at..]))
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
        let wi = (0..c.args.len())
            .find(|&a| blk(storage(c.args[a].src)).is_some())
            .ok_or("no Q4K / Q5K operand")?;
        let xi = (0..c.args.len())
            .find(|&a| storage(c.args[a].src) == Format::Q8x32)
            .ok_or("no Q8x32 operand")?;
        if !matches!(self.nodes[c.args[wi].src.n()].op, Op::Input(_)) {
            return Err("the block-format weight must be a graph input".into());
        }
        let b = blk(storage(c.args[wi].src)).unwrap();
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
        // W: [.., rows, k] with r and k as unit forms, nothing else on them
        let wl = ws.len();
        if wl < 2
            || !unit(&wa.view.forms[wl - 1], kd)
            || !unit(&wa.view.forms[wl - 2], rd)
            || wa.view.forms[..wl - 2]
                .iter()
                .any(|f| !free_of(f, kd) || !free_of(f, rd))
            || ws[wl - 1] != k
            || ws[wl - 2] != rows
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
        // X: [.., k], k a unit form, no form on r
        let xl = xs.len();
        if !unit(&xa.view.forms[xl - 1], kd)
            || xa.view.forms[..xl - 1].iter().any(|f| !free_of(f, kd))
            || xa.view.forms.iter().any(|f| !free_of(f, rd))
            || xs[xl - 1] != k
        {
            return Err("X must be read [.., k] independent of r".into());
        }
        if let Some(g) = &xa.view.gather {
            if g.coefs[xl - 1] != 0 {
                return Err("X's gather must index its leading dims".into());
            }
        }
        let param = |a: usize| -> Result<String, String> {
            let n = c.args[a].src.n();
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
        let ksplit = s.ksplit as usize;
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
    // the run this position starts (or exit)
    long long key0;
    {{ const long long P = pos;
{decode}      key0 = {wkey}; }}
    if (pos % 8 != 0) {{
        const long long P = pos - 1;
{decode}        if ({wkey} == key0) return;
    }}
    int m = 1;
    while (m < 8 - pos % 8 && pos + m < {npos}) {{
        const long long P = pos + m;
{decode}        if ({wkey} != key0) break;
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
        Ok((src, 256, npos * row_blocks))
    }
}
