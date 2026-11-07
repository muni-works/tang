//! FP8 contraction tiles on Hopper (sm_90a).
//!
//! A contraction whose two plain operands are stored block-scaled FP8
//! ([`Format::E4M3`] / [`Format::E5M2`]) runs on `wgmma` m64nNk32 with
//! f32 accumulation. The scale is part of the operand's type: its buffer
//! holds the bytes, then one f32 per `block` consecutive elements, so the
//! scale map is just another affine view of the same buffer — element
//! `a` reads byte `a` and scale `a / block`. The tile reads it per
//! 128-wide k chunk: each chunk's product goes to a scratch accumulator,
//! which is folded into the running sum times `sa[row] · sb[col]` (the
//! chunk's scales; constant over the chunk when the block is a multiple
//! of 128 along k). Per-tensor scales (block 0) are the same with one
//! scale.
//!
//! Warp specialized: one producer warpgroup stages A `[bm][128]` and B
//! `[bn][128]` bytes (128-byte rows, 16-byte chunks XOR-swizzled by row,
//! `cp.async`) plus the chunk's scales into a ring of `stages` buffers
//! (mbarriers full/empty); `bm / 64` consumer warpgroups issue the wgmma.
//! FP8 `wgmma` takes K-major operands only, so both operands must be
//! contiguous along the reduction.
//!
//! Everything else about the comp — other point dims, an epilogue map on
//! the accumulator, a literal factor, the output format — is handled like
//! the half tiles.

use std::fmt::Write;

use super::super::tile::emit_scalar;
use super::super::Format;
use super::super::{fixed, numel, TensorGraph};
use super::{f32_lit, float_literals, MmaShape, SM90_PRELUDE};

/// Hopper FP8 tile shapes (`bm` multiple of 64: one consumer warpgroup
/// per 64 rows; `bn` up to 256). `chunk` is fixed at 128 (one 128-byte
/// swizzle row). Each chunk's product is issued in `nsub` column slices
/// (2, or 4 past 128 columns) into two alternating scratch accumulators,
/// so slice `s + 1`'s wgmma runs while slice `s` is scaled into the sum:
/// `bn / 2 + bn / nsub` accumulator registers per thread.
pub const SM90_FP8_SHAPES: [MmaShape; 6] = [
    MmaShape::new(128, 256, 12, 1).with(128, 4, 1).wg(),
    MmaShape::new(128, 128, 12, 1).with(128, 6, 1).wg(),
    MmaShape::new(128, 192, 12, 1).with(128, 4, 1).wg(),
    MmaShape::new(128, 256, 12, 1).with(128, 3, 1).wg(),
    MmaShape::new(128, 64, 12, 1).with(128, 6, 1).wg(),
    MmaShape::new(64, 256, 8, 1).with(128, 6, 1).wg(),
];

pub(super) fn ptx_ty(f: Format) -> &'static str {
    match f {
        Format::E4M3(_) => "e4m3",
        _ => "e5m2",
    }
}

/// `wgmma` m64n`n`k32 over FP8, A and B from shared-memory descriptors
/// (both K-major); `acc` false zeroes the accumulator first.
fn wgmma_fp8(ta: &str, tb: &str, n: usize) -> String {
    wgmma_fp8_named("tang_wg8", ta, tb, n)
}

/// [`wgmma_fp8`] under another function name.
pub(super) fn wgmma_fp8_named(name: &str, ta: &str, tb: &str, n: usize) -> String {
    let regs = n / 2;
    let outs: Vec<String> = (0..regs).map(|i| format!("%{i}")).collect();
    let cons: Vec<String> = (0..regs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    format!(
        "static __device__ __forceinline__ void {name}(float* d, unsigned long long a, unsigned long long b, int acc) {{\n    asm volatile(\"{{ .reg .pred p; setp.ne.s32 p, %{}, 0; wgmma.mma_async.sync.aligned.m64n{n}k32.f32.{ta}.{tb} {{{}}}, %{}, %{}, p, 1, 1; }}\"\n        : {}\n        : \"l\"(a), \"l\"(b), \"r\"(acc));\n}}\n",
        regs + 2,
        outs.join(", "),
        regs,
        regs + 1,
        cons.join(", ")
    )
}

impl TensorGraph {
    /// CUDA source for `root` as an FP8 `wgmma` tile of shape `sh`, or
    /// `None` when it is not a single-term contraction of two FP8
    /// operands contiguous along k (or the shape does not fit).
    pub fn cuda_fp8_kernel_shaped(
        &self,
        root: usize,
        reads: &[usize],
        sh: MmaShape,
    ) -> Option<String> {
        let c = self.comp_at(root)?;
        let fmt = |j: usize| self.nodes[j].precision.storage;
        let bl = self.bilinear_tile(root, reads)?;
        if bl.outs.len() != 1 || bl.outs[0].len() != 1 {
            return None;
        }
        let t = &bl.outs[0][0];
        if t.map_a.is_some() || t.map_b.is_some() {
            return None;
        }
        let (ba, bb) = (t.buf_a, t.buf_b);
        let (fa, fb) = (fmt(ba), fmt(bb));
        if !fa.is_fp8() || !fb.is_fp8() {
            return None;
        }
        let accum = self.nodes[bl.red_node].precision.accum;
        if accum != Format::F32 && accum != Format::Fp8Accum {
            return None;
        }
        let out_f = fmt(root);
        if out_f.is_fp8() {
            return None;
        }
        let (aa, ab) = (&t.addr_a, &t.addr_b);
        let (kdim, dx, dy) = (bl.kdim, bl.dx, bl.dy);
        let (m, n, k_total) = (bl.m, bl.n, bl.k_total);
        // K-major operands with 16-byte aligned rows
        let aligned = |a: &super::super::msl_mma::Addr, keep: usize| {
            a.coefs[kdim] == 1
                && a.offset % 16 == 0
                && a.coefs
                    .iter()
                    .enumerate()
                    .all(|(d, &c)| d == kdim || d == keep || c % 16 == 0)
                && a.coefs[keep] % 16 == 0
        };
        if !aligned(aa, dy) || !aligned(ab, dx) || k_total % 16 != 0 {
            return None;
        }
        let (bm, bn, chunk) = (sh.bm, sh.bn, 128usize);
        let n_wg = bm / 64;
        if !sh.wgmma || bm % 64 != 0 || n_wg == 0 || bn % 16 != 0 || bn > 256 {
            return None;
        }
        let threads = 128 * (n_wg + 1);
        // scales: constant over a 128-wide k chunk when the block is a
        // multiple of 128 and chunk starts are block-aligned
        let numel_of = |j: usize| numel(&self.nodes[j].shape);
        let (na, nb) = (numel_of(ba), numel_of(bb));
        let (blk_a, blk_b) = (fa.block(na)?, fb.block(nb)?);
        for (blk, a, nn) in [(blk_a, aa, na), (blk_b, ab, nb)] {
            if blk != nn && (blk % chunk != 0 || a.offset % chunk as i64 != 0) {
                return None;
            }
            if blk != nn
                && a.coefs
                    .iter()
                    .enumerate()
                    .any(|(d, &c)| d != kdim && c % chunk as i64 != 0)
            {
                return None;
            }
        }
        let (sa_off, sb_off) = (Format::scale_offset(na), Format::scale_offset(nb));

        let pts = fixed(c.points());
        let others = &bl.others;
        let tiles_x = n.div_ceil(bn);
        let tiles_y = m.div_ceil(bm);
        let n_tiles = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();
        let w_out = c.width();
        let stages = sh.stages.max(2);
        // per stage: A, B bytes, then the chunk's scales (bm + bn floats)
        let a_bytes = bm * chunk;
        let b_bytes = bn * chunk;
        let sc_bytes = 4 * (bm + bn);
        let stage_bytes = (a_bytes + b_bytes + sc_bytes).next_multiple_of(1024);
        let bar_off = stages * stage_bytes;
        let smem = bar_off + 16 * stages + 1024;
        if smem > 227 * 1024 {
            return None;
        }
        let n_chunks = k_total.div_ceil(chunk);

        let addr =
            |a: &super::super::msl_mma::Addr, row_dim: usize, row: &str, k: &str| -> String {
                let mut e = format!("{}", a.offset);
                for (d, &coef) in a.coefs.iter().enumerate() {
                    if coef == 0 {
                        continue;
                    }
                    let v = if d == row_dim {
                        row.to_string()
                    } else if d == kdim {
                        k.to_string()
                    } else {
                        format!("p{d}")
                    };
                    e.push_str(&format!(" + {coef}ll * (long long)({v})"));
                }
                format!("({e})")
            };

        let dbg = std::env::var("TANG_FP8_DBG").unwrap_or_default();
        let mut s = String::new();
        writeln!(
            s,
            "// Auto-generated by tang-compile: mma (FP8 {}x{} wgmma) tile",
            ptx_ty(fa),
            ptx_ty(fb)
        )
        .unwrap();
        writeln!(s, "// block {bm}x{bn}, {n_wg} consumer warpgroups, chunk {chunk}, {stages} stages, block scales {blk_a}/{blk_b}").unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        writeln!(s, "#define TANG_SMEM {smem}").unwrap();
        writeln!(
            s,
            "#define TANG_OUT_LEN {}",
            pts.iter().product::<usize>() * w_out
        )
        .unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(super::MMA_PRELUDE);
        s.push_str(super::HALF_PRELUDE);
        s.push_str(SM90_PRELUDE);
        // TANG_FP8_OVERLAP=1: column slices with the next slice's wgmma in
        // flight during the fold (ptxas tends to serialize it); default:
        // the whole chunk, then the fold (DeepGEMM's order — the other
        // consumer warpgroup keeps the tensor cores busy meanwhile)
        // per-tensor scales on both sides: no per-chunk fold, the wgmma
        // accumulate into the sum and the scales apply in the epilogue
        // (only when the schedule accepts tensor-core accumulation: with an
        // f32 accum the partial sums are promoted every chunk)
        let tensorwise = blk_a == na && blk_b == nb && accum == Format::Fp8Accum;
        if accum == Format::Fp8Accum && !tensorwise {
            return None;
        }
        let overlap = !tensorwise && std::env::var("TANG_FP8_OVERLAP").ok().as_deref() == Some("1");
        s.push_str(&wgmma_fp8(
            ptx_ty(fa),
            ptx_ty(fb),
            if overlap {
                bn / if bn > 128 { 4 } else { 2 }
            } else {
                bn
            },
        ));
        s.push_str("static __device__ __forceinline__ void tang_cp4s(unsigned d, const void* src, bool ok) {\n    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4, %2;\" :: \"r\"(d), \"l\"(src), \"r\"(ok ? 4 : 0) : \"memory\");\n}\n");
        let elem = |j: usize| match fmt(j) {
            Format::F32 => "float",
            f if f.is_fp8() => "unsigned char",
            _ => "unsigned short",
        };
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const {}* __restrict__ b{j}", elem(j)))
            .collect();
        params.push(format!("{}* __restrict__ out", elem(root)));
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}, 1) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    extern __shared__ __align__(16) float smem[];").unwrap();
        writeln!(
            s,
            "    unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;"
        )
        .unwrap();
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        // grouped rasterization: blocks of `grp` tile rows walk their
        // columns together, so the tiles resident at once share B in L2
        let grp = 8usize.min(tiles_y);
        writeln!(s, "    unsigned pid = tile % {}u;", tiles_x * tiles_y).unwrap();
        writeln!(
            s,
            "    unsigned first_y = (pid / {}u) * {grp}u;",
            grp * tiles_x
        )
        .unwrap();
        writeln!(s, "    unsigned gsize = min({tiles_y}u - first_y, {grp}u);").unwrap();
        writeln!(
            s,
            "    unsigned tile_y = first_y + (pid % {}u) % gsize;",
            grp * tiles_x
        )
        .unwrap();
        writeln!(
            s,
            "    unsigned tile_x = (pid % {}u) / gsize;",
            grp * tiles_x
        )
        .unwrap();
        writeln!(s, "    unsigned lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    unsigned p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    (void)lead;").unwrap();
        writeln!(
            s,
            "    unsigned row0 = tile_y * {bm}u, col0 = tile_x * {bn}u;"
        )
        .unwrap();
        writeln!(
            s,
            "    unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem) + 1023u) & ~1023u;"
        )
        .unwrap();
        writeln!(s, "    const float* smf = (const float*)((const char*)smem + (sbase - (unsigned)__cvta_generic_to_shared(smem)));").unwrap();
        writeln!(
            s,
            "    unsigned bar_f = sbase + {bar_off}u, bar_e = bar_f + {}u;",
            8 * stages
        )
        .unwrap();
        writeln!(s, "    if (tid == 0u) {{").unwrap();
        writeln!(s, "        for (unsigned q = 0u; q < {stages}u; q++) {{ tang_bar_init(bar_f + 8u * q, 128u); tang_bar_init(bar_e + 8u * q, {}u); }}", 128 * n_wg).unwrap();
        writeln!(
            s,
            "        asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");"
        )
        .unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(
            s,
            "    const float* sa_g = (const float*)((const unsigned char*)b{ba} + {sa_off}u);"
        )
        .unwrap();
        writeln!(
            s,
            "    const float* sb_g = (const float*)((const unsigned char*)b{bb} + {sb_off}u);"
        )
        .unwrap();

        // producer
        writeln!(s, "    if (warp < 4u) {{").unwrap();
        writeln!(s, "        tang_setmaxnreg_dec<40>();").unwrap();
        // thread t copies 16-byte column c16 = t % 8 of rows t / 8 + 16·i:
        // its global row pointers step by 16 rows, its swizzled shared
        // offsets by 2 KB, and the swizzle (row mod 8) is fixed
        let k_ragged = k_total % chunk != 0;
        let mut copies = String::new();
        if dbg != "noload" {
            for (side, rows, ok_n, base) in [
                ("a", bm, m % bm == 0, 0usize),
                ("b", bn, n % bn == 0, a_bytes),
            ] {
                for i in 0..rows / 16 {
                    let lim = if side == "a" { m } else { n };
                    let org = if side == "a" { "row0" } else { "col0" };
                    let mut ok = Vec::new();
                    if !ok_n {
                        ok.push(format!("{org} + r0 + {}u < {lim}u", 16 * i));
                    }
                    if k_ragged {
                        ok.push(format!("k0 + c16 * 16u < {k_total}u"));
                    }
                    let okx = if ok.is_empty() {
                        "true".to_string()
                    } else {
                        ok.join(" && ")
                    };
                    let step = if side == "a" {
                        aa.coefs[dy]
                    } else {
                        ab.coefs[dx]
                    } * 16
                        * i as i64;
                    writeln!(copies, "            {{ bool ok = {okx}; tang_cp16s(st + {}u + sw + {}u, p{side} + (ok ? {step}ll + (long long)k0 : 0ll), ok); }}", base, 2048 * i).unwrap();
                }
            }
        }
        writeln!(s, "        unsigned r0 = tid / 8u, c16 = tid % 8u, sw = r0 * 128u + ((c16 ^ (r0 % 8u)) * 16u);").unwrap();
        writeln!(
            s,
            "        const unsigned char* pa = b{ba} + ({}) + c16 * 16u;",
            addr(aa, dy, &format!("min(row0 + r0, {}u)", m - 1), "0")
        )
        .unwrap();
        writeln!(
            s,
            "        const unsigned char* pb = b{bb} + ({}) + c16 * 16u;",
            addr(ab, dx, &format!("min(col0 + r0, {}u)", n - 1), "0")
        )
        .unwrap();
        writeln!(s, "        (void)pa; (void)pb; (void)sw;").unwrap();
        writeln!(s, "        for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
        writeln!(s, "            unsigned q = c % {stages}u;").unwrap();
        writeln!(
            s,
            "            tang_bar_wait(bar_e + 8u * q, ((c / {stages}u) & 1u) ^ 1u);"
        )
        .unwrap();
        writeln!(s, "            unsigned st = sbase + q * {stage_bytes}u;").unwrap();
        writeln!(s, "            unsigned k0 = c * {chunk}u;").unwrap();
        s.push_str(&copies);
        // the chunk's scales: one per A row, one per B column
        let sidx =
            |a: &super::super::msl_mma::Addr, blk: usize, nn: usize, row_dim: usize, row: &str| {
                if blk == nn {
                    "0ll".to_string()
                } else {
                    format!("({} / {blk}ll)", addr(a, row_dim, row, "k0"))
                }
            };
        writeln!(
            s,
            "            for (unsigned r = tid; r < {}u; r += 128u) {{",
            bm + bn
        )
        .unwrap();
        writeln!(s, "                if (r < {bm}u) {{ unsigned gr = row0 + r; bool ok = gr < {m}u; tang_cp4s(st + {}u + 4u * r, sa_g + (ok ? {} : 0ll), ok); }}", a_bytes + b_bytes, sidx(aa, blk_a, na, dy, "gr")).unwrap();
        writeln!(s, "                else {{ unsigned gc = col0 + r - {bm}u; bool ok = gc < {n}u; tang_cp4s(st + {}u + 4u * r, sb_g + (ok ? {} : 0ll), ok); }}", a_bytes + b_bytes, sidx(ab, blk_b, nb, dx, "gc")).unwrap();
        writeln!(s, "            }}").unwrap();
        // the full barrier completes when every producer thread's copies
        // of this chunk have landed (no wait here: the ring stays full)
        writeln!(s, "            asm volatile(\"cp.async.mbarrier.arrive.noinc.shared::cta.\" \"b\" \"64 [%0];\" :: \"r\"(bar_f + 8u * q) : \"memory\");").unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        tang_cp_wait<0>();").unwrap();
        writeln!(s, "        return;").unwrap();
        writeln!(s, "    }}").unwrap();

        // consumers: steps (chunk c, slice sl) in order, slice sl into
        // scratch T[sl % 2]; the next step is issued before this one is
        // folded in
        let regs = bn / 2;
        let nsub = if !overlap {
            1
        } else if bn > 128 {
            4
        } else {
            2
        };
        let nw = bn / nsub;
        let tr = nw / 2;
        writeln!(s, "    tang_setmaxnreg_inc<232>();").unwrap();
        writeln!(s, "    unsigned cwi = warp / 4u - 1u;").unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        writeln!(s, "    unsigned lr0 = cwi * 64u + (warp % 4u) * 16u + gq;").unwrap();
        if overlap {
            writeln!(s, "    float acc[{regs}], t0[{tr}], t1[{tr}];").unwrap();
        } else if tensorwise {
            writeln!(s, "    float acc[{regs}];").unwrap();
        } else {
            writeln!(s, "    float acc[{regs}], t0[{tr}];").unwrap();
        }
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int i = 0; i < {regs}; i++) acc[i] = 0.0f;").unwrap();
        // issue slice `sl` of the chunk in stage address `st` into `t`
        let issue = |s: &mut String, ind: &str, t: &str, sl: usize| {
            writeln!(s, "{ind}tang_wg_fence();").unwrap();
            writeln!(s, "{ind}#pragma unroll").unwrap();
            writeln!(
                s,
                "{ind}for (unsigned kb = 0u; kb < {chunk}u; kb += 32u) {{"
            )
            .unwrap();
            writeln!(
                s,
                "{ind}    unsigned long long da = tang_desc(st + cwi * {}u + kb, 16u, 1024u);",
                64 * 128
            )
            .unwrap();
            writeln!(
                s,
                "{ind}    unsigned long long db = tang_desc(st + {}u + kb, 16u, 1024u);",
                a_bytes + sl * nw * 128
            )
            .unwrap();
            if dbg != "nomma" {
                writeln!(s, "{ind}    tang_wg8({t}, da, db, kb != 0u);").unwrap();
            }
            writeln!(s, "{ind}}}").unwrap();
            writeln!(s, "{ind}tang_wg_commit();").unwrap();
        };
        if tensorwise {
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned q = c % {stages}u;").unwrap();
            writeln!(
                s,
                "        tang_bar_wait(bar_f + 8u * q, (c / {stages}u) & 1u); tang_fence_async();"
            )
            .unwrap();
            writeln!(s, "        unsigned st = sbase + q * {stage_bytes}u;").unwrap();
            writeln!(s, "        tang_wg_fence();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(
                s,
                "        for (unsigned kb = 0u; kb < {chunk}u; kb += 32u) {{"
            )
            .unwrap();
            writeln!(
                s,
                "            unsigned long long da = tang_desc(st + cwi * {}u + kb, 16u, 1024u);",
                64 * 128
            )
            .unwrap();
            writeln!(
                s,
                "            unsigned long long db = tang_desc(st + {a_bytes}u + kb, 16u, 1024u);"
            )
            .unwrap();
            writeln!(s, "            tang_wg8(acc, da, db, 1);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_wg_commit();").unwrap();
            writeln!(s, "        tang_wg_wait<1>();").unwrap();
            writeln!(
                s,
                "        if (c > 0u) tang_bar_arrive(bar_e + 8u * ((c - 1u) % {stages}u));"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    tang_wg_wait<0>();").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int i = 0; i < {regs}; i++) asm volatile(\"\" : \"+f\"(acc[i]) :: \"memory\");").unwrap();
            writeln!(s, "    float sab = sa_g[0] * sb_g[0];").unwrap();
        } else if !overlap {
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned q = c % {stages}u;").unwrap();
            writeln!(
                s,
                "        tang_bar_wait(bar_f + 8u * q, (c / {stages}u) & 1u); tang_fence_async();"
            )
            .unwrap();
            writeln!(s, "        unsigned st = sbase + q * {stage_bytes}u;").unwrap();
            issue(&mut s, "        ", "t0", 0);
            writeln!(s, "        tang_wg_wait<0>();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int i = 0; i < {tr}; i++) asm volatile(\"\" : \"+f\"(t0[i]) :: \"memory\");").unwrap();
            writeln!(
                s,
                "        const float* scl = smf + (q * {stage_bytes}u + {}u) / 4u;",
                a_bytes + b_bytes
            )
            .unwrap();
            writeln!(s, "        float s_lo = scl[lr0], s_hi = scl[lr0 + 8u];").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int j = 0; j < {}; j++) {{", bn / 8).unwrap();
            writeln!(
                s,
                "            float2 sb = *(const float2*)(scl + {bm}u + 8u * j + 2u * tq);"
            )
            .unwrap();
            writeln!(
                s,
                "            acc[4 * j + 0] += t0[4 * j + 0] * s_lo * sb.x;"
            )
            .unwrap();
            writeln!(
                s,
                "            acc[4 * j + 1] += t0[4 * j + 1] * s_lo * sb.y;"
            )
            .unwrap();
            writeln!(
                s,
                "            acc[4 * j + 2] += t0[4 * j + 2] * s_hi * sb.x;"
            )
            .unwrap();
            writeln!(
                s,
                "            acc[4 * j + 3] += t0[4 * j + 3] * s_hi * sb.y;"
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_bar_arrive(bar_e + 8u * q);").unwrap();
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    unsigned st = sbase;").unwrap();
        if overlap {
            writeln!(s, "    tang_bar_wait(bar_f, 0u); tang_fence_async();").unwrap();
            issue(&mut s, "    ", "t0", 0);
        }
        // chunk c's slices; `last`: no next chunk to issue (the final
        // chunk is peeled so the loop's issue is unconditional — a wgmma
        // issued under a branch makes ptxas wait on it at the merge)
        let body = |s: &mut String, last: bool| {
            writeln!(s, "        unsigned q = c % {stages}u;").unwrap();
            writeln!(
                s,
                "        const float* scl = smf + (q * {stage_bytes}u + {}u) / 4u;",
                a_bytes + b_bytes
            )
            .unwrap();
            writeln!(s, "        float s_lo = scl[lr0], s_hi = scl[lr0 + 8u];").unwrap();
            for sl in 0..nsub {
                let (cur, nxt) = if sl % 2 == 0 {
                    ("t0", "t1")
                } else {
                    ("t1", "t0")
                };
                if sl + 1 < nsub {
                    issue(s, "        ", nxt, sl + 1);
                    writeln!(s, "        tang_wg_wait<1>();").unwrap();
                } else if !last {
                    writeln!(s, "        unsigned q1 = (c + 1u) % {stages}u;").unwrap();
                    writeln!(s, "        tang_bar_wait(bar_f + 8u * q1, ((c + 1u) / {stages}u) & 1u); tang_fence_async();").unwrap();
                    writeln!(s, "        st = sbase + q1 * {stage_bytes}u;").unwrap();
                    issue(s, "        ", nxt, 0);
                    writeln!(s, "        tang_wg_wait<1>();").unwrap();
                } else {
                    writeln!(s, "        tang_wg_wait<0>();").unwrap();
                }
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int i = 0; i < {tr}; i++) asm volatile(\"\" : \"+f\"({cur}[i]) :: \"memory\");").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int j = 0; j < {}; j++) {{", nw / 8).unwrap();
                writeln!(
                    s,
                    "            float2 sb = *(const float2*)(scl + {}u + 8u * j + 2u * tq);",
                    bm + sl * nw
                )
                .unwrap();
                let o = sl * nw / 2;
                writeln!(
                    s,
                    "            acc[{o} + 4 * j + 0] += {cur}[4 * j + 0] * s_lo * sb.x;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            acc[{o} + 4 * j + 1] += {cur}[4 * j + 1] * s_lo * sb.y;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            acc[{o} + 4 * j + 2] += {cur}[4 * j + 2] * s_hi * sb.x;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            acc[{o} + 4 * j + 3] += {cur}[4 * j + 3] * s_hi * sb.y;"
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
            }
            // every wgmma reading stage q has retired (the one in flight
            // reads the next stage)
            writeln!(s, "        tang_bar_arrive(bar_e + 8u * q);").unwrap();
        };
        if overlap {
            writeln!(s, "    unsigned c = 0u;").unwrap();
            writeln!(s, "    for (; c + 1u < {n_chunks}u; c++) {{").unwrap();
            body(&mut s, false);
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    {{").unwrap();
            body(&mut s, true);
            writeln!(s, "    }}").unwrap();
        }

        // epilogue straight from the accumulators
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int j = 0; j < {}; j++)", bn / 8).unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
        writeln!(s, "        unsigned p{dy} = row0 + lr0 + 8u * (r / 2);").unwrap();
        writeln!(
            s,
            "        unsigned p{dx} = col0 + 8u * j + 2u * tq + (r % 2);"
        )
        .unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        writeln!(
            s,
            "            float red0 = acc[4 * j + r]{};",
            if tensorwise { " * sab" } else { "" }
        )
        .unwrap();
        if bl.scales[0] != 1.0 {
            writeln!(s, "            red0 *= {}f;", f32_lit(bl.scales[0])).unwrap();
        }
        let storage = |j: usize| self.storage(super::super::TensorId(j as u32)).n();
        let ld = |j: usize, idx: &str| -> Option<String> {
            Some(match fmt(j) {
                Format::F32 => format!("b{j}[{idx}]"),
                Format::F16 => format!("tang_h2f(b{j}[{idx}])"),
                Format::BF16 => format!("tang_b2f(b{j}[{idx}])"),
                _ => return None,
            })
        };
        let vals: Vec<String> = if bl.epilogue {
            let prank = c.points().len();
            let mut arg_vals = Vec::new();
            for a in &c.args {
                if a.src.n() == bl.red_node {
                    arg_vals.push(format!("red{}", self.epilogue_slot(a)));
                    continue;
                }
                let ad = self.flat_addr(a, prank);
                let mut e = format!("{}", ad.offset);
                for (d, &coef) in ad.coefs.iter().enumerate() {
                    if coef != 0 {
                        e.push_str(&format!(" + {coef} * (int)p{d}"));
                    }
                }
                arg_vals.push(ld(storage(a.src.n()), &format!("(unsigned)({e})"))?);
            }
            let mut body = String::new();
            let outs = emit_scalar(&mut body, &c.body, &c.body_outs, "e0_", &|v| {
                arg_vals[v as usize].clone()
            });
            for line in body.lines() {
                writeln!(
                    s,
                    "            {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            outs
        } else {
            vec!["red0".to_string()]
        };
        let mut flat = String::from("0u");
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np}u + p{d}");
        }
        writeln!(s, "            unsigned g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            let v = match out_f {
                Format::F32 => v.clone(),
                Format::F16 => format!("tang_f2h({v})"),
                _ => format!("tang_f2b({v})"),
            };
            writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(s)
    }
}

// ---------------------------------------------------------------------------
// FP8 flash attention (forward) on the Hopper flash schedule
// ---------------------------------------------------------------------------

/// e4m3 pair packing and unpacking for the softmax weights.
pub(super) const FP8_FLASH_PRELUDE: &str = r#"static __device__ __forceinline__ unsigned short tang_f2e4x2(float lo, float hi) { unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r; }
static __device__ __forceinline__ float2 tang_e4x2f(unsigned short v) {
    unsigned h; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    float2 f; asm("{ .reg .f16 a, b; mov." "b" "32 {a, b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b; }" : "=f"(f.x), "=f"(f.y) : "r"(h));
    return f;
}
static __device__ __forceinline__ void tang_cp4s(unsigned d, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(ok ? 4 : 0) : "memory");
}
"#;

/// Layout of an FP8 flash tile: formats, block sizes (`None`: one scale
/// for the tensor), byte offsets of each operand's scales in its buffer,
/// and the shared-memory map.
pub(super) struct Fp8FlashPlan {
    pub fq: Format,
    pub fk: Format,
    pub fv: Format,
    blk: [Option<usize>; 3],
    scale_off: [usize; 3],
    bufs: [usize; 3],
    addrs: [super::super::msl_mma::Addr; 3],
    qsc_off: usize,
    p_off: usize,
    st_off: usize,
    k_bytes: usize,
    v_bytes: usize,
    stage_bytes: usize,
    bar_off: usize,
    pub smem: usize,
}

/// Shape facts the FP8 flash body needs from the schedule.
pub(super) struct Fp8FlashCtx {
    pub bj: usize,
    pub d: usize,
    pub stages: usize,
    pub threads: usize,
    pub nsf: usize,
    pub nof: usize,
}

type Addr = super::super::msl_mma::Addr;

impl TensorGraph {
    /// Whether the flash tile `f` can run on FP8 `wgmma`, and its layout:
    /// one term, Q and K contiguous along the 128-wide head dim (one
    /// 128-byte row each), V contiguous along the keys (stored `[d, n]`:
    /// FP8 `wgmma` reads K-major B only), 128 keys per step, scales per
    /// row (Q, K: blocks of 128 along the head dim) or per 128 keys (V),
    /// or per tensor; the reduction's `operand` format E4M3 (the softmax
    /// weights are rounded to it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fp8_flash_plan(
        &self,
        f: &super::Flash,
        rc: &super::super::Comp,
        plain_v: &(usize, Addr),
        h: super::Sm90Flash,
        bj: usize,
        kd: usize,
        d: usize,
        stages: usize,
    ) -> Option<Fp8FlashPlan> {
        if f.terms.len() != 1 || f.score_buf.is_some() {
            return None;
        }
        if !matches!(self.nodes[f.red_node].precision.operand, Format::E4M3(_)) {
            return None;
        }
        if kd != 128
            || bj != 128
            || !d.is_multiple_of(64)
            || d > 256
            || !f.n_keys.is_multiple_of(16)
        {
            return None;
        }
        let (qb, qa, kb, ka) = &f.terms[0];
        let (vb, va) = plain_v;
        let inner = f.score_to_red.len();
        let jdim = rc.rank() - 1;
        let fmt = |j: usize| self.nodes[j].precision.storage;
        let (fq, fk, fv) = (fmt(*qb), fmt(*kb), fmt(*vb));
        if !(fq.is_fp8() && fk.is_fp8() && fv.is_fp8()) {
            return None;
        }
        // contiguous along `along`, 16-byte aligned rows
        let rows16 = |a: &Addr, along: usize| {
            a.coefs[along] == 1
                && a.offset % 16 == 0
                && a.coefs
                    .iter()
                    .enumerate()
                    .all(|(k, &c)| k == along || c % 16 == 0)
        };
        if !rows16(qa, inner) || !rows16(ka, inner) || !rows16(va, jdim) {
            return None;
        }
        // a block scale must be constant over the 128 elements a step
        // reads along the contiguous dim
        let blk = |j: usize, a: &Addr, along: usize| -> Option<Option<usize>> {
            let n = numel(&self.nodes[j].shape);
            let b = fmt(j).block(n)?;
            if b == n {
                return Some(None);
            }
            let aligned = b % 128 == 0
                && a.offset % 128 == 0
                && a.coefs
                    .iter()
                    .enumerate()
                    .all(|(k, &c)| k == along || c % b as i64 == 0 || c % 128 == 0);
            aligned.then_some(Some(b))
        };
        let bq = blk(*qb, qa, inner)?;
        let bk = blk(*kb, ka, inner)?;
        let bv = blk(*vb, va, jdim)?;
        let bm = 64 * h.consumers;
        let up = |x: usize| x.next_multiple_of(1024);
        let q_bytes = bm * 128;
        let qsc_off = q_bytes;
        let p_off = up(qsc_off + 4 * bm);
        let st_off = up(p_off + h.consumers * 64 * bj);
        let k_bytes = bj * 128;
        let v_bytes = d * 128;
        let stage_bytes = up(k_bytes + v_bytes + 4 * (bj + d));
        let bar_off = st_off + stages * stage_bytes;
        let smem = bar_off + 16 * stages + 1024;
        if smem > 227 * 1024 {
            return None;
        }
        let soff = |j: usize| Format::scale_offset(numel(&self.nodes[j].shape));
        Some(Fp8FlashPlan {
            fq,
            fk,
            fv,
            blk: [bq, bk, bv],
            scale_off: [soff(*qb), soff(*kb), soff(*vb)],
            bufs: [*qb, *kb, *vb],
            addrs: [qa.clone(), ka.clone(), va.clone()],
            qsc_off,
            p_off,
            st_off,
            k_bytes,
            v_bytes,
            stage_bytes,
            bar_off,
            smem,
        })
    }

    /// The FP8 Hopper flash schedule's body: Q (and its scales) staged by
    /// every thread, then a producer warpgroup ringing K, Vᵀ and their
    /// scales through `stages` buffers, and consumer warpgroups of 64
    /// rows: S = Q·Kᵀ on FP8 `wgmma`, scaled by `sq[row]·sk[key]`, the
    /// online softmax with the weights P rounded to e4m3 (the row sum
    /// adds the rounded weights, as the emulator does), P through shared
    /// memory into P·V on FP8 `wgmma`, scaled by `sv[col]` into O. Leaves
    /// `o`, `m_row` (natural log) and `l_row` for the shared epilogue.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fp8_flash_body(
        &self,
        s: &mut String,
        f: &super::Flash,
        rc: &super::super::Comp,
        pl: &Fp8FlashPlan,
        h: super::Sm90Flash,
        cx: Fp8FlashCtx,
        row_coords: &dyn Fn(&mut String, &str, &str),
        arg_value: &dyn Fn(&super::super::Arg) -> String,
        emit_body: &dyn Fn(
            &mut String,
            &str,
            &tang_expr::ExprGraph,
            &[tang_expr::ExprId],
            &str,
            &[String],
        ) -> Vec<String>,
        coord: &dyn Fn(usize) -> String,
        score_coord: &dyn Fn(usize) -> String,
        addr_expr: &dyn Fn(&Addr, &dyn Fn(usize) -> String) -> String,
    ) {
        let Fp8FlashCtx {
            bj,
            d,
            stages,
            threads: t_all,
            nsf,
            nof,
        } = cx;
        let cw = h.consumers;
        let bm = 64 * cw;
        let [qb, kb, vb] = pl.bufs;
        let [qa, ka, va] = &pl.addrs;
        let swz = |r: &str, c16: &str| format!("({r}) * 128u + ((({c16}) ^ (({r}) % 8u)) * 16u)");
        let sidx = |i: usize, a: &Addr, cf: &dyn Fn(usize) -> String| match pl.blk[i] {
            None => "0u".to_string(),
            Some(b) => format!("(unsigned)(({}) / {b})", addr_expr(a, cf)),
        };
        let (st_off, stage_bytes, k_bytes, v_bytes) =
            (pl.st_off, pl.stage_bytes, pl.k_bytes, pl.v_bytes);
        let w = |s: &mut String, line: String| {
            s.push_str(&line);
            s.push('\n');
        };
        w(
            s,
            "    unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem) + 1023u) & ~1023u;"
                .into(),
        );
        w(s, "    const float* smf = (const float*)((const char*)smem + (sbase - (unsigned)__cvta_generic_to_shared(smem)));".into());
        w(
            s,
            format!(
                "    unsigned bar_f = sbase + {}u, bar_e = bar_f + {}u;",
                pl.bar_off,
                8 * stages
            ),
        );
        for (i, (nm, b)) in [("sq_g", qb), ("sk_g", kb), ("sv_g", vb)]
            .iter()
            .enumerate()
        {
            w(
                s,
                format!(
                    "    const float* {nm} = (const float*)((const unsigned char*)b{b} + {}u);",
                    pl.scale_off[i]
                ),
            );
        }
        w(s, "    if (tid == 0u) {".into());
        w(s, format!("        for (unsigned q = 0u; q < {stages}u; q++) {{ tang_bar_init(bar_f + 8u * q, 128u); tang_bar_init(bar_e + 8u * q, {}u); }}", 128 * cw));
        w(
            s,
            "        asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");".into(),
        );
        w(s, "    }".into());
        // Q bytes and scales, by every thread
        w(
            s,
            format!(
                "    for (unsigned idx = tid; idx < {}u; idx += {t_all}u) {{",
                bm * 8
            ),
        );
        w(
            s,
            "        unsigned r = idx / 8u, c16 = idx % 8u, dd = c16 * 16u;".into(),
        );
        row_coords(s, "        ", "r");
        w(s, format!("        bool ok = ii < {}u;", f.m));
        w(
            s,
            format!(
                "        tang_cp16s(sbase + {}, b{qb} + (ok ? (unsigned)({}) : 0u), ok);",
                swz("r", "c16"),
                addr_expr(qa, score_coord)
            ),
        );
        w(s, "    }".into());
        w(
            s,
            format!("    for (unsigned r = tid; r < {bm}u; r += {t_all}u) {{"),
        );
        w(s, "        unsigned dd = 0u; (void)dd;".into());
        row_coords(s, "        ", "r");
        w(s, format!("        bool ok = ii < {}u;", f.m));
        w(
            s,
            format!(
                "        tang_cp4s(sbase + {}u + 4u * r, sq_g + (ok ? {} : 0u), ok);",
                pl.qsc_off,
                sidx(0, qa, score_coord)
            ),
        );
        w(s, "    }".into());
        w(
            s,
            "    tang_cp_commit(); tang_cp_wait<0>(); tang_fence_async();".into(),
        );
        w(s, "    __syncthreads();".into());

        // producer
        w(s, "    if (warp < 4u) {".into());
        w(
            s,
            format!("        tang_setmaxnreg_dec<{}>();", h.producer_regs),
        );
        w(s, "        unsigned it = 0u;".into());
        w(
            s,
            "        for (unsigned c = c_lo; c < c_hi; c++, it++) {".into(),
        );
        w(s, format!("            unsigned q = it % {stages}u;"));
        w(
            s,
            format!("            tang_bar_wait(bar_e + 8u * q, ((it / {stages}u) & 1u) ^ 1u);"),
        );
        w(
            s,
            format!("            unsigned st = sbase + {st_off}u + q * {stage_bytes}u;"),
        );
        w(
            s,
            format!(
                "            for (unsigned idx = tid; idx < {}u; idx += 128u) {{",
                bj * 8
            ),
        );
        w(s, format!("                unsigned r = idx / 8u, c16 = idx % 8u, dd = c16 * 16u, jj = c * {bj}u + r;"));
        w(s, format!("                bool ok = jj < {}u;", f.n_keys));
        w(
            s,
            format!(
                "                tang_cp16s(st + {}, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                swz("r", "c16"),
                addr_expr(ka, score_coord)
            ),
        );
        w(s, "            }".into());
        w(
            s,
            format!(
                "            for (unsigned idx = tid; idx < {}u; idx += 128u) {{",
                d * 8
            ),
        );
        w(s, format!("                unsigned nn = idx / 8u, c16 = idx % 8u, jj = c * {bj}u + c16 * 16u;"));
        w(s, format!("                bool ok = jj < {}u;", f.n_keys));
        w(s, format!("                tang_cp16s(st + {k_bytes}u + {}, b{vb} + (ok ? (unsigned)({}) : 0u), ok);", swz("nn", "c16"), addr_expr(va, coord)));
        w(s, "            }".into());
        w(
            s,
            format!(
                "            for (unsigned r = tid; r < {}u; r += 128u) {{",
                bj + d
            ),
        );
        w(s, format!("                if (r < {bj}u) {{ unsigned dd = 0u, jj = c * {bj}u + r; (void)dd; bool ok = jj < {}u; tang_cp4s(st + {}u + 4u * r, sk_g + (ok ? {} : 0u), ok); }}", f.n_keys, k_bytes + v_bytes, sidx(1, ka, score_coord)));
        w(s, format!("                else {{ unsigned nn = r - {bj}u, jj = c * {bj}u; tang_cp4s(st + {}u + 4u * r, sv_g + {}, true); }}", k_bytes + v_bytes, sidx(2, va, coord)));
        w(s, "            }".into());
        w(s, "            asm volatile(\"cp.async.mbarrier.arrive.noinc.shared::cta.\" \"b\" \"64 [%0];\" :: \"r\"(bar_f + 8u * q) : \"memory\");".into());
        w(s, "        }".into());
        w(s, "        tang_cp_wait<0>();".into());
        w(s, "        return;".into());
        w(s, "    }".into());

        // consumers
        w(
            s,
            format!("    tang_setmaxnreg_inc<{}>();", h.consumer_regs),
        );
        w(s, "    unsigned cwi = warp / 4u - 1u;".into());
        w(
            s,
            "    unsigned wrow = cwi * 64u + (warp % 4u) * 16u;".into(),
        );
        w(
            s,
            format!("    unsigned qaddr = sbase + cwi * {}u;", 64 * 128),
        );
        w(
            s,
            format!(
                "    unsigned pbase = sbase + {}u + cwi * {}u;",
                pl.p_off,
                64 * bj
            ),
        );
        if pl.blk[2].is_none() {
            w(s, format!("    float o[{nof}][4];"));
        } else {
            w(s, format!("    float o[{nof}][4], ot[{nof}][4];"));
        }
        w(s, "    #pragma unroll".into());
        w(
            s,
            format!(
                "    for (int j = 0; j < {nof}; j++) for (int r = 0; r < 4; r++) o[j][r] = 0.0f;"
            ),
        );
        w(s, "    float m_row[2] = {-1.0e30f, -1.0e30f};".into());
        w(s, "    float l_row[2] = {0.0f, 0.0f};".into());
        w(
            s,
            format!(
                "    float sq_r[2] = {{smf[{q}u + wrow + gq], smf[{q}u + wrow + gq + 8u]}};",
                q = pl.qsc_off / 4
            ),
        );
        w(s, "    unsigned it = 0u;".into());
        w(
            s,
            "    for (unsigned c = c_lo; c < c_hi; c++, it++) {".into(),
        );
        w(s, format!("        unsigned q = it % {stages}u;"));
        w(
            s,
            format!(
                "        tang_bar_wait(bar_f + 8u * q, (it / {stages}u) & 1u); tang_fence_async();"
            ),
        );
        w(s, format!("        unsigned kaddr = sbase + {st_off}u + q * {stage_bytes}u, vaddr = kaddr + {k_bytes}u;"));
        w(
            s,
            format!(
                "        const float* skp = smf + ({st_off}u + q * {stage_bytes}u + {}u) / 4u;",
                k_bytes + v_bytes
            ),
        );
        w(s, format!("        const float* svp = skp + {bj}u;"));
        w(s, format!("        float sc[{nsf}][4];"));
        w(s, "        tang_wg_fence();".into());
        w(s, "        #pragma unroll".into());
        w(s, "        for (int kk = 0; kk < 4; kk++)".into());
        w(s, "            tang_wgs(&sc[0][0], tang_desc(qaddr + kk * 32u, 16u, 1024u), tang_desc(kaddr + kk * 32u, 16u, 1024u), kk > 0);".into());
        w(s, "        tang_wg_commit(); tang_wg_wait<0>();".into());
        w(s, format!("        unsigned j0 = c * {bj}u;"));
        w(s, "        #pragma unroll".into());
        w(s, "        for (int h = 0; h < 2; h++) {".into());
        row_coords(s, "            ", "wrow + gq + 8u * h");
        w(s, "            float mloc = -1.0e30f;".into());
        w(s, "            #pragma unroll".into());
        w(s, format!("            for (int j = 0; j < {nsf}; j++)"));
        w(s, "            #pragma unroll".into());
        w(s, "            for (int e = 0; e < 2; e++) {".into());
        w(
            s,
            "                unsigned jj = j0 + 8u * j + 2u * tq + e;".into(),
        );
        w(s, format!("                float sval = sc[j][2 * h + e] * ({}f * sq_r[h] * skp[8u * j + 2u * tq + e]);", super::f32_lit(f.scale)));
        {
            let vals: Vec<String> = rc
                .args
                .iter()
                .enumerate()
                .map(|(k, a)| {
                    if k == f.score_arg {
                        "sval".to_string()
                    } else {
                        arg_value(a)
                    }
                })
                .collect();
            let o = emit_body(
                s,
                "                ",
                &rc.body,
                &[rc.body_outs[0]],
                "s0_",
                &vals,
            );
            w(
                s,
                format!(
                    "                float v = (jj < {}u) ? ({}) * 1.4426950408889634f : -1.0e30f;",
                    f.n_keys, o[0]
                ),
            );
        }
        w(s, "                sc[j][2 * h + e] = v;".into());
        w(s, "                mloc = fmaxf(mloc, v);".into());
        w(s, "            }".into());
        w(
            s,
            "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 1));".into(),
        );
        w(
            s,
            "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 2));".into(),
        );
        w(s, "            float m_new = fmaxf(m_row[h], mloc);".into());
        w(
            s,
            "            float alpha = exp2f(m_row[h] - m_new);".into(),
        );
        w(s, "            float lsum = 0.0f;".into());
        w(s, "            #pragma unroll".into());
        w(s, format!("            for (int j = 0; j < {nsf}; j++) {{"));
        // the weights rounded to e4m3; the sum adds the rounded values
        // scaled by 2^FP8_SOFTMAX_SHIFT into e4m3's normal range (O and l
        // carry the same factor, which the normalization cancels)
        let sh = super::super::precision::FP8_SOFTMAX_SHIFT;
        w(s, format!("                float2 pr = tang_e4x2f(tang_f2e4x2(exp2f(sc[j][2 * h] - m_new + {sh}.0f), exp2f(sc[j][2 * h + 1] - m_new + {sh}.0f)));"));
        w(
            s,
            "                sc[j][2 * h] = pr.x; sc[j][2 * h + 1] = pr.y; lsum += pr.x + pr.y;"
                .into(),
        );
        w(s, "            }".into());
        w(
            s,
            "            l_row[h] = l_row[h] * alpha + lsum; m_row[h] = m_new;".into(),
        );
        w(s, "            #pragma unroll".into());
        w(s, format!("            for (int j = 0; j < {nof}; j++) {{ o[j][2 * h] *= alpha; o[j][2 * h + 1] *= alpha; }}"));
        w(s, "        }".into());
        // P to shared memory, K-major 128-byte swizzled rows (the warpgroup's
        // previous P·V has retired: its wgmma were waited on)
        w(
            s,
            "        asm volatile(\"bar.sync %0, 128;\" :: \"r\"(1u + cwi) : \"memory\");".into(),
        );
        w(s, "        #pragma unroll".into());
        w(s, format!("        for (int j = 0; j < {nsf}; j++)"));
        w(s, "        #pragma unroll".into());
        w(s, "        for (int h = 0; h < 2; h++) {".into());
        w(
            s,
            "            unsigned lr = (warp % 4u) * 16u + gq + 8u * h, col = 8u * j + 2u * tq;"
                .into(),
        );
        w(
            s,
            "            unsigned short pv = tang_f2e4x2(sc[j][2 * h], sc[j][2 * h + 1]);".into(),
        );
        w(s, "            asm volatile(\"st.shared.u16 [%0], %1;\" :: \"r\"(pbase + lr * 128u + (((col / 16u) ^ (lr % 8u)) * 16u) + col % 16u), \"h\"(pv) : \"memory\");".into());
        w(s, "        }".into());
        w(s, "        tang_fence_async();".into());
        w(
            s,
            "        asm volatile(\"bar.sync %0, 128;\" :: \"r\"(1u + cwi) : \"memory\");".into(),
        );
        w(s, "        tang_wg_fence();".into());
        w(s, "        #pragma unroll".into());
        w(
            s,
            format!("        for (int kk = 0; kk < {}; kk++)", bj / 32),
        );
        if pl.blk[2].is_none() {
            // one V scale: P·V accumulates into O, scaled at the end
            w(s, "            tang_wgo(&o[0][0], tang_desc(pbase + kk * 32u, 16u, 1024u), tang_desc(vaddr + kk * 32u, 16u, 1024u), 1);".into());
            w(s, "        tang_wg_commit(); tang_wg_wait<0>();".into());
            w(s, "        (void)svp;".into());
        } else {
            w(s, "            tang_wgo(&ot[0][0], tang_desc(pbase + kk * 32u, 16u, 1024u), tang_desc(vaddr + kk * 32u, 16u, 1024u), kk > 0);".into());
            w(s, "        tang_wg_commit(); tang_wg_wait<0>();".into());
            w(s, "        #pragma unroll".into());
            w(s, format!("        for (int j = 0; j < {nof}; j++) {{"));
            w(
                s,
                "            float2 sv = *(const float2*)(svp + 8u * j + 2u * tq);".into(),
            );
            w(s, "            o[j][0] += ot[j][0] * sv.x; o[j][1] += ot[j][1] * sv.y; o[j][2] += ot[j][2] * sv.x; o[j][3] += ot[j][3] * sv.y;".into());
            w(s, "        }".into());
        }
        w(s, "        tang_bar_arrive(bar_e + 8u * q);".into());
        w(s, "    }".into());
        if pl.blk[2].is_none() {
            w(s, "    { float sv0 = sv_g[0];".into());
            w(s, "    #pragma unroll".into());
            w(s, format!("    for (int j = 0; j < {nof}; j++) for (int r = 0; r < 4; r++) o[j][r] *= sv0; }}"));
        }
        // the weights were scaled by 2^shift: the state's max moves with them
        let sh = super::super::precision::FP8_SOFTMAX_SHIFT;
        w(s, format!("    m_row[0] = (m_row[0] - {sh}.0f) * 0.6931471805599453f; m_row[1] = (m_row[1] - {sh}.0f) * 0.6931471805599453f;"));
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Format, Precision, TensorGraph, TensorId};

    fn matmul(m: usize, k: usize, n: usize, weight: bool, f: Format) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let a = g.input(&[m, k]);
        let b = g.input(&if weight { [n, k] } else { [k, n] });
        let bt = if weight {
            g.transpose(b, &[1, 0]).unwrap()
        } else {
            b
        };
        let c = g.matmul(a, bt).unwrap();
        let (mut g, outs) = g.substitute_maps(&[c]);
        g.schedule_auto(&outs);
        g.set_precision(a, Precision::storage(f));
        g.set_precision(b, Precision::storage(f));
        (g, outs[0])
    }

    #[test]
    fn fp8_matmul_lowers_k_major_only() {
        let (g, c) = matmul(256, 512, 384, true, Format::E4M3(128));
        assert!(g.cuda_fp8_lowers(&[c]));
        let src = &g.to_cuda_variant(&[c], Some(0)).kernels[0].source;
        assert!(src.contains("mma (FP8 e4m3xe4m3 wgmma) tile"), "{src}");
        assert!(src.contains("m64n256k32.f32.e4m3.e4m3"));
        // B [k, n] is contiguous along n: FP8 wgmma cannot read it
        let (g, c) = matmul(256, 512, 384, false, Format::E4M3(128));
        assert!(!g.cuda_fp8_lowers(&[c]));
        // a block that straddles k chunks
        let (g, c) = matmul(256, 512, 384, true, Format::E4M3(64));
        assert!(!g.cuda_fp8_lowers(&[c]));
    }

    #[test]
    fn fp8_emulation_is_block_quantization() {
        // eval_emulated of an FP8 matmul = the f64 product of the
        // block-quantized inputs (f32 accumulation is below its noise)
        let (m, k, n) = (8, 256, 4);
        let (g, c) = matmul(m, k, n, true, Format::E4M3(128));
        let a: Vec<f64> = (0..m * k)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 7.0)
            .collect();
        let b: Vec<f64> = (0..n * k)
            .map(|i| ((i * 53 % 89) as f64 - 44.0) / 3.0)
            .collect();
        let got = g.eval_emulated(&[a.clone(), b.clone()], &[c]);
        let (mut aq, mut bq) = (a, b);
        Format::E4M3(128).round_buf(&mut aq);
        Format::E4M3(128).round_buf(&mut bq);
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|kk| aq[i * k + kk] * bq[j * k + kk]).sum();
                let e = got[0][i * n + j];
                assert!(
                    (e - want).abs() <= want.abs() * 1e-5 + 1e-5,
                    "{e} vs {want}"
                );
            }
        }
        // and differs from the exact product by the quantization error
        assert!(g.precision_error(&[aq.clone(), bq.clone()], &[c]).max_rel < 1e-5);
    }
}
