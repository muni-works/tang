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

use super::super::Format;
use super::super::tile::emit_scalar;
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

fn ptx_ty(f: Format) -> &'static str {
    match f {
        Format::E4M3(_) => "e4m3",
        _ => "e5m2",
    }
}

/// `wgmma` m64n`n`k32 over FP8, A and B from shared-memory descriptors
/// (both K-major); `acc` false zeroes the accumulator first.
fn wgmma_fp8(ta: &str, tb: &str, n: usize) -> String {
    let regs = n / 2;
    let outs: Vec<String> = (0..regs).map(|i| format!("%{i}")).collect();
    let cons: Vec<String> = (0..regs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    format!(
        "static __device__ __forceinline__ void tang_wg8(float* d, unsigned long long a, unsigned long long b, int acc) {{\n    asm volatile(\"{{ .reg .pred p; setp.ne.s32 p, %{}, 0; wgmma.mma_async.sync.aligned.m64n{n}k32.f32.{ta}.{tb} {{{}}}, %{}, %{}, p, 1, 1; }}\"\n        : {}\n        : \"l\"(a), \"l\"(b), \"r\"(acc));\n}}\n",
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
            if blk != nn && a.coefs.iter().enumerate().any(|(d, &c)| d != kdim && c % chunk as i64 != 0) {
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

        let addr = |a: &super::super::msl_mma::Addr, row_dim: usize, row: &str, k: &str| -> String {
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
        writeln!(s, "// Auto-generated by tang-compile: mma (FP8 {}x{} wgmma) tile", ptx_ty(fa), ptx_ty(fb)).unwrap();
        writeln!(s, "// block {bm}x{bn}, {n_wg} consumer warpgroups, chunk {chunk}, {stages} stages, block scales {blk_a}/{blk_b}").unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        writeln!(s, "#define TANG_SMEM {smem}").unwrap();
        writeln!(s, "#define TANG_OUT_LEN {}", pts.iter().product::<usize>() * w_out).unwrap();
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
        s.push_str(&wgmma_fp8(ptx_ty(fa), ptx_ty(fb), if overlap { bn / if bn > 128 { 4 } else { 2 } } else { bn }));
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
        writeln!(s, "    unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;").unwrap();
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        // grouped rasterization: blocks of `grp` tile rows walk their
        // columns together, so the tiles resident at once share B in L2
        let grp = 8usize.min(tiles_y);
        writeln!(s, "    unsigned pid = tile % {}u;", tiles_x * tiles_y).unwrap();
        writeln!(s, "    unsigned first_y = (pid / {}u) * {grp}u;", grp * tiles_x).unwrap();
        writeln!(s, "    unsigned gsize = min({tiles_y}u - first_y, {grp}u);").unwrap();
        writeln!(s, "    unsigned tile_y = first_y + (pid % {}u) % gsize;", grp * tiles_x).unwrap();
        writeln!(s, "    unsigned tile_x = (pid % {}u) / gsize;", grp * tiles_x).unwrap();
        writeln!(s, "    unsigned lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    unsigned p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    (void)lead;").unwrap();
        writeln!(s, "    unsigned row0 = tile_y * {bm}u, col0 = tile_x * {bn}u;").unwrap();
        writeln!(s, "    unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem) + 1023u) & ~1023u;").unwrap();
        writeln!(s, "    const float* smf = (const float*)((const char*)smem + (sbase - (unsigned)__cvta_generic_to_shared(smem)));").unwrap();
        writeln!(s, "    unsigned bar_f = sbase + {bar_off}u, bar_e = bar_f + {}u;", 8 * stages).unwrap();
        writeln!(s, "    if (tid == 0u) {{").unwrap();
        writeln!(s, "        for (unsigned q = 0u; q < {stages}u; q++) {{ tang_bar_init(bar_f + 8u * q, 128u); tang_bar_init(bar_e + 8u * q, {}u); }}", 128 * n_wg).unwrap();
        writeln!(s, "        asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    __syncthreads();").unwrap();
        writeln!(s, "    const float* sa_g = (const float*)((const unsigned char*)b{ba} + {sa_off}u);").unwrap();
        writeln!(s, "    const float* sb_g = (const float*)((const unsigned char*)b{bb} + {sb_off}u);").unwrap();

        // producer
        writeln!(s, "    if (warp < 4u) {{").unwrap();
        writeln!(s, "        tang_setmaxnreg_dec<40>();").unwrap();
        // thread t copies 16-byte column c16 = t % 8 of rows t / 8 + 16·i:
        // its global row pointers step by 16 rows, its swizzled shared
        // offsets by 2 KB, and the swizzle (row mod 8) is fixed
        let k_ragged = k_total % chunk != 0;
        let mut copies = String::new();
        if dbg != "noload" {
            for (side, rows, ok_n, base) in [("a", bm, m % bm == 0, 0usize), ("b", bn, n % bn == 0, a_bytes)] {
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
                    let okx = if ok.is_empty() { "true".to_string() } else { ok.join(" && ") };
                    let step = if side == "a" { aa.coefs[dy] } else { ab.coefs[dx] } * 16 * i as i64;
                    writeln!(copies, "            {{ bool ok = {okx}; tang_cp16s(st + {}u + sw + {}u, p{side} + (ok ? {step}ll + (long long)k0 : 0ll), ok); }}", base, 2048 * i).unwrap();
                }
            }
        }
        writeln!(s, "        unsigned r0 = tid / 8u, c16 = tid % 8u, sw = r0 * 128u + ((c16 ^ (r0 % 8u)) * 16u);").unwrap();
        writeln!(s, "        const unsigned char* pa = b{ba} + ({}) + c16 * 16u;", addr(aa, dy, &format!("min(row0 + r0, {}u)", m - 1), "0")).unwrap();
        writeln!(s, "        const unsigned char* pb = b{bb} + ({}) + c16 * 16u;", addr(ab, dx, &format!("min(col0 + r0, {}u)", n - 1), "0")).unwrap();
        writeln!(s, "        (void)pa; (void)pb; (void)sw;").unwrap();
        writeln!(s, "        for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
        writeln!(s, "            unsigned q = c % {stages}u;").unwrap();
        writeln!(s, "            tang_bar_wait(bar_e + 8u * q, ((c / {stages}u) & 1u) ^ 1u);").unwrap();
        writeln!(s, "            unsigned st = sbase + q * {stage_bytes}u;").unwrap();
        writeln!(s, "            unsigned k0 = c * {chunk}u;").unwrap();
        s.push_str(&copies);
        // the chunk's scales: one per A row, one per B column
        let sidx = |a: &super::super::msl_mma::Addr, blk: usize, nn: usize, row_dim: usize, row: &str| {
            if blk == nn {
                "0ll".to_string()
            } else {
                format!("({} / {blk}ll)", addr(a, row_dim, row, "k0"))
            }
        };
        writeln!(s, "            for (unsigned r = tid; r < {}u; r += 128u) {{", bm + bn).unwrap();
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
        let nsub = if !overlap { 1 } else if bn > 128 { 4 } else { 2 };
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
            writeln!(s, "{ind}for (unsigned kb = 0u; kb < {chunk}u; kb += 32u) {{").unwrap();
            writeln!(s, "{ind}    unsigned long long da = tang_desc(st + cwi * {}u + kb, 16u, 1024u);", 64 * 128).unwrap();
            writeln!(s, "{ind}    unsigned long long db = tang_desc(st + {}u + kb, 16u, 1024u);", a_bytes + sl * nw * 128).unwrap();
            if dbg != "nomma" {
                writeln!(s, "{ind}    tang_wg8({t}, da, db, kb != 0u);").unwrap();
            }
            writeln!(s, "{ind}}}").unwrap();
            writeln!(s, "{ind}tang_wg_commit();").unwrap();
        };
        if tensorwise {
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned q = c % {stages}u;").unwrap();
            writeln!(s, "        tang_bar_wait(bar_f + 8u * q, (c / {stages}u) & 1u); tang_fence_async();").unwrap();
            writeln!(s, "        unsigned st = sbase + q * {stage_bytes}u;").unwrap();
            writeln!(s, "        tang_wg_fence();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (unsigned kb = 0u; kb < {chunk}u; kb += 32u) {{").unwrap();
            writeln!(s, "            unsigned long long da = tang_desc(st + cwi * {}u + kb, 16u, 1024u);", 64 * 128).unwrap();
            writeln!(s, "            unsigned long long db = tang_desc(st + {a_bytes}u + kb, 16u, 1024u);").unwrap();
            writeln!(s, "            tang_wg8(acc, da, db, 1);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_wg_commit();").unwrap();
            writeln!(s, "        tang_wg_wait<1>();").unwrap();
            writeln!(s, "        if (c > 0u) tang_bar_arrive(bar_e + 8u * ((c - 1u) % {stages}u));").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    tang_wg_wait<0>();").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int i = 0; i < {regs}; i++) asm volatile(\"\" : \"+f\"(acc[i]) :: \"memory\");").unwrap();
            writeln!(s, "    float sab = sa_g[0] * sb_g[0];").unwrap();
        } else if !overlap {
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned q = c % {stages}u;").unwrap();
            writeln!(s, "        tang_bar_wait(bar_f + 8u * q, (c / {stages}u) & 1u); tang_fence_async();").unwrap();
            writeln!(s, "        unsigned st = sbase + q * {stage_bytes}u;").unwrap();
            issue(&mut s, "        ", "t0", 0);
            writeln!(s, "        tang_wg_wait<0>();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int i = 0; i < {tr}; i++) asm volatile(\"\" : \"+f\"(t0[i]) :: \"memory\");").unwrap();
            writeln!(s, "        const float* scl = smf + (q * {stage_bytes}u + {}u) / 4u;", a_bytes + b_bytes).unwrap();
            writeln!(s, "        float s_lo = scl[lr0], s_hi = scl[lr0 + 8u];").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int j = 0; j < {}; j++) {{", bn / 8).unwrap();
            writeln!(s, "            float2 sb = *(const float2*)(scl + {bm}u + 8u * j + 2u * tq);").unwrap();
            writeln!(s, "            acc[4 * j + 0] += t0[4 * j + 0] * s_lo * sb.x;").unwrap();
            writeln!(s, "            acc[4 * j + 1] += t0[4 * j + 1] * s_lo * sb.y;").unwrap();
            writeln!(s, "            acc[4 * j + 2] += t0[4 * j + 2] * s_hi * sb.x;").unwrap();
            writeln!(s, "            acc[4 * j + 3] += t0[4 * j + 3] * s_hi * sb.y;").unwrap();
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
            writeln!(s, "        const float* scl = smf + (q * {stage_bytes}u + {}u) / 4u;", a_bytes + b_bytes).unwrap();
            writeln!(s, "        float s_lo = scl[lr0], s_hi = scl[lr0 + 8u];").unwrap();
            for sl in 0..nsub {
                let (cur, nxt) = if sl % 2 == 0 { ("t0", "t1") } else { ("t1", "t0") };
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
                writeln!(s, "            float2 sb = *(const float2*)(scl + {}u + 8u * j + 2u * tq);", bm + sl * nw).unwrap();
                let o = sl * nw / 2;
                writeln!(s, "            acc[{o} + 4 * j + 0] += {cur}[4 * j + 0] * s_lo * sb.x;").unwrap();
                writeln!(s, "            acc[{o} + 4 * j + 1] += {cur}[4 * j + 1] * s_lo * sb.y;").unwrap();
                writeln!(s, "            acc[{o} + 4 * j + 2] += {cur}[4 * j + 2] * s_hi * sb.x;").unwrap();
                writeln!(s, "            acc[{o} + 4 * j + 3] += {cur}[4 * j + 3] * s_hi * sb.y;").unwrap();
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
        writeln!(s, "        unsigned p{dx} = col0 + 8u * j + 2u * tq + (r % 2);").unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        writeln!(s, "            float red0 = acc[4 * j + r]{};", if tensorwise { " * sab" } else { "" }).unwrap();
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
                writeln!(s, "            {}", float_literals(&line.replacen("let ", "float ", 1))).unwrap();
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
