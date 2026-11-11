//! Native CUDA lowering of matmul-shaped tiles: tensor cores via `mma`.
//!
//! The same bilinear tile the Metal lowering recognizes
//! (`bilinear_tile`) emitted as CUDA C++: an [`MmaShape`] block (64×64 on
//! four warps up to 128×128 on eight), operands staged through
//! double-buffered shared memory from any affine layout (`cp.async`,
//! 16-byte where the layout allows), `mma.sync` m16n8k8 TF32 fragments
//! with fp32 accumulation, and the comp's epilogue run on the
//! accumulator registers. Flash attention uses the same fragments
//! ([`FlashShape`]).
//!
//! tang-gpu runs on wgpu, which cannot dispatch CUDA; this emitter
//! produces source for an external harness (`scripts/cuda_bench.cu`,
//! compiled with nvcc) that times the kernel against cuBLAS on the same
//! device. The kernel takes the read buffers in `reads` order, then
//! `out`; `TANG_TILES` blocks of `TANG_THREADS` threads.
use std::fmt::Write;

use super::msl_mma::Bilinear;
use super::tile::emit_scalar;
use super::wgsl::{Target, WgslProgram};
use super::{TensorGraph, TensorId};

#[path = "tensor_cuda_attn_bwd.rs"]
mod attn_bwd;

thread_local! {
    static CUDA_SM: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The compute capability (`90` for sm_90a) the lowerings may assume, 0
/// when unknown. Every variant of a program must share one structure, so
/// a lowering that only fits one arch (the d=256 attention backward needs
/// Hopper's shared memory) asks here instead of guessing from a variant.
pub fn cuda_sm() -> u32 {
    CUDA_SM.with(|c| c.get())
}

/// Run `f` with [`cuda_sm`] reporting `sm`.
pub fn with_cuda_sm<T>(sm: u32, f: impl FnOnce() -> T) -> T {
    let prev = CUDA_SM.with(|c| c.replace(sm));
    struct Reset(u32);
    impl Drop for Reset {
        fn drop(&mut self) {
            CUDA_SM.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(prev);
    f()
}
#[path = "tensor_cuda_fp8.rs"]
mod fp8;
pub use fp8::SM90_FP8_SHAPES;
use fp8::{Fp8FlashCtx, FP8_FLASH_PRELUDE};

/// A tensor-core tile's block shape: a `bm × bn` output block per thread
/// block of `warps_m × warps_n` warps (each a `bm/warps_m × bn/warps_n`
/// warp tile of 16×8 `mma` fragments), the reduction staged `chunk` at a
/// time through a ring of `stages` shared-memory buffers (`cp.async`
/// where the operand is a plain read, so the next chunks' copies overlap
/// this chunk's math), and split-K up to about `split_target` blocks
/// (1: never split).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaShape {
    pub bm: usize,
    pub bn: usize,
    pub warps_m: usize,
    pub warps_n: usize,
    pub chunk: usize,
    pub stages: usize,
    pub split_target: usize,
    /// Hopper (sm_90a): the block's warpgroups (four warps each, stacked
    /// along m) issue `wgmma` m64n`bn`k16 from 128-byte-swizzled stages
    /// instead of `mma.sync` fragments. Half operands only.
    pub wgmma: bool,
}

impl MmaShape {
    pub const fn new(bm: usize, bn: usize, warps_m: usize, warps_n: usize) -> Self {
        Self {
            bm,
            bn,
            warps_m,
            warps_n,
            chunk: 32,
            stages: 3,
            split_target: 100,
            wgmma: false,
        }
    }
    const fn wg(mut self) -> Self {
        self.wgmma = true;
        self
    }
    const fn with(mut self, chunk: usize, stages: usize, split_target: usize) -> Self {
        self.chunk = chunk;
        self.stages = stages;
        self.split_target = split_target;
        self
    }
    fn threads(&self) -> usize {
        32 * self.warps_m * self.warps_n
    }
    /// Row strides of the staged operands. Fragments are read as `float2`
    /// (see the k and column pairing in the kernel): A `[bm][chunk]` as 8
    /// rows × 4 pairs per half-warp-pair (stride ≡ 8 mod 32 is
    /// conflict-free), B `[chunk][bn]` as 4 rows × 4 pairs per half warp
    /// (≡ 4 mod 32). Both stay multiples of 4 floats for 16-byte
    /// `cp.async`.
    fn lda(&self) -> usize {
        self.chunk + 8
    }
    fn ldb(&self) -> usize {
        self.bn + 4
    }
}

/// The block shapes a tuner chooses between, per kernel; the first is the
/// default. 64×64 on four warps fills the device at the model's sizes;
/// the larger blocks halve the staging per flop where there are enough
/// output tiles; split-K only pays when tiles are very few.
pub const MMA_SHAPES: [MmaShape; 7] = [
    MmaShape::new(64, 64, 2, 2),
    MmaShape::new(64, 64, 2, 2).with(32, 3, 1),
    MmaShape::new(64, 64, 2, 2).with(16, 4, 1),
    MmaShape::new(64, 64, 2, 2).with(32, 2, 246),
    MmaShape::new(128, 64, 2, 2),
    MmaShape::new(128, 128, 2, 4).with(32, 2, 100),
    // 32 rows on two warps: twice the blocks where tiles are few (the
    // attention-backward tiles at short sequences)
    MmaShape::new(32, 64, 1, 2),
];

/// Hopper contraction tiles (`wgmma`): `Program::tuned` tries these past
/// [`MMA_SHAPES`] on an sm_90a device. A tile that cannot take one
/// (f32 operands, a ragged block) falls back to the `mma.sync` shape.
pub const SM90_MMA_SHAPES: [MmaShape; 5] = [
    MmaShape::new(128, 128, 8, 1).with(32, 4, 100).wg(),
    MmaShape::new(128, 256, 8, 1).with(32, 3, 100).wg(),
    MmaShape::new(64, 128, 4, 1).with(32, 4, 100).wg(),
    MmaShape::new(128, 64, 8, 1).with(32, 4, 100).wg(),
    MmaShape::new(64, 64, 4, 1).with(32, 4, 246).wg(),
];

/// A flash-attention tile's shape: `bi` queries per folded head per
/// block (0: chosen from the problem so the device fills), `bj` keys per
/// warp group per pipeline step, a `cp.async` ring of `stages` key/value
/// buffers, and `kw` warp groups splitting each step's keys (their
/// softmax states merged at the end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashShape {
    pub bi: usize,
    pub bj: usize,
    pub stages: usize,
    pub kw: usize,
    /// Hopper schedule: warp-specialized `wgmma` (`None`: `mma.sync`).
    pub sm90: Option<Sm90Flash>,
}

/// The Hopper flash schedule: one producer warpgroup staging Q, K and V
/// into 128-byte-swizzled shared memory (16-byte `cp.async`, a ring of
/// mbarriers) and `consumers` warpgroups of 64 query rows each on `wgmma`
/// m64nNk16, registers rebalanced with `setmaxnreg`. Half-precision
/// operands only (bf16 or f16); the keys per step are the shape's `bj`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90Flash {
    pub consumers: usize,
    pub producer_regs: usize,
    pub consumer_regs: usize,
}

/// The flash shapes a tuner chooses between, paired with [`MMA_SHAPES`]
/// variants by index modulo their count; the first is the default.
pub const FLASH_SHAPES: [FlashShape; 6] = [
    FlashShape {
        bi: 0,
        bj: 32,
        stages: 2,
        kw: 2,
        sm90: None,
    },
    FlashShape {
        bi: 0,
        bj: 64,
        stages: 2,
        kw: 1,
        sm90: None,
    },
    FlashShape {
        bi: 0,
        bj: 16,
        stages: 3,
        kw: 2,
        sm90: None,
    },
    FlashShape {
        bi: 0,
        bj: 16,
        stages: 2,
        kw: 4,
        sm90: None,
    },
    FlashShape {
        bi: 32,
        bj: 32,
        stages: 2,
        kw: 2,
        sm90: None,
    },
    FlashShape {
        bi: 0,
        bj: 32,
        stages: 3,
        kw: 1,
        sm90: None,
    },
];

/// Hopper flash shapes (sm_90a): `Program::tuned` adds these as
/// variants after [`MMA_SHAPES`] on a Hopper device.
pub const SM90_FLASH_SHAPES: [FlashShape; 4] = [
    FlashShape {
        bi: 0,
        bj: 128,
        stages: 2,
        kw: 1,
        sm90: Some(Sm90Flash {
            consumers: 2,
            producer_regs: 40,
            consumer_regs: 232,
        }),
    },
    FlashShape {
        bi: 0,
        bj: 64,
        stages: 3,
        kw: 1,
        sm90: Some(Sm90Flash {
            consumers: 2,
            producer_regs: 40,
            consumer_regs: 232,
        }),
    },
    FlashShape {
        bi: 0,
        bj: 128,
        stages: 2,
        kw: 1,
        sm90: Some(Sm90Flash {
            consumers: 1,
            producer_regs: 40,
            consumer_regs: 232,
        }),
    },
    FlashShape {
        bi: 0,
        bj: 64,
        stages: 2,
        kw: 1,
        sm90: Some(Sm90Flash {
            consumers: 2,
            producer_regs: 40,
            consumer_regs: 232,
        }),
    },
];

/// The PTX helpers the tensor-core kernels share: TF32 rounding, `mma`
/// m16n8k8, and `cp.async` (zero-filling when the source is out of range).
/// Hopper FP8 flash shapes (128 keys per step: one 128-byte row of Vᵀ
/// and of P per row).
pub const SM90_FP8_FLASH_SHAPES: [FlashShape; 4] = {
    const fn sh(consumers: usize, stages: usize) -> FlashShape {
        FlashShape {
            bi: 0,
            bj: 128,
            stages,
            kw: 1,
            sm90: Some(Sm90Flash {
                consumers,
                producer_regs: 40,
                consumer_regs: 232,
            }),
        }
    }
    [sh(2, 2), sh(2, 3), sh(2, 4), sh(1, 4)]
};

const MMA_PRELUDE: &str = r#"static __device__ __forceinline__ unsigned tang_tf32(float x) {
    // round to nearest, ties away (cvt.rna): the mma reads only the top
    // 19 bits, so adding half a TF32 ulp is the whole conversion
    return __float_as_uint(x) + 0x1000u;
}
static __device__ __forceinline__ void tang_mma(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static __device__ __forceinline__ void tang_cp4(float* dst, const float* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(ok ? 4 : 0));
}
static __device__ __forceinline__ void tang_cp16(float* dst, const float* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}
static __device__ __forceinline__ void tang_cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> static __device__ __forceinline__ void tang_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
"#;

/// `mma` m16n8k16 with f16 or bf16 operands (fp32 accumulation) and a
/// 16-byte `cp.async` of halves, for tiles whose operands are stored in
/// half precision.
fn mma16_prelude(f: super::Format) -> String {
    let ty = if f == super::Format::BF16 {
        "bf16"
    } else {
        "f16"
    };
    format!(
        r#"static __device__ __forceinline__ void tang_mma16(float* c, const unsigned* a, const unsigned* b) {{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{ty}.{ty}.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}}
static __device__ __forceinline__ unsigned tang_pack2(float lo, float hi) {{ unsigned d; asm("cvt.rn.{ty}x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }}
static __device__ __forceinline__ void tang_cp16h(unsigned short* dst, const unsigned short* src, bool ok) {{
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}}
"#
    )
}

/// `x` as a C float literal body (append `f`).
fn f32_lit(x: f64) -> String {
    format!("{:e}", x as f32)
}

/// Hopper (sm_90a) helpers (`.b64` is split into string pieces: the
/// program's canonical renaming would take `b64` for a buffer): mbarriers, the async-proxy fence, register
/// reallocation between warpgroups, `cp.async` to a shared address, and
/// `wgmma` shared-memory descriptors (128-byte swizzle).
const SM90_PRELUDE: &str = r#"static __device__ __forceinline__ void tang_bar_init(unsigned a, unsigned n) { asm volatile("mbarrier.init.shared::cta." "b" "64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
static __device__ __forceinline__ void tang_bar_arrive(unsigned a) { asm volatile("{ .reg ." "b" "64 st; mbarrier.arrive.shared::cta." "b" "64 st, [%0]; }" :: "r"(a) : "memory"); }
static __device__ __forceinline__ void tang_bar_wait(unsigned a, unsigned ph) {
    asm volatile("{ .reg .pred p; LAB_WAIT: mbarrier.try_wait.parity.shared::cta." "b" "64 p, [%0], %1; @!p bra LAB_WAIT; }" :: "r"(a), "r"(ph) : "memory");
}
static __device__ __forceinline__ void tang_fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int N> static __device__ __forceinline__ void tang_setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> static __device__ __forceinline__ void tang_setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
static __device__ __forceinline__ void tang_cp16s(unsigned d, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
static __device__ __forceinline__ void tang_wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
static __device__ __forceinline__ void tang_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> static __device__ __forceinline__ void tang_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
static __device__ __forceinline__ unsigned long long tang_desc(unsigned addr, unsigned lbo, unsigned sbo) {
    return (unsigned long long)((addr & 0x3FFFFu) >> 4) | ((unsigned long long)(lbo >> 4) << 16)
        | ((unsigned long long)(sbo >> 4) << 32) | (1ull << 62);
}
"#;

/// `wgmma` m64n`n`k16 with f32 accumulators: `tang_wg_ss{n}` (A and B
/// from shared-memory descriptors, both K-major; `acc` false zeroes the
/// accumulator) and `tang_wg_rs{n}` (A from four registers, B MN-major).
fn sm90_wgmma(ty: &str, n: usize, rs: bool) -> String {
    let regs = n / 2;
    let outs: Vec<String> = (0..regs).map(|i| format!("%{i}")).collect();
    let cons: Vec<String> = (0..regs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    if rs {
        format!(
            "static __device__ __forceinline__ void tang_wg_rs{n}(float* d, const unsigned* a, unsigned long long b) {{\n    asm volatile(\"{{ .reg .pred p; setp.ne.s32 p, %{}, 0; wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{ty}.{ty} {{{}}}, {{%{}, %{}, %{}, %{}}}, %{}, p, 1, 1, 1; }}\"\n        : {}\n        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(b), \"r\"(1));\n}}\n",
            regs + 5,
            outs.join(", "),
            regs,
            regs + 1,
            regs + 2,
            regs + 3,
            regs + 4,
            cons.join(", ")
        )
    } else {
        format!(
            "static __device__ __forceinline__ void tang_wg_ss{n}(float* d, unsigned long long a, unsigned long long b, bool acc) {{\n    asm volatile(\"{{ .reg .pred p; setp.ne.s32 p, %{}, 0; wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{ty}.{ty} {{{}}}, %{}, %{}, p, 1, 1, 0, 0; }}\"\n        : {}\n        : \"l\"(a), \"l\"(b), \"r\"((int)acc));\n}}\n",
            regs + 2,
            outs.join(", "),
            regs,
            regs + 1,
            cons.join(", ")
        )
    }
}

/// `wgmma` m64n`n`k16, A and B from shared-memory descriptors, with the
/// transpose immediates (`ta`: A MN-major, `tb`: B MN-major):
/// `tang_wg_ss{n}_{ta}{tb}`, accumulating.
fn sm90_wgmma_ss(ty: &str, n: usize, ta: bool, tb: bool) -> String {
    let regs = n / 2;
    let outs: Vec<String> = (0..regs).map(|i| format!("%{i}")).collect();
    let cons: Vec<String> = (0..regs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    let (ta, tb) = (ta as u8, tb as u8);
    format!(
        "static __device__ __forceinline__ void tang_wg_ss{n}_{ta}{tb}(float* d, unsigned long long a, unsigned long long b) {{\n    asm volatile(\"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{ty}.{ty} {{{}}}, %{}, %{}, 1, 1, 1, {ta}, {tb};\"\n        : {}\n        : \"l\"(a), \"l\"(b));\n}}\n",
        outs.join(", "),
        regs,
        regs + 1,
        cons.join(", ")
    )
}

impl TensorGraph {
    /// CUDA source for `root` as a tensor-core tile in the default block
    /// shape, or `None` when it is not matmul-shaped.
    pub fn cuda_mma_kernel(&self, root: usize, reads: &[usize]) -> Option<String> {
        self.cuda_mma_kernel_shaped(root, reads, MMA_SHAPES[0])
    }

    /// CUDA source for `root` as a tensor-core tile of block shape `sh`,
    /// or `None` when it is not matmul-shaped (or does not fit). Several
    /// terms (a sum of products) and several outputs (a tuple-valued
    /// reduction such as merged gate/up projections) share one kernel:
    /// distinct operands are staged once, each output accumulates its own
    /// fragments, and the epilogue runs on the accumulator registers.
    pub fn cuda_mma_kernel_shaped(
        &self,
        root: usize,
        reads: &[usize],
        sh: MmaShape,
    ) -> Option<String> {
        let c = self.comp_at(root)?;
        let Bilinear {
            red_node,
            epilogue,
            outs,
            pts,
            dx,
            dy,
            others,
            m,
            n,
            k_total,
            kdim,
            scales,
            ..
        } = self.bilinear_tile(root, reads)?;
        let (bm, bn, chunk) = (sh.bm, sh.bn, sh.chunk);
        let threads = sh.threads();
        // fp32 accumulation only (an f16 accumulator stays transpiled)
        if self.nodes[red_node].precision.accum != super::Format::F32 {
            return None;
        }
        let fmt = |j: usize| self.nodes[j].precision.storage;
        // fp8 operands have their own tile (tensor_cuda_fp8)
        if reads.iter().chain([&root]).any(|&j| fmt(j).is_fp8()) {
            return None;
        }
        // buffer j's element at `idx`, as f32
        let ld = |j: usize, idx: &str| -> String {
            match fmt(j) {
                super::Format::F32 => format!("b{j}[{idx}]"),
                super::Format::F16 => format!("tang_h2f(b{j}[{idx}])"),
                super::Format::BF16 => format!("tang_b2f(b{j}[{idx}])"),
                super::Format::Q4K
                | super::Format::Q5K
                | super::Format::Q6K
                | super::Format::Q8x32
                | super::Format::Q8x32F
                | super::Format::Fp8B32
                | super::Format::Mxfp4 => {
                    unreachable!("block formats lower only through the int8 MMA kernel")
                }
                _ => unreachable!("fp8 has its own tile"),
            }
        };
        let elem = |f: super::Format| {
            if f == super::Format::F32 {
                "float"
            } else {
                "unsigned short"
            }
        };
        // per warp: fm × fn fragments of 16×8 (wgmma: per warpgroup, fm
        // m64 blocks of bn columns)
        let wg = sh.wgmma;
        let n_wg = (sh.warps_m * sh.warps_n) / 4;
        let (wtm, wtn) = if wg {
            (bm / n_wg.max(1), bn)
        } else {
            (bm / sh.warps_m, bn / sh.warps_n)
        };
        let (fm, fnn) = if wg {
            (wtm / 64, bn / 8)
        } else {
            (wtm / 16, wtn / 8)
        };
        if wg
            && (n_wg == 0
                || !(sh.warps_m * sh.warps_n).is_multiple_of(4)
                || bm % (64 * n_wg) != 0
                || bn % 8 != 0
                || bn > 256)
        {
            return None;
        }
        if fm == 0 || wtm % 16 != 0 || wtn % 16 != 0 || chunk % 8 != 0 {
            return None;
        }
        let n_outs = outs.len();
        let typed = reads
            .iter()
            .chain([&root])
            .any(|&j| fmt(j) != super::Format::F32);
        // distinct operand stages; each (output, term) is an (A, B) pair
        let mut a_stages: Vec<(usize, super::msl_mma::Addr, Option<usize>)> = Vec::new();
        let mut b_stages: Vec<(usize, super::msl_mma::Addr, Option<usize>)> = Vec::new();
        let mut uses: Vec<Vec<(usize, usize)>> = Vec::new();
        for terms in &outs {
            let mut u = Vec::new();
            for t in terms {
                let ai = a_stages
                    .iter()
                    .position(|(b, a, _)| *b == t.buf_a && *a == t.addr_a)
                    .unwrap_or_else(|| {
                        a_stages.push((t.buf_a, t.addr_a.clone(), t.map_a));
                        a_stages.len() - 1
                    });
                let bi = b_stages
                    .iter()
                    .position(|(b, a, _)| *b == t.buf_b && *a == t.addr_b)
                    .unwrap_or_else(|| {
                        b_stages.push((t.buf_b, t.addr_b.clone(), t.map_b));
                        b_stages.len() - 1
                    });
                u.push((ai, bi));
            }
            uses.push(u);
        }
        // Operands stored in f16 (bf16) run on m16n8k16 f16 (bf16)
        // fragments, staged as halves; f32 operands on TF32. Plain
        // operands must agree; a map operand is rounded to the format.
        let plain: Vec<super::Format> = a_stages
            .iter()
            .chain(&b_stages)
            .filter(|st| st.2.is_none())
            .map(|st| fmt(st.0))
            .collect();
        // f32 operands beside half ones (fp32 master weights) are
        // converted to the half format while staging
        let halves: Vec<super::Format> = plain
            .iter()
            .copied()
            .filter(|&f| f != super::Format::F32)
            .collect();
        // no half plain operand, but the maps read half buffers (dV =
        // P̃ᵀ·(dO/l) beside bf16 Q, K, dO): the tile runs in that format,
        // the f32 operands rounded while staging (TANG_HALF_MAPS=0: TF32)
        let map_half = || {
            if std::env::var("TANG_HALF_MAPS").ok().as_deref() == Some("0")
                || !a_stages.iter().chain(&b_stages).any(|st| st.2.is_some())
            {
                return None;
            }
            let fs: Vec<super::Format> = reads
                .iter()
                .map(|&j| fmt(j))
                .filter(|&f| f != super::Format::F32)
                .collect();
            let f = *fs.first()?;
            fs.iter().all(|&g| g == f).then_some(f)
        };
        let half = match halves.first() {
            None => map_half(),
            Some(&f) if halves.iter().any(|&g| g != f) => return None,
            Some(&f) => Some(f),
        };
        // Half elements are half the bytes and a k16 step twice the work
        // of a TF32 k8 step: the k chunk doubles (TANG_HALF_CHUNK=1 keeps
        // it), so a stage holds the same bytes and every barrier covers
        // as many mma as on TF32.
        let chunk = match half {
            Some(_) => {
                chunk
                    * std::env::var("TANG_HALF_CHUNK")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(2usize)
            }
            None => chunk,
        };
        if half.is_some() && !chunk.is_multiple_of(16) {
            return None;
        }
        if wg && (half.is_none() || !chunk.is_multiple_of(64)) {
            return None;
        }
        let esz = if half.is_some() { 2 } else { 4 };
        let (lda, ldb) = match half {
            Some(_) => (chunk + 8, sh.bn + 8),
            None => (sh.lda(), sh.ldb()),
        };
        // Half tiles stage an operand in the layout its global reads are
        // contiguous in. A B operand contiguous along k (a weight stored
        // [out, in]) is staged [n][k]: m16n8k16's B fragments pair k, so
        // each register is one 32-bit load. An A operand contiguous along
        // m (a transposed activation in a weight gradient) is staged
        // [k][m], its fragment rows paired so that one 32-bit word holds
        // rows gq and gq + 8 (split with byte permutes). Every plain
        // operand on the side must agree; the epilogue's row and column
        // maps follow. TF32 tiles keep [m][k] and [k][n].
        let all_plain = |stages: &[(usize, super::msl_mma::Addr, Option<usize>)],
                         f: &dyn Fn(&super::msl_mma::Addr) -> bool| {
            let mut plain = stages.iter().filter(|st| st.2.is_none()).peekable();
            plain.peek().is_some() && plain.all(|st| f(&st.1))
        };
        let b_col =
            half.is_some() && all_plain(&b_stages, &|a| a.coefs[kdim] == 1 && a.coefs[dx] != 1);
        let a_pair =
            half.is_some() && all_plain(&a_stages, &|a| a.coefs[dy] == 1 && a.coefs[kdim] != 1);
        // shared row strides: an odd number of 16-byte units ([n][k] B,
        // like A), ≡ 8 mod 16 halves for the row-paired [k][m] A (like B)
        let ldat = bm + 8;
        let ldbt = chunk + 8;
        // accumulators: n_outs × fm × fnn × 4 floats per thread
        if n_outs * fm * fnn * 4 > 128 {
            return None;
        }
        // wgmma: an MN-major operand is staged in 64-wide column groups
        if wg && ((a_pair && bm % 64 != 0) || (!b_col && bn % 64 != 0)) {
            return None;
        }
        // contractions nested in a map operand (S = Q·Kᵀ in dS): per
        // staged operand, A stages first, then B
        let rc = self.comp_at(red_node)?;
        let rank = rc.rank();
        let nwarps = sh.warps_m * sh.warps_n;
        let mut nested: Vec<Vec<Nested>> = Vec::new();
        let mut nest_regs = 0;
        for (side_a, stages) in [(true, &a_stages), (false, &b_stages)] {
            for (_, _, map) in stages.iter() {
                let (pdim, block) = if side_a { (dy, bm) } else { (dx, bn) };
                let plan = match map {
                    Some(v) => self.nested_plan(rc, rank, *v, pdim, kdim, reads)?,
                    None => Vec::new(),
                };
                if !plan.is_empty()
                    && (!block.is_multiple_of(nwarps) || !(block / nwarps).is_multiple_of(16))
                {
                    return None;
                }
                let nfm = block / nwarps / 16;
                for nst in &plan {
                    // the invariant side's fragments, and the accumulators
                    nest_regs += nst.terms.len() * nfm * (nst.e / 8) * 4 + nfm * (chunk / 8) * 4;
                }
                nested.push(plan);
            }
        }
        if nest_regs
            > std::env::var("TANG_NEST_REGS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(192)
        {
            return None;
        }
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let tiles_x = n.div_ceil(bn);
        let tiles_y = m.div_ceil(bm);
        let n_base = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();
        let splits = split_k(n_base, k_total, chunk, sh.split_target);
        let n_tiles = n_base * splits;
        let w_out = c.width();
        // wgmma stages are unpadded (the swizzle spreads the banks)
        let a_floats = if wg {
            bm * chunk
        } else if a_pair {
            chunk * ldat
        } else {
            bm * lda
        };
        let b_floats = if wg {
            bn * chunk
        } else if b_col {
            bn * ldbt
        } else {
            chunk * ldb
        };
        let stage_floats = a_stages.len() * a_floats + b_stages.len() * b_floats;
        let smem_floats = sh.stages * stage_floats;
        if wg && sh.stages < 2 {
            return None;
        }
        // dynamic shared memory, within the 3090's 99 KB per block (a
        // Hopper block: 227 KB, less the 1 KB swizzle alignment)
        let smem_cap = if wg { 226 * 1024 } else { 99 * 1024 };
        if smem_floats * esz > smem_cap {
            // Many distinct operands (a multi-output, multi-term tile such
            // as the gate/up backward): shrink the ring, then the chunk,
            // rather than leave the shape without a lowering.
            let mut fit = sh;
            if sh.stages > 2 {
                fit.stages -= 1;
            } else if sh.chunk > 16 {
                fit.chunk /= 2;
            } else {
                return None;
            }
            return self.cuda_mma_kernel_shaped(root, reads, fit);
        }

        let mut s = String::new();
        let kind = match (half, wg) {
            (None, _) => "TF32",
            (Some(super::Format::BF16), false) => "BF16",
            (Some(_), false) => "F16",
            (Some(super::Format::BF16), true) => "BF16 wgmma",
            (Some(_), true) => "F16 wgmma",
        };
        writeln!(s, "// Auto-generated by tang-compile: mma ({kind}) tile").unwrap();
        writeln!(
            s,
            "// block {bm}x{bn}, {}x{} warps, chunk {chunk}, {} stages",
            sh.warps_m, sh.warps_n, sh.stages
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {threads}").unwrap();
        writeln!(
            s,
            "#define TANG_SMEM {}",
            smem_floats * esz + if wg { 1024 } else { 0 }
        )
        .unwrap();
        writeln!(s, "#define TANG_OUT_LEN {}", numel_pts(&pts) * w_out).unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        if typed {
            s.push_str(HALF_PRELUDE);
        }
        if let Some(f) = half {
            s.push_str(&mma16_prelude(f));
            if wg {
                s.push_str(SM90_PRELUDE);
                let ty = if f == super::Format::BF16 {
                    "bf16"
                } else {
                    "f16"
                };
                s.push_str(&sm90_wgmma_ss(ty, bn, a_pair, !b_col));
            }
        }
        if splits > 1 {
            // per-split partial blocks, and an arrival counter per output
            // tile (the last split to arrive reduces and runs the
            // epilogue, then resets its counter for the next launch)
            writeln!(
                s,
                "__device__ float tang_partial[{}];",
                n_tiles * n_outs * bm * bn
            )
            .unwrap();
            writeln!(s, "__device__ unsigned tang_arrived[{n_base}];").unwrap();
        }
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const {}* __restrict__ b{j}", elem(fmt(j))))
            .collect();
        params.push(format!("{}* __restrict__ out", elem(fmt(root))));
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({threads}) main0({}) {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    extern __shared__ __align__(16) float smem[];").unwrap();
        if wg {
            // 128-byte swizzle atoms are 1 KB aligned
            writeln!(s, "    unsigned short* smem_h = (unsigned short*)(((size_t)smem + 1023) & ~(size_t)1023);").unwrap();
        }
        writeln!(s, "    unsigned tile = blockIdx.x;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) return;").unwrap();
        let per_split = k_total.div_ceil(chunk).div_ceil(splits) * chunk;
        writeln!(s, "    unsigned split = tile / {n_base}u;").unwrap();
        writeln!(s, "    tile = tile % {n_base}u;").unwrap();
        writeln!(s, "    unsigned k_begin = split * {per_split}u;").unwrap();
        writeln!(
            s,
            "    unsigned k_end = min(k_begin + {per_split}u, {k_total}u);"
        )
        .unwrap();
        writeln!(
            s,
            "    unsigned n_chunks = k_end > k_begin ? (k_end - k_begin + {}u) / {chunk}u : 0u;",
            chunk - 1
        )
        .unwrap();
        writeln!(s, "    unsigned tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    unsigned tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    unsigned lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    unsigned p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    unsigned row0 = tile_y * {bm}u;").unwrap();
        writeln!(s, "    unsigned col0 = tile_x * {bn}u;").unwrap();
        // A staged map in every term that is provably zero where an
        // affine condition holds (the causal mask inside P̃ = exp(S + mask
        // − m), so inside dS too): chunks where it holds over the whole
        // block contribute nothing and are left out of the k range.
        let common = |side_a: bool| -> Option<usize> {
            let mut it = uses
                .iter()
                .flatten()
                .map(|&(ai, bi)| if side_a { ai } else { bi });
            let first = it.next()?;
            it.all(|x| x == first).then_some(first)
        };
        let zero_cond = [(true, common(true)), (false, common(false))]
            .into_iter()
            .find_map(|(side_a, st)| {
                let st = st?;
                let map = if side_a {
                    a_stages[st].2
                } else {
                    b_stages[st].2
                };
                self.map_zero_cond(rc, map?)
            })
            .filter(|f| others.iter().all(|&d| f.coefs[d] == 0));
        if let Some(f) = &zero_cond {
            // the condition's minimum over the block × chunk box
            let lo_hi = |d: usize, lo: String, hi: String| -> String {
                let cf = f.coefs[d];
                match cf.cmp(&0) {
                    std::cmp::Ordering::Greater => format!(" + {cf} * (int)({lo})"),
                    std::cmp::Ordering::Less => format!(" + {cf} * (int)({hi})"),
                    std::cmp::Ordering::Equal => String::new(),
                }
            };
            let min = format!(
                "{}{}{}{}",
                f.offset,
                lo_hi(
                    dy,
                    "row0".into(),
                    format!("min(row0 + {}u, {}u)", bm - 1, m - 1)
                ),
                lo_hi(
                    dx,
                    "col0".into(),
                    format!("min(col0 + {}u, {}u)", bn - 1, n - 1)
                ),
                lo_hi(
                    kdim,
                    "k0".into(),
                    format!("min(k0 + {}u, {}u)", chunk - 1, k_total - 1)
                ),
            );
            writeln!(s, "    {{").unwrap();
            writeln!(s, "        unsigned lo = k_end, hi = k_begin;").unwrap();
            writeln!(s, "        for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "            unsigned k0 = k_begin + c * {chunk}u;").unwrap();
            writeln!(s, "            if ({min} > 0) continue;").unwrap();
            writeln!(
                s,
                "            lo = min(lo, k0); hi = max(hi, min(k0 + {chunk}u, k_end));"
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(
                s,
                "        if (hi > lo) {{ k_begin = lo; k_end = hi; }} else k_end = k_begin;"
            )
            .unwrap();
            writeln!(
                s,
                "        n_chunks = k_end > k_begin ? (k_end - k_begin + {}u) / {chunk}u : 0u;",
                chunk - 1
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
        let lead_term = |addr: &super::msl_mma::Addr| -> String {
            let mut t = format!("{}", addr.offset);
            for &d in &others {
                if addr.coefs[d] != 0 {
                    t.push_str(&format!(" + {} * (int)p{d}", addr.coefs[d]));
                }
            }
            t
        };
        for (t, (_, addr, _)) in a_stages.iter().enumerate() {
            writeln!(s, "    int base_a{t} = {};", lead_term(addr)).unwrap();
        }
        for (t, (_, addr, _)) in b_stages.iter().enumerate() {
            writeln!(s, "    int base_b{t} = {};", lead_term(addr)).unwrap();
        }
        writeln!(s, "    unsigned tid = threadIdx.x;").unwrap();
        writeln!(s, "    unsigned warp = tid / 32u;").unwrap();
        writeln!(s, "    unsigned lane = tid % 32u;").unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        writeln!(s, "    unsigned wr = (warp / {}u) * {wtm}u;", sh.warps_n).unwrap();
        writeln!(s, "    unsigned wc = (warp % {}u) * {wtn}u;", sh.warps_n).unwrap();
        writeln!(s, "    float acc[{n_outs}][{fm}][{fnn}][4];").unwrap();
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int w = 0; w < {n_outs}; w++) for (int i = 0; i < {fm}; i++) for (int j = 0; j < {fnn}; j++) for (int r = 0; r < 4; r++) acc[w][i][j][r] = 0.0f;").unwrap();
        // The side of each nested contraction that does not move with k
        // (the staged operand's own rows: K for S in dK) is held in mma A
        // fragments for the whole kernel; warp w owns rows rpw·w … of
        // the block. k slots (tq, tq + 4) take e = 8q + 2tq, +1, the
        // pairing the flash forward uses, so S is the same sum it was.
        let n_a_stages = a_stages.len();
        for (o, plan) in nested.iter().enumerate() {
            let (origin, extent, block) = if o < n_a_stages {
                ("row0", m, bm)
            } else {
                ("col0", n, bn)
            };
            let nfm = block / nwarps / 16;
            for (q, nst) in plan.iter().enumerate() {
                let ne = nst.e / 8;
                for (t, term) in nst.terms.iter().enumerate() {
                    let name = format!("nf{o}_{q}_{t}");
                    writeln!(s, "    unsigned {name}[{nfm}][{ne}][4];").unwrap();
                    let (cp, ce) = (term.p_addr.coefs[nst.pdim], term.p_addr.coefs[rank]);
                    let base = lead_term(&term.p_addr);
                    writeln!(s, "    #pragma unroll").unwrap();
                    writeln!(s, "    for (int i = 0; i < {nfm}; i++)").unwrap();
                    writeln!(s, "    #pragma unroll").unwrap();
                    writeln!(s, "    for (int q = 0; q < {ne}; q++)").unwrap();
                    writeln!(s, "    #pragma unroll").unwrap();
                    writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
                    writeln!(
                        s,
                        "        unsigned gp = {origin} + warp * {}u + 16u * i + gq + 8u * (r % 2);",
                        block / nwarps
                    )
                    .unwrap();
                    writeln!(s, "        int e = 8 * q + 2 * (int)tq + r / 2;").unwrap();
                    writeln!(
                        s,
                        "        float x = gp < {extent}u ? {} : 0.0f;",
                        ld(
                            term.p_buf,
                            &format!("(unsigned)({base} + {cp} * (int)gp + {ce} * e)")
                        )
                    )
                    .unwrap();
                    let scale = if term.coef == 1.0 {
                        String::new()
                    } else {
                        format!("{:?}f * ", term.coef as f32)
                    };
                    writeln!(s, "        {name}[i][q][r] = tang_tf32({scale}x);").unwrap();
                    writeln!(s, "    }}").unwrap();
                }
            }
        }

        // staging of one chunk into buffer `buf`, walking the global
        // buffer in address order
        writeln!(s, "    auto load = [&](unsigned buf, unsigned k0) {{").unwrap();
        if wg {
            writeln!(
                s,
                "        unsigned short* st = smem_h + buf * {stage_floats}u;"
            )
            .unwrap();
        } else if half.is_some() {
            writeln!(
                s,
                "        unsigned short* st = (unsigned short*)smem + buf * {stage_floats}u;"
            )
            .unwrap();
        } else {
            writeln!(s, "        float* st = smem + buf * {stage_floats}u;").unwrap();
        }
        let stage = |s: &mut String,
                     o: usize,
                     off: usize,
                     base: &str,
                     addr: &super::msl_mma::Addr,
                     pdim: usize,
                     origin: &str,
                     extent: usize,
                     block: usize,
                     buf: usize,
                     map: Option<usize>,
                     row_major: bool,
                     ldst: usize| {
            let (cp, ck) = (addr.coefs[pdim], addr.coefs[kdim]);
            let slot = |pt: &str, kk: &str| {
                // wgmma: 64-element rows of 128 bytes along the staged
                // row (k when K-major, the block when MN-major), 16-byte
                // chunks XOR-swizzled by the row index mod 8
                if wg && row_major {
                    format!("st + {off}u + (({kk}) / 64u) * {}u + ({pt}) * 64u + (((({kk}) % 64u) / 8u ^ (({pt}) % 8u)) * 8u) + ({kk}) % 8u", block * 64)
                } else if wg {
                    format!("st + {off}u + (({pt}) / 64u) * {}u + ({kk}) * 64u + (((({pt}) % 64u) / 8u ^ (({kk}) % 8u)) * 8u) + ({pt}) % 8u", chunk * 64)
                } else if row_major {
                    format!("st + {off}u + {pt} * {ldst}u + {kk}")
                } else {
                    format!("st + {off}u + {kk} * {ldst}u + {pt}")
                }
            };
            // 16-byte copies along the shared row: the global stride
            // there is 1 and every vector start is 4-aligned and either
            // wholly in range or wholly out
            // an f32 operand of a half tile: converted while staging
            let conv = half.is_some() && map.is_none() && fmt(buf) == super::Format::F32;
            // elements per vector (16 bytes of the operand)
            let w: i64 = if half.is_some() && !conv { 8 } else { 4 };
            let lead_aligned =
                addr.offset % w == 0 && others.iter().all(|&d| addr.coefs[d] % w == 0);
            let wu = w as usize;
            let vec4 = map.is_none()
                && lead_aligned
                && if row_major {
                    ck == 1 && cp % w == 0 && k_total % wu == 0 && chunk % wu == 0
                } else {
                    cp == 1 && ck % w == 0 && extent.is_multiple_of(wu) && block.is_multiple_of(wu)
                };
            if vec4 {
                let (inner, outer) = if row_major {
                    (chunk, block)
                } else {
                    (block, chunk)
                };
                let nv = inner / wu;
                let total = inner * outer / wu;
                // a fixed trip count where the threads tile the block
                // evenly: every thread keeps its column, rows step by a
                // constant, and the index math hoists out of the k loop
                let (o, i) = if row_major {
                    ("pt", "kk")
                } else {
                    ("kk", "pt")
                };
                if total.is_multiple_of(threads) && threads.is_multiple_of(nv) {
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(
                        s,
                        "        for (unsigned q = 0u; q < {}u; q++) {{",
                        total / threads
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned {i} = (tid % {nv}u) * {wu}u; unsigned {o} = tid / {nv}u + q * {}u;",
                        threads / nv
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "        for (unsigned idx = tid; idx < {total}u; idx += {threads}u) {{"
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            unsigned {i} = (idx % {nv}u) * {wu}u; unsigned {o} = idx / {nv}u;"
                    )
                    .unwrap();
                }
                writeln!(
                    s,
                    "            unsigned gp = {origin} + pt; unsigned gk = k0 + kk;"
                )
                .unwrap();
                writeln!(
                    s,
                    "            bool ok = gp < {extent}u && gk < {k_total}u;"
                )
                .unwrap();
                if conv {
                    writeln!(
                        s,
                        "            float4 x = ok ? __ldg((const float4*)(b{buf} + (unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "            *(uint2*)({}) = make_uint2(tang_pack2(x.x, x.y), tang_pack2(x.z, x.w));",
                        slot("pt", "kk")
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            {}({}, b{buf} + (ok ? (unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk) : 0u), ok);",
                        if half.is_some() { "tang_cp16h" } else { "tang_cp16" },
                        slot("pt", "kk")
                    )
                    .unwrap();
                }
                writeln!(s, "        }}").unwrap();
                return;
            }
            // nested contractions: this chunk's block of each, by mma,
            // in registers; the map is then evaluated on the accumulator
            // elements (rows: warp w's rpw of the block; columns: the
            // chunk) and written straight to its stage
            let plan = &nested[o];
            let fused = !plan.is_empty();
            let mut ra_of: Vec<Option<usize>> =
                vec![None; map.map_or(0, |v| self.comp_at(rc.args[v].src.n()).unwrap().args.len())];
            if fused {
                let rpw = block / nwarps;
                let nfm = rpw / 16;
                let nfn = chunk / 8;
                writeln!(s, "        {{").unwrap();
                for (q, nst) in plan.iter().enumerate() {
                    ra_of[nst.arg] = Some(q);
                    let ne = nst.e / 8;
                    writeln!(s, "        float ra{q}[{nfm}][{nfn}][4];").unwrap();
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int i = 0; i < {nfm}; i++) for (int j = 0; j < {nfn}; j++) for (int r = 0; r < 4; r++) ra{q}[i][j][r] = 0.0f;").unwrap();
                    for (t, term) in nst.terms.iter().enumerate() {
                        let (ck, ce) = (term.k_addr.coefs[kdim], term.k_addr.coefs[rank]);
                        let base = lead_term(&term.k_addr);
                        writeln!(s, "        #pragma unroll").unwrap();
                        writeln!(s, "        for (int q = 0; q < {ne}; q++)").unwrap();
                        writeln!(s, "        #pragma unroll").unwrap();
                        writeln!(s, "        for (int j = 0; j < {nfn}; j++) {{").unwrap();
                        writeln!(s, "            unsigned gk = k0 + 8u * j + gq; int e = 8 * q + 2 * (int)tq;").unwrap();
                        writeln!(s, "            bool ok = gk < {k_total}u;").unwrap();
                        let kb = term.k_buf;
                        writeln!(s, "            const {}* src = b{kb} + (ok ? (unsigned)({base} + {ck} * (int)gk + {ce} * e) : 0u);", elem(fmt(kb))).unwrap();
                        let (s0, s1) = match fmt(kb) {
                            super::Format::F32 => ("src[0]".to_string(), format!("src[{ce}]")),
                            super::Format::F16 => (
                                "tang_h2f(src[0])".to_string(),
                                format!("tang_h2f(src[{ce}])"),
                            ),
                            super::Format::BF16 => (
                                "tang_b2f(src[0])".to_string(),
                                format!("tang_b2f(src[{ce}])"),
                            ),
                            super::Format::Q4K
                            | super::Format::Q5K
                            | super::Format::Q6K
                            | super::Format::Q8x32
                            | super::Format::Q8x32F
                            | super::Format::Fp8B32
                            | super::Format::Mxfp4 => {
                                unreachable!("block formats lower only through the int8 MMA kernel")
                            }
                            _ => unreachable!("fp8 has its own tile"),
                        };
                        writeln!(s, "            unsigned kf[2] = {{tang_tf32(ok ? {s0} : 0.0f), tang_tf32(ok ? {s1} : 0.0f)}};").unwrap();
                        writeln!(s, "            #pragma unroll").unwrap();
                        writeln!(s, "            for (int i = 0; i < {nfm}; i++) tang_mma(ra{q}[i][j], nf{o}_{q}_{t}[i][q], kf);").unwrap();
                        writeln!(s, "        }}").unwrap();
                    }
                }
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int i = 0; i < {nfm}; i++)").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int j = 0; j < {nfn}; j++)").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int r = 0; r < 4; r++) {{").unwrap();
                writeln!(
                    s,
                    "            unsigned pt = warp * {rpw}u + 16u * i + gq + 8u * (r / 2); unsigned kk = 8u * j + 2u * tq + (r % 2);"
                )
                .unwrap();
            } else {
                let k_inner = ck.unsigned_abs() <= cp.unsigned_abs();
                writeln!(
                    s,
                    "        for (unsigned idx = tid; idx < {}u; idx += {threads}u) {{",
                    block * chunk
                )
                .unwrap();
                if k_inner {
                    writeln!(
                        s,
                        "            unsigned kk = idx % {chunk}u; unsigned pt = idx / {chunk}u;"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "            unsigned pt = idx % {block}u; unsigned kk = idx / {block}u;"
                    )
                    .unwrap();
                }
            }
            writeln!(
                s,
                "            unsigned gp = {origin} + pt; unsigned gk = k0 + kk;"
            )
            .unwrap();
            let Some(v) = map else {
                writeln!(
                    s,
                    "            bool ok = gp < {extent}u && gk < {k_total}u;"
                )
                .unwrap();
                if half.is_some() {
                    // cp.async copies 4 bytes at least: a plain copy (an
                    // f32 operand converted)
                    let x = format!("b{buf}[(unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk)]");
                    let x = match (conv, half) {
                        (false, _) => x,
                        (true, Some(super::Format::BF16)) => format!("tang_f2b({x})"),
                        (true, _) => format!("tang_f2h({x})"),
                    };
                    writeln!(
                        s,
                        "            *({}) = ok ? {x} : (unsigned short)0;",
                        slot("pt", "kk")
                    )
                    .unwrap();
                    writeln!(s, "        }}").unwrap();
                    return;
                }
                writeln!(
                    s,
                    "            tang_cp4({}, b{buf} + (ok ? (unsigned)({base} + {cp} * (int)gp + {ck} * (int)gk) : 0u), ok);",
                    slot("pt", "kk")
                )
                .unwrap();
                writeln!(s, "        }}").unwrap();
                return;
            };
            // a map operand: its body over the args it reads, each at its
            // address composed through the map's view
            let outer = &rc.args[v];
            let mc = self.comp_at(outer.src.n()).unwrap();
            let coord = |d: usize| -> String {
                if d == pdim {
                    "(int)gp".into()
                } else if d == kdim {
                    "(int)gk".into()
                } else {
                    format!("(int)p{d}")
                }
            };
            let affine = |offset: i64, coefs: &[i64]| -> String {
                let mut e = format!("{offset}");
                for (d, &cf) in coefs.iter().enumerate() {
                    if cf != 0 {
                        e.push_str(&format!(" + {cf} * {}", coord(d)));
                    }
                }
                e
            };
            let vals: Vec<String> = mc
                .args
                .iter()
                .enumerate()
                .map(|(ai, sa)| {
                    if let Some(q) = ra_of[ai] {
                        return format!("ra{q}[i][j][r]");
                    }
                    let view = sa.view.compose(&outer.view, rank);
                    match &self.nodes[sa.src.n()].op {
                        super::Op::Iota(dd) => {
                            let f = &view.forms[*dd];
                            format!("(float)({})", affine(f.offset, &f.coefs))
                        }
                        _ => {
                            let composed = super::Arg { src: sa.src, view };
                            let a = self.flat_addr(&composed, rank);
                            ld(
                                storage(sa.src.n()),
                                &format!("(unsigned)({})", affine(a.offset, &a.coefs)),
                            )
                        }
                    }
                })
                .collect();
            writeln!(s, "            float v = 0.0f;").unwrap();
            writeln!(s, "            if (gp < {extent}u && gk < {k_total}u) {{").unwrap();
            let mut body = String::new();
            let outs = emit_scalar(
                &mut body,
                &mc.body,
                &mc.body_outs,
                &format!("m{off}_"),
                &|x| vals[x as usize].clone(),
            );
            for line in body.lines() {
                writeln!(
                    s,
                    "                {}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            writeln!(s, "                v = {};", outs[0]).unwrap();
            writeln!(s, "            }}").unwrap();
            match half {
                Some(super::Format::BF16) => {
                    writeln!(s, "            *({}) = tang_f2b(v);", slot("pt", "kk")).unwrap()
                }
                Some(_) => {
                    writeln!(s, "            *({}) = tang_f2h(v);", slot("pt", "kk")).unwrap()
                }
                None => writeln!(s, "            *({}) = v;", slot("pt", "kk")).unwrap(),
            }
            writeln!(s, "        }}").unwrap();
            if fused {
                writeln!(s, "        }}").unwrap();
            }
        };
        for (t, (buf, addr, map)) in a_stages.iter().enumerate() {
            stage(
                &mut s,
                t,
                t * a_floats,
                &format!("base_a{t}"),
                addr,
                dy,
                "row0",
                m,
                bm,
                *buf,
                *map,
                !a_pair,
                if a_pair { ldat } else { lda },
            );
        }
        for (t, (buf, addr, map)) in b_stages.iter().enumerate() {
            stage(
                &mut s,
                a_stages.len() + t,
                a_stages.len() * a_floats + t * b_floats,
                &format!("base_b{t}"),
                addr,
                dx,
                "col0",
                n,
                bn,
                *buf,
                *map,
                b_col,
                if b_col { ldbt } else { ldb },
            );
        }
        writeln!(s, "        tang_cp_commit();").unwrap();
        writeln!(s, "    }};").unwrap();

        // fragments for k step `kk` of buffer `buf` into register set
        // `par`. A reduction is order-free, so within each 8-wide k step
        // the mma k slots (tq, tq + 4) read staged columns (2tq, 2tq + 1)
        // of A and rows (2tq, 2tq + 1) of B: one float2 per A row pair.
        // Columns are ours to place too: fragments (2jp, 2jp + 1) take
        // staged columns 2gq and 2gq + 1 of their 16, one float2 per B row.
        let n_a = a_stages.len();
        let n_b = b_stages.len();
        writeln!(s, "    unsigned af[2][{n_a}][{fm}][4];").unwrap();
        writeln!(s, "    unsigned bf[2][{n_b}][{fnn}][2];").unwrap();
        // m16n8k16 halves: A fragment registers are (row gq | gq + 8) ×
        // (k 2tq, 2tq+1 | 2tq+8, 2tq+9), one 32-bit load each; B's k
        // pairs come from two staged rows, whose 32-bit words hold the
        // staged columns (2gq, 2gq + 1) of a fragment pair, split with
        // byte permutes (low halves: fragment 2jp, high: 2jp + 1).
        let frags16 = |s: &mut String, par: usize, buf: &str, kk: &str| {
            writeln!(s, "        {{").unwrap();
            writeln!(
                s,
                "        const unsigned short* st = (const unsigned short*)smem + ({buf}) * {stage_floats}u;"
            )
            .unwrap();
            for t in 0..n_a {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int i = 0; i < {fm}; i++) {{").unwrap();
                if a_pair {
                    // rows gq, gq + 8 of fragment i are staged rows
                    // 16i + 2gq, + 1: one word, both rows
                    writeln!(s, "            const unsigned short* p = st + {}u + (({kk}) + 2u * tq) * {ldat}u + wr + 16u * i + 2u * gq;", t * a_floats).unwrap();
                    writeln!(s, "            unsigned r0 = *(const unsigned*)p, r1 = *(const unsigned*)(p + {ldat}u);").unwrap();
                    writeln!(s, "            unsigned r8 = *(const unsigned*)(p + {}u), r9 = *(const unsigned*)(p + {}u);", 8 * ldat, 9 * ldat).unwrap();
                    writeln!(s, "            af[{par}][{t}][i][0] = __byte_perm(r0, r1, 0x5410); af[{par}][{t}][i][1] = __byte_perm(r0, r1, 0x7632);").unwrap();
                    writeln!(s, "            af[{par}][{t}][i][2] = __byte_perm(r8, r9, 0x5410); af[{par}][{t}][i][3] = __byte_perm(r8, r9, 0x7632);").unwrap();
                    writeln!(s, "        }}").unwrap();
                    continue;
                }
                writeln!(s, "            const unsigned short* p = st + {}u + (wr + 16u * i + gq) * {lda}u + ({kk}) + 2u * tq;", t * a_floats).unwrap();
                writeln!(s, "            af[{par}][{t}][i][0] = *(const unsigned*)p; af[{par}][{t}][i][1] = *(const unsigned*)(p + {}u);", 8 * lda).unwrap();
                writeln!(s, "            af[{par}][{t}][i][2] = *(const unsigned*)(p + 8u); af[{par}][{t}][i][3] = *(const unsigned*)(p + {}u);", 8 * lda + 8).unwrap();
                writeln!(s, "        }}").unwrap();
            }
            for t in 0..n_b {
                writeln!(s, "        #pragma unroll").unwrap();
                if b_col {
                    // [n][k]: column gq of fragment j is staged row 8j + gq
                    writeln!(s, "        for (int j = 0; j < {fnn}; j++) {{").unwrap();
                    writeln!(s, "            const unsigned short* p = st + {}u + (wc + 8u * j + gq) * {ldbt}u + ({kk}) + 2u * tq;", n_a * a_floats + t * b_floats).unwrap();
                    writeln!(s, "            bf[{par}][{t}][j][0] = *(const unsigned*)p; bf[{par}][{t}][j][1] = *(const unsigned*)(p + 8u);").unwrap();
                    writeln!(s, "        }}").unwrap();
                    continue;
                }
                writeln!(s, "        for (int jp = 0; jp < {}; jp++) {{", fnn / 2).unwrap();
                writeln!(s, "            const unsigned short* p = st + {}u + (({kk}) + 2u * tq) * {ldb}u + wc + 16u * jp + 2u * gq;", n_a * a_floats + t * b_floats).unwrap();
                writeln!(s, "            unsigned r0 = *(const unsigned*)p, r1 = *(const unsigned*)(p + {ldb}u);").unwrap();
                writeln!(s, "            unsigned r8 = *(const unsigned*)(p + {}u), r9 = *(const unsigned*)(p + {}u);", 8 * ldb, 9 * ldb).unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp][0] = __byte_perm(r0, r1, 0x5410); bf[{par}][{t}][2 * jp][1] = __byte_perm(r8, r9, 0x5410);").unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp + 1][0] = __byte_perm(r0, r1, 0x7632); bf[{par}][{t}][2 * jp + 1][1] = __byte_perm(r8, r9, 0x7632);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
        };
        let frags32 = |s: &mut String, par: usize, buf: &str, kk: &str| {
            writeln!(s, "        {{").unwrap();
            writeln!(
                s,
                "        const float* st = smem + ({buf}) * {stage_floats}u;"
            )
            .unwrap();
            for t in 0..n_a {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int i = 0; i < {fm}; i++) {{").unwrap();
                writeln!(s, "            const float* p = st + {}u + (wr + 16u * i + gq) * {lda}u + ({kk}) + 2u * tq;", t * a_floats).unwrap();
                writeln!(
                    s,
                    "            float2 lo = *(const float2*)p, hi = *(const float2*)(p + {}u);",
                    8 * lda
                )
                .unwrap();
                writeln!(s, "            af[{par}][{t}][i][0] = tang_tf32(lo.x); af[{par}][{t}][i][1] = tang_tf32(hi.x);").unwrap();
                writeln!(s, "            af[{par}][{t}][i][2] = tang_tf32(lo.y); af[{par}][{t}][i][3] = tang_tf32(hi.y);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            for t in 0..n_b {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int jp = 0; jp < {}; jp++) {{", fnn / 2).unwrap();
                writeln!(s, "            const float* p = st + {}u + (({kk}) + 2u * tq) * {ldb}u + wc + 16u * jp + 2u * gq;", n_a * a_floats + t * b_floats).unwrap();
                writeln!(
                    s,
                    "            float2 r0 = *(const float2*)p, r1 = *(const float2*)(p + {ldb}u);"
                )
                .unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp][0] = tang_tf32(r0.x); bf[{par}][{t}][2 * jp][1] = tang_tf32(r1.x);").unwrap();
                writeln!(s, "            bf[{par}][{t}][2 * jp + 1][0] = tang_tf32(r0.y); bf[{par}][{t}][2 * jp + 1][1] = tang_tf32(r1.y);").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        }}").unwrap();
        };
        let frags = |s: &mut String, par: usize, buf: &str, kk: &str| {
            if half.is_some() {
                frags16(s, par, buf, kk)
            } else {
                frags32(s, par, buf, kk)
            }
        };
        let mma_fn = if half.is_some() {
            "tang_mma16"
        } else {
            "tang_mma"
        };
        let mmas = |s: &mut String, par: usize| {
            for (w, u) in uses.iter().enumerate() {
                for (ai, bi) in u {
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int i = 0; i < {fm}; i++) for (int j = 0; j < {fnn}; j++) {mma_fn}(acc[{w}][i][j], af[{par}][{ai}][i], bf[{par}][{bi}][j]);").unwrap();
                }
            }
        };
        // k per mma step
        let kstep = if half.is_some() { 16 } else { 8 };
        let steps = chunk / kstep;
        if wg {
            // Hopper: chunk c's wgmma are issued asynchronously, then the
            // staging of chunk c + P runs (copies, or the maps and nested
            // contractions computed into their stage) while they execute.
            // With one wgmma group left in flight (ring ≥ 3), after the
            // barrier of iteration c everyone's chunk c - 2 is done, so
            // chunks c - 1 … c + P - 1 hold P + 1 buffers and c + P takes
            // the last: P = stages - 2 (stages 2: P = 1, no group left in
            // flight).
            let st = sh.stages;
            // TANG_WG_SYNC=1 (or nested contractions, whose registers
            // would make ptxas serialize the wgmma anyway): stage the
            // chunk ahead before issuing, and drain the group at once
            let has_nested = nested.iter().any(|p| !p.is_empty());
            let sync = match std::env::var("TANG_WG_SYNC").ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => has_nested,
            };
            let (pf, wgw) = if sync {
                (st - 1, 0)
            } else if st >= 3 {
                (st - 2, 1)
            } else {
                (1, 0)
            };
            let wpm = bm / n_wg;
            let bytes = stage_floats * 2;
            writeln!(
                s,
                "    unsigned sbase = (unsigned)__cvta_generic_to_shared(smem_h);"
            )
            .unwrap();
            writeln!(s, "    unsigned wgi = warp / 4u;").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < {pf}u; c++) {{ if (c < n_chunks) load(c, k_begin + c * {chunk}u); else tang_cp_commit(); }}").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "        tang_cp_wait<{}>();", pf - 1).unwrap();
            writeln!(s, "        tang_fence_async();").unwrap();
            writeln!(s, "        __syncthreads();").unwrap();
            writeln!(s, "        unsigned so = sbase + (c % {st}u) * {bytes}u;").unwrap();
            let ahead = format!("        if (c + {pf}u < n_chunks) load((c + {pf}u) % {st}u, k_begin + (c + {pf}u) * {chunk}u); else tang_cp_commit();");
            if sync {
                writeln!(s, "{ahead}").unwrap();
            }
            writeln!(s, "        tang_wg_fence();").unwrap();
            let ta = a_pair as u8;
            let tb = (!b_col) as u8;
            for (w, u) in uses.iter().enumerate() {
                for &(ai, bi) in u {
                    let a_off = ai * a_floats * 2;
                    let b_off = (n_a * a_floats + bi * b_floats) * 2;
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int i = 0; i < {fm}; i++) {{").unwrap();
                    writeln!(s, "            unsigned mb = wgi * {wpm}u + 64u * i;").unwrap();
                    writeln!(s, "            #pragma unroll").unwrap();
                    writeln!(
                        s,
                        "            for (unsigned kb = 0u; kb < {chunk}u; kb += 16u) {{"
                    )
                    .unwrap();
                    if a_pair {
                        // [bm/64][chunk][64], MN-major
                        writeln!(s, "                unsigned long long da = tang_desc(so + {a_off}u + (mb / 64u) * {}u + kb * 128u, {}u, 1024u);", chunk * 128, chunk * 128).unwrap();
                    } else {
                        // [chunk/64][bm][64], K-major
                        writeln!(s, "                unsigned long long da = tang_desc(so + {a_off}u + (kb / 64u) * {}u + mb * 128u + (kb % 64u) * 2u, 16u, 1024u);", bm * 128).unwrap();
                    }
                    if b_col {
                        writeln!(s, "                unsigned long long db = tang_desc(so + {b_off}u + (kb / 64u) * {}u + (kb % 64u) * 2u, 16u, 1024u);", bn * 128).unwrap();
                    } else {
                        writeln!(s, "                unsigned long long db = tang_desc(so + {b_off}u + kb * 128u, {}u, 1024u);", chunk * 128).unwrap();
                    }
                    writeln!(
                        s,
                        "                tang_wg_ss{bn}_{ta}{tb}(&acc[{w}][i][0][0], da, db);"
                    )
                    .unwrap();
                    writeln!(s, "            }}").unwrap();
                    writeln!(s, "        }}").unwrap();
                }
            }
            writeln!(s, "        tang_wg_commit();").unwrap();
            if !sync {
                writeln!(s, "{ahead}").unwrap();
            }
            writeln!(s, "        tang_wg_wait<{wgw}>();").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    tang_wg_wait<0>();").unwrap();
        } else if sh.stages >= 2 {
            // a ring of `stages` buffers, chunk c + stages - 1 in flight
            // while chunk c is multiplied; the fragments of the next k step
            // load while this one's mma issue (across the chunk boundary
            // too, after the one barrier per chunk)
            let st = sh.stages;
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < {}u; c++) {{ if (c < n_chunks) load(c, k_begin + c * {chunk}u); else tang_cp_commit(); }}", st - 1).unwrap();
            writeln!(s, "    tang_cp_wait<{}>();", st - 2).unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (n_chunks > 0u) {{").unwrap();
            frags(&mut s, 0, "0u", "0u");
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "        unsigned nc = c + {}u;", st - 1).unwrap();
            writeln!(s, "        if (nc < n_chunks) load(nc % {st}u, k_begin + nc * {chunk}u); else tang_cp_commit();").unwrap();
            for ks in 0..steps {
                let par = ks % 2;
                if ks + 1 < steps {
                    frags(
                        &mut s,
                        1 - par,
                        &format!("c % {st}u"),
                        &format!("{}u", (ks + 1) * kstep),
                    );
                } else {
                    writeln!(s, "        tang_cp_wait<{}>();", st - 2).unwrap();
                    writeln!(s, "        __syncthreads();").unwrap();
                    writeln!(s, "        if (c + 1u < n_chunks) {{").unwrap();
                    frags(&mut s, 1 - par, &format!("(c + 1u) % {st}u"), "0u");
                    writeln!(s, "        }}").unwrap();
                }
                mmas(&mut s, par);
            }
            if steps % 2 == 1 {
                // odd step count: the next chunk's first fragments landed
                // in set 1; move them to set 0
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int t = 0; t < {n_a}; t++) for (int i = 0; i < {fm}; i++) for (int e = 0; e < 4; e++) af[0][t][i][e] = af[1][t][i][e];").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int t = 0; t < {n_b}; t++) for (int j = 0; j < {fnn}; j++) for (int e = 0; e < 2; e++) bf[0][t][j][e] = bf[1][t][j][e];").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(s, "    for (unsigned c = 0u; c < n_chunks; c++) {{").unwrap();
            writeln!(s, "        load(0u, k_begin + c * {chunk}u);").unwrap();
            writeln!(s, "        tang_cp_wait<0>();").unwrap();
            writeln!(s, "        __syncthreads();").unwrap();
            for ks in 0..steps {
                frags(&mut s, 0, "0u", &format!("{}u", ks * kstep));
                mmas(&mut s, 0);
            }
            writeln!(s, "        __syncthreads();").unwrap();
            writeln!(s, "    }}").unwrap();
        }

        // the epilogue straight from the accumulators: element r of
        // fragment (i, j) is row gq + 8·(r/2) and mma column 2·tq + r%2,
        // staged column 2·(that) + j%2 of the fragment pair's 16
        let bb = bm * bn;
        let frag_loop = |s: &mut String| {
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int i = 0; i < {fm}; i++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int j = 0; j < {fnn}; j++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
            if wg {
                // wgmma m64nN: warp w of the group holds rows 16w + gq
                // (+ 8), each n8 group laid out as an mma.sync C fragment
                writeln!(s, "        unsigned lr = (warp / 4u) * {}u + 64u * i + 16u * (warp % 4u) + gq + 8u * (r / 2);", bm / n_wg.max(1)).unwrap();
                writeln!(s, "        unsigned lc = 8u * j + 2u * tq + (r % 2);").unwrap();
            } else if a_pair {
                writeln!(s, "        unsigned lr = wr + 16u * i + 2u * gq + (r / 2);").unwrap();
            } else {
                writeln!(s, "        unsigned lr = wr + 16u * i + gq + 8u * (r / 2);").unwrap();
            }
            if wg {
            } else if b_col {
                writeln!(s, "        unsigned lc = wc + 8u * j + 2u * tq + (r % 2);").unwrap();
            } else {
                writeln!(
                    s,
                    "        unsigned lc = wc + 16u * (j / 2) + 4u * tq + 2u * (r % 2) + (j % 2);"
                )
                .unwrap();
            }
        };
        if splits > 1 {
            writeln!(
                s,
                "    float* part = tang_partial + (split * {n_base}u + tile) * {}u;",
                n_outs * bb
            )
            .unwrap();
            frag_loop(&mut s);
            writeln!(s, "        for (int w = 0; w < {n_outs}; w++) part[w * {bb}u + lr * {bn}u + lc] = acc[w][i][j][r];").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __threadfence();").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    __shared__ unsigned last;").unwrap();
            writeln!(s, "    if (tid == 0u) {{").unwrap();
            writeln!(
                s,
                "        last = atomicAdd(&tang_arrived[tile], 1u) == {}u;",
                splits - 1
            )
            .unwrap();
            writeln!(s, "        if (last) tang_arrived[tile] = 0u;").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (!last) return;").unwrap();
            writeln!(s, "    __threadfence();").unwrap();
        }
        frag_loop(&mut s);
        writeln!(s, "        unsigned p{dy} = row0 + lr;").unwrap();
        writeln!(s, "        unsigned p{dx} = col0 + lc;").unwrap();
        writeln!(s, "        if (p{dy} < {m}u && p{dx} < {n}u) {{").unwrap();
        for w in 0..n_outs {
            if splits > 1 {
                // the splits in a fixed order: deterministic
                writeln!(s, "            float red{w} = 0.0f;").unwrap();
                writeln!(s, "            for (unsigned sp = 0u; sp < {splits}u; sp++) red{w} += __ldcg(tang_partial + (sp * {n_base}u + tile) * {}u + {}u + lr * {bn}u + lc);", n_outs * bb, w * bb).unwrap();
            } else {
                writeln!(s, "            float red{w} = acc[{w}][i][j][r];").unwrap();
            }
            if scales[w] != 1.0 {
                writeln!(s, "            red{w} *= {}f;", f32_lit(scales[w])).unwrap();
            }
        }
        let vals: Vec<String> = if epilogue {
            let prank = c.points().len();
            let mut arg_vals = Vec::new();
            for a in &c.args {
                if a.src.n() == red_node {
                    arg_vals.push(format!("red{}", self.epilogue_slot(a)));
                    continue;
                }
                let addr = self.flat_addr(a, prank);
                let mut e = format!("{}", addr.offset);
                for (d, &coef) in addr.coefs.iter().enumerate() {
                    if coef != 0 {
                        e.push_str(&format!(" + {coef} * (int)p{d}"));
                    }
                }
                arg_vals.push(ld(storage(a.src.n()), &format!("(unsigned)({e})")));
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
            (0..n_outs).map(|w| format!("red{w}")).collect()
        };
        let mut flat = String::from("0u");
        for (d, &np) in pts.iter().enumerate() {
            flat = format!("({flat}) * {np}u + p{d}");
        }
        writeln!(s, "            unsigned g = {flat};").unwrap();
        for (slot, v) in vals.iter().enumerate() {
            let v = match fmt(root) {
                super::Format::F32 => v.clone(),
                super::Format::F16 => format!("tang_f2h({v})"),
                super::Format::BF16 => format!("tang_f2b({v})"),
                super::Format::Q4K
                | super::Format::Q5K
                | super::Format::Q6K
                | super::Format::Q8x32
                | super::Format::Q8x32F
                | super::Format::Fp8B32
                | super::Format::Mxfp4 => {
                    unreachable!("block formats lower only through the int8 MMA kernel")
                }
                _ => unreachable!("fp8 has its own tile"),
            };
            writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        Some(s)
    }

    /// The CUDA tensor-core kernel for the one tiled matmul in `outputs`'
    /// program, with its read buffers in binding order. For the harness.
    pub fn cuda_mma_program(&self, outputs: &[TensorId]) -> Option<(String, Vec<TensorId>)> {
        self.cuda_mma_program_shaped(outputs, MMA_SHAPES[0])
    }

    /// [`cuda_mma_program`](Self::cuda_mma_program) in block shape `sh`.
    pub fn cuda_mma_program_shaped(
        &self,
        outputs: &[TensorId],
        sh: MmaShape,
    ) -> Option<(String, Vec<TensorId>)> {
        let live = self.live(outputs);
        for root in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            if self.nodes[root].tile.0 == 0 {
                continue;
            }
            let order = self.inline_closure(root);
            let reads: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&j| {
                    j != root
                        && !self.nodes[j].inline
                        && !matches!(self.nodes[j].op, super::Op::Iota(_) | super::Op::Reshape(_))
                })
                .collect();
            if let Some(src) = self.cuda_mma_kernel_shaped(root, &reads, sh) {
                return Some((src, reads.iter().map(|&j| TensorId(j as u32)).collect()));
            }
        }
        None
    }
}

/// How many ways to split a tile's reduction (split-K): up to `target`
/// blocks (the shape's, or `TANG_SPLITK_TARGET`) while every split keeps
/// at least four chunks of the reduction.
/// A sum of products `Σ_e U(row, e)·V(k, e)` nested in a staged map
/// operand: `U` moves with the operand's rows only (held in registers),
/// `V` with the outer reduction's k (read per chunk).
struct Nested {
    /// the map's arg it replaces
    arg: usize,
    /// the nested reduce dim's extent
    e: usize,
    pdim: usize,
    terms: Vec<NestedTerm>,
}

struct NestedTerm {
    coef: f64,
    p_buf: usize,
    /// addresses over the outer space plus the nested dim (last)
    p_addr: Addr,
    k_buf: usize,
    k_addr: Addr,
}

impl TensorGraph {
    /// An affine form over `rc`'s space where map operand `v` of `rc` is
    /// zero wherever it is positive: the map's value has a factor
    /// `exp2(… + κ·select(cond, ≤ −1e20, 0))` with `κ > 0` (a masked
    /// exponential), the other terms finite.
    fn map_zero_cond(&self, rc: &super::Comp, v: usize) -> Option<super::Form> {
        let rank = rc.rank();
        let outer = &rc.args[v];
        let mc = self.comp_at(outer.src.n())?;
        // the map with its reads seen from rc's space
        let seen = super::Comp {
            out: rc.points().to_vec(),
            tuple: 0,
            red: rc.red.clone(),
            args: mc
                .args
                .iter()
                .map(|a| Arg {
                    src: a.src,
                    view: a.view.compose(&outer.view, rank),
                })
                .collect(),
            body: mc.body.clone(),
            body_outs: mc.body_outs.clone(),
            monoid: None,
        };
        let nodes = seen.body.nodes_slice();
        let lit = |e: tang_expr::ExprId| nodes[e.index() as usize].as_f64();
        // where the exponent is −∞ in effect
        fn neg_inf(
            g: &TensorGraph,
            c: &super::Comp,
            nodes: &[ENode],
            e: tang_expr::ExprId,
        ) -> Option<super::Form> {
            let lit = |e: tang_expr::ExprId| nodes[e.index() as usize].as_f64();
            match nodes[e.index() as usize] {
                ENode::Add(a, b) => neg_inf(g, c, nodes, a).or_else(|| neg_inf(g, c, nodes, b)),
                ENode::Mul(a, b) => match (lit(a), lit(b)) {
                    (Some(k), _) if k > 0.0 => neg_inf(g, c, nodes, b),
                    (_, Some(k)) if k > 0.0 => neg_inf(g, c, nodes, a),
                    _ => None,
                },
                ENode::Select(cnd, t, f) => match (lit(t), lit(f)) {
                    (Some(neg), Some(zero)) if neg <= -1e20 && zero == 0.0 => g
                        .affine_of(c, cnd)
                        .filter(|(_, k)| k.fract() == 0.0)
                        .map(|(mut f, k)| {
                            f.offset += k as i64;
                            f
                        }),
                    _ => None,
                },
                _ => None,
            }
        }
        let mut stack = vec![seen.body_outs[0]];
        while let Some(e) = stack.pop() {
            match nodes[e.index() as usize] {
                ENode::Mul(a, b) => stack.extend([a, b]),
                ENode::Exp2(x) => {
                    if let Some(f) = neg_inf(self, &seen, nodes, x) {
                        return Some(f);
                    }
                }
                _ => {}
            }
        }
        let _ = lit;
        None
    }

    /// The contractions nested in map operand `v` of `rc`, each a sum of
    /// products of buffers splitting into a side along `pdim` (the
    /// staged rows) and one along `kdim`. `None` when a nested arg does
    /// not have that shape.
    fn nested_plan(
        &self,
        rc: &super::Comp,
        rank: usize,
        v: usize,
        pdim: usize,
        kdim: usize,
        reads: &[usize],
    ) -> Option<Vec<Nested>> {
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let outer = &rc.args[v];
        let mc = self.comp_at(outer.src.n())?;
        let mut plan = Vec::new();
        for (ai, sa) in mc.args.iter().enumerate() {
            let j = sa.src.n();
            let Some(r) = self.comp_at(j) else { continue };
            if !self.nodes[j].inline || r.red.is_empty() {
                continue;
            }
            let e = r.red[0].fixed()?;
            if r.red.len() != 1 || !e.is_multiple_of(8) || e > 256 {
                return None;
            }
            let terms = super::bilinear_terms(&r.body, r.body_outs[0])?;
            // the nested reduction's points over the outer space, its
            // reduce coord appended as coord `rank`
            let pv = sa.view.compose(&outer.view, rank);
            let inner = super::View {
                guard: false,
                gather: None,
                forms: pv
                    .forms
                    .iter()
                    .map(|f| {
                        let mut f = f.clone();
                        f.coefs.push(0);
                        f
                    })
                    .chain(std::iter::once(super::Form::unit(rank + 1, rank)))
                    .collect(),
            };
            let leaf = |x: u16| -> Option<(usize, Addr)> {
                let a = &r.args[x as usize];
                if a.view.guard || !reads.contains(&storage(a.src.n())) {
                    return None;
                }
                let view = a.view.compose(&inner, rank + 1);
                Some((
                    storage(a.src.n()),
                    self.flat_addr(&Arg { src: a.src, view }, rank + 1),
                ))
            };
            let mut nterms = Vec::new();
            for (coef, x, y) in terms {
                let (bx, ax) = leaf(x)?;
                let (by, ay) = leaf(y)?;
                let p_side = |a: &Addr| a.coefs[kdim] == 0;
                let k_side = |a: &Addr| a.coefs[pdim] == 0;
                let ((p_buf, p_addr), (k_buf, k_addr)) = if p_side(&ax) && k_side(&ay) {
                    ((bx, ax), (by, ay))
                } else if p_side(&ay) && k_side(&ax) {
                    ((by, ay), (bx, ax))
                } else {
                    return None;
                };
                nterms.push(NestedTerm {
                    coef,
                    p_buf,
                    p_addr,
                    k_buf,
                    k_addr,
                });
            }
            plan.push(Nested {
                arg: ai,
                e,
                pdim,
                terms: nterms,
            });
        }
        Some(plan)
    }
}

fn split_k(n_tiles: usize, k_total: usize, chunk: usize, target: usize) -> usize {
    let target: usize = std::env::var("TANG_SPLITK_TARGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(target);
    let mut s = 1;
    while n_tiles * s * 2 <= target && k_total.div_ceil(chunk) >= s * 2 * 4 && s < 16 {
        s *= 2;
    }
    s
}

fn numel_pts(pts: &[usize]) -> usize {
    pts.iter().product()
}

/// The CUDA calling convention every kernel of a [`to_cuda`](crate::tensor::TensorGraph::to_cuda) program
/// follows: `extern "C" __global__ void main0(const float* b{j}…, float* out,
/// unsigned count)` with the reads in `WgslKernel::reads` order, launched
/// as `count / workgroup_size` blocks of `workgroup_size` threads.
pub const CUDA_ENTRY: &str = "main0";

impl TensorGraph {
    /// The program as CUDA C++ (for nvrtc): matmul-shaped tiles as mma
    /// tensor-core kernels, everything else the WGSL kernels transpiled.
    /// Same `WgslProgram` shape — sources, reads, counts — so a runtime
    /// drives both the same way.
    pub fn to_cuda(&self, outputs: &[TensorId]) -> WgslProgram {
        self.to_cuda_with(outputs, true)
    }

    /// [`to_cuda`](Self::to_cuda), with the native (tensor-core)
    /// lowerings on or off. Both programs have the same kernels in the
    /// same order with the same bindings, so a tuner can pick per kernel.
    pub fn to_cuda_with(&self, outputs: &[TensorId], native: bool) -> WgslProgram {
        self.to_cuda_variant(outputs, native.then_some(0))
    }

    /// [`to_cuda`](Self::to_cuda) with the matmul tiles in block shape
    /// `MMA_SHAPES[v]`, or every kernel transpiled for `None`. All
    /// variants have the same kernels in the same order with the same
    /// bindings, so a tuner can pick a variant per kernel.
    ///
    /// Short contractions read only by tiles are nested in them
    /// ([`schedule_nested_contractions`](Self::schedule_nested_contractions)):
    /// the tensor-core tiles recompute them per staged block, so every
    /// variant lowers that schedule.
    pub fn to_cuda_variant(&self, outputs: &[TensorId], variant: Option<usize>) -> WgslProgram {
        let mut nested = self.clone();
        nested.schedule_nested_contractions(outputs);
        // the attention backward as one kernel per key block, in every
        // variant alike (its dQ half is a zero fill the kernel adds to)
        if let Some(plan) = nested.attn_bwd_plan(outputs) {
            let mut fused = nested.clone();
            fused.attn_bwd_schedule(&plan);
            let mut program = fused.to_cuda_variant_scheduled(outputs, variant);
            if fused.attn_bwd_patch(&plan, &mut program, variant) {
                return program.canonical();
            }
        }
        nested.to_cuda_variant_scheduled(outputs, variant)
    }

    /// [`to_cuda_variant`](Self::to_cuda_variant) for a graph scheduled by
    /// [`place`](Self::place): `nest` is the placement's own decision.
    /// With `nest` false no contraction moves into a tile and no kernel
    /// fuses (the attention backward included), so a `Hbm` placement stays
    /// one in the program.
    pub fn to_cuda_placed(
        &self,
        outputs: &[TensorId],
        nest: bool,
        variant: Option<usize>,
    ) -> WgslProgram {
        if nest {
            self.to_cuda_variant(outputs, variant)
        } else {
            self.to_cuda_variant_scheduled(outputs, variant)
        }
    }

    fn to_cuda_variant_scheduled(
        &self,
        outputs: &[TensorId],
        variant: Option<usize>,
    ) -> WgslProgram {
        // Block formats (Q4K / Q5K / Q8x32) have no WGSL element type: the
        // program is laid out with them as f32, then their buffers get their
        // formats back and the kernels touching them are replaced below.
        let blocky = |g: &TensorGraph, i: usize| g.nodes[i].precision.storage.is_block();
        let mut plain = self.clone();
        for n in &mut plain.nodes {
            if n.precision.storage.is_block() {
                n.precision.storage = super::Format::F32;
            }
        }
        let mut program = super::wgsl::with_dialect(super::wgsl::Dialect::Cuda, || {
            plain.to_program_numbered(outputs, Target::Wgsl)
        });
        for b in &mut program.buffers {
            b.format = self.nodes[b.node.index()].precision.storage;
        }
        for k in &mut program.kernels {
            let root = k.node.index();
            let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
            if let Some(s) = self.nodes[root].int_mma {
                let (src, threads, blocks) = self
                    .cuda_int8_kernel(root, &reads, s)
                    .unwrap_or_else(|e| panic!("int8 MMA schedule on node {root}: {e}"));
                k.source = src;
                k.workgroup_size = threads as u32;
                k.count = (threads * blocks) as u32;
                continue;
            }
            assert!(
                !reads.iter().any(|&r| blocky(self, r)),
                "node {root} reads a block-format buffer without an int8 MMA schedule"
            );
            if blocky(self, root) {
                let st = self.nodes[root].precision.storage;
                assert!(
                    matches!(st, super::Format::Q8x32 | super::Format::Q8x32F),
                    "only Q8x32 / Q8x32F can be written"
                );
                let src = wgsl_to_cuda(&k.source, k.workgroup_size);
                k.source = super::cuda_int8::q8x32_writer(&src, k.points as usize, st == super::Format::Q8x32F)
                    .unwrap_or_else(|| panic!("node {root}: a Q8x32 store needs a one-thread-per-point kernel over a multiple of 32 points"));
                continue;
            }
            // FP8 operands: only the sm90 FP8 tile reads them (variant v
            // picks its shape)
            let fp8 = reads
                .iter()
                .any(|&j| self.nodes[j].precision.storage.is_fp8());
            let native = match variant {
                _ if fp8 => {
                    let v = variant.unwrap_or(0);
                    let sh = SM90_FP8_SHAPES[v % SM90_FP8_SHAPES.len()];
                    let fs = SM90_FP8_FLASH_SHAPES[v % SM90_FP8_FLASH_SHAPES.len()];
                    Some(
                        self.cuda_fp8_kernel_shaped(root, &reads, sh)
                            .or_else(|| {
                                self.cuda_fp8_kernel_shaped(root, &reads, SM90_FP8_SHAPES[0])
                            })
                            .or_else(|| self.cuda_flash_kernel_shaped(root, &reads, fs))
                            .or_else(|| {
                                self.cuda_flash_kernel_shaped(
                                    root,
                                    &reads,
                                    SM90_FP8_FLASH_SHAPES[0],
                                )
                            })
                            .unwrap_or_else(|| {
                                panic!(
                                    "kernel {root} reads FP8 storage but is not an FP8 tile \
                                     (a contraction of two FP8 operands contiguous along k, \
                                     or attention with FP8 Q, K, Vᵀ and an E4M3 operand format)"
                                )
                            }),
                    )
                }
                Some(v) if self.nodes[root].tile.0 > 0 => v
                    .checked_sub(MMA_SHAPES.len())
                    .and_then(|h| {
                        // variants past MMA_SHAPES: the Hopper wgmma tiles
                        self.cuda_mma_kernel_shaped(
                            root,
                            &reads,
                            SM90_MMA_SHAPES[h % SM90_MMA_SHAPES.len()],
                        )
                    })
                    .or_else(|| {
                        self.cuda_mma_kernel_shaped(root, &reads, MMA_SHAPES[v % MMA_SHAPES.len()])
                    })
                    .or_else(|| {
                        // variants past MMA_SHAPES: the Hopper flash shapes
                        let fs = match v.checked_sub(MMA_SHAPES.len()) {
                            Some(h) => {
                                SM90_FLASH_SHAPES
                                    [(h / SM90_MMA_SHAPES.len()) % SM90_FLASH_SHAPES.len()]
                            }
                            None => FLASH_SHAPES[v % FLASH_SHAPES.len()],
                        };
                        self.cuda_flash_kernel_shaped(root, &reads, fs)
                    }),
                _ => None,
            };
            match native {
                Some(src) => {
                    k.source = src;
                    let threads = src_define(&k.source, "TANG_THREADS");
                    k.workgroup_size = threads as u32;
                    k.count = {
                        let n_tiles: usize = src_define(&k.source, "TANG_TILES");
                        (n_tiles * threads) as u32
                    };
                }
                None => k.source = wgsl_to_cuda(&k.source, k.workgroup_size),
            }
        }
        program.canonical()
    }
}

impl TensorGraph {
    /// Whether every kernel of the CUDA program for `outputs` that reads
    /// an FP8 buffer lowers to an FP8 tile (a contraction of two FP8
    /// operands contiguous along k, or FP8 attention). A precision search
    /// skips FP8 candidates that do not: no other kernel reads FP8.
    pub fn cuda_fp8_lowers(&self, outputs: &[TensorId]) -> bool {
        // FP8 is an operand format: inputs only (no kernel writes it)
        if self
            .nodes
            .iter()
            .any(|n| n.precision.storage.is_fp8() && !matches!(n.op, Op::Input(_)))
        {
            return false;
        }
        let mut g = self.clone();
        g.schedule_nested_contractions(outputs);
        let program = super::wgsl::with_dialect(super::wgsl::Dialect::Cuda, || {
            g.to_program_numbered(outputs, Target::Wgsl)
        });
        program.kernels.iter().all(|k| {
            let root = k.node.index();
            let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
            !reads.iter().any(|&j| g.nodes[j].precision.storage.is_fp8())
                || g.cuda_fp8_kernel_shaped(root, &reads, SM90_FP8_SHAPES[0])
                    .is_some()
                || g.cuda_flash_kernel_shaped(root, &reads, SM90_FP8_FLASH_SHAPES[0])
                    .is_some()
        })
    }

    /// The online-softmax reductions of the attention kernels in the CUDA
    /// program for `outputs`, with the buffers each stages as plain
    /// operands (Q, K, V): where an FP8 precision move sets the
    /// reduction's `operand` format.
    pub fn cuda_flash_reductions(&self, outputs: &[TensorId]) -> Vec<(TensorId, Vec<TensorId>)> {
        let mut g = self.clone();
        g.schedule_nested_contractions(outputs);
        let program = super::wgsl::with_dialect(super::wgsl::Dialect::Cuda, || {
            g.to_program_numbered(outputs, Target::Wgsl)
        });
        program
            .kernels
            .iter()
            .filter_map(|k| {
                let root = k.node.index();
                if g.nodes[root].tile.0 == 0 {
                    return None;
                }
                let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
                let f = g.flash_tile(root, &reads)?;
                let rc = g.comp_at(f.red_node)?;
                let mut bufs: Vec<TensorId> = f
                    .terms
                    .iter()
                    .flat_map(|(qb, _, kb, _)| [*qb, *kb])
                    .map(|b| TensorId(b as u32))
                    .collect();
                if let ENode::Var(v) = rc.body.nodes_slice()[rc.body_outs[2].index() as usize] {
                    bufs.push(TensorId(g.storage(rc.args[v as usize].src).n() as u32));
                }
                Some((TensorId(f.red_node as u32), bufs))
            })
            .collect()
    }

    /// The tensor-core kernels of [`to_cuda`](Self::to_cuda) and, for
    /// each, the buffers its tiles stage as plain operands (a matmul's two
    /// factors, attention's Q, K and V): stored in f16 (bf16) together,
    /// the kernel runs on m16n8k16 f16 (bf16) fragments. What a precision
    /// search moves as one candidate.
    pub fn cuda_tile_operands(&self, outputs: &[TensorId]) -> Vec<(TensorId, Vec<TensorId>)> {
        let mut g = self.clone();
        g.schedule_nested_contractions(outputs);
        let program = super::wgsl::with_dialect(super::wgsl::Dialect::Cuda, || {
            g.to_program_numbered(outputs, Target::Wgsl)
        });
        program
            .kernels
            .iter()
            .filter_map(|k| {
                let root = k.node.index();
                if g.nodes[root].tile.0 == 0 {
                    return None;
                }
                let reads: Vec<usize> = k.reads.iter().map(|t| t.index()).collect();
                let mut bufs: Vec<usize> = Vec::new();
                if g.cuda_mma_kernel(root, &reads).is_some() {
                    let b = g.bilinear_tile(root, &reads)?;
                    for t in b.outs.iter().flatten() {
                        if t.map_a.is_none() {
                            bufs.push(t.buf_a);
                        }
                        if t.map_b.is_none() {
                            bufs.push(t.buf_b);
                        }
                    }
                } else if g.cuda_flash_kernel(root, &reads).is_some() {
                    let f = g.flash_tile(root, &reads)?;
                    if f.score_buf.is_none() {
                        for (qb, _, kb, _) in &f.terms {
                            bufs.push(*qb);
                            bufs.push(*kb);
                        }
                    }
                    let rc = g.comp_at(f.red_node)?;
                    let e_v = rc.body_outs[2];
                    if let ENode::Var(v) = rc.body.nodes_slice()[e_v.index() as usize] {
                        let a = &rc.args[v as usize];
                        if !matches!(g.nodes[a.src.n()].op, Op::Iota(_)) {
                            bufs.push(g.storage(a.src).n());
                        }
                    }
                } else {
                    return None;
                }
                bufs.sort_unstable();
                bufs.dedup();
                (!bufs.is_empty()).then(|| {
                    (
                        k.node,
                        bufs.into_iter().map(|b| TensorId(b as u32)).collect(),
                    )
                })
            })
            .collect()
    }
}

fn src_define(src: &str, name: &str) -> usize {
    src.lines()
        .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
        .and_then(|v| v.trim().parse().ok())
        .expect("define")
}

/// Generated WGSL is a small, regular language: typed `fn`s over `f32`,
/// `u32` and `array<f32, N>`, `let`/`var` bindings, `for` loops over
/// `u32`, workgroup arrays, and a `main` over builtin ids. That maps to
/// CUDA C++ line by line with a short prelude (`Arr<N>` for arrays,
/// `Vec4` for vec4, `select` as a macro) — no second emitter.
pub fn wgsl_to_cuda(wgsl: &str, workgroup_size: u32) -> String {
    let mut out = String::new();
    out.push_str(CUDA_PRELUDE);
    if wgsl.contains("tang_") {
        out.push_str(HALF_PRELUDE);
    }
    let mut reads: Vec<(String, &str)> = Vec::new();
    let mut out_elem = "float";
    let mut body = String::new();
    for line in wgsl.lines() {
        let t = line.trim_start();
        let indent = &line[..line.len() - t.len()];
        if t.starts_with("struct Params") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("@group(0) @binding(") {
            // `N) var<storage, read> bJ: array<f32>;`
            if rest.contains("var<uniform>") {
                continue;
            }
            let name = rest
                .split("var<")
                .nth(1)
                .and_then(|r| r.split('>').nth(1))
                .map(|r| {
                    r.trim()
                        .trim_end_matches(':')
                        .split(':')
                        .next()
                        .unwrap()
                        .trim()
                        .to_string()
                })
                .expect("binding name");
            // `array<f16>` / `array<bf16>`: 16-bit elements (Dialect::Cuda)
            let elem = if rest.contains("array<f16>") || rest.contains("array<bf16>") {
                "unsigned short"
            } else {
                "float"
            };
            if name == "out" {
                body.push_str(&format!("__shared__ {elem}* out;\n"));
                out_elem = elem;
            } else {
                body.push_str(&format!("__shared__ const {elem}* {name};\n"));
                reads.push((name, elem));
            }
            continue;
        }
        if t.starts_with("@compute") {
            continue;
        }
        if t.starts_with("fn main(") {
            let mut params: Vec<String> = reads
                .iter()
                .map(|(n, e)| format!("const {e}* p_{n}"))
                .collect();
            params.push(format!("{out_elem}* p_out"));
            params.push("unsigned count".into());
            body.push_str(&format!(
                "extern \"C\" __global__ void __launch_bounds__({workgroup_size}) main0({}) {{\n",
                params.join(", ")
            ));
            body.push_str("    if (threadIdx.x == 0u) {\n");
            for (n, _) in &reads {
                body.push_str(&format!("        {n} = p_{n};\n"));
            }
            body.push_str("        out = p_out;\n        params.count = count;\n    }\n    __syncthreads();\n");
            continue;
        }
        if let Some(rest) = t.strip_prefix("var<workgroup> ") {
            // `name: array<T, N>;`
            let (name, ty) = rest
                .trim_end_matches(';')
                .split_once(": ")
                .expect("workgroup var");
            body.push_str(&format!(
                "{}__shared__ {};\n",
                indent,
                cuda_array_decl(name, ty)
            ));
            continue;
        }
        if let Some(rest) = t.strip_prefix("fn ") {
            // `name(params) -> T {`
            let (sig, ret) = rest.split_once(" -> ").expect("fn return");
            let ret = ret.trim_end_matches(" {").trim_end_matches('{').trim();
            let (name, params) = sig.split_once('(').expect("fn params");
            let params = params.trim_end_matches(')');
            let params: Vec<String> = split_top(params)
                .into_iter()
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let (n, ty) = p.split_once(": ").expect("param");
                    format!("{} {n}", cuda_type(ty))
                })
                .collect();
            body.push_str(&format!(
                "{indent}__device__ __forceinline__ {} {name}({}) {{\n",
                cuda_type(ret),
                params.join(", ")
            ));
            continue;
        }
        body.push_str(indent);
        body.push_str(&cuda_stmt(t));
        body.push('\n');
    }
    out.push_str("__shared__ Params params;\n");
    out.push_str(&body);
    out
}

/// 16-bit conversions for `Dialect::Cuda` buffers (PTX `cvt`, so no
/// headers), added only to kernels that use them: f32 sources stay
/// byte-identical.
pub(super) const HALF_PRELUDE: &str = r#"__device__ __forceinline__ float tang_h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ unsigned short tang_f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ float tang_b2f(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ unsigned short tang_f2b(float f) { unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
"#;

const CUDA_PRELUDE: &str = r#"// Auto-generated by tang-compile: WGSL kernel as CUDA C++
struct Params { unsigned count; };
template <int N> struct Arr {
    float v[N];
    __device__ __forceinline__ float& operator[](unsigned i) { return v[i]; }
    __device__ __forceinline__ float operator[](unsigned i) const { return v[i]; }
};
struct Vec4 {
    float v[4];
    __device__ __forceinline__ float& operator[](unsigned i) { return v[i]; }
    __device__ __forceinline__ float operator[](unsigned i) const { return v[i]; }
};
#define select(f, t, c) ((c) ? (t) : (f))
__device__ __forceinline__ unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
"#;

/// Split on `, ` outside angle brackets (`a: array<f32, 2>, b: f32`).
pub(super) fn split_top(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b',' if depth == 0 && b.get(i + 1) == Some(&b' ') => {
                out.push(&s[start..i]);
                start = i + 2;
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

fn cuda_type(ty: &str) -> String {
    let ty = ty.trim();
    if ty == "f32" {
        return "float".into();
    }
    if ty == "u32" {
        return "unsigned".into();
    }
    if ty == "i32" {
        return "int".into();
    }
    if ty == "bool" {
        return "bool".into();
    }
    if let Some(n) = ty
        .strip_prefix("array<f32, ")
        .and_then(|r| r.strip_suffix('>'))
    {
        return format!("Arr<{n}>");
    }
    if ty == "vec4<f32>" {
        return "Vec4".into();
    }
    panic!("wgsl type {ty}")
}

/// `name: array<T, N>` → `T name[N]`, with `T` itself an `Arr<M>` for a
/// nested array (workgroup accumulators) and `Vec4` for vec4 stages.
fn cuda_array_decl(name: &str, ty: &str) -> String {
    let ty = ty.trim();
    if let Some(inner) = ty.strip_prefix("array<").and_then(|r| r.strip_suffix('>')) {
        if let Some((elem, n)) = inner.rsplit_once(", ") {
            let elem = elem.trim();
            // outer dimension is a C array; the element type is a value type
            if elem.starts_with("array<") || elem == "vec4<f32>" {
                return format!("{} {name}[{n}]", cuda_type(elem));
            }
        }
    }
    format!("{} {name}", cuda_type(ty))
}

/// Float literals get an `f` suffix so the math stays single precision.
pub(super) fn float_literals(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let prev_ident = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if c.is_ascii_digit() && !prev_ident {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i < b.len() && b[i] == b'.' {
                is_float = true;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            // an exponent makes it a float too (`1e30`)
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            out.push_str(&s[start..i]);
            if is_float {
                out.push('f');
            }
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// One statement line. Bindings become `auto`, typed `var`s their type,
/// loops over `u32`, casts and constructors their C++ spellings.
fn cuda_stmt(t: &str) -> String {
    let mut s = t.to_string();
    if let Some(rest) = s.strip_prefix("let ") {
        // `let x: T = e;` keeps its type; `let x = e;` is `auto`
        s = match rest.split_once(" = ") {
            Some((l, r)) if l.contains(": ") => {
                let (n, ty) = l.split_once(": ").unwrap();
                format!("const {} = {r}", cuda_array_decl(n, ty))
            }
            _ => format!("auto {rest}"),
        };
    } else if let Some(rest) = s.strip_prefix("var ") {
        // `var x: T = e;` or `var x: T;` or `var x = e;`
        let (lhs, rhs) = match rest.split_once(" = ") {
            Some((l, r)) => (l, Some(r)),
            None => (rest.trim_end_matches(';'), None),
        };
        let decl = match lhs.split_once(": ") {
            Some((n, ty)) => cuda_array_decl(n, ty),
            None => format!("auto {lhs}"),
        };
        s = match rhs {
            Some(r) => format!("{decl} = {r}"),
            None => format!("{decl};"),
        };
    } else if let Some(rest) = s.strip_prefix("for (var ") {
        // `for (var i: u32 = a; …`
        let (n, rest) = rest.split_once(": u32 = ").expect("for var");
        s = format!("for (unsigned {n} = {rest}");
    }
    s = s
        .replace("array<f32, ", "Arr<")
        .replace("vec4<f32>(", "Vec4{")
        .replace("u32(", "(unsigned)(")
        .replace("i32(", "(int)(")
        .replace("f32(", "(float)(")
        .replace("workgroupBarrier()", "__syncthreads()")
        .replace("gid.x", "(blockIdx.x * blockDim.x + threadIdx.x)")
        .replace("gid.y", "blockIdx.y")
        .replace("wid.x", "blockIdx.x")
        .replace("wid.y", "blockIdx.y")
        .replace("lid.x", "threadIdx.x");
    // `Arr<N>(a, b)` → `Arr<N>{{a, b}}`; `Vec4{a, …)` → `Vec4{{a, …}}`
    s = rewrite_ctor(&s, "Arr<");
    s = rewrite_ctor(&s, "Vec4{");
    float_literals(&s)
}

/// Turn `PREFIX…(args)` constructor calls into brace initialization.
fn rewrite_ctor(s: &str, prefix: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find(prefix) {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        // find the opening paren/brace of the argument list
        let open = if prefix.ends_with('{') {
            prefix.len() - 1
        } else {
            match after.find('(') {
                Some(p) if after[..p].contains('>') => p,
                _ => {
                    out.push_str(&after[..prefix.len()]);
                    rest = &after[prefix.len()..];
                    continue;
                }
            }
        };
        // Arr<N> followed by a space (a declaration), not a constructor
        if !prefix.ends_with('{') && after[..open].contains(' ') {
            out.push_str(&after[..prefix.len()]);
            rest = &after[prefix.len()..];
            continue;
        }
        let mut depth = 0usize;
        let mut close = None;
        for (k, ch) in after[open..].char_indices() {
            match ch {
                '(' | '{' => depth += 1,
                ')' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + k);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.expect("ctor close");
        out.push_str(&after[..open]);
        out.push_str("{{");
        out.push_str(&after[open + 1..close]);
        out.push_str("}}");
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// Attention on tensor cores
// ---------------------------------------------------------------------------

/// The shape of a fused attention: an online-softmax reduction over keys
/// whose score is an inlined bilinear comp (`Σ_d q[i,d]·k[j,d]`, possibly
/// several such terms), plus a mask over (i, j); whose value is a read
/// along (j, lane) — and a map reading it point-for-point (the
/// normalizer). Recognized by [`TensorGraph::flash_tile`].
struct Flash {
    red_node: usize,
    /// index of the arg carrying the score comp, and that comp's node
    score_arg: usize,
    score_node: usize,
    /// (q buffer, q address over the score space, k buffer, k address)
    terms: Vec<(usize, Addr, usize, Addr)>,
    /// the score read from a buffer (storage, address over the reduction
    /// space) instead of an inlined bilinear comp: the state-materializing
    /// forward of a training step, whose scores are kept for the backward
    score_buf: Option<(usize, Addr)>,
    /// whether `root` is a normalizer over the reduction (else the root is
    /// the reduction, writing its `(m, l, acc)` state)
    epilogue: bool,
    /// score space dim ↔ reduction space dim, for the score's point dims
    score_to_red: Vec<usize>,
    /// reduction space: query dim, lane dim, others (incl. unit dims)
    dy: usize,
    dx: usize,
    others: Vec<usize>,
    pts: Vec<usize>,
    m: usize,
    n_keys: usize,
    d_lanes: usize,
    kd: usize,
    /// a constant factor on every product of the score (the 1/√d the
    /// query map folds into the score body)
    scale: f64,
}

use super::msl_mma::Addr;
use super::vars_reachable;
use super::{Arg, Op};
use tang_expr::Node as ENode;

impl TensorGraph {
    fn flash_tile(&self, root: usize, reads: &[usize]) -> Option<Flash> {
        let c = self.comp_at(root)?;
        let red_node = self.tiled_reduction(root)?;
        let epilogue = red_node != root;
        let rc = self.comp_at(red_node)?;
        let m = rc.monoid.as_ref()?;
        if m.kind() != super::MonoidKind::OnlineSoftmax || m.width != 3 || rc.red.len() != 1 {
            return None;
        }
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        let is_buffer = |j: usize| reads.contains(&storage(j));
        let pts = super::fixed(rc.points());
        let rank = rc.rank();
        let jdim = rank - 1;
        let dx = (0..pts.len()).rev().find(|&d| pts[d] > 1)?;
        let dy = super::tile::largest_other(&pts, dx)?;
        let others: Vec<usize> = (0..pts.len()).filter(|&d| d != dx && d != dy).collect();
        let d_lanes = pts[dx];
        if !d_lanes.is_multiple_of(16) || d_lanes > 256 {
            return None;
        }
        // body outs: (score + mask, 1, value)
        let nodes = rc.body.nodes_slice();
        let [e_s, e_one, e_v] = rc.body_outs[..] else {
            return None;
        };
        match nodes[e_one.index() as usize] {
            ENode::Lit(b) if f64::from_bits(b) == 1.0 => {}
            _ => return None,
        }
        // the one inlined comp the score reads; everything else it touches
        // is an iota or a buffer not varying along the lanes
        let s_vars = vars_reachable(&rc.body, e_s);
        let mut score_arg = None;
        // buffers the score reads along both queries and keys: candidates
        // for a materialized score
        let mut score_bufs = Vec::new();
        for &v in &s_vars {
            let a = &rc.args[v as usize];
            let p = a.src.n();
            match &self.nodes[p].op {
                Op::Comp(_) if self.nodes[p].inline => {
                    if score_arg.is_some() {
                        return None;
                    }
                    score_arg = Some(v as usize);
                }
                Op::Iota(_) => {}
                _ if is_buffer(p) => {
                    let addr = self.flat_addr(a, rank);
                    if addr.coefs[dx] != 0 {
                        return None;
                    }
                    if addr.coefs[dy] != 0 && addr.coefs[jdim] != 0 {
                        score_bufs.push(v as usize);
                    }
                }
                _ => return None,
            }
        }
        let mut score_buf = None;
        if score_arg.is_none() {
            let [v] = score_bufs[..] else {
                return None;
            };
            let a = &rc.args[v];
            score_buf = Some((storage(a.src.n()), self.flat_addr(a, rank)));
            score_arg = Some(v);
        } else if !score_bufs.is_empty() {
            return None;
        }
        let score_arg = score_arg?;
        // the value: buffers/iotas not varying along the query dim
        for &v in &vars_reachable(&rc.body, e_v) {
            let a = &rc.args[v as usize];
            let p = a.src.n();
            let ok = match &self.nodes[p].op {
                Op::Iota(_) => true,
                _ => is_buffer(p),
            };
            if !ok || self.flat_addr(a, rank).coefs[dy] != 0 {
                return None;
            }
        }
        if score_buf.is_some() {
            if epilogue {
                self.epilogue_read(c, red_node)?;
                for a in &c.args {
                    if a.src.n() != red_node && !is_buffer(a.src.n()) {
                        return None;
                    }
                }
            }
            return Some(Flash {
                red_node,
                score_arg,
                score_node: rc.args[score_arg].src.n(),
                terms: Vec::new(),
                score_buf,
                epilogue,
                score_to_red: Vec::new(),
                dy,
                dx,
                others,
                m: pts[dy],
                n_keys: rc.red[0].n(),
                d_lanes,
                kd: 8,
                pts,
                scale: 1.0,
            });
        }
        // the score comp: Σ_d of products, each q-like × k-like
        let sa = &rc.args[score_arg];
        let score_node = sa.src.n();
        let sc = self.comp_at(score_node)?;
        if sc.red.len() != 1
            || sc.monoid.as_ref()?.kind() != super::MonoidKind::Sum
            || sc.width() != 1
        {
            return None;
        }
        // score dims ↔ reduction dims: the view is a pick
        let score_to_red: Vec<usize> = sa
            .view
            .forms
            .iter()
            .map(|f| {
                let d = (0..rank).find(|&d| f.coefs[d] != 0)?;
                (f.coefs[d] == 1 && f.offset == 0).then_some(d)
            })
            .collect::<Option<_>>()?;
        if score_to_red.len() != sc.points().len() {
            return None;
        }
        let s_i = score_to_red.iter().position(|&d| d == dy)?;
        let s_j = score_to_red.iter().position(|&d| d == jdim)?;
        let srank = sc.rank();
        let sd = srank - 1;
        let mut pairs = Vec::new();
        let snodes = sc.body.nodes_slice();
        // a product of two vars and any literals: (p, q, Π literals)
        fn factors(
            nodes: &[ENode],
            e: tang_expr::ExprId,
            vars: &mut Vec<usize>,
            k: &mut f64,
        ) -> bool {
            match nodes[e.index() as usize] {
                ENode::Mul(a, b) => factors(nodes, a, vars, k) && factors(nodes, b, vars, k),
                ENode::Var(p) => {
                    vars.push(p as usize);
                    true
                }
                ENode::Lit(b) => {
                    *k *= f64::from_bits(b);
                    true
                }
                _ => false,
            }
        }
        fn products(
            nodes: &[ENode],
            e: tang_expr::ExprId,
            out: &mut Vec<(usize, usize, f64)>,
        ) -> bool {
            match nodes[e.index() as usize] {
                ENode::Add(a, b) => products(nodes, a, out) && products(nodes, b, out),
                ENode::Mul(..) => {
                    let (mut vars, mut k) = (Vec::new(), 1.0);
                    if !factors(nodes, e, &mut vars, &mut k) || vars.len() != 2 {
                        return false;
                    }
                    out.push((vars[0], vars[1], k));
                    true
                }
                _ => false,
            }
        }
        if !products(snodes, sc.body_outs[0], &mut pairs) {
            return None;
        }
        // one scale for every term (it multiplies the summed score)
        let scale = pairs.first().map_or(1.0, |t| t.2);
        if pairs.iter().any(|t| t.2 != scale) {
            return None;
        }
        let mut terms = Vec::new();
        for (p, q, _) in pairs {
            let (ap, aq) = (&sc.args[p], &sc.args[q]);
            if !is_buffer(ap.src.n()) || !is_buffer(aq.src.n()) {
                return None;
            }
            let (addr_p, addr_q) = (self.flat_addr(ap, srank), self.flat_addr(aq, srank));
            let q_like = |a: &Addr| a.coefs[s_j] == 0 && a.coefs[sd] != 0;
            let k_like = |a: &Addr| a.coefs[s_i] == 0 && a.coefs[sd] != 0;
            if q_like(&addr_p) && k_like(&addr_q) {
                terms.push((storage(ap.src.n()), addr_p, storage(aq.src.n()), addr_q));
            } else if q_like(&addr_q) && k_like(&addr_p) {
                terms.push((storage(aq.src.n()), addr_q, storage(ap.src.n()), addr_p));
            } else {
                return None;
            }
        }
        let kd = sc.red[0].n();
        if kd % 8 != 0 {
            return None;
        }
        // the normalizer reads the reduction point-for-point and nothing
        // but buffers otherwise
        if epilogue {
            self.epilogue_read(c, red_node)?;
            for a in &c.args {
                if a.src.n() != red_node && !is_buffer(a.src.n()) {
                    return None;
                }
            }
        }
        Some(Flash {
            red_node,
            score_arg,
            score_node,
            terms,
            score_buf: None,
            epilogue,
            score_to_red,
            scale,
            dy,
            dx,
            others,
            m: pts[dy],
            n_keys: rc.red[0].n(),
            d_lanes,
            kd,
            pts,
        })
    }

    /// A lead dim of the flash tile that neither the keys nor anything
    /// else of the reduction but the queries depends on, for
    /// `cuda_flash_kernel` to fold into its rows (grouped-query heads).
    fn flash_fold(&self, f: &Flash, rc: &super::Comp) -> Option<usize> {
        if f.score_buf.is_some() || std::env::var_os("TANG_FLASH_NOFOLD").is_some() {
            return None;
        }
        let rank = rc.rank();
        f.others.iter().copied().rev().find(|&fd| {
            f.pts[fd] > 1
                && f.terms.iter().all(|(_, _, _, ka)| {
                    (0..f.score_to_red.len())
                        .all(|sd| f.score_to_red[sd] != fd || ka.coefs[sd] == 0)
                })
                && rc.args.iter().enumerate().all(|(k, a)| {
                    k == f.score_arg
                        || match &self.nodes[a.src.n()].op {
                            Op::Iota(dd) => a.view.forms[*dd].coefs[fd] == 0,
                            _ => self.flat_addr(a, rank).coefs[fd] == 0,
                        }
                })
        })
    }

    /// CUDA source for `root` as a tensor-core flash-attention tile in
    /// the default shape ([`FLASH_SHAPES`]`[0]`).
    pub fn cuda_flash_kernel(&self, root: usize, reads: &[usize]) -> Option<String> {
        self.cuda_flash_kernel_shaped(root, reads, FLASH_SHAPES[0])
    }

    /// CUDA source for `root` as a flash-attention tile on `mma.sync`
    /// m16n8k8 TF32, or `None` when it is not attention-shaped. Each warp
    /// owns 16 rows (queries × folded heads); per key chunk it computes
    /// `S = Σ_t Q_t·K_tᵀ` into accumulator registers, runs the comp's
    /// mask and the online softmax on them (row statistics reduced
    /// across the fragment's quad), and feeds the same registers as the
    /// A operand of `O += P·V`: the accumulator's column pair (2tq,
    /// 2tq + 1) is the A fragment's k pair (tq, tq + 4) once the keys of
    /// each 8-wide step are permuted, which a reduction allows. O and the
    /// row statistics stay in registers across chunks; K and V come
    /// through a `cp.async` ring of `sh.stages` buffers; `sh.kw` warp
    /// groups split each chunk's keys and merge their states at the end.
    pub fn cuda_flash_kernel_shaped(
        &self,
        root: usize,
        reads: &[usize],
        sh: FlashShape,
    ) -> Option<String> {
        let f = self.flash_tile(root, reads)?;
        let c = self.comp_at(root)?;
        let rc = self.comp_at(f.red_node)?;
        let rank = rc.rank();
        let jdim = rank - 1;
        let storage = |j: usize| self.storage(TensorId(j as u32)).n();
        if self.nodes[f.red_node].precision.accum != super::Format::F32 {
            return None;
        }
        let fmt = |j: usize| self.nodes[j].precision.storage;
        let ld = |j: usize, idx: &str| -> String {
            match fmt(j) {
                super::Format::F32 => format!("b{j}[{idx}]"),
                super::Format::F16 => format!("tang_h2f(b{j}[{idx}])"),
                super::Format::BF16 => format!("tang_b2f(b{j}[{idx}])"),
                super::Format::Q4K
                | super::Format::Q5K
                | super::Format::Q6K
                | super::Format::Q8x32
                | super::Format::Q8x32F
                | super::Format::Fp8B32
                | super::Format::Mxfp4 => {
                    unreachable!("block formats lower only through the int8 MMA kernel")
                }
                // FP8 operands are staged by the tile; a stray scalar read
                // fails to compile
                _ => format!("tang_fp8_scalar_read(b{j}[{idx}])"),
            }
        };
        let elem = |f: super::Format| {
            if f == super::Format::F32 {
                "float"
            } else if f.is_fp8() {
                "unsigned char"
            } else {
                "unsigned short"
            }
        };
        let typed = reads
            .iter()
            .chain([&root])
            .any(|&j| fmt(j) != super::Format::F32);
        let sh = match std::env::var("TANG_FLASH_SHAPE") {
            Ok(v) => {
                let p: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                match p[..] {
                    [bi, bj, stages, kw] => FlashShape {
                        bi,
                        bj,
                        stages,
                        kw,
                        sm90: None,
                    },
                    _ => sh,
                }
            }
            Err(_) => sh,
        };
        let hop = sh.sm90;
        // FP8 operands: the Hopper schedule only
        if hop.is_none() && reads.iter().chain([&root]).any(|&j| fmt(j).is_fp8()) {
            return None;
        }
        let (bj, kw) = (sh.bj, if hop.is_some() { 1 } else { sh.kw });
        // keys per pipeline step, over all warp groups
        let bjs = bj * kw;
        if !bj.is_multiple_of(8) || sh.stages < 2 || kw == 0 {
            return None;
        }
        let lead: usize = f.others.iter().map(|&d| f.pts[d]).product();
        // a lead dim the keys and values do not depend on (the query
        // heads sharing a grouped K/V head) folds into the block's rows:
        // K and V are staged once for all of them
        let fold = self.flash_fold(&f, rc);
        if hop.is_some() && fold.is_some() {
            return None;
        }
        let nf = fold.map_or(1, |fd| f.pts[fd]);
        let bi = match std::env::var("TANG_FLASH_BI")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(b) => b,
            None if hop.is_some() => 64 * hop.unwrap().consumers,
            None if sh.bi > 0 => sh.bi,
            None if fold.is_some() => {
                // the largest query slice that still fills the device
                let blocks = |b: usize| f.m.div_ceil(b) * lead / nf;
                [64, 32, 16]
                    .into_iter()
                    .find(|&b| blocks(b) >= 82 && nf * b <= 128)
                    .unwrap_or(16)
            }
            None if f.m.div_ceil(64) * lead >= 164 => 64,
            None => 32,
        };
        // BI queries of each of the nf folded heads: R rows, 16 per warp
        #[allow(non_snake_case)]
        let (BI, R) = (bi, nf * bi);
        if !R.is_multiple_of(16) {
            return None;
        }
        let row_warps = R / 16;
        #[allow(non_snake_case)]
        let T = match hop {
            Some(h) => 128 * (h.consumers + 1),
            None => 32 * row_warps * kw,
        };
        if T > 1024 {
            return None;
        }
        let d = f.d_lanes;
        let kd = f.kd;
        let nq = kd / 8;
        let n_terms = f.terms.len();
        let has_k = f.score_buf.is_none();
        // the value: a plain read, or an expression evaluated at staging
        let e_v = rc.body_outs[2];
        let plain_v = match rc.body.nodes_slice()[e_v.index() as usize] {
            ENode::Var(v) => {
                let a = &rc.args[v as usize];
                (!matches!(self.nodes[a.src.n()].op, Op::Iota(_)))
                    .then(|| (storage(a.src.n()), self.flat_addr(a, rank)))
            }
            _ => None,
        };
        // Q, K and V stored in f16 (bf16): m16n8k16 fragments for S and
        // for P·V, P rounded to the format from the score registers.
        // Plain operands must agree.
        let mut plain: Vec<super::Format> = Vec::new();
        if has_k {
            for (qb, _, kb, _) in &f.terms {
                plain.push(fmt(*qb));
                plain.push(fmt(*kb));
            }
        }
        if let Some((vb, _)) = &plain_v {
            plain.push(fmt(*vb));
        }
        // FP8 Q, K and V (each with its own block scales): the Hopper
        // FP8 schedule below
        let fp8 = !plain.is_empty() && plain.iter().all(|f| f.is_fp8());
        if !fp8 && plain.iter().any(|f| f.is_fp8()) {
            return None;
        }
        let half = match plain.first() {
            _ if fp8 => None,
            None => None,
            Some(&g) if plain.iter().any(|&x| x != g) => return None,
            Some(&super::Format::F32) => None,
            Some(&g) => Some(g),
        };
        if half.is_some() && (!kd.is_multiple_of(16) || !bj.is_multiple_of(16)) {
            return None;
        }
        let esz = if half.is_some() { 2 } else { 4 };
        // fragments in registers: Q (n_terms × nq × 4), S (bj/8 × 4), O (d/8 × 4)
        if hop.is_none() && n_terms * nq * 4 + bj / 2 + d / 2 > 160 {
            return None;
        }
        // row strides: K rows are read 8 keys × 4 float2 per half warp
        // (≡ 8 mod 32 is conflict-free), V rows 4 keys × 4 float2 (≡ 4)
        // halves: K and Q rows are read 8 rows × 4 words (a row stride of
        // an odd number of 16-byte units is conflict-free), V rows 4 × 8
        // words two rows apart (likewise)
        let (ldk, ldv) = if half.is_some() {
            (kd + 8, d + 8)
        } else {
            (kd + 8, d + 4)
        };
        let k_floats = if has_k { n_terms * bjs * ldk } else { 0 };
        let stage_floats = k_floats + bjs * ldv;
        let q_floats = if has_k { n_terms * R * ldk } else { 0 };
        let ldo = d + 4;
        let merge_floats = if kw > 1 {
            2 * kw * R + (kw - 1) * R * ldo
        } else {
            0
        };
        // staging and Q in elements of `esz` bytes; the merge in floats
        let smem_bytes = (sh.stages * stage_floats * esz)
            .max(q_floats * esz)
            .max(merge_floats * 4);
        if hop.is_none() && smem_bytes > 99 * 1024 {
            return None;
        }
        let fp8_plan = match hop {
            Some(h) if fp8 => {
                Some(self.fp8_flash_plan(&f, rc, plain_v.as_ref()?, h, bj, kd, d, sh.stages)?)
            }
            _ => None,
        };
        let smem_bytes = match hop {
            None => smem_bytes,
            Some(_) if fp8 => fp8_plan.as_ref()?.smem,
            Some(h) => {
                // Q, the K/V ring, the mbarriers; 1024 bytes of alignment slack
                let ok = half.is_some()
                    && has_k
                    && n_terms == 1
                    && kd.is_multiple_of(64)
                    && d.is_multiple_of(64)
                    && d <= 256
                    && bj.is_multiple_of(16)
                    && bj <= 256
                    && plain_v.as_ref().is_some_and(|(_, va)| {
                        va.coefs[f.dx] == 1
                            && va.offset % 8 == 0
                            && va
                                .coefs
                                .iter()
                                .enumerate()
                                .all(|(k, &c)| k == f.dx || c % 8 == 0)
                    })
                    && f.terms.iter().all(|(_, qa, _, ka)| {
                        [qa, ka].iter().all(|a| {
                            let inner = f.score_to_red.len();
                            a.coefs[inner] == 1
                                && a.offset % 8 == 0
                                && a.coefs
                                    .iter()
                                    .enumerate()
                                    .all(|(k, &c)| k == inner || c % 8 == 0)
                        })
                    });
                let b = 64 * h.consumers * kd * 2
                    + sh.stages * bj * (kd + d) * 2
                    + 16 * sh.stages
                    + 1024;
                if !ok || b > 227 * 1024 {
                    return None;
                }
                b
            }
        };
        let tiles_i = f.m.div_ceil(BI);
        let n_tiles = tiles_i * lead / nf;
        let n_chunks = f.n_keys.div_ceil(bjs);
        let w_out = c.width();

        // coordinate names in the reduction's space
        let coord = |dim: usize| -> String {
            if dim == jdim {
                "(int)jj".into()
            } else if dim == f.dy {
                "(int)ii".into()
            } else if dim == f.dx {
                "(int)nn".into()
            } else {
                format!("(int)p{dim}")
            }
        };
        let addr_expr = |a: &Addr, coord: &dyn Fn(usize) -> String| -> String {
            let mut e = format!("{}", a.offset);
            for (dim, &cf) in a.coefs.iter().enumerate() {
                if cf != 0 {
                    e.push_str(&format!(" + {cf} * {}", coord(dim)));
                }
            }
            e
        };
        // an arg of the reduction as a CUDA expression at the current coords
        let arg_value = |a: &Arg| -> String {
            match &self.nodes[a.src.n()].op {
                Op::Iota(dd) => {
                    let fm = &a.view.forms[*dd];
                    let mut e = format!("{}", fm.offset);
                    for (dim, &cf) in fm.coefs.iter().enumerate() {
                        if cf != 0 {
                            e.push_str(&format!(" + {cf} * {}", coord(dim)));
                        }
                    }
                    format!("(float)({e})")
                }
                _ => ld(
                    storage(a.src.n()),
                    &format!(
                        "(unsigned)({})",
                        addr_expr(&self.flat_addr(a, rank), &coord)
                    ),
                ),
            }
        };
        // score-space coordinate names: its point dims map to reduction
        // dims; its reduce dim is the head-dim index `dd`
        let score_coord = |dim: usize| -> String {
            if dim == f.score_to_red.len() {
                "(int)dd".into()
            } else {
                coord(f.score_to_red[dim])
            }
        };
        // a read that is contiguous along `inner` and 16-byte aligned at
        // every vector start (the other coefficients and the offset are
        // multiples of the vector width), so it stages with 16-byte
        // `cp.async`
        let wv: usize = if half.is_some() { 8 } else { 4 };
        let vec4 = |a: &Addr, inner: usize, extent: usize| {
            a.coefs[inner] == 1
                && a.offset % wv as i64 == 0
                && extent.is_multiple_of(wv)
                && a.coefs
                    .iter()
                    .enumerate()
                    .all(|(dim, &cf)| dim == inner || cf % wv as i64 == 0)
        };
        // the row's query and folded head, where per-row code needs them
        let row_coords = |s: &mut String, indent: &str, r: &str| match fold {
            Some(fd) => writeln!(
                s,
                "{indent}unsigned p{fd} = ({r}) / {BI}u; unsigned ii = i0 + ({r}) % {BI}u;"
            )
            .unwrap(),
            None => writeln!(s, "{indent}unsigned ii = i0 + ({r});").unwrap(),
        };
        let emit_body = |s: &mut String,
                         indent: &str,
                         body: &tang_expr::ExprGraph,
                         outs: &[tang_expr::ExprId],
                         prefix: &str,
                         vals: &[String]|
         -> Vec<String> {
            let mut b = String::new();
            let o = emit_scalar(&mut b, body, outs, prefix, &|v| vals[v as usize].clone());
            for line in b.lines() {
                writeln!(
                    s,
                    "{indent}{}",
                    float_literals(&line.replacen("let ", "float ", 1))
                )
                .unwrap();
            }
            o
        };

        let mut s = String::new();
        let kind = match half {
            _ if fp8 => "FP8",
            None => "TF32",
            Some(super::Format::BF16) => "BF16",
            Some(_) => "F16",
        };
        writeln!(
            s,
            "// Auto-generated by tang-compile: flash attention on {} ({kind})",
            if hop.is_some() { "wgmma" } else { "mma" }
        )
        .unwrap();
        writeln!(
            s,
            "// {R} rows ({nf} x {BI}), keys {bj} x {kw} groups, {} stages",
            sh.stages
        )
        .unwrap();
        writeln!(s, "#define TANG_TILES {n_tiles}").unwrap();
        writeln!(s, "#define TANG_THREADS {T}").unwrap();
        writeln!(s, "#define TANG_SMEM {smem_bytes}").unwrap();
        writeln!(s, "#define select(f, t, c) ((c) ? (t) : (f))").unwrap();
        s.push_str(MMA_PRELUDE);
        if typed {
            s.push_str(HALF_PRELUDE);
        }
        if let Some(pl) = &fp8_plan {
            s.push_str(SM90_PRELUDE);
            s.push_str(&fp8::wgmma_fp8_named(
                "tang_wgs",
                fp8::ptx_ty(pl.fq),
                fp8::ptx_ty(pl.fk),
                bj,
            ));
            s.push_str(&fp8::wgmma_fp8_named(
                "tang_wgo",
                "e4m3",
                fp8::ptx_ty(pl.fv),
                d,
            ));
            s.push_str(FP8_FLASH_PRELUDE);
        }
        if let Some(hf) = half {
            s.push_str(&mma16_prelude(hf));
            if hop.is_some() {
                let ty = if hf == super::Format::BF16 {
                    "bf16"
                } else {
                    "f16"
                };
                s.push_str(SM90_PRELUDE);
                s.push_str(&sm90_wgmma(ty, bj, false));
                s.push_str(&sm90_wgmma(ty, d, true));
            }
        }
        let mut params: Vec<String> = reads
            .iter()
            .map(|&j| format!("const {}* __restrict__ b{j}", elem(fmt(j))))
            .collect();
        params.push(format!("{}* __restrict__ out", elem(fmt(root))));
        params.push("unsigned count".into());
        writeln!(
            s,
            "extern \"C\" __global__ void __launch_bounds__({T}{}) main0({}) {{",
            if hop.is_some() { ", 1" } else { "" },
            params.join(", ")
        )
        .unwrap();
        writeln!(s, "    (void)count;").unwrap();
        writeln!(s, "    extern __shared__ __align__(16) float smem[];").unwrap();
        // the longest (least masked) row blocks first
        writeln!(s, "    unsigned tile = {}u - blockIdx.x;", n_tiles - 1).unwrap();
        writeln!(s, "    if (blockIdx.x >= {n_tiles}u) return;").unwrap();
        writeln!(s, "    unsigned tile_i = tile % {tiles_i}u;").unwrap();
        writeln!(s, "    unsigned lead = tile / {tiles_i}u;").unwrap();
        for &dim in f.others.iter().rev().filter(|&&dim| Some(dim) != fold) {
            writeln!(s, "    unsigned p{dim} = lead % {}u;", f.pts[dim]).unwrap();
            writeln!(s, "    lead = lead / {}u;", f.pts[dim]).unwrap();
        }
        writeln!(s, "    unsigned i0 = tile_i * {BI}u;").unwrap();
        writeln!(s, "    unsigned tid = threadIdx.x;").unwrap();
        writeln!(s, "    unsigned warp = tid / 32u;").unwrap();
        writeln!(s, "    unsigned lane = tid % 32u;").unwrap();
        writeln!(s, "    unsigned gq = lane / 4u, tq = lane % 4u;").unwrap();
        // warp group (which keys of a chunk) and the warp's first row
        if hop.is_none() {
            writeln!(s, "    unsigned wg = warp / {row_warps}u;").unwrap();
            writeln!(s, "    unsigned wrow = (warp % {row_warps}u) * 16u;").unwrap();
            writeln!(s, "    unsigned kbase = wg * {bj}u;").unwrap();
        }

        // the key-chunk range: chunks the affine mask covers entirely
        // (block-uniform) are left out at either end
        writeln!(s, "    unsigned c_lo = 0u, c_hi = {n_chunks}u;").unwrap();
        if let Some(fm) = self.score_mask(rc, f.score_arg) {
            let mut e = format!("{}", fm.offset);
            for (dim, &cf) in fm.coefs.iter().enumerate() {
                if cf == 0 {
                    continue;
                }
                let (lo, hi) = if dim == jdim {
                    (
                        "(int)j0".to_string(),
                        format!("(int)min(j0 + {bjs}u, {}u) - 1", f.n_keys),
                    )
                } else if dim == f.dy {
                    (
                        "(int)i0".to_string(),
                        format!("(int)min(i0 + {BI}u, {}u) - 1", f.m),
                    )
                } else if dim == f.dx {
                    ("0".to_string(), format!("{}", d - 1))
                } else if Some(dim) == fold {
                    ("0".to_string(), format!("{}", nf - 1))
                } else {
                    (format!("(int)p{dim}"), format!("(int)p{dim}"))
                };
                let at = if cf > 0 { lo } else { hi };
                e.push_str(&format!(" + {cf} * ({at})"));
            }
            writeln!(s, "    c_lo = {n_chunks}u; c_hi = 0u;").unwrap();
            writeln!(s, "    for (unsigned c = 0u; c < {n_chunks}u; c++) {{").unwrap();
            writeln!(s, "        unsigned j0 = c * {bjs}u;").unwrap();
            writeln!(
                s,
                "        if (!({e} > 0)) {{ c_lo = min(c_lo, c); c_hi = c + 1u; }}"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }

        let nsf = bj / 8;
        let nof = d / 8;
        if let (Some(h), Some(pl)) = (hop, &fp8_plan) {
            self.fp8_flash_body(
                &mut s,
                &f,
                rc,
                pl,
                h,
                Fp8FlashCtx {
                    bj,
                    d,
                    stages: sh.stages,
                    threads: T,
                    nsf,
                    nof,
                },
                &row_coords,
                &arg_value,
                &emit_body,
                &coord,
                &score_coord,
                &addr_expr,
            );
        } else if let Some(h) = hop {
            // Hopper: one producer warpgroup stages Q, K and V with 16-byte
            // `cp.async` into 128-byte-swizzled tiles and signals a ring of
            // mbarriers; `h.consumers` warpgroups of 64 rows each run
            // S = Q·Kᵀ and O += P·V as `wgmma` (Q, K, V from shared-memory
            // descriptors, P from the score registers).
            let (qb, qa, kb, ka) = &f.terms[0];
            let (vb, va) = plain_v.clone()?;
            let cw = h.consumers;
            let bm = 64 * cw;
            let q_bytes = bm * kd * 2;
            let k_bytes = bj * kd * 2;
            let stage_bytes = k_bytes + bj * d * 2;
            let bar_off = q_bytes + sh.stages * stage_bytes;
            // one swizzled 16-byte chunk of a `[cols/64][rows][64]` tile
            let swz = |rows: usize, r: &str, c8: &str| {
                format!(
                    "(({c8}) / 8u) * {}u + ({r}) * 128u + (((({c8}) % 8u) ^ (({r}) % 8u)) * 16u)",
                    rows * 128
                )
            };
            writeln!(
                s,
                "    unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem) + 1023u) & ~1023u;"
            )
            .unwrap();
            writeln!(
                s,
                "    unsigned bar_f = sbase + {bar_off}u, bar_e = bar_f + {}u;",
                8 * sh.stages
            )
            .unwrap();
            writeln!(s, "    if (tid == 0u) {{").unwrap();
            writeln!(s, "        for (unsigned q = 0u; q < {}u; q++) {{ tang_bar_init(bar_f + 8u * q, 128u); tang_bar_init(bar_e + 8u * q, {}u); }}", sh.stages, 128 * cw).unwrap();
            writeln!(
                s,
                "        asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            // Q, by every thread, before the roles split
            writeln!(
                s,
                "    for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                bm * kd / 8
            )
            .unwrap();
            writeln!(
                s,
                "        unsigned r = idx / {}u, c8 = idx % {}u, dd = c8 * 8u, ii = i0 + r;",
                kd / 8,
                kd / 8
            )
            .unwrap();
            writeln!(s, "        bool ok = ii < {}u;", f.m).unwrap();
            writeln!(
                s,
                "        tang_cp16s(sbase + {}, b{qb} + (ok ? (unsigned)({}) : 0u), ok);",
                swz(bm, "r", "c8"),
                addr_expr(qa, &score_coord)
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(
                s,
                "    tang_cp_commit(); tang_cp_wait<0>(); tang_fence_async();"
            )
            .unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            // producer
            writeln!(s, "    if (warp < 4u) {{").unwrap();
            writeln!(s, "        tang_setmaxnreg_dec<{}>();", h.producer_regs).unwrap();
            writeln!(s, "        unsigned it = 0u;").unwrap();
            writeln!(s, "        for (unsigned c = c_lo; c < c_hi; c++, it++) {{").unwrap();
            writeln!(s, "            unsigned q = it % {}u;", sh.stages).unwrap();
            writeln!(
                s,
                "            tang_bar_wait(bar_e + 8u * q, ((it / {}u) & 1u) ^ 1u);",
                sh.stages
            )
            .unwrap();
            writeln!(
                s,
                "            unsigned st = sbase + {q_bytes}u + q * {stage_bytes}u;"
            )
            .unwrap();
            writeln!(
                s,
                "            for (unsigned idx = tid; idx < {}u; idx += 128u) {{",
                bj * kd / 8
            )
            .unwrap();
            writeln!(s, "                unsigned r = idx / {}u, c8 = idx % {}u, dd = c8 * 8u, jj = c * {bj}u + r;", kd / 8, kd / 8).unwrap();
            writeln!(s, "                bool ok = jj < {}u;", f.n_keys).unwrap();
            writeln!(
                s,
                "                tang_cp16s(st + {}, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                swz(bj, "r", "c8"),
                addr_expr(ka, &score_coord)
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(
                s,
                "            for (unsigned idx = tid; idx < {}u; idx += 128u) {{",
                bj * d / 8
            )
            .unwrap();
            writeln!(s, "                unsigned r = idx / {}u, c8 = idx % {}u, nn = c8 * 8u, jj = c * {bj}u + r;", d / 8, d / 8).unwrap();
            writeln!(s, "                bool ok = jj < {}u;", f.n_keys).unwrap();
            writeln!(
                s,
                "                tang_cp16s(st + {k_bytes}u + {}, b{vb} + (ok ? (unsigned)({}) : 0u), ok);",
                swz(bj, "r", "c8"),
                addr_expr(&va, &coord)
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "            tang_cp_commit();").unwrap();
            writeln!(s, "            if (c > c_lo) {{ tang_cp_wait<1>(); tang_fence_async(); tang_bar_arrive(bar_f + 8u * ((it - 1u) % {}u)); }}", sh.stages).unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        if (c_hi > c_lo) {{ tang_cp_wait<0>(); tang_fence_async(); tang_bar_arrive(bar_f + 8u * ((it - 1u) % {}u)); }}", sh.stages).unwrap();
            writeln!(s, "        return;").unwrap();
            writeln!(s, "    }}").unwrap();
            // consumers
            writeln!(s, "    tang_setmaxnreg_inc<{}>();", h.consumer_regs).unwrap();
            writeln!(s, "    unsigned cwi = warp / 4u - 1u;").unwrap();
            writeln!(s, "    unsigned wrow = cwi * 64u + (warp % 4u) * 16u;").unwrap();
            writeln!(s, "    unsigned qaddr = sbase + cwi * {}u;", 64 * 128).unwrap();
            writeln!(s, "    float o[{nof}][4];").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(
                s,
                "    for (int j = 0; j < {nof}; j++) for (int r = 0; r < 4; r++) o[j][r] = 0.0f;"
            )
            .unwrap();
            // the row max in log2 units (exp2 below)
            writeln!(s, "    float m_row[2] = {{-1.0e30f, -1.0e30f}};").unwrap();
            writeln!(s, "    float l_row[2] = {{0.0f, 0.0f}};").unwrap();
            writeln!(s, "    unsigned it = 0u;").unwrap();
            writeln!(s, "    for (unsigned c = c_lo; c < c_hi; c++, it++) {{").unwrap();
            writeln!(s, "        unsigned q = it % {}u;", sh.stages).unwrap();
            writeln!(
                s,
                "        tang_bar_wait(bar_f + 8u * q, (it / {}u) & 1u);",
                sh.stages
            )
            .unwrap();
            writeln!(s, "        unsigned kaddr = sbase + {q_bytes}u + q * {stage_bytes}u, vaddr = kaddr + {k_bytes}u;").unwrap();
            writeln!(s, "        float sc[{nsf}][4];").unwrap();
            writeln!(s, "        tang_wg_fence();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int kk = 0; kk < {}; kk++) {{", kd / 16).unwrap();
            writeln!(s, "            unsigned long long da = tang_desc(qaddr + (kk / 4) * {}u + (kk % 4) * 32u, 16u, 1024u);", bm * 128).unwrap();
            writeln!(s, "            unsigned long long db = tang_desc(kaddr + (kk / 4) * {}u + (kk % 4) * 32u, 16u, 1024u);", bj * 128).unwrap();
            writeln!(s, "            tang_wg_ss{bj}(&sc[0][0], da, db, kk > 0);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_wg_commit(); tang_wg_wait<0>();").unwrap();
            writeln!(s, "        unsigned j0 = c * {bj}u;").unwrap();
            // mask and online softmax per row half, as the mma path
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int h = 0; h < 2; h++) {{").unwrap();
            row_coords(&mut s, "            ", "wrow + gq + 8u * h");
            writeln!(s, "            float mloc = -1.0e30f;").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int e = 0; e < 2; e++) {{").unwrap();
            writeln!(
                s,
                "                unsigned jj = j0 + 8u * j + 2u * tq + e;"
            )
            .unwrap();
            writeln!(
                s,
                "                float sval = sc[j][2 * h + e] * {}f;",
                f32_lit(f.scale)
            )
            .unwrap();
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
                    &mut s,
                    "                ",
                    &rc.body,
                    &[rc.body_outs[0]],
                    "s0_",
                    &vals,
                );
                writeln!(
                    s,
                    "                float v = (jj < {}u) ? ({}) * 1.4426950408889634f : -1.0e30f;",
                    f.n_keys, o[0]
                )
                .unwrap();
            }
            writeln!(s, "                sc[j][2 * h + e] = v;").unwrap();
            writeln!(s, "                mloc = fmaxf(mloc, v);").unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(
                s,
                "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 1));"
            )
            .unwrap();
            writeln!(
                s,
                "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 2));"
            )
            .unwrap();
            writeln!(s, "            float m_new = fmaxf(m_row[h], mloc);").unwrap();
            writeln!(s, "            float alpha = exp2f(m_row[h] - m_new);").unwrap();
            writeln!(s, "            float lsum = 0.0f;").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int e = 0; e < 2; e++) {{ float p = exp2f(sc[j][2 * h + e] - m_new); sc[j][2 * h + e] = p; lsum += p; }}").unwrap();
            writeln!(
                s,
                "            l_row[h] = l_row[h] * alpha + lsum; m_row[h] = m_new;"
            )
            .unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int j = 0; j < {nof}; j++) {{ o[j][2 * h] *= alpha; o[j][2 * h + 1] *= alpha; }}").unwrap();
            writeln!(s, "        }}").unwrap();
            // P as wgmma A fragments: keys 16kb.. are score groups 2kb, 2kb + 1
            writeln!(s, "        unsigned pf[{}][4];", bj / 16).unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int kb = 0; kb < {}; kb++) {{", bj / 16).unwrap();
            writeln!(s, "            pf[kb][0] = tang_pack2(sc[2 * kb][0], sc[2 * kb][1]); pf[kb][1] = tang_pack2(sc[2 * kb][2], sc[2 * kb][3]);").unwrap();
            writeln!(s, "            pf[kb][2] = tang_pack2(sc[2 * kb + 1][0], sc[2 * kb + 1][1]); pf[kb][3] = tang_pack2(sc[2 * kb + 1][2], sc[2 * kb + 1][3]);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_wg_fence();").unwrap();
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int kb = 0; kb < {}; kb++) {{", bj / 16).unwrap();
            // V is MN-major (head dim contiguous): 64-wide column blocks
            // `bj · 128` bytes apart, 8 keys per 1024 bytes
            writeln!(
                s,
                "            unsigned long long dv = tang_desc(vaddr + kb * 2048u, {}u, 1024u);",
                bj * 128
            )
            .unwrap();
            writeln!(s, "            tang_wg_rs{d}(&o[0][0], pf[kb], dv);").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        tang_wg_commit(); tang_wg_wait<0>();").unwrap();
            writeln!(s, "        tang_bar_arrive(bar_e + 8u * q);").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(
                s,
                "    m_row[0] *= 0.6931471805599453f; m_row[1] *= 0.6931471805599453f;"
            )
            .unwrap();
        } else {
            // Q fragments, staged once through shared memory and kept as TF32
            // (the head-dim pair 2tq, 2tq + 1 in mma k slots tq, tq + 4)
            if has_k {
                writeln!(s, "    unsigned qf[{n_terms}][{nq}][4];").unwrap();
                for (t, (qb, qa, _, _)) in f.terms.iter().enumerate() {
                    writeln!(
                        s,
                        "    for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                        R * kd
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "        unsigned r = idx / {kd}u; unsigned dd = idx % {kd}u;"
                    )
                    .unwrap();
                    row_coords(&mut s, "        ", "r");
                    if half.is_some() {
                        writeln!(
                        s,
                        "        ((unsigned short*)smem)[{}u + r * {ldk}u + dd] = ii < {}u ? b{qb}[(unsigned)({})] : (unsigned short)0;",
                        t * R * ldk,
                        f.m,
                        addr_expr(qa, &score_coord)
                    )
                    .unwrap();
                    } else {
                        writeln!(
                        s,
                        "        smem[{}u + r * {ldk}u + dd] = ii < {}u ? b{qb}[(unsigned)({})] : 0.0f;",
                        t * R * ldk,
                        f.m,
                        addr_expr(qa, &score_coord)
                    )
                    .unwrap();
                    }
                    writeln!(s, "    }}").unwrap();
                }
                writeln!(s, "    __syncthreads();").unwrap();
                writeln!(s, "    #pragma unroll").unwrap();
                writeln!(s, "    for (int t = 0; t < {n_terms}; t++)").unwrap();
                writeln!(s, "    #pragma unroll").unwrap();
                if half.is_some() {
                    // m16n8k16 A: (row gq | gq + 8) × (dd 2tq.. | 2tq + 8..)
                    writeln!(s, "    for (int q = 0; q < {}; q++) {{", kd / 16).unwrap();
                    writeln!(
                    s,
                    "        const unsigned short* p = (const unsigned short*)smem + t * {}u + (wrow + gq) * {ldk}u + q * 16u + 2u * tq;",
                    R * ldk
                )
                .unwrap();
                    writeln!(
                    s,
                    "        qf[t][q][0] = *(const unsigned*)p; qf[t][q][1] = *(const unsigned*)(p + {}u);",
                    8 * ldk
                )
                .unwrap();
                    writeln!(
                    s,
                    "        qf[t][q][2] = *(const unsigned*)(p + 8u); qf[t][q][3] = *(const unsigned*)(p + {}u);",
                    8 * ldk + 8
                )
                .unwrap();
                    writeln!(s, "    }}").unwrap();
                    writeln!(s, "    __syncthreads();").unwrap();
                } else {
                    writeln!(s, "    for (int q = 0; q < {nq}; q++) {{").unwrap();
                    writeln!(
                s,
                "        const float* p = smem + t * {}u + (wrow + gq) * {ldk}u + q * 8u + 2u * tq;",
                R * ldk
            )
            .unwrap();
                    writeln!(
                        s,
                        "        float2 lo = *(const float2*)p, hi = *(const float2*)(p + {}u);",
                        8 * ldk
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "        qf[t][q][0] = tang_tf32(lo.x); qf[t][q][1] = tang_tf32(hi.x);"
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "        qf[t][q][2] = tang_tf32(lo.y); qf[t][q][3] = tang_tf32(hi.y);"
                    )
                    .unwrap();
                    writeln!(s, "    }}").unwrap();
                    writeln!(s, "    __syncthreads();").unwrap();
                }
            }

            // staging of one key chunk (K terms, then V) into buffer `buf`
            writeln!(s, "    auto load = [&](unsigned buf, unsigned j0) {{").unwrap();
            if half.is_some() {
                writeln!(
                    s,
                    "        unsigned short* st = (unsigned short*)smem + buf * {stage_floats}u;"
                )
                .unwrap();
            } else {
                writeln!(s, "        float* st = smem + buf * {stage_floats}u;").unwrap();
            }
            let cp16 = if half.is_some() {
                "tang_cp16h"
            } else {
                "tang_cp16"
            };
            if has_k {
                for (t, (_, _, kb, ka)) in f.terms.iter().enumerate() {
                    let off = t * bjs * ldk;
                    if vec4(ka, f.score_to_red.len(), kd) {
                        writeln!(
                            s,
                            "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                            bjs * kd / wv
                        )
                        .unwrap();
                        writeln!(
                        s,
                        "            unsigned r = idx / {}u; unsigned dd = (idx % {}u) * {wv}u; unsigned jj = j0 + r;",
                        kd / wv,
                        kd / wv
                    )
                    .unwrap();
                        writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                        writeln!(
                        s,
                        "            {cp16}(st + {off}u + r * {ldk}u + dd, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(ka, &score_coord)
                    )
                    .unwrap();
                    } else {
                        writeln!(
                            s,
                            "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                            bjs * kd
                        )
                        .unwrap();
                        writeln!(
                        s,
                        "            unsigned r = idx / {kd}u; unsigned dd = idx % {kd}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                        writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                        if half.is_some() {
                            writeln!(
                            s,
                            "            st[{off}u + r * {ldk}u + dd] = ok ? b{kb}[(unsigned)({})] : (unsigned short)0;",
                            addr_expr(ka, &score_coord)
                        )
                        .unwrap();
                        } else {
                            writeln!(
                            s,
                            "            tang_cp4(st + {off}u + r * {ldk}u + dd, b{kb} + (ok ? (unsigned)({}) : 0u), ok);",
                            addr_expr(ka, &score_coord)
                        )
                        .unwrap();
                        }
                    }
                    writeln!(s, "        }}").unwrap();
                }
            }
            {
                // the value: a plain read stages asynchronously; any other
                // expression is evaluated while staging
                match plain_v.clone() {
                    Some((vb, va)) if vec4(&va, f.dx, d) => {
                        writeln!(
                            s,
                            "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                            bjs * d / wv
                        )
                        .unwrap();
                        writeln!(
                        s,
                        "            unsigned r = idx / {}u; unsigned nn = (idx % {}u) * {wv}u; unsigned jj = j0 + r;",
                        d / wv,
                        d / wv
                    )
                    .unwrap();
                        writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                        writeln!(
                        s,
                        "            {cp16}(st + {k_floats}u + r * {ldv}u + nn, b{vb} + (ok ? (unsigned)({}) : 0u), ok);",
                        addr_expr(&va, &coord)
                    )
                    .unwrap();
                    }
                    Some((vb, va)) => {
                        writeln!(
                            s,
                            "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                            bjs * d
                        )
                        .unwrap();
                        writeln!(
                        s,
                        "            unsigned r = idx / {d}u; unsigned nn = idx % {d}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                        writeln!(s, "            bool ok = jj < {}u;", f.n_keys).unwrap();
                        if half.is_some() {
                            writeln!(
                            s,
                            "            st[{k_floats}u + r * {ldv}u + nn] = ok ? b{vb}[(unsigned)({})] : (unsigned short)0;",
                            addr_expr(&va, &coord)
                        )
                        .unwrap();
                        } else {
                            writeln!(
                            s,
                            "            tang_cp4(st + {k_floats}u + r * {ldv}u + nn, b{vb} + (ok ? (unsigned)({}) : 0u), ok);",
                            addr_expr(&va, &coord)
                        )
                        .unwrap();
                        }
                    }
                    None => {
                        writeln!(
                            s,
                            "        for (unsigned idx = tid; idx < {}u; idx += {T}u) {{",
                            bjs * d
                        )
                        .unwrap();
                        writeln!(
                        s,
                        "            unsigned r = idx / {d}u; unsigned nn = idx % {d}u; unsigned jj = j0 + r;"
                    )
                    .unwrap();
                        writeln!(s, "            float v = 0.0f;").unwrap();
                        writeln!(s, "            if (jj < {}u) {{", f.n_keys).unwrap();
                        let vals: Vec<String> = rc.args.iter().map(arg_value).collect();
                        let o =
                            emit_body(&mut s, "                ", &rc.body, &[e_v], "v0_", &vals);
                        writeln!(s, "                v = {};", o[0]).unwrap();
                        writeln!(s, "            }}").unwrap();
                        let v = match half {
                            Some(super::Format::BF16) => "tang_f2b(v)",
                            Some(_) => "tang_f2h(v)",
                            None => "v",
                        };
                        writeln!(s, "            st[{k_floats}u + r * {ldv}u + nn] = {v};")
                            .unwrap();
                    }
                }
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "        tang_cp_commit();").unwrap();
            writeln!(s, "    }};").unwrap();

            // per thread: rows wrow + gq and wrow + gq + 8 (h = 0, 1)
            let nsf = bj / 8;
            let nof = d / 8;
            writeln!(s, "    float o[{nof}][4];").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(
                s,
                "    for (int j = 0; j < {nof}; j++) for (int r = 0; r < 4; r++) o[j][r] = 0.0f;"
            )
            .unwrap();
            writeln!(s, "    float m_row[2] = {{-3.0e38f, -3.0e38f}};").unwrap();
            // per-thread partial row sums (the quad's are summed at the end)
            writeln!(s, "    float l_row[2] = {{0.0f, 0.0f}};").unwrap();
            let st = sh.stages;
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(
            s,
            "    for (unsigned c = 0u; c < {}u; c++) {{ if (c_lo + c < c_hi) load((c_lo + c) % {st}u, (c_lo + c) * {bjs}u); else tang_cp_commit(); }}",
            st - 1
        )
        .unwrap();
            writeln!(s, "    for (unsigned c = c_lo; c < c_hi; c++) {{").unwrap();
            writeln!(s, "        tang_cp_wait<{}>();", st - 2).unwrap();
            writeln!(s, "        __syncthreads();").unwrap();
            writeln!(s, "        {{").unwrap();
            writeln!(s, "            unsigned nc = c + {}u;", st - 1).unwrap();
            writeln!(
                s,
                "            if (nc < c_hi) load(nc % {st}u, nc * {bjs}u); else tang_cp_commit();"
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            if half.is_some() {
                writeln!(
                s,
                "        const unsigned short* st = (const unsigned short*)smem + (c % {st}u) * {stage_floats}u;"
            )
            .unwrap();
            } else {
                writeln!(
                    s,
                    "        const float* st = smem + (c % {st}u) * {stage_floats}u;"
                )
                .unwrap();
            }
            writeln!(s, "        unsigned j0 = c * {bjs}u + kbase;").unwrap();
            writeln!(s, "        float sc[{nsf}][4];").unwrap();
            if has_k {
                // S = Σ_t Q_t · K_tᵀ: key fragment j holds keys 8j .. 8j + 7
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int j = 0; j < {nsf}; j++) for (int r = 0; r < 4; r++) sc[j][r] = 0.0f;").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int t = 0; t < {n_terms}; t++)").unwrap();
                writeln!(s, "        #pragma unroll").unwrap();
                if half.is_some() {
                    // m16n8k16 B: key 8j + gq, dd (2tq.. | 2tq + 8..)
                    writeln!(s, "        for (int q = 0; q < {}; q++)", kd / 16).unwrap();
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int j = 0; j < {nsf}; j++) {{").unwrap();
                    writeln!(
                    s,
                    "            const unsigned short* kp = st + t * {}u + (kbase + 8u * j + gq) * {ldk}u + 16u * q + 2u * tq;",
                    bjs * ldk
                )
                .unwrap();
                    writeln!(
                    s,
                    "            unsigned kf[2] = {{*(const unsigned*)kp, *(const unsigned*)(kp + 8u)}};"
                )
                .unwrap();
                    writeln!(s, "            tang_mma16(sc[j], qf[t][q], kf);").unwrap();
                    writeln!(s, "        }}").unwrap();
                } else {
                    writeln!(s, "        for (int q = 0; q < {nq}; q++)").unwrap();
                    writeln!(s, "        #pragma unroll").unwrap();
                    writeln!(s, "        for (int j = 0; j < {nsf}; j++) {{").unwrap();
                    writeln!(
                s,
                "            float2 kv = *(const float2*)(st + t * {}u + (kbase + 8u * j + gq) * {ldk}u + 8u * q + 2u * tq);",
                bjs * ldk
            )
            .unwrap();
                    writeln!(
                        s,
                        "            unsigned kf[2] = {{tang_tf32(kv.x), tang_tf32(kv.y)}};"
                    )
                    .unwrap();
                    writeln!(s, "            tang_mma(sc[j], qf[t][q], kf);").unwrap();
                    writeln!(s, "        }}").unwrap();
                }
            }
            // the comp's score expression (mask) on each element, then the
            // online softmax per row half
            writeln!(s, "        #pragma unroll").unwrap();
            writeln!(s, "        for (int h = 0; h < 2; h++) {{").unwrap();
            row_coords(&mut s, "            ", "wrow + gq + 8u * h");
            writeln!(s, "            float mloc = -3.0e38f;").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int e = 0; e < 2; e++) {{").unwrap();
            writeln!(
                s,
                "                unsigned jj = j0 + 8u * j + 2u * tq + e;"
            )
            .unwrap();
            if let Some((sb, sa)) = &f.score_buf {
                writeln!(
                    s,
                    "                float sval = (ii < {}u && jj < {}u) ? {} : 0.0f;",
                    f.m,
                    f.n_keys,
                    ld(*sb, &format!("(unsigned)({})", addr_expr(sa, &coord)))
                )
                .unwrap();
            } else {
                writeln!(
                    s,
                    "                float sval = sc[j][2 * h + e] * {}f;",
                    f32_lit(f.scale)
                )
                .unwrap();
            }
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
                    &mut s,
                    "                ",
                    &rc.body,
                    &[rc.body_outs[0]],
                    "s0_",
                    &vals,
                );
                writeln!(
                    s,
                    "                float v = (jj < {}u) ? ({}) : -3.0e38f;",
                    f.n_keys, o[0]
                )
                .unwrap();
            }
            writeln!(s, "                sc[j][2 * h + e] = v;").unwrap();
            writeln!(s, "                mloc = fmaxf(mloc, v);").unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(
                s,
                "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 1));"
            )
            .unwrap();
            writeln!(
                s,
                "            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, 2));"
            )
            .unwrap();
            writeln!(s, "            float m_new = fmaxf(m_row[h], mloc);").unwrap();
            writeln!(s, "            float alpha = expf(m_row[h] - m_new);").unwrap();
            writeln!(s, "            float lsum = 0.0f;").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int j = 0; j < {nsf}; j++)").unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(s, "            for (int e = 0; e < 2; e++) {{").unwrap();
            writeln!(
            s,
            "                float p = expf(sc[j][2 * h + e] - m_new); sc[j][2 * h + e] = p; lsum += p;"
        )
        .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(
                s,
                "            l_row[h] = l_row[h] * alpha + lsum; m_row[h] = m_new;"
            )
            .unwrap();
            writeln!(s, "            #pragma unroll").unwrap();
            writeln!(
            s,
            "            for (int j = 0; j < {nof}; j++) {{ o[j][2 * h] *= alpha; o[j][2 * h + 1] *= alpha; }}"
        )
        .unwrap();
            writeln!(s, "        }}").unwrap();
            // O += P · V, P straight from the score registers
            if half.is_some() {
                // m16n8k16 A over keys 16jk..: score fragments 2jk, 2jk + 1;
                // B: V rows 16jk + 2tq, +1 (| +8, +9), column words split
                // into the fragment pair as in the tiles
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int jk = 0; jk < {}; jk++) {{", nsf / 2).unwrap();
                writeln!(s, "            unsigned pf[4] = {{tang_pack2(sc[2 * jk][0], sc[2 * jk][1]), tang_pack2(sc[2 * jk][2], sc[2 * jk][3]), tang_pack2(sc[2 * jk + 1][0], sc[2 * jk + 1][1]), tang_pack2(sc[2 * jk + 1][2], sc[2 * jk + 1][3])}};").unwrap();
                writeln!(s, "            #pragma unroll").unwrap();
                writeln!(s, "            for (int jp = 0; jp < {}; jp++) {{", d / 16).unwrap();
                writeln!(
                s,
                "                const unsigned short* vp = st + {k_floats}u + (kbase + 16u * jk + 2u * tq) * {ldv}u + 16u * jp + 2u * gq;"
            )
            .unwrap();
                writeln!(
                s,
                "                unsigned r0 = *(const unsigned*)vp, r1 = *(const unsigned*)(vp + {ldv}u), r8 = *(const unsigned*)(vp + {}u), r9 = *(const unsigned*)(vp + {}u);",
                8 * ldv,
                9 * ldv
            )
            .unwrap();
                writeln!(s, "                unsigned v0[2] = {{__byte_perm(r0, r1, 0x5410), __byte_perm(r8, r9, 0x5410)}}, v1[2] = {{__byte_perm(r0, r1, 0x7632), __byte_perm(r8, r9, 0x7632)}};").unwrap();
                writeln!(
                s,
                "                tang_mma16(o[2 * jp], pf, v0); tang_mma16(o[2 * jp + 1], pf, v1);"
            )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "        }}").unwrap();
            } else {
                writeln!(s, "        #pragma unroll").unwrap();
                writeln!(s, "        for (int j = 0; j < {nsf}; j++) {{").unwrap();
                writeln!(
            s,
            "            unsigned pf[4] = {{tang_tf32(sc[j][0]), tang_tf32(sc[j][2]), tang_tf32(sc[j][1]), tang_tf32(sc[j][3])}};"
        )
        .unwrap();
                writeln!(s, "            #pragma unroll").unwrap();
                writeln!(s, "            for (int jp = 0; jp < {}; jp++) {{", d / 16).unwrap();
                writeln!(
            s,
            "                const float* vp = st + {k_floats}u + (kbase + 8u * j + 2u * tq) * {ldv}u + 16u * jp + 2u * gq;"
        )
        .unwrap();
                writeln!(
            s,
            "                float2 r0 = *(const float2*)vp, r1 = *(const float2*)(vp + {ldv}u);"
        )
                .unwrap();
                writeln!(
            s,
            "                unsigned v0[2] = {{tang_tf32(r0.x), tang_tf32(r1.x)}}, v1[2] = {{tang_tf32(r0.y), tang_tf32(r1.y)}};"
        )
        .unwrap();
                writeln!(
                    s,
                    "                tang_mma(o[2 * jp], pf, v0); tang_mma(o[2 * jp + 1], pf, v1);"
                )
                .unwrap();
                writeln!(s, "            }}").unwrap();
                writeln!(s, "        }}").unwrap();
            }
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    tang_cp_wait<0>();").unwrap();
        }
        // the quad's partial row sums
        writeln!(s, "    #pragma unroll").unwrap();
        writeln!(s, "    for (int h = 0; h < 2; h++) {{").unwrap();
        writeln!(
            s,
            "        l_row[h] += __shfl_xor_sync(0xffffffffu, l_row[h], 1);"
        )
        .unwrap();
        writeln!(
            s,
            "        l_row[h] += __shfl_xor_sync(0xffffffffu, l_row[h], 2);"
        )
        .unwrap();
        writeln!(s, "    }}").unwrap();
        // element r of O fragment j: row gq + 8·(r/2), lane (the V column
        // pairing above) 16·(j/2) + 4·tq + 2·(r%2) + j%2
        let frag_loop = |s: &mut String| {
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int j = 0; j < {nof}; j++)").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int r = 0; r < 4; r++) {{").unwrap();
            writeln!(s, "        unsigned lr = wrow + gq + 8u * (r / 2);").unwrap();
            if hop.is_some() {
                writeln!(s, "        unsigned lc = 8u * j + 2u * tq + (r % 2);").unwrap();
            } else {
                writeln!(
                    s,
                    "        unsigned lc = 16u * (j / 2) + 4u * tq + 2u * (r % 2) + (j % 2);"
                )
                .unwrap();
            }
        };
        if kw > 1 {
            // groups 1.. hand their (m, l, O) to group 0 through shared memory
            let (mo, lo, oo) = (0, kw * R, 2 * kw * R);
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (gq < 8u && tq == 0u) {{").unwrap();
            writeln!(s, "        for (int h = 0; h < 2; h++) {{").unwrap();
            writeln!(s, "            unsigned lr = wrow + gq + 8u * h;").unwrap();
            writeln!(s, "            smem[{mo}u + wg * {R}u + lr] = m_row[h];").unwrap();
            writeln!(s, "            smem[{lo}u + wg * {R}u + lr] = l_row[h];").unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    if (wg > 0u) {{").unwrap();
            frag_loop(&mut s);
            writeln!(
                s,
                "        smem[{oo}u + ((wg - 1u) * {R}u + lr) * {ldo}u + lc] = o[j][r];"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    }}").unwrap();
            writeln!(s, "    __syncthreads();").unwrap();
            writeln!(s, "    if (wg > 0u) return;").unwrap();
            writeln!(s, "    float sc_g[{kw}][2];").unwrap();
            writeln!(s, "    #pragma unroll").unwrap();
            writeln!(s, "    for (int h = 0; h < 2; h++) {{").unwrap();
            writeln!(s, "        unsigned lr = wrow + gq + 8u * h;").unwrap();
            writeln!(s, "        float mx = m_row[h];").unwrap();
            writeln!(
                s,
                "        for (int g = 1; g < {kw}; g++) mx = fmaxf(mx, smem[{mo}u + g * {R}u + lr]);"
            )
            .unwrap();
            writeln!(s, "        float l = 0.0f;").unwrap();
            writeln!(s, "        for (int g = 0; g < {kw}; g++) {{").unwrap();
            writeln!(
                s,
                "            float mg = g == 0 ? m_row[h] : smem[{mo}u + g * {R}u + lr];"
            )
            .unwrap();
            writeln!(
                s,
                "            float lg = g == 0 ? l_row[h] : smem[{lo}u + g * {R}u + lr];"
            )
            .unwrap();
            writeln!(
                s,
                "            sc_g[g][h] = expf(mg - mx); l += lg * sc_g[g][h];"
            )
            .unwrap();
            writeln!(s, "        }}").unwrap();
            writeln!(s, "        m_row[h] = mx; l_row[h] = l;").unwrap();
            writeln!(s, "    }}").unwrap();
            frag_loop(&mut s);
            writeln!(s, "        float acc = o[j][r] * sc_g[0][r / 2];").unwrap();
            writeln!(
                s,
                "        for (int g = 1; g < {kw}; g++) acc += smem[{oo}u + ((g - 1) * {R}u + lr) * {ldo}u + lc] * sc_g[g][r / 2];"
            )
            .unwrap();
            writeln!(s, "        o[j][r] = acc;").unwrap();
            writeln!(s, "    }}").unwrap();
        }
        // the output per (row, lane): the state, or the normalizer
        frag_loop(&mut s);
        writeln!(s, "        unsigned nn = lc;").unwrap();
        row_coords(&mut s, "        ", "lr");
        writeln!(
            s,
            "        float m_v = m_row[r / 2], l_v = l_row[r / 2], acc_v = o[j][r];"
        )
        .unwrap();
        writeln!(s, "        if (ii < {}u) {{", f.m).unwrap();
        if !f.epilogue {
            // the root is the reduction: its (m, l, acc) state per point
            let mut flat = String::from("0u");
            for (dd, &np) in f.pts.iter().enumerate() {
                flat = format!("({flat}) * {np}u + (unsigned)({})", coord(dd));
            }
            writeln!(s, "            unsigned g = {flat};").unwrap();
            let st_v = |v: &str| match fmt(root) {
                super::Format::F32 => v.to_string(),
                super::Format::F16 => format!("tang_f2h({v})"),
                super::Format::BF16 => format!("tang_f2b({v})"),
                super::Format::Q4K
                | super::Format::Q5K
                | super::Format::Q6K
                | super::Format::Q8x32
                | super::Format::Q8x32F
                | super::Format::Fp8B32
                | super::Format::Mxfp4 => {
                    unreachable!("block formats lower only through the int8 MMA kernel")
                }
                _ => unreachable!("fp8 has its own tile"),
            };
            writeln!(s, "            out[g * {w_out}u + 0u] = {};", st_v("m_v")).unwrap();
            writeln!(s, "            out[g * {w_out}u + 1u] = {};", st_v("l_v")).unwrap();
            writeln!(s, "            out[g * {w_out}u + 2u] = {};", st_v("acc_v")).unwrap();
        } else {
            // the normalizer at (ii, nn), written in root's layout
            let ep = self.epilogue_read(c, f.red_node)?;
            let root_rank = c.points().len();
            let mut root_coord: Vec<Option<String>> = vec![None; root_rank];
            for (rdim, fm) in ep.iter().enumerate() {
                if let Some(dd) = (0..root_rank).find(|&dd| fm.coefs[dd] != 0) {
                    root_coord[dd] = Some(coord(rdim));
                }
            }
            let root_coord: Vec<String> = root_coord.into_iter().collect::<Option<_>>()?;
            let rcoord = |dim: usize| root_coord[dim].clone();
            let vals: Vec<String> = c
                .args
                .iter()
                .map(|a| {
                    if a.src.n() == f.red_node {
                        match self.epilogue_slot(a) {
                            0 => "m_v".to_string(),
                            1 => "l_v".to_string(),
                            _ => "acc_v".to_string(),
                        }
                    } else {
                        let addr = self.flat_addr(a, root_rank);
                        ld(
                            storage(a.src.n()),
                            &format!("(unsigned)({})", addr_expr(&addr, &rcoord)),
                        )
                    }
                })
                .collect();
            let outs = emit_body(&mut s, "            ", &c.body, &c.body_outs, "e0_", &vals);
            let pts_root = super::fixed(c.points());
            let mut flat = String::from("0u");
            for (dd, &np) in pts_root.iter().enumerate() {
                flat = format!("({flat}) * {np}u + (unsigned)({})", root_coord[dd]);
            }
            writeln!(s, "            unsigned g = {flat};").unwrap();
            for (slot, v) in outs.iter().enumerate() {
                let v = match fmt(root) {
                    super::Format::F32 => v.clone(),
                    super::Format::F16 => format!("tang_f2h({v})"),
                    super::Format::BF16 => format!("tang_f2b({v})"),
                    super::Format::Q4K
                    | super::Format::Q5K
                    | super::Format::Q6K
                    | super::Format::Q8x32
                    | super::Format::Q8x32F
                    | super::Format::Fp8B32
                    | super::Format::Mxfp4 => {
                        unreachable!("block formats lower only through the int8 MMA kernel")
                    }
                    _ => unreachable!("fp8 has its own tile"),
                };
                writeln!(s, "            out[g * {w_out}u + {slot}u] = {v};").unwrap();
            }
        }
        writeln!(s, "        }}").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        let _ = f.score_node;
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::{cuda_stmt, float_literals};

    /// A typed `let` (tile unroll offsets) keeps its type in C++.
    #[test]
    fn typed_let_is_typed() {
        assert_eq!(cuda_stmt("let u0: u32 = 1u;"), "const unsigned u0 = 1u;");
        assert_eq!(cuda_stmt("let x = a + b;"), "auto x = a + b;");
    }

    #[test]
    fn literals_stay_single_precision() {
        assert_eq!(float_literals("auto x = -1e30;"), "auto x = -1e30f;");
        assert_eq!(
            float_literals("let y = 1.5 * e4 + 2u;"),
            "let y = 1.5f * e4 + 2u;"
        );
        assert_eq!(float_literals("c0_14 = 1.442695;"), "c0_14 = 1.442695f;");
        assert_eq!(
            float_literals("v[3] = 3.4028235e38"),
            "v[3] = 3.4028235e38f"
        );
        assert_eq!(
            float_literals("stage_0[idx * 64u + 1u]"),
            "stage_0[idx * 64u + 1u]"
        );
    }
}
