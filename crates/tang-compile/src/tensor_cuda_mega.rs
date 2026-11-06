//! A whole decode program as one persistent CUDA kernel: the schedule
//! with no kernel boundaries.
//!
//! [`megakernel`] takes the kernels of a lowered program (CUDA sources in
//! the `to_cuda`/`to_cuda_decode` calling convention: `main0(b0.., out,
//! count)`), each with its launch shape and the buffer slot behind each
//! parameter, and emits one kernel that a single launch of
//! `grid × threads` runs to completion:
//!
//! - **Stages.** Every kernel becomes a `__device__` function in its own
//!   namespace (so helper names never collide), run over its original
//!   grid as *virtual blocks*: `blockIdx`/`gridDim`/`threadIdx`/`blockDim`
//!   are rewritten to the virtual ones, and a physical block of `threads`
//!   runs `threads / w` virtual blocks of a `w`-thread kernel side by side,
//!   each half (quarter, ...) synchronizing on its own named barrier
//!   (`bar.sync id, w`) where the kernel had `__syncthreads()`. Function
//!   local `__shared__` arrays are carved out of one dynamic pool, per
//!   side-by-side slot, so stages reuse the same bytes.
//! - **Phases.** Kernels are levelled ASAP over their buffer hazards
//!   (read after write, write after read, write after write, on slots):
//!   a phase is a set of kernels with no hazard between them, and phases
//!   are separated by a grid-wide barrier (an arrival counter in global
//!   memory, acquire/release at GPU scope; the grid is sized to be
//!   co-resident, so it cannot deadlock). A boundary costs one barrier
//!   instead of a launch, and independent kernels share a phase.
//! - **Coherence.** Data written by one block and read by another within
//!   the launch must not come through the non-coherent read-only path, so
//!   `__ldg` becomes a plain load and `__restrict__` is dropped.
//!
//! The buffers are passed as one table of device pointers, in the order
//! of [`Mega::table`] (slots by first use, so the same run of kernels over
//! other buffers — the next layer — has the same source and compiles
//! once), and the barrier state as two words (`sync`, zero before the
//! first launch; the kernel leaves them zero).
use std::fmt::Write;

/// One kernel of the program, in launch order.
pub struct MegaStage<'a> {
    pub source: &'a str,
    /// Threads per block of the standalone launch.
    pub threads: u32,
    /// Blocks of the standalone launch.
    pub blocks: u32,
    /// The kernel's `count` argument.
    pub count: u32,
    /// Buffer slot per pointer parameter: reads in order, then `out`.
    pub args: Vec<usize>,
}

/// What [`megakernel`] produced.
pub struct Mega {
    pub source: String,
    /// Threads per block of the persistent launch.
    pub threads: u32,
    /// Dynamic shared memory per block, bytes.
    pub smem: u32,
    /// Kernel indices per phase (grid barriers = phases − 1).
    pub phases: Vec<Vec<usize>>,
    /// Slot behind each entry of the launch's pointer table.
    pub table: Vec<usize>,
}

/// Knobs of [`megakernel`].
#[derive(Clone, Debug, Default)]
pub struct MegaOpts {
    /// Threads per block (a multiple of every stage's block size, at most
    /// 15 stage blocks per physical block). 0: 256.
    pub threads: u32,
    /// Stages of at most this many blocks are replicated: every physical
    /// block runs all of their virtual blocks, so their readers need no
    /// grid barrier. 0: none.
    pub dup: u32,
    /// `__launch_bounds__` minimum blocks per SM (caps registers). 0: none.
    pub min_blocks: u32,
    /// Bytes of upcoming phases' constant buffers (slots no stage writes:
    /// weights) to prefetch into L2 while a block waits at a barrier, so
    /// DRAM keeps streaming across the boundary. 0: off.
    pub prefetch: usize,
    /// Bytes per slot (for `prefetch`).
    pub slot_bytes: Vec<usize>,
    /// Block 0 records `%globaltimer` at the start of every phase (and at
    /// the end) as u64 at `sync + 2` (the sync buffer then needs
    /// `2 + 2 · (phases + 1)` words).
    pub trace: bool,
}

/// Stages small enough to replicate: at most `dup` blocks and not
/// reading their own output (so every block writing the same values is
/// harmless).
pub fn replicated(stages: &[MegaStage], dup: u32) -> Vec<bool> {
    stages
        .iter()
        .map(|st| {
            let (r, w) = split(&st.args);
            st.blocks <= dup && !r.contains(&w)
        })
        .collect()
}

/// The phase of each kernel: one past the latest earlier kernel it has a
/// slot hazard with (RAW, WAR, WAW) — or the same phase, after it, when
/// the only hazard is reading a replicated stage's output (every block
/// computed its own copy).
pub fn phases(stages: &[MegaStage], dup: &[bool]) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; stages.len()];
    for k in 0..stages.len() {
        let (kr, kw) = split(&stages[k].args);
        let mut lv = 0;
        for j in 0..k {
            let (jr, jw) = split(&stages[j].args);
            let raw = kr.contains(&jw);
            let other = jr.contains(&kw) || jw == kw;
            if other || (raw && !dup[j]) {
                lv = lv.max(level[j] + 1);
            } else if raw {
                lv = lv.max(level[j]);
            }
        }
        level[k] = lv;
    }
    let n = level.iter().max().map_or(0, |m| m + 1);
    let mut out = vec![Vec::new(); n];
    for (k, &l) in level.iter().enumerate() {
        out[l].push(k);
    }
    out
}

fn split(args: &[usize]) -> (&[usize], usize) {
    let (w, r) = args.split_last().expect("a kernel writes out");
    (r, *w)
}

const HEADER: &str = r#"// tang megakernel
template <class T> __device__ __forceinline__ T tang_ld(const T* p) { return *p; }
// barrier ids are immediates: with a register id ptxas reserves all 16
// hardware barriers per block, which caps blocks per SM below what the
// occupancy API reports, and a grid barrier over a grid that is not all
// resident deadlocks
template <unsigned ID, unsigned N> __device__ __forceinline__ void tang_bar() {
    asm volatile("bar.sync %0, %1;" :: "n"(ID), "n"(N) : "memory");
}
extern __shared__ __align__(16) char tang_dyn[];
__device__ __forceinline__ void tang_prefetch(unsigned long long base, unsigned long long bytes) {
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x * 128ull;
    for (unsigned long long o = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 128ull; o < bytes; o += step)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(base + o));
}
__device__ __forceinline__ void tang_grid_sync(unsigned* s, unsigned target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(s, 1u);
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(s) : "memory");
        } while (v < target);
        __threadfence();
    }
    __syncthreads();
}
"#;

/// Bytes of a scalar type spelled in a `__shared__` declaration.
fn type_bytes(t: &str) -> usize {
    match t.trim() {
        "float" | "int" | "unsigned" | "unsigned int" | "uint" => 4,
        "half" | "__half" | "unsigned short" | "short" => 2,
        "double" | "float2" | "uint2" | "int2" | "long long" | "unsigned long long" => 8,
        "float4" | "uint4" | "int4" => 16,
        t => panic!("megakernel: shared array of unknown type {t}"),
    }
}

/// One kernel's source as a namespaced device function `k{i}::run`; the
/// pool bytes its function-local shared arrays take.
/// `__ldg` stays on buffers no stage writes (weights: the read-only path
/// is fine for data constant across the launch); every other read
/// becomes a plain, coherent load. A read is attributed to the pointer
/// parameter `bN` it indexes; the packed-load helpers' bare `b` reads
/// weights (4-bit, bf16), except `tang_f16` (a KV cache).
fn coherent_loads(src: &str, konst: &[bool]) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_f16 = false;
    for line in src.split_inclusive('\n') {
        if line.starts_with("__device__") {
            in_f16 = line.contains("tang_f16(");
        }
        let mut rest = line;
        while let Some(at) = rest.find("__ldg(") {
            out.push_str(&rest[..at]);
            let after = &rest[at + "__ldg(".len()..];
            // skip casts and parens to the pointer token
            let mut t = after.trim_start();
            loop {
                if let Some(r) = t.strip_prefix('(') {
                    t = r.trim_start();
                    if t.starts_with("const ") {
                        let close = t.find(')').unwrap_or(0);
                        t = t[close + 1..].trim_start();
                    }
                } else {
                    break;
                }
            }
            let ident: String = t
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let keep = match ident
                .strip_prefix('b')
                .and_then(|n| n.parse::<usize>().ok())
            {
                Some(n) => konst.get(n).copied().unwrap_or(false),
                None => ident == "b" && !in_f16,
            };
            out.push_str(if keep { "__ldg(" } else { "tang_ld(" });
            rest = after;
        }
        out.push_str(rest);
    }
    out
}

fn stage_source(i: usize, st: &MegaStage, konst: &[bool]) -> (String, usize) {
    let coherent = coherent_loads(st.source, konst);
    let src = coherent.as_str();
    let at = src
        .find("main0(")
        .unwrap_or_else(|| panic!("megakernel: kernel {i} has no main0"));
    let line_start = src[..at].rfind('\n').map_or(0, |p| p + 1);
    let close = at + src[at..].find(')').expect("signature");
    let params = &src[at + "main0(".len()..close];
    let (head, body) = (&src[..line_start], &src[close + 1..]);
    let w = st.threads;
    let mut pool = 0usize;
    let mut out = String::new();
    writeln!(out, "namespace k{i} {{").unwrap();
    out.push_str(head);
    write!(
        out,
        "template <unsigned TANG_BAR> __device__ __noinline__ void run({params}, unsigned tang_bid, unsigned tang_tid, unsigned tang_bid_n, char* tang_sm)"
    )
    .unwrap();
    for line in body.lines() {
        let t = line.trim_start();
        let l = if let Some(decl) = t.strip_prefix("__shared__ ") {
            // `__shared__ T name[a][b];` inside the function: a slice of the pool
            let decl = decl.trim_end().trim_end_matches(';');
            let br = decl.find('[').expect("shared array");
            let (ty_name, dims) = decl.split_at(br);
            let sp = ty_name.trim_end().rfind(' ').expect("shared decl");
            let (ty, name) = (ty_name[..sp].trim(), ty_name[sp..].trim());
            let n: usize = dims
                .split(|c| c == '[' || c == ']')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().parse::<usize>().expect("constant dim"))
                .product();
            let off = pool;
            pool += (n * type_bytes(ty)).div_ceil(16) * 16;
            format!("{ty} (&{name}){dims} = *reinterpret_cast<{ty}(*){dims}>(tang_sm + {off});")
        } else if t.starts_with("extern __shared__") {
            panic!("megakernel: dynamic shared memory in kernel {i}");
        } else {
            line.to_string()
        };
        let l = l
            .replace("__syncthreads()", &format!("tang_bar<TANG_BAR, {w}>()"))
            .replace("blockIdx.x", "tang_bid")
            .replace("blockIdx.y", "0u")
            .replace("threadIdx.x", "tang_tid")
            .replace("blockDim.x", &format!("{w}u"))
            .replace("gridDim.x", "tang_bid_n");
        out.push_str(&l);
        out.push('\n');
    }
    writeln!(out, "}} // namespace k{i}").unwrap();
    let out = out.replace("extern \"C\" ", "").replace("__restrict__", "");
    (out, pool)
}

/// The program as one persistent kernel (see [`MegaOpts`]). Launch it
/// with a co-resident grid of `threads`-thread blocks, `smem` dynamic
/// shared bytes, and arguments `(tab, sync)`.
pub fn megakernel(stages: &[MegaStage], opts: &MegaOpts) -> Mega {
    let threads = if opts.threads == 0 { 256 } else { opts.threads };
    let dup = replicated(stages, opts.dup);
    let phases = phases(stages, &dup);
    let written: Vec<usize> = stages.iter().map(|st| split(&st.args).1).collect();
    let mut table: Vec<usize> = Vec::new();
    for st in stages {
        for &s in &st.args {
            if !table.contains(&s) {
                table.push(s);
            }
        }
    }
    let local = |s: usize| table.iter().position(|&t| t == s).unwrap();
    // per boundary: (slot, bytes) to prefetch, from the phases after it
    let prefetch_before = |p: usize| -> Vec<(usize, usize)> {
        let mut want: Vec<(usize, usize)> = Vec::new();
        let mut total = 0usize;
        for ks in &phases[p..] {
            if total >= opts.prefetch {
                break;
            }
            for &k in ks {
                for &s in split(&stages[k].args).0 {
                    if !written.contains(&s) && !want.iter().any(|w| w.0 == s) {
                        let b = opts.slot_bytes.get(s).copied().unwrap_or(0);
                        want.push((s, b));
                        total += b;
                    }
                }
            }
        }
        if total == 0 {
            return want;
        }
        let f = (opts.prefetch as f64 / total as f64).min(1.0);
        want.into_iter()
            .map(|(s, b)| (s, ((b as f64 * f) as usize).div_ceil(128) * 128))
            .filter(|w| w.1 > 0)
            .collect()
    };
    let mut src = String::from(HEADER);
    let mut smem = 0usize;
    let mut pools = Vec::with_capacity(stages.len());
    for (i, st) in stages.iter().enumerate() {
        assert!(
            threads % st.threads == 0 && threads / st.threads <= 15,
            "megakernel: stage {i} has {} threads, block has {threads}",
            st.threads
        );
        let konst: Vec<bool> = split(&st.args)
            .0
            .iter()
            .map(|s| !written.contains(s))
            .collect();
        let (s, pool) = stage_source(i, st, &konst);
        smem = smem.max(pool * (threads / st.threads) as usize);
        pools.push(pool);
        src.push_str(&s);
    }
    writeln!(src, "#define TANG_SMEM {smem}").unwrap();
    writeln!(
        src,
        "extern \"C\" __global__ void __launch_bounds__({threads}{}) main0(const unsigned long long* tab, unsigned* sync) {{",
        if opts.min_blocks > 0 { format!(", {}", opts.min_blocks) } else { String::new() }
    )
    .unwrap();
    src.push_str("    const unsigned tid = threadIdx.x;\n");
    let stamp = |src: &mut String, p: usize| {
        if opts.trace {
            writeln!(src, "    if (blockIdx.x == 0 && tid == 0) {{ unsigned long long t; asm volatile(\"mov.u64 %0, %globaltimer;\" : \"=l\"(t)); ((unsigned long long*)(sync + 2))[{p}] = t; }}").unwrap();
        }
    };
    for (p, ks) in phases.iter().enumerate() {
        if p > 0 {
            if opts.prefetch > 0 {
                for (slot, bytes) in prefetch_before(p) {
                    writeln!(src, "    tang_prefetch(tab[{}], {bytes}ull);", local(slot)).unwrap();
                }
            }
            writeln!(src, "    tang_grid_sync(sync, {p}u * gridDim.x);").unwrap();
        }
        stamp(&mut src, p);
        for (n, &k) in ks.iter().enumerate() {
            let st = &stages[k];
            let w = st.threads;
            let per = threads / w;
            if n > 0 {
                src.push_str("    __syncthreads();\n");
            }
            let ptrs: Vec<String> = st
                .args
                .iter()
                .enumerate()
                .map(|(a, &s)| {
                    let ty = if a + 1 == st.args.len() {
                        "float*"
                    } else {
                        "const float*"
                    };
                    format!("({ty})tab[{}]", local(s))
                })
                .collect();
            let pool = if per > 1 {
                // slot stride: this stage's pool
                format!("tang_dyn + (tid / {w}u) * {}u", pools[k])
            } else {
                "tang_dyn".to_string()
            };
            // one branch per side-by-side slot, each with its own barrier id
            for slot in 0..per {
                let (first, stride) = if dup[k] {
                    (format!("{slot}u"), format!("{per}u"))
                } else {
                    (
                        format!("blockIdx.x * {per}u + {slot}u"),
                        format!("gridDim.x * {per}u"),
                    )
                };
                let guard = if per == 1 {
                    "    {".to_string()
                } else {
                    format!(
                        "    {}if (tid / {w}u == {slot}u) {{",
                        if slot > 0 { "else " } else { "" }
                    )
                };
                let bar = slot + 1;
                writeln!(
                    src,
                    "{guard}\n    for (unsigned vb = {first}; vb < {nb}u; vb += {stride}) {{\n        k{k}::run<{bar}>({args}, {count}u, vb, tid % {w}u, {nb}u, {pool});\n        tang_bar<{bar}, {w}>();\n    }} }}",
                    nb = st.blocks,
                    args = ptrs.join(", "),
                    count = st.count,
                )
                .unwrap();
            }
        }
    }
    if opts.trace {
        writeln!(
            src,
            "    tang_grid_sync(sync, {}u * gridDim.x);",
            phases.len()
        )
        .unwrap();
        stamp(&mut src, phases.len());
    }
    // the last block out resets the barrier state for the next launch
    src.push_str(
        "    __syncthreads();\n    if (tid == 0) {\n        __threadfence();\n        if (atomicAdd(sync + 1, 1u) == gridDim.x - 1) { sync[0] = 0; sync[1] = 0; __threadfence(); }\n    }\n}\n",
    );
    Mega {
        source: src,
        threads,
        smem: smem as u32,
        phases,
        table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(args: &[usize], blocks: u32) -> MegaStage<'static> {
        MegaStage {
            source: "extern \"C\" __global__ void __launch_bounds__(128) main0(const float* __restrict__ b0, float* __restrict__ out, unsigned count) {\n    __shared__ float part[4][2];\n    if (threadIdx.x == 0) part[0][0] = __ldg(b0 + blockIdx.x);\n    __syncthreads();\n    out[blockIdx.x] = part[0][0] + (float)gridDim.x;\n}\n",
            threads: 128,
            blocks,
            count: 128 * blocks,
            args: args.to_vec(),
        }
    }

    #[test]
    fn hazards_level_phases() {
        // 1 reads 0's output; 2 is independent; 3 overwrites what 1 read
        let s = [
            st(&[10, 1], 4),
            st(&[1, 2], 4),
            st(&[11, 3], 4),
            st(&[12, 10], 4),
        ];
        let none = vec![false; 4];
        assert_eq!(phases(&s, &none), vec![vec![0, 2], vec![1, 3]]);
        // a replicated one-block producer lets its reader share its phase
        let s = [st(&[10, 1], 1), st(&[1, 2], 4)];
        assert_eq!(phases(&s, &replicated(&s, 1)), vec![vec![0, 1]]);
    }

    #[test]
    fn stage_rewrites() {
        let s = [st(&[10, 1], 4), st(&[1, 2], 4)];
        let m = megakernel(&s, &MegaOpts::default());
        assert_eq!(m.phases.len(), 2);
        assert_eq!(m.table, vec![10, 1, 2]);
        let src = &m.source;
        // virtual block coordinates, pooled shared memory, immediate barriers
        assert!(src.contains("namespace k0 {") && src.contains("namespace k1 {"));
        assert!(src.contains("tang_bid_n") && !src.contains("gridDim.x + "));
        assert!(
            src.contains("float (&part)[4][2] = *reinterpret_cast<float(*)[4][2]>(tang_sm + 0);")
        );
        assert!(src.contains("tang_bar<TANG_BAR, 128>()"));
        assert!(src.contains("k0::run<1>") && src.contains("k0::run<2>"));
        // the graph input keeps the read-only path, the intermediate does not
        assert!(src.contains("__ldg(b0 + tang_bid)"));
        assert!(src.contains("tang_ld(b0 + tang_bid)"));
        assert!(!src.contains("__restrict__"));
        // 2 slots of 128 threads, 32 bytes of pool each
        assert_eq!(m.smem, 64);
    }
}
