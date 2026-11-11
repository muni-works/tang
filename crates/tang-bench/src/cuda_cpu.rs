//! Generated CUDA kernels run on the CPU: every kernel of a decode program
//! (`to_cuda_decode`) compiled as C++ by the host `g++` against a small
//! emulation of the CUDA builtins, each block's threads as OS threads
//! (`__syncthreads` a block barrier, warp shuffles through a per-warp
//! exchange with a warp barrier), blocks one after the other. A check of
//! the lowering itself (the very kernel sources a GPU would run, fused
//! ones included) against the interpreter without a GPU. f32 buffers
//! only; small graphs only (one OS thread per CUDA thread).

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use tang_compile::tensor::cuda_gemv::DecodeProgram;
use tang_compile::tensor::TensorId;

const HEADER: &str = r#"
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
using std::sqrt; using std::exp; using std::log; using std::round; using std::floor;
using std::ceil; using std::fabs; using std::exp2; using std::log2; using std::tanh;
struct tang_dim3 { unsigned x, y, z; };
static thread_local tang_dim3 threadIdx;
static tang_dim3 blockIdx, blockDim, gridDim;
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
static inline float2 make_float2(float x, float y) { return {x, y}; }
static inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
struct uint4 { unsigned x, y, z, w; };
struct uint2 { unsigned x, y; };
struct int4 { int x, y, z, w; };
#define __device__
#define __host__
#define __forceinline__ inline
#define __global__
#define __shared__ static
#define __launch_bounds__(...)
#define __restrict__ __restrict
#define __noinline__
template <class T> static inline T __ldg(const T* p) { return *p; }
static inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
static inline int __float_as_int(float f) { int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
static inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
static inline float min(float a, float b) { return std::fmin(a, b); }
static inline float max(float a, float b) { return std::fmax(a, b); }
static inline float __fdividef(float a, float b) { return a / b; }
static inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
static inline float __frcp_rn(float x) { return 1.0f / x; }
static inline float tang_sw_h2f(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}
static inline unsigned short tang_sw_f2h(float f) {
    // round to nearest even through the hardware-free route: good enough
    // for the f32 checks here (no f16 buffers)
    unsigned u = __float_as_uint(f), s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 255) - 127 + 15;
    unsigned m = u & 0x7fffffu;
    if (e <= 0) return (unsigned short)s;
    if (e >= 31) return (unsigned short)(s | 0x7c00u);
    unsigned r = (unsigned)(e << 10) | (m >> 13);
    unsigned rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
    return (unsigned short)(s | r);
}
static std::barrier<>* tang_block;
static std::barrier<>* tang_warp[32];
static uint64_t tang_xch[32][32];
static inline void __syncthreads() { tang_block->arrive_and_wait(); }
static inline void __syncwarp(unsigned = 0xffffffffu) { tang_warp[threadIdx.x >> 5]->arrive_and_wait(); }
template <class T> static inline T tang_shfl(T v, int src_lane) {
    unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    uint64_t bits = 0; std::memcpy(&bits, &v, sizeof(T));
    tang_xch[w][lane] = bits;
    tang_warp[w]->arrive_and_wait();
    T r; uint64_t got = tang_xch[w][src_lane & 31]; std::memcpy(&r, &got, sizeof(T));
    tang_warp[w]->arrive_and_wait();
    return r;
}
template <class T> static inline T __shfl_xor_sync(unsigned, T v, int off, int = 32) { return tang_shfl(v, (int)(threadIdx.x & 31) ^ off); }
template <class T> static inline T __shfl_down_sync(unsigned, T v, unsigned off, int = 32) {
    int l = (int)(threadIdx.x & 31) + (int)off; return tang_shfl(v, l < 32 ? l : (int)(threadIdx.x & 31));
}
template <class T> static inline T __shfl_up_sync(unsigned, T v, unsigned off, int = 32) {
    int l = (int)(threadIdx.x & 31) - (int)off; return tang_shfl(v, l >= 0 ? l : (int)(threadIdx.x & 31));
}
template <class T> static inline T __shfl_sync(unsigned, T v, int src, int width = 32) {
    int base = (int)(threadIdx.x & 31) & ~(width - 1); return tang_shfl(v, base + (src % width));
}
template <class F> static void tang_launch(unsigned blocks, unsigned threads, F f) {
    gridDim = {blocks, 1, 1};
    blockDim = {threads, 1, 1};
    for (unsigned b = 0; b < blocks; b++) {
        blockIdx = {b, 0, 0};
        std::barrier<> block((std::ptrdiff_t)threads);
        tang_block = &block;
        std::vector<std::barrier<>*> ws;
        for (unsigned w = 0; w * 32 < threads; w++) {
            unsigned n = threads - w * 32 < 32 ? threads - w * 32 : 32;
            ws.push_back(new std::barrier<>((std::ptrdiff_t)n));
            tang_warp[w] = ws.back();
        }
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < threads; t++) {
            ts.emplace_back([t, &f] {
                threadIdx = {t, 0, 0};
                f();
                tang_warp[t >> 5]->arrive_and_drop();
                tang_block->arrive_and_drop();
            });
        }
        for (auto& t : ts) t.join();
        for (auto* w : ws) delete w;
    }
}
"#;

/// A kernel source made host C++: the PTX `asm` conversions replaced,
/// `extern "C"` dropped (each kernel lives in its own namespace), dynamic
/// shared memory as a static array of `TANG_SMEM` bytes.
fn host_source(src: &str) -> String {
    let mut s = src
        .replace(
            r#"asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(s));"#,
            "f = tang_sw_h2f(s);",
        )
        .replace(
            r#"asm("cvt.rn.f16.f32 %0, %1;" : "=h"(s) : "f"(f));"#,
            "s = tang_sw_f2h(f);",
        )
        .replace("extern \"C\" ", "");
    let smem = src
        .lines()
        .find_map(|l| l.strip_prefix("#define TANG_SMEM "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    if let Some(i) = s.find("extern __shared__") {
        // `extern __shared__ T name[];` → a static array as wide
        let end = i + s[i..].find(';').unwrap();
        let decl = s[i..end].replace("extern __shared__", "static");
        let decl = decl.replace("[]", &format!("[{}]", smem.max(16) / 4 + 4));
        s.replace_range(i..end, &decl);
    }
    s
}

/// Runs `dp` on the CPU with graph input `k` holding `inputs[k]` (every
/// other buffer zero) and returns the values of `outs`.
pub fn run(dp: &DecodeProgram, inputs: &[Vec<f32>], outs: &[TensorId]) -> Vec<Vec<f32>> {
    let p = &dp.program;
    let bi = |t: TensorId| {
        let t = p.storage(t);
        p.buffers
            .iter()
            .position(|b| b.node == t)
            .expect("tensor has no buffer")
    };
    let mut cpp = String::from(HEADER);
    for (j, k) in p.kernels.iter().enumerate() {
        writeln!(cpp, "namespace k{j} {{\n{}\n}}", host_source(&k.source)).unwrap();
    }
    cpp.push_str("int main(int argc, char** argv) {\n");
    writeln!(
        cpp,
        "    std::vector<std::vector<float>> B({});",
        p.buffers.len()
    )
    .unwrap();
    for (j, b) in p.buffers.iter().enumerate() {
        writeln!(cpp, "    B[{j}].assign({} + 4, 0.0f);", b.len.max(1)).unwrap();
    }
    cpp.push_str("    FILE* in = std::fopen(argv[1], \"rb\");\n");
    for (j, b) in p.buffers.iter().enumerate() {
        if let Some(k) = b.input {
            let n = inputs[k].len();
            assert!(n <= b.len.max(1), "input {k}: {n} values, buffer {}", b.len);
            writeln!(
                cpp,
                "    if (std::fread(B[{j}].data(), 4, {n}, in) != {n}) return 2;"
            )
            .unwrap();
        }
    }
    cpp.push_str("    std::fclose(in);\n");
    for (j, k) in p.kernels.iter().enumerate() {
        let mut args: Vec<String> = k
            .reads
            .iter()
            .map(|&t| format!("B[{}].data()", bi(t)))
            .collect();
        for t in k.outputs() {
            args.push(format!("B[{}].data()", bi(t)));
        }
        args.push(format!("{}u", k.points));
        let blocks = k.count.div_ceil(k.workgroup_size);
        writeln!(
            cpp,
            "    tang_launch({blocks}u, {}u, [&] {{ k{j}::main0({}); }});",
            k.workgroup_size,
            args.join(", ")
        )
        .unwrap();
    }
    cpp.push_str("    FILE* o = std::fopen(argv[2], \"wb\");\n");
    for &t in outs {
        let j = bi(t);
        writeln!(
            cpp,
            "    std::fwrite(B[{j}].data(), 4, {}, o);",
            p.buffers[j].len
        )
        .unwrap();
    }
    cpp.push_str("    std::fclose(o);\n    return 0;\n}\n");

    let dir: PathBuf = std::env::temp_dir().join(format!(
        "tang-cuda-cpu-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, exe, fin, fout) = (
        dir.join("prog.cpp"),
        dir.join("prog"),
        dir.join("in.bin"),
        dir.join("out.bin"),
    );
    std::fs::write(&src, &cpp).unwrap();
    let cc = Command::new("g++")
        .args(["-std=c++20", "-O1", "-w", "-pthread", "-o"])
        .arg(&exe)
        .arg(&src)
        .output()
        .expect("g++");
    assert!(
        cc.status.success(),
        "g++ failed on {}:\n{}",
        src.display(),
        String::from_utf8_lossy(&cc.stderr)
            .lines()
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mut bytes = Vec::new();
    for b in &p.buffers {
        if let Some(k) = b.input {
            bytes.extend(inputs[k].iter().flat_map(|x| x.to_le_bytes()));
        }
    }
    std::fs::write(&fin, bytes).unwrap();
    let st = Command::new(&exe).arg(&fin).arg(&fout).status().unwrap();
    assert!(st.success(), "{} exited {st}", exe.display());
    let raw = std::fs::read(&fout).unwrap();
    let mut words = raw
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let got = outs
        .iter()
        .map(|&t| {
            let n = p.buffers[bi(t)].len;
            (&mut words).take(n).collect()
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    got
}

#[cfg(test)]
mod tests {
    use crate::derive_all::*;
    use tang_compile::tensor::cuda_gemv::DecodeOpts;
    use tang_compile::tensor::step::{Rows, StepInput};
    use tang_compile::tensor::TensorId;

    /// The tiny coder-next step (prefill 5 tokens on the interpreter, one
    /// derived step) as its CUDA decode program run on the CPU, against
    /// the interpreter's step: `(worst rel err, kernels)`.
    pub(crate) fn coder_step_on_cpu(
        opts: &DecodeOpts,
        keep_residual: bool,
        staged: bool,
        cokernels: bool,
    ) -> (f64, usize) {
        let m = coder();
        let w = weights(&m);
        let p0 = 5;
        let k = 1;
        let toks: Vec<f64> = (0..p0 + k)
            .map(|i| ((37 * i + 5) % vocab(&m)) as f64)
            .collect();
        let ins = |t: &[f64]| -> Vec<Vec<f64>> {
            m.inputs
                .iter()
                .map(|(n, _)| {
                    if n == "tokens" {
                        t.to_vec()
                    } else {
                        w[n].clone()
                    }
                })
                .collect()
        };
        let st = derive(&m, k, Rows::All).unwrap();
        let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
        let pre =
            m.g.bind(&[("T", p0)])
                .unwrap()
                .eval(&ins(&toks[..p0]), &nodes);
        let caches: Vec<Vec<f64>> = st
            .caches
            .iter()
            .zip(pre)
            .map(|(c, v)| seed(&m, c, v, p0))
            .collect();
        let step_ins: Vec<Vec<f64>> = st
            .inputs
            .iter()
            .map(|&x| match x {
                StepInput::Pos => vec![p0 as f64],
                StepInput::New(_) => toks[p0..p0 + k].to_vec(),
                StepInput::Same(i) => w[&m.inputs[i].0].clone(),
                StepInput::Cache(c) => caches[c].clone(),
            })
            .collect();
        let mut ids = st.outputs.clone();
        ids.extend(st.caches.iter().map(|c| c.updated));
        let want = st.graph.eval(&step_ins, &ids);
        let n = ids.len();
        let mut keep = ids.clone();
        if keep_residual {
            let pg =
                crate::ir_coder_prefill::prefill_graph(&crate::ir_coder_prefill::CoderDims::tiny());
            keep.extend(crate::ir_coder_real::residual_stream(&st.graph, &pg));
        }
        let (g, keep) = st.graph.substitute_maps(&keep);
        let (mut g, keep) = g.eliminate_common(&keep);
        let outs = keep[..n].to_vec();
        if staged {
            let k = g.schedule_staged_stats(&outs, &[]);
            assert!(k > 0, "no statistic staged");
            let m = g.schedule_small_maps(&outs, &[]);
            assert!(m > 0, "no small map inlined");
        }
        if cokernels {
            let (a, b) = g.schedule_cokernels(&outs, opts);
            eprintln!("cokernels: {a} inlined and stored, {b} side jobs");
            assert!(a + b > 0, "no multi-output kernel");
            if std::env::var("LIST").is_ok() {
                for (i, nd) in g.nodes().iter().enumerate() {
                    if let Some(h) = nd.host {
                        eprintln!("host n{i} {:?} -> n{}", nd.shape, h.index());
                    }
                }
            }
        }
        let dp = g.to_cuda_decode(&outs, opts);
        let nk = dp.program.kernels.len();
        if std::env::var("LIST").is_ok() {
            for k in &dp.program.kernels {
                eprintln!(
                    "K n{} writes {:?} | {}",
                    k.node.index(),
                    k.writes.iter().map(|t| t.index()).collect::<Vec<_>>(),
                    k.source.lines().next().unwrap_or("")
                );
            }
        }
        // NaN in unwritten cache rows: zero for the run (masked anyway),
        // skipped in the comparison
        let words: Vec<Vec<f32>> = step_ins
            .iter()
            .map(|v| {
                v.iter()
                    .map(|&x| if x.is_nan() { 0.0 } else { x as f32 })
                    .collect()
            })
            .collect();
        let got = super::run(&dp, &words, &outs);
        let mut worst = 0.0f64;
        for (oi, (gv, wv)) in got.iter().zip(&want).enumerate() {
            let (gk, wk): (Vec<f64>, Vec<f64>) = gv
                .iter()
                .zip(wv)
                .filter(|(_, w)| !w.is_nan())
                .map(|(a, b)| (*a as f64, *b))
                .unzip();
            let e = rel(&gk, &wk);
            if e > 1e-4 {
                eprintln!("output {oi} (node {}): rel err {e:.2e}", outs[oi].index());
            }
            worst = worst.max(e);
        }
        (worst, nk)
    }

    /// The tiny coder step with multi-output kernels
    /// (`schedule_cokernels`: residual adds folded into their norms, the
    /// conv window into the conv, ring and state updates as side jobs)
    /// against the interpreter, and the kernel counts with and without.
    #[test]
    fn coder_step_cokernels_run_on_the_cpu() {
        let (e0, n0) = coder_step_on_cpu(&DecodeOpts::default(), true, true, false);
        let (e1, n1) = coder_step_on_cpu(&DecodeOpts::default(), true, true, true);
        eprintln!(
            "tiny coder step: {n0} kernels (rel err {e0:.2e}) -> {n1} with multi-output \
             kernels (rel err {e1:.2e} vs interpreter)"
        );
        assert!(e1 < 1e-4, "{e1}");
        assert!(n1 < n0, "{n1} >= {n0}");
    }

    #[test]
    fn coder_step_kernels_run_on_the_cpu() {
        for staged in [false, true] {
            let (e, nk) = coder_step_on_cpu(&DecodeOpts::default(), true, staged, false);
            eprintln!(
                "tiny coder step (staged stats {staged}): {nk} CUDA kernels on the CPU vs \
                 interpreter: rel err {e:.2e}"
            );
            assert!(e < 1e-4, "{e}");
        }
    }

    /// The shapes `schedule_cokernels` fuses at real size, on a small
    /// graph: a residual add read by its RMSNorm (whose sum of squares
    /// is staged) and by a later map, the norm read by its Q8x32 copy
    /// and the later map; a zero-padded conv window (old ring rows, then
    /// the new row) read by a depthwise conv and by the shifted ring;
    /// an independent state update. Multi-output kernels against the
    /// one-output kernels, on the CPU: the same bytes, fewer launches.
    #[test]
    fn cokernel_patterns_match_unfused() {
        use tang::Scalar;
        use tang_compile::tensor::{Arg, Dim, Form, Format, Monoid, Precision, TensorGraph, View};
        let (d, c) = (256usize, 64usize);
        let mut g = TensorGraph::new();
        let x = g.input(&[1, d]);
        let r = g.input(&[1, d]);
        let w = g.input(&[1, d]);
        let ring = g.input(&[3, c]);
        let new = g.input(&[1, c]);
        let kw = g.input(&[c, 4]);
        let st = g.input(&[c, 8]);
        let a = g.map(&[x, r], |v| v[0] + v[1]).unwrap();
        let sq = g.map(&[a], |v| v[0] * v[0]).unwrap();
        let ss = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let ssb = g.broadcast(ss, &[Dim::Fixed(1), Dim::Fixed(d)]).unwrap();
        let nrm = g
            .map(&[a, ssb, w], |v| {
                let one = tang_expr::ExprId::ONE;
                let k = tang_expr::ExprId::from_f64(1.0 / d as f64);
                let eps = tang_expr::ExprId::from_f64(1e-6);
                v[0] * (one / (v[1] * k + eps).sqrt()) * v[2]
            })
            .unwrap();
        let q = g.map(&[nrm], |v| v[0]).unwrap();
        g.set_precision(q, Precision::storage(Format::Q8x32));
        let m = g.map(&[nrm, a], |v| v[0] * v[1]).unwrap();
        // win[t, ch] = ring[t, ch] (t < 3), new[t - 3, ch] (t = 3)
        let f = |coefs: &[i64], offset: i64| Form {
            coefs: coefs.to_vec(),
            offset,
        };
        let pad = |forms: Vec<Form>| {
            View {
                forms,
                guard: false,
                gather: None,
            }
            .guarded()
        };
        let win = g
            .comp(
                &[4, c],
                &[] as &[usize],
                vec![
                    Arg {
                        src: ring,
                        view: pad(vec![f(&[1, 0], 0), f(&[0, 1], 0)]),
                    },
                    Arg {
                        src: new,
                        view: pad(vec![f(&[1, 0], -3), f(&[0, 1], 0)]),
                    },
                ],
                None,
                |v| vec![v[0] + v[1]],
            )
            .unwrap();
        let conv = g
            .comp(
                &[c],
                &[4],
                vec![
                    Arg {
                        src: win,
                        view: View::pick(2, &[Some(1), Some(0)]),
                    },
                    Arg {
                        src: kw,
                        view: View::pick(2, &[Some(0), Some(1)]),
                    },
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap();
        let shifted = g
            .comp(
                &[3, c],
                &[] as &[usize],
                vec![Arg {
                    src: win,
                    view: View {
                        forms: vec![f(&[1, 0], 1), f(&[0, 1], 0)],
                        guard: false,
                        gather: None,
                    },
                }],
                None,
                |v| vec![v[0]],
            )
            .unwrap();
        let upd = g
            .comp(
                &[c, 8],
                &[] as &[usize],
                vec![
                    Arg {
                        src: st,
                        view: View::identity(2, 2),
                    },
                    Arg {
                        src: new,
                        view: View::pick(2, &[None, Some(0)]),
                    },
                ],
                None,
                |v| vec![v[0] * tang_expr::ExprId::from_f64(0.5) + v[1]],
            )
            .unwrap();
        let outs0 = [a, m, q, conv, shifted, upd];
        let mut keep = outs0.to_vec();
        keep.extend([nrm, win]);
        let (mut g, keep) = g.substitute_maps(&keep);
        let outs = keep[..outs0.len()].to_vec();
        assert!(g.schedule_staged_stats(&outs, &[]) > 0);
        let ins: Vec<Vec<f32>> = [d, d, d, 3 * c, c, 4 * c, 8 * c]
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                (0..n)
                    .map(|i| (((i * 37 + k * 11) % 101) as f32 - 50.0) / 23.0)
                    .collect()
            })
            .collect();
        let run = |g: &TensorGraph| {
            let dp = g.to_cuda_decode(&outs, &DecodeOpts::default());
            (super::run(&dp, &ins, &outs), dp.program.kernels.len())
        };
        let (one, n1) = run(&g);
        let mut fg = g.clone();
        let (ai, bj) = fg.schedule_cokernels(&outs, &DecodeOpts::default());
        let (multi, nm) = run(&fg);
        let bits = |v: &[Vec<f32>]| {
            v.iter()
                .map(|o| o.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        eprintln!(
            "cokernel patterns: {n1} one-output kernels -> {nm} ({ai} inlined and stored, \
             {bj} side jobs); bytes identical: {}",
            bits(&one) == bits(&multi)
        );
        assert!(one.iter().all(|o| o.iter().any(|&x| x != 0.0)));
        assert!(nm < n1);
        assert_eq!(bits(&one), bits(&multi));
    }

    /// An RMSNorm stored as Q8x32 (what an int8 GEMV reads): the sum of
    /// squares staged into the norm's kernel (`schedule_staged_stats`,
    /// one launch) against its own warp kernel plus the transpiled Q8x32
    /// writer (two launches), on the CPU: the same bytes, codes and
    /// scales.
    #[test]
    fn staged_q8_norm_matches_two_kernels() {
        use tang::Scalar;
        use tang_compile::tensor::{Dim, Format, Monoid, Precision, TensorGraph};
        for f in [Format::Q8x32, Format::Q8x32F] {
            let d = 256;
            let mut g = TensorGraph::new();
            let x = g.input(&[Dim::Fixed(2), Dim::Fixed(d)]);
            let w = g.input(&[Dim::Fixed(1), Dim::Fixed(d)]);
            let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
            let ss = g.reduce(sq, 1, Monoid::sum()).unwrap();
            let ssb = g.broadcast(ss, &[Dim::Fixed(2), Dim::Fixed(d)]).unwrap();
            let wb = g.broadcast(w, &[Dim::Fixed(2), Dim::Fixed(d)]).unwrap();
            let y = g
                .map(&[x, ssb, wb], |v| {
                    let one = tang_expr::ExprId::ONE;
                    let k = tang_expr::ExprId::from_f64(1.0 / d as f64);
                    let eps = tang_expr::ExprId::from_f64(1e-6);
                    v[0] * (one / (v[1] * k + eps).sqrt()) * v[2]
                })
                .unwrap();
            g.set_precision(y, Precision::storage(f));
            let (g, outs) = g.substitute_maps(&[y]);
            let xs: Vec<f32> = (0..2 * d)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
                .collect();
            let ws: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 9.0).collect();
            let run = |g: &TensorGraph| {
                let dp = g.to_cuda_decode(&outs, &DecodeOpts::default());
                let nk = dp.program.kernels.len();
                (
                    super::run(&dp, &[xs.clone(), ws.clone()], &outs).remove(0),
                    nk,
                )
            };
            let (two, n2) = run(&g);
            let mut sg = g.clone();
            assert_eq!(sg.schedule_staged_stats(&outs, &[]), 1);
            let (one, n1) = run(&sg);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            eprintln!(
                "{f:?} norm: {n2} kernels unfused, {n1} staged; bytes identical: {}",
                bits(&two) == bits(&one)
            );
            assert_eq!((n2, n1), (2, 1));
            assert!(two.iter().any(|&x| x != 0.0));
            assert_eq!(bits(&two), bits(&one), "{f:?}");
        }
    }
}
