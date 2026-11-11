//! Generated CUDA kernels run on the CPU: every kernel of a decode program
//! (`to_cuda_decode`) compiled as C++ by the host `g++` against a small
//! emulation of the CUDA builtins, each block's threads as OS threads
//! (`__syncthreads` a block barrier, warp shuffles through a per-warp
//! exchange with a warp barrier), blocks one after the other. A check of
//! the lowering itself (the very kernel sources a GPU would run, fused
//! ones included) against the interpreter without a GPU. Block-format
//! buffers (GGUF / DeepSeek weights, Q8x32 activations) hold their raw
//! words as on the device; `dp4a`, `__byte_perm` and the f16 conversion are
//! emulated (not the int8 MMA: the int8 GEMV's `Gemv` stage only). One OS
//! thread per CUDA thread, blocks one after the other.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use tang_compile::tensor::cuda_gemv::DecodeProgram;
use tang_compile::tensor::TensorId;

const HEADER: &str = r#"
#pragma once
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
inline thread_local tang_dim3 threadIdx;
inline tang_dim3 blockIdx, blockDim, gridDim;
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
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
static inline float2 make_float2(float x, float y) { return {x, y}; }
static inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
static inline int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += (int)(signed char)(a >> (8 * i)) * (int)(signed char)(b >> (8 * i));
    return c;
}
static inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    const uint64_t v = ((uint64_t)y << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; i++) r |= (unsigned)((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
    return r;
}
static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned sh) {
    sh &= 31; return sh ? (lo >> sh) | (hi << (32 - sh)) : lo;
}
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
inline std::barrier<>* tang_block;
inline std::barrier<>* tang_warp[32];
inline uint64_t tang_xch[32][32];
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
        .replace(
            r#"asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));"#,
            "f = tang_sw_h2f(h);",
        )
        .replace("extern \"C\" ", "");
    // the int8 MMA (not emulated: only the Gemv stage runs here)
    if let Some(i) = s.find("asm volatile(\"mma.sync") {
        let end = i + s[i..].find(");\n").unwrap() + 2;
        s.replace_range(i..end, "std::abort();");
    }
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
/// other buffer zero; a block-format input as its raw words, f32 bits) and
/// returns the values of `outs` (f32 outputs).
pub fn run(dp: &DecodeProgram, inputs: &[Vec<f32>], outs: &[TensorId]) -> Vec<Vec<f32>> {
    let p = &dp.program;
    let bi = |t: TensorId| {
        let t = p.storage(t);
        p.buffers
            .iter()
            .position(|b| b.node == t)
            .expect("tensor has no buffer")
    };
    // kernels in translation units of `PER_TU`, compiled in parallel (one
    // file of a few hundred kernels takes g++ more memory than a step's
    // weights)
    const PER_TU: usize = 8;
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "tang-cuda-cpu-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tang_cpu.h"), HEADER).unwrap();
    let launch = |j: usize| -> String {
        let k = &p.kernels[j];
        let mut args: Vec<String> = k
            .reads
            .iter()
            .map(|&t| format!("B[{}].data()", bi(t)))
            .collect();
        args.push(format!("B[{}].data()", bi(k.node)));
        args.push(format!("{}u", k.points));
        let blocks = k.count.div_ceil(k.workgroup_size);
        format!(
            "void run_k{j}(std::vector<std::vector<float>>& B) {{ tang_launch({blocks}u, {}u, [&] {{ k{j}::main0({}); }}); }}\n",
            k.workgroup_size,
            args.join(", ")
        )
    };
    let nk = p.kernels.len();
    let mut units = Vec::new();
    for (u, j0) in (0..nk).step_by(PER_TU).enumerate() {
        let mut cpp = String::from("#include \"tang_cpu.h\"\n");
        for j in j0..(j0 + PER_TU).min(nk) {
            writeln!(
                cpp,
                "namespace k{j} {{\n{}\n}}",
                host_source(&p.kernels[j].source)
            )
            .unwrap();
            cpp.push_str(&launch(j));
        }
        let f = dir.join(format!("k{u}.cpp"));
        std::fs::write(&f, cpp).unwrap();
        units.push(f);
    }
    let mut cpp = String::from("#include \"tang_cpu.h\"\n");
    for j in 0..nk {
        writeln!(cpp, "void run_k{j}(std::vector<std::vector<float>>& B);").unwrap();
    }
    cpp.push_str("int main(int argc, char** argv) {\n");
    writeln!(
        cpp,
        "    std::vector<std::vector<float>> B({});",
        p.buffers.len()
    )
    .unwrap();
    for (j, b) in p.buffers.iter().enumerate() {
        let words = b.format.words(b.len).max(1);
        writeln!(cpp, "    B[{j}].assign({words} + 4, 0.0f);").unwrap();
    }
    cpp.push_str("    FILE* in = std::fopen(argv[1], \"rb\");\n");
    for (j, b) in p.buffers.iter().enumerate() {
        if let Some(k) = b.input {
            let n = inputs[k].len();
            let words = b.format.words(b.len).max(1);
            assert!(n <= words, "input {k}: {n} words, buffer {words}");
            writeln!(
                cpp,
                "    if (std::fread(B[{j}].data(), 4, {n}, in) != {n}) return 2;"
            )
            .unwrap();
        }
    }
    cpp.push_str("    std::fclose(in);\n");
    for j in 0..nk {
        writeln!(cpp, "    run_k{j}(B);").unwrap();
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
    let main = dir.join("main.cpp");
    std::fs::write(&main, &cpp).unwrap();
    units.push(main);
    let (exe, fin, fout) = (dir.join("prog"), dir.join("in.bin"), dir.join("out.bin"));
    let jobs = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8);
    let objs: Vec<PathBuf> = units.iter().map(|u| u.with_extension("o")).collect();
    for chunk in units.iter().zip(&objs).collect::<Vec<_>>().chunks(jobs) {
        let kids: Vec<_> = chunk
            .iter()
            .map(|(u, o)| {
                Command::new("g++")
                    .args(["-std=c++20", "-O1", "-w", "-pthread", "-c", "-o"])
                    .arg(o)
                    .arg(u)
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .expect("g++")
            })
            .collect();
        for ((u, _), kid) in chunk.iter().zip(kids) {
            let out = kid.wait_with_output().unwrap();
            assert!(
                out.status.success(),
                "g++ failed on {}:\n{}",
                u.display(),
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .take(40)
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
    let ld = Command::new("g++")
        .args(["-pthread", "-o"])
        .arg(&exe)
        .args(&objs)
        .output()
        .expect("g++");
    assert!(
        ld.status.success(),
        "link failed: {}",
        String::from_utf8_lossy(&ld.stderr)
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
    pub(crate) fn coder_step_on_cpu(opts: &DecodeOpts, keep_residual: bool) -> (f64, usize) {
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
        let (g, keep) = g.eliminate_common(&keep);
        let outs = keep[..n].to_vec();
        let dp = g.to_cuda_decode(&outs, opts);
        let nk = dp.program.kernels.len();
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
        for (gv, wv) in got.iter().zip(&want) {
            let (gk, wk): (Vec<f64>, Vec<f64>) = gv
                .iter()
                .zip(wv)
                .filter(|(_, w)| !w.is_nan())
                .map(|(a, b)| (*a as f64, *b))
                .unzip();
            worst = worst.max(rel(&gk, &wk));
        }
        (worst, nk)
    }

    #[test]
    fn coder_step_kernels_run_on_the_cpu() {
        let (e, nk) = coder_step_on_cpu(&DecodeOpts::default(), true);
        eprintln!("tiny coder step, {nk} CUDA kernels on the CPU vs interpreter: rel err {e:.2e}");
        assert!(e < 1e-4, "{e}");
    }
}
