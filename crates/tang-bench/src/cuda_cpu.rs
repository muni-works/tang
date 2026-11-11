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
    writeln!(cpp, "    std::vector<std::vector<float>> B({});", p.buffers.len()).unwrap();
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
        args.push(format!("B[{}].data()", bi(k.node)));
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
                .map(|(n, _)| if n == "tokens" { t.to_vec() } else { w[n].clone() })
                .collect()
        };
        let st = derive(&m, k, Rows::All).unwrap();
        let nodes: Vec<TensorId> = st.caches.iter().map(|c| c.node).collect();
        let pre = m.g.bind(&[("T", p0)]).unwrap().eval(&ins(&toks[..p0]), &nodes);
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
            let pg = crate::ir_coder_prefill::prefill_graph(
                &crate::ir_coder_prefill::CoderDims::tiny(),
            );
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
            .map(|v| v.iter().map(|&x| if x.is_nan() { 0.0 } else { x as f32 }).collect())
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
    /// Chunked prefill from an empty cache, then decode, with every step
    /// run as its CUDA program on the CPU: each chunk is the derived step
    /// with `k = chunk` (it writes the caches), each decode step `k = 1`
    /// reads them. The logits of every step against the full prefill on
    /// the interpreter: `(worst rel err, kernels per step)`.
    pub(crate) fn chunked_then_decode(m: &Model, ks: &[usize]) -> (f64, Vec<usize>) {
        let w = weights(m);
        let total: usize = ks.iter().sum();
        assert!(total <= m.cap);
        let toks: Vec<f64> = (0..total)
            .map(|i| ((37 * i + 5) % vocab(m)) as f64)
            .collect();
        let ins = |t: &[f64]| -> Vec<Vec<f64>> {
            m.inputs
                .iter()
                .map(|(n, _)| if n == "tokens" { t.to_vec() } else { w[n].clone() })
                .collect()
        };
        let mut caches: Option<Vec<Vec<f32>>> = None;
        let (mut pos, mut worst, mut nks) = (0usize, 0.0f64, vec![]);
        let vocab = vocab(m);
        for &k in ks {
            let st = derive(m, k, Rows::All).unwrap();
            let cs = caches.get_or_insert_with(|| {
                // empty session: zero rows (masked) and zero states
                st.caches
                    .iter()
                    .map(|c| {
                        let n: usize = concrete(st.graph.shape(c.input), 1).iter().product();
                        vec![0.0f32; n]
                    })
                    .collect()
            });
            let step_ins: Vec<Vec<f32>> = st
                .inputs
                .iter()
                .map(|&x| match x {
                    StepInput::Pos => vec![pos as f32],
                    StepInput::New(_) => toks[pos..pos + k].iter().map(|&t| t as f32).collect(),
                    StepInput::Same(i) => w[&m.inputs[i].0].iter().map(|&v| v as f32).collect(),
                    StepInput::Cache(c) => cs[c].clone(),
                })
                .collect();
            let mut ids = st.outputs.clone();
            ids.extend(st.caches.iter().map(|c| c.updated));
            let (g, outs) = st.graph.substitute_maps(&ids);
            let (g, outs) = g.eliminate_common(&outs);
            let dp = g.to_cuda_decode(&outs, &DecodeOpts::default());
            nks.push(dp.program.kernels.len());
            let got = super::run(&dp, &step_ins, &outs);
            let grown = pos + k;
            let want = m
                .g
                .bind(&[("T", grown)])
                .unwrap()
                .eval(&ins(&toks[..grown]), &[m.logits]);
            let g64: Vec<f64> = got[0].iter().map(|&x| x as f64).collect();
            worst = worst.max(rel(&g64, &want[0][pos * vocab..grown * vocab]));
            *cs = got[1..].to_vec();
            pos = grown;
        }
        (worst, nks)
    }

    #[test]
    fn chunked_prefill_then_decode_on_the_cpu() {
        for m in [qwen(), coder()] {
            for ks in [&[8usize, 8, 1, 1][..], &[16, 1, 1, 1][..], &[5, 3, 1][..]] {
                let (e, nk) = chunked_then_decode(&m, ks);
                eprintln!("{} chunks {ks:?}: {nk:?} kernels, logits rel err {e:.2e}", m.name);
                assert!(e < 1e-4, "{} {ks:?}: {e}", m.name);
            }
        }
    }
}
