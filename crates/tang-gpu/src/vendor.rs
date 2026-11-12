//! Vendor GEMMs: a kernel whose source starts with
//! [`VENDOR_TAG`](tang_compile::tensor::VENDOR_TAG) (an int8 contraction
//! over many positions, `TANG_VENDOR_GEMM`) runs as a host call instead of
//! its stub: the GGUF weight and the Q8x32 activation are dequantized to
//! f16 scratch, then cuBLASLt computes `out[P, r] = Σ_k W[r, k]·X[P, k]`
//! with f32 accumulation and output. Every launch goes on the caller's
//! stream, so a vendor kernel captures into a CUDA graph like any other.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cublaslt::{result as lt, sys};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};

use crate::cuda::CudaDevice;

/// A parsed vendor GEMM (see `vendor_gemm_source` in tang-compile).
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub w: String,
    pub rows: usize,
    pub k: usize,
    pub npos: usize,
    /// W's first element (r = k = 0).
    pub wb: usize,
    /// W's element count (DS formats: where its scales start).
    pub wn: usize,
    /// X's first element and row stride, in elements.
    pub xb: usize,
    pub xld: usize,
    /// X's element count (where its Q8x32 scales start, in bytes).
    pub xn: usize,
    /// Read bindings of W and X.
    pub wq: usize,
    pub xq: usize,
}

/// `source`'s vendor GEMM, if it is one.
pub fn parse(source: &str) -> Option<Spec> {
    let line = source
        .strip_prefix(tang_compile::tensor::VENDOR_TAG)?
        .lines()
        .next()?;
    let kv: HashMap<&str, &str> = line
        .split_whitespace()
        .filter_map(|t| t.split_once('='))
        .collect();
    let n = |k: &str| kv.get(k)?.parse::<usize>().ok();
    Some(Spec {
        w: kv.get("w")?.to_string(),
        rows: n("rows")?,
        k: n("k")?,
        npos: n("npos")?,
        wb: n("wb")?,
        wn: n("wn").unwrap_or(0),
        xb: n("xb")?,
        xld: n("xld")?,
        xn: n("xn")?,
        wq: n("wq")?,
        xq: n("xq")?,
    })
}

pub fn is_vendor(source: &str) -> bool {
    source.starts_with(tang_compile::tensor::VENDOR_TAG)
}

/// GGUF / Q8x32 to f16. Element `e` of the output is element `base + e` of
/// a `[.., k]` matrix (rows `ld` apart in the source for the activation).
const DEQUANT: &str = r#"
__device__ __forceinline__ float h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ unsigned short f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int* sc, int* m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ __forceinline__ float e8m0(unsigned s) { return __uint_as_float(s ? (s << 23) : 0x00400000u); }
__device__ __forceinline__ float fp8e4m3(unsigned b) {
    const unsigned ex = (b >> 3) & 15u, m = b & 7u;
    const float f = ex ? __uint_as_float(((ex + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (b & 0x80u) ? -f : f;
}
// fmt 0 = Q4_K, 1 = Q5_K, 2 = Q8_0, 3 = Fp8B32, 4 = Mxfp4; one thread per output element.
// k: W's row length, wn: W's element count (DS scales follow the codes)
extern "C" __global__ void main0(const unsigned char* __restrict__ w, unsigned short* __restrict__ out,
                                 unsigned long long base, unsigned long long n, int fmt,
                                 unsigned long long k, unsigned long long wn) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long e = base + i;
    float v;
    if (fmt == 3) {
        const unsigned long long row = e / k, col = e % k;
        v = fp8e4m3(w[e]) * e8m0(w[wn + (row / 32) * (k / 32) + col / 32]);
    } else if (fmt == 4) {
        const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const unsigned q = (w[e / 2] >> (4 * (e & 1))) & 15u;
        v = (q & 8u ? -lut[q & 7u] : lut[q & 7u]) * e8m0(w[wn / 2 + e / 32]);
    } else if (fmt == 2) {
        const unsigned char* b = w + (e / 32) * 34;
        v = h2f(b[0] | (b[1] << 8)) * (float)(signed char)b[2 + (e % 32)];
    } else {
        const unsigned bytes = fmt == 0 ? 144u : 176u;
        const unsigned char* b = w + (e / 256) * bytes;
        const int t = (int)(e % 256), c = t / 64, wi = t % 64, j = 2 * c + wi / 32, l = wi % 32;
        int sc, m; scale_min_k4(j, b + 4, &sc, &m);
        const unsigned char* qs = b + (fmt == 0 ? 16 : 48);
        int q = wi < 32 ? (qs[32 * c + l] & 0xF) : (qs[32 * c + l] >> 4);
        if (fmt == 1) q += ((b[16 + l] >> (2 * c + (wi >= 32))) & 1) << 4;
        v = h2f(b[0] | (b[1] << 8)) * (float)sc * (float)q - h2f(b[2] | (b[3] << 8)) * (float)m;
    }
    out[i] = f2h(v);
}
"#;

/// Q8x32 rows to a dense f16 `[npos, k]`: codes at byte `x`, the
/// `(d, d·Σc)` pairs after `xn` codes.
const DEQUANT_X: &str = r#"
__device__ __forceinline__ unsigned short f2h(float f) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
extern "C" __global__ void main0(const signed char* __restrict__ x, unsigned short* __restrict__ out,
                                 unsigned long long xb, unsigned long long ld, unsigned long long xn,
                                 unsigned long long k, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long e = xb + (i / k) * ld + (i % k);
    const float d = ((const float2*)((const unsigned char*)x + xn))[e / 32].x;
    out[i] = f2h(d * (float)x[e]);
}
"#;

/// One vendor kernel's scratch: f16 W and X and a cuBLASLt workspace.
pub struct Scratch {
    w: CudaSlice<u16>,
    x: CudaSlice<u16>,
    ws: CudaSlice<u8>,
}

impl Scratch {
    pub fn new(stream: &Arc<CudaStream>, s: &Spec) -> Self {
        let ws = std::env::var("TANG_VENDOR_WS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize << 20);
        Self {
            w: stream.alloc_zeros(s.rows * s.k).expect("vendor W scratch"),
            x: stream.alloc_zeros(s.npos * s.k).expect("vendor X scratch"),
            ws: stream.alloc_zeros(ws.max(1)).expect("vendor workspace"),
        }
    }
}

struct Lt {
    handle: sys::cublasLtHandle_t,
    deq: CudaFunction,
    deq_x: CudaFunction,
    algos: HashMap<(usize, usize, usize), sys::cublasLtMatmulAlgo_t>,
}

thread_local! {
    static LT: RefCell<Option<Lt>> = const { RefCell::new(None) };
}

fn fmt_code(w: &str) -> i32 {
    match w {
        "Q4K" => 0,
        "Q5K" => 1,
        "Q8_0" => 2,
        "Fp8B32" => 3,
        "Mxfp4" => 4,
        _ => panic!("vendor GEMM: no dequant for {w}"),
    }
}

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Run vendor GEMM `s` on `stream`: `w` and `x` are its bound buffers,
/// `out` the f32 `[npos, rows]` output.
pub fn run(
    d: &CudaDevice,
    stream: &Arc<CudaStream>,
    s: &Spec,
    sc: &mut Scratch,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    out: &CudaSlice<f32>,
) {
    LT.with(|cell| {
        let mut g = cell.borrow_mut();
        let lt = g.get_or_insert_with(|| Lt {
            handle: lt::create_handle().expect("cublasLt handle"),
            deq: d.function(DEQUANT),
            deq_x: d.function(DEQUANT_X),
            algos: HashMap::new(),
        });
        // dequantize W and X
        let nw = (s.rows * s.k) as u64;
        let mut l = stream.launch_builder(&lt.deq);
        let (base, fmt) = (s.wb as u64, fmt_code(&s.w));
        let (wk, wn) = (s.k as u64, s.wn as u64);
        l.arg(w)
            .arg(&mut sc.w)
            .arg(&base)
            .arg(&nw)
            .arg(&fmt)
            .arg(&wk)
            .arg(&wn);
        unsafe { l.launch(grid(nw as usize)) }.expect("vendor dequant W");
        let nx = (s.npos * s.k) as u64;
        let (xb, ld, xn, k) = (s.xb as u64, s.xld as u64, s.xn as u64, s.k as u64);
        let mut l = stream.launch_builder(&lt.deq_x);
        l.arg(x)
            .arg(&mut sc.x)
            .arg(&xb)
            .arg(&ld)
            .arg(&xn)
            .arg(&k)
            .arg(&nx);
        unsafe { l.launch(grid(nx as usize)) }.expect("vendor dequant X");
        unsafe { gemm(lt, stream, s, sc, out) };
    });
}

/// Column-major: `D (rows × npos) = Wᵀ · X` with W stored `k × rows`
/// (ld k) and X `k × npos` (ld k), D ld rows.
unsafe fn gemm(
    lt: &mut Lt,
    stream: &Arc<CudaStream>,
    s: &Spec,
    sc: &mut Scratch,
    out: &CudaSlice<f32>,
) {
    use sys::cudaDataType_t::{CUDA_R_16F, CUDA_R_32F};
    let (m, n, k) = (s.rows as u64, s.npos as u64, s.k as u64);
    let desc = lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, CUDA_R_32F)
        .expect("matmul desc");
    let ta: i32 = 1; // CUBLAS_OP_T
    lt::set_matmul_desc_attribute(
        desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &ta as *const _ as *const _,
        std::mem::size_of_val(&ta),
    )
    .expect("transa");
    let a = lt::create_matrix_layout(CUDA_R_16F, k, m, k as i64).expect("A layout");
    let b = lt::create_matrix_layout(CUDA_R_16F, k, n, k as i64).expect("B layout");
    let c = lt::create_matrix_layout(CUDA_R_32F, m, n, m as i64).expect("C layout");
    let ws_len = sc.ws.len();
    let algo = *lt.algos.entry((s.rows, s.npos, s.k)).or_insert_with(|| {
        let pref = lt::create_matmul_pref().expect("pref");
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_len as *const _ as *const _,
            std::mem::size_of::<usize>(),
        )
        .expect("pref ws");
        let h = lt::get_matmul_algo_heuristic(lt.handle, desc, a, b, c, c, pref)
            .expect("cublasLt has no algorithm for this GEMM");
        lt::destroy_matmul_pref(pref).ok();
        h.algo
    });
    let (alpha, beta) = (1.0f32, 0.0f32);
    {
        let (wp, _g1) = sc.w.device_ptr(stream);
        let (xp, _g2) = sc.x.device_ptr(stream);
        let (wsp, _g3) = sc.ws.device_ptr_mut(stream);
        let (op, _g4) = out.device_ptr(stream);
        lt::matmul(
            lt.handle,
            desc,
            &alpha as *const f32 as *const _,
            &beta as *const f32 as *const _,
            wp as *const _,
            a,
            xp as *const _,
            b,
            op as *const _,
            c,
            op as *mut _,
            c,
            &algo,
            wsp as *mut _,
            ws_len,
            stream.cu_stream() as _,
        )
        .expect("cublasLtMatmul");
    }
    lt::destroy_matrix_layout(a).ok();
    lt::destroy_matrix_layout(b).ok();
    lt::destroy_matrix_layout(c).ok();
    lt::destroy_matmul_desc(desc).ok();
}
