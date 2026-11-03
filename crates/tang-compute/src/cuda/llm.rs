//! LLM inference on CUDA: the `ComputeDevice` methods tang-llm's forward pass uses, over the
//! kernels in `kernels::llm_cuda`. Same semantics as the Metal backend: activations f32,
//! weights f32, bf16 or MLX 4-bit; attention with sliding windows and bidirectional blocks for
//! any head dim up to 256.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchArgs, LaunchConfig, PushKernelArg};

use super::{CudaBuffer, CudaComputeDevice, CudaStorage, Q4Weight};
use crate::device::ComputeDevice;
use crate::kernels::{kvdec_cuda, llm_cuda};

/// Push a KV cache (f32 or bf16) as a kernel argument.
fn kv_arg<'a>(lb: &mut LaunchArgs<'a>, b: &'a CudaBuffer) {
    match b.storage() {
        CudaStorage::F32(s) => lb.arg(s),
        CudaStorage::Bf16(s) => lb.arg(s),
        CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("a KV cache is f32 or bf16"),
    };
}

/// [`kv_arg`] for a cache the kernel writes.
fn kv_arg_mut<'a>(lb: &mut LaunchArgs<'a>, b: &'a mut CudaBuffer) {
    match b.storage_mut() {
        CudaStorage::F32(s) => lb.arg(s),
        CudaStorage::Bf16(s) => lb.arg(s),
        CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("a KV cache is f32 or bf16"),
    };
}

/// A paged cache's block table (u32 bits, as `upload_u32` stores them) and `log2` of its block
/// size, for the attention kernels (`None`: contiguous).
pub(super) type Paging<'a> = Option<(&'a CudaBuffer, u32)>;

/// The table argument for a kernel: the block table, or any device pointer when contiguous
/// (the kernel never reads it then).
fn table_of<'a>(pages: Paging<'a>, fallback: &'a CudaBuffer) -> (&'a CudaSlice<f32>, u32) {
    match pages {
        Some((t, sh)) => (t.f32_data(), sh),
        None => (fallback.f32_data(), 0),
    }
}

/// One thread per element, 256 per block.
fn per_elem(n: usize) -> LaunchConfig {
    LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn blocks(grid: (usize, usize, usize), threads: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        shared_mem_bytes: 0,
    }
}

/// Dequantized weight rows staged per prefill GEMM (floats): bounds scratch memory.
const DEQUANT_CHUNK: usize = 32 << 20;

/// Batches up to this many rows run the small-batch GEMM straight off the packed weights
/// (`gemm_small_*`) instead of dequantizing for cuBLAS; up to 8 rows use the GEMV.
const SMALL_GEMM_ROWS: usize = 32;
/// Warps a block of `gemv_q4_xs*` (the kernel source's `Q4_WARPS`).
const Q4_WARPS: usize = 8;

/// Forwards of up to this many queries use split-KV attention (`attn_multi_*`) instead of the
/// tiled prefill kernel, which launches too few blocks to stream a long cache quickly.
const SMALL_ATTN_ROWS: usize = 32;

/// K split for `gemv_q4_xs4_m*`: `(splits, words a split)` so `col_blocks` column blocks
/// launch about 4 blocks an SM (a 4096-wide output is 128 column blocks, one an SM on an H200,
/// too few loads in flight to cover HBM latency). Splits are whole 128-word stages. It depends
/// on the shape alone, never on the row count, so a row's sum order is batch-invariant.
fn gemv_q4_splits(col_blocks: usize, words: usize) -> (usize, usize) {
    const XW: usize = 128;
    let stages = words.div_ceil(XW);
    let want = (4 * sm_count()).div_ceil(col_blocks).clamp(1, stages);
    let wsplit = stages.div_ceil(want) * XW;
    (words.div_ceil(wsplit), wsplit)
}

/// Split-KV decode: how many splits a query seeing `span` keys uses. A split per 64 keys, at
/// most 128 (`ATTN_SPLIT_KEYS` / `ATTN_MAX_SPLITS` in the kernel source, which must agree: a
/// row of a batched decode splits in-kernel the way a decode alone does here). It depends on
/// the span only, so a row's sums never depend on what it's batched with.
fn decode_splits(span: usize) -> usize {
    span.div_ceil(64).clamp(1, 128)
}

/// Streaming multiprocessors on the current device (for sizing split-KV / split-K grids).
pub(super) fn sm_count() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        use cudarc::driver::sys;
        let mut n: i32 = 0;
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut n,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            );
        }
        if n > 0 {
            n as usize
        } else {
            80
        }
    })
}

impl CudaComputeDevice {
    /// Kernel `name` from `source` (one of `llm_cuda`'s), compiled once.
    pub(super) fn llm_func(&self, source: &str, name: &'static str) -> CudaFunction {
        if let Some(f) = self.llm_funcs.borrow().get(name) {
            return f.clone();
        }
        let (_module, f) = self.get_func(source, name);
        self.llm_funcs.borrow_mut().insert(name, f.clone());
        f
    }

    /// Kernel `name` from a KV-reading source (`llm_cuda::kv_source`), for f32 or bf16 caches.
    fn kv_func(&self, source: &'static str, name: &'static str, bf16: bool) -> CudaFunction {
        if !bf16 {
            return self.llm_func(llm_cuda::kv_source(source, false), name);
        }
        let key: &'static str = {
            use std::sync::Mutex;
            static KEYS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
            let want = format!("{name}/bf16");
            let mut keys = KEYS.lock().unwrap();
            match keys.iter().find(|k| **k == want) {
                Some(k) => k,
                None => {
                    let k: &'static str = Box::leak(want.into_boxed_str());
                    keys.push(k);
                    k
                }
            }
        };
        if let Some(f) = self.llm_funcs.borrow().get(key) {
            return f.clone();
        }
        let (_module, f) = self.get_func(llm_cuda::kv_source(source, true), name);
        self.llm_funcs.borrow_mut().insert(key, f.clone());
        f
    }

    /// Whether K and V caches are bf16 (they must agree).
    fn kv_bf16(k: &CudaBuffer, v: &CudaBuffer) -> bool {
        assert_eq!(k.is_bf16(), v.is_bf16(), "K and V caches differ in format");
        k.is_bf16()
    }

    /// `buf` as f32 on device: itself, or a widened copy of bf16 activations.
    fn as_f32<'a>(&self, buf: &'a CudaBuffer, tmp: &'a mut Option<CudaBuffer>) -> &'a CudaBuffer {
        if buf.is_bf16() {
            tmp.insert(self.convert_bf16_to_f32(buf))
        } else {
            buf
        }
    }

    /// Back to the device's activation precision.
    fn finish(&self, out: CudaBuffer) -> CudaBuffer {
        if self.mixed_precision {
            self.convert_f32_to_bf16(&out)
        } else {
            out
        }
    }

    pub(super) fn upload_q4_impl(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        group: usize,
    ) -> CudaBuffer {
        assert!(group % 8 == 0, "4-bit groups must be a multiple of 8");
        let q = Q4Weight {
            packed: self.stream.memcpy_stod(packed).unwrap(),
            scales: self.stream.memcpy_stod(scales).unwrap(),
            biases: self.stream.memcpy_stod(biases).unwrap(),
            group,
        };
        Self::make_buf_unpooled(CudaStorage::Q4(Box::new(q)), packed.len() * 8)
    }

    pub(super) fn upload_bf16_impl(&self, bits: &[u16]) -> CudaBuffer {
        let slice = self.stream.memcpy_stod(bits).unwrap();
        Self::make_buf_unpooled(CudaStorage::Bf16(slice), bits.len())
    }

    /// `embedding` from bf16 or 4-bit weights into f32 rows. None for other weights.
    pub(super) fn embedding_llm(
        &self,
        weight: &CudaBuffer,
        ids: &CudaBuffer,
        seq_len: usize,
        dim: usize,
    ) -> Option<CudaBuffer> {
        let total = seq_len * dim;
        let (s, d) = (seq_len as u32, dim as u32);
        let mut out = self.pool_alloc_uninit_f32(total);
        match weight.storage() {
            CudaStorage::Q4(q) => {
                let f = self.llm_func(llm_cuda::EMBED_CUDA, "embedding_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(ids.f32_data())
                        .arg(out.f32_data_mut())
                        .arg(&s)
                        .arg(&d)
                        .arg(&group)
                        .launch(per_elem(total))
                        .unwrap();
                }
            }
            // bf16 activations (mixed precision) keep the training path.
            CudaStorage::Bf16(w) if !self.mixed_precision => {
                let f = self.llm_func(llm_cuda::EMBED_CUDA, "embedding_bf16");
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(w)
                        .arg(ids.f32_data())
                        .arg(out.f32_data_mut())
                        .arg(&s)
                        .arg(&d)
                        .launch(per_elem(total))
                        .unwrap();
                }
            }
            _ => return None,
        }
        Some(self.finish(out))
    }

    /// `linear` over bf16 or 4-bit weights with f32 activations. None for other combinations.
    /// Decode (`m <= 8`) is a GEMV straight off the packed weights; larger batches widen the
    /// weights to f32 a chunk of rows at a time and run cuBLAS SGEMM into the output columns.
    pub(super) fn linear_llm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<CudaBuffer> {
        let q4 = matches!(w.storage(), CudaStorage::Q4(_));
        if !q4 && !(w.is_bf16() && !x.is_bf16()) {
            return None;
        }
        let mut tmp = None;
        let x = self.as_f32(x, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(m * n);
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        // Column-per-thread GEMM from 4 rows (4-bit) or 9 rows (bf16, whose GEMV stays
        // memory-bound up to 8); the GEMV below is faster for fewer.
        let cols_ok = match w.storage() {
            CudaStorage::Q4(q) => m >= 4 && q.group % 64 == 0 && k % 128 == 0,
            _ => m >= 9 && k % 128 == 0,
        };
        // 4-bit, up to 8 rows: `gemv_q4_x{v,s}4_m{m}`, the same sums whatever m is.
        if q4 && m <= 8 && k % 8 == 0 {
            const NAMES: [&str; 8] = [
                "gemv_q4_xs4_m1",
                "gemv_q4_xs4_m2",
                "gemv_q4_xs4_m3",
                "gemv_q4_xs4_m4",
                "gemv_q4_xs4_m5",
                "gemv_q4_xs4_m6",
                "gemv_q4_xs4_m7",
                "gemv_q4_xs4_m8",
            ];
            const XV_NAMES: [&str; 8] = [
                "gemv_q4_xv4_m1",
                "gemv_q4_xv4_m2",
                "gemv_q4_xv4_m3",
                "gemv_q4_xv4_m4",
                "gemv_q4_xv4_m5",
                "gemv_q4_xv4_m6",
                "gemv_q4_xv4_m7",
                "gemv_q4_xv4_m8",
            ];
            if let CudaStorage::Q4(q) = w.storage() {
                // 128-bit weight loads when the groups allow (every MLX model). The pick
                // depends on the shape alone, never m.
                let names = if k % 32 == 0 && q.group % 32 == 0 {
                    &XV_NAMES
                } else {
                    &NAMES
                };
                let f = self.llm_func(llm_cuda::GEMV_CUDA, names[m - 1]);
                let group = q.group as u32;
                let col_blocks = n.div_ceil(Q4_WARPS * 4);
                let (splits, wsplit) = gemv_q4_splits(col_blocks, k / 8);
                let mut partial = (splits > 1).then(|| self.pool_alloc_uninit_f32(splits * m * n));
                let ws = wsplit as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(match partial.as_mut() {
                            Some(p) => p.f32_data_mut(),
                            None => out.f32_data_mut(),
                        })
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&group)
                        .arg(&ws)
                        .launch(blocks((col_blocks, splits, 1), Q4_WARPS as u32 * 32))
                        .unwrap();
                }
                if let Some(p) = partial {
                    let f = self.llm_func(llm_cuda::GEMV_CUDA, "sum_splits");
                    let (total, s) = ((m * n) as u32, splits as u32);
                    unsafe {
                        self.stream
                            .launch_builder(&f)
                            .arg(p.f32_data())
                            .arg(out.f32_data_mut())
                            .arg(&total)
                            .arg(&s)
                            .launch(per_elem(m * n))
                            .unwrap();
                    }
                }
                return Some(self.finish(out));
            }
        }
        if (9..=SMALL_GEMM_ROWS).contains(&m) && k % 32 == 0 && self.tc_q4_ok(w) {
            self.gemm_tc_q4(x, w, &mut out, m, k, n);
            return Some(self.finish(out));
        }
        if m <= SMALL_GEMM_ROWS && cols_ok {
            self.gemm_cols(x, w, &mut out, m, k, n);
            return Some(self.finish(out));
        }
        if m <= 8 && k % 8 == 0 {
            let cfg = blocks((n.div_ceil(8), 1, 1), 256);
            match w.storage() {
                CudaStorage::Q4(q) => {
                    let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemv_q4");
                    let group = q.group as u32;
                    unsafe {
                        self.stream
                            .launch_builder(&f)
                            .arg(x.f32_data())
                            .arg(&q.packed)
                            .arg(&q.scales)
                            .arg(&q.biases)
                            .arg(out.f32_data_mut())
                            .arg(&mu)
                            .arg(&ku)
                            .arg(&nu)
                            .arg(&group)
                            .launch(cfg)
                            .unwrap();
                    }
                }
                _ => {
                    let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemv_bf16");
                    unsafe {
                        self.stream
                            .launch_builder(&f)
                            .arg(x.f32_data())
                            .arg(w.bf16_data())
                            .arg(out.f32_data_mut())
                            .arg(&mu)
                            .arg(&ku)
                            .arg(&nu)
                            .launch(cfg)
                            .unwrap();
                    }
                }
            }
            return Some(self.finish(out));
        }

        if m <= SMALL_GEMM_ROWS && k % 64 == 0 {
            self.gemm_small(x, w, &mut out, m, k, n);
            return Some(self.finish(out));
        }

        let chunk = (DEQUANT_CHUNK / k).clamp(1, n);
        let mut scratch = self.pool_alloc_uninit_f32(chunk * k);
        let mut row0 = 0;
        while row0 < n {
            let rows = chunk.min(n - row0);
            self.dequant_rows(w, scratch.f32_data_mut(), row0, rows, k);
            // Column-major view: out^T[row0.., :] = W_chunk (rows x k) · x^T (k x m).
            let mut dst = out.f32_data_mut().slice_mut(row0..);
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_T,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: rows as i32,
                            n: m as i32,
                            k: k as i32,
                            alpha: 1.0f32,
                            lda: k as i32,
                            ldb: k as i32,
                            beta: 0.0f32,
                            ldc: n as i32,
                        },
                        scratch.f32_data(),
                        x.f32_data(),
                        &mut dst,
                    )
                    .expect("cuBLAS sgemm (dequantized weights) failed");
            }
            row0 += rows;
        }
        Some(self.finish(out))
    }

    /// Whether `gemm_tc_q4` can take `w`: 4-bit with groups of a multiple of 32, on sm_80+
    /// (TF32 `mma.sync`), unless `TANG_TC_GEMM=0`.
    fn tc_q4_ok(&self, w: &CudaBuffer) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| {
            use cudarc::driver::sys;
            let mut major: i32 = 0;
            unsafe {
                sys::cuDeviceGetAttribute(
                    &mut major,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    0,
                );
            }
            major >= 8 && std::env::var("TANG_TC_GEMM").map_or(true, |v| v != "0")
        });
        on && matches!(w.storage(), CudaStorage::Q4(q) if q.group % 32 == 0)
    }

    /// `out = x · wᵀ` for 9..=32 rows on tensor cores (`gemm_tc_q4_m{16,32}`), K split (by
    /// shape alone, partials summed by `sum_splits`) so the column blocks fill the GPU.
    fn gemm_tc_q4(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let CudaStorage::Q4(q) = w.storage() else {
            unreachable!()
        };
        let name = if m <= 16 {
            "gemm_tc_q4_m16"
        } else {
            "gemm_tc_q4_m32"
        };
        let f = match self.llm_funcs.borrow().get(name) {
            Some(f) => Some(f.clone()),
            None => None,
        };
        let f = f.unwrap_or_else(|| {
            let (_m, f) = self.get_func_with_arch(llm_cuda::TC_GEMM_CUDA, name, "sm_80");
            self.llm_funcs.borrow_mut().insert(name, f.clone());
            f
        });
        let col_blocks = n.div_ceil(128);
        let stages = k / 32;
        let want = (4 * sm_count()).div_ceil(col_blocks).clamp(1, stages);
        let k_split = stages.div_ceil(want) * 32;
        let splits = k.div_ceil(k_split);
        let mut partial = (splits > 1).then(|| self.pool_alloc_uninit_f32(splits * m * n));
        let (mu, ku, nu, gu, ks) = (m as u32, k as u32, n as u32, q.group as u32, k_split as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(&q.packed)
                .arg(&q.scales)
                .arg(&q.biases)
                .arg(match partial.as_mut() {
                    Some(p) => p.f32_data_mut(),
                    None => out.f32_data_mut(),
                })
                .arg(&mu)
                .arg(&ku)
                .arg(&nu)
                .arg(&gu)
                .arg(&ks)
                .launch(blocks((col_blocks, splits, 1), 256))
                .unwrap();
        }
        if let Some(p) = partial {
            let f = self.llm_func(llm_cuda::GEMV_CUDA, "sum_splits");
            let (total, s) = ((m * n) as u32, splits as u32);
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(p.f32_data())
                    .arg(out.f32_data_mut())
                    .arg(&total)
                    .arg(&s)
                    .launch(per_elem(m * n))
                    .unwrap();
            }
        }
    }

    /// `out = x · wᵀ` for 2..=32 rows with `gemm_cols{8,16,32}_*`: a thread per output
    /// column, K split (partials summed by `sum_splits`) when the columns can't fill the GPU.
    fn gemm_cols(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let col_blocks = n.div_ceil(256);
        let kc = 128;
        let chunks = k / kc;
        // Aim for three blocks per SM; each split keeps at least 2 chunks of K.
        let want = (3 * sm_count()).div_ceil(col_blocks);
        let per = chunks.div_ceil(want.clamp(1, chunks.div_ceil(2).max(1)));
        let splits = chunks.div_ceil(per);
        let k_split = (per * kc) as u32;
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        let cfg = blocks((col_blocks, splits, 1), 256);
        let mr = match m {
            ..=8 => "gemm_cols8",
            ..=16 => "gemm_cols16",
            _ => "gemm_cols32",
        };
        let mut partial = (splits > 1).then(|| self.pool_alloc_uninit_f32(splits * m * n));
        let dst = match partial.as_mut() {
            Some(p) => p.f32_data_mut(),
            None => out.f32_data_mut(),
        };
        match w.storage() {
            CudaStorage::Q4(q) => {
                let name = match mr {
                    "gemm_cols8" => "gemm_cols8_q4",
                    "gemm_cols16" => "gemm_cols16_q4",
                    _ => "gemm_cols32_q4",
                };
                let f = self.llm_func(llm_cuda::GEMV_CUDA, name);
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(dst)
                        .arg(&mu)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&group)
                        .arg(&k_split)
                        .launch(cfg)
                        .unwrap();
                }
            }
            _ => {
                let name = match mr {
                    "gemm_cols8" => "gemm_cols8_bf16",
                    "gemm_cols16" => "gemm_cols16_bf16",
                    _ => "gemm_cols32_bf16",
                };
                let f = self.llm_func(llm_cuda::GEMV_CUDA, name);
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(w.bf16_data())
                        .arg(dst)
                        .arg(&mu)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&k_split)
                        .launch(cfg)
                        .unwrap();
                }
            }
        }
        if let Some(p) = partial {
            let f = self.llm_func(llm_cuda::GEMV_CUDA, "sum_splits");
            let (total, s) = ((m * n) as u32, splits as u32);
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(p.f32_data())
                    .arg(out.f32_data_mut())
                    .arg(&total)
                    .arg(&s)
                    .launch(per_elem(m * n))
                    .unwrap();
            }
        }
    }

    /// `out = x · wᵀ` for `m <= 32` rows with `gemm_small_*`: 128 output columns per block,
    /// and a K split (partials summed by `sum_splits`) when that alone can't fill the GPU.
    fn gemm_small(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let col_blocks = n.div_ceil(128);
        let chunks = k / 64;
        // Aim for two blocks per SM; each split keeps at least 4 chunks of K.
        let want = (2 * sm_count()).div_ceil(col_blocks);
        let per = chunks.div_ceil(want.clamp(1, chunks.div_ceil(4).max(1)));
        let splits = chunks.div_ceil(per);
        let k_split = (per * 64) as u32;
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        let cfg = blocks((col_blocks, splits, 1), 256);
        let mut partial = (splits > 1).then(|| self.pool_alloc_uninit_f32(splits * m * n));
        let dst = match partial.as_mut() {
            Some(p) => p.f32_data_mut(),
            None => out.f32_data_mut(),
        };
        match w.storage() {
            CudaStorage::Q4(q) => {
                let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemm_small_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(dst)
                        .arg(&mu)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&group)
                        .arg(&k_split)
                        .launch(cfg)
                        .unwrap();
                }
            }
            _ => {
                let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemm_small_bf16");
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(x.f32_data())
                        .arg(w.bf16_data())
                        .arg(dst)
                        .arg(&mu)
                        .arg(&ku)
                        .arg(&nu)
                        .arg(&k_split)
                        .launch(cfg)
                        .unwrap();
                }
            }
        }
        if let Some(p) = partial {
            let f = self.llm_func(llm_cuda::GEMV_CUDA, "sum_splits");
            let (total, s) = ((m * n) as u32, splits as u32);
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(p.f32_data())
                    .arg(out.f32_data_mut())
                    .arg(&total)
                    .arg(&s)
                    .launch(per_elem(m * n))
                    .unwrap();
            }
        }
    }

    /// Widen weight rows `row0 .. row0 + rows` (bf16 or 4-bit) into `dst` as `[rows, k]` f32.
    fn dequant_rows(
        &self,
        w: &CudaBuffer,
        dst: &mut CudaSlice<f32>,
        row0: usize,
        rows: usize,
        k: usize,
    ) {
        let (r0, r, ku) = (row0 as u32, rows as u32, k as u32);
        match w.storage() {
            CudaStorage::Q4(q) => {
                let f = self.llm_func(llm_cuda::DEQUANT_CUDA, "dequant_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(dst)
                        .arg(&r0)
                        .arg(&r)
                        .arg(&ku)
                        .arg(&group)
                        .launch(per_elem(rows * k / 8))
                        .unwrap();
                }
            }
            CudaStorage::Bf16(s) => {
                let f = self.llm_func(llm_cuda::DEQUANT_CUDA, "dequant_bf16");
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(s)
                        .arg(dst)
                        .arg(&r0)
                        .arg(&r)
                        .arg(&ku)
                        .launch(per_elem(rows * k))
                        .unwrap();
                }
            }
            CudaStorage::F32(_) | CudaStorage::Q2(_) => unreachable!(),
        }
    }

    /// Fused attention prologue (see `ComputeDevice::attention_prep`). None when the shape or
    /// precision isn't covered (bf16 activations, odd or > 256 head dims).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prep_llm(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_cache: &mut CudaBuffer,
        v_cache: &mut CudaBuffer,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
        pages: Paging,
    ) -> Option<CudaBuffer> {
        if qkv.is_bf16() || hd % 2 != 0 || hd > 256 {
            return None;
        }
        let bf16 = Self::kv_bf16(k_cache, v_cache);
        let kvd = nkv * hd;
        assert!(
            pages.is_some()
                || (k_cache.len >= (pos + seq) * kvd && v_cache.len >= (pos + seq) * kvd),
            "KV cache too small"
        );
        let (table, sh) = table_of(pages, qkv);
        let mut q = self.pool_alloc_uninit_f32(seq * nh * hd);
        let f = self.kv_func(llm_cuda::ATTN_PREP_CUDA, "attention_prep", bf16);
        let qn = q_norm.unwrap_or(qkv);
        let kn = k_norm.unwrap_or(qkv);
        let (nh_u, nkv_u, hd_u, pos_u) = (nh as u32, nkv as u32, hd as u32, pos as u32);
        let (has_qn, has_kn) = (q_norm.is_some() as u32, k_norm.is_some() as u32);
        let mut lb = self.stream.launch_builder(&f);
        lb.arg(qkv.f32_data())
            .arg(qn.f32_data())
            .arg(kn.f32_data())
            .arg(cos.f32_data())
            .arg(sin.f32_data())
            .arg(q.f32_data_mut());
        kv_arg_mut(&mut lb, k_cache);
        kv_arg_mut(&mut lb, v_cache);
        unsafe {
            lb.arg(&nh_u)
                .arg(&nkv_u)
                .arg(&hd_u)
                .arg(&pos_u)
                .arg(&eps)
                .arg(&has_qn)
                .arg(&has_kn)
                .arg(table)
                .arg(&sh)
                .launch(blocks((nh + 2 * nkv, seq, 1), 32))
                .unwrap();
        }
        Some(q)
    }

    /// Attention over a KV cache (see `ComputeDevice::kv_attention_window`): split-KV for
    /// decode, tiled for batches. `bidir`: every query sees the whole batch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_llm(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        cache_start: usize,
        q_len: usize,
        (nh, nkv, d): (usize, usize, usize),
        window: usize,
        bidir: bool,
        pages: Paging,
    ) -> CudaBuffer {
        assert!(
            d <= 256 && nh % nkv == 0,
            "attention needs head_dim at most 256 and n_heads a multiple of n_kv_heads"
        );
        // K and V are read as stored (f32 or bf16); q is widened if it's bf16.
        let mut tq = None;
        let q = self.as_f32(q, &mut tq);
        let bf16 = Self::kv_bf16(k, v);
        let longest = cache_start + q_len;
        assert!(pages.is_some() || (k.len >= longest * nkv * d && v.len >= longest * nkv * d));
        let mut out = self.pool_alloc_uninit_f32(q_len * nh * d);
        let (table, sh) = table_of(pages, q);
        let (cs, ql, nh_u, nkv_u, d_u, win, bi) = (
            cache_start as u32,
            q_len as u32,
            nh as u32,
            nkv as u32,
            d as u32,
            window as u32,
            bidir as u32,
        );

        if q_len > 1 && q_len <= SMALL_ATTN_ROWS && d % 4 == 0 {
            self.attention_multi(
                q,
                k,
                v,
                &mut out,
                cache_start,
                q_len,
                (nh, nkv, d),
                window,
                bidir,
                bf16,
                pages,
            );
            return self.finish(out);
        }
        if q_len > 1 {
            let bq: u32 = if d <= 128 { 32 } else { 16 };
            let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_prefill", bf16);
            let mut lb = self.stream.launch_builder(&f);
            lb.arg(q.f32_data());
            kv_arg(&mut lb, k);
            kv_arg(&mut lb, v);
            unsafe {
                lb.arg(out.f32_data_mut())
                    .arg(&cs)
                    .arg(&ql)
                    .arg(&nh_u)
                    .arg(&nkv_u)
                    .arg(&d_u)
                    .arg(&win)
                    .arg(&bi)
                    .arg(&bq)
                    .arg(table)
                    .arg(&sh)
                    .launch(blocks((q_len.div_ceil(bq as usize), nh, 1), 256))
                    .unwrap();
            }
            return self.finish(out);
        }

        // Decode: split the keys a window can see across blocks, then combine.
        let base = if window > 0 {
            longest.saturating_sub(window)
        } else {
            0
        };
        let span = longest - base;
        let n_splits = decode_splits(span);
        let split_len = span.div_ceil(n_splits);
        let mut partial = self.pool_alloc_uninit_f32(nh * n_splits * (d + 2));
        let (ns, sl, base_u) = (n_splits as u32, split_len as u32, base as u32);
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_partial", bf16);
        let mut lb = self.stream.launch_builder(&f);
        lb.arg(q.f32_data());
        kv_arg(&mut lb, k);
        kv_arg(&mut lb, v);
        unsafe {
            lb.arg(partial.f32_data_mut())
                .arg(&cs)
                .arg(&ql)
                .arg(&nh_u)
                .arg(&nkv_u)
                .arg(&d_u)
                .arg(&ns)
                .arg(&sl)
                .arg(&win)
                .arg(&base_u)
                .arg(&bi)
                .arg(table)
                .arg(&sh)
                .launch(blocks((nh, n_splits, 1), 256))
                .unwrap();
        }
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_combine", false);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(partial.f32_data())
                .arg(out.f32_data_mut())
                .arg(&nh_u)
                .arg(&d_u)
                .arg(&ns)
                .launch(blocks((nh, 1, 1), d.next_multiple_of(32).min(256) as u32))
                .unwrap();
        }
        self.finish(out)
    }

    /// Batched decode attention (see `ComputeDevice::attention_paged_rows`): one launch each
    /// of `attention_prep_rows`, `attn_partial_rows` and `attn_combine` for every row. None
    /// when the shape or precision isn't covered.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_rows_llm(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_pool: &mut CudaBuffer,
        v_pool: &mut CudaBuffer,
        rows: &crate::PagedRows<CudaBuffer>,
        (nh, nkv, hd): (usize, usize, usize),
        eps: f32,
        query_scale: f32,
    ) -> Option<CudaBuffer> {
        let n = rows.pos.len();
        let Some(first) = rows.pages.first() else {
            return None;
        };
        if qkv.is_bf16() || hd % 2 != 0 || hd > 256 || nh % nkv != 0 {
            return None;
        }
        let sh = first.shift();
        let bf16 = Self::kv_bf16(k_pool, v_pool);
        let mut q = self.pool_alloc_uninit_f32(n * nh * hd);
        let f = self.kv_func(llm_cuda::ATTN_PREP_CUDA, "attention_prep_rows", bf16);
        let qn = q_norm.unwrap_or(qkv);
        let kn = k_norm.unwrap_or(qkv);
        let (nh_u, nkv_u, hd_u) = (nh as u32, nkv as u32, hd as u32);
        let (has_qn, has_kn) = (q_norm.is_some() as u32, k_norm.is_some() as u32);
        let mut lb = self.stream.launch_builder(&f);
        lb.arg(qkv.f32_data())
            .arg(qn.f32_data())
            .arg(kn.f32_data())
            .arg(cos.f32_data())
            .arg(sin.f32_data())
            .arg(q.f32_data_mut());
        kv_arg_mut(&mut lb, k_pool);
        kv_arg_mut(&mut lb, v_pool);
        unsafe {
            lb.arg(&nh_u)
                .arg(&nkv_u)
                .arg(&hd_u)
                .arg(&eps)
                .arg(&has_qn)
                .arg(&has_kn)
                .arg(rows.tables.f32_data())
                .arg(&sh)
                .arg(rows.positions.f32_data())
                .arg(rows.offsets.f32_data())
                .launch(blocks((nh + 2 * nkv, n, 1), 32))
                .unwrap();
        }
        if query_scale != 1.0 {
            crate::ComputeDevice::scale_buffer(self, &mut q, query_scale);
        }
        // As many splits as the longest row's decode would use; each row uses its own.
        let span = rows.pos.iter().max().map_or(1, |p| p + 1);
        let n_splits = decode_splits(span);
        let mut partial = self.pool_alloc_uninit_f32(n * nh * n_splits * (hd + 2));
        let ns = n_splits as u32;
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_partial_rows", bf16);
        let mut lb = self.stream.launch_builder(&f);
        lb.arg(q.f32_data());
        kv_arg(&mut lb, k_pool);
        kv_arg(&mut lb, v_pool);
        unsafe {
            lb.arg(partial.f32_data_mut())
                .arg(&nh_u)
                .arg(&nkv_u)
                .arg(&hd_u)
                .arg(&ns)
                .arg(rows.tables.f32_data())
                .arg(&sh)
                .arg(rows.positions.f32_data())
                .arg(rows.offsets.f32_data())
                .launch(blocks((nh, n_splits, n), 256))
                .unwrap();
        }
        let mut out = self.pool_alloc_uninit_f32(n * nh * hd);
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_combine", false);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(partial.f32_data())
                .arg(out.f32_data_mut())
                .arg(&nh_u)
                .arg(&hd_u)
                .arg(&ns)
                .launch(blocks((nh, n, 1), hd.next_multiple_of(32).min(256) as u32))
                .unwrap();
        }
        Some(self.finish(out))
    }

    /// Split-KV attention for a few queries (`attn_multi_*` then `attn_combine`): blocks over
    /// (KV head, key split, group of (query, head) rows), each reading its split's keys once
    /// for all the rows.
    #[allow(clippy::too_many_arguments)]
    fn attention_multi(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        out: &mut CudaBuffer,
        cache_start: usize,
        q_len: usize,
        (nh, nkv, d): (usize, usize, usize),
        window: usize,
        bidir: bool,
        bf16: bool,
        pages: Paging,
    ) {
        let gqa_rows = q_len * (nh / nkv);
        let (name, rows, tile) = match (d <= 128, gqa_rows) {
            (false, _) => ("attn_multi_d256", 16, 16),
            (true, ..=8) => ("attn_multi_r8", 8, 32),
            (true, ..=16) => ("attn_multi_r16", 16, 32),
            _ => ("attn_multi_r32", 32, 32),
        };
        let longest = cache_start + q_len;
        // The earliest key any query can see.
        let base = if window > 0 && !bidir {
            (cache_start + 1).saturating_sub(window)
        } else {
            0
        };
        let span = longest - base;
        let groups = gqa_rows.div_ceil(rows);
        // Enough blocks for a few per SM, splits a whole number of key tiles.
        let want = (4 * sm_count()).div_ceil(nkv * groups).max(1);
        let split_len = span.div_ceil(want).max(2 * tile).next_multiple_of(tile);
        let n_splits = span.div_ceil(split_len);
        let mut partial = self.pool_alloc_uninit_f32(q_len * nh * n_splits * (d + 2));
        let (cs, ql, nh_u, nkv_u, d_u, win, bi) = (
            cache_start as u32,
            q_len as u32,
            nh as u32,
            nkv as u32,
            d as u32,
            window as u32,
            bidir as u32,
        );
        let (ns, sl, base_u) = (n_splits as u32, split_len as u32, base as u32);
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, name, bf16);
        let (table, sh) = table_of(pages, q);
        let mut lb = self.stream.launch_builder(&f);
        lb.arg(q.f32_data());
        kv_arg(&mut lb, k);
        kv_arg(&mut lb, v);
        unsafe {
            lb.arg(partial.f32_data_mut())
                .arg(&cs)
                .arg(&ql)
                .arg(&nh_u)
                .arg(&nkv_u)
                .arg(&d_u)
                .arg(&ns)
                .arg(&sl)
                .arg(&win)
                .arg(&base_u)
                .arg(&bi)
                .arg(table)
                .arg(&sh)
                .launch(blocks((nkv, n_splits, groups), 256))
                .unwrap();
        }
        let f = self.kv_func(llm_cuda::ATTENTION_CUDA, "attn_combine", false);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(partial.f32_data())
                .arg(out.f32_data_mut())
                .arg(&nh_u)
                .arg(&d_u)
                .arg(&ns)
                .launch(blocks(
                    (nh, q_len, 1),
                    d.next_multiple_of(32).min(256) as u32,
                ))
                .unwrap();
        }
    }

    /// Elementwise kernels from `FUSED_CUDA` taking `(in, out, a[, b])` with f32 buffers.
    fn fused_unary(
        &self,
        name: &'static str,
        x: &CudaBuffer,
        n_out: usize,
        a: u32,
        b: Option<u32>,
    ) -> CudaBuffer {
        let mut tmp = None;
        let x = self.as_f32(x, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(n_out);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, name);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(x.f32_data()).arg(out.f32_data_mut()).arg(&a);
            if let Some(b) = b.as_ref() {
                l.arg(b);
            }
            l.launch(per_elem(n_out)).unwrap();
        }
        self.finish(out)
    }

    pub(super) fn gelu_tanh_llm(&self, x: &CudaBuffer, n: usize) -> CudaBuffer {
        self.fused_unary("gelu_tanh", x, n, n as u32, None)
    }

    pub(super) fn gelu_erf_llm(&self, x: &CudaBuffer, n: usize) -> CudaBuffer {
        self.fused_unary("gelu_erf", x, n, n as u32, None)
    }

    pub(super) fn gated_split_llm(
        &self,
        name: &'static str,
        gu: &CudaBuffer,
        rows: usize,
        ff: usize,
    ) -> CudaBuffer {
        self.fused_unary(name, gu, rows * ff, rows as u32, Some(ff as u32))
    }

    /// `a + b` over f32 buffers. None for bf16 inputs.
    /// `ComputeDevice::top_k_rows` on the device (`top_k_rows` kernel) for k <= 64 and f32
    /// rows; None otherwise. Downloads `rows * 2k` floats instead of the rows.
    pub(super) fn top_k_rows_llm(
        &self,
        x: &CudaBuffer,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Option<(Vec<f32>, Vec<u32>)> {
        if k == 0 || k > 64 || k > n || !matches!(x.storage(), CudaStorage::F32(_)) {
            return None;
        }
        let mut out = self.pool_alloc_uninit_f32(rows * 2 * k);
        let f = self.llm_func(llm_cuda::TOPK_CUDA, "top_k_rows");
        let (nu, ku) = (n as u32, k as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(out.f32_data_mut())
                .arg(&nu)
                .arg(&ku)
                .launch(blocks((rows, 1, 1), 1024))
                .unwrap();
        }
        let v = crate::ComputeDevice::download(self, &out);
        let (mut vals, mut idx) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
        for r in v.chunks(2 * k) {
            vals.extend_from_slice(&r[..k]);
            idx.extend(r[k..].iter().map(|f| f.to_bits()));
        }
        Some((vals, idx))
    }

    pub(super) fn add_llm(&self, a: &CudaBuffer, b: &CudaBuffer, n: usize) -> Option<CudaBuffer> {
        if a.is_bf16() || b.is_bf16() {
            return None;
        }
        let mut out = self.pool_alloc_uninit_f32(n);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "add_f32");
        let n_u = n as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(a.f32_data())
                .arg(b.f32_data())
                .arg(out.f32_data_mut())
                .arg(&n_u)
                .launch(per_elem(n))
                .unwrap();
        }
        Some(out)
    }

    pub(super) fn layer_norm_llm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        b: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        let (mut tx, mut tw, mut tb) = (None, None, None);
        let (x, w, b) = (
            self.as_f32(x, &mut tx),
            self.as_f32(w, &mut tw),
            self.as_f32(b, &mut tb),
        );
        let mut out = self.pool_alloc_uninit_f32(rows * dim);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "layer_norm");
        let dim_u = dim as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(w.f32_data())
                .arg(b.f32_data())
                .arg(out.f32_data_mut())
                .arg(&dim_u)
                .arg(&eps)
                .launch(blocks((rows, 1, 1), 256))
                .unwrap();
        }
        self.finish(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_half_llm(
        &self,
        input: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let mut tmp = None;
        let x = self.as_f32(input, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(x.len);
        // Heads past `seq_len * n_heads` (if any) pass through unrotated, as on the host.
        if out.len > seq_len * n_heads * head_dim {
            self.stream
                .memcpy_dtod(x.f32_data(), out.f32_data_mut())
                .unwrap();
        }
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "rope_half");
        let (s, h, d, p) = (
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            start_pos as u32,
        );
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(cos.f32_data())
                .arg(sin.f32_data())
                .arg(out.f32_data_mut())
                .arg(&s)
                .arg(&h)
                .arg(&d)
                .arg(&p)
                .launch(per_elem(seq_len * n_heads * head_dim / 2))
                .unwrap();
        }
        self.finish(out)
    }

    /// Compressed-KV decode (see `ComputeDevice::tkv_decode`): one thread per lane,
    /// tables from `cram_codec::kvgpu::Batch`. The out buffer is f32 or bf16 (a cache
    /// the caller allocated with `alloc` / `alloc_bf16`); the kernel is compiled per
    /// kind. The u32 tables ride as f32 bit patterns (see `upload_u32`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tkv_decode(
        &self,
        stream: &CudaBuffer,
        lanes: &CudaBuffer,
        chans: &CudaBuffer,
        prior: &CudaBuffer,
        scratch: &CudaBuffer,
        out: &CudaBuffer,
        rope: Option<(&CudaBuffer, &CudaBuffer)>,
        n_lanes: usize,
    ) {
        assert!(matches!(
            out.storage(),
            CudaStorage::F32(_) | CudaStorage::Bf16(_)
        ));
        let bf16 = out.is_bf16();
        let f = self.kv_func(kvdec_cuda::KVDEC_CUDA, "tkv_decode", bf16);
        let (cs, sn) = rope.map_or((prior, prior), |(c, s)| (c, s));
        let nl = n_lanes as u32;
        unsafe {
            let mut lb = self.stream.launch_builder(&f);
            lb.arg(stream.f32_data())
                .arg(lanes.f32_data())
                .arg(chans.f32_data())
                .arg(prior.f32_data())
                .arg(scratch.f32_data())
                .arg(kv_arg_out(out))
                .arg(cs.f32_data())
                .arg(sn.f32_data())
                .arg(&nl);
            lb.launch(blocks((n_lanes.div_ceil(32), 1, 1), 32)).unwrap();
        }
    }

    /// KIVI-style integer dequant (see `ComputeDevice::kivi_dequant`), CUDA.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kivi_dequant(
        &self,
        codes: &CudaBuffer,
        sz: &CudaBuffer,
        out: &CudaBuffer,
        pages: Option<(&CudaBuffer, u32)>,
        p: [u32; 6],
    ) {
        assert!(matches!(
            out.storage(),
            CudaStorage::F32(_) | CudaStorage::Bf16(_)
        ));
        let bf16 = out.is_bf16();
        let f = self.kv_func(kvdec_cuda::KVDEC_CUDA, "kivi_dequant", bf16);
        let prm = [p[0], p[1], p[2], p[3], p[4], p[5], pages.map_or(0, |x| x.1)];
        let pb = self.upload_u32(&prm);
        let t = pages.map_or(&pb, |x| &x.0);
        unsafe {
            let mut lb = self.stream.launch_builder(&f);
            lb.arg(codes.f32_data())
                .arg(sz.f32_data())
                .arg(kv_arg_out(out))
                .arg(pb.f32_data())
                .arg(t.f32_data());
            lb.launch(per_elem((p[3] * p[2] as u32) as usize * p[4] as usize))
                .unwrap();
        }
    }
}

/// The cache argument for a kernel that WRITES it through `&CudaBuffer` (the trait
/// method takes the shared ref): the same slice `kv_arg` pushes. The device pointer is
/// what the kernel gets and it writes through it (as `kv_arg_mut` would with `&mut`).
fn kv_arg_out<'a>(b: &'a CudaBuffer) -> &'a CudaSlice<f32> {
    match b.storage() {
        CudaStorage::F32(s) => s,
        // bf16 rides as u16 slices; erase the element type for the launch (same pointer)
        CudaStorage::Bf16(s) => unsafe { &*(s as *const CudaSlice<u16> as *const CudaSlice<f32>) },
        CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("a KV cache is f32 or bf16"),
    }
}

/// Parity against `CpuDevice` (the trait's portable reference implementations). Each test
/// skips, with a note, when there's no CUDA driver or GPU.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeBuffer, ComputeDevice, CpuDevice};

    /// Runs `check` on the GPU, or skips when there's no CUDA device (unless
    /// `TANG_REQUIRE_CUDA=1`, for the GPU box, where skipping would hide a broken setup).
    fn on_gpu(check: fn(&CudaComputeDevice)) {
        match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(dev)) => check(&dev),
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => eprintln!("no CUDA device: skipping"),
        }
    }

    /// Deterministic values in [-scale/2, scale/2).
    fn vals(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5)
            .map(|v| v * scale)
            .collect()
    }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what} at {i}: {g} vs {w}"
            );
        }
    }

    fn bf16_bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| super::super::f32_to_bf16(x)).collect()
    }

    /// Random MLX-style 4-bit weights for an `[n, k]` matrix.
    fn q4(n: usize, k: usize, group: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
        let packed = (0..n * k / 8)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x5a5a_a5a5)
            .collect();
        let scales = bf16_bits(&vals(n * k / group, 1, 0.02));
        let biases = bf16_bits(&vals(n * k / group, 2, 0.1));
        (packed, scales, biases)
    }

    fn q4_and_bf16_linear_and_embedding_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        // (k 4096 and 2048, few columns: the 4-bit GEMV's K split, even and ragged.)
        for &(m, k, n, group) in &[
            (1, 256, 96, 64),
            (8, 512, 40, 32),
            (37, 128, 72, 128),
            (1, 4096, 96, 64),
            (3, 2560, 40, 64),
        ] {
            let x = vals(m * k, 3, 2.0);
            let (p, s, b) = q4(n, k, group);
            let (gw, cw) = (
                g.upload_q4(&p, &s, &b, group),
                c.upload_q4(&p, &s, &b, group),
            );
            close(&gw.to_vec(), &c.download(&cw), 0.0, "q4 to_vec");
            let got = g.download(&g.linear(&g.upload(&x), &gw, m, k, n));
            let want = c.download(&c.linear(&c.upload(&x), &cw, m, k, n));
            close(&got, &want, 1e-4, &format!("q4 linear {m}x{k}x{n}"));

            let wb = bf16_bits(&vals(n * k, 4, 1.0));
            let (gb, cb) = (g.upload_bf16(&wb), c.upload_bf16(&wb));
            let got = g.download(&g.linear(&g.upload(&x), &gb, m, k, n));
            let want = c.download(&c.linear(&c.upload(&x), &cb, m, k, n));
            close(&got, &want, 1e-4, &format!("bf16 linear {m}x{k}x{n}"));

            let ids: Vec<u32> = (0..5).map(|i| (i * 13 % n) as u32).collect();
            for (gw, cw) in [(&gw, &cw), (&gb, &cb)] {
                let got = g.download(&g.embedding(gw, &g.upload_u32(&ids), 5, k));
                let want = c.download(&c.embedding(cw, &c.upload_u32(&ids), 5, k));
                close(&got, &want, 0.0, "embedding");
            }
        }
        // Small batches (the split-K GEMM): row counts around the GEMV / GEMM boundaries,
        // column counts off the 128-column tile, one and several K splits.
        for &(m, k, n, group) in &[
            (2, 512, 100, 64),
            (5, 512, 300, 64),
            (9, 256, 200, 64),
            (16, 1024, 130, 32),
            (32, 4096, 384, 64),
            (21, 2048, 1000, 128),
        ] {
            let x = vals(m * k, 30, 2.0);
            let (p, s, b) = q4(n, k, group);
            let got =
                g.download(&g.linear(&g.upload(&x), &g.upload_q4(&p, &s, &b, group), m, k, n));
            let want =
                c.download(&c.linear(&c.upload(&x), &c.upload_q4(&p, &s, &b, group), m, k, n));
            // 9+ rows run on tensor cores in TF32 (10-bit mantissas) on sm_80+.
            let tol = if m >= 9 { 3e-3 } else { 1e-4 };
            close(&got, &want, tol, &format!("q4 small gemm {m}x{k}x{n}"));
            let wb = bf16_bits(&vals(n * k, 31, 1.0));
            let got = g.download(&g.linear(&g.upload(&x), &g.upload_bf16(&wb), m, k, n));
            let want = c.download(&c.linear(&c.upload(&x), &c.upload_bf16(&wb), m, k, n));
            close(&got, &want, 1e-4, &format!("bf16 small gemm {m}x{k}x{n}"));
        }
        // K not a multiple of 8 (bf16 goes through the dequantized GEMM).
        let (m, k, n) = (2, 30, 17);
        let x = vals(m * k, 5, 1.0);
        let wb = bf16_bits(&vals(n * k, 6, 1.0));
        let got = g.download(&g.linear(&g.upload(&x), &g.upload_bf16(&wb), m, k, n));
        let want = c.download(&c.linear(&c.upload(&x), &c.upload_bf16(&wb), m, k, n));
        close(&got, &want, 1e-4, "bf16 linear, odd K");
    }

    /// Several dequant chunks: many output rows against a long K.
    fn q4_linear_chunked_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let (m, k, n) = (9, 4096, DEQUANT_CHUNK / 4096 * 2 + 40);
        let x = vals(m * k, 7, 1.0);
        let (p, s, b) = q4(n, k, 64);
        let got = g.download(&g.linear(&g.upload(&x), &g.upload_q4(&p, &s, &b, 64), m, k, n));
        let want = c.download(&c.linear(&c.upload(&x), &c.upload_q4(&p, &s, &b, 64), m, k, n));
        close(&got, &want, 1e-3, "chunked q4 linear");
    }

    /// `x` rounded to bf16 (as f32).
    fn bf16_round(x: &[f32]) -> Vec<f32> {
        bf16_bits(x)
            .iter()
            .map(|&b| super::super::bf16_to_f32(b))
            .collect()
    }

    /// A KV cache holding `x` (and room for `rows` elements): bf16 or f32.
    fn kv_cache<D: ComputeDevice>(g: &D, x: &[f32], rows: usize, bf16: bool) -> D::Buffer {
        if !bf16 {
            return g.upload(x);
        }
        let mut b = g.alloc_bf16(rows.max(x.len()));
        g.write_into(&mut b, 0, &g.upload(x));
        b
    }

    /// With `bf16`, the GPU's caches are bf16 and the CPU's results are rounded to compare.
    fn attention_prep_vs_cpu<D: ComputeDevice>(g: &D, bf16: bool) {
        let c = CpuDevice::new();
        for &(nh, nkv, hd, seq, pos, norms) in &[
            (4, 2, 128, 3, 5, true),
            (8, 4, 256, 1, 9, true),
            (2, 1, 64, 4, 0, false),
            (2, 2, 72, 2, 1, true),
        ] {
            let qkv = vals(seq * (nh + 2 * nkv) * hd, 8, 2.0);
            let (qn, kn) = (vals(hd, 9, 1.0), vals(hd, 10, 1.0));
            let max = 16;
            let (cos, sin) = (vals(max * hd / 2, 11, 2.0), vals(max * hd / 2, 12, 2.0));
            let cache = max * nkv * hd;
            let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>, Vec<f32>)| d();
            let got = run(&|| {
                let (mut kc, mut vc) = if bf16 {
                    (g.alloc_bf16(cache), g.alloc_bf16(cache))
                } else {
                    (g.alloc(cache), g.alloc(cache))
                };
                let (qn, kn) = (g.upload(&qn), g.upload(&kn));
                let q = g.attention_prep(
                    &g.upload(&qkv),
                    norms.then_some(&qn),
                    norms.then_some(&kn),
                    &g.upload(&cos),
                    &g.upload(&sin),
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                );
                (g.download(&q), g.download(&kc), g.download(&vc))
            });
            let want = run(&|| {
                let (mut kc, mut vc) = (c.alloc(cache), c.alloc(cache));
                let (qn, kn) = (c.upload(&qn), c.upload(&kn));
                let q = c.attention_prep(
                    &c.upload(&qkv),
                    norms.then_some(&qn),
                    norms.then_some(&kn),
                    &c.upload(&cos),
                    &c.upload(&sin),
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                );
                let (k, v) = (c.download(&kc), c.download(&vc));
                if bf16 {
                    (c.download(&q), bf16_round(&k), bf16_round(&v))
                } else {
                    (c.download(&q), k, v)
                }
            });
            let what = format!("attention_prep nh={nh} nkv={nkv} hd={hd} bf16={bf16}");
            close(&got.0, &want.0, 1e-4, &format!("{what} q"));
            // A k within 1e-4 can still round to the neighbouring bf16.
            let k_tol = if bf16 { 8e-3 } else { 1e-4 };
            close(&got.1, &want.1, k_tol, &format!("{what} k cache"));
            close(&got.2, &want.2, 0.0, &format!("{what} v cache"));
        }
    }

    /// With `bf16`, K and V are bf16 caches on the GPU (rows rounded up to 32, as the model
    /// allocates them) and bf16-rounded values on the CPU.
    fn attention_vs_cpu<D: ComputeDevice>(g: &D, bf16: bool) {
        let c = CpuDevice::new();
        // (nh, nkv, hd, cache_start, q_len, window, causal)
        let cases = [
            (4, 2, 128, 0, 1, 0, true),     // first decode step
            (4, 2, 128, 999, 1, 0, true),   // long decode: several splits
            (8, 4, 256, 700, 1, 512, true), // Gemma local layer, decode past the window
            (4, 1, 64, 0, 45, 0, true),     // prefill
            (4, 2, 128, 30, 70, 0, true),   // prefill after a cached prefix
            (4, 2, 256, 20, 50, 16, true),  // windowed prefill, D = 256
            (4, 4, 72, 3, 33, 0, false),    // image block: bidirectional within the batch
            (2, 2, 72, 0, 1, 0, false),
            // A few queries on a cache (split-KV multi-query path).
            (4, 2, 128, 999, 5, 0, true),
            (32, 8, 128, 1500, 8, 0, true), // 8B-shaped verify step
            (4, 1, 64, 30, 32, 0, true),    // 4 row groups
            (8, 4, 256, 700, 3, 512, true), // D = 256, past the window
            (4, 2, 256, 40, 20, 16, true),  // window shorter than the batch
            (4, 2, 80, 100, 2, 0, true),    // D not a multiple of 32
            (4, 4, 72, 3, 7, 0, false),     // bidirectional
            (2, 1, 128, 0, 6, 0, true),     // empty cache
        ];
        for &(nh, nkv, hd, cs, ql, window, causal) in &cases {
            let total = cs + ql;
            let q = vals(ql * nh * hd, 13, 2.0);
            let mut k = vals(total * nkv * hd, 14, 2.0);
            let mut v = vals(total * nkv * hd, 15, 2.0);
            if bf16 {
                (k, v) = (bf16_round(&k), bf16_round(&v));
            }
            let rows = total.next_multiple_of(32) * nkv * hd;
            let got = g.download(&g.kv_attention_window(
                &g.upload(&q),
                &kv_cache(g, &k, rows, bf16),
                &kv_cache(g, &v, rows, bf16),
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            let want = c.download(&c.kv_attention_window(
                &c.upload(&q),
                &c.upload(&k),
                &c.upload(&v),
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            let what = format!("attention nh={nh} nkv={nkv} hd={hd} start={cs} len={ql} window={window} causal={causal} bf16={bf16}");
            close(&got, &want, 1e-4, &what);
        }
        // Vision tower shape (SigLIP: 16 heads of 72), smaller grid.
        let (n, nh, hd) = (100, 16, 72);
        let (q, k, v) = (
            vals(n * nh * hd, 16, 2.0),
            vals(n * nh * hd, 17, 2.0),
            vals(n * nh * hd, 18, 2.0),
        );
        let got =
            g.download(&g.attention_full(&g.upload(&q), &g.upload(&k), &g.upload(&v), n, nh, hd));
        let want =
            c.download(&c.attention_full(&c.upload(&q), &c.upload(&k), &c.upload(&v), n, nh, hd));
        close(&got, &want, 1e-4, "attention_full");
    }

    /// A paged copy of contiguous K or V rows `x` (`kvd` wide): blocks of `block` rows, placed
    /// in a pool of `ids.len() + 3` blocks in the scrambled order `ids`.
    fn paged<D: ComputeDevice>(
        g: &D,
        x: &[f32],
        kvd: usize,
        ids: &[u32],
        block: usize,
        bf16: bool,
    ) -> D::Buffer {
        let mut pool = vec![0.0f32; (ids.len() + 3) * block * kvd];
        for p in 0..x.len() / kvd {
            let row = ids[p / block] as usize * block + p % block;
            pool[row * kvd..(row + 1) * kvd].copy_from_slice(&x[p * kvd..(p + 1) * kvd]);
        }
        kv_cache(g, &pool, pool.len(), bf16)
    }

    /// Block ids for `n` blocks, out of order, within a pool of `n + 3`: odd blocks from the
    /// top down, even ones from the bottom up past a gap.
    fn scrambled(n: usize) -> Vec<u32> {
        (0..n)
            .map(|i| if i % 2 == 1 { n + 2 - i / 2 } else { 1 + i / 2 } as u32)
            .collect()
    }

    /// Paged attention and prologue against the contiguous ones on the same device, and the
    /// paged attention against the CPU's paged fallback.
    fn paged_vs_contiguous<D: ComputeDevice>(g: &D, bf16: bool) {
        let c = CpuDevice::new();
        // (nh, nkv, hd, cache_start, q_len, window, causal, block)
        let cases = [
            (4, 2, 128, 999, 1, 0, true, 32),    // decode, several splits
            (4, 2, 128, 2100, 1, 0, true, 256),  // lane-per-key decode
            (8, 4, 256, 700, 1, 512, true, 64),  // windowed decode, D = 256
            (4, 1, 64, 0, 45, 0, true, 32),      // prefill
            (4, 2, 128, 300, 100, 0, true, 64),  // prefill after a prefix
            (4, 2, 256, 20, 50, 16, true, 32),   // windowed prefill, D = 256
            (4, 4, 72, 3, 33, 0, false, 32),     // bidirectional, D not a multiple of 64
            (4, 2, 128, 999, 5, 0, true, 128),   // multi-query
            (32, 8, 128, 1500, 8, 0, true, 256), // verify step
            (8, 4, 256, 700, 3, 512, true, 32),  // multi-query, D = 256, windowed
            (4, 2, 80, 100, 2, 0, true, 32),     // D not a multiple of 32
        ];
        for &(nh, nkv, hd, cs, ql, window, causal, block) in &cases {
            let (total, kvd) = (cs + ql, nkv * hd);
            let q = vals(ql * nh * hd, 21, 2.0);
            let mut k = vals(total * kvd, 22, 2.0);
            let mut v = vals(total * kvd, 23, 2.0);
            if bf16 {
                (k, v) = (bf16_round(&k), bf16_round(&v));
            }
            let ids = scrambled(total.div_ceil(block));
            let what = format!("paged attention nh={nh} nkv={nkv} hd={hd} start={cs} len={ql} window={window} causal={causal} block={block} bf16={bf16}");
            let run = |d: &D, cpu: bool| -> Vec<f32> {
                let _ = cpu;
                let table = d.upload_u32(&ids);
                let pages = crate::Pages {
                    ids: &ids,
                    table: &table,
                    block,
                };
                d.download(&d.kv_attention_paged(
                    &d.upload(&q),
                    &paged(d, &k, kvd, &ids, block, bf16),
                    &paged(d, &v, kvd, &ids, block, bf16),
                    &pages,
                    cs,
                    ql,
                    (nh, nkv, hd),
                    window,
                    causal,
                ))
            };
            let got = run(g, false);
            let rows = total.next_multiple_of(32) * kvd;
            let flat = g.download(&g.kv_attention_window(
                &g.upload(&q),
                &kv_cache(g, &k, rows, bf16),
                &kv_cache(g, &v, rows, bf16),
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            close(&got, &flat, 1e-5, &format!("{what} vs contiguous"));
            let table = c.upload_u32(&ids);
            let pages = crate::Pages {
                ids: &ids,
                table: &table,
                block,
            };
            let want = c.download(&c.kv_attention_paged(
                &c.upload(&q),
                &paged(&c, &k, kvd, &ids, block, false),
                &paged(&c, &v, kvd, &ids, block, false),
                &pages,
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            close(&got, &want, 1e-4, &format!("{what} vs cpu"));
        }
        // The prologue writes each position's k and v where the table puts it.
        for &(nh, nkv, hd, seq, pos, block) in &[
            (4, 2, 128, 40, 50, 32),
            (8, 4, 256, 1, 300, 256),
            (2, 2, 72, 5, 30, 32),
        ] {
            let kvd = nkv * hd;
            let qkv = vals(seq * (nh + 2 * nkv) * hd, 24, 2.0);
            let (qn, kn) = (vals(hd, 25, 1.0), vals(hd, 26, 1.0));
            let max = 512;
            let (cos, sin) = (vals(max * hd / 2, 27, 2.0), vals(max * hd / 2, 28, 2.0));
            let ids = scrambled((pos + seq).div_ceil(block));
            let table = g.upload_u32(&ids);
            let pages = crate::Pages {
                ids: &ids,
                table: &table,
                block,
            };
            let pool = (ids.len() + 3) * block * kvd;
            let (mut kp, mut vp) = if bf16 {
                (g.alloc_bf16(pool), g.alloc_bf16(pool))
            } else {
                (g.alloc(pool), g.alloc(pool))
            };
            let (qn, kn) = (g.upload(&qn), g.upload(&kn));
            let (qkv, cos, sin) = (g.upload(&qkv), g.upload(&cos), g.upload(&sin));
            let q1 = g.download(&g.attention_prep_paged(
                &qkv,
                Some(&qn),
                Some(&kn),
                &cos,
                &sin,
                &mut kp,
                &mut vp,
                &pages,
                seq,
                (nh, nkv, hd),
                pos,
                1e-6,
            ));
            let rows = (pos + seq) * kvd;
            let (mut kc, mut vc) = if bf16 {
                (g.alloc_bf16(rows), g.alloc_bf16(rows))
            } else {
                (g.alloc(rows), g.alloc(rows))
            };
            let q2 = g.download(&g.attention_prep(
                &qkv,
                Some(&qn),
                Some(&kn),
                &cos,
                &sin,
                &mut kc,
                &mut vc,
                seq,
                (nh, nkv, hd),
                pos,
                1e-6,
            ));
            let what = format!("paged prologue nh={nh} hd={hd} seq={seq} pos={pos} bf16={bf16}");
            close(&q1, &q2, 0.0, &format!("{what} q"));
            let (kp, vp, kc, vc) = (
                g.download(&kp),
                g.download(&vp),
                g.download(&kc),
                g.download(&vc),
            );
            for p in pos..pos + seq {
                let row = pages.row(p);
                let at = |x: &[f32], r: usize| x[r * kvd..(r + 1) * kvd].to_vec();
                close(&at(&kp, row), &at(&kc, p), 0.0, &format!("{what} k at {p}"));
                close(&at(&vp, row), &at(&vc, p), 0.0, &format!("{what} v at {p}"));
            }
        }
    }

    fn elementwise_and_norms_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let (rows, ff, dim) = (3, 300, 1152);
        // Large gate values exercise the tanh clamp.
        let gu = vals(rows * 2 * ff, 19, 40.0);
        close(
            &g.download(&g.geglu_split(&g.upload(&gu), rows, ff)),
            &c.download(&c.geglu_split(&c.upload(&gu), rows, ff)),
            1e-5,
            "geglu_split",
        );
        close(
            &g.download(&g.swiglu_split(&g.upload(&gu), rows, ff)),
            &c.download(&c.swiglu_split(&c.upload(&gu), rows, ff)),
            1e-5,
            "swiglu_split",
        );
        close(
            &g.download(&g.gelu_tanh(&g.upload(&gu), gu.len())),
            &c.download(&c.gelu_tanh(&c.upload(&gu), gu.len())),
            1e-5,
            "gelu_tanh",
        );
        close(
            &g.download(&g.gelu_erf(&g.upload(&gu), gu.len())),
            &c.download(&c.gelu_erf(&c.upload(&gu), gu.len())),
            1e-5,
            "gelu_erf",
        );
        let a = vals(rows * dim, 20, 3.0);
        let b = vals(rows * dim, 21, 3.0);
        close(
            &g.download(&g.add_tensors_buf(&g.upload(&a), &g.upload(&b), a.len())),
            &c.download(&c.add_tensors_buf(&c.upload(&a), &c.upload(&b), a.len())),
            0.0,
            "add",
        );
        let (w, bias) = (vals(dim, 22, 1.0), vals(dim, 23, 1.0));
        close(
            &g.download(&g.layer_norm(
                &g.upload(&a),
                &g.upload(&w),
                &g.upload(&bias),
                rows,
                dim,
                1e-6,
            )),
            &c.download(&c.layer_norm(
                &c.upload(&a),
                &c.upload(&w),
                &c.upload(&bias),
                rows,
                dim,
                1e-6,
            )),
            1e-4,
            "layer_norm",
        );
        close(
            &g.download(&g.rms_norm(&g.upload(&a), &g.upload(&w), rows, dim, 1e-6)),
            &c.download(&c.rms_norm(&c.upload(&a), &c.upload(&w), rows, dim, 1e-6)),
            1e-4,
            "rms_norm",
        );
        let (seq, nh, hd, pos) = (3, 4, 128, 7);
        let x = vals(seq * nh * hd, 24, 2.0);
        let (cos, sin) = (vals(16 * hd / 2, 25, 2.0), vals(16 * hd / 2, 26, 2.0));
        close(
            &g.download(&g.rope_half_cached(
                &g.upload(&x),
                &g.upload(&cos),
                &g.upload(&sin),
                seq,
                nh,
                hd,
                pos,
            )),
            &c.download(&c.rope_half_cached(
                &c.upload(&x),
                &c.upload(&cos),
                &c.upload(&sin),
                seq,
                nh,
                hd,
                pos,
            )),
            1e-5,
            "rope_half",
        );
    }

    /// A row of the 9..32-row 4-bit GEMM gets the same bits whatever the row count.
    fn q4_small_gemm_rows_are_batch_invariant<D: ComputeDevice>(g: &D) {
        let (k, n, group) = (4096, 1000, 64);
        let x = vals(32 * k, 33, 2.0);
        let (p, s, b) = q4(n, k, group);
        let w = g.upload_q4(&p, &s, &b, group);
        let all = g.download(&g.linear(&g.upload(&x), &w, 32, k, n));
        for m in [9, 16, 17, 31] {
            let got = g.download(&g.linear(&g.upload(&x[..m * k]), &w, m, k, n));
            assert_eq!(got, all[..m * n], "{m} rows vs 32");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_q4_small_gemm_rows_are_batch_invariant() {
        on_gpu(|g| q4_small_gemm_rows_are_batch_invariant(g));
    }

    #[test]
    fn cuda_top_k_rows_vs_host() {
        on_gpu(|g| {
            for &(rows, n, k) in &[(1, 151936, 20), (3, 1000, 64), (2, 64, 64), (4, 5000, 1)] {
                // `vals` repeats every 1009 columns: plenty of ties, at the K-th place too.
                let v = vals(rows * n, 9, 4.0);
                let want = crate::top_k_rows_host(&v, n, k);
                let got = g.top_k_rows(&g.upload(&v), rows, n, k);
                assert_eq!(got.1, want.1, "top_k_rows idx {rows}x{n} k{k}");
                assert_eq!(got.0, want.0, "top_k_rows vals {rows}x{n} k{k}");
            }
        });
    }

    /// Timings of the small-k kernels (not a check) for a model shape
    /// `(heads, kv heads, head dim, hidden, ff)`.
    fn bench_small_k_on<D: ComputeDevice>(
        g: &D,
        (nh, nkv, hd, h, ff): (usize, usize, usize, usize, usize),
    ) {
        let time = |f: &mut dyn FnMut()| {
            f();
            g.sync();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            g.sync();
            t.elapsed().as_secs_f64() * 1e6 / 20.0
        };
        let ks = [1, 2, 4, 8, 16, 32];
        for ctx in [4096, 16384] {
            let k = g.upload(&vals((ctx + 32) * nkv * hd, 1, 2.0));
            let v = g.upload(&vals((ctx + 32) * nkv * hd, 2, 2.0));
            let mut line = format!("attention ctx {ctx:>5} (us):");
            for ql in ks {
                let q = g.upload(&vals(ql * nh * hd, 3, 2.0));
                let us = time(&mut || {
                    g.kv_attention_window(&q, &k, &v, ctx, ql, (nh, nkv, hd), 0, true);
                });
                line += &format!(" k{ql} {us:.0}");
            }
            eprintln!("{line}");
        }
        for (kk, n) in [(nh * hd, h), (h, (nh + 2 * nkv) * hd), (h, 2 * ff), (ff, h)] {
            let (p, s, b) = q4(n, kk, 64);
            let w = g.upload_q4(&p, &s, &b, 64);
            let mut line = format!("q4 linear {kk:>5}x{n:>5} (us):");
            for m in [1, 2, 4, 8, 9, 16, 32] {
                let x = g.upload(&vals(m * kk, 4, 1.0));
                let us = time(&mut || {
                    g.linear(&x, &w, m, kk, n);
                });
                line += &format!(" m{m} {us:.0}");
            }
            eprintln!("{line}");
            let wb = g.upload_bf16(&bf16_bits(&vals(n * kk, 5, 1.0)));
            let mut line = format!("bf16 linear {kk:>5}x{n:>5} (us):");
            for m in [1, 2, 4, 8, 9, 16, 32] {
                let x = g.upload(&vals(m * kk, 4, 1.0));
                let us = time(&mut || {
                    g.linear(&x, &wb, m, kk, n);
                });
                line += &format!(" m{m} {us:.0}");
            }
            eprintln!("{line}");
        }
    }

    /// `cargo test --release -p tang-compute --features cuda --lib bench_small_k -- --ignored --nocapture`
    /// 4-bit decode GEMV timings at Qwen3-8B shapes (not a check).
    #[test]
    #[ignore]
    fn bench_q4_gemv_cuda() {
        on_gpu(|g| {
            for &(k, n) in &[
                (4096, 4096),
                (4096, 6144),
                (4096, 24576),
                (12288, 4096),
                (4096, 151936),
            ] {
                let (p, s, b) = q4(n, k, 64);
                let w = g.upload_q4(&p, &s, &b, 64);
                let bytes = n * k / 2 + n * k / 64 * 4;
                for m in [1, 8] {
                    let x = g.upload(&vals(m * k, 3, 2.0));
                    g.linear(&x, &w, m, k, n);
                    g.sync();
                    let t = std::time::Instant::now();
                    for _ in 0..50 {
                        g.linear(&x, &w, m, k, n);
                    }
                    g.sync();
                    let us = t.elapsed().as_secs_f64() * 1e6 / 50.0;
                    println!(
                        "q4 gemv m{m} {k}x{n}: {us:7.1} us {:6.0} GB/s",
                        bytes as f64 / us / 1e3
                    );
                }
            }
        });
    }

    #[test]
    #[ignore]
    fn bench_small_k_cuda() {
        on_gpu(|g| bench_small_k_on(g, (32, 8, 128, 4096, 12288)));
    }

    /// Qwen3-4B shapes on Metal.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn bench_small_k_metal() {
        bench_small_k_on(
            &crate::MetalDevice::new().expect("no Metal device"),
            (32, 8, 128, 2560, 9728),
        );
    }

    #[test]
    fn cuda_q4_and_bf16_linear_and_embedding_vs_cpu() {
        on_gpu(q4_and_bf16_linear_and_embedding_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q4_linear_chunked_vs_cpu() {
        on_gpu(q4_linear_chunked_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_attention_prep_vs_cpu() {
        on_gpu(|g| attention_prep_vs_cpu(g, false));
        on_gpu(|g| attention_prep_vs_cpu(g, true));
    }

    #[test]
    fn cuda_attention_vs_cpu() {
        on_gpu(|g| attention_vs_cpu(g, false));
        on_gpu(|g| attention_vs_cpu(g, true));
    }

    #[test]
    fn cuda_paged_vs_contiguous() {
        on_gpu(|g| paged_vs_contiguous(g, false));
        on_gpu(|g| paged_vs_contiguous(g, true));
    }

    #[test]
    fn cuda_elementwise_and_norms_vs_cpu() {
        on_gpu(elementwise_and_norms_vs_cpu::<CudaComputeDevice>);
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_q4_and_bf16_linear_and_embedding_vs_cpu() {
        q4_and_bf16_linear_and_embedding_vs_cpu(
            &crate::MetalDevice::new().expect("no Metal device"),
        );
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_q4_linear_chunked_vs_cpu() {
        q4_linear_chunked_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_prep_vs_cpu() {
        let g = crate::MetalDevice::new().expect("no Metal device");
        attention_prep_vs_cpu(&g, false);
        attention_prep_vs_cpu(&g, true);
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_vs_cpu() {
        let g = crate::MetalDevice::new().expect("no Metal device");
        attention_vs_cpu(&g, false);
        attention_vs_cpu(&g, true);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_paged_vs_contiguous() {
        let g = crate::MetalDevice::new().expect("no Metal device");
        paged_vs_contiguous(&g, false);
        paged_vs_contiguous(&g, true);
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_elementwise_and_norms_vs_cpu() {
        elementwise_and_norms_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }
}
