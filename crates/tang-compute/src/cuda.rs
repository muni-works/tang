//! CUDA compute backend via cudarc.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{attention_cuda, backward_cuda, reduce_cuda, util_cuda};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

// ---- bf16 CPU conversion helpers ----

fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---- Buffer storage ----

enum CudaStorage {
    F32(CudaSlice<f32>),
    Bf16(CudaSlice<u16>),
    /// Read-only 4-bit weights for `linear`/`embedding` (see `ComputeDevice::upload_q4`).
    Q4(Box<Q4Weight>),
    /// Read-only Q2_0 weights in the repacked layout (`crate::flash::q2_repack`), for the Q2
    /// ops only (`ComputeDevice::q2_linear_into`).
    Q2(Box<Q2Weight>),
}

/// A repacked Q2_0 `[n, k]` matrix: `[n][k/4]` code bytes, then `[n][k/64]` fp16 scales.
struct Q2Weight {
    data: CudaSlice<u8>,
    n: usize,
    k: usize,
}

/// 4-bit affine weights in MLX's layout: `packed` holds 8 weights per u32 (low nibble first)
/// for a row-major `[n, k]` matrix; `scales` and `biases` are bf16, one per `group` weights.
struct Q4Weight {
    packed: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    group: usize,
}

/// CUDA buffer wrapping either a CudaSlice<f32> or CudaSlice<u16> (bf16).
/// When dropped, automatically returns storage to the buffer pool (if pooled).
pub struct CudaBuffer {
    storage: Option<CudaStorage>,
    len: usize,
    pool: Option<Arc<Mutex<BufferPool>>>,
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if let (Some(storage), Some(pool)) = (self.storage.take(), self.pool.as_ref()) {
            let mut p = pool.lock().unwrap();
            match storage {
                CudaStorage::F32(slice) => p.put_f32(slice, self.len),
                CudaStorage::Bf16(slice) => p.put_bf16(slice, self.len),
                CudaStorage::Q4(_) | CudaStorage::Q2(_) => {}
            }
        }
        // If no pool ref, storage is dropped normally (cudaFree)
    }
}

impl CudaBuffer {
    fn storage(&self) -> &CudaStorage {
        self.storage.as_ref().expect("buffer storage already taken")
    }

    fn storage_mut(&mut self) -> &mut CudaStorage {
        self.storage.as_mut().expect("buffer storage already taken")
    }
}

/// A [`CudaBuffer`]'s storage, read-only, for extensions that launch their own kernels on it
/// (see [`CudaBuffer::raw`]). A kernel may write through these (a KV cache, say) only where the
/// matching built-in op would.
pub enum RawCuda<'a> {
    F32(&'a CudaSlice<f32>),
    /// bf16 bits.
    Bf16(&'a CudaSlice<u16>),
    /// MLX 4-bit `[n, k]`: `packed` 8 weights a u32 (low nibble first), bf16 `scales` and
    /// `biases` one per `group` weights.
    Q4 {
        packed: &'a CudaSlice<u32>,
        scales: &'a CudaSlice<u16>,
        biases: &'a CudaSlice<u16>,
        group: usize,
    },
    /// Repacked Q2_0 (`crate::flash::q2_repack`): `[n][k/4]` code bytes, then `[n][k/64]`
    /// fp16 scales.
    Q2 {
        data: &'a CudaSlice<u8>,
        n: usize,
        k: usize,
    },
}

impl CudaBuffer {
    /// The storage behind this buffer.
    pub fn raw(&self) -> RawCuda<'_> {
        match self.storage() {
            CudaStorage::F32(s) => RawCuda::F32(s),
            CudaStorage::Bf16(s) => RawCuda::Bf16(s),
            CudaStorage::Q4(q) => RawCuda::Q4 {
                packed: &q.packed,
                scales: &q.scales,
                biases: &q.biases,
                group: q.group,
            },
            CudaStorage::Q2(q) => RawCuda::Q2 {
                data: &q.data,
                n: q.n,
                k: q.k,
            },
        }
    }
}

impl ComputeBuffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn to_vec(&self) -> Vec<f32> {
        match self.storage() {
            CudaStorage::F32(s) => s.stream().memcpy_dtov(s).unwrap(),
            CudaStorage::Bf16(s) => {
                let u16_data: Vec<u16> = s.stream().memcpy_dtov(s).unwrap();
                u16_data.iter().map(|&b| bf16_to_f32(b)).collect()
            }
            // The raw repacked bytes, 4 per element (what the portable `upload_q2` stores).
            CudaStorage::Q2(q) => {
                let bytes: Vec<u8> = q.data.stream().memcpy_dtov(&q.data).unwrap();
                crate::flash::words_of(&crate::flash::bytes_to_words(&bytes))
            }
            CudaStorage::Q4(q) => {
                let stream = q.packed.stream();
                let packed: Vec<u32> = stream.memcpy_dtov(&q.packed).unwrap();
                let scales: Vec<u16> = stream.memcpy_dtov(&q.scales).unwrap();
                let biases: Vec<u16> = stream.memcpy_dtov(&q.biases).unwrap();
                (0..self.len)
                    .map(|i| {
                        let v = (packed[i / 8] >> (4 * (i % 8))) & 0xf;
                        let g = i / q.group;
                        bf16_to_f32(scales[g]) * v as f32 + bf16_to_f32(biases[g])
                    })
                    .collect()
            }
        }
    }
}

impl CudaBuffer {
    /// Whether this buffer stores bf16 data.
    pub fn is_bf16(&self) -> bool {
        matches!(self.storage(), CudaStorage::Bf16(_))
    }

    /// Get the underlying f32 slice (immutable). Panics if bf16.
    fn f32_data(&self) -> &CudaSlice<f32> {
        match self.storage() {
            CudaStorage::F32(s) => s,
            _ => panic!("expected f32 buffer"),
        }
    }

    /// Get the underlying f32 slice (mutable). Panics if bf16.
    fn f32_data_mut(&mut self) -> &mut CudaSlice<f32> {
        match self.storage_mut() {
            CudaStorage::F32(s) => s,
            _ => panic!("expected f32 buffer"),
        }
    }

    /// Get the underlying u16 (bf16) slice (immutable). Panics if f32.
    fn bf16_data(&self) -> &CudaSlice<u16> {
        match self.storage() {
            CudaStorage::Bf16(s) => s,
            _ => panic!("expected bf16 buffer"),
        }
    }

    /// Get the underlying u16 (bf16) slice (mutable). Panics if f32.
    fn bf16_data_mut(&mut self) -> &mut CudaSlice<u16> {
        match self.storage_mut() {
            CudaStorage::Bf16(s) => s,
            _ => panic!("expected bf16 buffer"),
        }
    }
}

use crate::pool::BufferPool;

mod flash;
mod llm;
mod moe;

/// Extended pool diagnostics.
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub evictions: u64,
    pub pressure_clears: u64,
    pub cached_mb: f64,
    pub n_buckets: usize,
}

/// CUDA compute device.
pub struct CudaComputeDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cublas: CudaBlas,
    module_cache: RefCell<HashMap<u64, Arc<CudaModule>>>, // hash → module
    mixed_precision: bool,
    pool: Arc<Mutex<BufferPool>>,
    /// LLM kernels by name, so decode doesn't hash kernel sources on every launch.
    llm_funcs: RefCell<HashMap<&'static str, CudaFunction>>,
    /// Self-resetting grid-barrier and completion counters for the fused Flash-Next kernels
    /// (`cuda/flash.rs`), allocated on first use.
    flash_sync: RefCell<Option<CudaSlice<u32>>>,
    /// The flash module's source for the current [`crate::flash::Geometry`] (Flash-Next until
    /// `set_flash_geometry`).
    flash_src: RefCell<Option<&'static str>>,
}

/// GPU 0's name, free and total memory, without making a device (no stream or cuBLAS handle):
/// it holds the primary context only while asking. Free memory counts every process's use.
pub fn cuda_memory_info() -> Result<(String, usize, usize), cudarc::driver::DriverError> {
    let ctx = CudaContext::new(0)?;
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    Ok((ctx.name()?, free, total))
}

impl CudaComputeDevice {
    /// Create a CUDA device on `ordinal` (the default [`CudaComputeDevice::new`] is 0).
    pub fn on(ordinal: usize) -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        // Single-stream: disable event tracking to avoid per-op cuEventRecord overhead
        unsafe {
            ctx.disable_event_tracking();
        }
        // Use non-blocking stream to support CUDA graph capture
        let stream = ctx.new_stream()?;
        let cublas = CudaBlas::new(stream.clone()).expect("Failed to create cuBLAS handle");
        if std::env::var("GAIA_TF32")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *cublas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                );
            }
        }
        Ok(CudaComputeDevice {
            ctx,
            stream,
            cublas,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: false,
            pool: Arc::new(Mutex::new(BufferPool::new())),
            llm_funcs: RefCell::new(HashMap::new()),
            flash_sync: RefCell::new(None),
            flash_src: RefCell::new(None),
        })
    }

    /// Create a new CUDA device (ordinal 0), f32 precision.
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(0)?;
        // Single-stream: disable event tracking to avoid per-op cuEventRecord overhead
        unsafe {
            ctx.disable_event_tracking();
        }
        // Use non-blocking stream to support CUDA graph capture
        let stream = ctx.new_stream()?;
        let cublas = CudaBlas::new(stream.clone()).expect("Failed to create cuBLAS handle");
        // TF32 tensor cores: ~2× matmul throughput on Ampere+ but reduces mantissa
        // from 23 to 10 bits. Disable by default for training stability.
        // Enable with GAIA_TF32=1 if needed for inference speed.
        if std::env::var("GAIA_TF32")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *cublas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                );
            }
        }
        Ok(CudaComputeDevice {
            ctx,
            stream,
            cublas,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: false,
            pool: Arc::new(Mutex::new(BufferPool::new())),
            llm_funcs: RefCell::new(HashMap::new()),
            flash_sync: RefCell::new(None),
            flash_src: RefCell::new(None),
        })
    }

    /// Switch cuBLAS f32 GEMMs between TF32 tensor cores (`on`) and full FP32 (the default,
    /// unless `GAIA_TF32=1`). TF32 keeps 10 mantissa bits in the products (f32 accumulate):
    /// fine for inference prefill, riskier for training.
    pub fn set_tf32(&self, on: bool) {
        let mode = if on {
            cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
        } else {
            cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH
        };
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(*self.cublas.handle(), mode);
        }
    }

    /// Create a new CUDA device with bf16 mixed precision.
    /// Weights and activations stored in bf16, compute in f32 internally.
    pub fn new_mixed_precision() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(0)?;
        // Single-stream: disable event tracking to avoid per-op cuEventRecord overhead
        unsafe {
            ctx.disable_event_tracking();
        }
        // Use non-blocking stream to support CUDA graph capture
        let stream = ctx.new_stream()?;
        let cublas = CudaBlas::new(stream.clone()).expect("Failed to create cuBLAS handle");
        // TF32 tensor cores: ~2× matmul throughput on Ampere+ but reduces mantissa
        // from 23 to 10 bits. Disable by default for training stability.
        // Enable with GAIA_TF32=1 if needed for inference speed.
        if std::env::var("GAIA_TF32")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *cublas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                );
            }
        }
        Ok(CudaComputeDevice {
            ctx,
            stream,
            cublas,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: true,
            pool: Arc::new(Mutex::new(BufferPool::new())),
            llm_funcs: RefCell::new(HashMap::new()),
            flash_sync: RefCell::new(None),
            flash_src: RefCell::new(None),
        })
    }

    /// Wrap storage + len into a pooled CudaBuffer (returned to pool on drop).
    fn make_buf(&self, storage: CudaStorage, len: usize) -> CudaBuffer {
        CudaBuffer {
            storage: Some(storage),
            len,
            pool: Some(Arc::clone(&self.pool)),
        }
    }

    /// Wrap storage + len into an unpooled CudaBuffer (freed on drop).
    /// Use for uploads and temporaries with unique sizes.
    fn make_buf_unpooled(storage: CudaStorage, len: usize) -> CudaBuffer {
        CudaBuffer {
            storage: Some(storage),
            len,
            pool: None,
        }
    }

    /// Allocate an f32 buffer using the pool. Zeros the memory.
    fn pool_alloc_f32(&self, len: usize) -> CudaBuffer {
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(mut slice) = pool.get_f32(len) {
                self.stream.memset_zeros(&mut slice).unwrap();
                return self.make_buf(CudaStorage::F32(slice), len);
            }
            // Only a miss allocates, so only a miss can push VRAM low: check here, not on
            // every allocation (cuMemGetInfo cost ~6 µs of CPU a launch).
            pool.maybe_evict_under_pressure();
        }
        match self.stream.alloc_zeros::<f32>(len) {
            Ok(slice) => self.make_buf(CudaStorage::F32(slice), len),
            Err(_) => {
                self.pool.lock().unwrap().clear();
                let slice = self.stream.alloc_zeros::<f32>(len).unwrap();
                self.make_buf(CudaStorage::F32(slice), len)
            }
        }
    }

    /// Allocate an f32 buffer using the pool WITHOUT zeroing.
    /// Caller must fully overwrite before reading (e.g., matmul output).
    fn pool_alloc_uninit_f32(&self, len: usize) -> CudaBuffer {
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(slice) = pool.get_f32(len) {
                return self.make_buf(CudaStorage::F32(slice), len);
            }
            // Only a miss allocates, so only a miss can push VRAM low: check here, not on
            // every allocation (cuMemGetInfo cost ~6 µs of CPU a launch).
            pool.maybe_evict_under_pressure();
        }
        match unsafe { self.stream.alloc::<f32>(len) } {
            Ok(slice) => self.make_buf(CudaStorage::F32(slice), len),
            Err(_) => {
                // OOM — evict pool and retry
                self.pool.lock().unwrap().clear();
                let slice = unsafe { self.stream.alloc::<f32>(len).unwrap() };
                self.make_buf(CudaStorage::F32(slice), len)
            }
        }
    }

    /// Allocate a bf16 buffer using the pool. Zeros the memory.
    fn pool_alloc_bf16(&self, len: usize) -> CudaBuffer {
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(mut slice) = pool.get_bf16(len) {
                self.stream.memset_zeros(&mut slice).unwrap();
                return self.make_buf(CudaStorage::Bf16(slice), len);
            }
            // Only a miss allocates, so only a miss can push VRAM low: check here, not on
            // every allocation (cuMemGetInfo cost ~6 µs of CPU a launch).
            pool.maybe_evict_under_pressure();
        }
        match self.stream.alloc_zeros::<u16>(len) {
            Ok(slice) => self.make_buf(CudaStorage::Bf16(slice), len),
            Err(_) => {
                self.pool.lock().unwrap().clear();
                let slice = self.stream.alloc_zeros::<u16>(len).unwrap();
                self.make_buf(CudaStorage::Bf16(slice), len)
            }
        }
    }

    /// Allocate a bf16 buffer using the pool WITHOUT zeroing.
    fn pool_alloc_uninit_bf16(&self, len: usize) -> CudaBuffer {
        {
            let mut pool = self.pool.lock().unwrap();
            if let Some(slice) = pool.get_bf16(len) {
                return self.make_buf(CudaStorage::Bf16(slice), len);
            }
            // Only a miss allocates, so only a miss can push VRAM low: check here, not on
            // every allocation (cuMemGetInfo cost ~6 µs of CPU a launch).
            pool.maybe_evict_under_pressure();
        }
        match unsafe { self.stream.alloc::<u16>(len) } {
            Ok(slice) => self.make_buf(CudaStorage::Bf16(slice), len),
            Err(_) => {
                self.pool.lock().unwrap().clear();
                let slice = unsafe { self.stream.alloc::<u16>(len).unwrap() };
                self.make_buf(CudaStorage::Bf16(slice), len)
            }
        }
    }

    /// Return a buffer to the pool for reuse.
    /// With auto-reclaim, just dropping the buffer works too.
    pub fn reclaim(&self, _buf: CudaBuffer) {
        // Drop impl handles returning storage to pool
    }

    /// Get pool diagnostics: (hits, misses, hit_rate, evictions, cached_mb, buckets, pressure_clears).
    pub fn pool_stats(&self) -> (u64, u64, f64, u64) {
        let pool = self.pool.lock().unwrap();
        (pool.hits, pool.misses, pool.hit_rate(), pool.evictions)
    }

    /// Get extended pool diagnostics.
    pub fn pool_stats_extended(&self) -> PoolStats {
        let pool = self.pool.lock().unwrap();
        PoolStats {
            hits: pool.hits,
            misses: pool.misses,
            hit_rate: pool.hit_rate(),
            evictions: pool.evictions,
            pressure_clears: pool.pressure_clears,
            cached_mb: pool.cached_bytes() as f64 / (1024.0 * 1024.0),
            n_buckets: pool.n_buckets(),
        }
    }

    /// Drop all cached buffers in the pool, returning memory to CUDA.
    pub fn pool_clear(&self) {
        self.pool.lock().unwrap().clear();
    }

    // ---- Raw access, for extensions (`crate::ext`) ----

    /// Whether activations are bf16 (`new_mixed_precision`) rather than f32.
    pub fn is_mixed_precision(&self) -> bool {
        self.mixed_precision
    }

    // ---- CUDA Graph capture ----

    /// Begin capturing GPU operations into a CUDA graph.
    /// All subsequent GPU ops on this device's stream are recorded, not executed.
    /// Call `end_capture()` to finalize and get a replayable graph.
    pub fn begin_capture(&self) {
        self.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .expect("failed to begin CUDA graph capture");
    }

    /// End graph capture and return the executable graph.
    /// Returns None if no operations were captured.
    pub fn end_capture(&self) -> Option<CudaGraph> {
        self.stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        ).expect("failed to end CUDA graph capture")
    }

    /// Upload f32 data into an existing buffer (async, graph-capturable).
    /// Buffer must be f32 and have len >= data.len().
    pub fn upload_into_f32(&self, buf: &mut CudaBuffer, data: &[f32]) {
        assert!(buf.len >= data.len(), "buffer too small for upload_into");
        let dst = buf.f32_data_mut();
        self.stream.memcpy_htod(data, dst).unwrap();
    }

    /// Upload f32 data into `buf` at element `offset`, leaving the rest alone (async,
    /// ordered on this device's stream).
    pub fn upload_into_f32_at(&self, buf: &mut CudaBuffer, offset: usize, data: &[f32]) {
        assert!(
            offset + data.len() <= buf.len,
            "upload_into_f32_at out of bounds"
        );
        let mut view = buf
            .f32_data_mut()
            .try_slice_mut(offset..offset + data.len())
            .unwrap();
        self.stream.memcpy_htod(data, &mut view).unwrap();
    }

    /// Upload u32 data into an existing f32 buffer (async, graph-capturable).
    /// Reinterprets u32 as f32 bits (for token IDs stored as f32 bit patterns).
    pub fn upload_into_u32(&self, buf: &mut CudaBuffer, data: &[u32]) {
        let f32_data: Vec<f32> = data.iter().map(|&x| f32::from_bits(x)).collect();
        self.upload_into_f32(buf, &f32_data);
    }

    /// Compile and load a CUDA kernel, returning the module.
    fn get_module(&self, source: &str) -> Arc<CudaModule> {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(module) = self.module_cache.borrow().get(&hash) {
            return Arc::clone(module);
        }

        let ptx = nvrtc::compile_ptx(source).expect("Failed to compile CUDA kernel");
        let module = self.ctx.load_module(ptx).expect("Failed to load PTX");

        self.module_cache
            .borrow_mut()
            .insert(hash, Arc::clone(&module));
        module
    }

    /// Compile and load a CUDA kernel with explicit GPU architecture.
    /// Required for kernels using wmma/tensor cores (needs sm_80+).
    fn get_module_with_arch(&self, source: &str, arch: &'static str) -> Arc<CudaModule> {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        arch.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(module) = self.module_cache.borrow().get(&hash) {
            return Arc::clone(module);
        }

        // Find CUDA include path for mma.h etc
        let mut include_paths = Vec::new();
        for candidate in &[
            "/usr/local/cuda/include",
            "/usr/local/cuda/targets/x86_64-linux/include",
        ] {
            if std::path::Path::new(candidate).exists() {
                include_paths.push(candidate.to_string());
            }
        }

        let opts = nvrtc::CompileOptions {
            arch: Some(arch),
            include_paths,
            ..Default::default()
        };
        let ptx = nvrtc::compile_ptx_with_opts(source, opts)
            .expect("Failed to compile CUDA kernel with arch");
        let module = self.ctx.load_module(ptx).expect("Failed to load PTX");

        self.module_cache
            .borrow_mut()
            .insert(hash, Arc::clone(&module));
        module
    }

    /// Get a function from a module by name.
    fn get_func(&self, source: &str, fn_name: &str) -> (Arc<CudaModule>, CudaFunction) {
        let module = self.get_module(source);
        let func = module
            .load_function(fn_name)
            .expect("Failed to load CUDA function");
        (module, func)
    }

    /// Get a function from a module compiled with explicit arch.
    fn get_func_with_arch(
        &self,
        source: &str,
        fn_name: &str,
        arch: &'static str,
    ) -> (Arc<CudaModule>, CudaFunction) {
        let module = self.get_module_with_arch(source, arch);
        let func = module
            .load_function(fn_name)
            .expect("Failed to load CUDA function");
        (module, func)
    }

    /// Allocate a zero-initialized f32 buffer regardless of mixed_precision mode.
    /// Used for gradient accumulation buffers that need f32 precision (atomicAdd targets).
    fn alloc_f32(&self, len: usize) -> CudaBuffer {
        self.pool_alloc_f32(len)
    }

    /// Upload data as f32 regardless of mixed_precision mode.
    /// Used for elementwise kernels and other f32-only operations.
    fn upload_f32(&self, data: &[f32]) -> CudaBuffer {
        let slice = self.stream.memcpy_stod(data).unwrap();
        Self::make_buf_unpooled(CudaStorage::F32(slice), data.len())
    }

    /// Extract columns [col_start, col_start + col_count) from a [batch, total_cols] matrix.
    /// Returns a contiguous [batch, col_count] buffer. Supports both f32 and bf16.
    pub fn extract_columns(
        &self,
        buf: &CudaBuffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> CudaBuffer {
        let total = batch * col_count;
        let tg = 256;
        let cfg = LaunchConfig {
            block_dim: (tg as u32, 1, 1),
            grid_dim: (((total + tg - 1) / tg) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_u32 = batch as u32;
        let total_cols_u32 = total_cols as u32;
        let col_start_u32 = col_start as u32;
        let col_count_u32 = col_count as u32;

        if buf.is_bf16() {
            let (_module, func) =
                self.get_func(util_cuda::EXTRACT_COLUMNS_BF16_CUDA, "extract_columns_bf16");
            let mut out = self.pool_alloc_uninit_bf16(total);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(buf.bf16_data())
                    .arg(out.bf16_data_mut())
                    .arg(&batch_u32)
                    .arg(&total_cols_u32)
                    .arg(&col_start_u32)
                    .arg(&col_count_u32)
                    .launch(cfg)
                    .unwrap();
            }
            out
        } else {
            let (_module, func) = self.get_func(util_cuda::EXTRACT_COLUMNS_CUDA, "extract_columns");
            let mut out = self.pool_alloc_uninit_f32(total);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(buf.f32_data())
                    .arg(out.f32_data_mut())
                    .arg(&batch_u32)
                    .arg(&total_cols_u32)
                    .arg(&col_start_u32)
                    .arg(&col_count_u32)
                    .launch(cfg)
                    .unwrap();
            }
            out
        }
    }

    /// Write `src[batch, col_count]` into columns [col_start, col_start + col_count) of `dst[batch, total_cols]`.
    /// Inverse of `extract_columns`. Supports both f32 and bf16.
    pub fn concat_columns(
        &self,
        dst: &mut CudaBuffer,
        src: &CudaBuffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let total = batch * col_count;
        let tg = 256;
        let cfg = LaunchConfig {
            block_dim: (tg as u32, 1, 1),
            grid_dim: (((total + tg - 1) / tg) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_u32 = batch as u32;
        let total_cols_u32 = total_cols as u32;
        let col_start_u32 = col_start as u32;
        let col_count_u32 = col_count as u32;

        if src.is_bf16() {
            let (_module, func) =
                self.get_func(util_cuda::CONCAT_COLUMNS_BF16_CUDA, "concat_columns_bf16");
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(src.bf16_data())
                    .arg(dst.bf16_data_mut())
                    .arg(&batch_u32)
                    .arg(&total_cols_u32)
                    .arg(&col_start_u32)
                    .arg(&col_count_u32)
                    .launch(cfg)
                    .unwrap();
            }
        } else {
            let (_module, func) = self.get_func(util_cuda::CONCAT_COLUMNS_CUDA, "concat_columns");
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(src.f32_data())
                    .arg(dst.f32_data_mut())
                    .arg(&batch_u32)
                    .arg(&total_cols_u32)
                    .arg(&col_start_u32)
                    .arg(&col_count_u32)
                    .launch(cfg)
                    .unwrap();
            }
        }
    }

    /// Fused residual add + RMS normalization.
    /// Computes `rms_norm(input + residual, weight, eps)`.
    /// Returns `(normed_output, pre_norm_sum)` — the sum is needed for backward.
    pub fn rms_norm_residual(
        &self,
        input: &CudaBuffer,
        residual: &CudaBuffer,
        weight: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (CudaBuffer, CudaBuffer) {
        let tg = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if input.is_bf16() && residual.is_bf16() && weight.is_bf16() {
            // Native bf16 path: single kernel, no conversions
            let (_module, func) = self.get_func(
                util_cuda::RMS_NORM_RESIDUAL_BF16_CUDA,
                "rms_norm_residual_bf16",
            );
            let mut output = self.pool_alloc_uninit_bf16(n_groups * dim);
            let mut sum_out = self.pool_alloc_uninit_bf16(n_groups * dim);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(residual.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(sum_out.bf16_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }
            (output, sum_out)
        } else {
            // Fall back to f32 path for mixed or pure f32 types
            let any_bf16 = input.is_bf16() || residual.is_bf16() || weight.is_bf16();
            let inp_conv = if input.is_bf16() {
                Some(self.convert_bf16_to_f32(input))
            } else {
                None
            };
            let res_conv = if residual.is_bf16() {
                Some(self.convert_bf16_to_f32(residual))
            } else {
                None
            };
            let wt_conv = if weight.is_bf16() {
                Some(self.convert_bf16_to_f32(weight))
            } else {
                None
            };
            let inp_ref = inp_conv.as_ref().unwrap_or(input);
            let res_ref = res_conv.as_ref().unwrap_or(residual);
            let wt_ref = wt_conv.as_ref().unwrap_or(weight);

            let (_module, func) =
                self.get_func(util_cuda::RMS_NORM_RESIDUAL_CUDA, "rms_norm_residual");
            let mut output = self.pool_alloc_uninit_f32(n_groups * dim);
            let mut sum_out = self.pool_alloc_uninit_f32(n_groups * dim);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(inp_ref.f32_data())
                    .arg(res_ref.f32_data())
                    .arg(wt_ref.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(sum_out.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }
            if any_bf16 {
                (
                    self.convert_f32_to_bf16(&output),
                    self.convert_f32_to_bf16(&sum_out),
                )
            } else {
                (output, sum_out)
            }
        }
    }

    /// Convert a bf16 buffer to f32 entirely on GPU (no CPU round-trip).
    pub fn convert_bf16_to_f32(&self, buf: &CudaBuffer) -> CudaBuffer {
        if !buf.is_bf16() {
            return self.copy_buffer_impl(buf);
        }
        let n = buf.len;
        let tg = 256usize;
        let cfg = LaunchConfig {
            block_dim: (tg as u32, 1, 1),
            grid_dim: (((n + tg - 1) / tg) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(util_cuda::BF16_TO_F32_CUDA, "bf16_to_f32");
        let mut out = self.pool_alloc_uninit_f32(n);
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(buf.bf16_data())
                .arg(out.f32_data_mut())
                .arg(&n_u32)
                .launch(cfg)
                .unwrap();
        }
        out
    }

    /// Convert a f32 buffer to bf16 entirely on GPU (no CPU round-trip).
    pub fn convert_f32_to_bf16(&self, buf: &CudaBuffer) -> CudaBuffer {
        if buf.is_bf16() {
            return self.copy_buffer_impl(buf);
        }
        let n = buf.len;
        let tg = 256usize;
        let cfg = LaunchConfig {
            block_dim: (tg as u32, 1, 1),
            grid_dim: (((n + tg - 1) / tg) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(util_cuda::F32_TO_BF16_CUDA, "f32_to_bf16");
        let mut out = self.pool_alloc_uninit_bf16(n);
        let n_u32 = n as u32;
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(buf.f32_data())
                .arg(out.bf16_data_mut())
                .arg(&n_u32)
                .launch(cfg)
                .unwrap();
        }
        out
    }

    /// Internal copy_buffer without going through the trait.
    fn copy_buffer_impl(&self, src: &CudaBuffer) -> CudaBuffer {
        match src.storage() {
            CudaStorage::F32(s) => {
                let mut out = self.pool_alloc_uninit_f32(src.len);
                self.stream.memcpy_dtod(s, out.f32_data_mut()).unwrap();
                out
            }
            CudaStorage::Bf16(s) => {
                let mut out = self.pool_alloc_uninit_bf16(src.len);
                self.stream.memcpy_dtod(s, out.bf16_data_mut()).unwrap();
                out
            }
            CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("4-bit weights can't be copied"),
        }
    }

    /// Slice a contiguous sub-range from a buffer (GPU-side copy, no CPU round-trip).
    /// Returns a new buffer with `len` elements starting at `offset`.
    pub fn slice_buffer(&self, buf: &CudaBuffer, offset: usize, len: usize) -> CudaBuffer {
        assert!(offset + len <= buf.len, "slice out of bounds");
        match buf.storage() {
            CudaStorage::F32(s) => {
                let view = s.try_slice(offset..offset + len).unwrap();
                let mut out = self.pool_alloc_uninit_f32(len);
                self.stream.memcpy_dtod(&view, out.f32_data_mut()).unwrap();
                out
            }
            CudaStorage::Bf16(s) => {
                let view = s.try_slice(offset..offset + len).unwrap();
                let mut out = self.pool_alloc_uninit_bf16(len);
                self.stream.memcpy_dtod(&view, out.bf16_data_mut()).unwrap();
                out
            }
            CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("4-bit weights can't be sliced"),
        }
    }

    /// Write `src` into `dst` starting at `offset` (GPU-side copy).
    pub fn write_into(&self, dst: &mut CudaBuffer, offset: usize, src: &CudaBuffer) {
        assert!(offset + src.len <= dst.len, "write_into out of bounds");
        let src_len = src.len;
        match (dst.storage_mut(), src.storage()) {
            (CudaStorage::F32(d), CudaStorage::F32(s)) => {
                let mut view = d.try_slice_mut(offset..offset + src_len).unwrap();
                self.stream.memcpy_dtod(s, &mut view).unwrap();
            }
            (CudaStorage::Bf16(d), CudaStorage::Bf16(s)) => {
                let mut view = d.try_slice_mut(offset..offset + src_len).unwrap();
                self.stream.memcpy_dtod(s, &mut view).unwrap();
            }
            // Across formats (a bf16 KV cache written from f32 rows, or read back): convert.
            (CudaStorage::Bf16(_), CudaStorage::F32(_)) => {
                let conv = self.convert_f32_to_bf16(src);
                self.write_into(dst, offset, &conv);
            }
            (CudaStorage::F32(_), CudaStorage::Bf16(_)) => {
                let conv = self.convert_bf16_to_f32(src);
                self.write_into(dst, offset, &conv);
            }
            _ => panic!("write_into: mismatched storage types"),
        }
    }

    /// Tensor-core flash attention (bf16 native, wmma m16n16k16).
    /// Used for both single and batched causal attention when inputs are bf16.
    fn causal_attention_tc(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        let total_dim = n_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let batch_size_u32 = batch_size as u32;

        let q_tiles = (seq_len + 15) / 16;

        // Shared memory: Q_smem + KV_smem (bf16) + S + O_acc + row_m + row_l (f32) + P_bf16 + S_warp
        let smem_bytes = (16 * head_dim) * 2       // Q_smem bf16
            + (16 * head_dim) * 2     // KV_smem bf16
            + (16 * 16) * 4           // S f32
            + (16 * head_dim) * 4     // O_acc f32
            + 16 * 4                  // row_m f32
            + 16 * 4                  // row_l f32
            + (16 * 16) * 2           // P_bf16
            + 4 * (16 * 16) * 4; // S_warp[4][256] f32

        let cfg = LaunchConfig {
            block_dim: (128, 1, 1),
            grid_dim: (q_tiles as u32, n_heads_u32, batch_size_u32),
            shared_mem_bytes: smem_bytes as u32,
        };

        let (_module, func) = self.get_func_with_arch(
            attention_cuda::CAUSAL_ATTENTION_FLASH_TC_CUDA,
            "causal_attention_flash_tc",
            "sm_86",
        );

        let mut output = self.pool_alloc_uninit_bf16(batch_size * seq_len * total_dim);

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(q.bf16_data())
                .arg(k.bf16_data())
                .arg(v.bf16_data())
                .arg(output.bf16_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size_u32)
                .launch(cfg)
                .unwrap();
        }

        output
    }

    /// Tensor-core flash attention backward (bf16 native, wmma m16n16k16).
    /// Used for both single and batched causal attention backward when inputs are bf16.
    /// Gradient outputs are f32 for precision.
    fn causal_attention_backward_tc(
        &self,
        grad_output: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        output: &CudaBuffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CudaBuffer, CudaBuffer, CudaBuffer) {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let batch_size_u32 = batch_size as u32;

        // Step 1: Precompute D[i,h] = sum_d(dO[i,d] * O[i,d]) — bf16 native
        let d_tg = std::cmp::min(head_dim, 256).next_power_of_two();
        let d_cfg = LaunchConfig {
            block_dim: (d_tg as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, batch_size_u32),
            shared_mem_bytes: (d_tg * 4) as u32,
        };
        let (_module, d_func) = self.get_func_with_arch(
            backward_cuda::FLASH_ATTN_BWD_PRECOMPUTE_D_BF16_CUDA,
            "flash_attn_bwd_precompute_d_bf16",
            "sm_86",
        );
        let mut d_buf = self.pool_alloc_uninit_f32(batch_size * seq_len * n_heads);
        unsafe {
            self.stream
                .launch_builder(&d_func)
                .arg(grad_output.bf16_data())
                .arg(output.bf16_data())
                .arg(d_buf.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size_u32)
                .launch(d_cfg)
                .unwrap();
        }

        // Step 2: TC flash backward kernel
        let tile_kv: usize = 16; // must match TC_BWD_TILE_KV in kernel
        let n_kv_tiles = (seq_len + tile_kv - 1) / tile_kv;
        let q_split: usize = 8;
        let bwd_tg: usize = 128;

        // Shared memory: K_smem + V_smem + Q_smem + dO_smem (bf16)
        //              + S + dK_acc + dV_acc + row_m + row_l + D_cache (f32)
        //              + P_bf16 + dS_bf16 (bf16)
        //              + S_warp[4][256] (f32)
        let smem_bytes = (tile_kv * head_dim) * 2         // K_smem bf16
            + (tile_kv * head_dim) * 2       // V_smem bf16
            + (16 * head_dim) * 2            // Q_smem bf16
            + (16 * head_dim) * 2            // dO_smem bf16
            + (16 * 16) * 4                  // S f32
            + (tile_kv * head_dim) * 4       // dK_acc f32
            + (tile_kv * head_dim) * 4       // dV_acc f32
            + 16 * 4                         // row_m f32
            + 16 * 4                         // row_l f32
            + 16 * 4                         // D_cache f32
            + (16 * 16) * 2                  // P_bf16
            + (16 * 16) * 2                  // dS_bf16
            + 4 * (16 * 16) * 4; // S_warp[4][256] f32

        let bwd_cfg = LaunchConfig {
            block_dim: (bwd_tg as u32, 1, 1),
            grid_dim: (
                (n_kv_tiles * batch_size) as u32,
                n_heads as u32,
                q_split as u32,
            ),
            shared_mem_bytes: smem_bytes as u32,
        };
        let (_module, bwd_func) = self.get_func_with_arch(
            backward_cuda::CAUSAL_ATTENTION_BACKWARD_FLASH_TC_CUDA,
            "causal_attention_backward_flash_tc",
            "sm_86",
        );

        // Gradient outputs are f32 (atomicAdd targets)
        let mut grad_q = self.alloc_f32(batch_size * seq_len * total_dim);
        let mut grad_k = self.alloc_f32(batch_size * seq_len * kv_dim);
        let mut grad_v = self.alloc_f32(batch_size * seq_len * kv_dim);

        let q_split_u32 = q_split as u32;
        let n_kv_tiles_u32 = n_kv_tiles as u32;
        unsafe {
            self.stream
                .launch_builder(&bwd_func)
                .arg(grad_output.bf16_data())
                .arg(q.bf16_data())
                .arg(k.bf16_data())
                .arg(v.bf16_data())
                .arg(d_buf.f32_data())
                .arg(grad_q.f32_data_mut())
                .arg(grad_k.f32_data_mut())
                .arg(grad_v.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&q_split_u32)
                .arg(&batch_size_u32)
                .arg(&n_kv_tiles_u32)
                .launch(bwd_cfg)
                .unwrap();
        }

        (grad_q, grad_k, grad_v)
    }

    /// Graph-capturable CE loss: writes loss to pre-allocated device buffer, no CPU download.
    /// Returns grad buffer. Caller reads loss_buf after graph completes.
    pub fn cross_entropy_fwd_bwd_graph(
        &self,
        logits: &CudaBuffer,
        targets: &CudaBuffer,
        loss_buf: &mut CudaBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
        non_pad_count: u32,
    ) -> CudaBuffer {
        let tg_size = std::cmp::min(vocab_size, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_positions as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_positions_u32 = n_positions as u32;
        let vocab_size_u32 = vocab_size as u32;

        // Zero loss accumulator (graph-safe: async memset)
        self.stream.memset_zeros(loss_buf.f32_data_mut()).unwrap();

        if logits.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::CROSS_ENTROPY_BF16_CUDA,
                "cross_entropy_fwd_bwd_bf16",
            );
            let mut grad = self.alloc_f32(n_positions * vocab_size);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.bf16_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&non_pad_count)
                    .launch(cfg)
                    .unwrap();
            }
            grad
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::CROSS_ENTROPY_CUDA, "cross_entropy_fwd_bwd");
            let mut grad = self.alloc_f32(n_positions * vocab_size);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.f32_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&non_pad_count)
                    .launch(cfg)
                    .unwrap();
            }
            grad
        }
    }
}

impl ComputeDevice for CudaComputeDevice {
    fn bf16_storage_bytes(&self) -> usize {
        2
    }
    type Buffer = CudaBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::Cuda
    }

    fn peak_flops_f32(&self) -> Option<f64> {
        // Query SM count and clock from CUDA driver
        use cudarc::driver::sys;
        let mut sm_count: i32 = 0;
        let mut clock_khz: i32 = 0;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut sm_count,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            );
            sys::cuDeviceGetAttribute(
                &mut clock_khz,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                0,
            );
            sys::cuDeviceGetAttribute(
                &mut major,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                0,
            );
            sys::cuDeviceGetAttribute(
                &mut minor,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                0,
            );
        }
        if sm_count == 0 || clock_khz == 0 {
            return None;
        }
        // FP32 cores per SM by compute capability
        let fp32_per_sm = match (major, minor) {
            (7, 0) => 64,   // V100
            (7, 5) => 64,   // Turing (RTX 2xxx)
            (8, 0) => 64,   // A100
            (8, 6) => 128,  // Ampere consumer (RTX 3xxx)
            (8, 9) => 128,  // Ada Lovelace (RTX 4xxx)
            (9, 0) => 128,  // Hopper (H100)
            (10, 0) => 128, // Blackwell
            _ => 64,        // conservative default
        };
        // Peak FLOPS = SM_count × FP32_per_SM × 2 (FMA) × clock_Hz
        let clock_hz = clock_khz as f64 * 1000.0;
        Some(sm_count as f64 * fp32_per_sm as f64 * 2.0 * clock_hz)
    }

    fn total_memory_bytes(&self) -> usize {
        cudarc::driver::result::mem_get_info()
            .map(|(_, total)| total)
            .unwrap_or(0)
    }

    fn free_memory_bytes(&self) -> usize {
        cudarc::driver::result::mem_get_info()
            .map(|(free, _)| free)
            .unwrap_or(0)
    }

    fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_default()
    }

    fn arch(&self) -> String {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let cc = |a| self.ctx.attribute(a).unwrap_or(0);
        format!(
            "sm_{}{}",
            cc(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            cc(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        )
    }

    fn pool_clear(&self) {
        CudaComputeDevice::pool_clear(self);
    }

    fn upload(&self, data: &[f32]) -> CudaBuffer {
        if self.mixed_precision {
            let bf16_data: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
            let slice = self.stream.memcpy_stod(&bf16_data).unwrap();
            Self::make_buf_unpooled(CudaStorage::Bf16(slice), data.len())
        } else {
            let slice = self.stream.memcpy_stod(data).unwrap();
            Self::make_buf_unpooled(CudaStorage::F32(slice), data.len())
        }
    }

    fn upload_u32(&self, data: &[u32]) -> CudaBuffer {
        // Always f32 — these are integer IDs stored as f32 bit patterns, not learnable weights
        let f32_data: Vec<f32> = data.iter().map(|&x| f32::from_bits(x)).collect();
        self.upload_f32(&f32_data)
    }

    fn alloc(&self, len: usize) -> CudaBuffer {
        if self.mixed_precision {
            self.pool_alloc_bf16(len)
        } else {
            self.pool_alloc_f32(len)
        }
    }

    fn upload_f32(&self, data: &[f32]) -> CudaBuffer {
        CudaComputeDevice::upload_f32(self, data)
    }

    fn alloc_f32(&self, len: usize) -> CudaBuffer {
        self.pool_alloc_f32(len)
    }

    fn alloc_uninit(&self, len: usize) -> CudaBuffer {
        if self.mixed_precision {
            self.pool_alloc_uninit_bf16(len)
        } else {
            self.pool_alloc_uninit_f32(len)
        }
    }

    fn alloc_uninit_f32(&self, len: usize) -> CudaBuffer {
        self.pool_alloc_uninit_f32(len)
    }

    fn download(&self, buf: &CudaBuffer) -> Vec<f32> {
        buf.to_vec()
    }

    fn upload_into_f32(&self, buf: &mut CudaBuffer, data: &[f32]) {
        CudaComputeDevice::upload_into_f32(self, buf, data)
    }

    fn upload_into_u32(&self, buf: &mut CudaBuffer, data: &[u32]) {
        CudaComputeDevice::upload_into_u32(self, buf, data)
    }

    fn elementwise(
        &self,
        inputs: &[&CudaBuffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> CudaBuffer {
        let n_inputs = inputs.len();

        let (graph, output) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });
        let kernel = graph.to_kernel(&[output], n_inputs, Dialect::Cuda);
        let (_module, func) = self.get_func(&kernel.source, kernel.entry_point);

        // If inputs are bf16, convert to f32 on GPU (no CPU round-trip).
        let converted: Vec<CudaBuffer>;
        let f32_inputs: Vec<&CudaSlice<f32>> = if self.mixed_precision {
            converted = inputs
                .iter()
                .map(|inp| self.convert_bf16_to_f32(inp))
                .collect();
            converted.iter().map(|b| b.f32_data()).collect()
        } else {
            inputs.iter().map(|inp| inp.f32_data()).collect()
        };

        let mut output_buf = self.pool_alloc_uninit_f32(numel);
        let count = numel as u32;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((numel as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        // Pass each input as a separate kernel argument (matches the generated kernel signature)
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            for slice in &f32_inputs {
                builder.arg(*slice);
            }
            builder
                .arg(output_buf.f32_data_mut())
                .arg(&count)
                .launch(cfg)
                .unwrap();
        }

        // Convert output to bf16 if mixed precision (GPU-side, no CPU round-trip)
        if self.mixed_precision {
            self.convert_f32_to_bf16(&output_buf)
        } else {
            output_buf
        }
    }

    fn matmul(&self, a: &CudaBuffer, b: &CudaBuffer, m: usize, k: usize, n: usize) -> CudaBuffer {
        // cuBLAS is column-major. For row-major C[m,n] = A[m,k] * B[k,n]:
        //   C_col^T = B_col^T * A_col^T
        // So we call gemm with swapped A/B and cublas m=n, n=m.
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        // Handle mixed types: convert f32 operand to bf16, use tensor core gemm_ex
        if a.is_bf16() != b.is_bf16() {
            let a_conv = if !a.is_bf16() {
                Some(self.convert_f32_to_bf16(a))
            } else {
                None
            };
            let b_conv = if !b.is_bf16() {
                Some(self.convert_f32_to_bf16(b))
            } else {
                None
            };
            let a_ref = a_conv.as_ref().unwrap_or(a);
            let b_ref = b_conv.as_ref().unwrap_or(b);
            return self.matmul(a_ref, b_ref, m, k, n);
        }

        if a.is_bf16() {
            let mut output = self.pool_alloc_uninit_bf16(m * n);

            // Use gemm_ex for bf16 (stored as u16) with f32 compute
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = output.bf16_data_mut().device_ptr_mut(&self.stream);

                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m, // lda = n (row stride of B)
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k, // ldb = k (row stride of A)
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m, // ldc = n (row stride of C)
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex failed");
                }
            } // drop _rec_c borrow here

            output
        } else {
            let mut output = self.pool_alloc_uninit_f32(m * n);

            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_N,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: cublas_m,
                            n: cublas_n,
                            k: cublas_k,
                            alpha: 1.0f32,
                            lda: cublas_m, // n (row stride of B)
                            ldb: cublas_k, // k (row stride of A)
                            beta: 0.0f32,
                            ldc: cublas_m, // n (row stride of C)
                        },
                        b.f32_data(),
                        a.f32_data(),
                        output.f32_data_mut(),
                    )
                    .expect("cuBLAS sgemm failed");
            }

            output
        }
    }

    fn softmax(&self, data: &CudaBuffer, n_rows: usize, row_len: usize) -> CudaBuffer {
        let tg_size = std::cmp::min(row_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_rows as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_u32 = n_rows as u32;
        let row_len_u32 = row_len as u32;

        if data.is_bf16() {
            let (_module, func) = self.get_func(reduce_cuda::SOFTMAX_BF16_CUDA, "softmax_bf16");
            let mut output = self.pool_alloc_uninit_bf16(data.len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(reduce_cuda::SOFTMAX_CUDA, "softmax");
            let mut output = self.pool_alloc_uninit_f32(data.len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn rms_norm(
        &self,
        data: &CudaBuffer,
        weight: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if data.is_bf16() {
            let (_module, func) = self.get_func(reduce_cuda::RMS_NORM_BF16_CUDA, "rms_norm_bf16");
            let mut output = self.pool_alloc_uninit_bf16(data.len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let v4 = dim % 4 == 0;
            let name = if v4 { "rms_norm_v4" } else { "rms_norm" };
            let (_module, func) = self.get_func(reduce_cuda::RMS_NORM_CUDA, name);
            let mut output = self.pool_alloc_uninit_f32(data.len);
            let cfg = if v4 {
                // A float4 a thread, whole warps, at most 1024: fixed by dim alone.
                let threads = (dim / 4).div_ceil(32).clamp(1, 32) * 32;
                LaunchConfig {
                    block_dim: (threads as u32, 1, 1),
                    ..cfg
                }
            } else {
                cfg
            };

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(weight.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn embedding(
        &self,
        weight: &CudaBuffer,
        ids: &CudaBuffer,
        seq_len: usize,
        dim: usize,
    ) -> CudaBuffer {
        if let Some(out) = self.embedding_llm(weight, ids, seq_len, dim) {
            return out;
        }
        let total = seq_len * dim;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u32 = seq_len as u32;
        let dim_u32 = dim as u32;

        if weight.is_bf16() {
            let (_module, func) = self.get_func(
                reduce_cuda::EMBEDDING_GATHER_BF16_CUDA,
                "embedding_gather_bf16",
            );
            let mut output = self.pool_alloc_uninit_bf16(total);

            // ids is always f32 storage (u32 bit patterns) — cast to unsigned int* on GPU side
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(weight.bf16_data())
                    .arg(ids.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) =
                self.get_func(reduce_cuda::EMBEDDING_GATHER_CUDA, "embedding_gather");
            let mut output = self.pool_alloc_uninit_f32(total);

            // ids is always f32 storage (u32 bit patterns) — cast to unsigned int* on GPU side
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(weight.f32_data())
                    .arg(ids.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn reduce_sum(&self, data: &CudaBuffer, shape: &[usize], axis: usize) -> CudaBuffer {
        let ndim = shape.len();
        assert!(axis < ndim);

        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let out_len = outer * inner;

        if out_len == 0 {
            return self.alloc(0);
        }

        let tg_size = std::cmp::min(axis_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (out_len as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let outer_u32 = outer as u32;
        let axis_len_u32 = axis_len as u32;
        let inner_u32 = inner as u32;

        if data.is_bf16() {
            let (_module, func) =
                self.get_func(reduce_cuda::REDUCE_SUM_BF16_CUDA, "reduce_sum_bf16");
            let mut output = self.pool_alloc_uninit_bf16(out_len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(reduce_cuda::REDUCE_SUM_CUDA, "reduce_sum");
            let mut output = self.pool_alloc_uninit_f32(out_len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn causal_attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        let total_dim = n_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        // Use tensor core kernel for bf16 inputs with head_dim multiple of 16
        if q.is_bf16() && head_dim % 16 == 0 {
            return self.causal_attention_tc(q, k, v, seq_len, 1, n_heads, n_kv_heads, head_dim);
        }

        // f32 fallback: convert bf16 if needed
        let q_conv = if q.is_bf16() {
            Some(self.convert_bf16_to_f32(q))
        } else {
            None
        };
        let k_conv = if k.is_bf16() {
            Some(self.convert_bf16_to_f32(k))
        } else {
            None
        };
        let v_conv = if v.is_bf16() {
            Some(self.convert_bf16_to_f32(v))
        } else {
            None
        };
        let q_ref = q_conv.as_ref().unwrap_or(q);
        let k_ref = k_conv.as_ref().unwrap_or(k);
        let v_ref = v_conv.as_ref().unwrap_or(v);
        let return_bf16 = q.is_bf16();

        let tile_kv: usize = 64;
        let smem_bytes = (tile_kv + head_dim + head_dim) * 4;
        let batch_size: u32 = 1;
        let cfg = LaunchConfig {
            block_dim: (head_dim as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, batch_size),
            shared_mem_bytes: smem_bytes as u32,
        };
        let (_module, func) = self.get_func(
            attention_cuda::CAUSAL_ATTENTION_FLASH_CUDA,
            "causal_attention_flash",
        );
        let mut output = self.pool_alloc_uninit_f32(seq_len * total_dim);

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(q_ref.f32_data())
                .arg(k_ref.f32_data())
                .arg(v_ref.f32_data())
                .arg(output.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size)
                .launch(cfg)
                .unwrap();
        }

        if return_bf16 {
            self.convert_f32_to_bf16(&output)
        } else {
            output
        }
    }

    fn kv_attention(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        let total_dim = n_heads * head_dim;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        if q_len == 1 {
            // Single-query decode
            let total_len = cache_start + 1;
            let tg_size = std::cmp::min(total_len, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (n_heads as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_len_u32 = total_len as u32;

            if q.is_bf16() {
                let (_module, func) =
                    self.get_func(attention_cuda::KV_ATTENTION_BF16_CUDA, "kv_attention_bf16");
                let mut output = self.pool_alloc_uninit_bf16(total_dim);

                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(q.bf16_data())
                        .arg(k_cache.bf16_data())
                        .arg(v_cache.bf16_data())
                        .arg(output.bf16_data_mut())
                        .arg(&total_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            } else {
                let (_module, func) =
                    self.get_func(attention_cuda::KV_ATTENTION_CUDA, "kv_attention");
                let mut output = self.pool_alloc_uninit_f32(total_dim);

                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(q.f32_data())
                        .arg(k_cache.f32_data())
                        .arg(v_cache.f32_data())
                        .arg(output.f32_data_mut())
                        .arg(&total_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            }
        } else {
            // Batched prefill
            let max_attend = cache_start + q_len;
            let tg_size = std::cmp::min(max_attend, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (q_len as u32, n_heads as u32, 1),
                shared_mem_bytes: 0,
            };
            let cache_start_u32 = cache_start as u32;
            let q_len_u32 = q_len as u32;

            if q.is_bf16() {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_PREFILL_BF16_CUDA,
                    "kv_attention_prefill_bf16",
                );
                let mut output = self.pool_alloc_uninit_bf16(q_len * total_dim);

                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(q.bf16_data())
                        .arg(k_cache.bf16_data())
                        .arg(v_cache.bf16_data())
                        .arg(output.bf16_data_mut())
                        .arg(&cache_start_u32)
                        .arg(&q_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            } else {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_PREFILL_CUDA,
                    "kv_attention_prefill",
                );
                let mut output = self.pool_alloc_uninit_f32(q_len * total_dim);

                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(q.f32_data())
                        .arg(k_cache.f32_data())
                        .arg(v_cache.f32_data())
                        .arg(output.f32_data_mut())
                        .arg(&cache_start_u32)
                        .arg(&q_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            }
        }
    }

    fn transpose_2d(&self, buf: &CudaBuffer, rows: usize, cols: usize) -> CudaBuffer {
        let cfg = LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim: (((cols as u32) + 15) / 16, ((rows as u32) + 15) / 16, 1),
            shared_mem_bytes: 0,
        };
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        if buf.is_bf16() {
            let (_module, func) =
                self.get_func(backward_cuda::TRANSPOSE_2D_BF16_CUDA, "transpose_2d_bf16");
            let mut output = self.pool_alloc_uninit_bf16(rows * cols);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(buf.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(backward_cuda::TRANSPOSE_2D_CUDA, "transpose_2d");
            let mut output = self.pool_alloc_uninit_f32(rows * cols);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn softmax_backward(
        &self,
        softmax_out: &CudaBuffer,
        grad_output: &CudaBuffer,
        n_rows: usize,
        row_len: usize,
    ) -> CudaBuffer {
        let tg_size = std::cmp::min(row_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_rows as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_u32 = n_rows as u32;
        let row_len_u32 = row_len as u32;

        if softmax_out.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::SOFTMAX_BACKWARD_BF16_CUDA,
                "softmax_backward_bf16",
            );
            let mut output = self.pool_alloc_uninit_bf16(n_rows * row_len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(softmax_out.bf16_data())
                    .arg(grad_output.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::SOFTMAX_BACKWARD_CUDA, "softmax_backward");
            let mut output = self.pool_alloc_uninit_f32(n_rows * row_len);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(softmax_out.f32_data())
                    .arg(grad_output.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn rms_norm_backward(
        &self,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (CudaBuffer, CudaBuffer) {
        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if input.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::RMS_NORM_BACKWARD_BF16_CUDA,
                "rms_norm_backward_bf16",
            );
            let mut grad_input = self.pool_alloc_uninit_bf16(n_groups * dim); // bf16, fully written by kernel
            let mut grad_weight = self.alloc_f32(dim); // f32 for atomic accumulation — MUST be zero

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(grad_output.bf16_data())
                    .arg(grad_input.bf16_data_mut())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            (grad_input, grad_weight)
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::RMS_NORM_BACKWARD_CUDA, "rms_norm_backward");
            let mut grad_input = self.pool_alloc_uninit_f32(n_groups * dim); // fully written by kernel
            let mut grad_weight = self.alloc_f32(dim); // atomicAdd target — MUST be zero

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.f32_data())
                    .arg(weight.f32_data())
                    .arg(grad_output.f32_data())
                    .arg(grad_input.f32_data_mut())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            (grad_input, grad_weight)
        }
    }

    fn embedding_backward(
        &self,
        grad_output: &CudaBuffer,
        ids: &CudaBuffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> CudaBuffer {
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((seq_len as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let vocab_size_u32 = vocab_size as u32;
        let seq_len_u32 = seq_len as u32;
        let dim_u32 = dim as u32;

        // grad_weight is always f32 (atomicAdd accumulator)
        let mut grad_weight = self.alloc_f32(vocab_size * dim);

        if grad_output.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::EMBEDDING_BACKWARD_BF16_CUDA,
                "embedding_backward_bf16",
            );

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(ids.f32_data())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&vocab_size_u32)
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::EMBEDDING_BACKWARD_CUDA, "embedding_backward");

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.f32_data())
                    .arg(ids.f32_data())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&vocab_size_u32)
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }
        }

        grad_weight
    }

    fn causal_attention_backward(
        &self,
        grad_output: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CudaBuffer, CudaBuffer, CudaBuffer) {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let tg_size = std::cmp::min(head_dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        // For mixed precision: use TC backward when bf16 + head_dim%16==0,
        // otherwise convert to f32 and use scalar flash backward.
        let any_bf16 = q.is_bf16() || k.is_bf16() || v.is_bf16() || grad_output.is_bf16();

        if any_bf16 && q.is_bf16() && head_dim % 16 == 0 {
            // TC path: recompute forward O (bf16), then TC precompute-D + TC backward
            let o_buf =
                self.causal_attention_tc(q, k, v, seq_len, 1, n_heads, n_kv_heads, head_dim);
            return self.causal_attention_backward_tc(
                grad_output,
                q,
                k,
                v,
                &o_buf,
                seq_len,
                1,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        } else if any_bf16 {
            let go_conv = if grad_output.is_bf16() {
                Some(self.convert_bf16_to_f32(grad_output))
            } else {
                None
            };
            let q_conv = if q.is_bf16() {
                Some(self.convert_bf16_to_f32(q))
            } else {
                None
            };
            let k_conv = if k.is_bf16() {
                Some(self.convert_bf16_to_f32(k))
            } else {
                None
            };
            let v_conv = if v.is_bf16() {
                Some(self.convert_bf16_to_f32(v))
            } else {
                None
            };
            let go_ref = go_conv.as_ref().unwrap_or(grad_output);
            let q_ref = q_conv.as_ref().unwrap_or(q);
            let k_ref = k_conv.as_ref().unwrap_or(k);
            let v_ref = v_conv.as_ref().unwrap_or(v);
            return self.causal_attention_backward(
                go_ref, q_ref, k_ref, v_ref, seq_len, n_heads, n_kv_heads, head_dim,
            );
        } else {
            // Check if user requested the simple (non-flash) backward kernel
            let use_simple_bwd = std::env::var("GAIA_SIMPLE_BWD")
                .map(|v| v == "1")
                .unwrap_or(false);

            if use_simple_bwd {
                // Simple backward: stores all scores in shared memory, no O recompute.
                // Slower but simpler — useful for debugging.
                let (_module, func) = self.get_func(
                    backward_cuda::CAUSAL_ATTENTION_BACKWARD_CUDA,
                    "causal_attention_backward",
                );
                // grad_K/V need atomicAdd accumulation across Q positions
                let mut grad_q = self.alloc_f32(seq_len * total_dim);
                let mut grad_k = self.alloc_f32(seq_len * kv_dim);
                let mut grad_v = self.alloc_f32(seq_len * kv_dim);
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(grad_output.f32_data())
                        .arg(q.f32_data())
                        .arg(k.f32_data())
                        .arg(v.f32_data())
                        .arg(grad_q.f32_data_mut())
                        .arg(grad_k.f32_data_mut())
                        .arg(grad_v.f32_data_mut())
                        .arg(&seq_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }
                (grad_q, grad_k, grad_v)
            } else {
                // FlashAttention-2 backward: recompute O, precompute D, then tiled backward.

                // Step 1: Recompute forward output O (flash attention trades compute for memory)
                let o_buf = self.causal_attention(q, k, v, seq_len, n_heads, n_kv_heads, head_dim);

                // Step 2: Precompute D[i,h] = sum_d(dO[i,d] * O[i,d])
                let batch_size_u32: u32 = 1;
                let d_tg = std::cmp::min(head_dim, 256).next_power_of_two();
                let d_cfg = LaunchConfig {
                    block_dim: (d_tg as u32, 1, 1),
                    grid_dim: (seq_len as u32, n_heads as u32, batch_size_u32),
                    shared_mem_bytes: (d_tg * 4) as u32,
                };
                let (_module, d_func) = self.get_func(
                    backward_cuda::FLASH_ATTN_BWD_PRECOMPUTE_D_CUDA,
                    "flash_attn_bwd_precompute_d",
                );
                let mut d_buf = self.pool_alloc_uninit_f32(seq_len * n_heads);
                unsafe {
                    self.stream
                        .launch_builder(&d_func)
                        .arg(grad_output.f32_data())
                        .arg(o_buf.f32_data())
                        .arg(d_buf.f32_data_mut())
                        .arg(&seq_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&batch_size_u32)
                        .launch(d_cfg)
                        .unwrap();
                }

                // Step 3: Flash backward kernel (v3 — Q/dO cached in shared memory)
                let tile_kv: usize = 32; // must match FA_BWD_TILE in kernel
                let n_kv_tiles = (seq_len + tile_kv - 1) / tile_kv;
                let q_split: usize = 16; // split Q loop across this many blocks
                let bwd_tg: usize = 128;
                // smem: dK_acc[T*D] + dV_acc[T*D] + scores[T] + reduce[tg] + Q_cache[D] + dO_cache[D]
                let smem_bytes = (2 * tile_kv * head_dim + tile_kv + bwd_tg + 2 * head_dim) * 4;
                let bwd_cfg = LaunchConfig {
                    block_dim: (bwd_tg as u32, 1, 1),
                    grid_dim: (
                        (n_kv_tiles as u32) * batch_size_u32,
                        n_heads as u32,
                        q_split as u32,
                    ),
                    shared_mem_bytes: smem_bytes as u32,
                };
                let (_module, bwd_func) = self.get_func(
                    backward_cuda::CAUSAL_ATTENTION_BACKWARD_FLASH_CUDA,
                    "causal_attention_backward_flash",
                );

                // All grads need zero-init (atomicAdd targets from multiple blocks)
                let mut grad_q = self.alloc_f32(seq_len * total_dim);
                let mut grad_k = self.alloc_f32(seq_len * kv_dim);
                let mut grad_v = self.alloc_f32(seq_len * kv_dim);

                let q_split_u32 = q_split as u32;
                let n_kv_tiles_u32 = n_kv_tiles as u32;
                unsafe {
                    self.stream
                        .launch_builder(&bwd_func)
                        .arg(grad_output.f32_data())
                        .arg(q.f32_data())
                        .arg(k.f32_data())
                        .arg(v.f32_data())
                        .arg(d_buf.f32_data())
                        .arg(grad_q.f32_data_mut())
                        .arg(grad_k.f32_data_mut())
                        .arg(grad_v.f32_data_mut())
                        .arg(&seq_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .arg(&q_split_u32)
                        .arg(&batch_size_u32)
                        .arg(&n_kv_tiles_u32)
                        .launch(bwd_cfg)
                        .unwrap();
                }

                (grad_q, grad_k, grad_v)
            }
        }
    }

    fn causal_attention_backward_with_output(
        &self,
        grad_output: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        output: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CudaBuffer, CudaBuffer, CudaBuffer) {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        // Convert bf16 inputs to f32 on GPU, or use TC path if eligible
        let any_bf16 =
            q.is_bf16() || k.is_bf16() || v.is_bf16() || grad_output.is_bf16() || output.is_bf16();
        if any_bf16 && q.is_bf16() && head_dim % 16 == 0 {
            return self.causal_attention_backward_tc(
                grad_output,
                q,
                k,
                v,
                output,
                seq_len,
                1,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        } else if any_bf16 {
            let go_conv = if grad_output.is_bf16() {
                Some(self.convert_bf16_to_f32(grad_output))
            } else {
                None
            };
            let q_conv = if q.is_bf16() {
                Some(self.convert_bf16_to_f32(q))
            } else {
                None
            };
            let k_conv = if k.is_bf16() {
                Some(self.convert_bf16_to_f32(k))
            } else {
                None
            };
            let v_conv = if v.is_bf16() {
                Some(self.convert_bf16_to_f32(v))
            } else {
                None
            };
            let o_conv = if output.is_bf16() {
                Some(self.convert_bf16_to_f32(output))
            } else {
                None
            };
            let go_ref = go_conv.as_ref().unwrap_or(grad_output);
            let q_ref = q_conv.as_ref().unwrap_or(q);
            let k_ref = k_conv.as_ref().unwrap_or(k);
            let v_ref = v_conv.as_ref().unwrap_or(v);
            let o_ref = o_conv.as_ref().unwrap_or(output);
            return self.causal_attention_backward_with_output(
                go_ref, q_ref, k_ref, v_ref, o_ref, seq_len, n_heads, n_kv_heads, head_dim,
            );
        }

        // Step 1: Precompute D[i,h] = sum_d(dO[i,d] * O[i,d]) using cached O
        let batch_size_u32: u32 = 1;
        let d_tg = std::cmp::min(head_dim, 256).next_power_of_two();
        let d_cfg = LaunchConfig {
            block_dim: (d_tg as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, batch_size_u32),
            shared_mem_bytes: (d_tg * 4) as u32,
        };
        let (_module, d_func) = self.get_func(
            backward_cuda::FLASH_ATTN_BWD_PRECOMPUTE_D_CUDA,
            "flash_attn_bwd_precompute_d",
        );
        let mut d_buf = self.pool_alloc_uninit_f32(seq_len * n_heads);
        unsafe {
            self.stream
                .launch_builder(&d_func)
                .arg(grad_output.f32_data())
                .arg(output.f32_data())
                .arg(d_buf.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size_u32)
                .launch(d_cfg)
                .unwrap();
        }

        // Step 2: Flash backward kernel
        let tile_kv: usize = 32;
        let n_kv_tiles = (seq_len + tile_kv - 1) / tile_kv;
        let q_split: usize = 16;
        let bwd_tg: usize = 128;
        let smem_bytes = (2 * tile_kv * head_dim + tile_kv + bwd_tg + 2 * head_dim) * 4;
        let bwd_cfg = LaunchConfig {
            block_dim: (bwd_tg as u32, 1, 1),
            grid_dim: (
                (n_kv_tiles as u32) * batch_size_u32,
                n_heads as u32,
                q_split as u32,
            ),
            shared_mem_bytes: smem_bytes as u32,
        };
        let (_module, bwd_func) = self.get_func(
            backward_cuda::CAUSAL_ATTENTION_BACKWARD_FLASH_CUDA,
            "causal_attention_backward_flash",
        );

        let mut grad_q = self.alloc_f32(seq_len * total_dim);
        let mut grad_k = self.alloc_f32(seq_len * kv_dim);
        let mut grad_v = self.alloc_f32(seq_len * kv_dim);

        let q_split_u32 = q_split as u32;
        let n_kv_tiles_u32 = n_kv_tiles as u32;
        unsafe {
            self.stream
                .launch_builder(&bwd_func)
                .arg(grad_output.f32_data())
                .arg(q.f32_data())
                .arg(k.f32_data())
                .arg(v.f32_data())
                .arg(d_buf.f32_data())
                .arg(grad_q.f32_data_mut())
                .arg(grad_k.f32_data_mut())
                .arg(grad_v.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&q_split_u32)
                .arg(&batch_size_u32)
                .arg(&n_kv_tiles_u32)
                .launch(bwd_cfg)
                .unwrap();
        }

        (grad_q, grad_k, grad_v)
    }

    fn batched_causal_attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        // Use tensor core kernel for bf16 inputs with head_dim multiple of 16
        if q.is_bf16() && head_dim % 16 == 0 {
            return self
                .causal_attention_tc(q, k, v, seq_len, batch_size, n_heads, n_kv_heads, head_dim);
        }

        if batch_size == 1 {
            return self.causal_attention(q, k, v, seq_len, n_heads, n_kv_heads, head_dim);
        }
        let total_dim = n_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let batch_size_u32 = batch_size as u32;

        // f32 path
        let q_conv = if q.is_bf16() {
            Some(self.convert_bf16_to_f32(q))
        } else {
            None
        };
        let k_conv = if k.is_bf16() {
            Some(self.convert_bf16_to_f32(k))
        } else {
            None
        };
        let v_conv = if v.is_bf16() {
            Some(self.convert_bf16_to_f32(v))
        } else {
            None
        };
        let q_ref = q_conv.as_ref().unwrap_or(q);
        let k_ref = k_conv.as_ref().unwrap_or(k);
        let v_ref = v_conv.as_ref().unwrap_or(v);
        let return_bf16 = q.is_bf16();

        let tile_kv: usize = 64;
        let smem_bytes = (tile_kv + head_dim + head_dim) * 4;
        let cfg = LaunchConfig {
            block_dim: (head_dim as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, batch_size_u32),
            shared_mem_bytes: smem_bytes as u32,
        };
        let (_module, func) = self.get_func(
            attention_cuda::CAUSAL_ATTENTION_FLASH_CUDA,
            "causal_attention_flash",
        );
        let mut output = self.pool_alloc_uninit_f32(batch_size * seq_len * total_dim);

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(q_ref.f32_data())
                .arg(k_ref.f32_data())
                .arg(v_ref.f32_data())
                .arg(output.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size_u32)
                .launch(cfg)
                .unwrap();
        }

        if return_bf16 {
            self.convert_f32_to_bf16(&output)
        } else {
            output
        }
    }

    fn batched_causal_attention_backward(
        &self,
        grad_output: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        output: &CudaBuffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CudaBuffer, CudaBuffer, CudaBuffer) {
        // Use TC backward for bf16 batched path
        if q.is_bf16() && head_dim % 16 == 0 {
            return self.causal_attention_backward_tc(
                grad_output,
                q,
                k,
                v,
                output,
                seq_len,
                batch_size,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }

        if batch_size == 1 {
            return self.causal_attention_backward_with_output(
                grad_output,
                q,
                k,
                v,
                output,
                seq_len,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let batch_size_u32 = batch_size as u32;

        // Convert bf16 to f32 on GPU
        let any_bf16 =
            q.is_bf16() || k.is_bf16() || v.is_bf16() || grad_output.is_bf16() || output.is_bf16();
        let go_conv = if any_bf16 && grad_output.is_bf16() {
            Some(self.convert_bf16_to_f32(grad_output))
        } else {
            None
        };
        let q_conv = if any_bf16 && q.is_bf16() {
            Some(self.convert_bf16_to_f32(q))
        } else {
            None
        };
        let k_conv = if any_bf16 && k.is_bf16() {
            Some(self.convert_bf16_to_f32(k))
        } else {
            None
        };
        let v_conv = if any_bf16 && v.is_bf16() {
            Some(self.convert_bf16_to_f32(v))
        } else {
            None
        };
        let o_conv = if any_bf16 && output.is_bf16() {
            Some(self.convert_bf16_to_f32(output))
        } else {
            None
        };
        let go_ref = go_conv.as_ref().unwrap_or(grad_output);
        let q_ref = q_conv.as_ref().unwrap_or(q);
        let k_ref = k_conv.as_ref().unwrap_or(k);
        let v_ref = v_conv.as_ref().unwrap_or(v);
        let o_ref = o_conv.as_ref().unwrap_or(output);

        // Step 1: Precompute D
        let d_tg = std::cmp::min(head_dim, 256).next_power_of_two();
        let d_cfg = LaunchConfig {
            block_dim: (d_tg as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, batch_size_u32),
            shared_mem_bytes: (d_tg * 4) as u32,
        };
        let (_module, d_func) = self.get_func(
            backward_cuda::FLASH_ATTN_BWD_PRECOMPUTE_D_CUDA,
            "flash_attn_bwd_precompute_d",
        );
        let mut d_buf = self.pool_alloc_uninit_f32(batch_size * seq_len * n_heads);
        unsafe {
            self.stream
                .launch_builder(&d_func)
                .arg(go_ref.f32_data())
                .arg(o_ref.f32_data())
                .arg(d_buf.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&head_dim_u32)
                .arg(&batch_size_u32)
                .launch(d_cfg)
                .unwrap();
        }

        // Step 2: Flash backward kernel (batch folded into gridDim.x)
        let tile_kv: usize = 32;
        let n_kv_tiles = (seq_len + tile_kv - 1) / tile_kv;
        let q_split: usize = 16;
        let bwd_tg: usize = 128;
        let smem_bytes = (2 * tile_kv * head_dim + tile_kv + bwd_tg + 2 * head_dim) * 4;
        let bwd_cfg = LaunchConfig {
            block_dim: (bwd_tg as u32, 1, 1),
            grid_dim: (
                (n_kv_tiles * batch_size) as u32,
                n_heads as u32,
                q_split as u32,
            ),
            shared_mem_bytes: smem_bytes as u32,
        };
        let (_module, bwd_func) = self.get_func(
            backward_cuda::CAUSAL_ATTENTION_BACKWARD_FLASH_CUDA,
            "causal_attention_backward_flash",
        );

        let mut grad_q = self.alloc_f32(batch_size * seq_len * total_dim);
        let mut grad_k = self.alloc_f32(batch_size * seq_len * kv_dim);
        let mut grad_v = self.alloc_f32(batch_size * seq_len * kv_dim);

        let q_split_u32 = q_split as u32;
        let n_kv_tiles_u32 = n_kv_tiles as u32;
        unsafe {
            self.stream
                .launch_builder(&bwd_func)
                .arg(go_ref.f32_data())
                .arg(q_ref.f32_data())
                .arg(k_ref.f32_data())
                .arg(v_ref.f32_data())
                .arg(d_buf.f32_data())
                .arg(grad_q.f32_data_mut())
                .arg(grad_k.f32_data_mut())
                .arg(grad_v.f32_data_mut())
                .arg(&seq_len_u32)
                .arg(&n_heads_u32)
                .arg(&n_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&q_split_u32)
                .arg(&batch_size_u32)
                .arg(&n_kv_tiles_u32)
                .launch(bwd_cfg)
                .unwrap();
        }

        (grad_q, grad_k, grad_v)
    }

    fn cross_entropy_forward_backward(
        &self,
        logits: &CudaBuffer,
        targets: &CudaBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, CudaBuffer) {
        // Pre-count non-padded positions on CPU (tiny download, same as Metal path)
        let target_data = self.download(targets);
        let count = target_data
            .iter()
            .take(n_positions)
            .filter(|t| t.to_bits() != pad_id)
            .count() as u32;

        let tg_size = std::cmp::min(vocab_size, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_positions as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_positions_u32 = n_positions as u32;
        let vocab_size_u32 = vocab_size as u32;

        // loss_out is always f32
        let mut loss_buf = self.alloc_f32(1);

        if logits.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::CROSS_ENTROPY_BF16_CUDA,
                "cross_entropy_fwd_bwd_bf16",
            );
            let mut grad = self.alloc_f32(n_positions * vocab_size); // f32 for precision

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.bf16_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::CROSS_ENTROPY_CUDA, "cross_entropy_fwd_bwd");
            let mut grad = self.alloc_f32(n_positions * vocab_size);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.f32_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        }
    }

    fn cross_entropy_forward_backward_counted(
        &self,
        logits: &CudaBuffer,
        targets: &CudaBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
        non_pad_count: u32,
    ) -> (f32, CudaBuffer) {
        let tg_size = std::cmp::min(vocab_size, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_positions as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_positions_u32 = n_positions as u32;
        let vocab_size_u32 = vocab_size as u32;

        let mut loss_buf = self.alloc_f32(1);

        if logits.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::CROSS_ENTROPY_BF16_CUDA,
                "cross_entropy_fwd_bwd_bf16",
            );
            let mut grad = self.alloc_f32(n_positions * vocab_size);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.bf16_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&non_pad_count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::CROSS_ENTROPY_CUDA, "cross_entropy_fwd_bwd");
            let mut grad = self.alloc_f32(n_positions * vocab_size);

            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(logits.f32_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&non_pad_count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        }
    }

    fn sync(&self) {
        self.stream.synchronize().unwrap();
    }

    fn matmul_accumulate(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        // Mixed input types: convert f32→bf16 for tensor cores
        if a.is_bf16() != b.is_bf16() {
            let a_conv = if !a.is_bf16() {
                Some(self.convert_f32_to_bf16(a))
            } else {
                None
            };
            let b_conv = if !b.is_bf16() {
                Some(self.convert_f32_to_bf16(b))
            } else {
                None
            };
            let a_ref = a_conv.as_ref().unwrap_or(a);
            let b_ref = b_conv.as_ref().unwrap_or(b);
            self.matmul_accumulate(a_ref, b_ref, c, m, k, n);
            return;
        }

        if a.is_bf16() && c.is_bf16() {
            // All bf16: bf16 gemm_ex with beta=1 accumulate
            let alpha: f32 = 1.0;
            let beta: f32 = 1.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = c.bf16_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex accumulate failed");
                }
            }
        } else if a.is_bf16() && !c.is_bf16() {
            // bf16 inputs, f32 accumulator: gemm_ex with bf16 A/B, f32 C
            let alpha: f32 = 1.0;
            let beta: f32 = 1.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = c.f32_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS mixed bf16→f32 gemm_ex accumulate failed");
                }
            }
        } else {
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_N,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: cublas_m,
                            n: cublas_n,
                            k: cublas_k,
                            alpha: 1.0f32,
                            lda: cublas_m,
                            ldb: cublas_k,
                            beta: 1.0f32,
                            ldc: cublas_m,
                        },
                        b.f32_data(),
                        a.f32_data(),
                        c.f32_data_mut(),
                    )
                    .expect("cuBLAS sgemm accumulate failed");
            }
        }
    }

    fn matmul_a_transposed(
        &self,
        a: &CudaBuffer, // [k, m] row-major
        b: &CudaBuffer, // [k, n] row-major
        m: usize,
        k: usize,
        n: usize,
    ) -> CudaBuffer {
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        // Mixed input types: convert f32→bf16 for tensor cores
        if a.is_bf16() != b.is_bf16() {
            let a_conv = if !a.is_bf16() {
                Some(self.convert_f32_to_bf16(a))
            } else {
                None
            };
            let b_conv = if !b.is_bf16() {
                Some(self.convert_f32_to_bf16(b))
            } else {
                None
            };
            let a_ref = a_conv.as_ref().unwrap_or(a);
            let b_ref = b_conv.as_ref().unwrap_or(b);
            return self.matmul_a_transposed(a_ref, b_ref, m, k, n);
        }

        if a.is_bf16() {
            let mut output = self.pool_alloc_uninit_bf16(m * n);
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = output.bf16_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_T,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_n,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex a_transposed failed");
                }
            }
            output
        } else {
            let mut output = self.pool_alloc_uninit_f32(m * n);
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_N,
                            transb: cublasOperation_t::CUBLAS_OP_T,
                            m: cublas_m,
                            n: cublas_n,
                            k: cublas_k,
                            alpha: 1.0f32,
                            lda: cublas_m,
                            ldb: cublas_n,
                            beta: 0.0f32,
                            ldc: cublas_m,
                        },
                        b.f32_data(),
                        a.f32_data(),
                        output.f32_data_mut(),
                    )
                    .expect("cuBLAS sgemm a_transposed failed");
            }
            output
        }
    }

    fn matmul_b_transposed(
        &self,
        a: &CudaBuffer, // [m, k] row-major
        b: &CudaBuffer, // [n, k] row-major (logically transposed to [k, n])
        m: usize,
        k: usize,
        n: usize,
    ) -> CudaBuffer {
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        // Mixed input types: convert f32→bf16 for tensor cores
        if a.is_bf16() != b.is_bf16() {
            let a_conv = if !a.is_bf16() {
                Some(self.convert_f32_to_bf16(a))
            } else {
                None
            };
            let b_conv = if !b.is_bf16() {
                Some(self.convert_f32_to_bf16(b))
            } else {
                None
            };
            let a_ref = a_conv.as_ref().unwrap_or(a);
            let b_ref = b_conv.as_ref().unwrap_or(b);
            return self.matmul_b_transposed(a_ref, b_ref, m, k, n);
        }

        if a.is_bf16() {
            let mut output = self.pool_alloc_uninit_bf16(m * n);
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = output.bf16_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_T,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex b_transposed failed");
                }
            }
            output
        } else {
            let mut output = self.pool_alloc_uninit_f32(m * n);
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_T,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: cublas_m,
                            n: cublas_n,
                            k: cublas_k,
                            alpha: 1.0f32,
                            lda: cublas_k,
                            ldb: cublas_k,
                            beta: 0.0f32,
                            ldc: cublas_m,
                        },
                        b.f32_data(),
                        a.f32_data(),
                        output.f32_data_mut(),
                    )
                    .expect("cuBLAS sgemm b_transposed failed");
            }
            output
        }
    }

    fn matmul_accumulate_a_transposed(
        &self,
        a: &CudaBuffer,     // [k, m] row-major
        b: &CudaBuffer,     // [k, n] row-major
        c: &mut CudaBuffer, // [m, n] accumulated
        m: usize,
        k: usize,
        n: usize,
    ) {
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        // Mixed input types: convert f32→bf16 for tensor cores
        if a.is_bf16() != b.is_bf16() {
            let a_conv = if !a.is_bf16() {
                Some(self.convert_f32_to_bf16(a))
            } else {
                None
            };
            let b_conv = if !b.is_bf16() {
                Some(self.convert_f32_to_bf16(b))
            } else {
                None
            };
            let a_ref = a_conv.as_ref().unwrap_or(a);
            let b_ref = b_conv.as_ref().unwrap_or(b);
            self.matmul_accumulate_a_transposed(a_ref, b_ref, c, m, k, n);
            return;
        }

        if a.is_bf16() && c.is_bf16() {
            let alpha: f32 = 1.0;
            let beta: f32 = 1.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = c.bf16_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_T,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_n,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex a_transposed accumulate failed");
                }
            }
        } else if a.is_bf16() && !c.is_bf16() {
            let alpha: f32 = 1.0;
            let beta: f32 = 1.0;
            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = c.f32_data_mut().device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_T,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m,
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_n,
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
                        cublas_m,
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS mixed bf16→f32 gemm_ex a_transposed accumulate failed");
                }
            }
        } else {
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_N,
                            transb: cublasOperation_t::CUBLAS_OP_T,
                            m: cublas_m,
                            n: cublas_n,
                            k: cublas_k,
                            alpha: 1.0f32,
                            lda: cublas_m,
                            ldb: cublas_n,
                            beta: 1.0f32,
                            ldc: cublas_m,
                        },
                        b.f32_data(),
                        a.f32_data(),
                        c.f32_data_mut(),
                    )
                    .expect("cuBLAS sgemm a_transposed accumulate failed");
            }
        }
    }

    fn rms_norm_backward_accumulate(
        &self,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
        grad_weight_acc: &mut CudaBuffer,
    ) -> CudaBuffer {
        // The RMS norm backward kernel uses atomicAdd for grad_weight,
        // so we can pass the existing accumulator directly instead of a zeroed buffer.
        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if input.is_bf16() && grad_output.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::RMS_NORM_BACKWARD_BF16_CUDA,
                "rms_norm_backward_bf16",
            );
            let mut grad_input = self.pool_alloc_uninit_bf16(n_groups * dim);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(grad_output.bf16_data())
                    .arg(grad_input.bf16_data_mut())
                    .arg(grad_weight_acc.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }
            grad_input
        } else if input.is_bf16() || weight.is_bf16() || grad_output.is_bf16() {
            // Mixed types: convert bf16 inputs to f32 on GPU
            let input_f32 = if input.is_bf16() {
                self.convert_bf16_to_f32(input)
            } else {
                self.copy_buffer_impl(input)
            };
            let weight_f32 = if weight.is_bf16() {
                self.convert_bf16_to_f32(weight)
            } else {
                self.copy_buffer_impl(weight)
            };
            let grad_f32 = if grad_output.is_bf16() {
                self.convert_bf16_to_f32(grad_output)
            } else {
                self.copy_buffer_impl(grad_output)
            };
            self.rms_norm_backward_accumulate(
                &input_f32,
                &weight_f32,
                &grad_f32,
                n_groups,
                dim,
                eps,
                grad_weight_acc,
            )
        } else {
            let (_module, func) =
                self.get_func(backward_cuda::RMS_NORM_BACKWARD_CUDA, "rms_norm_backward");
            let mut grad_input = self.pool_alloc_uninit_f32(n_groups * dim);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.f32_data())
                    .arg(weight.f32_data())
                    .arg(grad_output.f32_data())
                    .arg(grad_input.f32_data_mut())
                    .arg(grad_weight_acc.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }
            grad_input
        }
    }

    fn rms_norm_backward_residual_accumulate(
        &self,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        residual_grad: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
        grad_weight_acc: &mut CudaBuffer,
    ) -> CudaBuffer {
        // For bf16 inputs, fall back to default (convert + unfused)
        if input.is_bf16() || weight.is_bf16() || grad_output.is_bf16() || residual_grad.is_bf16() {
            let gi = self.rms_norm_backward_accumulate(
                input,
                weight,
                grad_output,
                n_groups,
                dim,
                eps,
                grad_weight_acc,
            );
            return self.add_tensors_buf(&gi, residual_grad, n_groups * dim);
        }

        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        let (_module, func) = self.get_func(
            backward_cuda::RMS_NORM_BACKWARD_RESIDUAL_CUDA,
            "rms_norm_backward_residual",
        );
        let mut grad_input = self.pool_alloc_uninit_f32(n_groups * dim);
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(input.f32_data())
                .arg(weight.f32_data())
                .arg(grad_output.f32_data())
                .arg(residual_grad.f32_data())
                .arg(grad_input.f32_data_mut())
                .arg(grad_weight_acc.f32_data_mut())
                .arg(&n_groups_u32)
                .arg(&dim_u32)
                .arg(&eps)
                .launch(cfg)
                .unwrap();
        }
        grad_input
    }

    fn reduce_sum_accumulate(
        &self,
        data: &CudaBuffer,
        shape: &[usize],
        axis: usize,
        dst: &mut CudaBuffer,
    ) {
        // Use a kernel that atomicAdds the reduction result into dst.
        let ndim = shape.len();
        assert!(axis < ndim);

        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let out_len = outer * inner;

        if out_len == 0 {
            return;
        }

        let tg_size = std::cmp::min(axis_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (out_len as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let outer_u32 = outer as u32;
        let axis_len_u32 = axis_len as u32;
        let inner_u32 = inner as u32;

        if data.is_bf16() {
            // Fused bf16→f32 reduce+accumulate (no separate conversion)
            let (_module, func) = self.get_func(
                reduce_cuda::REDUCE_SUM_ACCUMULATE_BF16_CUDA,
                "reduce_sum_accumulate_bf16",
            );
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(dst.f32_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }
        } else {
            let (_module, func) = self.get_func(
                reduce_cuda::REDUCE_SUM_ACCUMULATE_CUDA,
                "reduce_sum_accumulate",
            );
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(dst.f32_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }
        }
    }

    fn copy_buffer(&self, src: &CudaBuffer) -> CudaBuffer {
        self.copy_buffer_impl(src)
    }

    fn extract_columns(
        &self,
        buf: &CudaBuffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> CudaBuffer {
        CudaComputeDevice::extract_columns(self, buf, batch, total_cols, col_start, col_count)
    }

    fn concat_columns(
        &self,
        dst: &mut CudaBuffer,
        src: &CudaBuffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        CudaComputeDevice::concat_columns(self, dst, src, batch, total_cols, col_start, col_count)
    }

    fn rms_norm_residual(
        &self,
        input: &CudaBuffer,
        residual: &CudaBuffer,
        weight: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (CudaBuffer, CudaBuffer) {
        CudaComputeDevice::rms_norm_residual(self, input, residual, weight, n_groups, dim, eps)
    }

    fn slice_buffer(&self, buf: &CudaBuffer, offset: usize, len: usize) -> CudaBuffer {
        CudaComputeDevice::slice_buffer(self, buf, offset, len)
    }

    fn write_into(&self, dst: &mut CudaBuffer, offset: usize, src: &CudaBuffer) {
        CudaComputeDevice::write_into(self, dst, offset, src)
    }

    fn bias_add(
        &self,
        matrix: &CudaBuffer,
        bias: &CudaBuffer,
        numel: usize,
        dim: usize,
    ) -> CudaBuffer {
        let numel_u32 = numel as u32;
        let dim_u32 = dim as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((numel_u32 + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        if matrix.is_bf16() && bias.is_bf16() {
            // Native bf16 bias_add: single kernel, no conversions
            let (_module, func) = self.get_func(util_cuda::BIAS_ADD_BF16_CUDA, "bias_add_bf16");
            let mut output = self.pool_alloc_uninit_bf16(numel);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(matrix.bf16_data())
                    .arg(bias.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&numel_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }
            output
        } else {
            // Fall back to f32 path for mixed or pure f32 types
            let any_bf16 = matrix.is_bf16() || bias.is_bf16();
            let mat_conv = if matrix.is_bf16() {
                Some(self.convert_bf16_to_f32(matrix))
            } else {
                None
            };
            let bias_conv = if bias.is_bf16() {
                Some(self.convert_bf16_to_f32(bias))
            } else {
                None
            };
            let mat_ref = mat_conv.as_ref().unwrap_or(matrix);
            let bias_ref = bias_conv.as_ref().unwrap_or(bias);

            let mut output = self.pool_alloc_uninit_f32(numel);
            let (_module, func) = self.get_func(util_cuda::BIAS_ADD_CUDA, "bias_add");
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(mat_ref.f32_data())
                    .arg(bias_ref.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&numel_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }
            if any_bf16 {
                self.convert_f32_to_bf16(&output)
            } else {
                output
            }
        }
    }

    fn rope_forward(
        &self,
        input: &CudaBuffer,
        cos_table: &[f32],
        sin_table: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let half_dim = head_dim / 2;
        let total = (seq_len * n_heads * half_dim) as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (total.div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u = seq_len as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        let half_dim_u = half_dim as u32;
        let start_pos_u = start_pos as u32;

        // Upload cos/sin tables (f32 always — small, ~256KB for seq=2048 dim=64)
        let cos_buf = self.upload_f32(cos_table);
        let sin_buf = self.upload_f32(sin_table);

        if input.is_bf16() {
            let (_module, func) =
                self.get_func(util_cuda::ROPE_FORWARD_BF16_CUDA, "rope_forward_bf16");
            let mut output = self.pool_alloc_uninit_bf16(input.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_forward_bf16 launch failed");
            }
            output
        } else {
            let (_module, func) = self.get_func(util_cuda::ROPE_FORWARD_CUDA, "rope_forward");
            let mut output = self.pool_alloc_uninit_f32(input.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.f32_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_forward launch failed");
            }
            output
        }
    }

    fn rope_forward_cached(
        &self,
        input: &CudaBuffer,
        cos_buf: &CudaBuffer,
        sin_buf: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let half_dim = head_dim / 2;
        let total = (seq_len * n_heads * half_dim) as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (total.div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u = seq_len as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        let half_dim_u = half_dim as u32;
        let start_pos_u = start_pos as u32;

        if input.is_bf16() {
            let (_module, func) =
                self.get_func(util_cuda::ROPE_FORWARD_BF16_CUDA, "rope_forward_bf16");
            let mut output = self.pool_alloc_uninit_bf16(input.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_forward_bf16 cached launch failed");
            }
            output
        } else {
            let (_module, func) = self.get_func(util_cuda::ROPE_FORWARD_CUDA, "rope_forward");
            let mut output = self.pool_alloc_uninit_f32(input.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(input.f32_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_forward cached launch failed");
            }
            output
        }
    }

    fn rope_backward(
        &self,
        grad_output: &CudaBuffer,
        cos_table: &[f32],
        sin_table: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let half_dim = head_dim / 2;
        let total = (seq_len * n_heads * half_dim) as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (total.div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u = seq_len as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        let half_dim_u = half_dim as u32;
        let start_pos_u = start_pos as u32;

        let cos_buf = self.upload_f32(cos_table);
        let sin_buf = self.upload_f32(sin_table);

        if grad_output.is_bf16() {
            let (_module, func) =
                self.get_func(util_cuda::ROPE_BACKWARD_BF16_CUDA, "rope_backward_bf16");
            let mut output = self.pool_alloc_uninit_bf16(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward_bf16 launch failed");
            }
            output
        } else {
            let (_module, func) = self.get_func(util_cuda::ROPE_BACKWARD_CUDA, "rope_backward");
            let mut output = self.pool_alloc_uninit_f32(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.f32_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward launch failed");
            }
            output
        }
    }

    fn rope_backward_batched(
        &self,
        grad_output: &CudaBuffer,
        cos_table: &[f32],
        sin_table: &[f32],
        total_rows: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let half_dim = head_dim / 2;
        let total = (total_rows * n_heads * half_dim) as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (total.div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        let total_rows_u = total_rows as u32;
        let seq_len_u = seq_len as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        let half_dim_u = half_dim as u32;
        let start_pos_u = start_pos as u32;

        let cos_buf = self.upload_f32(cos_table);
        let sin_buf = self.upload_f32(sin_table);

        if grad_output.is_bf16() {
            let (_module, func) = self.get_func(
                util_cuda::ROPE_BACKWARD_BATCHED_BF16_CUDA,
                "rope_backward_batched_bf16",
            );
            let mut output = self.pool_alloc_uninit_bf16(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&total_rows_u)
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward_batched_bf16 launch failed");
            }
            output
        } else {
            let (_module, func) = self.get_func(
                util_cuda::ROPE_BACKWARD_BATCHED_CUDA,
                "rope_backward_batched",
            );
            let mut output = self.pool_alloc_uninit_f32(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.f32_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&total_rows_u)
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward_batched launch failed");
            }
            output
        }
    }

    fn rope_backward_batched_cached(
        &self,
        grad_output: &CudaBuffer,
        cos_buf: &CudaBuffer,
        sin_buf: &CudaBuffer,
        total_rows: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let half_dim = head_dim / 2;
        let total = (total_rows * n_heads * half_dim) as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (total.div_ceil(256), 1, 1),
            shared_mem_bytes: 0,
        };
        let total_rows_u = total_rows as u32;
        let seq_len_u = seq_len as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        let half_dim_u = half_dim as u32;
        let start_pos_u = start_pos as u32;

        if grad_output.is_bf16() {
            let (_module, func) = self.get_func(
                util_cuda::ROPE_BACKWARD_BATCHED_BF16_CUDA,
                "rope_backward_batched_bf16",
            );
            let mut output = self.pool_alloc_uninit_bf16(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&total_rows_u)
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward_batched_bf16 cached launch failed");
            }
            output
        } else {
            let (_module, func) = self.get_func(
                util_cuda::ROPE_BACKWARD_BATCHED_CUDA,
                "rope_backward_batched",
            );
            let mut output = self.pool_alloc_uninit_f32(grad_output.len);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad_output.f32_data())
                    .arg(cos_buf.f32_data())
                    .arg(sin_buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&total_rows_u)
                    .arg(&seq_len_u)
                    .arg(&n_heads_u)
                    .arg(&head_dim_u)
                    .arg(&half_dim_u)
                    .arg(&start_pos_u)
                    .launch(cfg)
                    .expect("rope_backward_batched cached launch failed");
            }
            output
        }
    }

    fn add_tensors_buf(&self, a: &CudaBuffer, b: &CudaBuffer, numel: usize) -> CudaBuffer {
        if let Some(out) = self.add_llm(a, b, numel) {
            return out;
        }
        if a.is_bf16() && b.is_bf16() {
            let n = numel as u32;
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: ((n + 255) / 256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (_module, func) =
                self.get_func(util_cuda::ADD_TENSORS_BF16_CUDA, "add_tensors_bf16");
            let mut out = self.pool_alloc_uninit_bf16(numel);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(a.bf16_data())
                    .arg(b.bf16_data())
                    .arg(out.bf16_data_mut())
                    .arg(&n)
                    .launch(cfg)
                    .unwrap();
            }
            out
        } else {
            // Fall back to generic elementwise for f32
            self.elementwise(&[a, b], numel, &|ids| ids[0] + ids[1])
        }
    }

    fn swiglu_fused_buf(&self, gate: &CudaBuffer, up: &CudaBuffer, numel: usize) -> CudaBuffer {
        if gate.is_bf16() && up.is_bf16() {
            let n = numel as u32;
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: ((n + 255) / 256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (_module, func) =
                self.get_func(util_cuda::SWIGLU_FUSED_BF16_CUDA, "swiglu_fused_bf16");
            let mut out = self.pool_alloc_uninit_bf16(numel);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(gate.bf16_data())
                    .arg(up.bf16_data())
                    .arg(out.bf16_data_mut())
                    .arg(&n)
                    .launch(cfg)
                    .unwrap();
            }
            out
        } else {
            use tang::Scalar;
            use tang_expr::node::ExprId;
            self.elementwise(&[gate, up], numel, &|ids: &[ExprId]| {
                let one = ExprId::from_f64(1.0);
                let neg_gate = -ids[0];
                let exp_neg = Scalar::exp(neg_gate);
                let sigmoid = one / (one + exp_neg);
                ids[0] * sigmoid * ids[1]
            })
        }
    }

    fn swiglu_backward_buf(
        &self,
        grad: &CudaBuffer,
        gate: &CudaBuffer,
        up: &CudaBuffer,
        numel: usize,
    ) -> (CudaBuffer, CudaBuffer) {
        if grad.is_bf16() && gate.is_bf16() && up.is_bf16() {
            let n = numel as u32;
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: ((n + 255) / 256, 1, 1),
                shared_mem_bytes: 0,
            };
            let (_module, func) =
                self.get_func(util_cuda::SWIGLU_BACKWARD_BF16_CUDA, "swiglu_backward_bf16");
            let mut grad_gate = self.pool_alloc_uninit_bf16(numel);
            let mut grad_up = self.pool_alloc_uninit_bf16(numel);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(grad.bf16_data())
                    .arg(gate.bf16_data())
                    .arg(up.bf16_data())
                    .arg(grad_gate.bf16_data_mut())
                    .arg(grad_up.bf16_data_mut())
                    .arg(&n)
                    .launch(cfg)
                    .unwrap();
            }
            (grad_gate, grad_up)
        } else {
            use tang::Scalar;
            use tang_expr::node::ExprId;
            let grad_up = self.elementwise(&[grad, gate], numel, &|ids: &[ExprId]| {
                let one = ExprId::from_f64(1.0);
                let neg_gate = -ids[1];
                let exp_neg = Scalar::exp(neg_gate);
                let sigmoid = one / (one + exp_neg);
                ids[0] * ids[1] * sigmoid
            });
            let grad_gate = self.elementwise(&[grad, gate, up], numel, &|ids: &[ExprId]| {
                let one = ExprId::from_f64(1.0);
                let neg_gate = -ids[1];
                let exp_neg = Scalar::exp(neg_gate);
                let sigmoid = one / (one + exp_neg);
                let dsilu = sigmoid * (one + ids[1] * (one - sigmoid));
                ids[0] * ids[2] * dsilu
            });
            (grad_gate, grad_up)
        }
    }

    fn add_norm_relative_noise(
        &self,
        buf: &mut CudaBuffer,
        epsilon: f32,
        seed: u64,
        rows: usize,
        cols: usize,
    ) {
        assert_eq!(buf.len, rows * cols);
        let block_dim = cols.min(256) as u32;
        let cfg = LaunchConfig {
            block_dim: (block_dim, 1, 1),
            grid_dim: (rows as u32, 1, 1),
            shared_mem_bytes: block_dim * 4,
        };
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        match buf.storage_mut() {
            CudaStorage::F32(s) => {
                let (_module, func) = self.get_func(
                    crate::kernels::noise_cuda::NORM_RELATIVE_NOISE_F32_CUDA,
                    "norm_relative_noise_f32",
                );
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(s)
                        .arg(&epsilon)
                        .arg(&seed)
                        .arg(&rows_u32)
                        .arg(&cols_u32)
                        .launch(cfg)
                        .unwrap();
                }
            }
            CudaStorage::Bf16(s) => {
                let (_module, func) = self.get_func(
                    crate::kernels::noise_cuda::NORM_RELATIVE_NOISE_BF16_CUDA,
                    "norm_relative_noise_bf16",
                );
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(s)
                        .arg(&epsilon)
                        .arg(&seed)
                        .arg(&rows_u32)
                        .arg(&cols_u32)
                        .launch(cfg)
                        .unwrap();
                }
            }
            CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("4-bit weights are read-only"),
        }
    }

    fn upload_bf16(&self, bits: &[u16]) -> CudaBuffer {
        self.upload_bf16_impl(bits)
    }

    fn upload_bytes(&self, bytes: &[u8]) -> CudaBuffer {
        self.upload_bytes_impl(bytes)
    }

    fn alloc_bf16(&self, len: usize) -> CudaBuffer {
        self.pool_alloc_bf16(len)
    }

    fn buffer_addr(&self, buf: &CudaBuffer) -> u64 {
        self.buffer_addr_impl(buf)
    }

    fn upload_q2(&self, raw: &[u8], n: usize, k: usize) -> CudaBuffer {
        self.upload_q2_impl(raw, n, k)
    }

    fn linear_into(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.linear_into_impl(x, w, out, m, k, n)
    }

    fn quantize_act_into(&self, x: &CudaBuffer, xq: &mut CudaBuffer, m: usize, k: usize) {
        self.quantize_act_impl(x, xq, m, k)
    }

    fn q2_linear_into(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.q2_linear_impl(xq, w, out, m, k, n)
    }

    fn q8x_linear_into(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.q8x_linear_impl(xq, w, out, m, k, n)
    }

    fn native_linear_into(
        &self,
        ty: crate::flash_native::NatType,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.native_linear_impl(ty, xq, w, out, 0, n, m, k, n)
    }

    fn bf16_linear_out_into(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        off: usize,
        ostride: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.bf16_linear_out_impl(x, w, out, off, ostride, m, k, n)
    }

    fn native_stack(
        &self,
        segs: Vec<crate::flash_native::NatSeg<CudaBuffer>>,
        n: usize,
        k: usize,
    ) -> crate::flash_native::NatStack<CudaBuffer> {
        self.native_stack_impl(segs, n, k)
    }

    fn native_stack_into(
        &self,
        s: &crate::flash_native::NatStack<CudaBuffer>,
        x: &CudaBuffer,
        xq: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
    ) {
        self.native_stack_into_impl(s, x, xq, out, m)
    }

    fn native_linear_out_into(
        &self,
        ty: crate::flash_native::NatType,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        off: usize,
        ostride: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.native_linear_impl(ty, xq, w, out, off, ostride, m, k, n)
    }

    fn q4x_linear_into(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.q4x_linear_impl(xq, w, out, m, k, n)
    }

    fn hc_read_into(
        &self,
        r: &mut CudaBuffer,
        pending: Option<crate::flash::HcPending<'_, CudaBuffer>>,
        w: &crate::flash::HcWeights<'_, CudaBuffer>,
        x: &mut CudaBuffer,
        xq: Option<&mut CudaBuffer>,
        inj: Option<&mut CudaBuffer>,
        scratch: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        self.hc_read_impl(r, pending, w, x, xq, inj, scratch, t, eps)
    }

    fn hc_write(&self, r: &mut CudaBuffer, y: &CudaBuffer, inj: &CudaBuffer, t: usize) {
        self.hc_write_impl(r, y, inj, t)
    }

    fn gdn_conv_into(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        conv: &CudaBuffer,
        h: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        self.gdn_conv_impl(proj, stride, hist, conv, h, t, eps)
    }

    fn gdn_conv_commit(
        &self,
        hist: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        t: usize,
    ) {
        self.gdn_conv_commit_impl(hist, proj, stride, win, t)
    }

    fn gdn_step(
        &self,
        state: &mut CudaBuffer,
        h: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        p: &crate::flash::GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: crate::flash::GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        self.gdn_step_impl(state, h, proj, stride, p, y, yq, t, mode, eps)
    }

    fn gdn_conv_step(
        &self,
        state: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        p: &crate::flash::GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: crate::flash::GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        self.gdn_conv_step_impl(state, proj, stride, hist, p, y, yq, t, mode, eps)
    }

    fn router_topk_into(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        t: usize,
    ) {
        self.router_topk_impl(logits, stride, n_expert, ids, w, t)
    }

    fn moe_plan_into(
        &self,
        ids: &CudaBuffer,
        table: &CudaBuffer,
        shared: u64,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        self.moe_plan_impl(ids, table, shared, plan, t)
    }

    fn moe_route_into(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        forced: Option<&CudaBuffer>,
        table: &CudaBuffer,
        shared: u64,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        self.moe_route_impl(
            logits, stride, n_expert, forced, table, shared, ids, w, plan, t,
        )
    }

    unsafe fn moe_grouped_into(
        &self,
        xq: &CudaBuffer,
        plan: &CudaBuffer,
        scratch: &mut CudaBuffer,
        parts: &mut CudaBuffer,
        t: usize,
    ) {
        self.moe_grouped_impl(xq, plan, scratch, parts, t)
    }

    fn moe_combine_into(
        &self,
        parts: &CudaBuffer,
        w: &CudaBuffer,
        logits: &CudaBuffer,
        stride: usize,
        sg: Option<usize>,
        y: &mut CudaBuffer,
        t: usize,
    ) {
        self.moe_combine_impl(parts, w, logits, stride, sg, y, t)
    }

    fn qsa_prep(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        norms: &crate::flash::QsaNorms<'_, CudaBuffer>,
        rope: (&CudaBuffer, &CudaBuffer),
        q: &mut CudaBuffer,
        cache: crate::flash::QsaCache<'_, CudaBuffer>,
        t: usize,
        eps: f32,
    ) {
        self.qsa_prep_impl(proj, stride, win, norms, rope, q, cache, t, eps)
    }

    fn qsa_select_into(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        self.qsa_select_impl(pooled, q, win, scores, ids, max_blocks, t)
    }

    fn qsa_select_union_into(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        union: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        self.qsa_select_union_impl(pooled, q, win, scores, ids, union, max_blocks, t)
    }

    fn qsa_attend_union_into(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        _ids: &CudaBuffer,
        union: &CudaBuffer,
        max_blocks: usize,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        self.qsa_attend_union_impl(
            q, k_cache, v_cache, union, max_blocks, proj, stride, win, scratch, out, outq, t,
        )
    }

    fn qsa_attend_into(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        ids: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        self.qsa_attend_impl(
            q, k_cache, v_cache, ids, proj, stride, win, scratch, out, outq, t,
        )
    }

    fn bf16_storage(&self) -> bool {
        true
    }

    fn download_bf16(&self, buf: &CudaBuffer) -> Vec<u16> {
        match buf.storage() {
            CudaStorage::Bf16(s) => self.stream.memcpy_dtov(s).unwrap(),
            _ => buf.to_vec().into_iter().map(f32_to_bf16).collect(),
        }
    }

    fn upload_q4(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        group: usize,
    ) -> CudaBuffer {
        self.upload_q4_impl(packed, scales, biases, group)
    }

    fn top_k_rows(&self, x: &CudaBuffer, rows: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
        self.top_k_rows_llm(x, rows, n, k).unwrap_or_else(|| {
            let v = self.download(x);
            crate::top_k_rows_host(&v[..rows * n], n, k)
        })
    }

    fn linear(&self, x: &CudaBuffer, w: &CudaBuffer, m: usize, k: usize, n: usize) -> CudaBuffer {
        self.linear_llm(x, w, m, k, n)
            .unwrap_or_else(|| self.matmul_b_transposed(x, w, m, k, n))
    }

    fn attention_prep(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_cache: &mut CudaBuffer,
        v_cache: &mut CudaBuffer,
        seq: usize,
        shape: (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> CudaBuffer {
        if let Some(q) = self.attention_prep_llm(
            qkv, q_norm, k_norm, cos, sin, k_cache, v_cache, seq, shape, pos, eps, None,
        ) {
            return q;
        }
        crate::device::attention_prep_default(
            self, qkv, q_norm, k_norm, cos, sin, k_cache, v_cache, seq, shape, pos, eps,
        )
    }

    fn kv_attention_window(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        cache_start: usize,
        q_len: usize,
        shape: (usize, usize, usize),
        window: usize,
        causal: bool,
    ) -> CudaBuffer {
        self.attention_llm(
            q,
            k_cache,
            v_cache,
            cache_start,
            q_len,
            shape,
            window,
            !causal,
            None,
        )
    }

    fn attention_prep_paged(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_pool: &mut CudaBuffer,
        v_pool: &mut CudaBuffer,
        pages: &crate::Pages<CudaBuffer>,
        seq: usize,
        shape: (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> CudaBuffer {
        let pg = Some((pages.table, pages.shift()));
        if let Some(q) = self.attention_prep_llm(
            qkv, q_norm, k_norm, cos, sin, k_pool, v_pool, seq, shape, pos, eps, pg,
        ) {
            return q;
        }
        crate::device::attention_prep_paged_default(
            self, qkv, q_norm, k_norm, cos, sin, k_pool, v_pool, pages, seq, shape, pos, eps,
        )
    }

    fn kv_attention_paged(
        &self,
        q: &CudaBuffer,
        k_pool: &CudaBuffer,
        v_pool: &CudaBuffer,
        pages: &crate::Pages<CudaBuffer>,
        cache_start: usize,
        q_len: usize,
        shape: (usize, usize, usize),
        window: usize,
        causal: bool,
    ) -> CudaBuffer {
        let (nh, nkv, d) = shape;
        if d > 256 || nh % nkv != 0 {
            return crate::device::kv_attention_paged_default(
                self,
                q,
                k_pool,
                v_pool,
                pages,
                cache_start,
                q_len,
                shape,
                window,
                causal,
            );
        }
        self.attention_llm(
            q,
            k_pool,
            v_pool,
            cache_start,
            q_len,
            shape,
            window,
            !causal,
            Some((pages.table, pages.shift())),
        )
    }

    fn attention_paged_rows(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_pool: &mut CudaBuffer,
        v_pool: &mut CudaBuffer,
        rows: &crate::PagedRows<CudaBuffer>,
        shape: (usize, usize, usize),
        eps: f32,
        query_scale: f32,
    ) -> CudaBuffer {
        if let Some(out) = self.attention_rows_llm(
            qkv,
            q_norm,
            k_norm,
            cos,
            sin,
            k_pool,
            v_pool,
            rows,
            shape,
            eps,
            query_scale,
        ) {
            return out;
        }
        crate::device::attention_paged_rows_default(
            self,
            qkv,
            q_norm,
            k_norm,
            cos,
            sin,
            k_pool,
            v_pool,
            rows,
            shape,
            eps,
            query_scale,
        )
    }

    fn attention_full(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        n: usize,
        nh: usize,
        hd: usize,
    ) -> CudaBuffer {
        self.attention_llm(q, k, v, 0, n, (nh, nh, hd), 0, true, None)
    }

    fn layer_norm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        b: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        self.layer_norm_llm(x, w, b, rows, dim, eps)
    }

    fn gelu_tanh(&self, x: &CudaBuffer, n: usize) -> CudaBuffer {
        self.gelu_tanh_llm(x, n)
    }

    fn gelu_erf(&self, x: &CudaBuffer, n: usize) -> CudaBuffer {
        self.gelu_erf_llm(x, n)
    }

    fn geglu_split(&self, gu: &CudaBuffer, rows: usize, ff: usize) -> CudaBuffer {
        self.gated_split_llm("geglu_split", gu, rows, ff)
    }

    fn swiglu_split(&self, gu: &CudaBuffer, rows: usize, ff: usize) -> CudaBuffer {
        self.gated_split_llm("swiglu_split", gu, rows, ff)
    }

    fn moe_route(
        &self,
        logits: &CudaBuffer,
        rows: usize,
        experts: usize,
        k: usize,
        router: crate::moe::Router,
    ) -> (CudaBuffer, CudaBuffer) {
        (!moe::host_op("route"))
            .then(|| self.moe_route_llm(logits, rows, experts, k, router))
            .flatten()
            .unwrap_or_else(|| {
                let (ids, w) = crate::moe::route(&self.download(logits), rows, experts, k, router);
                (self.upload_u32(&ids), self.upload(&w))
            })
    }

    fn moe_linear(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        fmt: crate::moe::ExpertFmt,
        experts: usize,
        bias: Option<&CudaBuffer>,
        ids: &CudaBuffer,
        k_top: usize,
        k: usize,
        n: usize,
        per_pick: bool,
    ) -> CudaBuffer {
        if let Some(y) = (!moe::host_op("linear"))
            .then(|| self.moe_linear_llm(x, w, fmt, experts, bias, ids, k_top, k, n, per_pick))
            .flatten()
        {
            return y;
        }
        crate::device::moe_linear_default(
            self, x, w, fmt, experts, bias, ids, k_top, k, n, per_pick,
        )
    }

    fn swiglu_clamped(
        &self,
        gu: &CudaBuffer,
        rows: usize,
        ff: usize,
        alpha: f32,
        limit: f32,
    ) -> CudaBuffer {
        if moe::host_op("swiglu") {
            let y = crate::moe::swiglu_clamped(&self.download(gu), rows, ff, alpha, limit);
            return self.upload(&y);
        }
        self.swiglu_clamped_llm(gu, rows, ff, alpha, limit)
    }

    fn moe_combine(
        &self,
        parts: &CudaBuffer,
        w: &CudaBuffer,
        rows: usize,
        k_top: usize,
        h: usize,
    ) -> CudaBuffer {
        if moe::host_op("combine") {
            let y =
                crate::moe::moe_combine(&self.download(parts), &self.download(w), rows, k_top, h);
            return self.upload(&y);
        }
        self.moe_combine_llm(parts, w, rows, k_top, h)
    }

    fn attention_paged_rows_sinks(
        &self,
        qkv: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_pool: &mut CudaBuffer,
        v_pool: &mut CudaBuffer,
        rows: &crate::PagedRows<CudaBuffer>,
        shape: (usize, usize, usize),
        window: usize,
        sinks: &CudaBuffer,
    ) -> CudaBuffer {
        let (nh, nkv, hd) = shape;
        let (qd, w) = (nh * hd, (nh + 2 * nkv) * hd);
        let n = rows.pos.len();
        // Prep each row (RoPE, K/V into its cache), then one sink-attention launch for all.
        let mut q = self.pool_alloc_uninit_f32(n * qd);
        for (i, (pages, &pos)) in rows.pages.iter().zip(rows.pos).enumerate() {
            let row = self.slice_buffer(qkv, i * w, w);
            let qi = self.attention_prep_paged(
                &row, None, None, cos, sin, k_pool, v_pool, pages, 1, shape, pos, 0.0,
            );
            self.write_into(&mut q, i * qd, &qi);
        }
        let sh = rows.pages.first().map_or(0, |p| p.shift());
        if !moe::host_op("sinks") {
            if let Some(o) = self.attention_sinks_llm(
                &q,
                k_pool,
                v_pool,
                rows.tables,
                sh,
                0,
                n,
                shape,
                window,
                sinks,
                Some((rows.positions, rows.offsets, rows.pos)),
            ) {
                return o;
            }
        }
        let mut out = self.pool_alloc_uninit_f32(n * qd);
        for (i, (pages, &pos)) in rows.pages.iter().zip(rows.pos).enumerate() {
            let qi = self.slice_buffer(&q, i * qd, qd);
            let o = self.kv_attention_paged_sinks(
                &qi, k_pool, v_pool, pages, pos, 1, shape, window, true, sinks,
            );
            self.write_into(&mut out, i * qd, &o);
        }
        out
    }

    fn kv_attention_paged_sinks(
        &self,
        q: &CudaBuffer,
        k_pool: &CudaBuffer,
        v_pool: &CudaBuffer,
        pages: &crate::Pages<CudaBuffer>,
        cache_start: usize,
        q_len: usize,
        shape: (usize, usize, usize),
        window: usize,
        causal: bool,
        sinks: &CudaBuffer,
    ) -> CudaBuffer {
        if causal && !moe::host_op("sinks") {
            if let Some(o) = self.attention_sinks_llm(
                q,
                k_pool,
                v_pool,
                pages.table,
                pages.shift(),
                cache_start,
                q_len,
                shape,
                window,
                sinks,
                None,
            ) {
                return o;
            }
        }
        crate::device::kv_attention_paged_sinks_default(
            self,
            q,
            k_pool,
            v_pool,
            pages,
            cache_start,
            q_len,
            shape,
            window,
            causal,
            sinks,
        )
    }

    fn rope_half_cached(
        &self,
        input: &CudaBuffer,
        cos_buf: &CudaBuffer,
        sin_buf: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        self.rope_half_llm(
            input, cos_buf, sin_buf, seq_len, n_heads, head_dim, start_pos,
        )
    }

    fn add_assign(&self, dst: &mut CudaBuffer, src: &CudaBuffer) {
        assert_eq!(dst.len, src.len);
        let n = dst.len as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        match (dst.storage_mut(), src.storage()) {
            (CudaStorage::F32(d), CudaStorage::F32(s)) => {
                let (_module, func) =
                    self.get_func(crate::kernels::adamw_cuda::ADD_ASSIGN_CUDA, "add_assign");
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(d)
                        .arg(s)
                        .arg(&n)
                        .launch(cfg)
                        .unwrap();
                }
            }
            (CudaStorage::F32(d), CudaStorage::Bf16(s)) => {
                let (_module, func) = self.get_func(
                    crate::kernels::adamw_cuda::ADD_ASSIGN_BF16_TO_F32_CUDA,
                    "add_assign_bf16_to_f32",
                );
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(d)
                        .arg(s)
                        .arg(&n)
                        .launch(cfg)
                        .unwrap();
                }
            }
            (CudaStorage::Bf16(d), CudaStorage::Bf16(s)) => {
                // Native bf16 += bf16: single kernel, no conversions
                let (_module, func) = self.get_func(
                    crate::kernels::adamw_cuda::ADD_ASSIGN_BF16_CUDA,
                    "add_assign_bf16",
                );
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(d)
                        .arg(s)
                        .arg(&n)
                        .launch(cfg)
                        .unwrap();
                }
            }
            (CudaStorage::Bf16(d), CudaStorage::F32(s)) => {
                // Native bf16 += f32: single kernel, no conversions
                let (_module, func) = self.get_func(
                    crate::kernels::adamw_cuda::ADD_ASSIGN_F32_TO_BF16_CUDA,
                    "add_assign_f32_to_bf16",
                );
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(d)
                        .arg(s)
                        .arg(&n)
                        .launch(cfg)
                        .unwrap();
                }
            }
            _ => panic!("add_assign: unsupported storage"),
        }
    }

    fn zero_buffer(&self, buf: &mut CudaBuffer) {
        // Use memset instead of a kernel launch — float 0.0 is all-zero bits.
        match buf.storage_mut() {
            CudaStorage::F32(s) => {
                self.stream.memset_zeros(s).unwrap();
            }
            CudaStorage::Bf16(s) => {
                self.stream.memset_zeros(s).unwrap();
            }
            CudaStorage::Q4(_) | CudaStorage::Q2(_) => panic!("4-bit weights are read-only"),
        }
    }

    fn adamw_step(
        &self,
        param: &mut CudaBuffer,
        grad: &CudaBuffer,
        m: &mut CudaBuffer,
        v: &mut CudaBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: usize,
    ) {
        let n = param.len as u32;
        let beta1_pow = beta1.powi(step_t as i32);
        let beta2_pow = beta2.powi(step_t as i32);
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        if param.is_bf16() {
            let (_module, func) = self.get_func(
                crate::kernels::adamw_cuda::ADAMW_STEP_BF16_CUDA,
                "adamw_step_bf16",
            );
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(param.bf16_data_mut())
                    .arg(grad.f32_data())
                    .arg(m.f32_data_mut())
                    .arg(v.f32_data_mut())
                    .arg(&lr)
                    .arg(&beta1)
                    .arg(&beta2)
                    .arg(&eps)
                    .arg(&weight_decay)
                    .arg(&beta1_pow)
                    .arg(&beta2_pow)
                    .arg(&n)
                    .launch(cfg)
                    .unwrap();
            }
        } else {
            let (_module, func) =
                self.get_func(crate::kernels::adamw_cuda::ADAMW_STEP_CUDA, "adamw_step");
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(param.f32_data_mut())
                    .arg(grad.f32_data())
                    .arg(m.f32_data_mut())
                    .arg(v.f32_data_mut())
                    .arg(&lr)
                    .arg(&beta1)
                    .arg(&beta2)
                    .arg(&eps)
                    .arg(&weight_decay)
                    .arg(&beta1_pow)
                    .arg(&beta2_pow)
                    .arg(&n)
                    .launch(cfg)
                    .unwrap();
            }
        }
    }

    fn reduce_sum_sq_accumulate(&self, src: &CudaBuffer, acc: &mut CudaBuffer) {
        // Convert bf16 grad buffers to f32 if needed
        let src_conv = if src.is_bf16() {
            Some(self.convert_bf16_to_f32(src))
        } else {
            None
        };
        let src_ref = src_conv.as_ref().unwrap_or(src);
        let n = src_ref.len as u32;
        // 4 elements per thread, 256 threads per block
        let blocks = (n + 1023) / 1024;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::REDUCE_SUM_SQ_CUDA,
            "reduce_sum_sq",
        );
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(src_ref.f32_data())
                .arg(acc.f32_data_mut())
                .arg(&n)
                .launch(cfg)
                .unwrap();
        }
    }

    fn fused_sum_sq(&self, bufs: &[&CudaBuffer]) -> CudaBuffer {
        if bufs.is_empty() {
            return self.upload_f32(&[0.0f32]);
        }

        // Convert bf16 buffers to f32 if needed
        let converted: Vec<Option<CudaBuffer>> = bufs
            .iter()
            .map(|b| {
                if b.is_bf16() {
                    Some(self.convert_bf16_to_f32(b))
                } else {
                    None
                }
            })
            .collect();

        // Collect raw device pointers and cumulative offsets
        let mut ptrs: Vec<u64> = Vec::with_capacity(bufs.len());
        let mut offsets: Vec<u32> = Vec::with_capacity(bufs.len() + 1);
        let mut _records = Vec::new(); // keep SyncOnDrop alive until launch
        offsets.push(0);
        for (i, buf) in bufs.iter().enumerate() {
            let src = converted[i].as_ref().unwrap_or(buf);
            let (ptr, record) = src.f32_data().device_ptr(&self.stream);
            ptrs.push(ptr);
            _records.push(record);
            offsets.push(offsets.last().unwrap() + src.len as u32);
        }
        let total_n = *offsets.last().unwrap();
        let n_bufs = bufs.len() as u32;

        // Upload pointer and offset arrays to GPU
        let ptrs_gpu = self.stream.memcpy_stod(&ptrs).unwrap();
        let offsets_gpu = self.stream.memcpy_stod(&offsets).unwrap();
        let mut acc = self.stream.alloc_zeros::<f32>(1).unwrap();

        // Single fused kernel: all buffers in one launch
        let blocks = (total_n + 1023) / 1024; // 4 elems/thread, 256 threads/block
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (blocks, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::MULTI_BUFFER_SUM_SQ_CUDA,
            "multi_buffer_sum_sq",
        );
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&ptrs_gpu)
                .arg(&offsets_gpu)
                .arg(&n_bufs)
                .arg(&total_n)
                .arg(&mut acc)
                .launch(cfg)
                .unwrap();
        }
        drop(_records);
        self.make_buf(CudaStorage::F32(acc), 1)
    }

    fn scale_buffer(&self, buf: &mut CudaBuffer, scale: f32) {
        if buf.is_bf16() {
            // Convert to f32, scale, convert back
            let mut f32_buf = self.convert_bf16_to_f32(buf);
            self.scale_buffer(&mut f32_buf, scale);
            *buf = self.convert_f32_to_bf16(&f32_buf);
            return;
        }
        let n = buf.len as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::SCALE_BUFFER_CUDA,
            "scale_buffer",
        );
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(buf.f32_data_mut())
                .arg(&scale)
                .arg(&n)
                .launch(cfg)
                .unwrap();
        }
    }

    fn clip_grad_norm(&self, bufs: &mut [&mut CudaBuffer], max_norm: f32) -> f32 {
        if bufs.is_empty() {
            return 0.0;
        }

        // Step 1: Fused sum-of-squares (single kernel, no sync)
        let buf_refs: Vec<&CudaBuffer> = bufs.iter().map(|b| &**b).collect();
        let norm_sq_buf = self.fused_sum_sq(&buf_refs);

        // Step 2: Fused clip+scale on GPU (reads norm_sq, conditionally scales all buffers)
        // Precompute offsets from lengths before taking mutable device pointers
        let n_bufs = bufs.len() as u32;
        let mut offsets: Vec<u32> = Vec::with_capacity(bufs.len() + 1);
        offsets.push(0);
        for buf in bufs.iter() {
            offsets.push(offsets.last().unwrap() + buf.len as u32);
        }
        let total_n = *offsets.last().unwrap();

        // Now take mutable device pointers
        let mut ptrs: Vec<u64> = Vec::with_capacity(bufs.len());
        let mut _records = Vec::new();
        for buf in bufs.iter_mut() {
            let (ptr, record) = buf.f32_data_mut().device_ptr_mut(&self.stream);
            ptrs.push(ptr);
            _records.push(record);
        }

        let ptrs_gpu = self.stream.memcpy_stod(&ptrs).unwrap();
        let offsets_gpu = self.stream.memcpy_stod(&offsets).unwrap();

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((total_n + 255) / 256), 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) =
            self.get_func(backward_cuda::FUSED_CLIP_SCALE_CUDA, "fused_clip_scale");
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(norm_sq_buf.f32_data())
                .arg(&ptrs_gpu)
                .arg(&offsets_gpu)
                .arg(&n_bufs)
                .arg(&total_n)
                .arg(&max_norm)
                .launch(cfg)
                .unwrap();
        }
        drop(_records);

        // Download norm for logging (memcpy_dtov syncs stream internally)
        let norm_sq_val = self.stream.memcpy_dtov(norm_sq_buf.f32_data()).unwrap()[0];
        (norm_sq_val as f64).sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuDevice;
    use crate::ComputeDevice;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{label}: length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "{label}: max diff {max_diff} exceeds tolerance {tol}"
        );
    }

    #[test]
    fn flash_backward_matches_cpu_small() {
        // Small GQA config: seq_len=4, n_heads=2, n_kv_heads=1, head_dim=4
        let seq_len = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic test data
        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 0.5).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.3).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.1).sin())
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.3 + 0.7).cos())
            .collect();

        // CPU reference
        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        // CUDA flash backward
        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        approx_eq(&gq_cuda, &gq_ref, 1e-3, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 1e-3, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 1e-3, "grad_V");
    }

    #[test]
    fn flash_backward_matches_cpu_mha() {
        // MHA config: seq_len=8, n_heads=4, n_kv_heads=4, head_dim=8
        let seq_len = 8;
        let n_heads = 4;
        let n_kv_heads = 4;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 1.0).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.3).sin())
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.25 + 0.1).cos())
            .collect();

        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        approx_eq(&gq_cuda, &gq_ref, 1e-3, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 1e-3, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 1e-3, "grad_V");
    }

    #[test]
    fn flash_backward_matches_cpu_350m_shape() {
        // 350M model shape at reduced seq_len: exercises multi-tile path
        let seq_len = 128;
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01 - 2.0).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02 + 1.0).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015 - 0.5).sin() * 0.1)
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.03 + 0.2).cos() * 0.1)
            .collect();

        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        // Wider tolerance for larger configs (f32 accumulation order differences
        // between flash tiled path and sequential CPU path)
        approx_eq(&gq_cuda, &gq_ref, 2e-1, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 2e-1, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 2e-1, "grad_V");
    }

    #[test]
    fn flash_forward_matches_cpu() {
        // Verify flash forward also matches CPU
        let seq_len = 16;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 1.0).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.3).sin())
            .collect();

        let cpu = CpuDevice::new();
        let o_cpu = cpu.causal_attention(
            &cpu.upload(&q),
            &cpu.upload(&k),
            &cpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_ref = cpu.download(&o_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let o_gpu = gpu.causal_attention(
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_cuda = gpu.download(&o_gpu);

        approx_eq(&o_cuda, &o_ref, 1e-4, "forward output");
    }

    #[test]
    fn flash_backward_bench_350m() {
        // Benchmark flash backward at actual 350M training shape
        let seq_len = 512;
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015).sin() * 0.1)
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.03).cos() * 0.1)
            .collect();

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);

        // Warmup
        let _ = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu.sync();

        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention_backward(
                &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!(
            "  flash backward 350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0,
            iters,
            elapsed
        );

        // Also bench forward for reference
        let _ = gpu.causal_attention(
            &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention(
                &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!(
            "  flash forward  350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0,
            iters,
            elapsed
        );
        eprintln!();
    }

    #[test]
    fn tc_forward_matches_cpu() {
        // Verify tensor-core flash attention matches CPU reference
        // head_dim must be multiple of 16 for wmma
        let seq_len = 32;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 1.0).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.3).sin() * 0.5)
            .collect();

        // CPU reference
        let cpu = CpuDevice::new();
        let o_cpu = cpu.causal_attention(
            &cpu.upload(&q),
            &cpu.upload(&k),
            &cpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_ref = cpu.download(&o_cpu);

        // GPU with bf16 (triggers TC kernel)
        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let o_gpu = gpu.causal_attention(
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_cuda = gpu.download(&o_gpu);

        // bf16 precision: tolerance ~1e-2 (bf16 has ~3 decimal digits)
        approx_eq(&o_cuda, &o_ref, 5e-2, "TC forward output");
    }

    #[test]
    fn tc_forward_matches_cpu_350m_shape() {
        // Test at actual 350M model shape: seq=64, n_heads=16, n_kv_heads=4, head_dim=64
        let seq_len = 64;
        let n_heads = 16;
        let n_kv_heads = 4;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015).sin() * 0.1)
            .collect();

        let cpu = CpuDevice::new();
        let o_cpu = cpu.causal_attention(
            &cpu.upload(&q),
            &cpu.upload(&k),
            &cpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_ref = cpu.download(&o_cpu);

        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let o_gpu = gpu.causal_attention(
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_cuda = gpu.download(&o_gpu);

        approx_eq(&o_cuda, &o_ref, 5e-2, "TC forward 350m shape");
    }

    #[test]
    fn tc_forward_batched_matches_cpu() {
        // Test batched TC attention
        let seq_len = 32;
        let batch_size = 4;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..batch_size * seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 1.0).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..batch_size * seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..batch_size * seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.3).sin() * 0.5)
            .collect();

        // CPU reference: run each batch separately
        let cpu = CpuDevice::new();
        let mut o_ref = Vec::new();
        for b in 0..batch_size {
            let q_b: Vec<f32> = q[b * seq_len * total_dim..(b + 1) * seq_len * total_dim].to_vec();
            let k_b: Vec<f32> = k[b * seq_len * kv_dim..(b + 1) * seq_len * kv_dim].to_vec();
            let v_b: Vec<f32> = v[b * seq_len * kv_dim..(b + 1) * seq_len * kv_dim].to_vec();
            let o = cpu.causal_attention(
                &cpu.upload(&q_b),
                &cpu.upload(&k_b),
                &cpu.upload(&v_b),
                seq_len,
                n_heads,
                n_kv_heads,
                head_dim,
            );
            o_ref.extend(cpu.download(&o));
        }

        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let o_gpu = gpu.batched_causal_attention(
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            batch_size,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let o_cuda = gpu.download(&o_gpu);

        approx_eq(&o_cuda, &o_ref, 5e-2, "TC batched forward");
    }

    #[test]
    fn tc_forward_bench_350m() {
        // Benchmark TC attention at 350M shape
        let seq_len = 512;
        let n_heads = 16;
        let n_kv_heads = 4;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015).sin() * 0.1)
            .collect();

        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);

        // Warmup
        for _ in 0..5 {
            let _ = gpu.causal_attention(
                &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
        }
        gpu.stream.synchronize().unwrap();

        let iters = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention(
                &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
        }
        gpu.stream.synchronize().unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!(
            "  TC flash forward 350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0,
            iters,
            elapsed
        );
    }

    #[test]
    fn tc_backward_matches_cpu() {
        // Small shape: seq=32, n_heads=4, n_kv_heads=2, head_dim=16
        let seq_len = 32;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 16;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.1 - 1.0).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.15 - 0.3).sin() * 0.1)
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.25 + 0.1).cos() * 0.1)
            .collect();

        // CPU reference
        let cpu = CpuDevice::new();
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &cpu.upload(&grad_out),
            &cpu.upload(&q),
            &cpu.upload(&k),
            &cpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        // GPU with bf16 (triggers TC backward kernel)
        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &gpu.upload(&grad_out),
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        // bf16 backward: tile-local softmax + bf16 quantization → wider tolerance
        approx_eq(&gq_cuda, &gq_ref, 3e-1, "TC grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 3e-1, "TC grad_K");
        approx_eq(&gv_cuda, &gv_ref, 3e-1, "TC grad_V");
    }

    #[test]
    fn tc_backward_matches_cpu_350m_shape() {
        // 350M model shape at reduced seq_len
        let seq_len = 64;
        let n_heads = 16;
        let n_kv_heads = 4;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01 - 2.0).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02 + 1.0).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015 - 0.5).sin() * 0.1)
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.03 + 0.2).cos() * 0.1)
            .collect();

        let cpu = CpuDevice::new();
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &cpu.upload(&grad_out),
            &cpu.upload(&q),
            &cpu.upload(&k),
            &cpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &gpu.upload(&grad_out),
            &gpu.upload(&q),
            &gpu.upload(&k),
            &gpu.upload(&v),
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        // bf16 multi-tile backward with tile-local softmax recompute
        approx_eq(&gq_cuda, &gq_ref, 3e-1, "TC 350m grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 3e-1, "TC 350m grad_K");
        approx_eq(&gv_cuda, &gv_ref, 3e-1, "TC 350m grad_V");
    }

    #[test]
    fn tc_backward_bench_350m() {
        // Benchmark TC backward at actual 350M training shape
        let seq_len = 512;
        let n_heads = 16;
        let n_kv_heads = 4;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i as f32) * 0.015).sin() * 0.1)
            .collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i as f32) * 0.03).cos() * 0.1)
            .collect();

        let gpu = CudaComputeDevice::new_mixed_precision().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);

        // Warmup
        let _ = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu.sync();

        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention_backward(
                &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!(
            "  TC backward 350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0,
            iters,
            elapsed
        );

        // Also bench f32 flash backward for comparison
        let gpu_f32 = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_f32 = gpu_f32.upload(&q);
        let k_f32 = gpu_f32.upload(&k);
        let v_f32 = gpu_f32.upload(&v);
        let go_f32 = gpu_f32.upload(&grad_out);
        let _ = gpu_f32.causal_attention_backward(
            &go_f32, &q_f32, &k_f32, &v_f32, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu_f32.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu_f32.causal_attention_backward(
                &go_f32, &q_f32, &k_f32, &v_f32, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu_f32.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!(
            "  f32 flash backward 350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0,
            iters,
            elapsed
        );
        eprintln!();
    }
}
