//! A thin layer over the CUDA driver API (through cudarc's `sys`), for what tang-compute's
//! single-stream device doesn't expose: several streams, events, raw device addresses, mapped
//! host memory and graph capture.
//!
//! Everything here assumes the context is current on the calling thread
//! ([`Gpu::bind`]). Handles free themselves on drop.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext};

#[derive(Clone, Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error(format!("{e:?}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Turn a `CUresult` into a `Result`, naming the call.
pub fn check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error(format!("{what}: {r:?}")))
    }
}

macro_rules! ck {
    ($e:expr) => {
        $crate::gpu::check(unsafe { $e }, stringify!($e))
    };
}
pub(crate) use ck;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
}

impl Gpu {
    /// Device 0's primary context. Errors (instead of panicking) when there is no driver.
    pub fn new() -> Result<Self> {
        Gpu::at(0)
    }

    /// Device `ordinal`'s primary context (device 0 for [`Gpu::new`]). Errors (instead of
    /// panicking) when there is no driver, or no such device.
    pub fn at(ordinal: usize) -> Result<Self> {
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|_| Error("no CUDA driver".into()))??;
        Ok(Gpu { ctx })
    }

    /// For tests: the GPU, or `None` with a note (panics if `TANG_REQUIRE_CUDA=1`).
    pub fn for_test() -> Option<Self> {
        match Gpu::new() {
            Ok(g) => Some(g),
            Err(e) if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device: {e}")
            }
            Err(e) => {
                eprintln!("no CUDA device ({e}): skipping");
                None
            }
        }
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.ctx.bind_to_thread()?)
    }

    /// `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        ck!(sys::cuMemGetInfo_v2(&mut free, &mut total))?;
        Ok((free, total))
    }

    pub fn attribute(&self, a: sys::CUdevice_attribute) -> Result<i32> {
        Ok(self.ctx.attribute(a)?)
    }

    /// Compile CUDA C with NVRTC for this device's architecture and load it.
    pub fn module(&self, src: &str) -> Result<Module> {
        let major =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        // the toolkit's headers, for sources that include them (`<mma.h>`, `<cuda_fp16.h>`)
        let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let include = format!("{include}/include");
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            include_paths: if src.contains("#include <") && std::path::Path::new(&include).exists()
            {
                vec![include]
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        // NVRTC from an older CUDA doesn't know newer architectures (e.g. B300's sm_103
        // on the CUDA 12.8 wheel). Retry at sm_90, NOT arch-less: arch-less PTX targets
        // the NVRTC default (compute_75 on the 12.8 wheel), where __dp4a is UNDEFINED
        // (the whole point of an explicit arch — tang-compute's flash.rs compiles its
        // dp4a kernels at sm_86 for the same reason). PTX JITs to the actual device at
        // cuModuleLoadData.
        // NVRTC is the expensive part of a load (seconds per big source, twice on an
        // arch the wheel doesn't know). Per-layer owners (GLM-5's 34 resident KDA layers
        // each build three modules) compile identical sources, so memoise the PTX text by
        // (arch, source): the bytes handed to cuModuleLoadData are unchanged.
        static PTX: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<(String, String), String>>,
        > = std::sync::OnceLock::new();
        let cache = PTX.get_or_init(Default::default);
        let key = (arch.to_string(), src.to_string());
        if let Some(p) = cache.lock().unwrap().get(&key).cloned() {
            return self.load_ptx(&p);
        }
        let ptx = match cudarc::nvrtc::compile_ptx_with_opts(src, opts.clone()) {
            Ok(p) => p,
            Err(e)
                if format!("{e:?}").contains("INVALID_OPTION")
                    || format!("{e:?}").contains("invalid value") =>
            {
                let opts = cudarc::nvrtc::CompileOptions {
                    arch: Some("sm_90"),
                    ..opts
                };
                cudarc::nvrtc::compile_ptx_with_opts(src, opts)
                    .map_err(|e| Error(format!("nvrtc: {e:?}")))?
            }
            Err(e) => return Err(Error(format!("nvrtc: {e:?}"))),
        };
        let ptx = ptx.to_src();
        cache.lock().unwrap().insert(key, ptx.clone());
        self.load_ptx(&ptx)
    }

    /// Load PTX compiled elsewhere (e.g. for an arch-specific target such as `sm_90a`).
    pub fn module_from_ptx(&self, ptx: &str) -> Result<Module> {
        self.load_ptx(ptx)
    }

    fn load_ptx(&self, ptx: &str) -> Result<Module> {
        let src = CString::new(ptx).unwrap();
        let mut m: sys::CUmodule = std::ptr::null_mut();
        ck!(sys::cuModuleLoadData(&mut m, src.as_ptr() as *const c_void))?;
        Ok(Module(m))
    }
}

pub struct Module(sys::CUmodule);

impl Module {
    pub fn func(&self, name: &str) -> Result<sys::CUfunction> {
        let c = CString::new(name).unwrap();
        let mut f: sys::CUfunction = std::ptr::null_mut();
        ck!(sys::cuModuleGetFunction(&mut f, self.0, c.as_ptr()))?;
        trace::named(f, name);
        Ok(f)
    }
}

/// Opt-in accounting of every launch, copy and allocation that goes through this module,
/// for profilers (GLM-5's `TANG_GLM5_TRACE`). Off (the default): one relaxed atomic load
/// per call. On ([`trace::enable`]):
///
/// - counters: launches, host→device / device→host bytes and calls, the host time spent in
///   synchronous copies (which includes waiting for the work they order after), and
///   `cuMemAlloc`/`cuMemFree` calls and time;
/// - every [`launch`] outside graph capture gets one timing event recorded after it and is
///   queued (thread-local: one thread drives the GPU) with the caller's current
///   [`trace::tag`], the kernel (named at [`Module::func`]) and an optional byte count the
///   caller declared for it ([`trace::bytes`]). The previous launch's end event on the same
///   stream is the next one's start (the stream orders them), so it is one driver call per
///   launch; [`trace::drain`] resolves the queue into GPU milliseconds, one event sync per
///   stream, not per launch.
///
/// The process drives the GPU from one thread at a time, so a global tag is enough.
pub mod trace {
    use super::sys;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
    use std::sync::Mutex;

    static ON: AtomicBool = AtomicBool::new(false);
    static TAG: AtomicU32 = AtomicU32::new(0);
    static HINT: AtomicU64 = AtomicU64::new(0);

    /// Running totals since the process started tracing (never reset; take deltas).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Snap {
        pub launches: u64,
        pub h2d_bytes: u64,
        pub h2d_calls: u64,
        pub d2h_bytes: u64,
        pub d2h_calls: u64,
        /// Host nanoseconds inside synchronous copies (transfer + the wait they imply).
        pub copy_ns: u64,
        pub allocs: u64,
        pub alloc_bytes: u64,
        /// Host nanoseconds inside cuMemAlloc + cuMemFree.
        pub alloc_ns: u64,
        /// Explicit host waits: `Stream::sync` + `Event::sync` (sync copies are d2h/h2d_calls).
        pub syncs: u64,
        /// Host nanoseconds blocked in those waits.
        pub sync_ns: u64,
    }

    impl Snap {
        /// `self - earlier`, field by field.
        pub fn since(&self, e: &Snap) -> Snap {
            Snap {
                launches: self.launches - e.launches,
                h2d_bytes: self.h2d_bytes - e.h2d_bytes,
                h2d_calls: self.h2d_calls - e.h2d_calls,
                d2h_bytes: self.d2h_bytes - e.d2h_bytes,
                d2h_calls: self.d2h_calls - e.d2h_calls,
                copy_ns: self.copy_ns - e.copy_ns,
                allocs: self.allocs - e.allocs,
                alloc_bytes: self.alloc_bytes - e.alloc_bytes,
                alloc_ns: self.alloc_ns - e.alloc_ns,
                syncs: self.syncs - e.syncs,
                sync_ns: self.sync_ns - e.sync_ns,
            }
        }
        pub fn add(&mut self, o: &Snap) {
            self.launches += o.launches;
            self.h2d_bytes += o.h2d_bytes;
            self.h2d_calls += o.h2d_calls;
            self.d2h_bytes += o.d2h_bytes;
            self.d2h_calls += o.d2h_calls;
            self.copy_ns += o.copy_ns;
            self.allocs += o.allocs;
            self.alloc_bytes += o.alloc_bytes;
            self.alloc_ns += o.alloc_ns;
            self.syncs += o.syncs;
            self.sync_ns += o.sync_ns;
        }
    }

    static C: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
    const LAUNCHES: usize = 0;
    const H2D_B: usize = 1;
    const H2D_N: usize = 2;
    const D2H_B: usize = 3;
    const D2H_N: usize = 4;
    const COPY_NS: usize = 5;
    const ALLOCS: usize = 6;
    const ALLOC_B: usize = 7;
    const ALLOC_NS: usize = 8;
    const SYNCS: usize = 9;
    const SYNC_NS: usize = 10;

    /// One traced launch.
    struct Pending {
        tag: u32,
        func: usize,
        bytes: u64,
        /// (stream, start, end) — `None` inside graph capture (no timing). `start` is the
        /// previous launch's end event on that stream (0: none was recorded, unmeasurable).
        ev: Option<(usize, usize, usize)>,
    }

    thread_local! {
        /// Stream-capture depth ([`super::Graph::capture`] brackets its body): launches
        /// inside it are counted but get no timing events (a recorded event would be
        /// captured into the graph).
        static CAPTURE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        /// This thread's queued launches (the GPU is driven from one thread at a time).
        static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
        /// Reusable timing events (no context: one event works across contexts).
        static POOL: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
        /// Per stream: the last end event recorded there (the next launch's start), kept
        /// across drains (0: none).
        static LAST: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());
    }
    static NAMES: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);

    /// A resolved launch: its tag, kernel, declared bytes and GPU time (`None`: captured).
    #[derive(Clone, Debug)]
    pub struct Launch {
        pub tag: u32,
        pub func: usize,
        pub bytes: u64,
        pub gpu_ms: Option<f32>,
    }

    /// Turn tracing on or off (process-wide).
    pub fn enable(on: bool) {
        ON.store(on, Relaxed);
    }

    /// Bracket a graph capture: launches inside get no timing events (counted only).
    /// [`super::Graph::capture`] calls this around its body; nesting is fine.
    pub fn capture(on: bool) {
        CAPTURE.set(if on {
            CAPTURE.get() + 1
        } else {
            CAPTURE.get() - 1
        });
    }

    #[inline(always)]
    pub fn on() -> bool {
        ON.load(Relaxed)
    }

    /// Set the tag later launches are attributed to; returns the previous one.
    pub fn tag(t: u32) -> u32 {
        TAG.swap(t, Relaxed)
    }

    /// Declare the bytes the NEXT launch reads + writes (for GB/s; consumed by it).
    #[inline(always)]
    pub fn bytes(n: u64) {
        if on() {
            HINT.store(n, Relaxed);
        }
    }

    /// A host wait ended (`t0`: when it began).
    pub fn sync(t0: std::time::Instant) {
        C[SYNCS].fetch_add(1, Relaxed);
        C[SYNC_NS].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }

    pub fn snap() -> Snap {
        let g = |i: usize| C[i].load(Relaxed);
        Snap {
            launches: g(LAUNCHES),
            h2d_bytes: g(H2D_B),
            h2d_calls: g(H2D_N),
            d2h_bytes: g(D2H_B),
            d2h_calls: g(D2H_N),
            copy_ns: g(COPY_NS),
            allocs: g(ALLOCS),
            alloc_bytes: g(ALLOC_B),
            alloc_ns: g(ALLOC_NS),
            syncs: g(SYNCS),
            sync_ns: g(SYNC_NS),
        }
    }

    /// The kernel's name as given to [`super::Module::func`] (or its address).
    pub fn name(func: usize) -> String {
        NAMES
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(&func).cloned())
            .unwrap_or_else(|| format!("fn@{func:#x}"))
    }

    pub(super) fn named(f: sys::CUfunction, name: &str) {
        if !on() {
            return;
        }
        let mut g = NAMES.lock().unwrap();
        g.get_or_insert_with(HashMap::new)
            .entry(f as usize)
            .or_insert_with(|| name.to_string());
    }

    pub(super) fn copy(h2d: bool, bytes: usize, t0: Option<std::time::Instant>) {
        let (b, n) = if h2d { (H2D_B, H2D_N) } else { (D2H_B, D2H_N) };
        C[b].fetch_add(bytes as u64, Relaxed);
        C[n].fetch_add(1, Relaxed);
        if let Some(t0) = t0 {
            C[COPY_NS].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        }
    }

    pub(super) fn alloc(bytes: usize, t0: std::time::Instant) {
        C[ALLOCS].fetch_add(1, Relaxed);
        C[ALLOC_B].fetch_add(bytes as u64, Relaxed);
        C[ALLOC_NS].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }

    pub(super) fn free(t0: std::time::Instant) {
        C[ALLOC_NS].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }

    fn exact() -> bool {
        static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *E.get_or_init(|| std::env::var("TANG_TRACE_EXACT").is_ok_and(|v| v == "1"))
    }

    fn event() -> Option<usize> {
        let from_pool = POOL.with_borrow_mut(|p| p.pop());
        if from_pool.is_some() {
            return from_pool;
        }
        let mut e: sys::CUevent = std::ptr::null_mut();
        let r = unsafe { sys::cuEventCreate(&mut e, sys::CUevent_flags::CU_EVENT_DEFAULT as u32) };
        (r == sys::CUresult::CUDA_SUCCESS).then_some(e as usize)
    }

    /// Before a launch on `s`: the start event (outside capture), the kernel's declared bytes.
    pub(super) fn before(s: sys::CUstream) -> u64 {
        C[LAUNCHES].fetch_add(1, Relaxed);
        let bytes = HINT.swap(0, Relaxed);
        let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let capturing = unsafe { sys::cuStreamIsCapturing(s, &mut st) }
            != sys::CUresult::CUDA_SUCCESS
            || st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // a stream's first traced launch since tracing started has no base event: record
        // one (the next launch measures from it); graph capture gets nothing recorded
        // TANG_TRACE_EXACT=1: a fresh start event right before every launch, so a launch's
        // time is its own execution — the default (start = the previous launch's end on
        // the stream) also counts the host gap before it, which overstates GPU busy
        // whenever the host is the bottleneck
        if !capturing && (exact() || LAST.with_borrow(|m| !m.contains_key(&(s as usize)))) {
            if let Some(e) = event() {
                unsafe { sys::cuEventRecord(e as sys::CUevent, s) };
                // the replaced base (exact mode) is still some pending launch's end event:
                // drain() returns those to the pool, so just overwrite the slot
                LAST.with_borrow_mut(|m| m.insert(s as usize, e));
            }
        }
        bytes
    }

    pub(super) fn after(f: sys::CUfunction, s: sys::CUstream, bytes: u64) {
        // the previous end event on this stream is this launch's start; record ours
        let ev = LAST.with_borrow_mut(|m| {
            let prev = m.get(&(s as usize)).copied().unwrap_or(0);
            let Some(e1) = event() else { return None };
            unsafe { sys::cuEventRecord(e1 as sys::CUevent, s) };
            m.insert(s as usize, e1);
            (prev != 0).then_some((s as usize, prev, e1))
        });
        PENDING.with_borrow_mut(|q| {
            q.push(Pending {
                tag: TAG.load(Relaxed),
                func: f as usize,
                bytes,
                ev,
            })
        });
    }

    /// Resolve every queued launch (waits for each one's end event) and return them in
    /// launch order; their events go back to the pool.
    pub fn drain() -> Vec<Launch> {
        let pend = PENDING.take();
        // the last end event per stream: sync it once — everything earlier on that stream
        // is complete by then
        let mut last: HashMap<usize, usize> = HashMap::new();
        for p in &pend {
            if let Some((s, _, e1)) = p.ev {
                last.insert(s, e1);
            }
        }
        for &e1 in last.values() {
            unsafe { sys::cuEventSynchronize(e1 as sys::CUevent) };
        }
        let mut out = Vec::with_capacity(pend.len());
        let mut back = Vec::with_capacity(pend.len());
        for p in pend {
            let gpu_ms = p.ev.and_then(|(stream, e0, e1)| {
                unsafe {
                    let mut ms = 0f32;
                    let ok =
                        sys::cuEventElapsedTime(&mut ms, e0 as sys::CUevent, e1 as sys::CUevent)
                            == sys::CUresult::CUDA_SUCCESS;
                    back.push(e1); // the start event stays as the stream's base
                    if exact() {
                        back.push(e0); // exact mode: each launch's own start event
                    }
                    ok.then_some(ms)
                }
            });
            out.push(Launch {
                tag: p.tag,
                func: p.func,
                bytes: p.bytes,
                gpu_ms,
            });
        }
        POOL.with_borrow_mut(|p| p.extend(back));
        out
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            sys::cuModuleUnload(self.0);
        }
    }
}

pub struct Stream(pub sys::CUstream);

impl Stream {
    /// A non-blocking stream: does not synchronise with the legacy default stream, so it is
    /// capturable ([`Graph::capture`] requires it) — but a preceding synchronous
    /// `DevBuf::write`/`from_slice` HtoD is NOT ordered before work on it. Never use for
    /// data-path work that consumes host-staged buffers: use [`Stream::blocking`].
    pub fn new() -> Result<Self> {
        let mut s: sys::CUstream = std::ptr::null_mut();
        ck!(sys::cuStreamCreate(
            &mut s,
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32
        ))?;
        Ok(Stream(s))
    }

    /// A blocking stream (flags 0): orders against the legacy default stream, so a
    /// preceding synchronous `DevBuf::write`/`from_slice` HtoD is guaranteed visible to
    /// kernels launched on it. `DevBuf::write`/`from_slice` copy with the synchronous
    /// `cuMemcpyHtoD_v2`, which the driver enqueues on the legacy default stream — a
    /// `CU_STREAM_NON_BLOCKING` stream does not order against it, so a kernel launched
    /// right after an upload could read the buffer before the copy lands: rare,
    /// position-varying wrong outputs that stick via carried state (observed on mew as
    /// engine-vs-CPU parity flipping run-to-run, 13/170 → 161/170; see frog/tree-eval
    /// `eval/tree/stream-blocking.md`). Use this for every stream that reads host-staged
    /// buffers.
    pub fn blocking() -> Result<Self> {
        let mut s: sys::CUstream = std::ptr::null_mut();
        unsafe {
            ck!(sys::cuStreamCreate(&mut s, 0))?;
        }
        Ok(Stream(s))
    }

    /// Whether this stream orders against the legacy default stream (created by
    /// [`Stream::blocking`], not [`Stream::new`]'s `NON_BLOCKING`). Wrappers that stage
    /// inputs with the synchronous `DevBuf::write`/`from_slice` assert it.
    pub fn is_blocking(&self) -> bool {
        let mut f: std::ffi::c_uint = 0;
        let r = unsafe { sys::cuStreamGetFlags(self.0, &mut f) };
        r == sys::CUresult::CUDA_SUCCESS
            && f & sys::CUstream_flags::CU_STREAM_NON_BLOCKING as std::ffi::c_uint == 0
    }

    pub fn sync(&self) -> Result<()> {
        let t0 = trace::on().then(std::time::Instant::now);
        let r = ck!(sys::cuStreamSynchronize(self.0));
        if let Some(t0) = t0 {
            trace::sync(t0);
        }
        r
    }

    /// True when all work is done (`cuStreamQuery`).
    pub fn idle(&self) -> bool {
        unsafe { sys::cuStreamQuery(self.0) == sys::CUresult::CUDA_SUCCESS }
    }

    pub fn wait(&self, e: &Event) -> Result<()> {
        ck!(sys::cuStreamWaitEvent(self.0, e.0, 0))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.0);
        }
    }
}

pub struct Event(pub sys::CUevent);

impl Event {
    pub fn new(timing: bool) -> Result<Self> {
        let flags = if timing {
            sys::CUevent_flags::CU_EVENT_DEFAULT
        } else {
            sys::CUevent_flags::CU_EVENT_DISABLE_TIMING
        };
        let mut e: sys::CUevent = std::ptr::null_mut();
        ck!(sys::cuEventCreate(&mut e, flags as u32))?;
        Ok(Event(e))
    }

    pub fn record(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuEventRecord(self.0, s.0))
    }

    pub fn sync(&self) -> Result<()> {
        let t0 = trace::on().then(std::time::Instant::now);
        let r = ck!(sys::cuEventSynchronize(self.0));
        if let Some(t0) = t0 {
            trace::sync(t0);
        }
        r
    }

    pub fn done(&self) -> bool {
        unsafe { sys::cuEventQuery(self.0) == sys::CUresult::CUDA_SUCCESS }
    }

    /// Milliseconds from `start` to this event.
    pub fn since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        ck!(sys::cuEventElapsedTime(&mut ms, start.0, self.0))?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            sys::cuEventDestroy_v2(self.0);
        }
    }
}

/// A raw device allocation.
pub struct DevBuf {
    pub ptr: u64,
    pub len: usize,
}

impl DevBuf {
    /// GOTCHA (cost a glm5 first-tokens pod): `alloc`/`zeroed` take BYTES, but `read::<T>`/
    /// `write::<T>` take ELEMENT counts. Calling `zeroed(n * size_of::<T>())` and then
    /// `read::<T>` is fine, but passing an element count to `zeroed` (or bytes to `read`)
    /// leaves the buffer 4× (or 4× too small) and the kernel writes past the allocation —
    /// the assert fires later, in an unrelated read. When in doubt: `alloc_bytes(n * size)` +
    /// `read::<T>(0, n)` never mix.
    pub fn alloc(len: usize) -> Result<Self> {
        let t0 = trace::on().then(std::time::Instant::now);
        let mut p: sys::CUdeviceptr = 0;
        ck!(sys::cuMemAlloc_v2(&mut p, len.max(1)))?;
        if let Some(t0) = t0 {
            trace::alloc(len, t0);
        }
        Ok(DevBuf { ptr: p, len })
    }

    pub fn zeroed(len: usize) -> Result<Self> {
        let b = Self::alloc(len)?;
        ck!(sys::cuMemsetD8_v2(b.ptr, 0, len.max(1)))?;
        Ok(b)
    }

    /// `zeroed_elems::<T>(n)` zeroes n elements of T (n * size_of::<T>() bytes) — the
    /// clearer form for "a kernel will write n floats here" buffers.
    pub fn zeroed_elems<T>(n: usize) -> Result<Self> {
        Self::zeroed(n * std::mem::size_of::<T>())
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let b = Self::alloc(std::mem::size_of_val(data))?;
        b.write(0, data)?;
        Ok(b)
    }

    /// Synchronous copy of `data` to byte offset `off`.
    pub fn write<T: Copy>(&self, off: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(off + n <= self.len);
        if n == 0 {
            return Ok(());
        }
        let t0 = trace::on().then(std::time::Instant::now);
        let r = ck!(sys::cuMemcpyHtoD_v2(
            self.ptr + off as u64,
            data.as_ptr() as *const c_void,
            n
        ));
        if t0.is_some() {
            trace::copy(true, n, t0);
        }
        r
    }

    /// Async copy from pageable `data` (the driver stages it before returning).
    pub fn write_async<T: Copy>(&self, off: usize, data: &[T], s: &Stream) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(off + n <= self.len);
        if n == 0 {
            return Ok(());
        }
        if trace::on() {
            trace::copy(true, n, None);
        }
        ck!(sys::cuMemcpyHtoDAsync_v2(
            self.ptr + off as u64,
            data.as_ptr() as *const c_void,
            n,
            s.0
        ))
    }

    pub fn read<T: Copy + Default>(&self, off: usize, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        let bytes = n * std::mem::size_of::<T>();
        assert!(off + bytes <= self.len);
        if bytes > 0 {
            let t0 = trace::on().then(std::time::Instant::now);
            ck!(sys::cuMemcpyDtoH_v2(
                v.as_mut_ptr() as *mut c_void,
                self.ptr + off as u64,
                bytes
            ))?;
            if t0.is_some() {
                trace::copy(false, bytes, t0);
            }
        }
        Ok(v)
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        let t0 = trace::on().then(std::time::Instant::now);
        unsafe {
            sys::cuMemFree_v2(self.ptr);
        }
        if let Some(t0) = t0 {
            trace::free(t0);
        }
    }
}

/// Async device-to-device copy of `bytes` from `src` to `dst` on `s` (capturable as a
/// graph memcpy node). Both are raw device addresses the caller keeps valid.
pub fn copy_d2d_async(dst: u64, src: u64, bytes: usize, s: &Stream) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    ck!(sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, s.0))
}

/// Launch `f`. `args` are pointers to each kernel argument's value.
///
/// # Safety
/// `args` must match the kernel's parameter list in count and types, and every device
/// address passed must be valid for what the kernel does with it.
pub unsafe fn launch(
    f: sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    s: &Stream,
    args: &mut [*mut c_void],
) -> Result<()> {
    if trace::on() {
        let bytes = trace::before(s.0);
        let r = launch_raw(f, grid, block, smem, s, args);
        trace::after(f, s.0, bytes);
        return r;
    }
    launch_raw(f, grid, block, smem, s, args)
}

unsafe fn launch_raw(
    f: sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    s: &Stream,
    args: &mut [*mut c_void],
) -> Result<()> {
    check(
        sys::cuLaunchKernel(
            f,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            smem,
            s.0,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ),
        "cuLaunchKernel",
    )
}

/// Kernel arguments: `args![a, b, c]` gives the `&mut [*mut c_void]` that [`launch`] takes.
/// The values must outlive the launch call (they are copied at launch).
#[macro_export]
macro_rules! args {
    ($($a:expr),* $(,)?) => {
        &mut [$(&$a as *const _ as *mut ::std::ffi::c_void),*]
    };
}

/// An instantiated CUDA graph.
pub struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}

impl Graph {
    /// Record everything `body` enqueues on `s` (which must not be the legacy default stream)
    /// and instantiate it.
    pub fn capture(s: &Stream, body: impl FnOnce(&Stream) -> Result<()>) -> Result<Self> {
        trace::capture(true);
        let out = Self::capture_body(s, body);
        trace::capture(false);
        out
    }

    fn capture_body(s: &Stream, body: impl FnOnce(&Stream) -> Result<()>) -> Result<Self> {
        ck!(sys::cuStreamBeginCapture_v2(
            s.0,
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        ))?;
        let body_result = body(s);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end = ck!(sys::cuStreamEndCapture(s.0, &mut graph));
        body_result?;
        end?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        ck!(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0))?;
        ck!(sys::cuGraphUpload(exec, s.0))?;
        Ok(Graph { graph, exec })
    }

    pub fn launch(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuGraphLaunch(self.exec, s.0))
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}
