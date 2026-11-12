//! CUDA device for tensor-IR programs (feature `cuda`).
//!
//! The driver (`libcuda`) and nvrtc are loaded at run time by cudarc, so
//! a build with this feature runs on machines without NVIDIA hardware:
//! [`CudaDevice::new`] returns `None` there and
//! [`Runtime::select`](crate::program::Runtime::select) falls back to wgpu.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// The cudarc version this backend is built on, for callers that need
/// the driver directly.
pub use cudarc;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// One CUDA context and its stream, plus compiled kernels keyed by source
/// (a program lowered twice for tuning compiles each kernel once).
pub struct CudaDevice {
    pub ctx: Arc<CudaContext>,
    /// Every launch, copy and readback goes on this stream, in order.
    pub stream: Arc<CudaStream>,
    /// Device name, e.g. `"NVIDIA GeForce RTX 3090 (CUDA sm_86)"`.
    pub name: String,
    /// nvrtc target, e.g. `"sm_86"`, from the device's compute capability.
    pub arch: &'static str,
    functions: RefCell<HashMap<String, CudaFunction>>,
}

/// FNV-1a, for cache file names stable across processes and builds.
fn fnv64(parts: &[&str]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        for b in p.bytes().chain([0xff]) {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The on-disk PTX cache: `$TANG_PTX_CACHE`, else
/// `$XDG_CACHE_HOME/tang/ptx` or `~/.cache/tang/ptx`; `None` (no disk
/// cache) when `TANG_PTX_CACHE` is empty or there is no home.
pub fn ptx_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    match std::env::var_os("TANG_PTX_CACHE") {
        Some(d) if d.is_empty() => None,
        Some(d) => Some(PathBuf::from(d)),
        None => std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .map(|c| c.join("tang/ptx")),
    }
}

/// `<cache>/<hash>.ptx` for `source` on `arch`, with the source beside it
/// (`.cu`) to rule out hash collisions.
fn ptx_paths(source: &str, arch: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let (major, minor) = nvrtc_version();
    let v = format!("{major}.{minor}");
    let stem = format!("{:016x}", fnv64(&[source, arch, &v]));
    let dir = ptx_cache_dir()?;
    Some((
        dir.join(format!("{stem}.ptx")),
        dir.join(format!("{stem}.cu")),
    ))
}

fn nvrtc_version() -> (i32, i32) {
    let (mut major, mut minor) = (0, 0);
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    (major, minor)
}

/// Whether [`ptx`] would compile `source` (it is not on disk).
pub fn ptx_cached(source: &str, arch: &str) -> bool {
    ptx_paths(source, arch).is_some_and(|(ptx, cu)| {
        ptx.exists() && std::fs::read_to_string(cu).is_ok_and(|s| s == source)
    })
}

/// nvrtc every distinct source in `sources` not yet in the on-disk PTX
/// cache, on all cores. Needs no device (so it can run outside a GPU
/// lock); [`CudaDevice::function`] then only loads.
pub fn precompile<'a>(sources: impl IntoIterator<Item = &'a str>, arch: &'static str) {
    let mut todo: Vec<&str> = sources.into_iter().collect();
    todo.sort();
    todo.dedup();
    todo.retain(|s| !ptx_cached(s, arch));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(todo.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(src) = todo.get(i) else { break };
                ptx(src, arch);
            });
        }
    });
}

/// PTX for `source` on `arch` (e.g. `"sm_86"`): from the on-disk cache
/// ([`ptx_cache_dir`]), else nvrtc, written back. Needs no device, so a
/// caller can fill the cache from many threads up front.
pub fn ptx(source: &str, arch: &'static str) -> String {
    let paths = ptx_paths(source, arch);
    if let Some((ptx, cu)) = &paths {
        if std::fs::read_to_string(cu).is_ok_and(|s| s == source) {
            if let Ok(p) = std::fs::read_to_string(ptx) {
                return p;
            }
        }
    }
    let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![format!("{include}/include")],
        ..Default::default()
    };
    let out = compile_ptx_with_opts(source, opts)
        .unwrap_or_else(|e| panic!("nvrtc: {e:?}\n{source}"))
        .to_src();
    if let Some((ptx, cu)) = paths {
        // write-then-rename so concurrent writers never expose a torn file
        let tmp = |p: &std::path::Path| {
            p.with_extension(format!(
                "tmp{}.{:?}",
                std::process::id(),
                std::thread::current().id()
            ))
        };
        let put = |p: &std::path::Path, data: &str| -> std::io::Result<()> {
            let t = tmp(p);
            std::fs::write(&t, data)?;
            std::fs::rename(&t, p)
        };
        let _ = std::fs::create_dir_all(ptx.parent().unwrap())
            .and_then(|_| put(&ptx, &out))
            .and_then(|_| put(&cu, source));
    }
    out
}

/// `#define TANG_SMEM n` in a kernel's source: its dynamic shared memory
/// in bytes (0 when absent).
pub fn dynamic_smem(src: &str) -> u32 {
    src.lines()
        .find_map(|l| l.strip_prefix("#define TANG_SMEM "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

impl CudaDevice {
    /// Device `ordinal`, or `None` when there is no driver or no device.
    /// Never panics: cudarc panics when `libcuda` is missing, and that is
    /// caught here.
    pub fn new(ordinal: usize) -> Option<Self> {
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal).ok())
            .ok()
            .flatten()?;
        let stream = ctx.default_stream();
        Self::from_stream(ctx, stream)
    }

    /// Borrow an existing cudarc context and stream (e.g. phyz-gpu's or
    /// the host application's) so tang programs run on the same context,
    /// in order with the caller's work on `stream`, and can read and write
    /// the caller's `CudaSlice`s without copies. `stream` must belong to
    /// `ctx`. `None` if the device attributes cannot be queried.
    ///
    /// For [`Program::capture`](crate::program::Program::capture) the
    /// stream must not be the legacy default stream and the context must
    /// have event tracking disabled (see [`Self::capturable`]); that is
    /// the caller's choice, since it affects their allocations too.
    pub fn from_stream(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Option<Self> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let major = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .ok()?;
        let minor = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .ok()?;
        // Hopper builds for sm_90a: wgmma and setmaxnreg are arch-specific
        let suffix = if major == 9 { "a" } else { "" };
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}{suffix}").into_boxed_str());
        let name = format!("{} (CUDA {arch})", ctx.name().ok()?);
        Some(Self {
            ctx,
            stream,
            name,
            arch,
            functions: RefCell::new(HashMap::new()),
        })
    }

    /// [`Self::from_stream`] on `ctx`'s default stream.
    pub fn from_context(ctx: Arc<CudaContext>) -> Option<Self> {
        let stream = ctx.default_stream();
        Self::from_stream(ctx, stream)
    }

    /// Device `ordinal` set up for CUDA Graph capture
    /// ([`Program::capture`](crate::program::Program::capture)): work goes
    /// on a stream of its own (the legacy default stream cannot be
    /// captured) and cudarc's cross-stream event tracking is off (its
    /// events would break capture; everything here is on one stream, so
    /// there is nothing to order).
    pub fn new_capturable(ordinal: usize) -> Option<Self> {
        let mut d = Self::new(ordinal)?;
        // SAFETY: every allocation of this device is used only on `stream`
        unsafe { d.ctx.disable_event_tracking() };
        d.stream = d.ctx.new_stream().ok()?;
        Some(d)
    }

    /// Whether [`Program::capture`](crate::program::Program::capture)
    /// can capture on this device's stream.
    pub fn capturable(&self) -> bool {
        !self.stream.cu_stream().is_null() && !self.ctx.is_event_tracking()
    }

    /// nvrtc can be loaded too (a driver without the toolkit cannot
    /// compile programs).
    pub fn has_nvrtc() -> bool {
        std::panic::catch_unwind(|| {
            let (mut major, mut minor) = (0, 0);
            let r = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
            r == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
        })
        .unwrap_or(false)
    }

    /// The program entry point of `source`, compiled with nvrtc (cached).
    /// Kernels that want more than 48 KB of dynamic shared memory (their
    /// `TANG_SMEM`) get the attribute set.
    pub fn function(&self, source: &str) -> CudaFunction {
        if let Some(f) = self.functions.borrow().get(source) {
            return f.clone();
        }
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx(source, self.arch));
        let module = self.ctx.load_module(ptx).expect("load module");
        let f = module
            .load_function(tang_compile::tensor::cuda::CUDA_ENTRY)
            .expect("entry");
        let smem = dynamic_smem(source);
        if smem > 48 * 1024 {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .expect("smem attribute");
        }
        self.functions
            .borrow_mut()
            .insert(source.to_string(), f.clone());
        f
    }

    /// Wait for everything queued on the stream.
    pub fn sync(&self) {
        self.stream.synchronize().expect("cuda sync");
    }
}

static PDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Programmatic dependent launch for derived programs on `arch`: sm_90+
/// unless `TANG_PDL=0`. Remembered for [`pdl_active`] (the launches of
/// the kernels compiled with [`with_griddep`]).
pub fn pdl_on(arch: &str) -> bool {
    let cc: u32 = arch
        .trim_start_matches("sm_")
        .trim_end_matches(|c: char| c.is_alphabetic())
        .parse()
        .unwrap_or(0);
    let on = cc >= 90 && !std::env::var("TANG_PDL").is_ok_and(|v| v == "0");
    PDL.store(on, std::sync::atomic::Ordering::Relaxed);
    on
}

/// Whether derived kernels launch with programmatic stream serialization.
pub fn pdl_active() -> bool {
    PDL.load(std::sync::atomic::Ordering::Relaxed)
}

/// `src` (a kernel with entry point `main0`) that first lets its
/// dependents launch (`griddepcontrol.launch_dependents`), then waits for
/// every grid it depends on to complete and flush
/// (`griddepcontrol.wait`): its launch and block scheduling overlap the
/// producer's tail, its reads still see all of the producer's writes.
pub fn with_griddep(src: &str) -> String {
    let Some(at) = src.rfind("main0(") else {
        return src.to_string();
    };
    let Some(open) = src[at..].find('{').map(|i| at + i) else {
        return src.to_string();
    };
    let mut out = String::with_capacity(src.len() + 200);
    out.push_str(&src[..=open]);
    out.push_str(
        "\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n    asm volatile(\"griddepcontrol.launch_dependents;\");\n    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n#endif\n",
    );
    out.push_str(&src[open + 1..]);
    out
}
