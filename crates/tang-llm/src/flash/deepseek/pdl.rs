//! Programmatic dependent launch (`DS_PDL=1`): a launch with
//! `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` may start once its predecessor
//! has run `griddepcontrol.launch_dependents` (or finished); the kernel then runs its
//! independent prologue (its weight loads) and `griddepcontrol.wait`s before touching the
//! predecessor's outputs. sm_90+; the kernels compile the PTX away below it.

use cudarc::driver::sys;
use std::ffi::c_void;
use tang_moe::gpu::{self, Stream};

/// Whether `DS_PDL=1`.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PDL").is_ok_and(|v| v == "1"))
}

/// [`gpu::launch`] with the programmatic-serialization attribute when `pdl`.
///
/// # Safety
/// As [`gpu::launch`]; with `pdl`, `f` must `griddepcontrol.wait` before reading anything its
/// stream predecessor writes.
pub unsafe fn launch(
    f: sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    s: &Stream,
    args: &mut [*mut c_void],
    pdl: bool,
) -> gpu::Result<()> {
    if !pdl {
        return gpu::launch(f, grid, block, smem, s, args);
    }
    let mut attr: sys::CUlaunchAttribute = std::mem::zeroed();
    attr.id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
    let cfg = sys::CUlaunchConfig {
        gridDimX: grid.0,
        gridDimY: grid.1,
        gridDimZ: grid.2,
        blockDimX: block.0,
        blockDimY: block.1,
        blockDimZ: block.2,
        sharedMemBytes: smem,
        hStream: s.0,
        attrs: &mut attr,
        numAttrs: 1,
    };
    gpu::check(
        sys::cuLaunchKernelEx(&cfg, f, args.as_mut_ptr(), std::ptr::null_mut()),
        "cuLaunchKernelEx",
    )
}
