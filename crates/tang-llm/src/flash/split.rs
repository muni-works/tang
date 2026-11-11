//! The two-GPU layer split (`TANG_GPUS=2`): GPU 1's context and the peer-access setup.
//! The stage wiring itself is in `coder.rs` ([`Peer`], the stage-dispatched enqueue
//! functions and `enqueue_boundary`); this module keeps the driver-API plumbing separate:
//!
//! - GPU 1's primary context and its own stream ([`Ctx::for_device`]);
//! - enabling peer access both ways ([`Ctx::enable_peer`]), which the boundary copies and
//!   the cross-stage `attn_norm` reads need (an NVLink pair like 2×H200 has it; a kernel
//!   *reading* the other device's pointer cannot work without it, so the split requires
//!   it as built);
//! - `TANG_GPUS` parsing (the split point comes from the GGUF's per-layer bytes,
//!   `coder::split_point`, or `TANG_SPLIT_AT`).
//!
//! `TANG_GPUS=1` (the default) builds none of this: the same kernels on the same stream,
//! byte-for-byte the pre-split engine.

use anyhow::{anyhow, Result};
use tang_moe::gpu::{Gpu, Stream};

/// GPU 1's CUDA objects: its driver-API wrapper (the context bound to the thread when
/// loading and launching) and its stream. GPU 0's equivalents are the engine's own
/// `CudaComputeDevice` + `Gpu` + legacy stream; the peer stage launches on this `Ctx`'s
/// stream with kernels its own context compiled (`Kern::build_on`).
pub struct Ctx {
    pub gpu: Gpu,
    pub stream: Stream,
}

impl Ctx {
    /// GPU `ordinal`'s primary context (the split uses 1; 0 is the engine's own).
    pub fn for_device(ordinal: usize) -> Result<Self> {
        let gpu = Gpu::at(ordinal).map_err(|e| anyhow!("CUDA device {ordinal}: {e}"))?;
        gpu.bind().map_err(|e| anyhow!("{e}"))?;
        let stream = Stream::new().map_err(|e| anyhow!("{e}"))?;
        Ok(Ctx { gpu, stream })
    }

    /// Enable peer access GPU 1 ↔ GPU 0 (`cuCtxEnablePeerAccess` from each side's own
    /// context), so GPU 0's kernels can read GPU 1's weights (the first `attn_norm` at
    /// the boundary) and the boundary copies can be `cuMemcpyPeerAsync`. Returns whether
    /// the devices report mutual access (reported only — the split requires it either way).
    pub fn enable_peer(&self, other: &Gpu) -> Result<bool> {
        use cudarc::driver::sys;
        let mut a = 0i32;
        unsafe {
            let _ =
                sys::cuDeviceCanAccessPeer(&mut a, self.gpu.ctx.cu_device(), other.ctx.cu_device());
            // Enable ours→theirs from our context, theirs→ours from theirs.
            let _ = sys::cuCtxEnablePeerAccess(other.ctx.cu_ctx(), 0);
            other.bind().map_err(|e| anyhow!("{e}"))?;
            let _ = sys::cuCtxEnablePeerAccess(self.gpu.ctx.cu_ctx(), 0);
            self.gpu.bind().map_err(|e| anyhow!("{e}"))?;
        }
        Ok(a != 0)
    }

    /// Bind and synchronise GPU 1's context.
    pub fn sync(&self) -> Result<()> {
        self.gpu.ctx.synchronize().map_err(|e| anyhow!("{e:?}"))
    }
}

/// How many GPUs the engine uses: `TANG_GPUS` (default 1, clamped to 2).
pub fn gpus() -> usize {
    std::env::var("TANG_GPUS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .clamp(1, 2)
}
