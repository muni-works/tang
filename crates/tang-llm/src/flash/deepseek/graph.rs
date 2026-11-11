//! Per-step scratch and CUDA-graph plumbing for [`super::engine::GpuModel`].
//!
//! - [`Arena`]: every temporary of a step is bumped out of one persistent device block and
//!   the whole step is released at the next step's start. Nothing is allocated or freed on
//!   the token path, so no host sync is needed for buffer lifetimes: reuse is ordered by the
//!   one stream. A step that overflows the block gets extra blocks; the next step's start
//!   drains the stream and folds them into one bigger block, which moves every address
//!   ([`Arena::gen`] counts these; a captured graph is only valid for its generation).
//! - [`Staging`]: pinned host memory for a step's host inputs (token ids, Engram rows, the
//!   step parameters), copied with `cuMemcpyHtoDAsync` on the step's stream, so a captured
//!   graph holds the copies as memcpy nodes and the host refills the same pinned bytes
//!   before each launch.
//! - [`GKernels`]: kernels_graph.cu, the position-dependent kernels reading the position
//!   from the device.

use anyhow::{bail, Context, Result};
use cudarc::driver::sys;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

pub const SRC: &str = include_str!("kernels_graph.cu");

/// A view of device memory: arena scratch (valid until the next step starts) or a
/// persistent [`DevBuf`]'s.
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    pub ptr: u64,
    pub len: usize,
}

impl Buf {
    /// The view `bytes` in.
    pub fn at(self, bytes: usize) -> Buf {
        Buf {
            ptr: self.ptr + bytes as u64,
            len: self.len.saturating_sub(bytes),
        }
    }

    /// As a [`DevBuf`] that is never freed (for APIs that take one).
    pub fn dev(self) -> ManuallyDrop<DevBuf> {
        ManuallyDrop::new(DevBuf {
            ptr: self.ptr,
            len: self.len,
        })
    }
}

impl From<&DevBuf> for Buf {
    fn from(d: &DevBuf) -> Buf {
        Buf {
            ptr: d.ptr,
            len: d.len,
        }
    }
}

const ALIGN: usize = 256;

#[derive(Default)]
pub struct Arena {
    main: Option<DevBuf>,
    off: usize,
    extra: Vec<DevBuf>,
    extra_bytes: usize,
    /// Bumped whenever the block moves.
    pub gen: u64,
    /// While capturing: overflowing is an error (cuMemAlloc can't be captured).
    pub frozen: bool,
}

impl Arena {
    pub fn alloc(&mut self, bytes: usize) -> Result<Buf> {
        let n = bytes.max(1).next_multiple_of(ALIGN);
        if let Some(m) = &self.main {
            if self.off + n <= m.len {
                let b = Buf {
                    ptr: m.ptr + self.off as u64,
                    len: bytes,
                };
                self.off += n;
                return Ok(b);
            }
        }
        if self.frozen {
            bail!("arena overflow while capturing");
        }
        let d = DevBuf::alloc(n)?;
        super::obs::alloc();
        let b = Buf {
            ptr: d.ptr,
            len: bytes,
        };
        self.extra_bytes += n;
        self.extra.push(d);
        Ok(b)
    }

    /// Start a step: the stream must be drained (frees don't wait for it).
    pub fn reset(&mut self) -> Result<()> {
        if !self.extra.is_empty() {
            let cap = self.main.as_ref().map_or(0, |m| m.len) + self.extra_bytes;
            self.extra.clear();
            self.main = None;
            self.main = Some(DevBuf::alloc((cap + cap / 2).next_multiple_of(ALIGN))?);
            super::obs::alloc();
            self.extra_bytes = 0;
            self.gen += 1;
        }
        self.off = 0;
        Ok(())
    }

    /// Whether this step overflowed (the next start moves the block).
    pub fn overflowed(&self) -> bool {
        !self.extra.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.main.as_ref().map_or(0, |m| m.len)
    }
}

struct Pinned(*mut u8, usize);

impl Pinned {
    fn new(n: usize) -> Result<Self> {
        super::obs::alloc();
        let mut p: *mut c_void = std::ptr::null_mut();
        gpu::check(
            unsafe { sys::cuMemAllocHost_v2(&mut p, n.max(1)) },
            "cuMemAllocHost",
        )?;
        Ok(Pinned(p as *mut u8, n))
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemFreeHost(self.0 as *mut c_void);
        }
    }
}

/// Pinned host staging, bumped per step like [`Arena`].
#[derive(Default)]
pub struct Staging {
    main: Option<Pinned>,
    off: usize,
    extra: Vec<Pinned>,
    extra_bytes: usize,
    pub gen: u64,
}

impl Staging {
    pub fn alloc(&mut self, bytes: usize) -> Result<*mut u8> {
        let n = bytes.max(1).next_multiple_of(64);
        if let Some(m) = &self.main {
            if self.off + n <= m.1 {
                let p = unsafe { m.0.add(self.off) };
                self.off += n;
                return Ok(p);
            }
        }
        let p = Pinned::new(n)?;
        let ptr = p.0;
        self.extra_bytes += n;
        self.extra.push(p);
        Ok(ptr)
    }

    pub fn overflowed(&self) -> bool {
        !self.extra.is_empty()
    }

    /// Start a step: no copy from staging may be pending.
    pub fn reset(&mut self) -> Result<()> {
        if !self.extra.is_empty() {
            let cap = self.main.as_ref().map_or(0, |m| m.1) + self.extra_bytes;
            self.extra.clear();
            self.main = None;
            self.main = Some(Pinned::new(cap + cap / 2 + 4096)?);
            self.extra_bytes = 0;
            self.gen += 1;
        }
        self.off = 0;
        Ok(())
    }
}

/// A pending host-to-device copy from staging.
#[derive(Clone, Copy)]
pub struct Upload {
    pub dst: u64,
    pub src: *const u8,
    pub n: usize,
}

impl Upload {
    pub fn enqueue(&self, s: &Stream) -> Result<()> {
        if self.n == 0 {
            return Ok(());
        }
        super::obs::h2d(self.n);
        gpu::check(
            unsafe { sys::cuMemcpyHtoDAsync_v2(self.dst, self.src as *const c_void, self.n, s.0) },
            "cuMemcpyHtoDAsync",
        )?;
        Ok(())
    }
}

/// kernels_graph.cu, launched by name.
pub struct GKernels {
    _m: Module,
    f: HashMap<&'static str, sys::CUfunction>,
}

const NAMES: &[&str] = &[
    "dg_rope",
    "dg_win_idx",
    "dg_ring_write",
    "dg_row_write",
    "dg_spark_idx",
    "dg_index_score",
    "dg_mask_scores",
    "dg_candidates",
    "dg_topk",
    "dg_sparse_attn",
];

/// Threads per block of the selection kernels (`DG_SEL_THREADS`).
pub const SEL: (u32, u32, u32) = (512, 1, 1);

impl GKernels {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(SRC)?;
        let f = NAMES
            .iter()
            .map(|&n| Ok((n, m.func(n)?)))
            .collect::<Result<_>>()?;
        Ok(GKernels { _m: m, f })
    }

    /// # Safety
    /// As [`gpu::launch`].
    pub unsafe fn go(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        s: &Stream,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let f = *self
            .f
            .get(name)
            .with_context(|| format!("no kernel {name}"))?;
        Ok(super::obs::launch(f, grid, block, smem, s, args)?)
    }
}
