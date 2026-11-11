//! The device side on tang-moe's CUDA layer (feature `cuda`): poisoned buffers and
//! named read-backs. Everything assumes the context is bound on the calling thread.

pub use tang_moe::gpu::{DevBuf, Error, Gpu, Result};

use crate::Probe;

/// The poison value. A quiet NaN: every arithmetic op on it stays NaN, so an unwritten
/// element read by a downstream kernel poisons ITS output too, and [`crate::Stat`]
/// counts it (`nonfinite`).
pub const POISON: f32 = f32::NAN;

/// NaN-fill all of `b` (its whole byte length, as f32s). Synchronous (orders against
/// blocking streams; on a NON_BLOCKING stream sync first).
pub fn poison(b: &DevBuf) -> Result<()> {
    let n = b.len / std::mem::size_of::<f32>();
    b.write(0, &vec![POISON; n])
}

/// A fresh device buffer of `n` f32, NaN-filled.
pub fn poisoned(n: usize) -> Result<DevBuf> {
    let b = DevBuf::alloc(n * std::mem::size_of::<f32>())?;
    poison(&b)?;
    Ok(b)
}

/// The first `n` f32 of `b`.
pub fn read(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    b.read::<f32>(0, n)
}

/// The GPU for a test, bound, or `None` after [`crate::skip_or_fail`] (which panics
/// under `TANG_REQUIRE_CUDA=1`).
pub fn gpu_or_skip(what: &str) -> Option<Gpu> {
    let g = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            crate::skip_or_fail(&format!("{what}: no CUDA device ({e})"));
            return None;
        }
    };
    if let Err(e) = g.bind() {
        crate::skip_or_fail(&format!("{what}: CUDA bind failed ({e})"));
        return None;
    }
    Some(g)
}

impl Probe {
    /// NaN-fill each buffer when this probe poisons (no-op otherwise).
    pub fn poison(&self, bufs: &[&DevBuf]) -> Result<()> {
        if !self.poison {
            return Ok(());
        }
        bufs.iter().try_for_each(|b| poison(b))
    }

    /// A fresh output buffer of `n` f32: NaN-filled when poisoning, else zeroed.
    pub fn output(&self, n: usize) -> Result<DevBuf> {
        if self.poison {
            poisoned(n)
        } else {
            DevBuf::zeroed_elems::<f32>(n)
        }
    }

    /// Read `n` f32 of `b` into tap `name`.
    pub fn read(&mut self, name: &str, b: &DevBuf, n: usize) -> Result<()> {
        let v = read(b, n)?;
        self.taps.put(name, v);
        Ok(())
    }

    /// Read `n` f32 of `b` into tap `name` at element `at` (chunked taps; gaps stay NaN).
    pub fn read_at(&mut self, name: &str, at: usize, b: &DevBuf, n: usize) -> Result<()> {
        let v = read(b, n)?;
        self.taps.put_at(name, at, &v);
        Ok(())
    }
}
