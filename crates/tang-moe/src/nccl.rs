//! NCCL collectives on [`crate::gpu`]'s raw streams and buffers (feature `nccl`).
//!
//! One [`Comm`] per rank, each created on its own thread with its device's context bound:
//! `ncclCommInitRank` blocks until every rank of the clique has joined, so the ranks of one
//! process must initialise concurrently. cudarc loads `libnccl.so.2` at run time (its
//! `dynamic-loading` default), so the crate builds where NCCL is absent and only
//! [`unique_id`] / [`Comm::new`] fail there.

use std::ffi::c_void;

use cudarc::nccl::{result, sys};

use crate::gpu::{DevBuf, Error, Result, Stream};

fn nccl<T>(r: std::result::Result<T, result::NcclError>, what: &str) -> Result<()> {
    r.map(|_| ()).map_err(|e| Error(format!("{what}: {e:?}")))
}

/// The clique's id: made once (by rank 0, or the parent thread) and handed to every rank.
#[derive(Clone, Copy)]
pub struct UniqueId(pub sys::ncclUniqueId);

// a 128-byte opaque blob
unsafe impl Send for UniqueId {}
unsafe impl Sync for UniqueId {}

/// A fresh clique id. Errors when libnccl can't be loaded.
pub fn unique_id() -> Result<UniqueId> {
    std::panic::catch_unwind(result::get_uniqueid)
        .map_err(|_| Error("libnccl not loadable".into()))?
        .map(UniqueId)
        .map_err(|e| Error(format!("ncclGetUniqueId: {e:?}")))
}

/// One rank's communicator. Not `Send`: it lives on the thread that made it.
pub struct Comm {
    comm: sys::ncclComm_t,
    pub rank: usize,
    pub world: usize,
}

impl Comm {
    /// Join the clique as `rank` of `world`. The calling thread must have this rank's device
    /// context bound ([`crate::gpu::Gpu::bind`]); blocks until all `world` ranks have called it.
    pub fn new(rank: usize, world: usize, id: UniqueId) -> Result<Self> {
        let mut comm: sys::ncclComm_t = std::ptr::null_mut();
        nccl(
            unsafe { result::comm_init_rank(&mut comm, world as i32, id.0, rank as i32) },
            "ncclCommInitRank",
        )?;
        Ok(Comm { comm, rank, world })
    }

    fn stream(s: &Stream) -> sys::cudaStream_t {
        s.0 as sys::cudaStream_t
    }

    /// In-place sum of `n` f32s at `buf` across all ranks, enqueued on `s`. Every rank must
    /// call it with the same `n`, in the same order relative to its other collectives.
    pub fn all_reduce_sum_f32(&self, buf: &DevBuf, n: usize, s: &Stream) -> Result<()> {
        assert!(n * 4 <= buf.len);
        nccl(
            unsafe {
                result::all_reduce(
                    buf.ptr as *const c_void,
                    buf.ptr as *mut c_void,
                    n,
                    sys::ncclDataType_t::ncclFloat32,
                    sys::ncclRedOp_t::ncclSum,
                    self.comm,
                    Self::stream(s),
                )
            },
            "ncclAllReduce",
        )
    }

    /// In-place broadcast of `bytes` bytes at `buf` from `root` to every rank, on `s`.
    pub fn broadcast(&self, buf: &DevBuf, bytes: usize, root: usize, s: &Stream) -> Result<()> {
        assert!(bytes <= buf.len);
        nccl(
            unsafe {
                result::broadcast(
                    buf.ptr as *const c_void,
                    buf.ptr as *mut c_void,
                    bytes,
                    sys::ncclDataType_t::ncclUint8,
                    root as i32,
                    self.comm,
                    Self::stream(s),
                )
            },
            "ncclBroadcast",
        )
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        unsafe {
            let _ = result::comm_destroy(self.comm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Gpu;

    /// Needs ≥ 2 GPUs and libnccl:
    /// `cargo test --release -p tang-moe --features nccl --lib nccl -- --ignored`.
    /// Each rank fills `n` floats with `rank + 1` and broadcasts a marker from rank 0; after
    /// the all-reduce every element is `world * (world + 1) / 2`.
    #[test]
    #[ignore]
    fn nccl_all_reduce_and_broadcast() {
        let world = std::env::var("NCCL_TEST_WORLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2usize);
        let id = unique_id().expect("libnccl");
        let n = 1 << 20;
        let hs: Vec<_> = (0..world)
            .map(|rank| {
                std::thread::spawn(move || {
                    let g = Gpu::at(rank).unwrap();
                    g.bind().unwrap();
                    let s = Stream::blocking().unwrap();
                    let comm = Comm::new(rank, world, id).unwrap();
                    let buf = DevBuf::from_slice(&vec![(rank + 1) as f32; n]).unwrap();
                    comm.all_reduce_sum_f32(&buf, n, &s).unwrap();
                    let mark = DevBuf::from_slice(&[rank as u32 + 7; 4]).unwrap();
                    comm.broadcast(&mark, 16, 0, &s).unwrap();
                    s.sync().unwrap();
                    let want = (world * (world + 1) / 2) as f32;
                    assert!(buf.read::<f32>(0, n).unwrap().iter().all(|&v| v == want));
                    assert_eq!(mark.read::<u32>(0, 4).unwrap(), vec![7; 4]);
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }

    /// All-reduce latency at the decode MoE's sizes (t × 5120 f32, t = 1..8), `NCCL_TEST_WORLD`
    /// ranks, back-to-back on each rank's stream: µs per call (rank 0 prints). Set NCCL_ALGO /
    /// NCCL_PROTO / NCCL_NCHANNELS to compare. `--ignored --nocapture nccl_small_bench`.
    #[test]
    #[ignore]
    fn nccl_small_bench() {
        let world = std::env::var("NCCL_TEST_WORLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize);
        let id = unique_id().expect("libnccl");
        let hs: Vec<_> = (0..world)
            .map(|rank| {
                std::thread::spawn(move || {
                    let g = Gpu::at(rank).unwrap();
                    g.bind().unwrap();
                    let s = Stream::new().unwrap();
                    let comm = Comm::new(rank, world, id).unwrap();
                    let buf = DevBuf::zeroed(8 * 5120 * 4).unwrap();
                    for t in [1usize, 7, 8] {
                        let n = t * 5120;
                        for _ in 0..50 {
                            comm.all_reduce_sum_f32(&buf, n, &s).unwrap();
                        }
                        s.sync().unwrap();
                        let iters = 500;
                        let t0 = std::time::Instant::now();
                        for _ in 0..iters {
                            comm.all_reduce_sum_f32(&buf, n, &s).unwrap();
                        }
                        s.sync().unwrap();
                        let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
                        if rank == 0 {
                            println!(
                                "nccl all-reduce {world} ranks, {} KB: {us:.1} us",
                                n * 4 / 1024
                            );
                        }
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }
}
