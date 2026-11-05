//! VRAM-aware buffer pool for CUDA.
//!
//! Caches freed GPU buffers for reuse, keyed by (type, size). Automatically
//! evicts cached buffers when free VRAM drops below a configurable threshold,
//! preventing fragmentation-induced OOM.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;

/// Maximum buffers per size bucket.
const MAX_PER_BUCKET: usize = 32;

/// When free VRAM falls below this fraction of total, evict all cached buffers.
const LOW_VRAM_FRACTION: f64 = 0.08;

pub struct BufferPool {
    f32_free: HashMap<usize, Vec<CudaSlice<f32>>>,
    bf16_free: HashMap<usize, Vec<CudaSlice<u16>>>,
    /// Total bytes currently cached in the pool.
    cached_bytes: usize,
    /// Number of distinct size buckets.
    n_buckets: usize,
    // Stats
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub pressure_clears: u64,
    /// One clear per pressure episode: set after a clear, released when VRAM rises above
    /// the threshold again (see `maybe_evict_under_pressure`).
    pressure_hold: bool,
    /// Pool misses while held (the back-off counter; also the log throttle).
    pressure_holds: u64,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            f32_free: HashMap::new(),
            bf16_free: HashMap::new(),
            cached_bytes: 0,
            n_buckets: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            pressure_clears: 0,
            pressure_hold: false,
            pressure_holds: 0,
        }
    }

    pub fn get_f32(&mut self, len: usize) -> Option<CudaSlice<f32>> {
        if let Some(v) = self.f32_free.get_mut(&len) {
            if let Some(s) = v.pop() {
                self.hits += 1;
                self.cached_bytes -= len * 4;
                if v.is_empty() {
                    self.f32_free.remove(&len);
                    self.n_buckets -= 1;
                }
                return Some(s);
            }
        }
        self.misses += 1;
        None
    }

    pub fn get_bf16(&mut self, len: usize) -> Option<CudaSlice<u16>> {
        if let Some(v) = self.bf16_free.get_mut(&len) {
            if let Some(s) = v.pop() {
                self.hits += 1;
                self.cached_bytes -= len * 2;
                if v.is_empty() {
                    self.bf16_free.remove(&len);
                    self.n_buckets -= 1;
                }
                return Some(s);
            }
        }
        self.misses += 1;
        None
    }

    pub fn put_f32(&mut self, slice: CudaSlice<f32>, len: usize) {
        let bucket = self.f32_free.entry(len).or_insert_with(|| {
            self.n_buckets += 1;
            Vec::new()
        });
        if bucket.len() >= MAX_PER_BUCKET {
            self.evictions += 1;
            // drop slice — cudaFree
        } else {
            self.cached_bytes += len * 4;
            bucket.push(slice);
        }
    }

    pub fn put_bf16(&mut self, slice: CudaSlice<u16>, len: usize) {
        let bucket = self.bf16_free.entry(len).or_insert_with(|| {
            self.n_buckets += 1;
            Vec::new()
        });
        if bucket.len() >= MAX_PER_BUCKET {
            self.evictions += 1;
            // drop slice — cudaFree
        } else {
            self.cached_bytes += len * 2;
            bucket.push(slice);
        }
    }

    /// Drop all cached buffers, returning memory to CUDA.
    pub fn clear(&mut self) {
        self.f32_free.clear();
        self.bf16_free.clear();
        self.cached_bytes = 0;
        self.n_buckets = 0;
    }

    /// Check VRAM pressure and evict if needed.
    /// Call this before allocations that might push memory close to the limit.
    ///
    /// One clear per pressure EPISODE: when a clear doesn't raise free VRAM above the
    /// threshold, the pressure isn't from this pool (the pool is empty after the first
    /// clear) — another process owns the rest of the GPU. Clearing again then re-pools
    /// and re-clears ~600×/s (mew, 2026-11-04: 2900 clears in 14 s at "freed 140MB" while
    /// VRAM stayed at 1028/24123 MB, then the next allocation panicked the serve process).
    /// So the second clear in a row is skipped until free VRAM rises above the threshold
    /// again (someone else freed memory), or a full second's worth of allocations pass —
    /// whichever comes first.
    pub fn maybe_evict_under_pressure(&mut self) {
        if self.cached_bytes < 64 * 1024 * 1024 {
            return; // not worth checking if pool < 64MB
        }
        let (free, total) = cudarc::driver::result::mem_get_info().unwrap_or((usize::MAX, 1));
        if total == 0 || (free as f64) >= (total as f64 * LOW_VRAM_FRACTION) {
            // Pressure is over (or unmeasurable): re-arm the evictor.
            self.pressure_hold = false;
            self.pressure_holds = 0;
            return;
        }
        if self.pressure_hold {
            // The previous clear didn't lift the pressure: not our memory. Back off —
            // re-check every 512 pool misses, log once per ~8k, and never spin.
            self.pressure_holds += 1;
            if self.pressure_holds == 1 || self.pressure_holds % 4096 == 0 {
                eprintln!(
                    "  [pool] under VRAM pressure but the pool is already clear ({}MB held) — \
                     not ours, backing off (checked {}×)",
                    self.cached_bytes / (1024 * 1024),
                    self.pressure_holds,
                );
            }
            return;
        }
        let old_cached = self.cached_bytes;
        self.clear();
        self.pressure_clears += 1;
        self.pressure_hold = true;
        self.pressure_holds = 0;
        if self.pressure_clears <= 5 || self.pressure_clears % 100 == 0 {
            eprintln!(
                "  [pool] pressure eviction: freed {}MB, VRAM {}/{}MB free (total clears={})",
                old_cached / (1024 * 1024),
                free / (1024 * 1024),
                total / (1024 * 1024),
                self.pressure_clears,
            );
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }
}
