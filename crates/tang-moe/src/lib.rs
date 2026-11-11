//! tang-moe: the substrate for running MoE models whose experts don't fit in VRAM.
//!
//! - [`HostArena`]: pinned, device-mapped host memory holding expert blobs. Kernels read it
//!   directly over PCIe through its device addresses.
//! - [`CachePolicy`]: which experts live in VRAM slots, with decayed-LFU adaptation. CPU-only,
//!   tested anywhere.
//! - [`PointerTable`]: per (layer, expert) the device address of its blob, either a VRAM slot
//!   or the mapped host copy, so an expert kernel does one indirection and doesn't care where
//!   the blob lives.
//! - `ExpertCache` (feature `cuda`): the VRAM slot arena, the device-side residency and pointer
//!   tables, and swaps on a side stream. `ResidentCache`: the same in resident mode (host arena
//!   holds only non-VRAM experts; swaps are exchanges, see [`resident`]).
//! - The CPU miss path: [`contract`] (the data contract shared with the GPU kernels and the
//!   scalar spec), [`q2cpu`] (AVX-VNNI / AVX2 Q2_0 rows), [`pool`] (pinned spinning workers),
//!   [`miss`] (plan builder and two-phase executor), and `doorbell` (feature `cuda`: the
//!   per-layer GPU↔CPU protocol on mapped memory).
//!
//! See `DESIGN.md` for the measurements behind the design and `BENCH.md` for how to reproduce
//! them with `tang-moe-bench`.

pub mod arena;
pub mod contract;
pub mod miss;
pub mod policy;
pub mod pool;
pub mod q2cpu;
pub mod resident;

#[cfg(feature = "cuda")]
pub mod cache;
#[cfg(feature = "cuda")]
pub mod doorbell;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "nccl")]
pub mod nccl;

pub use arena::{ArenaOptions, Backing, HostArena};
#[cfg(feature = "cuda")]
pub use cache::{ExpertCache, SizeReport};
pub use policy::{AdaptParams, CachePolicy, Geometry, PointerTable, Stats, Swap};
