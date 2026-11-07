//! The tang stack behind one dependency.
//!
//! ```toml
//! tang-all = { version = "0.1", features = ["optim", "compile"] }
//! ```
//!
//! `tang`'s core math (`Scalar`, `Dual`, `Vec3`, …) is re-exported at the
//! root; each subcrate is a module behind the feature of the same name.
//! This is a facade only: every subcrate is still its own crate, so mixing
//! `tang-all` with direct subcrate dependencies resolves to the same types.
//! `la` is on by default; `gpu` and `train` stay off unless asked for.

#![no_std]

pub use tang::*;

#[cfg(feature = "ad")]
pub use tang_ad as ad;
#[cfg(feature = "compile")]
pub use tang_compile as compile;
#[cfg(feature = "expr")]
pub use tang_expr as expr;
#[cfg(feature = "gpu")]
pub use tang_gpu as gpu;
#[cfg(feature = "la")]
pub use tang_la as la;
#[cfg(feature = "optim")]
pub use tang_optim as optim;
#[cfg(feature = "sparse")]
pub use tang_sparse as sparse;
#[cfg(feature = "tensor")]
pub use tang_tensor as tensor;
#[cfg(feature = "train")]
pub use tang_train as train;
