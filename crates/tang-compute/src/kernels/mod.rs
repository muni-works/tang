//! Hand-optimized compute kernels for performance-critical operations.
//!
//! Each submodule contains kernel source code as `const &str` for a specific
//! dialect (MSL or CUDA). These are compiled at runtime and cached by the
//! respective backend.

#[cfg(feature = "metal")]
pub mod matmul_msl;

#[cfg(feature = "metal")]
pub mod reduce_msl;

#[cfg(feature = "metal")]
pub mod attention_msl;

#[cfg(feature = "metal")]
pub mod backward_msl;

#[cfg(feature = "metal")]
pub mod adamw_msl;

#[cfg(feature = "metal")]
pub mod llm_msl;

#[cfg(feature = "metal")]
pub mod kvdec_msl;

#[cfg(feature = "cuda")]
pub mod matmul_cuda;

#[cfg(feature = "cuda")]
pub mod reduce_cuda;

#[cfg(feature = "cuda")]
pub mod attention_cuda;

#[cfg(feature = "cuda")]
pub mod backward_cuda;

#[cfg(feature = "cuda")]
pub mod adamw_cuda;

#[cfg(feature = "cuda")]
pub mod util_cuda;

#[cfg(feature = "cuda")]
pub mod noise_cuda;

#[cfg(feature = "cuda")]
pub mod llm_cuda;

#[cfg(feature = "cuda")]
pub mod kvdec_cuda;

#[cfg(feature = "cuda")]
pub mod flash_cuda;

#[cfg(feature = "cuda")]
pub mod native_cuda;

#[cfg(feature = "cuda")]
pub mod moe_cuda;
