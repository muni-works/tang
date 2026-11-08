//! tang-compute — Native GPU compute backends via tang-expr multi-target codegen.
//!
//! Provides a unified `ComputeDevice` trait with backends for:
//! - CPU (always available, with Accelerate BLAS on macOS)
//! - Metal (macOS/iOS, with simdgroup_matrix acceleration)
//! - CUDA (NVIDIA GPUs, with wmma/tensor core acceleration)

pub mod cpu;
pub mod device;
pub mod ext;
pub mod flash;
pub mod flash_native;
pub mod kernels;
pub mod modules;
pub mod moe;
pub mod ops;
pub mod tensor;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod pool;

pub use cpu::CpuDevice;
pub use device::{
    f32_to_bf16, gelu_erf_host, top_k_rows_host, ComputeBuffer, ComputeDevice, PagedRows, Pages,
};
pub use tang_expr::codegen::Dialect;

pub use modules::{
    Embedding, EmbeddingCache, InterleavedRoPE, KVCache, Linear, LinearCache, RMSNorm, RMSNormCache,
};
pub use ops::{
    add_tensors, bias_add, causal_attention_backward, causal_attention_backward_with_output,
    swiglu_backward, swiglu_fused,
};
pub use tensor::ComputeTensor;

#[cfg(feature = "metal")]
pub use metal::MetalDevice;

#[cfg(feature = "cuda")]
pub use cuda::{cuda_memory_info, CudaComputeDevice, RawCuda, RowIn, RowKernel, RowOut};
/// The expression graph crate kernels are traced in (`elementwise`, row kernels).
pub use tang_expr;

/// The cudarc this crate is built on (0.19), so callers sharing a context use the same types.
#[cfg(feature = "cuda")]
pub use cudarc;
#[cfg(feature = "cuda")]
pub use cudarc::driver::CudaGraph;
