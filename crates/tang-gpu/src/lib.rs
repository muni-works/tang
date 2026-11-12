//! tang-gpu — GPU compute and training runtime.
//!
//! GPU runtime for tang-expr with fused kernels via wgpu, plus a complete
//! training pipeline: modules, backward passes, data loading, and optimization.
//!
//! # Key features
//!
//! - **Fused forward-backward kernels**: trace forward + backward into one
//!   expression graph, compile to a single WGSL kernel. One dispatch computes
//!   loss + all gradients with automatic CSE across the boundary.
//!
//! - **JIT kernel cache**: first call traces → diffs → simplifies → compiles.
//!   Subsequent calls just bind new data and dispatch.
//!
//! - **GPU training**: [`GpuTrainModule`] trait with forward/backward passes,
//!   [`GpuSequential`] for layer composition, [`GpuTrainer`] for training
//!   loops with [`GpuAdam`] optimization and [`GpuDataLoader`] batching.
//!
//! - **Same code, three backends**: `f64` for CPU, `ExprId` for symbolic,
//!   `GpuTensor` for GPU — all from the same generic `Scalar` code.
//!
//! # Example: fused elementwise kernel
//!
//! ```ignore
//! use tang_gpu::{GpuDevice, GpuTensor, KernelCache, map_elementwise};
//!
//! let device = GpuDevice::new().await?;
//! let mut cache = KernelCache::new();
//!
//! let a = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0], &[3]);
//! let b = GpuTensor::from_slice(&device, &[4.0, 5.0, 6.0], &[3]);
//!
//! // Fused (a+b)^2 in one kernel via tang-expr codegen
//! let c = map_elementwise(&device, &mut cache, &[&a, &b], |args| {
//!     let sum = args[0] + args[1];
//!     sum * sum
//! });
//!
//! assert_eq!(c.to_vec(&device).await, vec![25.0, 49.0, 81.0]);
//! ```
//!
//! # Example: GPU training
//!
//! ```ignore
//! use tang_gpu::*;
//!
//! let device = GpuDevice::new_sync()?;
//! let mut cache = KernelCache::new();
//!
//! let mut model = GpuSequential::new(vec![
//!     Box::new(GpuLinear::kaiming(&device, 2, 8, 42)),
//!     Box::new(GpuReLULayer::new()),
//!     Box::new(GpuLinear::kaiming(&device, 8, 1, 43)),
//! ]);
//!
//! let mut loader = GpuDataLoader::new(
//!     vec![0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0], // XOR inputs
//!     vec![0.0, 1.0, 1.0, 0.0],                   // XOR targets
//!     2, 1, 4,
//! );
//!
//! let losses = GpuTrainer::new(0.01, 500)
//!     .fit(&device, &mut cache, &mut model, &mut loader);
//! ```

pub mod autotune;
pub mod backward;
pub mod buffer;
/// CPU runtime: `to_c` compiled by the system compiler and `dlopen`ed.
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
/// The driver bindings tang-gpu's CUDA backend is built on, for code that
/// needs the driver directly (CUDA graph capture in tang-bench's planned
/// whole-model runner).
#[cfg(feature = "cuda")]
pub use cudarc;
pub mod device;
pub mod kernel;
pub mod llm;
pub mod matmul;
pub mod module;
pub mod nn;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod program;
pub mod realize;
pub mod reduce;
pub mod safetensors;
pub mod tensor;
pub mod train;

pub use backward::{forward_backward_gpu, fused_forward_backward, FusedKernel};
pub use buffer::GpuBuffer;
pub use device::{GpuDevice, GpuError};
pub use kernel::KernelCache;
pub use llm::{
    interleaved_rope_backward, kv_attention_fused, swiglu_fused_pub, GpuCausalAttention,
    GpuEmbedding, GpuInterleavedRoPE, GpuKVCache, GpuRMSNorm, GpuRoPE, GpuSwiGLU,
    GpuTrainTransformer, GpuTrainTransformerBlock,
};
pub use module::{GpuAdam, GpuLinear, GpuModule, GpuTrainModule};
pub use nn::{
    add_tensors, bias_add, gelu, relu, relu_backward, softmax, GpuAttention, GpuLayerNorm,
    GpuTransformerBlock,
};
pub use program::{BackendKind, BufferPlan, Program, Runtime, TuneReport};
pub use realize::{map_elementwise, map_elementwise_multi};
pub use reduce::{reduce_max, reduce_mean, reduce_sum, reduce_sum_all};
pub use safetensors::{load_safetensors, save_safetensors};
pub use tensor::GpuTensor;
pub use train::{
    gpu_cross_entropy_loss, gpu_mse_loss, GpuDataLoader, GpuReLULayer, GpuSequential, GpuTanhLayer,
    GpuTrainer, GradScaler,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The wgpu device, or `None` (the test skips, saying why) where wgpu
    /// finds no adapter (e.g. a container without a Vulkan driver).
    fn get_device() -> Option<GpuDevice> {
        match GpuDevice::new_sync() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipped: no wgpu adapter ({e})");
                None
            }
        }
    }

    #[test]
    fn device_creation() {
        let Some(_device) = get_device() else { return };
    }

    #[test]
    fn buffer_roundtrip() {
        let Some(device) = get_device() else { return };
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let buf = GpuBuffer::from_slice(&device, &data);
        let result = buf.to_vec_sync(&device);
        assert_eq!(result, data);
    }

    #[test]
    fn tensor_roundtrip() {
        let Some(device) = get_device() else { return };
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::from_slice(&device, &data, &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        let result = t.to_vec_sync(&device);
        assert_eq!(result, data);
    }

    #[test]
    fn elementwise_add() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let a = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0], &[3]);
        let b = GpuTensor::from_slice(&device, &[4.0, 5.0, 6.0], &[3]);
        let c = map_elementwise(&device, &mut cache, &[&a, &b], |args| args[0] + args[1]);
        let result = c.to_vec_sync(&device);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn elementwise_fused() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let a = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0], &[3]);
        let b = GpuTensor::from_slice(&device, &[4.0, 5.0, 6.0], &[3]);
        // Fused (a+b)^2
        let c = map_elementwise(&device, &mut cache, &[&a, &b], |args| {
            let sum = args[0] + args[1];
            sum * sum
        });
        let result = c.to_vec_sync(&device);
        assert_eq!(result, vec![25.0, 49.0, 81.0]);
    }

    #[test]
    fn elementwise_relu() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let a = GpuTensor::from_slice(&device, &[-1.0, 0.0, 1.0, 2.0], &[4]);
        let c = relu(&device, &mut cache, &a);
        let result = c.to_vec_sync(&device);
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn matmul_2x2() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = GpuTensor::from_slice(&device, &[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul::matmul(&device, &mut cache, &a, &b);
        let result = c.to_vec_sync(&device);
        assert_eq!(c.shape(), &[2, 2]);
        // Allow small floating point tolerance
        assert!((result[0] - 19.0).abs() < 0.01);
        assert!((result[1] - 22.0).abs() < 0.01);
        assert!((result[2] - 43.0).abs() < 0.01);
        assert!((result[3] - 50.0).abs() < 0.01);
    }

    #[test]
    fn reduce_sum_axis0() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        // [[1, 2, 3], [4, 5, 6]] -> sum axis 0 -> [5, 7, 9]
        let t = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = reduce_sum(&device, &mut cache, &t, 0);
        let result = s.to_vec_sync(&device);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_sum_axis1() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        // [[1, 2, 3], [4, 5, 6]] -> sum axis 1 -> [6, 15]
        let t = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = reduce_sum(&device, &mut cache, &t, 1);
        let result = s.to_vec_sync(&device);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(result, vec![6.0, 15.0]);
    }

    #[test]
    fn fused_forward_backward_quadratic() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // f(x, y, z) = x^2 + y^2 + z^2
        // df/dx = 2x, df/dy = 2y, df/dz = 2z
        let kernel = fused_forward_backward(3, |vars| {
            vars[0] * vars[0] + vars[1] * vars[1] + vars[2] * vars[2]
        });

        let (loss, grads) = kernel.run(&device, &mut cache, &[1.0, 2.0, 3.0]);
        assert!((loss - 14.0).abs() < 0.01, "loss = {loss}");
        assert!((grads[0] - 2.0).abs() < 0.01, "grad[0] = {}", grads[0]);
        assert!((grads[1] - 4.0).abs() < 0.01, "grad[1] = {}", grads[1]);
        assert!((grads[2] - 6.0).abs() < 0.01, "grad[2] = {}", grads[2]);
    }

    #[test]
    fn fused_forward_backward_product() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // f(x, y) = x * y
        // df/dx = y, df/dy = x
        let kernel = fused_forward_backward(2, |vars| vars[0] * vars[1]);

        let (loss, grads) = kernel.run(&device, &mut cache, &[3.0, 5.0]);
        assert!((loss - 15.0).abs() < 0.01, "loss = {loss}");
        assert!((grads[0] - 5.0).abs() < 0.01, "grad[0] = {}", grads[0]);
        assert!((grads[1] - 3.0).abs() < 0.01, "grad[1] = {}", grads[1]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = softmax(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum = {sum}, expected 1.0"
        );
        // Values should be monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn softmax_2d_rows() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        // 2 rows of 3 elements each
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 10.0, 0.0, 0.0], &[2, 3]);
        let output = softmax(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);
        // Each row should sum to 1
        let row1_sum: f32 = result[0..3].iter().sum();
        let row2_sum: f32 = result[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-5, "row1 sum = {row1_sum}");
        assert!((row2_sum - 1.0).abs() < 1e-5, "row2 sum = {row2_sum}");
        // Row 2: token 0 should dominate
        assert!(
            result[3] > 0.99,
            "row2[0] should dominate, got {}",
            result[3]
        );
    }

    #[test]
    fn layer_norm_output() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let ln = GpuLayerNorm::new(&device, 4, 1e-5);
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = ln.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        // Mean should be approximately 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5, "layernorm mean = {mean}");

        // Std should be approximately 1
        let var: f32 =
            result.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / result.len() as f32;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.1,
            "layernorm std = {std}, expected ~1.0"
        );
    }

    #[test]
    fn linear_forward() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // W = [[1, 0], [0, 1], [1, 1]], b = [0, 0, 0]
        // input = [3, 4]
        // output = W @ input + b = [3, 4, 7]
        let linear = GpuLinear::new(
            &device,
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            2,
            3,
        );
        let input = GpuTensor::from_slice(&device, &[3.0, 4.0], &[2]);
        let output = linear.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 4.0).abs() < 0.01);
        assert!((result[2] - 7.0).abs() < 0.01);
    }

    #[test]
    fn safetensors_roundtrip() {
        let Some(device) = get_device() else { return };
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "weight".to_string(),
            GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        );
        tensors.insert(
            "bias".to_string(),
            GpuTensor::from_slice(&device, &[0.5, 1.5], &[2]),
        );

        let path = std::env::temp_dir().join("tang_gpu_test.safetensors");
        save_safetensors(&tensors, &device, &path).unwrap();

        let loaded = load_safetensors(&device, &path).unwrap();
        assert_eq!(loaded.len(), 2);

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        let w_data = w.to_vec_sync(&device);
        assert_eq!(w_data, vec![1.0, 2.0, 3.0, 4.0]);

        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape(), &[2]);
        let b_data = b.to_vec_sync(&device);
        assert_eq!(b_data, vec![0.5, 1.5]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn adam_step_changes_params() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();
        let mut adam = GpuAdam::new(0.01);

        let mut param = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0], &[3]);
        let grads = vec![GpuTensor::from_slice(&device, &[0.1, 0.2, 0.3], &[3])];

        let before = param.to_vec_sync(&device);
        adam.step(&device, &mut cache, &mut [&mut param], &grads);
        let after = param.to_vec_sync(&device);

        // Parameters should have changed
        assert_ne!(before, after);
        // Parameters should have decreased (positive gradients, minimizing)
        for i in 0..3 {
            assert!(after[i] < before[i], "param[{i}] should decrease");
        }
    }
}
