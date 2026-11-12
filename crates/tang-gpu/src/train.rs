//! GPU training infrastructure: layers, sequential model, data loader, trainer.

use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::module::{GpuAdam, GpuTrainModule};
use crate::nn::{relu, relu_backward};
use crate::realize::map_elementwise;
use crate::reduce::reduce_sum_all;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// GpuTanhLayer
// ---------------------------------------------------------------------------

/// Tanh activation layer with cached output for backward.
pub struct GpuTanhLayer {
    cached_output: Option<GpuTensor>,
}

impl Default for GpuTanhLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuTanhLayer {
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }
}

impl GpuTrainModule for GpuTanhLayer {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        let output = map_elementwise(device, cache, &[input], |args| {
            use tang::Scalar;
            args[0].tanh()
        });
        self.cached_output = Some(output.clone_gpu_batched(device, cache));
        output
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        // d/dx tanh(x) = 1 - tanh(x)^2
        let output = self
            .cached_output
            .as_ref()
            .expect("must call forward_train before backward");
        map_elementwise(device, cache, &[grad_output, output], |args| {
            use tang::Scalar;
            use tang_expr::ExprId;
            let grad = args[0];
            let out = args[1];
            grad * (ExprId::from_f64(1.0) - out * out)
        })
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.cached_output = None;
    }
}

// ---------------------------------------------------------------------------
// GpuReLULayer
// ---------------------------------------------------------------------------

/// ReLU activation layer with cached input for backward.
pub struct GpuReLULayer {
    cached_input: Option<GpuTensor>,
}

impl Default for GpuReLULayer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuReLULayer {
    pub fn new() -> Self {
        Self { cached_input: None }
    }
}

impl GpuTrainModule for GpuReLULayer {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        self.cached_input = Some(input.clone_gpu_batched(device, cache));
        relu(device, cache, input)
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward_train before backward");
        relu_backward(device, cache, input, grad_output)
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.cached_input = None;
    }
}

// ---------------------------------------------------------------------------
// GpuSequential
// ---------------------------------------------------------------------------

/// Sequential container: chains layers left-to-right for forward,
/// right-to-left for backward.
pub struct GpuSequential {
    layers: Vec<Box<dyn GpuTrainModule>>,
}

impl GpuSequential {
    pub fn new(layers: Vec<Box<dyn GpuTrainModule>>) -> Self {
        Self { layers }
    }
}

impl GpuTrainModule for GpuSequential {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        let mut x = input.clone_gpu_batched(device, cache);
        for layer in &mut self.layers {
            x = layer.forward_train(device, cache, &x);
        }
        x
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let mut grad = grad_output.clone_gpu_batched(device, cache);
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(device, cache, &grad);
        }
        grad
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        self.layers.iter().flat_map(|l| l.gradients()).collect()
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

// ---------------------------------------------------------------------------
// gpu_mse_loss
// ---------------------------------------------------------------------------

/// Compute MSE loss and its gradient on GPU.
///
/// Returns `(loss_tensor, grad_tensor)` where loss is a [1] scalar tensor
/// and `grad = 2/n * (pred - target)`. No CPU readback — loss stays on GPU.
pub fn gpu_mse_loss(
    device: &GpuDevice,
    cache: &mut KernelCache,
    pred: &GpuTensor,
    target: &GpuTensor,
) -> (GpuTensor, GpuTensor) {
    let n = pred.numel();
    assert_eq!(n, target.numel());

    // diff = pred - target
    let diff = map_elementwise(device, cache, &[pred, target], |args| args[0] - args[1]);

    // squared = diff^2
    let sq = map_elementwise(device, cache, &[&diff], |args| args[0] * args[0]);

    // loss = sum(sq) / n — entirely on GPU, no readback
    let sum = reduce_sum_all(device, cache, &sq);
    let loss = sum.scale(device, cache, 1.0 / n as f32);

    // grad = 2/n * diff
    let scale = 2.0 / n as f32;
    let grad = diff.scale(device, cache, scale);

    (loss, grad)
}

// ---------------------------------------------------------------------------
// gpu_cross_entropy_loss
// ---------------------------------------------------------------------------

/// Compute cross-entropy loss and its gradient on GPU.
///
/// `pred` has shape `[batch, num_classes]` (logits) or `[num_classes]`.
/// `target` has shape `[batch, 1]` (class indices as f32) or `[1]`.
///
/// Returns `(loss_tensor, grad_tensor)` where loss is a `[1]` scalar and
/// grad has the same shape as `pred`.
pub fn gpu_cross_entropy_loss(
    device: &GpuDevice,
    cache: &mut KernelCache,
    pred: &GpuTensor,
    target: &GpuTensor,
) -> (GpuTensor, GpuTensor) {
    cache.flush(device);
    let logits_data = pred.buffer.to_vec_sync(device);
    let target_data = target.buffer.to_vec_sync(device);

    let (batch, num_classes) = if pred.ndim() == 2 {
        (pred.shape()[0], pred.shape()[1])
    } else {
        (1, pred.shape()[0])
    };

    let mut total_loss = 0.0f32;
    let mut grad = vec![0.0f32; batch * num_classes];

    for b in 0..batch {
        let offset = b * num_classes;
        let row = &logits_data[offset..offset + num_classes];
        let target_idx = target_data[b] as usize;

        // Numerically stable softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

        // Loss = -log(prob[target])
        total_loss += -probs[target_idx].max(1e-12).ln();

        // Grad = softmax - one_hot(target)
        grad[offset..offset + num_classes].copy_from_slice(&probs[..num_classes]);
        grad[offset + target_idx] -= 1.0;
    }

    // Average over batch
    let avg_loss = total_loss / batch as f32;
    let inv_batch = 1.0 / batch as f32;
    for g in &mut grad {
        *g *= inv_batch;
    }

    let loss = GpuTensor::from_slice(device, &[avg_loss], &[1]);
    let grad_tensor = GpuTensor::from_slice(device, &grad, pred.shape());
    (loss, grad_tensor)
}

// ---------------------------------------------------------------------------
// GpuDataLoader
// ---------------------------------------------------------------------------

/// Batched data loader that uploads CPU data to GPU per batch.
pub struct GpuDataLoader {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    input_dim: usize,
    target_dim: usize,
    n_samples: usize,
    batch_size: usize,
    position: usize,
}

impl GpuDataLoader {
    /// Create a data loader from flat f32 arrays.
    ///
    /// `inputs` has shape [n_samples, input_dim] flattened row-major.
    /// `targets` has shape [n_samples, target_dim] flattened row-major.
    pub fn new(
        inputs: Vec<f32>,
        targets: Vec<f32>,
        input_dim: usize,
        target_dim: usize,
        batch_size: usize,
    ) -> Self {
        let n_samples = inputs.len() / input_dim;
        assert_eq!(inputs.len(), n_samples * input_dim);
        assert_eq!(targets.len(), n_samples * target_dim);
        Self {
            inputs,
            targets,
            input_dim,
            target_dim,
            n_samples,
            batch_size,
            position: 0,
        }
    }

    /// Reset to the beginning of the dataset.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get the next batch as GPU tensors, or None if exhausted.
    pub fn next_batch(&mut self, device: &GpuDevice) -> Option<(GpuTensor, GpuTensor)> {
        if self.position >= self.n_samples {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.n_samples);
        let batch = end - self.position;

        let in_start = self.position * self.input_dim;
        let in_end = end * self.input_dim;
        let tgt_start = self.position * self.target_dim;
        let tgt_end = end * self.target_dim;

        let in_shape = if batch == 1 {
            vec![self.input_dim]
        } else {
            vec![batch, self.input_dim]
        };
        let tgt_shape = if batch == 1 {
            vec![self.target_dim]
        } else {
            vec![batch, self.target_dim]
        };

        let input = GpuTensor::from_slice(device, &self.inputs[in_start..in_end], &in_shape);
        let target = GpuTensor::from_slice(device, &self.targets[tgt_start..tgt_end], &tgt_shape);

        self.position = end;
        Some((input, target))
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.n_samples
    }

    /// Whether the loader is empty.
    pub fn is_empty(&self) -> bool {
        self.n_samples == 0
    }

    /// Number of batches per epoch.
    pub fn n_batches(&self) -> usize {
        self.n_samples.div_ceil(self.batch_size)
    }
}

// ---------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------

/// Training loop that mirrors tang-train's Trainer.
pub struct GpuTrainer {
    optimizer: GpuAdam,
    loss_fn: fn(&GpuDevice, &mut KernelCache, &GpuTensor, &GpuTensor) -> (GpuTensor, GpuTensor),
    num_epochs: usize,
}

impl GpuTrainer {
    pub fn new(lr: f32, num_epochs: usize) -> Self {
        Self {
            optimizer: GpuAdam::new(lr),
            loss_fn: gpu_mse_loss,
            num_epochs,
        }
    }

    /// Set a custom loss function.
    pub fn with_loss_fn(
        mut self,
        f: fn(&GpuDevice, &mut KernelCache, &GpuTensor, &GpuTensor) -> (GpuTensor, GpuTensor),
    ) -> Self {
        self.loss_fn = f;
        self
    }

    /// Train the model and return per-epoch average losses.
    pub fn fit(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        model: &mut dyn GpuTrainModule,
        loader: &mut GpuDataLoader,
    ) -> Vec<f32> {
        let mut epoch_losses = Vec::with_capacity(self.num_epochs);

        // Enable command batching for the training loop
        cache.begin_batch();

        for _ in 0..self.num_epochs {
            loader.reset();
            let mut batch_losses: Vec<GpuTensor> = Vec::new();

            while let Some((input, target)) = loader.next_batch(device) {
                // Zero gradients
                model.zero_grad();

                // Forward
                let pred = model.forward_train(device, cache, &input);

                // Loss — stays on GPU as a [1] scalar tensor
                let (loss_tensor, grad) = (self.loss_fn)(device, cache, &pred, &target);
                batch_losses.push(loss_tensor);

                // Backward
                model.backward(device, cache, &grad);

                // Collect grads (GPU-side copy, no readback)
                let grads: Vec<GpuTensor> = model
                    .gradients()
                    .into_iter()
                    .map(|g| {
                        let g = g.expect("gradient missing after backward");
                        g.clone_gpu_batched(device, cache)
                    })
                    .collect();

                // Update params in-place
                let mut params = model.parameters_mut();
                self.optimizer.step(device, cache, &mut params, &grads);
            }

            // Flush once per epoch, then read back all loss scalars
            cache.flush(device);
            let n_batches = batch_losses.len().max(1);
            let total_loss: f32 = batch_losses.iter().map(|t| t.to_vec_sync(device)[0]).sum();
            epoch_losses.push(total_loss / n_batches as f32);

            // Re-enable batching for next epoch
            cache.begin_batch();
        }

        epoch_losses
    }
}

// ---------------------------------------------------------------------------
// GradScaler — mixed-precision training support
// ---------------------------------------------------------------------------

/// Gradient scaler for mixed-precision training (AMP-style).
///
/// Scales the loss up before backward pass to prevent gradient underflow in
/// low-precision computations, then scales gradients back down before the
/// optimizer step. Dynamically adjusts the scale factor based on whether
/// gradients contain inf/NaN.
///
/// # Usage
/// ```ignore
/// let mut scaler = GradScaler::new();
/// let scaled_loss = scaler.scale_loss(device, cache, &loss);
/// // ... backward with scaled_loss ...
/// if scaler.unscale_and_step(device, cache, &mut optimizer, &mut params, &grads) {
///     // step was applied
/// }
/// scaler.update();
/// ```
pub struct GradScaler {
    /// Current scale factor (starts high, adjusts dynamically)
    scale: f32,
    /// Factor to grow scale by when no overflow detected
    growth_factor: f32,
    /// Factor to shrink scale by when overflow detected
    backoff_factor: f32,
    /// Number of consecutive successful steps before growing scale
    growth_interval: u32,
    /// Counter of consecutive successful steps
    growth_step: u32,
    /// Whether the last step had an overflow
    found_inf: bool,
}

impl GradScaler {
    /// Create a new gradient scaler with default settings.
    ///
    /// Default: initial scale = 65536, growth_factor = 2, backoff = 0.5,
    /// growth_interval = 2000 steps.
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_step: 0,
            found_inf: false,
        }
    }

    /// Create with custom initial scale.
    pub fn with_scale(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            ..Self::new()
        }
    }

    /// Current scale factor.
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Scale the loss tensor for backward pass.
    pub fn scale_loss(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        loss: &GpuTensor,
    ) -> GpuTensor {
        loss.scale(device, cache, self.scale)
    }

    /// Unscale gradients by 1/scale, check for inf/NaN, and optionally step the optimizer.
    ///
    /// Returns true if the optimizer step was applied (no overflow detected).
    /// Returns false if overflow was detected (optimizer step skipped).
    pub fn unscale_and_step(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        optimizer: &mut GpuAdam,
        params: &mut [&mut GpuTensor],
        grads: &[GpuTensor],
    ) -> bool {
        let inv_scale = 1.0 / self.scale;

        // Unscale gradients and check for inf/NaN
        let mut unscaled_grads = Vec::with_capacity(grads.len());
        let mut has_inf = false;

        for grad in grads {
            let unscaled = grad.scale(device, cache, inv_scale);
            cache.flush(device);
            let data = unscaled.to_vec_sync(device);
            if data.iter().any(|&v| v.is_infinite() || v.is_nan()) {
                has_inf = true;
                break;
            }
            unscaled_grads.push(unscaled);
        }

        self.found_inf = has_inf;

        if has_inf {
            // Skip optimizer step
            return false;
        }

        // Apply optimizer step with unscaled gradients
        optimizer.step(device, cache, params, &unscaled_grads);
        true
    }

    /// Update the scale factor after a training step.
    ///
    /// Call this after `unscale_and_step` each iteration.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale *= self.backoff_factor;
            self.growth_step = 0;
        } else {
            self.growth_step += 1;
            if self.growth_step >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_step = 0;
            }
        }
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{GpuLinear, GpuModule};

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
    fn linear_forward_backward_gradient_check() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // Create a simple linear layer: 2 -> 3
        let mut linear = GpuLinear::new(
            &device,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.1, 0.2, 0.3],
            2,
            3,
        );

        let input = GpuTensor::from_slice(&device, &[1.0, 0.5], &[2]);

        // Forward
        let output = linear.forward_train(&device, &mut cache, &input);
        let out_data = output.to_vec_sync(&device);
        // y = W @ x + b = [1*1+2*0.5+0.1, 3*1+4*0.5+0.2, 5*1+6*0.5+0.3] = [2.1, 5.2, 8.3]
        assert!((out_data[0] - 2.1).abs() < 0.01, "out[0]={}", out_data[0]);
        assert!((out_data[1] - 5.2).abs() < 0.01, "out[1]={}", out_data[1]);
        assert!((out_data[2] - 8.3).abs() < 0.01, "out[2]={}", out_data[2]);

        // Backward with grad_output = [1, 1, 1]
        let grad_out = GpuTensor::from_slice(&device, &[1.0, 1.0, 1.0], &[3]);
        let grad_input = linear.backward(&device, &mut cache, &grad_out);

        // Numerical gradient check for input
        let eps = 1e-3;
        let in_data = vec![1.0f32, 0.5];
        let gi_data = grad_input.to_vec_sync(&device);

        for dim in 0..2 {
            let mut plus = in_data.clone();
            plus[dim] += eps;
            let mut minus = in_data.clone();
            minus[dim] -= eps;

            let out_p = linear.forward(
                &device,
                &mut cache,
                &GpuTensor::from_slice(&device, &plus, &[2]),
            );
            let out_m = linear.forward(
                &device,
                &mut cache,
                &GpuTensor::from_slice(&device, &minus, &[2]),
            );

            let p_data = out_p.to_vec_sync(&device);
            let m_data = out_m.to_vec_sync(&device);
            // sum of outputs (since grad_out = [1,1,1])
            let numerical_grad: f32 = p_data
                .iter()
                .zip(m_data.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            assert!(
                (gi_data[dim] - numerical_grad).abs() < 0.05,
                "input grad[{dim}]: analytical={}, numerical={}",
                gi_data[dim],
                numerical_grad
            );
        }
    }

    #[test]
    fn sequential_forward_backward_shapes() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let mut model = GpuSequential::new(vec![
            Box::new(GpuLinear::kaiming(&device, 2, 4, 42)),
            Box::new(GpuReLULayer::new()),
            Box::new(GpuLinear::kaiming(&device, 4, 1, 43)),
        ]);

        // Batched forward
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let output = model.forward_train(&device, &mut cache, &input);
        assert_eq!(output.shape(), &[2, 1]);

        // Backward
        let grad = GpuTensor::from_slice(&device, &[1.0, 1.0], &[2, 1]);
        let grad_input = model.backward(&device, &mut cache, &grad);
        assert_eq!(grad_input.shape(), &[2, 2]);
    }

    #[test]
    fn data_loader_batching() {
        let Some(device) = get_device() else { return };

        // 5 samples, input_dim=2, target_dim=1, batch_size=2
        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut loader = GpuDataLoader::new(inputs, targets, 2, 1, 2);
        assert_eq!(loader.len(), 5);
        assert_eq!(loader.n_batches(), 3);

        // Batch 1: 2 samples
        let (inp, tgt) = loader.next_batch(&device).unwrap();
        assert_eq!(inp.shape(), &[2, 2]);
        assert_eq!(tgt.shape(), &[2, 1]);

        // Batch 2: 2 samples
        let (inp, tgt) = loader.next_batch(&device).unwrap();
        assert_eq!(inp.shape(), &[2, 2]);
        assert_eq!(tgt.shape(), &[2, 1]);

        // Batch 3: 1 remaining sample
        let (inp, tgt) = loader.next_batch(&device).unwrap();
        assert_eq!(inp.shape(), &[2]);
        assert_eq!(tgt.shape(), &[1]);

        // Exhausted
        assert!(loader.next_batch(&device).is_none());

        // Reset and iterate again
        loader.reset();
        assert!(loader.next_batch(&device).is_some());
    }

    #[test]
    fn xor_training_convergence() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // XOR dataset
        let inputs = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let targets = vec![0.0, 1.0, 1.0, 0.0];

        let mut loader = GpuDataLoader::new(inputs, targets, 2, 1, 4);

        let mut model = GpuSequential::new(vec![
            Box::new(GpuLinear::kaiming(&device, 2, 8, 123)),
            Box::new(GpuReLULayer::new()),
            Box::new(GpuLinear::kaiming(&device, 8, 1, 456)),
        ]);

        let mut trainer = GpuTrainer::new(0.01, 500);
        let losses = trainer.fit(&device, &mut cache, &mut model, &mut loader);

        // Loss should decrease significantly
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first * 0.1,
            "XOR training did not converge: first_loss={first}, last_loss={last}"
        );
    }
}
