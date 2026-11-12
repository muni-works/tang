//! GPU-accelerated components for LLM inference.
//!
//! Provides the core building blocks for running transformer-based language
//! models on GPU via wgpu compute shaders:
//!
//! - [`GpuEmbedding`]: token embedding lookup
//! - [`GpuRMSNorm`]: RMS normalization
//! - [`GpuSwiGLU`]: SwiGLU feed-forward network
//! - [`GpuRoPE`]: rotary position embeddings
//! - [`GpuCausalAttention`]: grouped-query attention with causal mask
//! - [`GpuKVCache`]: key-value cache for autoregressive inference

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;
use crate::kernel::{BindingSpec, KernelCache};
use crate::matmul::matmul;
use crate::module::GpuTrainModule;
use crate::nn::add_tensors;
use crate::realize::map_elementwise;
use crate::tensor::GpuTensor;

use std::hash::{Hash, Hasher};

fn hash_wgsl(wgsl: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    wgsl.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// GpuEmbedding
// ---------------------------------------------------------------------------

/// Token embedding lookup on GPU.
///
/// Stores an embedding table of shape `[vocab_size, dim]` and looks up
/// rows by token ID. Input: buffer of u32 token IDs. Output: `[seq_len, dim]`.
pub struct GpuEmbedding {
    /// Embedding weight table [vocab_size, dim].
    pub weight: GpuTensor,
    pub vocab_size: usize,
    pub dim: usize,
    // Training state
    cached_indices: Option<Vec<u32>>,
    cached_seq_len: Option<usize>,
    weight_grad: Option<GpuTensor>,
}

impl GpuEmbedding {
    /// Create an embedding layer from a weight table.
    pub fn new(device: &GpuDevice, weight: &[f32], vocab_size: usize, dim: usize) -> Self {
        assert_eq!(weight.len(), vocab_size * dim);
        Self {
            weight: GpuTensor::from_slice(device, weight, &[vocab_size, dim]),
            vocab_size,
            dim,
            cached_indices: None,
            cached_seq_len: None,
            weight_grad: None,
        }
    }

    /// Create an embedding layer with zero weights.
    pub fn zeros(device: &GpuDevice, vocab_size: usize, dim: usize) -> Self {
        Self::new(device, &vec![0.0f32; vocab_size * dim], vocab_size, dim)
    }

    /// Look up embeddings for a sequence of token IDs.
    ///
    /// `token_ids`: u32 buffer of length `seq_len`.
    /// Returns: `GpuTensor` of shape `[seq_len, dim]`.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        token_ids: &GpuBuffer,
        seq_len: usize,
    ) -> GpuTensor {
        let out = GpuTensor::uninit(device, &[seq_len, self.dim]);

        let wgsl = r#"// Embedding lookup: output[i, :] = weight[token_ids[i], :]

struct Params {
    seq_len: u32,
    dim: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.dim;
    if (idx >= total) { return; }
    let row = idx / params.dim;
    let col = idx % params.dim;
    let tok = token_ids[row];
    output[idx] = weight[tok * params.dim + col];
}
"#;

        let params: [u32; 4] = [seq_len as u32, self.dim as u32, 0, 0];

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },  // token_ids
            BindingSpec::Storage { read_only: true },  // weight
            BindingSpec::Storage { read_only: false }, // output
            BindingSpec::Uniform,                      // params
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embedding params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embedding dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * self.dim) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuRMSNorm
// ---------------------------------------------------------------------------

/// RMS normalization: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Operates on the last dimension. One workgroup per normalization group
/// with shared-memory reduction for the RMS statistic.
pub struct GpuRMSNorm {
    /// Scale parameter [dim].
    pub weight: GpuTensor,
    pub eps: f32,
    pub dim: usize,
    // Training state
    cached_input: Option<Vec<f32>>,
    cached_rms: Option<Vec<f32>>,
    weight_grad: Option<GpuTensor>,
}

impl GpuRMSNorm {
    /// Create RMS norm with ones for weight.
    pub fn new(device: &GpuDevice, dim: usize, eps: f32) -> Self {
        let ones = vec![1.0f32; dim];
        Self {
            weight: GpuTensor::from_slice(device, &ones, &[dim]),
            eps,
            dim,
            cached_input: None,
            cached_rms: None,
            weight_grad: None,
        }
    }

    /// Apply RMS normalization on GPU.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        let n_groups = input.numel() / self.dim;
        let out = GpuTensor::uninit(device, input.shape());

        let wg_size = (self.dim as u32).next_power_of_two().min(256);

        let wgsl = format!(
            r#"// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// One workgroup per normalization group

struct Params {{
    n_groups: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_buf: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let group = wg_id.x;
    if (group >= params.n_groups) {{ return; }}
    let tid = lid.x;
    let base = group * params.dim;

    // Compute mean(x^2)
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.dim; i = i + WG) {{
        let val = input[base + i];
        local_sum = local_sum + val * val;
    }}
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + stride];
        }}
        workgroupBarrier();
    }}
    let rms = sqrt(wg_buf[0] / f32(params.dim) + params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();

    // Normalize and scale
    for (var i = tid; i < params.dim; i = i + WG) {{
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }}
}}
"#,
            wg_size = wg_size,
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RmsParams {
            n_groups: u32,
            dim: u32,
            eps: f32,
            _pad: u32,
        }
        let uniform = RmsParams {
            n_groups: n_groups as u32,
            dim: self.dim as u32,
            eps: self.eps,
            _pad: 0,
        };

        let hash = hash_wgsl(&wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },  // input
            BindingSpec::Storage { read_only: true },  // weight
            BindingSpec::Storage { read_only: false }, // output
            BindingSpec::Uniform,                      // params
        ];
        let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsnorm params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rmsnorm dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_groups as u32, 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuSwiGLU
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
///
/// gate_proj: [dim, hidden_dim], up_proj: [dim, hidden_dim], down_proj: [hidden_dim, dim].
pub struct GpuSwiGLU {
    /// Gate projection [hidden_dim, dim] (stored transposed for matmul).
    pub gate_proj: GpuTensor,
    /// Up projection [hidden_dim, dim].
    pub up_proj: GpuTensor,
    /// Down projection [dim, hidden_dim].
    pub down_proj: GpuTensor,
    pub dim: usize,
    pub hidden_dim: usize,
    // Training state
    cached_input: Option<GpuTensor>,
    cached_gate_raw: Option<GpuTensor>,
    cached_up_raw: Option<GpuTensor>,
    cached_gate_silu: Option<GpuTensor>,
    cached_activated: Option<GpuTensor>,
    gate_proj_grad: Option<GpuTensor>,
    up_proj_grad: Option<GpuTensor>,
    down_proj_grad: Option<GpuTensor>,
}

impl GpuSwiGLU {
    /// Create a SwiGLU layer with given weights.
    ///
    /// Weight shapes: gate/up = `[hidden_dim, dim]`, down = `[dim, hidden_dim]`.
    pub fn new(
        device: &GpuDevice,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        dim: usize,
        hidden_dim: usize,
    ) -> Self {
        assert_eq!(gate.len(), hidden_dim * dim);
        assert_eq!(up.len(), hidden_dim * dim);
        assert_eq!(down.len(), dim * hidden_dim);
        Self {
            gate_proj: GpuTensor::from_slice(device, gate, &[hidden_dim, dim]),
            up_proj: GpuTensor::from_slice(device, up, &[hidden_dim, dim]),
            down_proj: GpuTensor::from_slice(device, down, &[dim, hidden_dim]),
            dim,
            hidden_dim,
            cached_input: None,
            cached_gate_raw: None,
            cached_up_raw: None,
            cached_gate_silu: None,
            cached_activated: None,
            gate_proj_grad: None,
            up_proj_grad: None,
            down_proj_grad: None,
        }
    }

    /// Create a SwiGLU layer with zero weights.
    pub fn zeros(device: &GpuDevice, dim: usize, hidden_dim: usize) -> Self {
        Self::new(
            device,
            &vec![0.0f32; hidden_dim * dim],
            &vec![0.0f32; hidden_dim * dim],
            &vec![0.0f32; dim * hidden_dim],
            dim,
            hidden_dim,
        )
    }

    /// Forward pass: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    ///
    /// Input: `[seq_len, dim]` or `[dim]`. Output: same shape as input.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        use crate::matmul::matmul;

        // Ensure 2D for matmul
        let was_1d = input.ndim() == 1;
        let input_2d = if was_1d {
            GpuTensor {
                buffer: input.buffer.clone_gpu_batched(device, cache),
                shape: vec![1, input.numel()],
            }
        } else {
            GpuTensor {
                buffer: input.buffer.clone_gpu_batched(device, cache),
                shape: input.shape().to_vec(),
            }
        };

        // gate = input @ gate_proj^T -> [seq_len, hidden_dim]
        let gate_t = self.gate_proj.transpose_gpu(device, cache);
        let gate = matmul(device, cache, &input_2d, &gate_t);

        // up = input @ up_proj^T -> [seq_len, hidden_dim]
        let up_t = self.up_proj.transpose_gpu(device, cache);
        let up = matmul(device, cache, &input_2d, &up_t);

        // silu(gate) * up -> fused in one kernel
        let activated = swiglu_fused(device, cache, &gate, &up);

        // down = activated @ down_proj^T -> [seq_len, dim]
        let down_t = self.down_proj.transpose_gpu(device, cache);
        let result = matmul(device, cache, &activated, &down_t);

        if was_1d {
            GpuTensor {
                buffer: result.buffer,
                shape: vec![self.dim],
            }
        } else {
            result
        }
    }
}

/// Fused SiLU(gate) * up kernel.
fn swiglu_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    gate: &GpuTensor,
    up: &GpuTensor,
) -> GpuTensor {
    assert_eq!(gate.numel(), up.numel());
    let numel = gate.numel() as u32;
    let out = GpuTensor::uninit(device, gate.shape());

    let wgsl = r#"// Fused SiLU(gate) * up: output[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let g = gate[idx];
    let silu_g = g / (1.0 + exp(-g));
    output[idx] = silu_g * up[idx];
}
"#;

    cache.dispatch_rr_w(
        device,
        wgsl,
        &gate.buffer,
        &up.buffer,
        &out.buffer,
        &[numel, 0, 0, 0],
    );
    out
}

// ---------------------------------------------------------------------------
// GpuRoPE
// ---------------------------------------------------------------------------

/// Rotary position embeddings (RoPE) with configurable frequency base.
///
/// Applies rotation to pairs of dimensions in the input. Supports
/// base frequencies up to 500K for long-context models.
pub struct GpuRoPE {
    /// Precomputed cos table [max_seq_len, head_dim/2].
    pub cos_table: GpuTensor,
    /// Precomputed sin table [max_seq_len, head_dim/2].
    pub sin_table: GpuTensor,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl GpuRoPE {
    /// Create RoPE with precomputed frequency tables.
    ///
    /// `base`: frequency base (10000.0 standard, up to 500000.0 for long context).
    /// `head_dim`: dimension of each attention head (must be even).
    /// `max_seq_len`: maximum sequence length to precompute.
    pub fn new(device: &GpuDevice, head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
        let half = head_dim / 2;

        let mut cos_data = vec![0.0f32; max_seq_len * half];
        let mut sin_data = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }

        Self {
            cos_table: GpuTensor::from_slice(device, &cos_data, &[max_seq_len, half]),
            sin_table: GpuTensor::from_slice(device, &sin_data, &[max_seq_len, half]),
            head_dim,
            max_seq_len,
        }
    }

    /// Apply rotary embeddings to input tensor.
    ///
    /// Input: `[seq_len, n_heads, head_dim]`. Output: same shape.
    /// `start_pos`: position offset (for KV cache continuation).
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
        start_pos: usize,
    ) -> GpuTensor {
        assert_eq!(
            input.ndim(),
            3,
            "RoPE input must be [seq_len, n_heads, head_dim]"
        );
        let seq_len = input.shape()[0];
        let n_heads = input.shape()[1];
        let head_dim = input.shape()[2];
        assert_eq!(head_dim, self.head_dim);
        assert!(
            start_pos + seq_len <= self.max_seq_len,
            "RoPE: position {} + seq_len {} exceeds max {}",
            start_pos,
            seq_len,
            self.max_seq_len
        );

        let half = head_dim / 2;
        let out = GpuTensor::uninit(device, input.shape());

        let wgsl = r#"// RoPE: rotate pairs of dimensions using precomputed cos/sin
// Input/output: [seq_len, n_heads, head_dim]
// cos/sin tables: [max_seq_len, head_dim/2]

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    start_pos: u32,
    max_seq_len: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.n_heads * params.half_dim;
    if (idx >= total) { return; }

    // Decode: idx -> (pos_in_seq, head, pair)
    let pair = idx % params.half_dim;
    let remainder = idx / params.half_dim;
    let head = remainder % params.n_heads;
    let pos_in_seq = remainder / params.n_heads;
    let pos = pos_in_seq + params.start_pos;

    // Input indices for the pair
    let base_idx = (pos_in_seq * params.n_heads + head) * params.head_dim;
    let i0 = base_idx + pair;
    let i1 = base_idx + pair + params.half_dim;

    // Lookup cos/sin for this position and dimension pair
    let table_idx = pos * params.half_dim + pair;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    let x0 = input[i0];
    let x1 = input[i1];

    // Apply rotation
    output[i0] = x0 * c - x1 * s;
    output[i1] = x0 * s + x1 * c;
}
"#;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            seq_len: u32,
            n_heads: u32,
            head_dim: u32,
            half_dim: u32,
            start_pos: u32,
            max_seq_len: u32,
            _pad2: u32,
            _pad3: u32,
        }
        let uniform = RopeParams {
            seq_len: seq_len as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half as u32,
            start_pos: start_pos as u32,
            max_seq_len: self.max_seq_len as u32,
            _pad2: 0,
            _pad3: 0,
        };

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },  // input
            BindingSpec::Storage { read_only: true },  // cos_table
            BindingSpec::Storage { read_only: true },  // sin_table
            BindingSpec::Storage { read_only: false }, // output
            BindingSpec::Uniform,                      // params
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rope params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cos_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sin_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rope dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * n_heads * half) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuInterleavedRoPE
// ---------------------------------------------------------------------------

/// Rotary position embeddings with interleaved pair convention.
///
/// Pairs `(d, d+1)` for `d = 0, 2, 4, ...` — matching the CPU `RotaryEmbedding`
/// convention used by gaia/tang-train. Use this when weights were trained with
/// interleaved pairs rather than the halved convention of [`GpuRoPE`].
pub struct GpuInterleavedRoPE {
    /// Precomputed cos table `[max_seq_len, head_dim/2]`.
    pub cos_table: GpuTensor,
    /// Precomputed sin table `[max_seq_len, head_dim/2]`.
    pub sin_table: GpuTensor,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl GpuInterleavedRoPE {
    /// Create with precomputed frequency tables matching CPU interleaved convention.
    ///
    /// `base`: frequency base (e.g. 500000.0 for long-context models).
    /// `head_dim`: dimension of each attention head (must be even).
    /// `max_seq_len`: maximum sequence length to precompute.
    pub fn new(device: &GpuDevice, head_dim: usize, max_seq_len: usize, base: f64) -> Self {
        assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
        let half = head_dim / 2;

        // theta_i = pos / base^(2i/dim) — identical to CPU RotaryEmbedding::with_base
        let mut cos_data = vec![0.0f32; max_seq_len * half];
        let mut sin_data = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = pos as f64 / base.powf(2.0 * i as f64 / head_dim as f64);
                cos_data[pos * half + i] = theta.cos() as f32;
                sin_data[pos * half + i] = theta.sin() as f32;
            }
        }

        Self {
            cos_table: GpuTensor::from_slice(device, &cos_data, &[max_seq_len, half]),
            sin_table: GpuTensor::from_slice(device, &sin_data, &[max_seq_len, half]),
            head_dim,
            max_seq_len,
        }
    }

    /// Apply interleaved RoPE to input tensor.
    ///
    /// Input: `[seq_len, n_heads, head_dim]`. Output: same shape.
    /// `start_pos`: position offset (for KV cache continuation).
    ///
    /// For each position and head, rotates interleaved pairs:
    /// - `output[2i]   = x[2i] * cos - x[2i+1] * sin`
    /// - `output[2i+1] = x[2i] * sin + x[2i+1] * cos`
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
        start_pos: usize,
    ) -> GpuTensor {
        assert_eq!(
            input.ndim(),
            3,
            "RoPE input must be [seq_len, n_heads, head_dim]"
        );
        let seq_len = input.shape()[0];
        let n_heads = input.shape()[1];
        let head_dim = input.shape()[2];
        assert_eq!(head_dim, self.head_dim);
        assert!(
            start_pos + seq_len <= self.max_seq_len,
            "RoPE: position {} + seq_len {} exceeds max {}",
            start_pos,
            seq_len,
            self.max_seq_len
        );

        let half = head_dim / 2;
        let out = GpuTensor::uninit(device, input.shape());

        let wgsl = r#"// Interleaved RoPE: rotate pairs (2i, 2i+1)

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    start_pos: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.n_heads * params.half_dim;
    if (idx >= total) { return; }

    let pair = idx % params.half_dim;
    let remainder = idx / params.half_dim;
    let head = remainder % params.n_heads;
    let pos_in_seq = remainder / params.n_heads;
    let pos = pos_in_seq + params.start_pos;

    let base_idx = (pos_in_seq * params.n_heads + head) * params.head_dim;
    let i0 = base_idx + pair * 2u;
    let i1 = base_idx + pair * 2u + 1u;

    let table_idx = pos * params.half_dim + pair;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    let x0 = input[i0];
    let x1 = input[i1];

    output[i0] = x0 * c - x1 * s;
    output[i1] = x0 * s + x1 * c;
}
"#;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            seq_len: u32,
            n_heads: u32,
            head_dim: u32,
            half_dim: u32,
            start_pos: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }
        let uniform = RopeParams {
            seq_len: seq_len as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half as u32,
            start_pos: start_pos as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: false },
            BindingSpec::Uniform,
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interleaved rope params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("interleaved rope bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cos_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sin_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interleaved rope dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("interleaved rope compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * n_heads * half) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuKVCache
// ---------------------------------------------------------------------------

/// Simple KV cache for autoregressive inference.
///
/// Stores key and value tensors and supports appending new entries
/// and retrieving the full sequence. Fixed maximum sequence length.
pub struct GpuKVCache {
    /// Key cache [max_seq_len, n_kv_heads, head_dim].
    pub keys: GpuBuffer,
    /// Value cache [max_seq_len, n_kv_heads, head_dim].
    pub values: GpuBuffer,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Current number of cached positions.
    pub len: usize,
}

impl GpuKVCache {
    /// Create an empty KV cache.
    pub fn new(device: &GpuDevice, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let size = max_seq_len * n_kv_heads * head_dim;
        Self {
            keys: GpuBuffer::uninit(device, size),
            values: GpuBuffer::uninit(device, size),
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Append new key and value entries to the cache.
    ///
    /// `new_keys`, `new_values`: `[new_seq_len, n_kv_heads, head_dim]`.
    pub fn append(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        new_keys: &GpuTensor,
        new_values: &GpuTensor,
    ) {
        let new_seq = new_keys.shape()[0];
        assert!(
            self.len + new_seq <= self.max_seq_len,
            "KV cache overflow: {} + {} > {}",
            self.len,
            new_seq,
            self.max_seq_len
        );

        let row_size = self.n_kv_heads * self.head_dim;
        let offset_bytes = (self.len * row_size * std::mem::size_of::<f32>()) as u64;
        let copy_bytes = (new_seq * row_size * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache append"),
            });

        encoder.copy_buffer_to_buffer(
            &new_keys.buffer.buffer,
            0,
            &self.keys.buffer,
            offset_bytes,
            copy_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &new_values.buffer.buffer,
            0,
            &self.values.buffer,
            offset_bytes,
            copy_bytes,
        );

        cache.submit_or_enqueue(device, encoder.finish());
        self.len += new_seq;
    }

    /// Get cached keys as a tensor of shape `[current_len, n_kv_heads, head_dim]`.
    pub fn get_keys(&self, device: &GpuDevice) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let data = self.keys.to_vec_sync(device);
        GpuTensor::from_slice(
            device,
            &data[..total],
            &[self.len, self.n_kv_heads, self.head_dim],
        )
    }

    /// Get cached values as a tensor of shape `[current_len, n_kv_heads, head_dim]`.
    pub fn get_values(&self, device: &GpuDevice) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let data = self.values.to_vec_sync(device);
        GpuTensor::from_slice(
            device,
            &data[..total],
            &[self.len, self.n_kv_heads, self.head_dim],
        )
    }

    /// Get cached keys as a tensor, staying on GPU (no CPU roundtrip).
    ///
    /// Returns `[current_len, n_kv_heads, head_dim]`.
    pub fn get_keys_gpu(&self, device: &GpuDevice, cache: &mut KernelCache) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let dst = GpuBuffer::uninit(device, total);
        let copy_bytes = (total * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache get_keys_gpu"),
            });
        encoder.copy_buffer_to_buffer(&self.keys.buffer, 0, &dst.buffer, 0, copy_bytes);
        cache.submit_or_enqueue(device, encoder.finish());

        GpuTensor {
            buffer: dst,
            shape: vec![self.len, self.n_kv_heads, self.head_dim],
        }
    }

    /// Get cached values as a tensor, staying on GPU (no CPU roundtrip).
    ///
    /// Returns `[current_len, n_kv_heads, head_dim]`.
    pub fn get_values_gpu(&self, device: &GpuDevice, cache: &mut KernelCache) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let dst = GpuBuffer::uninit(device, total);
        let copy_bytes = (total * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache get_values_gpu"),
            });
        encoder.copy_buffer_to_buffer(&self.values.buffer, 0, &dst.buffer, 0, copy_bytes);
        cache.submit_or_enqueue(device, encoder.finish());

        GpuTensor {
            buffer: dst,
            shape: vec![self.len, self.n_kv_heads, self.head_dim],
        }
    }

    /// Reset the cache (set length to 0).
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// GpuCausalAttention
// ---------------------------------------------------------------------------

/// Grouped-query attention with causal mask.
///
/// Supports variable numbers of query heads per KV head (GQA).
/// Uses a fused attention kernel with causal masking for autoregressive
/// inference.
pub struct GpuCausalAttention {
    /// Query projection [n_heads * head_dim, dim].
    pub wq: GpuTensor,
    /// Key projection [n_kv_heads * head_dim, dim].
    pub wk: GpuTensor,
    /// Value projection [n_kv_heads * head_dim, dim].
    pub wv: GpuTensor,
    /// Output projection [dim, n_heads * head_dim].
    pub wo: GpuTensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dim: usize,
    // Training state
    cached_input: Option<Vec<f32>>,
    cached_q: Option<Vec<f32>>,
    cached_k: Option<Vec<f32>>,
    cached_v: Option<Vec<f32>>,
    cached_attn_out: Option<Vec<f32>>,
    wq_grad: Option<GpuTensor>,
    wk_grad: Option<GpuTensor>,
    wv_grad: Option<GpuTensor>,
    wo_grad: Option<GpuTensor>,
}

impl GpuCausalAttention {
    /// Create attention with zero-initialized projections.
    pub fn zeros(
        device: &GpuDevice,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert_eq!(
            n_heads % n_kv_heads,
            0,
            "n_heads must be divisible by n_kv_heads"
        );
        let q_size = n_heads * head_dim * dim;
        let kv_size = n_kv_heads * head_dim * dim;
        let o_size = dim * n_heads * head_dim;
        Self {
            wq: GpuTensor::from_slice(device, &vec![0.0f32; q_size], &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(
                device,
                &vec![0.0f32; kv_size],
                &[n_kv_heads * head_dim, dim],
            ),
            wv: GpuTensor::from_slice(
                device,
                &vec![0.0f32; kv_size],
                &[n_kv_heads * head_dim, dim],
            ),
            wo: GpuTensor::from_slice(device, &vec![0.0f32; o_size], &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn_out: None,
            wq_grad: None,
            wk_grad: None,
            wv_grad: None,
            wo_grad: None,
        }
    }

    /// Forward pass with grouped-query attention and causal masking.
    ///
    /// Input: `[seq_len, dim]`. Output: `[seq_len, dim]`.
    ///
    /// The causal mask ensures position `i` can only attend to positions `<= i`.
    /// KV heads are shared across query head groups (GQA).
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        use crate::matmul::matmul;

        assert_eq!(input.ndim(), 2, "attention input must be [seq_len, dim]");
        let seq_len = input.shape()[0];

        // Q = input @ Wq^T -> [seq_len, n_heads * head_dim]
        let wq_t = self.wq.transpose_gpu(device, cache);
        let q_flat = matmul(device, cache, input, &wq_t);

        // K = input @ Wk^T -> [seq_len, n_kv_heads * head_dim]
        let wk_t = self.wk.transpose_gpu(device, cache);
        let k_flat = matmul(device, cache, input, &wk_t);

        // V = input @ Wv^T -> [seq_len, n_kv_heads * head_dim]
        let wv_t = self.wv.transpose_gpu(device, cache);
        let v_flat = matmul(device, cache, input, &wv_t);

        // Fused causal attention with GQA
        let attn_out = causal_attention_fused(
            device,
            cache,
            &q_flat,
            &k_flat,
            &v_flat,
            seq_len,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );

        // Output projection: attn_out @ Wo^T -> [seq_len, dim]
        let wo_t = self.wo.transpose_gpu(device, cache);
        matmul(device, cache, &attn_out, &wo_t)
    }
}

/// Fused causal attention with grouped-query support.
///
/// Q: [seq_len, n_heads * head_dim], K: [seq_len, n_kv_heads * head_dim],
/// V: [seq_len, n_kv_heads * head_dim] -> output: [seq_len, n_heads * head_dim].
fn causal_attention_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> GpuTensor {
    let heads_per_group = n_heads / n_kv_heads;
    let out = GpuTensor::uninit(device, &[seq_len, n_heads * head_dim]);

    // Use a workgroup per (query_pos, head) pair.
    // Each workgroup computes attention for one query position and one head.
    let wg_size = (seq_len as u32).next_power_of_two().clamp(1, 256);

    let wgsl = format!(
        r#"// Fused causal GQA attention
// One workgroup per (query_pos, head) pair
// Each thread covers one or more key positions

struct Params {{
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_per_group: u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
}}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_scores: array<f32, {wg_size}>;
var<workgroup> wg_max: array<f32, {wg_size}>;
var<workgroup> wg_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let query_pos = wg_id.x;
    let head = wg_id.y;
    if (query_pos >= params.seq_len || head >= params.n_heads) {{ return; }}
    let tid = lid.x;
    let kv_head = head / params.heads_per_group;

    // Q offset for this head: Q[query_pos, head * head_dim .. (head+1) * head_dim]
    let q_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    // K/V row stride
    let kv_stride = params.n_kv_heads * params.head_dim;

    // Phase 1: Compute dot products Q . K for each key position
    // Each thread handles a subset of key positions
    var local_max: f32 = -3.402823e+38;
    for (var kp = tid; kp < params.seq_len; kp = kp + WG) {{
        if (kp <= query_pos) {{
            // Causal: only attend to positions <= query_pos
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            let score = dot * params.scale;
            wg_scores[kp % WG] = score;
            local_max = max(local_max, score);
        }}
    }}
    wg_max[tid] = local_max;
    workgroupBarrier();

    // Reduce max
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_max[tid] = max(wg_max[tid], wg_max[tid + stride]);
        }}
        workgroupBarrier();
    }}
    let row_max = wg_max[0];
    workgroupBarrier();

    // Phase 2: Compute exp(score - max) and accumulate weighted V
    // For small seq_len, we can store all scores and do two passes.
    // For larger seq_len, we use online softmax.

    // Initialize output accumulator (per-thread partial sums)
    // Since head_dim could be large, we accumulate across all positions per dimension

    // Simple approach: iterate key positions, accumulate weighted V
    var local_exp_sum: f32 = 0.0;

    // Output accumulator: we need head_dim values.
    // We iterate over key positions and accumulate on the fly.
    // Since we only have WG shared memory slots, and head_dim could be > WG,
    // we compute in two passes: first get softmax weights, then weighted sum.

    // Pass 1: compute exp weights and sum
    for (var kp = tid; kp < params.seq_len; kp = kp + WG) {{
        if (kp <= query_pos) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            let w = exp(dot * params.scale - row_max);
            wg_scores[kp % WG] = w;
            local_exp_sum = local_exp_sum + w;
        }}
    }}
    wg_sum[tid] = local_exp_sum;
    workgroupBarrier();

    // Reduce sum
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
        }}
        workgroupBarrier();
    }}
    let total_sum = wg_sum[0];
    let inv_sum = 1.0 / total_sum;
    workgroupBarrier();

    // Phase 3: Compute weighted V sum, each thread handles a subset of head_dim
    let out_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    for (var d = tid; d < params.head_dim; d = d + WG) {{
        var acc: f32 = 0.0;
        for (var kp: u32 = 0u; kp <= query_pos; kp = kp + 1u) {{
            let v_base = kp * kv_stride + kv_head * params.head_dim;
            // Recompute weight (avoids needing seq_len shared memory)
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var dd: u32 = 0u; dd < params.head_dim; dd = dd + 1u) {{
                dot = dot + Q[q_base + dd] * K[k_base + dd];
            }}
            let w = exp(dot * params.scale - row_max) * inv_sum;
            acc = acc + w * V[v_base + d];
        }}
        output[out_base + d] = acc;
    }}
}}
"#,
        wg_size = wg_size,
    );

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttnParams {
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,
        scale: f32,
        _pad2: u32,
        _pad3: u32,
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let uniform = AttnParams {
        seq_len: seq_len as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        heads_per_group: heads_per_group as u32,
        scale,
        _pad2: 0,
        _pad3: 0,
    };

    let hash = hash_wgsl(&wgsl);
    let bindings = [
        BindingSpec::Storage { read_only: true },  // Q
        BindingSpec::Storage { read_only: true },  // K
        BindingSpec::Storage { read_only: true },  // V
        BindingSpec::Storage { read_only: false }, // output
        BindingSpec::Uniform,                      // params
    ];
    let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("causal attn params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("causal attn bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("causal attn dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("causal attn compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per (query_pos, head) pair
        pass.dispatch_workgroups(seq_len as u32, n_heads as u32, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());
    out
}

/// Fused causal attention with separate Q and KV sequence lengths.
///
/// Supports KV-cached inference where Q has `q_len` new positions and
/// K/V have `kv_len` total cached positions (including the new ones).
///
/// - `q`: `[q_len, n_heads * head_dim]`
/// - `k`: `[kv_len, n_kv_heads * head_dim]`
/// - `v`: `[kv_len, n_kv_heads * head_dim]`
/// - `start_pos`: absolute position of the first query token. Query at batch
///   index `i` can attend to key positions `0..=start_pos+i`.
///
/// Returns: `[q_len, n_heads * head_dim]`.
pub fn kv_attention_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    q_len: usize,
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> GpuTensor {
    let heads_per_group = n_heads / n_kv_heads;
    let out = GpuTensor::uninit(device, &[q_len, n_heads * head_dim]);

    let wg_size = (kv_len as u32).next_power_of_two().clamp(1, 256);

    let wgsl = format!(
        r#"// Fused causal GQA attention with separate Q/KV lengths
// One workgroup per (query_pos, head) pair

struct Params {{
    q_len: u32,
    kv_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_per_group: u32,
    scale: f32,
    start_pos: u32,
}}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_max: array<f32, {wg_size}>;
var<workgroup> wg_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let query_pos = wg_id.x;
    let head = wg_id.y;
    if (query_pos >= params.q_len || head >= params.n_heads) {{ return; }}
    let tid = lid.x;
    let kv_head = head / params.heads_per_group;

    let q_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    let kv_stride = params.n_kv_heads * params.head_dim;
    let max_attend = params.start_pos + query_pos;

    // Phase 1: compute dot products and find max
    var local_max: f32 = -3.402823e+38;
    for (var kp = tid; kp < params.kv_len; kp = kp + WG) {{
        if (kp <= max_attend) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            local_max = max(local_max, dot * params.scale);
        }}
    }}
    wg_max[tid] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_max[tid] = max(wg_max[tid], wg_max[tid + stride]);
        }}
        workgroupBarrier();
    }}
    let row_max = wg_max[0];
    workgroupBarrier();

    // Phase 2: compute exp weights and sum
    var local_exp_sum: f32 = 0.0;
    for (var kp = tid; kp < params.kv_len; kp = kp + WG) {{
        if (kp <= max_attend) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            local_exp_sum = local_exp_sum + exp(dot * params.scale - row_max);
        }}
    }}
    wg_sum[tid] = local_exp_sum;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
        }}
        workgroupBarrier();
    }}
    let total_sum = wg_sum[0];
    let inv_sum = 1.0 / total_sum;
    workgroupBarrier();

    // Phase 3: compute weighted V sum
    let out_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    for (var d = tid; d < params.head_dim; d = d + WG) {{
        var acc: f32 = 0.0;
        for (var kp: u32 = 0u; kp <= max_attend && kp < params.kv_len; kp = kp + 1u) {{
            let v_base = kp * kv_stride + kv_head * params.head_dim;
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var dd: u32 = 0u; dd < params.head_dim; dd = dd + 1u) {{
                dot = dot + Q[q_base + dd] * K[k_base + dd];
            }}
            let w = exp(dot * params.scale - row_max) * inv_sum;
            acc = acc + w * V[v_base + d];
        }}
        output[out_base + d] = acc;
    }}
}}
"#,
        wg_size = wg_size,
    );

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct KvAttnParams {
        q_len: u32,
        kv_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,
        scale: f32,
        start_pos: u32,
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let uniform = KvAttnParams {
        q_len: q_len as u32,
        kv_len: kv_len as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        heads_per_group: heads_per_group as u32,
        scale,
        start_pos: start_pos as u32,
    };

    let hash = hash_wgsl(&wgsl);
    let bindings = [
        BindingSpec::Storage { read_only: true },  // Q
        BindingSpec::Storage { read_only: true },  // K
        BindingSpec::Storage { read_only: true },  // V
        BindingSpec::Storage { read_only: false }, // output
        BindingSpec::Uniform,                      // params
    ];
    let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kv attn params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kv attn bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kv attn dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kv attn compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(q_len as u32, n_heads as u32, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());
    out
}

/// Fused SiLU(gate) * up kernel (public version for use outside this module).
///
/// `gate` and `up` must have the same number of elements.
/// Returns `output[i] = silu(gate[i]) * up[i]`.
pub fn swiglu_fused_pub(
    device: &GpuDevice,
    cache: &mut KernelCache,
    gate: &GpuTensor,
    up: &GpuTensor,
) -> GpuTensor {
    swiglu_fused(device, cache, gate, up)
}

// ===========================================================================
// GpuTrainModule implementations — backward passes for LLM components
// ===========================================================================

// ---------------------------------------------------------------------------
// GpuEmbedding backward (CPU scatter-add)
// ---------------------------------------------------------------------------

impl GpuTrainModule for GpuEmbedding {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // input is a 1D tensor of shape [seq_len] containing token IDs as f32.
        // We need to convert to u32 and upload as a GpuBuffer for the forward kernel.
        cache.flush(device);
        let input_f32 = input.buffer.to_vec_sync(device);
        let seq_len = input_f32.len();
        let token_ids: Vec<u32> = input_f32.iter().map(|&x| x as u32).collect();

        // Cache indices for backward
        self.cached_indices = Some(token_ids.clone());
        self.cached_seq_len = Some(seq_len);

        // Create u32 buffer for embedding lookup
        use wgpu::util::DeviceExt;
        let id_buf = GpuBuffer {
            buffer: device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embedding train token ids"),
                    contents: bytemuck::cast_slice(&token_ids),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                }),
            len: seq_len,
        };

        self.forward(device, cache, &id_buf, seq_len)
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let indices = self
            .cached_indices
            .as_ref()
            .expect("must call forward_train before backward");
        let seq_len = self.cached_seq_len.unwrap();

        // Download grad_output [seq_len, dim]
        cache.flush(device);
        let grad_data = grad_output.buffer.to_vec_sync(device);

        // Scatter-add into weight gradient [vocab_size, dim]
        let mut grad_w = vec![0.0f32; self.vocab_size * self.dim];
        for s in 0..seq_len {
            let idx = indices[s] as usize;
            for e in 0..self.dim {
                grad_w[idx * self.dim + e] += grad_data[s * self.dim + e];
            }
        }

        self.weight_grad = Some(GpuTensor::from_slice(
            device,
            &grad_w,
            &[self.vocab_size, self.dim],
        ));

        // No meaningful gradient for integer indices — return zeros
        GpuTensor::from_slice(device, &vec![0.0f32; seq_len], &[seq_len])
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.weight]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![self.weight_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.cached_indices = None;
        self.cached_seq_len = None;
    }
}

// ---------------------------------------------------------------------------
// GpuRMSNorm backward (CPU)
// ---------------------------------------------------------------------------

impl GpuTrainModule for GpuRMSNorm {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Cache input data on CPU for backward
        cache.flush(device);
        let input_data = input.buffer.to_vec_sync(device);
        let n_groups = input.numel() / self.dim;

        // Compute RMS per group on CPU
        let mut rms_vals = vec![0.0f32; n_groups];
        for g in 0..n_groups {
            let base = g * self.dim;
            let mut sum_sq = 0.0f32;
            for i in 0..self.dim {
                let v = input_data[base + i];
                sum_sq += v * v;
            }
            rms_vals[g] = (sum_sq / self.dim as f32 + self.eps).sqrt();
        }

        self.cached_input = Some(input_data);
        self.cached_rms = Some(rms_vals);

        // Run GPU forward as normal
        self.forward(device, cache, input)
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let input_data = self
            .cached_input
            .as_ref()
            .expect("must call forward_train before backward");
        let rms_vals = self
            .cached_rms
            .as_ref()
            .expect("must call forward_train before backward");

        cache.flush(device);
        let grad_out_data = grad_output.buffer.to_vec_sync(device);
        let weight_data = self.weight.buffer.to_vec_sync(device);

        let n_groups = rms_vals.len();
        let dim = self.dim;
        let n = dim as f32;

        let mut grad_input = vec![0.0f32; input_data.len()];
        let mut grad_weight = vec![0.0f32; dim];

        for g in 0..n_groups {
            let base = g * dim;
            let r = rms_vals[g];
            let inv_r = 1.0 / r;

            // dot = sum(weight[f] * grad_output[f] * input[f])
            let mut dot = 0.0f32;
            for f in 0..dim {
                dot += weight_data[f] * grad_out_data[base + f] * input_data[base + f];
            }

            for f in 0..dim {
                let w_grad = weight_data[f] * grad_out_data[base + f];
                grad_input[base + f] = (w_grad - input_data[base + f] * dot / (n * r * r)) * inv_r;
                // Accumulate weight gradient
                grad_weight[f] += grad_out_data[base + f] * input_data[base + f] * inv_r;
            }
        }

        self.weight_grad = Some(GpuTensor::from_slice(device, &grad_weight, &[dim]));
        GpuTensor::from_slice(device, &grad_input, grad_output.shape())
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.weight]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![self.weight_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.cached_input = None;
        self.cached_rms = None;
    }
}

// ---------------------------------------------------------------------------
// Interleaved RoPE backward
// ---------------------------------------------------------------------------

/// Backward pass for interleaved RoPE: transpose rotation.
///
/// For forward: y0 = x0*cos - x1*sin, y1 = x0*sin + x1*cos
/// Backward:   dx0 = dy0*cos + dy1*sin, dx1 = -dy0*sin + dy1*cos
///
/// No trainable parameters. `start_pos` must match the forward pass.
pub fn interleaved_rope_backward(
    rope: &GpuInterleavedRoPE,
    device: &GpuDevice,
    cache: &mut KernelCache,
    grad_output: &GpuTensor,
    start_pos: usize,
) -> GpuTensor {
    assert_eq!(
        grad_output.ndim(),
        3,
        "RoPE grad must be [seq_len, n_heads, head_dim]"
    );
    let seq_len = grad_output.shape()[0];
    let n_heads = grad_output.shape()[1];
    let head_dim = grad_output.shape()[2];
    assert_eq!(head_dim, rope.head_dim);

    let half = head_dim / 2;
    let out = GpuTensor::uninit(device, grad_output.shape());

    let wgsl = r#"// Interleaved RoPE backward: transposed rotation

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    start_pos: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> grad_out: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_in: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.n_heads * params.half_dim;
    if (idx >= total) { return; }

    let pair = idx % params.half_dim;
    let remainder = idx / params.half_dim;
    let head = remainder % params.n_heads;
    let pos_in_seq = remainder / params.n_heads;
    let pos = pos_in_seq + params.start_pos;

    let base_idx = (pos_in_seq * params.n_heads + head) * params.head_dim;
    let i0 = base_idx + pair * 2u;
    let i1 = base_idx + pair * 2u + 1u;

    let table_idx = pos * params.half_dim + pair;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    let dy0 = grad_out[i0];
    let dy1 = grad_out[i1];

    // Transposed rotation
    grad_in[i0] = dy0 * c + dy1 * s;
    grad_in[i1] = -dy0 * s + dy1 * c;
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RopeParams {
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        half_dim: u32,
        start_pos: u32,
        _pad1: u32,
        _pad2: u32,
        _pad3: u32,
    }
    let uniform = RopeParams {
        seq_len: seq_len as u32,
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        half_dim: half as u32,
        start_pos: start_pos as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    let hash = hash_wgsl(wgsl);
    let bindings = [
        BindingSpec::Storage { read_only: true },
        BindingSpec::Storage { read_only: true },
        BindingSpec::Storage { read_only: true },
        BindingSpec::Storage { read_only: false },
        BindingSpec::Uniform,
    ];
    let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope backward params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope backward bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grad_output.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rope.cos_table.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rope.sin_table.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope backward dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope backward compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let total = (seq_len * n_heads * half) as u32;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }
    cache.submit_or_enqueue(device, encoder.finish());
    out
}

// ---------------------------------------------------------------------------
// GpuSwiGLU backward (GPU matmuls + elementwise)
// ---------------------------------------------------------------------------

impl GpuTrainModule for GpuSwiGLU {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Ensure 2D
        let was_1d = input.ndim() == 1;
        let input_2d = if was_1d {
            GpuTensor {
                buffer: input.buffer.clone_gpu_batched(device, cache),
                shape: vec![1, input.numel()],
            }
        } else {
            input.clone_gpu_batched(device, cache)
        };

        // Cache input
        self.cached_input = Some(input_2d.clone_gpu_batched(device, cache));

        // gate_raw = input @ gate_proj^T -> [seq_len, hidden_dim]
        let gate_t = self.gate_proj.transpose_gpu(device, cache);
        let gate_raw = matmul(device, cache, &input_2d, &gate_t);
        self.cached_gate_raw = Some(gate_raw.clone_gpu_batched(device, cache));

        // up_raw = input @ up_proj^T -> [seq_len, hidden_dim]
        let up_t = self.up_proj.transpose_gpu(device, cache);
        let up_raw = matmul(device, cache, &input_2d, &up_t);
        self.cached_up_raw = Some(up_raw.clone_gpu_batched(device, cache));

        // gate_silu = silu(gate_raw)
        let gate_silu = map_elementwise(device, cache, &[&gate_raw], |args| {
            use tang::Scalar;
            let x = args[0];
            let one = tang_expr::ExprId::from_f64(1.0);
            x / (one + (-x).exp())
        });
        self.cached_gate_silu = Some(gate_silu.clone_gpu_batched(device, cache));

        // activated = gate_silu * up_raw
        let activated = map_elementwise(device, cache, &[&gate_silu, &up_raw], |args| {
            args[0] * args[1]
        });
        self.cached_activated = Some(activated.clone_gpu_batched(device, cache));

        // result = activated @ down_proj^T -> [seq_len, dim]
        let down_t = self.down_proj.transpose_gpu(device, cache);
        let result = matmul(device, cache, &activated, &down_t);

        if was_1d {
            GpuTensor {
                buffer: result.buffer,
                shape: vec![self.dim],
            }
        } else {
            result
        }
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("must call forward_train before backward");
        let gate_raw = self
            .cached_gate_raw
            .as_ref()
            .expect("cached_gate_raw missing");
        let up_raw = self.cached_up_raw.as_ref().expect("cached_up_raw missing");
        let gate_silu = self
            .cached_gate_silu
            .as_ref()
            .expect("cached_gate_silu missing");
        let activated = self
            .cached_activated
            .as_ref()
            .expect("cached_activated missing");

        // Ensure 2D grad_output
        let was_1d = grad_output.ndim() == 1;
        let grad_out_2d = if was_1d {
            GpuTensor {
                buffer: grad_output.buffer.clone_gpu_batched(device, cache),
                shape: vec![1, grad_output.numel()],
            }
        } else {
            grad_output.clone_gpu_batched(device, cache)
        };

        // 1. grad_hidden = grad_output @ down_proj -> [seq, hidden_dim]
        let grad_hidden = matmul(device, cache, &grad_out_2d, &self.down_proj);

        // 2. down_proj_grad = grad_output^T @ activated -> [dim, hidden_dim]
        let grad_out_t = grad_out_2d.transpose_gpu(device, cache);
        self.down_proj_grad = Some(matmul(device, cache, &grad_out_t, activated));

        // 3. grad_gate_silu = grad_hidden * up_raw
        let grad_gate_silu = map_elementwise(device, cache, &[&grad_hidden, up_raw], |args| {
            args[0] * args[1]
        });

        // 4. grad_up_raw = grad_hidden * gate_silu
        let grad_up_raw = map_elementwise(device, cache, &[&grad_hidden, gate_silu], |args| {
            args[0] * args[1]
        });

        // 5. SiLU derivative: dsilu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        // 6. grad_gate_raw = grad_gate_silu * dsilu
        let grad_gate_raw = map_elementwise(device, cache, &[&grad_gate_silu, gate_raw], |args| {
            use tang::Scalar;
            let grad = args[0];
            let x = args[1];
            let one = tang_expr::ExprId::from_f64(1.0);
            let sig = one / (one + (-x).exp());
            let dsilu = sig * (one + x * (one - sig));
            grad * dsilu
        });

        // 7. gate_proj_grad = grad_gate_raw^T @ input -> [hidden_dim, dim]
        let grad_gate_t = grad_gate_raw.transpose_gpu(device, cache);
        self.gate_proj_grad = Some(matmul(device, cache, &grad_gate_t, cached_input));

        // 8. up_proj_grad = grad_up_raw^T @ input -> [hidden_dim, dim]
        let grad_up_t = grad_up_raw.transpose_gpu(device, cache);
        self.up_proj_grad = Some(matmul(device, cache, &grad_up_t, cached_input));

        // 9. grad_input = grad_gate_raw @ gate_proj + grad_up_raw @ up_proj
        let gi_gate = matmul(device, cache, &grad_gate_raw, &self.gate_proj);
        let gi_up = matmul(device, cache, &grad_up_raw, &self.up_proj);
        let grad_input = add_tensors(device, cache, &gi_gate, &gi_up);

        if was_1d {
            GpuTensor {
                buffer: grad_input.buffer,
                shape: vec![self.dim],
            }
        } else {
            grad_input
        }
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.gate_proj, &self.up_proj, &self.down_proj]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![
            self.gate_proj_grad.as_ref(),
            self.up_proj_grad.as_ref(),
            self.down_proj_grad.as_ref(),
        ]
    }

    fn zero_grad(&mut self) {
        self.gate_proj_grad = None;
        self.up_proj_grad = None;
        self.down_proj_grad = None;
        self.cached_input = None;
        self.cached_gate_raw = None;
        self.cached_up_raw = None;
        self.cached_gate_silu = None;
        self.cached_activated = None;
    }
}

// ---------------------------------------------------------------------------
// GpuCausalAttention backward (CPU for correctness)
// ---------------------------------------------------------------------------

impl GpuTrainModule for GpuCausalAttention {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        assert_eq!(input.ndim(), 2, "attention input must be [seq_len, dim]");
        let seq_len = input.shape()[0];

        // Q, K, V projections on GPU
        let wq_t = self.wq.transpose_gpu(device, cache);
        let q_flat = matmul(device, cache, input, &wq_t);
        let wk_t = self.wk.transpose_gpu(device, cache);
        let k_flat = matmul(device, cache, input, &wk_t);
        let wv_t = self.wv.transpose_gpu(device, cache);
        let v_flat = matmul(device, cache, input, &wv_t);

        // Fused attention on GPU
        let attn_out = causal_attention_fused(
            device,
            cache,
            &q_flat,
            &k_flat,
            &v_flat,
            seq_len,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );

        // Output projection on GPU
        let wo_t = self.wo.transpose_gpu(device, cache);
        let output = matmul(device, cache, &attn_out, &wo_t);

        // Cache everything on CPU for backward
        cache.flush(device);
        self.cached_input = Some(input.buffer.to_vec_sync(device));
        self.cached_q = Some(q_flat.buffer.to_vec_sync(device));
        self.cached_k = Some(k_flat.buffer.to_vec_sync(device));
        self.cached_v = Some(v_flat.buffer.to_vec_sync(device));
        self.cached_attn_out = Some(attn_out.buffer.to_vec_sync(device));

        output
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let input_data = self.cached_input.as_ref().expect("cached_input missing");
        let q_data = self.cached_q.as_ref().expect("cached_q missing");
        let k_data = self.cached_k.as_ref().expect("cached_k missing");
        let v_data = self.cached_v.as_ref().expect("cached_v missing");
        let attn_out_data = self
            .cached_attn_out
            .as_ref()
            .expect("cached_attn_out missing");

        cache.flush(device);
        let grad_out_data = grad_output.buffer.to_vec_sync(device);
        let wo_data = self.wo.buffer.to_vec_sync(device);
        let wq_data = self.wq.buffer.to_vec_sync(device);
        let wk_data = self.wk.buffer.to_vec_sync(device);
        let wv_data = self.wv.buffer.to_vec_sync(device);

        let seq_len = grad_output.shape()[0];
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let heads_per_kv = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // wo backward: grad_attn_out = grad_output @ wo, wo_grad = grad_output^T @ attn_out
        // wo: [dim, q_dim]
        let mut grad_attn_out = vec![0.0f32; seq_len * q_dim];
        let mut wo_grad_data = vec![0.0f32; self.dim * q_dim];

        for i in 0..seq_len {
            for j in 0..q_dim {
                let mut sum = 0.0f32;
                for k in 0..self.dim {
                    // wo[k, j], grad_output[i, k]
                    sum += grad_out_data[i * self.dim + k] * wo_data[k * q_dim + j];
                }
                grad_attn_out[i * q_dim + j] = sum;
            }
        }
        for k in 0..self.dim {
            for j in 0..q_dim {
                let mut sum = 0.0f32;
                for i in 0..seq_len {
                    sum += grad_out_data[i * self.dim + k] * attn_out_data[i * q_dim + j];
                }
                wo_grad_data[k * q_dim + j] = sum;
            }
        }
        self.wo_grad = Some(GpuTensor::from_slice(
            device,
            &wo_grad_data,
            &[self.dim, q_dim],
        ));

        // Per-head attention backward
        let mut grad_q_full = vec![0.0f32; seq_len * q_dim];
        let mut grad_k_full = vec![0.0f32; seq_len * kv_dim];
        let mut grad_v_full = vec![0.0f32; seq_len * kv_dim];

        for h in 0..self.n_heads {
            let q_offset = h * self.head_dim;
            let kv_group = h / heads_per_kv;
            let kv_offset = kv_group * self.head_dim;

            // Recompute attention weights for this head
            // scores[i, j] = sum_d Q_h[i, d] * K_h[j, d] * scale (causal: j <= i)
            let mut attn_weights = vec![0.0f32; seq_len * seq_len];
            for qi in 0..seq_len {
                let mut row_max = f32::NEG_INFINITY;
                for kp in 0..=qi {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot +=
                            q_data[qi * q_dim + q_offset + d] * k_data[kp * kv_dim + kv_offset + d];
                    }
                    let s = dot * scale;
                    attn_weights[qi * seq_len + kp] = s;
                    row_max = row_max.max(s);
                }
                // Softmax with causal mask
                let mut exp_sum = 0.0f32;
                for kp in 0..=qi {
                    let e = (attn_weights[qi * seq_len + kp] - row_max).exp();
                    attn_weights[qi * seq_len + kp] = e;
                    exp_sum += e;
                }
                let inv_sum = 1.0 / exp_sum;
                for kp in 0..=qi {
                    attn_weights[qi * seq_len + kp] *= inv_sum;
                }
                for kp in (qi + 1)..seq_len {
                    attn_weights[qi * seq_len + kp] = 0.0;
                }
            }

            // grad_attn_h[i, j] = sum_d grad_attn_out_h[i, d] * V_h[j, d]
            let mut grad_attn_h = vec![0.0f32; seq_len * seq_len];
            for qi in 0..seq_len {
                for kp in 0..seq_len {
                    let mut sum = 0.0f32;
                    for d in 0..self.head_dim {
                        sum += grad_attn_out[qi * q_dim + q_offset + d]
                            * v_data[kp * kv_dim + kv_offset + d];
                    }
                    grad_attn_h[qi * seq_len + kp] = sum;
                }
            }

            // grad_V_h[kp, d] = sum_qi attn_weights[qi, kp] * grad_attn_out_h[qi, d]
            for kp in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for qi in 0..seq_len {
                        sum += attn_weights[qi * seq_len + kp]
                            * grad_attn_out[qi * q_dim + q_offset + d];
                    }
                    grad_v_full[kp * kv_dim + kv_offset + d] += sum;
                }
            }

            // Softmax backward: grad_scores[i,j] = attn[i,j] * (grad_attn[i,j] - dot_i)
            // where dot_i = sum_k grad_attn[i,k] * attn[i,k]
            let mut grad_scores = vec![0.0f32; seq_len * seq_len];
            for qi in 0..seq_len {
                let mut dot_i = 0.0f32;
                for kp in 0..seq_len {
                    dot_i += grad_attn_h[qi * seq_len + kp] * attn_weights[qi * seq_len + kp];
                }
                for kp in 0..seq_len {
                    grad_scores[qi * seq_len + kp] = attn_weights[qi * seq_len + kp]
                        * (grad_attn_h[qi * seq_len + kp] - dot_i)
                        * scale;
                }
            }

            // grad_Q_h = grad_scores @ K_h
            for qi in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for kp in 0..seq_len {
                        sum += grad_scores[qi * seq_len + kp] * k_data[kp * kv_dim + kv_offset + d];
                    }
                    grad_q_full[qi * q_dim + q_offset + d] += sum;
                }
            }

            // grad_K_h = grad_scores^T @ Q_h
            for kp in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for qi in 0..seq_len {
                        sum += grad_scores[qi * seq_len + kp] * q_data[qi * q_dim + q_offset + d];
                    }
                    grad_k_full[kp * kv_dim + kv_offset + d] += sum;
                }
            }
        }

        // Backward through Q, K, V projections
        // wq: [q_dim, dim], grad_q: [seq, q_dim]
        // wq_grad = grad_q^T @ input -> [q_dim, dim]
        // grad_input_q = grad_q @ wq -> [seq, dim]
        let mut wq_grad_data = vec![0.0f32; q_dim * self.dim];
        let mut wk_grad_data = vec![0.0f32; kv_dim * self.dim];
        let mut wv_grad_data = vec![0.0f32; kv_dim * self.dim];
        let mut grad_input = vec![0.0f32; seq_len * self.dim];

        // wq_grad and grad_input_q
        for r in 0..q_dim {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for i in 0..seq_len {
                    sum += grad_q_full[i * q_dim + r] * input_data[i * self.dim + c];
                }
                wq_grad_data[r * self.dim + c] = sum;
            }
        }
        for i in 0..seq_len {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for r in 0..q_dim {
                    sum += grad_q_full[i * q_dim + r] * wq_data[r * self.dim + c];
                }
                grad_input[i * self.dim + c] += sum;
            }
        }

        // wk_grad and grad_input_k
        for r in 0..kv_dim {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for i in 0..seq_len {
                    sum += grad_k_full[i * kv_dim + r] * input_data[i * self.dim + c];
                }
                wk_grad_data[r * self.dim + c] = sum;
            }
        }
        for i in 0..seq_len {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for r in 0..kv_dim {
                    sum += grad_k_full[i * kv_dim + r] * wk_data[r * self.dim + c];
                }
                grad_input[i * self.dim + c] += sum;
            }
        }

        // wv_grad and grad_input_v
        for r in 0..kv_dim {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for i in 0..seq_len {
                    sum += grad_v_full[i * kv_dim + r] * input_data[i * self.dim + c];
                }
                wv_grad_data[r * self.dim + c] = sum;
            }
        }
        for i in 0..seq_len {
            for c in 0..self.dim {
                let mut sum = 0.0f32;
                for r in 0..kv_dim {
                    sum += grad_v_full[i * kv_dim + r] * wv_data[r * self.dim + c];
                }
                grad_input[i * self.dim + c] += sum;
            }
        }

        self.wq_grad = Some(GpuTensor::from_slice(
            device,
            &wq_grad_data,
            &[q_dim, self.dim],
        ));
        self.wk_grad = Some(GpuTensor::from_slice(
            device,
            &wk_grad_data,
            &[kv_dim, self.dim],
        ));
        self.wv_grad = Some(GpuTensor::from_slice(
            device,
            &wv_grad_data,
            &[kv_dim, self.dim],
        ));

        GpuTensor::from_slice(device, &grad_input, &[seq_len, self.dim])
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.wq, &self.wk, &self.wv, &self.wo]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![
            self.wq_grad.as_ref(),
            self.wk_grad.as_ref(),
            self.wv_grad.as_ref(),
            self.wo_grad.as_ref(),
        ]
    }

    fn zero_grad(&mut self) {
        self.wq_grad = None;
        self.wk_grad = None;
        self.wv_grad = None;
        self.wo_grad = None;
        self.cached_input = None;
        self.cached_q = None;
        self.cached_k = None;
        self.cached_v = None;
        self.cached_attn_out = None;
    }
}

// ---------------------------------------------------------------------------
// GpuTrainTransformerBlock
// ---------------------------------------------------------------------------

/// A trainable transformer block: ln1 → attn → residual → ln2 → ffn → residual.
pub struct GpuTrainTransformerBlock {
    pub ln1: GpuRMSNorm,
    pub attn: GpuCausalAttention,
    pub ln2: GpuRMSNorm,
    pub ffn: GpuSwiGLU,
    // Cached residuals for backward
    cached_residual1: Option<GpuTensor>,
    cached_ln1_out: Option<GpuTensor>,
    cached_ln2_input: Option<GpuTensor>,
}

impl GpuTrainTransformerBlock {
    pub fn new(ln1: GpuRMSNorm, attn: GpuCausalAttention, ln2: GpuRMSNorm, ffn: GpuSwiGLU) -> Self {
        Self {
            ln1,
            attn,
            ln2,
            ffn,
            cached_residual1: None,
            cached_ln1_out: None,
            cached_ln2_input: None,
        }
    }
}

impl GpuTrainModule for GpuTrainTransformerBlock {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Cache original input for residual backward
        self.cached_residual1 = Some(input.clone_gpu_batched(device, cache));

        // ln1 → attn
        let ln1_out = self.ln1.forward_train(device, cache, input);
        self.cached_ln1_out = Some(ln1_out.clone_gpu_batched(device, cache));
        let attn_out = self.attn.forward_train(device, cache, &ln1_out);

        // Residual add: x + attn(ln1(x))
        let residual1 = add_tensors(device, cache, input, &attn_out);
        self.cached_ln2_input = Some(residual1.clone_gpu_batched(device, cache));

        // ln2 → ffn
        let ln2_out = self.ln2.forward_train(device, cache, &residual1);
        let ffn_out = self.ffn.forward_train(device, cache, &ln2_out);

        // Residual add: residual1 + ffn(ln2(residual1))
        add_tensors(device, cache, &residual1, &ffn_out)
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        // Backward through second residual: grad flows to both branches
        let grad_ffn_out = grad_output;
        let grad_residual1_from_add = grad_output;

        // FFN backward
        let grad_ln2_out = self.ffn.backward(device, cache, grad_ffn_out);

        // LN2 backward
        let grad_residual1_from_ln2 = self.ln2.backward(device, cache, &grad_ln2_out);

        // Sum gradients at residual1
        let grad_residual1 = add_tensors(
            device,
            cache,
            grad_residual1_from_add,
            &grad_residual1_from_ln2,
        );

        // Backward through first residual
        let grad_attn_out = &grad_residual1;
        let grad_input_from_add = &grad_residual1;

        // Attention backward
        let grad_ln1_out = self.attn.backward(device, cache, grad_attn_out);

        // LN1 backward
        let grad_input_from_ln1 = self.ln1.backward(device, cache, &grad_ln1_out);

        // Sum gradients at input
        add_tensors(device, cache, grad_input_from_add, &grad_input_from_ln1)
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ffn.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters_mut());
        params.extend(self.attn.parameters_mut());
        params.extend(self.ln2.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        let mut grads = Vec::new();
        grads.extend(self.ln1.gradients());
        grads.extend(self.attn.gradients());
        grads.extend(self.ln2.gradients());
        grads.extend(self.ffn.gradients());
        grads
    }

    fn zero_grad(&mut self) {
        self.ln1.zero_grad();
        self.attn.zero_grad();
        self.ln2.zero_grad();
        self.ffn.zero_grad();
        self.cached_residual1 = None;
        self.cached_ln1_out = None;
        self.cached_ln2_input = None;
    }
}

// ---------------------------------------------------------------------------
// GpuTrainTransformer — full model
// ---------------------------------------------------------------------------

/// A trainable transformer: embedding → blocks → ln_final → lm_head.
pub struct GpuTrainTransformer {
    pub embed: GpuEmbedding,
    pub blocks: Vec<GpuTrainTransformerBlock>,
    pub ln_final: GpuRMSNorm,
    pub lm_head: crate::module::GpuLinear,
}

impl GpuTrainModule for GpuTrainTransformer {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Embedding: input [seq_len] of token IDs -> [seq_len, dim]
        let mut x = self.embed.forward_train(device, cache, input);

        // Transformer blocks
        for block in &mut self.blocks {
            x = block.forward_train(device, cache, &x);
        }

        // Final norm
        x = self.ln_final.forward_train(device, cache, &x);

        // LM head
        self.lm_head.forward_train(device, cache, &x)
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        // LM head backward
        let mut grad = self.lm_head.backward(device, cache, grad_output);

        // Final norm backward
        grad = self.ln_final.backward(device, cache, &grad);

        // Blocks in reverse
        for block in self.blocks.iter_mut().rev() {
            grad = block.backward(device, cache, &grad);
        }

        // Embedding backward (returns zeros — indices have no gradient)
        self.embed.backward(device, cache, &grad)
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        let mut params = Vec::new();
        params.extend(self.embed.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_final.parameters());
        params.extend(GpuTrainModule::parameters(&self.lm_head));
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        let mut params = Vec::new();
        params.extend(self.embed.parameters_mut());
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.ln_final.parameters_mut());
        params.extend(GpuTrainModule::parameters_mut(&mut self.lm_head));
        params
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        let mut grads = Vec::new();
        grads.extend(self.embed.gradients());
        for block in &self.blocks {
            grads.extend(block.gradients());
        }
        grads.extend(self.ln_final.gradients());
        grads.extend(GpuTrainModule::gradients(&self.lm_head));
        grads
    }

    fn zero_grad(&mut self) {
        self.embed.zero_grad();
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.ln_final.zero_grad();
        self.lm_head.zero_grad();
    }
}

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
    fn embedding_lookup() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // vocab=3, dim=4. Embeddings:
        // 0 -> [1, 0, 0, 0]
        // 1 -> [0, 1, 0, 0]
        // 2 -> [0, 0, 1, 0]
        let weight = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];
        let emb = GpuEmbedding::new(&device, &weight, 3, 4);

        // Look up tokens [2, 0, 1]
        let token_ids: Vec<u32> = vec![2, 0, 1];
        use wgpu::util::DeviceExt;
        let id_buf = GpuBuffer {
            buffer: device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("token ids"),
                    contents: bytemuck::cast_slice(&token_ids),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                }),
            len: token_ids.len(),
        };

        let out = emb.forward(&device, &mut cache, &id_buf, 3);
        let result = out.to_vec_sync(&device);

        assert_eq!(out.shape(), &[3, 4]);
        // Token 2 -> [0, 0, 1, 0]
        assert_eq!(&result[0..4], &[0.0, 0.0, 1.0, 0.0]);
        // Token 0 -> [1, 0, 0, 0]
        assert_eq!(&result[4..8], &[1.0, 0.0, 0.0, 0.0]);
        // Token 1 -> [0, 1, 0, 0]
        assert_eq!(&result[8..12], &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rmsnorm_unit_weight() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let rms = GpuRMSNorm::new(&device, 4, 1e-6);
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = rms.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        // RMS = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps)
        // Expected: x / RMS
        let rms_val = (7.5f32 + 1e-6).sqrt();
        for (i, &val) in result.iter().enumerate() {
            let expected = (i + 1) as f32 / rms_val;
            assert!(
                (val - expected).abs() < 1e-4,
                "rmsnorm[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn rmsnorm_batched() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let rms = GpuRMSNorm::new(&device, 3, 1e-6);
        // 2 rows of 3 elements
        let input = GpuTensor::from_slice(&device, &[3.0, 4.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]);
        let output = rms.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        // Row 0: RMS = sqrt((9+16+0)/3 + eps)
        let rms0 = (25.0f32 / 3.0 + 1e-6).sqrt();
        assert!((result[0] - 3.0 / rms0).abs() < 1e-4);
        assert!((result[1] - 4.0 / rms0).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);

        // Row 1: RMS = sqrt((1+1+1)/3 + eps) = sqrt(1 + eps) ~= 1
        let rms1 = (1.0f32 + 1e-6).sqrt();
        for i in 3..6 {
            assert!(
                (result[i] - 1.0 / rms1).abs() < 1e-4,
                "rmsnorm[{i}] = {}, expected {}",
                result[i],
                1.0 / rms1
            );
        }
    }

    #[test]
    fn swiglu_fused_activation() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // Test the fused SiLU(gate) * up kernel directly
        let gate = GpuTensor::from_slice(&device, &[0.0, 1.0, -1.0, 2.0], &[4]);
        let up = GpuTensor::from_slice(&device, &[1.0, 1.0, 1.0, 1.0], &[4]);
        let out = swiglu_fused(&device, &mut cache, &gate, &up);
        let result = out.to_vec_sync(&device);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let expected: Vec<f32> = [0.0_f32, 1.0, -1.0, 2.0]
            .iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "swiglu[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn swiglu_layer_identity_like() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // With zero weights, output should be zero
        let ffn = GpuSwiGLU::zeros(&device, 4, 8);
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = ffn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-6, "swiglu zero output[{i}] = {val}");
        }
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // At position 0, all angles are 0, so cos=1, sin=0 -> no rotation
        let rope = GpuRoPE::new(&device, 4, 32, 10000.0);
        let input = GpuTensor::from_slice(
            &device,
            &[1.0, 2.0, 3.0, 4.0], // 1 seq, 1 head, dim=4
            &[1, 1, 4],
        );
        let output = rope.forward(&device, &mut cache, &input, 0);
        let result = output.to_vec_sync(&device);

        // At pos 0: cos=1, sin=0 for all freqs
        // output[0] = x[0]*1 - x[2]*0 = x[0]
        // output[2] = x[0]*0 + x[2]*1 = x[2]
        assert!((result[0] - 1.0).abs() < 1e-5, "rope[0] = {}", result[0]);
        assert!((result[1] - 2.0).abs() < 1e-5, "rope[1] = {}", result[1]);
        assert!((result[2] - 3.0).abs() < 1e-5, "rope[2] = {}", result[2]);
        assert!((result[3] - 4.0).abs() < 1e-5, "rope[3] = {}", result[3]);
    }

    #[test]
    fn rope_rotation_at_pos_nonzero() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let head_dim = 4;
        let base = 10000.0f32;
        let rope = GpuRoPE::new(&device, head_dim, 32, base);

        // Input: [1, 0, 0, 0] at position 5
        let input = GpuTensor::from_slice(&device, &[1.0, 0.0, 0.0, 0.0], &[1, 1, 4]);
        let output = rope.forward(&device, &mut cache, &input, 5);
        let result = output.to_vec_sync(&device);

        // At pos 5, pair 0: angle = 5 * 1/10000^(0/4) = 5.0
        let angle0 = 5.0f32 * 1.0 / base.powf(0.0 / 4.0);
        // x[0]=1, x[2]=0: out[0] = 1*cos(a) - 0*sin(a) = cos(a)
        //                   out[2] = 1*sin(a) + 0*cos(a) = sin(a)
        assert!(
            (result[0] - angle0.cos()).abs() < 1e-4,
            "rope pos5[0] = {}, expected {}",
            result[0],
            angle0.cos()
        );
        assert!(
            (result[2] - angle0.sin()).abs() < 1e-4,
            "rope pos5[2] = {}, expected {}",
            result[2],
            angle0.sin()
        );
    }

    #[test]
    fn rope_high_base() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // Test with 500K base (long context)
        let rope = GpuRoPE::new(&device, 4, 64, 500000.0);
        let input = GpuTensor::from_slice(&device, &[1.0, 1.0, 1.0, 1.0], &[1, 1, 4]);
        let output = rope.forward(&device, &mut cache, &input, 10);
        let result = output.to_vec_sync(&device);

        // Just verify it produces valid (non-NaN) output
        for (i, &val) in result.iter().enumerate() {
            assert!(!val.is_nan(), "rope 500K base output[{i}] is NaN");
            assert!(val.is_finite(), "rope 500K base output[{i}] is infinite");
        }
    }

    #[test]
    fn kv_cache_append_and_retrieve() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let mut kv = GpuKVCache::new(&device, 2, 4, 32); // 2 heads, dim 4, max 32

        // Append 2 positions
        let k1 = GpuTensor::from_slice(
            &device,
            &[
                1.0, 0.0, 0.0, 0.0, // head 0
                0.0, 1.0, 0.0, 0.0, // head 1
                0.0, 0.0, 1.0, 0.0, // head 0
                0.0, 0.0, 0.0, 1.0, // head 1
            ],
            &[2, 2, 4],
        );
        let v1 = GpuTensor::from_slice(
            &device,
            &[
                1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
            ],
            &[2, 2, 4],
        );

        kv.append(&device, &mut cache, &k1, &v1);
        assert_eq!(kv.len, 2);

        // Append 1 more position
        let k2 = GpuTensor::from_slice(
            &device,
            &[5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
            &[1, 2, 4],
        );
        let v2 = GpuTensor::from_slice(
            &device,
            &[7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0],
            &[1, 2, 4],
        );

        kv.append(&device, &mut cache, &k2, &v2);
        assert_eq!(kv.len, 3);

        // Retrieve and verify
        cache.flush(&device);
        let keys = kv.get_keys(&device);
        let key_data = keys.to_vec_sync(&device);
        assert_eq!(keys.shape(), &[3, 2, 4]);
        assert_eq!(&key_data[0..4], &[1.0, 0.0, 0.0, 0.0]); // pos 0, head 0
        assert_eq!(&key_data[16..20], &[5.0, 5.0, 5.0, 5.0]); // pos 2, head 0

        let values = kv.get_values(&device);
        let val_data = values.to_vec_sync(&device);
        assert_eq!(values.shape(), &[3, 2, 4]);
        assert_eq!(&val_data[16..20], &[7.0, 7.0, 7.0, 7.0]); // pos 2, head 0
    }

    #[test]
    fn kv_cache_clear() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let mut kv = GpuKVCache::new(&device, 1, 2, 16);
        let k = GpuTensor::from_slice(&device, &[1.0, 2.0], &[1, 1, 2]);
        let v = GpuTensor::from_slice(&device, &[3.0, 4.0], &[1, 1, 2]);
        kv.append(&device, &mut cache, &k, &v);
        assert_eq!(kv.len, 1);

        kv.clear();
        assert_eq!(kv.len, 0);
    }

    #[test]
    fn causal_attention_zeros() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // With zero weights, output should be zero
        let attn = GpuCausalAttention::zeros(&device, 8, 2, 2, 4);
        let input = GpuTensor::from_slice(
            &device,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            &[2, 8],
        );
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[2, 8]);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val.abs() < 1e-4 || val.is_nan(),
                "causal_attn zeros output[{i}] = {val}"
            );
        }
    }

    #[test]
    fn causal_attention_identity_kv() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // seq_len=1 with identity-like weights to verify the pipeline works
        let dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 4;

        // Set Wq, Wk, Wv to identity, Wo to identity
        let eye4 = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let attn = GpuCausalAttention {
            wq: GpuTensor::from_slice(&device, &eye4, &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(&device, &eye4, &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn_out: None,
            wq_grad: None,
            wk_grad: None,
            wv_grad: None,
            wo_grad: None,
        };

        // Single token: self-attention on 1 position just returns V (after softmax weight = 1.0)
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[1, 4]);
        // With identity projections and single token: output = V = input
        for (i, (&got, &expected)) in result.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.1,
                "causal_attn identity[{i}] = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn causal_attention_mask_works() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 4;

        let eye4 = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let attn = GpuCausalAttention {
            wq: GpuTensor::from_slice(&device, &eye4, &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(&device, &eye4, &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn_out: None,
            wq_grad: None,
            wk_grad: None,
            wv_grad: None,
            wo_grad: None,
        };

        // Two tokens: position 0 can only see itself, position 1 sees both
        let input = GpuTensor::from_slice(
            &device,
            &[
                1.0, 0.0, 0.0, 0.0, // token 0
                0.0, 0.0, 0.0, 1.0, // token 1
            ],
            &[2, 4],
        );
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[2, 4]);

        // Position 0: only attends to itself -> output ~= V[0] = [1, 0, 0, 0]
        assert!(
            (result[0] - 1.0).abs() < 0.1,
            "pos0[0] = {}, expected ~1.0",
            result[0]
        );

        // Position 1: attends to both -> weighted average of V[0] and V[1]
        // The exact values depend on the attention weights, but it should be
        // some mix of [1,0,0,0] and [0,0,0,1]
        let row1_sum: f32 = result[4..8].iter().sum();
        assert!(
            (row1_sum - 1.0).abs() < 0.2,
            "pos1 sum = {row1_sum}, expected ~1.0"
        );
    }

    #[test]
    fn causal_attention_gqa() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        // 4 query heads, 2 KV heads (GQA ratio 2:1)
        let dim = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 2;

        let attn = GpuCausalAttention::zeros(&device, dim, n_heads, n_kv_heads, head_dim);
        let input =
            GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 8]);

        // Should not panic -- just verifying GQA dispatch works
        let output = attn.forward(&device, &mut cache, &input);
        assert_eq!(output.shape(), &[1, 8]);
    }

    // -----------------------------------------------------------------------
    // Backward pass tests
    // -----------------------------------------------------------------------

    /// CPU-only RMSNorm for gradient checking: returns sum of outputs.
    fn cpu_rmsnorm_sum(inp: &[f32], weight: &[f32], dim: usize, eps: f32) -> f32 {
        let n_groups = inp.len() / dim;
        let mut out_sum = 0.0f32;
        for g in 0..n_groups {
            let base = g * dim;
            let mut sq_sum = 0.0f32;
            for i in 0..dim {
                sq_sum += inp[base + i] * inp[base + i];
            }
            let r = (sq_sum / dim as f32 + eps).sqrt();
            for i in 0..dim {
                out_sum += inp[base + i] / r * weight[i];
            }
        }
        out_sum
    }

    /// CPU-only finite-difference gradient (pure CPU, no GPU state).
    fn finite_diff_cpu(input: &[f32], eps: f32, f: &dyn Fn(&[f32]) -> f32) -> Vec<f32> {
        let mut grad = vec![0.0f32; input.len()];
        for i in 0..input.len() {
            let mut plus = input.to_vec();
            plus[i] += eps;
            let mut minus = input.to_vec();
            minus[i] -= eps;
            grad[i] = (f(&plus) - f(&minus)) / (2.0 * eps);
        }
        grad
    }

    #[test]
    fn rmsnorm_backward_gradient_check() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let dim = 4;
        let norm_eps = 1e-6;
        let weight_data = vec![1.5, 0.5, 2.0, 1.0];
        let mut rms = GpuRMSNorm {
            weight: GpuTensor::from_slice(&device, &weight_data, &[dim]),
            eps: norm_eps,
            dim,
            cached_input: None,
            cached_rms: None,
            weight_grad: None,
        };

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, -0.5];
        let input = GpuTensor::from_slice(&device, &input_data, &[2, 4]);

        let _output = rms.forward_train(&device, &mut cache, &input);
        let grad_out = GpuTensor::from_slice(&device, &[1.0f32; 8], &[2, 4]);
        let grad_input = rms.backward(&device, &mut cache, &grad_out);

        cache.flush(&device);
        let analytical = grad_input.to_vec_sync(&device);

        let w = weight_data.clone();
        let numerical = finite_diff_cpu(&input_data, 1e-3, &|inp| {
            cpu_rmsnorm_sum(inp, &w, dim, norm_eps)
        });

        for i in 0..8 {
            assert!(
                (analytical[i] - numerical[i]).abs() < 0.01,
                "rmsnorm grad[{i}]: analytical={}, numerical={}",
                analytical[i],
                numerical[i]
            );
        }
    }

    #[test]
    fn rope_backward_transposed_rotation() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let head_dim = 4;
        let rope = GpuInterleavedRoPE::new(&device, head_dim, 32, 10000.0);

        // Forward at position 3
        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 4]
        let input = GpuTensor::from_slice(&device, &input_data, &[1, 1, 4]);
        let output = rope.forward(&device, &mut cache, &input, 3);
        let _fwd = output.to_vec_sync(&device);

        // Backward with grad_output = [1, 0, 0, 0] at position 3
        let grad_out = GpuTensor::from_slice(&device, &[1.0, 0.0, 0.0, 0.0], &[1, 1, 4]);
        let grad_in = interleaved_rope_backward(&rope, &device, &mut cache, &grad_out, 3);
        let gi = grad_in.to_vec_sync(&device);

        // Numerical check: perturb input, measure change in output[0]
        let h = 1e-4;
        for dim_i in 0..4 {
            let mut plus = input_data.clone();
            plus[dim_i] += h;
            let mut minus = input_data.clone();
            minus[dim_i] -= h;

            let out_p = rope.forward(
                &device,
                &mut cache,
                &GpuTensor::from_slice(&device, &plus, &[1, 1, 4]),
                3,
            );
            let out_m = rope.forward(
                &device,
                &mut cache,
                &GpuTensor::from_slice(&device, &minus, &[1, 1, 4]),
                3,
            );
            let p = out_p.to_vec_sync(&device);
            let m = out_m.to_vec_sync(&device);
            let numerical = (p[0] - m[0]) / (2.0 * h);

            assert!(
                (gi[dim_i] - numerical).abs() < 1e-3,
                "rope grad[{dim_i}]: analytical={}, numerical={}",
                gi[dim_i],
                numerical
            );
        }
    }

    #[test]
    fn embedding_backward_scatter_add() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let weight_data = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];
        let mut emb = GpuEmbedding::new(&device, &weight_data, 3, 2);

        // Token IDs [1, 0, 2] as f32 (forward_train expects f32 input)
        let input = GpuTensor::from_slice(&device, &[1.0, 0.0, 2.0], &[3]);
        let output = emb.forward_train(&device, &mut cache, &input);

        cache.flush(&device);
        let out_data = output.to_vec_sync(&device);
        // Check forward: token 1 -> [3,4], token 0 -> [1,2], token 2 -> [5,6]
        assert!((out_data[0] - 3.0).abs() < 1e-5);
        assert!((out_data[1] - 4.0).abs() < 1e-5);
        assert!((out_data[2] - 1.0).abs() < 1e-5);
        assert!((out_data[3] - 2.0).abs() < 1e-5);
        assert!((out_data[4] - 5.0).abs() < 1e-5);
        assert!((out_data[5] - 6.0).abs() < 1e-5);

        // Backward with grad_output = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        let grad_out = GpuTensor::from_slice(&device, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]);
        let _ = emb.backward(&device, &mut cache, &grad_out);

        let wg = emb.weight_grad.as_ref().unwrap().to_vec_sync(&device);
        // token 0 appears at position 1: grad_weight[0] = [0.3, 0.4]
        assert!((wg[0] - 0.3).abs() < 1e-5, "wg[0]={}", wg[0]);
        assert!((wg[1] - 0.4).abs() < 1e-5, "wg[1]={}", wg[1]);
        // token 1 appears at position 0: grad_weight[1] = [0.1, 0.2]
        assert!((wg[2] - 0.1).abs() < 1e-5, "wg[2]={}", wg[2]);
        assert!((wg[3] - 0.2).abs() < 1e-5, "wg[3]={}", wg[3]);
        // token 2 appears at position 2: grad_weight[2] = [0.5, 0.6]
        assert!((wg[4] - 0.5).abs() < 1e-5, "wg[4]={}", wg[4]);
        assert!((wg[5] - 0.6).abs() < 1e-5, "wg[5]={}", wg[5]);
    }

    /// CPU-only SwiGLU forward sum for gradient checking.
    fn cpu_swiglu_sum(
        inp: &[f32],
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
        seq_len: usize,
        dim: usize,
        hidden_dim: usize,
    ) -> f32 {
        // gate = inp @ gate_w^T -> [seq, hidden]
        let mut gate = vec![0.0f32; seq_len * hidden_dim];
        for s in 0..seq_len {
            for h in 0..hidden_dim {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for d in 0..dim {
                    g += inp[s * dim + d] * gate_w[h * dim + d];
                    u += inp[s * dim + d] * up_w[h * dim + d];
                }
                let silu_g = g / (1.0 + (-g).exp());
                gate[s * hidden_dim + h] = silu_g * u;
            }
        }
        // result = activated @ down_w^T -> [seq, dim]
        let mut out_sum = 0.0f32;
        for s in 0..seq_len {
            for d in 0..dim {
                let mut val = 0.0f32;
                for h in 0..hidden_dim {
                    val += gate[s * hidden_dim + h] * down_w[d * hidden_dim + h];
                }
                out_sum += val;
            }
        }
        out_sum
    }

    #[test]
    fn swiglu_backward_gradient_check() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let dim = 3;
        let hidden_dim = 4;

        let gate_w: Vec<f32> = (0..hidden_dim * dim)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5)
            .collect();
        let up_w: Vec<f32> = (0..hidden_dim * dim)
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let down_w: Vec<f32> = (0..dim * hidden_dim)
            .map(|i| ((i * 13 + 7) % 11) as f32 * 0.1 - 0.5)
            .collect();

        let mut ffn = GpuSwiGLU::new(&device, &gate_w, &up_w, &down_w, dim, hidden_dim);

        let input_data = vec![0.5, -0.3, 0.8, 1.0, 0.2, -0.5];
        let input = GpuTensor::from_slice(&device, &input_data, &[2, dim]);

        let _output = ffn.forward_train(&device, &mut cache, &input);
        let grad_out = GpuTensor::from_slice(&device, &vec![1.0f32; 2 * dim], &[2, dim]);
        let grad_input = ffn.backward(&device, &mut cache, &grad_out);

        cache.flush(&device);
        let analytical = grad_input.to_vec_sync(&device);

        let gw = gate_w.clone();
        let uw = up_w.clone();
        let dw = down_w.clone();
        let numerical = finite_diff_cpu(&input_data, 1e-3, &|inp| {
            cpu_swiglu_sum(inp, &gw, &uw, &dw, 2, dim, hidden_dim)
        });

        for i in 0..6 {
            assert!(
                (analytical[i] - numerical[i]).abs() < 0.05,
                "swiglu grad[{i}]: analytical={}, numerical={}",
                analytical[i],
                numerical[i]
            );
        }
    }

    /// CPU-only causal attention forward sum for gradient checking.
    fn cpu_causal_attn_sum(
        inp: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        seq_len: usize,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> f32 {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q = inp @ wq^T
        let mut q = vec![0.0f32; seq_len * q_dim];
        let mut k = vec![0.0f32; seq_len * kv_dim];
        let mut v = vec![0.0f32; seq_len * kv_dim];
        for s in 0..seq_len {
            for j in 0..q_dim {
                let mut sum = 0.0f32;
                for d in 0..dim {
                    sum += inp[s * dim + d] * wq[j * dim + d];
                }
                q[s * q_dim + j] = sum;
            }
            for j in 0..kv_dim {
                let mut sk = 0.0f32;
                let mut sv = 0.0f32;
                for d in 0..dim {
                    sk += inp[s * dim + d] * wk[j * dim + d];
                    sv += inp[s * dim + d] * wv[j * dim + d];
                }
                k[s * kv_dim + j] = sk;
                v[s * kv_dim + j] = sv;
            }
        }

        // Per-head causal attention
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        for h in 0..n_heads {
            let q_off = h * head_dim;
            let kv_g = h / heads_per_kv;
            let kv_off = kv_g * head_dim;
            for qi in 0..seq_len {
                // Compute scores + causal softmax
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                let mut row_max = f32::NEG_INFINITY;
                for kp in 0..=qi {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qi * q_dim + q_off + d] * k[kp * kv_dim + kv_off + d];
                    }
                    scores[kp] = dot * scale;
                    row_max = row_max.max(scores[kp]);
                }
                let mut exp_sum = 0.0f32;
                for kp in 0..=qi {
                    scores[kp] = (scores[kp] - row_max).exp();
                    exp_sum += scores[kp];
                }
                for kp in 0..=qi {
                    scores[kp] /= exp_sum;
                }
                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for kp in 0..=qi {
                        acc += scores[kp] * v[kp * kv_dim + kv_off + d];
                    }
                    attn_out[qi * q_dim + q_off + d] = acc;
                }
            }
        }

        // Output projection + sum
        let mut out_sum = 0.0f32;
        for s in 0..seq_len {
            for d in 0..dim {
                let mut val = 0.0f32;
                for j in 0..q_dim {
                    val += attn_out[s * q_dim + j] * wo[d * q_dim + j];
                }
                out_sum += val;
            }
        }
        out_sum
    }

    #[test]
    fn attention_backward_gradient_check() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 2;

        let make_weights = |seed: usize, rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i * seed + 3) % 17) as f32 * 0.1 - 0.8)
                .collect()
        };

        let wq_data = make_weights(7, n_heads * head_dim, dim);
        let wk_data = make_weights(11, n_kv_heads * head_dim, dim);
        let wv_data = make_weights(13, n_kv_heads * head_dim, dim);
        let wo_data = make_weights(17, dim, n_heads * head_dim);

        let mut attn = GpuCausalAttention {
            wq: GpuTensor::from_slice(&device, &wq_data, &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(&device, &wk_data, &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(&device, &wv_data, &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(&device, &wo_data, &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn_out: None,
            wq_grad: None,
            wk_grad: None,
            wv_grad: None,
            wo_grad: None,
        };

        let input_data = vec![0.5, -0.3, 0.8, 0.2, 1.0, 0.1, -0.5, 0.7];
        let input = GpuTensor::from_slice(&device, &input_data, &[2, dim]);

        let _output = attn.forward_train(&device, &mut cache, &input);
        let grad_out = GpuTensor::from_slice(&device, &vec![1.0f32; 2 * dim], &[2, dim]);
        let grad_input = attn.backward(&device, &mut cache, &grad_out);

        cache.flush(&device);
        let analytical = grad_input.to_vec_sync(&device);

        let wq = wq_data.clone();
        let wk = wk_data.clone();
        let wv = wv_data.clone();
        let wo = wo_data.clone();
        let numerical = finite_diff_cpu(&input_data, 1e-3, &|inp| {
            cpu_causal_attn_sum(
                inp, &wq, &wk, &wv, &wo, 2, dim, n_heads, n_kv_heads, head_dim,
            )
        });

        for i in 0..8 {
            assert!(
                (analytical[i] - numerical[i]).abs() < 0.05,
                "attn grad[{i}]: analytical={}, numerical={}",
                analytical[i],
                numerical[i]
            );
        }
    }

    #[test]
    fn transformer_block_backward_shapes() {
        let Some(device) = get_device() else { return };
        let mut cache = KernelCache::new();

        let dim = 4;
        let hidden_dim = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 2;

        let ln1 = GpuRMSNorm::new(&device, dim, 1e-6);
        let attn = GpuCausalAttention::zeros(&device, dim, n_heads, n_kv_heads, head_dim);
        let ln2 = GpuRMSNorm::new(&device, dim, 1e-6);
        let ffn = GpuSwiGLU::zeros(&device, dim, hidden_dim);

        let mut block = GpuTrainTransformerBlock::new(ln1, attn, ln2, ffn);

        // Forward
        let input =
            GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let output = block.forward_train(&device, &mut cache, &input);
        assert_eq!(output.shape(), &[2, 4]);

        // Backward
        let grad_out = GpuTensor::from_slice(&device, &[1.0f32; 8], &[2, 4]);
        let grad_input = block.backward(&device, &mut cache, &grad_out);
        assert_eq!(grad_input.shape(), &[2, 4]);

        // All parameters should have gradients
        let grads = block.gradients();
        for (i, g) in grads.iter().enumerate() {
            assert!(g.is_some(), "gradient {i} is None");
        }
    }
}
