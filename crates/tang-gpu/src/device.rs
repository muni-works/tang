//! GPU device and queue initialization via wgpu.

use std::fmt;

/// Error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Device request failed.
    DeviceError(wgpu::RequestDeviceError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no suitable GPU adapter found"),
            Self::DeviceError(e) => write!(f, "GPU device error: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceError(e)
    }
}

/// GPU device wrapping wgpu `Device` and `Queue`.
pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Adapter name and backend, e.g. `"Apple M4 Max (Metal)"`.
    pub name: String,
    /// Native MSL kernels can be dispatched (Metal, with wgpu's
    /// shader passthrough).
    pub msl_passthrough: bool,
    /// SPIR-V can be dispatched as-is (Vulkan): the GLSL decode kernels,
    /// when built with the `glsl` feature.
    pub spirv_passthrough: bool,
    /// WGSL kernels may use `f64` (wgpu `SHADER_F64`: Vulkan and DX12 on
    /// most desktop GPUs; not Metal).
    pub shader_f64: bool,
    /// WGSL subgroup operations (wgpu `SUBGROUP`).
    pub subgroups: bool,
    /// WGSL `enable f16;` (wgpu `SHADER_F16`).
    pub shader_f16: bool,
}

impl GpuDevice {
    /// What generated WGSL may use on this device
    /// ([`TensorGraph::to_wgsl_caps`](tang_compile::tensor::TensorGraph::to_wgsl_caps)).
    /// `TANG_NO_SUBGROUPS` / `TANG_NO_F16` turn either off, for A/B runs.
    pub fn wgsl_caps(&self) -> tang_compile::tensor::WgslCaps {
        tang_compile::tensor::WgslCaps {
            subgroups: self.subgroups && std::env::var_os("TANG_NO_SUBGROUPS").is_none(),
            f16: self.shader_f16 && std::env::var_os("TANG_NO_F16").is_none(),
        }
    }
}

impl GpuDevice {
    /// Create a new GPU device with default settings.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        // Generated kernels bind one storage buffer per tensor they read;
        // a fused block kernel can need more than the default 8. Ask for
        // what the adapter has, within reason.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.min(16);
        // whole weight matrices are single buffers (an LM head's 4-bit
        // nibbles are 311 MB): as large as the adapter allows
        limits.max_buffer_size = adapter_limits.max_buffer_size.max(limits.max_buffer_size);
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max(limits.max_storage_buffer_binding_size);
        limits.max_compute_workgroup_storage_size = adapter_limits
            .max_compute_workgroup_storage_size
            .max(limits.max_compute_workgroup_storage_size);

        // Native shader passthrough (MSL on Metal) lets generated kernels
        // use what WGSL cannot express — simdgroup matrices.
        // f64 for simulation kernels that f32 cannot carry (a stiff contact
        // solve drifts off the f64 trajectory within a few steps).
        let required_features = adapter.features()
            & (wgpu::Features::MSL_SHADER_PASSTHROUGH
                | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                | wgpu::Features::SHADER_F64
                | wgpu::Features::SUBGROUP
                | wgpu::Features::SHADER_F16);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tang-gpu"),
                required_features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let msl_passthrough = required_features.contains(wgpu::Features::MSL_SHADER_PASSTHROUGH);
        let spirv_passthrough =
            required_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);
        Ok(Self {
            device,
            queue,
            name,
            msl_passthrough,
            spirv_passthrough,
            shader_f64: required_features.contains(wgpu::Features::SHADER_F64),
            subgroups: required_features.contains(wgpu::Features::SUBGROUP),
            shader_f16: required_features.contains(wgpu::Features::SHADER_F16),
        })
    }

    /// Create a GPU device synchronously (blocks on async).
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }
}
