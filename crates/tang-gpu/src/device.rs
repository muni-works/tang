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
    /// Compute passes can write GPU timestamps (wgpu `TIMESTAMP_QUERY`):
    /// kernel times from the GPU's clock instead of the host's.
    pub timestamps: bool,
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
                | wgpu::Features::TIMESTAMP_QUERY);
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
            timestamps: required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
        })
    }

    /// Create a GPU device synchronously (blocks on async).
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }
}
