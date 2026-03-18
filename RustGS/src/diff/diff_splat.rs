//! Differentiable splatting.
//!
//! This module will be populated from RustSLAM/src/fusion/diff_splat.rs in Story 9-5.

use candle_core::Device;

/// Differentiable splat renderer.
///
/// Uses Candle with Metal/MPS for GPU-accelerated differentiable rendering.
pub struct DiffSplatRenderer {
    device: Device,
}

impl DiffSplatRenderer {
    /// Create a new renderer with the given device.
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Create with Metal device.
    pub fn new_metal() -> candle_core::Result<Self> {
        let device = Device::new_metal(0)?;
        Ok(Self::new(device))
    }

    /// Create with CPU device (for testing).
    pub fn new_cpu() -> Self {
        Self::new(Device::Cpu)
    }
}