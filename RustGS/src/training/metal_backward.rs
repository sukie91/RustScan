use candle_core::{DType, Device, Tensor};

use crate::diff::diff_splat::DiffCamera;

use super::metal_runtime::{MetalRuntime, MetalTileBins};

pub(crate) struct MetalBackwardGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
    pub rotations: Tensor,
    pub opacity_logits: Tensor,
    pub colors: Tensor,
}

pub(crate) struct MetalBackwardPass {
    pub grads: MetalBackwardGrads,
    /// Gradient magnitudes per Gaussian (tiny readback for densification stats).
    pub grad_magnitudes: Vec<f32>,
}

/// GPU-accelerated backward pass using Metal compute kernel.
/// All computation runs on GPU; only per-Gaussian gradient magnitudes
/// are read back for densification decisions.
pub(crate) fn backward_weighted_l1(
    runtime: &mut MetalRuntime,
    device: &Device,
    visible_count: usize,
    tile_bins: &MetalTileBins,
    target_color: &[f32],
    target_depth: &[f32],
    n_gaussians: usize,
    camera: &DiffCamera,
) -> candle_core::Result<MetalBackwardPass> {
    let color_scale = 1.0 / target_color.len().max(1) as f32;
    let depth_scale = 0.1 / target_depth.len().max(1) as f32;

    // Upload targets to GPU
    runtime.write_target_data(target_color, target_depth, color_scale, depth_scale)?;

    // Dispatch Metal backward kernel
    let (frame, _profile) =
        runtime.rasterize_backward(visible_count, tile_bins, camera.width, camera.height)?;

    let grad_magnitude_tensor = runtime.compute_grad_magnitudes(n_gaussians)?;
    let grad_magnitudes = runtime.read_tensor_flat::<f32>(&grad_magnitude_tensor)?;

    let grads = MetalBackwardGrads {
        positions: frame.grad_positions,
        log_scales: frame.grad_log_scales,
        rotations: Tensor::zeros((n_gaussians, 4), DType::F32, device)?,
        opacity_logits: frame.grad_opacity_logits,
        colors: frame.grad_colors,
    };

    Ok(MetalBackwardPass {
        grads,
        grad_magnitudes,
    })
}
