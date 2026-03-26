use candle_core::Tensor;

use crate::diff::diff_splat::DiffCamera;

use super::metal_runtime::{MetalRuntime, MetalTileBins};

pub(crate) struct MetalBackwardGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
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
    tile_bins: &MetalTileBins,
    n_gaussians: usize,
    camera: &DiffCamera,
) -> candle_core::Result<MetalBackwardPass> {
    let (frame, _profile) =
        runtime.rasterize_backward(n_gaussians, tile_bins, camera.width, camera.height)?;

    let grad_magnitude_tensor = runtime.compute_grad_magnitudes(n_gaussians)?;
    let grad_magnitudes = runtime.read_tensor_flat::<f32>(&grad_magnitude_tensor)?;

    let grads = MetalBackwardGrads {
        positions: frame.grad_positions,
        log_scales: frame.grad_log_scales,
        opacity_logits: frame.grad_opacity_logits,
        colors: frame.grad_colors,
    };

    Ok(MetalBackwardPass {
        grads,
        grad_magnitudes,
    })
}
