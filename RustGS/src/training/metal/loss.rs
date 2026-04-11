use candle_core::op::BackpropOp;
use candle_core::{CpuStorage, CustomOp2, Layout, MetalStorage, Shape, Storage, Tensor};

use crate::diff::diff_splat::Splats;

use super::backward::{backward_loss_scales, MetalBackwardLossScales};
use super::forward::{ProjectedGaussians, RenderedFrame};
use super::parity_harness::ParityLossTerms;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalLossConfig {
    pub color_weight: f32,
    pub ssim_weight: f32,
    pub depth_weight: f32,
    pub scale_regularization_weight: f32,
    pub enable_transmittance_loss: bool,
    pub render_width: usize,
    pub render_height: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalLossTelemetry {
    pub loss_terms: ParityLossTerms,
    pub depth_valid_pixels: Option<usize>,
    pub depth_grad_scale: Option<f32>,
}

pub(crate) struct MetalStepLossContext {
    pub total_loss: f32,
    pub rendered_color_cpu: Vec<f32>,
    pub ssim_grads: Vec<f32>,
    pub backward_loss_scales: MetalBackwardLossScales,
    pub telemetry: MetalLossTelemetry,
}

/// Compute per-pixel SSIM gradient (dSSIM/d_rendered) for RGB images.
///
/// Uses an 11×11 uniform box-filter window.  Returns `(ssim_mean, grad)` where
/// `grad` has the same layout as `rendered` (H×W×3 interleaved RGB floats).
///
/// C1 = (0.01)², C2 = (0.03)² follow the standard SSIM definition assuming
/// pixel values in [0, 1].
pub(crate) fn ssim_gradient(
    rendered: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
) -> (f32, Vec<f32>) {
    const WINDOW_RADIUS: isize = 5; // 11×11 window
    const C1: f32 = 0.0001; // (0.01)²
    const C2: f32 = 0.0009; // (0.03)²

    let pixel_count = width * height;
    let mut ssim_sum = 0.0f32;
    let mut grad = vec![0.0f32; pixel_count * 3];

    for ch in 0..3usize {
        for py in 0..height {
            for px in 0..width {
                let mut sum_x = 0.0f32;
                let mut sum_y = 0.0f32;
                let mut sum_xx = 0.0f32;
                let mut sum_yy = 0.0f32;
                let mut sum_xy = 0.0f32;
                let mut count = 0usize;
                let cx_val;
                let cy_val;

                for dy in -WINDOW_RADIUS..=WINDOW_RADIUS {
                    let qy = py as isize + dy;
                    if qy < 0 || qy >= height as isize {
                        continue;
                    }
                    for dx in -WINDOW_RADIUS..=WINDOW_RADIUS {
                        let qx = px as isize + dx;
                        if qx < 0 || qx >= width as isize {
                            continue;
                        }
                        let qi = (qy as usize * width + qx as usize) * 3 + ch;
                        let xv = rendered[qi];
                        let yv = target[qi];
                        sum_x += xv;
                        sum_y += yv;
                        sum_xx += xv * xv;
                        sum_yy += yv * yv;
                        sum_xy += xv * yv;
                        count += 1;
                    }
                }

                let cnt = count as f32;
                let mu_x = sum_x / cnt;
                let mu_y = sum_y / cnt;
                let sigma_x2 = (sum_xx / cnt - mu_x * mu_x).max(0.0);
                let sigma_y2 = (sum_yy / cnt - mu_y * mu_y).max(0.0);
                let sigma_xy = sum_xy / cnt - mu_x * mu_y;

                let a = 2.0 * mu_x * mu_y + C1;
                let b = 2.0 * sigma_xy + C2;
                let c = mu_x * mu_x + mu_y * mu_y + C1;
                let d = sigma_x2 + sigma_y2 + C2;
                let cd = c * d;

                let ssim_p = if cd > 1e-10 { (a * b) / cd } else { 1.0 };
                ssim_sum += ssim_p;

                // Current pixel's contribution to gradient:
                //   dSSIM/dx_p = (da/dx_p * b * cd + a * db/dx_p * cd - a*b*(dc/dx_p*d + c*dd/dx_p)) / cd²
                // where:
                //   da/dx_p = 2*mu_y / cnt
                //   db/dx_p = 2*(y_p - mu_y) / cnt
                //   dc/dx_p = 2*mu_x / cnt
                //   dd/dx_p = 2*(x_p - mu_x) / cnt
                if cd > 1e-10 {
                    let pi = (py * width + px) * 3 + ch;
                    cx_val = rendered[pi];
                    cy_val = target[pi];

                    let da = 2.0 * mu_y / cnt;
                    let db = 2.0 * (cy_val - mu_y) / cnt;
                    let dc = 2.0 * mu_x / cnt;
                    let dd = 2.0 * (cx_val - mu_x) / cnt;

                    let numerator = da * b * cd + a * db * cd - a * b * (dc * d + c * dd);
                    let grad_p = numerator / (cd * cd);
                    grad[pi] = grad_p;
                }
            }
        }
    }

    let ssim_mean = ssim_sum / (pixel_count * 3) as f32;
    (ssim_mean, grad)
}

#[derive(Debug, Clone, Copy, Default)]
struct MeanAbsDiff;

pub(crate) fn mean_abs_diff(predicted: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    if predicted.shape() != target.shape() {
        candle_core::bail!(
            "mean_abs_diff shape mismatch: lhs={:?}, rhs={:?}",
            predicted.dims(),
            target.dims()
        );
    }
    predicted
        .contiguous()?
        .apply_op2(&target.contiguous()?, MeanAbsDiff)
}

pub(crate) fn masked_mean_abs_diff(
    predicted: &Tensor,
    target: &Tensor,
    valid_mask: &Tensor,
) -> candle_core::Result<Tensor> {
    if predicted.shape() != target.shape() || predicted.shape() != valid_mask.shape() {
        candle_core::bail!(
            "masked_mean_abs_diff shape mismatch: pred={:?}, target={:?}, mask={:?}",
            predicted.dims(),
            target.dims(),
            valid_mask.dims()
        );
    }

    let dtype = predicted.dtype();
    let mask = valid_mask.gt(0.0f64)?.to_dtype(dtype)?;
    let valid_count = mask.flatten_all()?.sum(0)?;
    if valid_count.to_vec0::<f32>()? <= 0.0 {
        return Tensor::zeros((), dtype, predicted.device());
    }

    let masked_diff = predicted.sub(target)?.abs()?.broadcast_mul(&mask)?;
    let masked_sum = masked_diff.flatten_all()?.sum(0)?;
    masked_sum.broadcast_div(&valid_count)
}

pub(crate) fn evaluate_training_step_loss(
    gaussians: &Splats,
    rendered: &RenderedFrame,
    projected: &ProjectedGaussians,
    target_color: &Tensor,
    target_depth: &Tensor,
    target_color_cpu: &[f32],
    target_depth_cpu: &[f32],
    config: MetalLossConfig,
) -> candle_core::Result<MetalStepLossContext> {
    let color_loss = mean_abs_diff(&rendered.color, target_color)?;
    let depth_loss = masked_mean_abs_diff(&rendered.depth, target_depth, target_depth)?;
    let rendered_color_cpu = rendered.color.flatten_all()?.to_vec1::<f32>()?;
    let (ssim_value, ssim_grads) = ssim_gradient(
        &rendered_color_cpu,
        target_color_cpu,
        config.render_width,
        config.render_height,
    );
    let ssim_loss_term = 1.0 - ssim_value;
    let depth_valid_pixels = if config.depth_weight > 0.0 {
        Some(valid_depth_sample_count(target_depth_cpu))
    } else {
        None
    };
    let depth_grad_scale = if config.depth_weight > 0.0 {
        Some(depth_backward_scale(config.depth_weight, target_depth_cpu))
    } else {
        None
    };
    let scale_reg_term = if config.scale_regularization_weight > 0.0 && projected.visible_count > 0
    {
        let visible_log_scales = gaussians
            .scales
            .as_tensor()
            .index_select(&projected.source_indices, 0)?;
        scale_regularization_term(&visible_log_scales)?
    } else {
        Tensor::new(0.0f32, color_loss.device())?
    };
    let transmittance_term = if config.enable_transmittance_loss {
        rendered.alpha.mean_all()?
    } else {
        Tensor::new(0.0f32, color_loss.device())?
    };

    let mut total = color_loss
        .affine(config.color_weight as f64, 0.0)?
        .broadcast_add(&Tensor::new(
            config.ssim_weight * ssim_loss_term,
            color_loss.device(),
        )?)?;
    if config.depth_weight > 0.0 {
        total = total.broadcast_add(&depth_loss.affine(config.depth_weight as f64, 0.0)?)?;
    }
    if config.scale_regularization_weight > 0.0 {
        total = total.broadcast_add(
            &scale_reg_term.affine(config.scale_regularization_weight as f64, 0.0)?,
        )?;
    }
    if config.enable_transmittance_loss {
        total = total.broadcast_add(&transmittance_term)?;
    }

    let total_loss = total.to_vec0::<f32>()?;
    let telemetry = MetalLossTelemetry {
        loss_terms: ParityLossTerms {
            l1: Some(color_loss.to_vec0::<f32>()?),
            ssim: Some(ssim_loss_term),
            scale_regularization: if config.scale_regularization_weight > 0.0 {
                Some(scale_reg_term.to_vec0::<f32>()?)
            } else {
                None
            },
            transmittance: if config.enable_transmittance_loss {
                Some(transmittance_term.to_vec0::<f32>()?)
            } else {
                None
            },
            depth: if config.depth_weight > 0.0 {
                Some(depth_loss.to_vec0::<f32>()?)
            } else {
                None
            },
            total: Some(total_loss),
        },
        depth_valid_pixels,
        depth_grad_scale,
    };
    let backward_loss_scales = backward_loss_scales(
        config.color_weight,
        config.ssim_weight,
        target_color_cpu.len(),
        depth_grad_scale.unwrap_or(0.0),
        config.enable_transmittance_loss,
        config.render_width * config.render_height,
    );

    Ok(MetalStepLossContext {
        total_loss,
        rendered_color_cpu,
        ssim_grads,
        backward_loss_scales,
        telemetry,
    })
}

fn is_valid_depth_sample(depth: f32) -> bool {
    depth.is_finite() && depth > 0.0
}

pub(crate) fn valid_depth_sample_count(depth: &[f32]) -> usize {
    depth
        .iter()
        .copied()
        .filter(|depth| is_valid_depth_sample(*depth))
        .count()
}

pub(crate) fn depth_backward_scale(depth_weight: f32, target_depth: &[f32]) -> f32 {
    if depth_weight <= 0.0 {
        return 0.0;
    }
    let valid_count = valid_depth_sample_count(target_depth);
    if valid_count == 0 {
        0.0
    } else {
        depth_weight / valid_count as f32
    }
}

pub(crate) fn scale_regularization_term(
    visible_log_scales: &Tensor,
) -> candle_core::Result<Tensor> {
    visible_log_scales.exp()?.sqr()?.mean_all()
}

pub(crate) fn scale_regularization_grad(
    visible_log_scales: &Tensor,
    weight: f32,
) -> candle_core::Result<Tensor> {
    let visible_elem_count = visible_log_scales.elem_count().max(1) as f32;
    visible_log_scales
        .exp()?
        .sqr()?
        .affine(((2.0 * weight) / visible_elem_count) as f64, 0.0)
}

pub(crate) fn optional_full_scale_regularization_grad(
    gaussians: &Splats,
    projected: &ProjectedGaussians,
    enabled: bool,
    weight: f32,
) -> candle_core::Result<Option<Tensor>> {
    if !enabled || weight <= 0.0 || projected.visible_count == 0 {
        return Ok(None);
    }

    let visible_log_scales = gaussians
        .scales
        .as_tensor()
        .index_select(&projected.source_indices, 0)?;
    let visible_reg_grad = scale_regularization_grad(&visible_log_scales, weight)?;
    Tensor::zeros_like(gaussians.scales.as_tensor())?
        .index_add(&projected.source_indices, &visible_reg_grad, 0)
        .map(Some)
}

impl CustomOp2 for MeanAbsDiff {
    fn name(&self) -> &'static str {
        "mean-abs-diff"
    }

    fn cpu_fwd(
        &self,
        lhs_storage: &CpuStorage,
        lhs_layout: &Layout,
        rhs_storage: &CpuStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        validate_layouts(self.name(), lhs_layout, rhs_layout)?;
        let lhs = Tensor::from_storage(
            Storage::Cpu(lhs_storage.clone()),
            lhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        let rhs = Tensor::from_storage(
            Storage::Cpu(rhs_storage.clone()),
            rhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        extract_cpu_scalar(&forward_loss(lhs, rhs)?)
    }

    fn metal_fwd(
        &self,
        lhs_storage: &MetalStorage,
        lhs_layout: &Layout,
        rhs_storage: &MetalStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        validate_layouts(self.name(), lhs_layout, rhs_layout)?;
        let lhs = Tensor::from_storage(
            Storage::Metal(lhs_storage.clone()),
            lhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        let rhs = Tensor::from_storage(
            Storage::Metal(rhs_storage.clone()),
            rhs_layout.shape().clone(),
            BackpropOp::none(),
            false,
        );
        extract_metal_scalar(&forward_loss(lhs, rhs)?)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        target: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        let dtype = arg.dtype();
        let diff = arg.sub(target)?;
        let positive = diff.gt(0.0f64)?.to_dtype(dtype)?;
        let negative = diff.lt(0.0f64)?.to_dtype(dtype)?;
        let sign = positive.broadcast_sub(&negative)?;
        let grad = sign
            .broadcast_mul(&grad_res.to_dtype(dtype)?)?
            .affine(1.0 / arg.elem_count() as f64, 0.0)?;
        Ok((Some(grad), None))
    }
}

fn validate_layouts(name: &str, lhs: &Layout, rhs: &Layout) -> candle_core::Result<()> {
    if lhs.shape() != rhs.shape() {
        candle_core::bail!(
            "{name} shape mismatch: lhs={:?}, rhs={:?}",
            lhs.shape().dims(),
            rhs.shape().dims()
        );
    }
    ensure_simple_layout(name, lhs)?;
    ensure_simple_layout(name, rhs)?;
    Ok(())
}

fn ensure_simple_layout(name: &str, layout: &Layout) -> candle_core::Result<()> {
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!(
            "{name} expects contiguous zero-offset tensors, got contiguous={} offset={}",
            layout.is_contiguous(),
            layout.start_offset()
        );
    }
    Ok(())
}

fn forward_loss(lhs: Tensor, rhs: Tensor) -> candle_core::Result<Tensor> {
    lhs.sub(&rhs)?.abs()?.mean_all()
}

fn extract_cpu_scalar(tensor: &Tensor) -> candle_core::Result<(CpuStorage, Shape)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(storage) => Ok((storage.clone(), layout.shape().clone())),
        other => candle_core::bail!("expected cpu scalar storage, got {:?}", other.device()),
    }
}

fn extract_metal_scalar(tensor: &Tensor) -> candle_core::Result<(MetalStorage, Shape)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(storage) => Ok((storage.clone(), layout.shape().clone())),
        other => candle_core::bail!("expected metal scalar storage, got {:?}", other.device()),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor, Var};

    use super::{masked_mean_abs_diff, mean_abs_diff};

    fn assert_close_scalar(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tol={tolerance}"
        );
    }

    fn assert_close_vec(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "idx={idx} expected {expected}, got {actual}, tol={tolerance}"
            );
        }
    }

    fn run_loss_parity_case(device: &Device, tolerance: f32) -> candle_core::Result<()> {
        let target = Tensor::from_vec(vec![0.2f32, -0.4, 0.9, -1.3], (2, 2), device)?;

        let baseline_pred = Var::from_tensor(&Tensor::from_vec(
            vec![0.5f32, -0.8, 0.3, -0.7],
            (2, 2),
            device,
        )?)?;
        let custom_pred = Var::from_tensor(&Tensor::from_vec(
            vec![0.5f32, -0.8, 0.3, -0.7],
            (2, 2),
            device,
        )?)?;

        let baseline = baseline_pred.sub(&target)?.abs()?.mean_all()?;
        let baseline_value = baseline.to_vec0::<f32>()?;
        let baseline_grads = baseline.backward()?;
        let baseline_grad = baseline_grads
            .get(&baseline_pred)
            .expect("baseline grad available")
            .flatten_all()?
            .to_vec1::<f32>()?;

        let custom = mean_abs_diff(custom_pred.as_tensor(), &target)?;
        let custom_value = custom.to_vec0::<f32>()?;
        let custom_grads = custom.backward()?;
        let custom_grad = custom_grads
            .get(&custom_pred)
            .expect("custom grad available")
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_close_scalar(custom_value, baseline_value, tolerance);
        assert_close_vec(&custom_grad, &baseline_grad, tolerance);
        Ok(())
    }

    #[test]
    fn mean_abs_diff_matches_baseline_on_cpu() -> candle_core::Result<()> {
        run_loss_parity_case(&Device::Cpu, 1e-6)
    }

    #[test]
    fn mean_abs_diff_matches_baseline_on_metal() -> candle_core::Result<()> {
        let Ok(device) = crate::try_metal_device() else {
            return Ok(());
        };
        run_loss_parity_case(&device, 1e-5)
    }

    #[test]
    fn masked_mean_abs_diff_ignores_invalid_entries() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let predicted = Tensor::from_vec(vec![1.0f32, 9.0, 3.0], 3, &device)?;
        let target = Tensor::from_vec(vec![2.0f32, 0.0, 1.0], 3, &device)?;
        let mask = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], 3, &device)?;

        let loss = masked_mean_abs_diff(&predicted, &target, &mask)?;
        assert_close_scalar(loss.to_vec0::<f32>()?, 1.5, 1e-6);
        Ok(())
    }

    #[test]
    fn masked_mean_abs_diff_returns_zero_without_valid_entries() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let predicted = Tensor::from_vec(vec![1.0f32, 9.0, 3.0], 3, &device)?;
        let target = Tensor::from_vec(vec![2.0f32, 0.0, 1.0], 3, &device)?;
        let mask = Tensor::zeros(3, candle_core::DType::F32, &device)?;

        let loss = masked_mean_abs_diff(&predicted, &target, &mask)?;
        assert_close_scalar(loss.to_vec0::<f32>()?, 0.0, 1e-6);
        Ok(())
    }

    #[test]
    fn ssim_gradient_identical_images_returns_one_and_zero_grad() {
        use super::ssim_gradient;
        // Identical rendered and target → SSIM = 1, gradient ≈ 0.
        let width: usize = 8;
        let height: usize = 8;
        let pixel_count = width * height * 3;
        let image: Vec<f32> = (0..pixel_count)
            .map(|i| (i as f32 % 256.0) / 255.0)
            .collect();

        let (ssim, grad) = ssim_gradient(&image, &image, width, height);
        assert!(
            (ssim - 1.0f32).abs() < 1e-4,
            "SSIM of identical images should be ~1.0, got {ssim}"
        );
        let max_grad: f32 = grad.iter().cloned().fold(f32::NEG_INFINITY, f32::max).abs();
        assert!(
            max_grad < 1e-4,
            "gradient of identical images should be ~0, max abs = {max_grad}"
        );
    }

    #[test]
    fn ssim_gradient_different_images_has_nonzero_grad() {
        use super::ssim_gradient;
        let width: usize = 8;
        let height: usize = 8;
        let pixel_count = width * height * 3;
        let rendered: Vec<f32> = (0..pixel_count)
            .map(|i| i as f32 / pixel_count as f32)
            .collect();
        let target: Vec<f32> = (0..pixel_count)
            .map(|i| 1.0f32 - i as f32 / pixel_count as f32)
            .collect();

        let (ssim, grad) = ssim_gradient(&rendered, &target, width, height);
        // SSIM < 1 for different images.
        assert!(
            ssim < 1.0f32,
            "SSIM should be < 1.0 for different images, got {ssim}"
        );
        // At least some gradients should be non-zero.
        let any_nonzero = grad.iter().any(|g: &f32| g.abs() > 1e-8);
        assert!(
            any_nonzero,
            "gradient should have non-zero entries for different images"
        );
    }
}
