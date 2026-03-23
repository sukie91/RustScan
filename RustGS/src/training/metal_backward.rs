use candle_core::{DType, Device, Tensor};

use crate::diff::analytical_backward::{self, GaussianRenderRecord};
#[cfg(test)]
use crate::diff::analytical_backward::ForwardIntermediate;
use crate::diff::diff_splat::DiffCamera;

use super::metal_runtime::{MetalBufferSlot, MetalRuntime};
use super::metal_trainer::{CpuProjectedGaussian, RenderedFrame};

pub(crate) struct MetalBackwardGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
    pub rotations: Tensor,
    pub opacity_logits: Tensor,
    pub colors: Tensor,
}

pub(crate) struct MetalBackwardPass {
    pub grads: MetalBackwardGrads,
    pub reference: analytical_backward::AnalyticalGradients,
}

pub(crate) fn backward_weighted_l1(
    runtime: &mut MetalRuntime,
    device: &Device,
    projected: &[CpuProjectedGaussian],
    rendered: &RenderedFrame,
    target_color: &[f32],
    target_depth: &[f32],
    n_gaussians: usize,
    camera: &DiffCamera,
) -> candle_core::Result<MetalBackwardPass> {
    let rendered_color = runtime.read_tensor_flat::<f32>(&rendered.color.flatten_all()?)?;
    let rendered_depth = runtime.read_tensor_flat::<f32>(&rendered.depth)?;
    let alpha_acc = runtime.read_tensor_flat::<f32>(&rendered.alpha)?;
    let records = build_records(projected);
    let reference = analytical_backward::backward_weighted_l1_from_buffers(
        &records,
        &rendered_color,
        &alpha_acc,
        &rendered_depth,
        camera.width,
        camera.height,
        target_color,
        target_depth,
        n_gaussians,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        1.0 / target_color.len().max(1) as f32,
        0.1 / target_depth.len().max(1) as f32,
    );

    let grads = MetalBackwardGrads {
        positions: runtime.stage_tensor_from_slice(
            MetalBufferSlot::GradPositions,
            &reference.positions,
            (n_gaussians, 3),
        )?,
        log_scales: runtime.stage_tensor_from_slice(
            MetalBufferSlot::GradScales,
            &reference.log_scales,
            (n_gaussians, 3),
        )?,
        rotations: Tensor::zeros((n_gaussians, 4), DType::F32, device)?,
        opacity_logits: runtime.stage_tensor_from_slice(
            MetalBufferSlot::GradOpacity,
            &reference.opacity_logits,
            (n_gaussians,),
        )?,
        colors: runtime.stage_tensor_from_slice(
            MetalBufferSlot::GradColors,
            &reference.colors,
            (n_gaussians, 3),
        )?,
    };

    Ok(MetalBackwardPass { grads, reference })
}

fn build_records(projected: &[CpuProjectedGaussian]) -> Vec<GaussianRenderRecord> {
    projected
        .iter()
        .map(|record| GaussianRenderRecord {
            gaussian_idx: record.source_idx as usize,
            u: record.u,
            v: record.v,
            sigma_x: record.sigma_x,
            sigma_y: record.sigma_y,
            z: record.depth,
            base_alpha: record.opacity.clamp(0.0, 1.0),
            color: record.color,
            min_x: record.min_x.floor().max(0.0) as usize,
            max_x: record.max_x.ceil().max(0.0) as usize,
            min_y: record.min_y.floor().max(0.0) as usize,
            max_y: record.max_y.ceil().max(0.0) as usize,
            raw_scale_2d_x: record.raw_sigma_x,
            raw_scale_2d_y: record.raw_sigma_y,
            raw_opacity: record.opacity,
            scale_3d: record.scale3d,
            opacity_logit: record.opacity_logit,
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn build_forward_intermediate(
    runtime: &MetalRuntime,
    projected: &[CpuProjectedGaussian],
    rendered: &RenderedFrame,
    width: usize,
    height: usize,
) -> candle_core::Result<ForwardIntermediate> {
    let mut records = Vec::with_capacity(projected.len());
    let rendered_color = runtime.read_tensor_flat::<f32>(&rendered.color.flatten_all()?)?;
    let rendered_depth = runtime.read_tensor_flat::<f32>(&rendered.depth)?;
    let alpha_acc = runtime.read_tensor_flat::<f32>(&rendered.alpha)?;
    records.extend(build_records(projected));

    Ok(ForwardIntermediate {
        records,
        rendered_color,
        alpha_acc,
        rendered_depth,
        width,
        height,
    })
}
