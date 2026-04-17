use burn::module::Param;
use burn::prelude::*;
use glam::{Quat, Vec3};

use crate::core::HostSplats;
use crate::training::topology::{TopologyMutationPlan, TopologyPlanRow};

use super::splats::{device_splats_to_host, host_splats_to_device, DeviceSplats};

pub(crate) async fn apply_mutations<B: Backend>(
    splats: &mut DeviceSplats<B>,
    plan: &TopologyMutationPlan,
    device: &B::Device,
) {
    if plan.rows.is_empty() {
        return;
    }

    let snapshot = device_splats_to_host(splats).await;
    let rebuilt = rebuild_host_snapshot(&snapshot, plan);
    let updated = host_splats_to_device(&rebuilt, device);

    splats.transforms = Param::initialized(splats.transforms.id, updated.transforms.val());
    splats.sh_coeffs = Param::initialized(splats.sh_coeffs.id, updated.sh_coeffs.val());
    splats.raw_opacities = Param::initialized(splats.raw_opacities.id, updated.raw_opacities.val());
    splats.sh_degree = updated.sh_degree;
}

fn rebuild_host_snapshot(
    snapshot: &HostSplats,
    plan: &TopologyMutationPlan,
) -> HostSplats {
    let sh_row_width = if snapshot.is_empty() {
        3
    } else {
        snapshot.sh_coeffs_row(0).len()
    };
    let mut positions = Vec::with_capacity(plan.rows.len() * 3);
    let mut log_scales = Vec::with_capacity(plan.rows.len() * 3);
    let mut rotations = Vec::with_capacity(plan.rows.len() * 4);
    let mut opacity_logits = Vec::with_capacity(plan.rows.len());
    let mut sh_coeffs = Vec::with_capacity(plan.rows.len() * sh_row_width);

    for row in &plan.rows {
        let source_idx = row.source_idx();
        let mut position = snapshot.position(source_idx);
        let mut log_scale = snapshot.log_scale(source_idx);
        let rotation = snapshot.rotation(source_idx);
        let mut opacity_logit = snapshot.opacity_logit(source_idx);

        match *row {
            TopologyPlanRow::Existing { .. } => {}
            TopologyPlanRow::LegacyOffsetClone { axis, .. } => {
                let scale = snapshot.scale(source_idx);
                if axis < 3 {
                    position[axis] += scale[axis].max(0.01) * 0.5;
                }
            }
            TopologyPlanRow::LegacySplit { direction, .. } => {
                let scale = snapshot.scale(source_idx);
                let max_scale = scale[0].max(scale[1]).max(scale[2]);
                position[0] += (direction as f32) * max_scale * 0.1;
                log_scale[0] = (max_scale * 0.5).max(1e-6).ln();
            }
            TopologyPlanRow::BrushRefineExisting { sample_scalar, .. } => {
                let offset =
                    brush_refine_offset(rotation, snapshot.scale(source_idx), sample_scalar);
                let refined_scale = brush_refine_scale(snapshot.scale(source_idx));
                position[0] -= offset[0];
                position[1] -= offset[1];
                position[2] -= offset[2];
                log_scale = refined_scale.map(|value| value.max(1e-6).ln());
                opacity_logit = brush_refine_opacity_logit(opacity_logit);
            }
            TopologyPlanRow::BrushRefineNew { sample_scalar, .. } => {
                let offset =
                    brush_refine_offset(rotation, snapshot.scale(source_idx), sample_scalar);
                let refined_scale = brush_refine_scale(snapshot.scale(source_idx));
                position[0] += offset[0];
                position[1] += offset[1];
                position[2] += offset[2];
                log_scale = refined_scale.map(|value| value.max(1e-6).ln());
                opacity_logit = brush_refine_opacity_logit(opacity_logit);
            }
        }

        positions.extend_from_slice(&position);
        log_scales.extend_from_slice(&log_scale);
        rotations.extend_from_slice(&rotation);
        opacity_logits.push(opacity_logit);
        sh_coeffs.extend_from_slice(snapshot.sh_coeffs_row(source_idx));
    }

    HostSplats::from_raw_parts(
        positions,
        log_scales,
        rotations,
        opacity_logits,
        sh_coeffs,
        snapshot.sh_degree(),
    )
    .expect("valid topology mutation rebuild")
}

fn brush_refine_offset(rotation: [f32; 4], scale: [f32; 3], sample_scalar: f32) -> [f32; 3] {
    let quat = Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    let rotated = quat * (Vec3::from_array(scale) * sample_scalar);
    rotated.to_array()
}

fn brush_refine_scale(scale: [f32; 3]) -> [f32; 3] {
    let mut refined = scale;
    let mut max_axis = 0usize;
    for axis in 1..3 {
        if refined[axis] > refined[max_axis] {
            max_axis = axis;
        }
    }
    refined[max_axis] *= 0.5;
    refined
}

fn brush_refine_opacity_logit(opacity_logit: f32) -> f32 {
    let opacity = sigmoid_scalar(opacity_logit).clamp(1.0 / 255.0, 1.0 - 1.0 / 255.0);
    let refined = (1.0 - (1.0 - opacity).sqrt()).clamp(1.0 / 255.0, 1.0 - 1.0 / 255.0);
    (refined / (1.0 - refined)).ln()
}

fn sigmoid_scalar(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
