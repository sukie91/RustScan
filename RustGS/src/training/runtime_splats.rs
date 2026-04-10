use candle_core::Device;

use crate::diff::diff_splat::{sh_coeff_count_for_degree, Splats as RuntimeSplats};

use super::splats::{row_slice, sigmoid_scalar};
#[cfg(test)]
use super::splats::HostSplats;
use super::topology::{TopologyMutationPlan, TopologyPlanRow};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TopologySplatMetrics {
    positions: Vec<f32>,
    log_scales: Vec<f32>,
    opacity_logits: Vec<f32>,
    retainable: Vec<bool>,
}

impl TopologySplatMetrics {
    #[cfg(test)]
    pub(super) fn from_snapshot(splats: &HostSplats) -> Self {
        Self {
            positions: splats.positions.clone(),
            log_scales: splats.log_scales.clone(),
            opacity_logits: splats.opacity_logits.clone(),
            retainable: (0..splats.len())
                .map(|idx| {
                    splats.position(idx).iter().all(|value| value.is_finite())
                        && splats.rotation(idx).iter().all(|value| value.is_finite())
                        && splats
                            .sh_coeffs_row(idx)
                            .iter()
                            .all(|value| value.is_finite())
                })
                .collect(),
        }
    }

    pub(super) fn from_runtime(gaussians: &RuntimeSplats) -> candle_core::Result<Self> {
        let positions = flatten_rows(gaussians.positions().to_vec2::<f32>()?);
        let log_scales = gaussians.scales.as_tensor().to_vec2::<f32>()?;
        let rotations = gaussians.rotations.as_tensor().to_vec2::<f32>()?;
        let sh_0 = gaussians.colors().to_vec2::<f32>()?;
        let sh_rest = gaussians.sh_rest().to_vec3::<f32>()?;
        let sh_rest_row_width = sh_coeff_count_for_degree(gaussians.sh_degree()).saturating_sub(1);
        let retainable = (0..gaussians.len())
            .map(|idx| {
                positions
                    .get(idx * 3..idx * 3 + 3)
                    .unwrap_or(&[])
                    .iter()
                    .all(|value| value.is_finite())
                    && rotations
                        .get(idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .all(|value| value.is_finite())
                    && sh_0
                        .get(idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .all(|value| value.is_finite())
                    && sh_rest
                        .get(idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .flat_map(|channel| channel.iter())
                        .count()
                        == sh_rest_row_width * 3
                    && sh_rest
                        .get(idx)
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                        .iter()
                        .flat_map(|channel| channel.iter())
                        .all(|value| value.is_finite())
            })
            .collect();
        Ok(Self {
            positions,
            log_scales: flatten_rows(log_scales),
            opacity_logits: gaussians.opacities.as_tensor().to_vec1::<f32>()?,
            retainable,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    pub(super) fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions.get(base).copied().unwrap_or_default(),
            self.positions.get(base + 1).copied().unwrap_or_default(),
            self.positions.get(base + 2).copied().unwrap_or_default(),
        ]
    }

    pub(super) fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales.get(base).copied().unwrap_or_default(),
            self.log_scales.get(base + 1).copied().unwrap_or_default(),
            self.log_scales.get(base + 2).copied().unwrap_or_default(),
        ]
    }

    pub(super) fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    pub(super) fn max_scale(&self, idx: usize) -> f32 {
        let scale = self.scale(idx);
        scale[0].max(scale[1]).max(scale[2])
    }

    pub(super) fn max_scale_axis(&self, idx: usize) -> usize {
        let scale = self.scale(idx);
        let mut max_axis = 0usize;
        let mut max_value = scale[0];
        for (axis, value) in scale.into_iter().enumerate().skip(1) {
            if value > max_value {
                max_axis = axis;
                max_value = value;
            }
        }
        max_axis
    }

    pub(super) fn opacity(&self, idx: usize) -> f32 {
        self.opacity_logits
            .get(idx)
            .copied()
            .map(sigmoid_scalar)
            .unwrap_or_default()
    }

    pub(super) fn retainable(&self, idx: usize) -> bool {
        self.retainable.get(idx).copied().unwrap_or(false)
    }
}

pub(super) fn apply_topology_plan(
    gaussians: &RuntimeSplats,
    plan: &TopologyMutationPlan,
    device: &Device,
) -> candle_core::Result<RuntimeSplats> {
    let positions = flatten_rows(gaussians.positions().to_vec2::<f32>()?);
    let log_scales = flatten_rows(gaussians.scales.as_tensor().to_vec2::<f32>()?);
    let rotations = flatten_rows(gaussians.rotations.as_tensor().to_vec2::<f32>()?);
    let opacity_logits = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
    let base_colors = flatten_rows(gaussians.colors().to_vec2::<f32>()?);
    let sh_degree = gaussians.sh_degree();
    let sh_rest_row_width = sh_coeff_count_for_degree(sh_degree).saturating_sub(1) * 3;
    let sh_rest = flatten_3d(gaussians.sh_rest().to_vec3::<f32>()?);

    let mut final_positions = Vec::with_capacity(plan.rows.len() * 3);
    let mut final_log_scales = Vec::with_capacity(plan.rows.len() * 3);
    let mut final_rotations = Vec::with_capacity(plan.rows.len() * 4);
    let mut final_opacity_logits = Vec::with_capacity(plan.rows.len());
    let mut final_colors = Vec::with_capacity(plan.rows.len() * 3);
    let mut final_sh_rest = Vec::with_capacity(plan.rows.len() * sh_rest_row_width);

    for row in &plan.rows {
        let source_idx = row.source_idx();
        let mut position = vec3_from_row(row_slice(&positions, 3, source_idx));
        let mut log_scale = vec3_from_row(row_slice(&log_scales, 3, source_idx));
        let rotation = vec4_from_row(row_slice(&rotations, 4, source_idx));
        let opacity_logit = opacity_logits.get(source_idx).copied().unwrap_or_default();
        let color = vec3_from_row(row_slice(&base_colors, 3, source_idx));
        let sh_rest_row = row_slice(&sh_rest, sh_rest_row_width, source_idx);

        match *row {
            TopologyPlanRow::Existing { .. } | TopologyPlanRow::LiteClone { .. } => {}
            TopologyPlanRow::LegacyOffsetClone { axis, .. } => {
                if axis < 3 {
                    position[axis] += log_scale[axis].exp().max(0.01) * 0.5;
                }
            }
            TopologyPlanRow::LegacySplit { direction, .. } => {
                let scale = [log_scale[0].exp(), log_scale[1].exp(), log_scale[2].exp()];
                let max_scale = scale[0].max(scale[1]).max(scale[2]);
                position[0] += (direction as f32) * max_scale * 0.1;
                log_scale[0] = (max_scale * 0.5).max(1e-6).ln();
            }
            TopologyPlanRow::LiteSplit { axis, .. } => {
                if axis < 3 {
                    let axis_scale = log_scale[axis].exp();
                    position[axis] += axis_scale * 0.5;
                    log_scale[axis] = (axis_scale / 1.6).max(1e-6).ln();
                }
            }
        }

        final_positions.extend_from_slice(&position);
        final_log_scales.extend_from_slice(&log_scale);
        final_rotations.extend_from_slice(&rotation);
        final_opacity_logits.push(opacity_logit);
        final_colors.extend_from_slice(&color);
        if sh_rest_row_width > 0 {
            final_sh_rest.extend_from_slice(sh_rest_row);
        }
    }

    match sh_degree {
        0 => RuntimeSplats::new(
            &final_positions,
            &final_log_scales,
            &final_rotations,
            &final_opacity_logits,
            &final_colors,
            device,
        ),
        degree => RuntimeSplats::new_with_sh(
            &final_positions,
            &final_log_scales,
            &final_rotations,
            &final_opacity_logits,
            &final_colors,
            &final_sh_rest,
            degree,
            device,
        ),
    }
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

fn vec3_from_row(row: &[f32]) -> [f32; 3] {
    [
        row.first().copied().unwrap_or_default(),
        row.get(1).copied().unwrap_or_default(),
        row.get(2).copied().unwrap_or_default(),
    ]
}

fn vec4_from_row(row: &[f32]) -> [f32; 4] {
    [
        row.first().copied().unwrap_or(1.0),
        row.get(1).copied().unwrap_or_default(),
        row.get(2).copied().unwrap_or_default(),
        row.get(3).copied().unwrap_or_default(),
    ]
}
