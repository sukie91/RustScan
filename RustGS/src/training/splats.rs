use candle_core::Device;
use serde::{Deserialize, Serialize};

use crate::diff::diff_splat::{
    rgb_to_sh0_value, sh0_to_rgb_value, sh_coeff_count_for_degree, SplatColorRepresentation,
    Splats as RuntimeSplats,
};

use super::{TrainingConfig, TrainingProfile};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSplats {
    pub(super) positions: Vec<f32>,
    pub(super) log_scales: Vec<f32>,
    pub(super) rotations: Vec<f32>,
    pub(super) opacity_logits: Vec<f32>,
    pub(super) sh_coeffs: Vec<f32>,
    pub(super) sh_degree: usize,
}

impl Default for HostSplats {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            log_scales: Vec::new(),
            rotations: Vec::new(),
            opacity_logits: Vec::new(),
            sh_coeffs: Vec::new(),
            sh_degree: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SplatView<'a> {
    pub positions: &'a [f32],
    pub log_scales: &'a [f32],
    pub rotations: &'a [f32],
    pub opacity_logits: &'a [f32],
    pub sh_coeffs: &'a [f32],
    pub sh_degree: usize,
}

#[cfg(test)]
pub(crate) type Splats = HostSplats;

impl HostSplats {
    pub(crate) fn with_sh_degree_capacity(sh_degree: usize, row_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(row_count * 3),
            log_scales: Vec::with_capacity(row_count * 3),
            rotations: Vec::with_capacity(row_count * 4),
            opacity_logits: Vec::with_capacity(row_count),
            sh_coeffs: Vec::with_capacity(row_count * sh_coeff_count_for_degree(sh_degree) * 3),
            sh_degree,
        }
    }

    pub fn from_raw_parts(
        positions: Vec<f32>,
        log_scales: Vec<f32>,
        rotations: Vec<f32>,
        opacity_logits: Vec<f32>,
        sh_coeffs: Vec<f32>,
        sh_degree: usize,
    ) -> candle_core::Result<Self> {
        let splats = Self {
            positions,
            log_scales,
            rotations,
            opacity_logits,
            sh_coeffs,
            sh_degree,
        };
        splats.validate()?;
        Ok(splats)
    }

    pub fn from_runtime(gaussians: &RuntimeSplats) -> candle_core::Result<Self> {
        let sh_degree = gaussians.sh_degree();
        let mut sh_coeffs =
            Vec::with_capacity(gaussians.len() * sh_coeff_count_for_degree(sh_degree) * 3);
        let sh_0 = flatten_rows(gaussians.colors().to_vec2::<f32>()?);
        let sh_rest = flatten_3d(gaussians.sh_rest().to_vec3::<f32>()?);
        if sh_degree == 0 && !gaussians.uses_spherical_harmonics() {
            sh_coeffs.extend(sh_0.into_iter().map(rgb_to_sh0_value));
        } else {
            let sh_rest_row_width = sh_coeff_count_for_degree(sh_degree).saturating_sub(1) * 3;
            for idx in 0..gaussians.len() {
                let base = idx * 3;
                sh_coeffs.extend_from_slice(&sh_0[base..base + 3]);
                let rest = row_slice(&sh_rest, sh_rest_row_width, idx);
                sh_coeffs.extend_from_slice(rest);
            }
        }
        let splats = Self {
            positions: flatten_rows(gaussians.positions().to_vec2::<f32>()?),
            log_scales: flatten_rows(gaussians.scales.as_tensor().to_vec2::<f32>()?),
            rotations: flatten_rows(gaussians.rotations.as_tensor().to_vec2::<f32>()?),
            opacity_logits: gaussians.opacities.as_tensor().to_vec1::<f32>()?,
            sh_coeffs,
            sh_degree,
        };
        splats.validate()?;
        Ok(splats)
    }

    pub fn upload(&self, device: &Device) -> candle_core::Result<RuntimeSplats> {
        self.validate()?;
        match self.sh_degree {
            0 => RuntimeSplats::new(
                &self.positions,
                &self.log_scales,
                &self.rotations,
                &self.opacity_logits,
                &self.rgb_colors(),
                device,
            ),
            degree => {
                let (sh_0, sh_rest) = self.split_sh_coeffs();
                RuntimeSplats::new_with_sh(
                    &self.positions,
                    &self.log_scales,
                    &self.rotations,
                    &self.opacity_logits,
                    &sh_0,
                    &sh_rest,
                    degree,
                    device,
                )
            }
        }
    }

    pub fn to_runtime(&self, device: &Device) -> candle_core::Result<RuntimeSplats> {
        self.upload(device)
    }

    pub fn as_view(&self) -> SplatView<'_> {
        SplatView {
            positions: &self.positions,
            log_scales: &self.log_scales,
            rotations: &self.rotations,
            opacity_logits: &self.opacity_logits,
            sh_coeffs: &self.sh_coeffs,
            sh_degree: self.sh_degree,
        }
    }

    pub fn validate(&self) -> candle_core::Result<()> {
        let row_count = self.opacity_logits.len();
        validate_component_len("positions", self.positions.len(), row_count, 3)?;
        validate_component_len("log_scales", self.log_scales.len(), row_count, 3)?;
        validate_component_len("rotations", self.rotations.len(), row_count, 4)?;
        validate_component_len(
            "sh_coeffs",
            self.sh_coeffs.len(),
            row_count,
            self.sh_coeffs_row_width(),
        )?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub(super) fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    pub(super) fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales[base],
            self.log_scales[base + 1],
            self.log_scales[base + 2],
        ]
    }

    pub(super) fn rotation(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    pub(super) fn sh_0(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.sh_coeffs[base],
            self.sh_coeffs[base + 1],
            self.sh_coeffs[base + 2],
        ]
    }

    pub(super) fn rgb_color(&self, idx: usize) -> [f32; 3] {
        self.sh_0(idx).map(sh0_to_rgb_value)
    }

    pub fn sh_degree(&self) -> usize {
        self.sh_degree
    }

    pub(super) fn sh_coeffs_row_width(&self) -> usize {
        sh_coeff_count_for_degree(self.sh_degree) * 3
    }

    pub(super) fn sh_rest_row_width(&self) -> usize {
        self.sh_coeffs_row_width().saturating_sub(3)
    }

    pub(super) fn sh_coeffs_row(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_coeffs, self.sh_coeffs_row_width(), idx)
    }

    pub(super) fn sh_rest(&self, idx: usize) -> &[f32] {
        self.sh_coeffs_row(idx).get(3..).unwrap_or(&[])
    }

    pub(super) fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    #[allow(dead_code)]
    pub(super) fn push(
        &mut self,
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity_logit: f32,
        sh_coeffs: &[f32],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.push_sh_coeffs_row(sh_coeffs);
    }

    pub(crate) fn push_rgb(
        &mut self,
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity_logit: f32,
        rgb: [f32; 3],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.sh_coeffs.extend(rgb.map(rgb_to_sh0_value));
        let sh_rest_width = self.sh_rest_row_width();
        if sh_rest_width > 0 {
            self.sh_coeffs
                .resize(self.sh_coeffs.len() + sh_rest_width, 0.0);
        }
    }

    pub(super) fn truncate_rows(&mut self, row_count: usize) {
        if row_count >= self.len() {
            return;
        }

        self.positions.truncate(row_count * 3);
        self.log_scales.truncate(row_count * 3);
        self.rotations.truncate(row_count * 4);
        self.opacity_logits.truncate(row_count);
        self.sh_coeffs
            .truncate(row_count * self.sh_coeffs_row_width());
    }

    #[allow(dead_code)]
    pub(super) fn retained_view(&self, row_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(row_count * 3),
            log_scales: Vec::with_capacity(row_count * 3),
            rotations: Vec::with_capacity(row_count * 4),
            opacity_logits: Vec::with_capacity(row_count),
            sh_coeffs: Vec::with_capacity(row_count * self.sh_coeffs_row_width()),
            sh_degree: self.sh_degree,
        }
    }

    pub(super) fn downsample_evenly(&mut self, target_count: usize) {
        if target_count == 0 || self.len() <= target_count {
            return;
        }

        let len = self.len();
        let mut sampled = self.retained_view(target_count);
        for out_idx in 0..target_count {
            let src_idx = out_idx.saturating_mul(len) / target_count;
            let src_idx = src_idx.min(len.saturating_sub(1));
            sampled.push(
                self.position(src_idx),
                self.log_scale(src_idx),
                self.rotation(src_idx),
                self.opacity_logits[src_idx],
                self.sh_coeffs_row(src_idx),
            );
        }
        *self = sampled;
    }

    pub(super) fn positions_vec3(&self) -> Vec<[f32; 3]> {
        (0..self.len()).map(|idx| self.position(idx)).collect()
    }

    pub(super) fn scene_extent(&self) -> f32 {
        if self.is_empty() {
            return 1.0;
        }

        let mut center = [0.0f32; 3];
        for idx in 0..self.len() {
            let position = self.position(idx);
            center[0] += position[0];
            center[1] += position[1];
            center[2] += position[2];
        }
        let inv = 1.0 / self.len().max(1) as f32;
        center[0] *= inv;
        center[1] *= inv;
        center[2] *= inv;

        let mut max_dist = 0.0f32;
        for idx in 0..self.len() {
            let position = self.position(idx);
            let dx = position[0] - center[0];
            let dy = position[1] - center[1];
            let dz = position[2] - center[2];
            max_dist = max_dist.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        max_dist.max(1e-3)
    }

    pub(super) fn push_sh_coeffs(&mut self, sh_0: [f32; 3], sh_rest: &[f32]) {
        self.sh_coeffs.extend_from_slice(&sh_0);
        let sh_rest_row_width = self.sh_rest_row_width();
        if sh_rest_row_width == 0 {
            return;
        }
        let copied = sh_rest.len().min(sh_rest_row_width);
        self.sh_coeffs.extend_from_slice(&sh_rest[..copied]);
        if copied < sh_rest_row_width {
            self.sh_coeffs
                .resize(self.sh_coeffs.len() + (sh_rest_row_width - copied), 0.0);
        }
    }

    fn push_sh_coeffs_row(&mut self, sh_coeffs: &[f32]) {
        let row_width = self.sh_coeffs_row_width();
        if row_width == 0 {
            return;
        }
        let copied = sh_coeffs.len().min(row_width);
        self.sh_coeffs.extend_from_slice(&sh_coeffs[..copied]);
        if copied < row_width {
            self.sh_coeffs
                .resize(self.sh_coeffs.len() + (row_width - copied), 0.0);
        }
    }

    fn split_sh_coeffs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut sh_0 = Vec::with_capacity(self.len() * 3);
        let mut sh_rest = Vec::with_capacity(self.len() * self.sh_rest_row_width());
        for idx in 0..self.len() {
            sh_0.extend_from_slice(&self.sh_0(idx));
            sh_rest.extend_from_slice(self.sh_rest(idx));
        }
        (sh_0, sh_rest)
    }

    fn rgb_colors(&self) -> Vec<f32> {
        let mut rgb = Vec::with_capacity(self.len() * 3);
        for idx in 0..self.len() {
            rgb.extend(self.rgb_color(idx));
        }
        rgb
    }
}

pub(super) fn splat_color_representation_for_config(
    config: &TrainingConfig,
) -> SplatColorRepresentation {
    if config.training_profile == TrainingProfile::LiteGsMacV1 {
        SplatColorRepresentation::SphericalHarmonics {
            degree: config.litegs.sh_degree,
        }
    } else {
        SplatColorRepresentation::Rgb
    }
}

pub(super) fn row_slice(values: &[f32], width: usize, idx: usize) -> &[f32] {
    let start = idx.saturating_mul(width);
    let end = start.saturating_add(width);
    values.get(start..end).unwrap_or(&[])
}

fn validate_component_len(
    name: &str,
    actual: usize,
    row_count: usize,
    row_width: usize,
) -> candle_core::Result<()> {
    let expected = row_count.saturating_mul(row_width);
    if actual != expected {
        candle_core::bail!(
            "splats invariant violated: {name} expected {expected} values for {row_count} gaussians, got {actual}"
        );
    }
    Ok(())
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

pub(super) fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

pub(super) fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::{splat_color_representation_for_config, Splats};
    use crate::diff::diff_splat::{
        rgb_to_sh0_value, SplatColorRepresentation, Splats as RuntimeSplats,
    };
    use crate::{LiteGsConfig, TrainingConfig, TrainingProfile};
    use candle_core::Device;

    #[test]
    fn validation_rejects_mismatched_component_lengths() {
        let invalid = Splats {
            positions: vec![0.0; 5],
            log_scales: vec![0.0; 6],
            rotations: vec![0.0; 8],
            opacity_logits: vec![0.0; 2],
            sh_coeffs: vec![0.0; 6],
            sh_degree: 0,
        };

        let err = invalid.validate().unwrap_err().to_string();
        assert!(err.contains("positions expected 6 values"));
    }

    #[test]
    fn rgb_splats_round_trip_trainables() {
        let device = Device::Cpu;
        let gaussians = RuntimeSplats::new(
            &[0.0, 0.0, 1.0, 1.0, 0.5, 2.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[1.0, 0.0, 0.0, 0.0, 0.707, 0.0, 0.707, 0.0],
            &[0.1, -0.2],
            &[1.0, 0.0, 0.0, 0.2, 0.4, 0.6],
            &device,
        )
        .unwrap();

        let splats = Splats::from_runtime(&gaussians).unwrap();
        let rebuilt = splats.to_runtime(&device).unwrap();

        assert_eq!(
            rebuilt.positions().to_vec2::<f32>().unwrap(),
            gaussians.positions().to_vec2::<f32>().unwrap()
        );
        let rebuilt_colors = rebuilt.colors().to_vec2::<f32>().unwrap();
        let source_colors = gaussians.colors().to_vec2::<f32>().unwrap();
        for (rebuilt_row, source_row) in rebuilt_colors.iter().zip(source_colors.iter()) {
            for (rebuilt, source) in rebuilt_row.iter().zip(source_row.iter()) {
                assert!((rebuilt - source).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn sh_splats_round_trip_trainables() {
        let device = Device::Cpu;
        let gaussians = RuntimeSplats::new_with_sh(
            &[0.0, 0.0, 1.0],
            &[0.1, 0.2, 0.3],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.1],
            &[
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ],
            &vec![0.5; 15 * 3],
            3,
            &device,
        )
        .unwrap();

        let splats = Splats::from_runtime(&gaussians).unwrap();
        let rebuilt = splats.to_runtime(&device).unwrap();

        assert!(rebuilt.uses_spherical_harmonics());
        assert_eq!(rebuilt.sh_degree(), 3);
        assert_eq!(
            rebuilt.sh_rest().to_vec3::<f32>().unwrap(),
            gaussians.sh_rest().to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn config_selects_internal_splat_color_representation() {
        let legacy = splat_color_representation_for_config(&TrainingConfig::default());
        assert_eq!(legacy, SplatColorRepresentation::Rgb);

        let litegs = splat_color_representation_for_config(&TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sh_degree: 3,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        });
        assert_eq!(
            litegs,
            SplatColorRepresentation::SphericalHarmonics { degree: 3 }
        );
    }
}
