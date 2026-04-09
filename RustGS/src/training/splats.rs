use candle_core::Device;
use serde::{Deserialize, Serialize};

use crate::core::{Gaussian3D, GaussianColorRepresentation, GaussianMap, GaussianState};
use crate::diff::diff_splat::{
    rgb_to_sh0_value, sh0_to_rgb_value, sh_coeff_count_for_degree, Splats as RuntimeSplats,
    TrainableColorRepresentation, TrainableGaussians,
};
use crate::Gaussian;

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

pub(crate) type Splats = HostSplats;

impl HostSplats {
    pub(super) fn with_sh_degree_capacity(sh_degree: usize, row_count: usize) -> Self {
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

    #[deprecated(note = "Use HostSplats::from_runtime(...) instead.")]
    pub fn from_trainable(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        Self::from_runtime(gaussians)
    }

    pub fn from_legacy_gaussians(
        gaussians: &[Gaussian3D],
        color_representation: TrainableColorRepresentation,
    ) -> candle_core::Result<Self> {
        let sh_degree = color_representation.sh_degree();
        let mut splats = Self {
            positions: Vec::with_capacity(gaussians.len() * 3),
            log_scales: Vec::with_capacity(gaussians.len() * 3),
            rotations: Vec::with_capacity(gaussians.len() * 4),
            opacity_logits: Vec::with_capacity(gaussians.len()),
            sh_coeffs: Vec::with_capacity(
                gaussians.len() * sh_coeff_count_for_degree(sh_degree) * 3,
            ),
            sh_degree,
        };

        for gaussian in gaussians {
            splats.positions.extend_from_slice(&[
                gaussian.position.x,
                gaussian.position.y,
                gaussian.position.z,
            ]);
            splats.log_scales.extend_from_slice(&[
                gaussian.scale.x.max(1e-6).ln(),
                gaussian.scale.y.max(1e-6).ln(),
                gaussian.scale.z.max(1e-6).ln(),
            ]);
            splats.rotations.extend_from_slice(&[
                gaussian.rotation.w,
                gaussian.rotation.x,
                gaussian.rotation.y,
                gaussian.rotation.z,
            ]);
            splats
                .opacity_logits
                .push(opacity_to_logit(gaussian.opacity));

            match color_representation {
                TrainableColorRepresentation::Rgb => {
                    splats.push_sh_coeffs(gaussian.color.map(rgb_to_sh0_value), &[]);
                }
                TrainableColorRepresentation::SphericalHarmonics { degree } => {
                    match gaussian.color_representation {
                        GaussianColorRepresentation::Rgb => {
                            splats.push_sh_coeffs(gaussian.color.map(rgb_to_sh0_value), &[]);
                        }
                        GaussianColorRepresentation::SphericalHarmonics {
                            degree: stored_degree,
                        } => {
                            if stored_degree != degree {
                                candle_core::bail!(
                                    "gaussian map mixes SH degree {stored_degree} with requested degree {degree}"
                                );
                            }
                            splats.push_sh_coeffs(
                                gaussian
                                    .sh_dc
                                    .unwrap_or_else(|| gaussian.color.map(rgb_to_sh0_value)),
                                gaussian.sh_rest.as_deref().unwrap_or(&[]),
                            );
                        }
                    }
                }
            }
        }

        splats.validate()?;
        Ok(splats)
    }

    pub fn from_legacy_gaussians_for_config(
        gaussians: &[Gaussian3D],
        config: &TrainingConfig,
    ) -> candle_core::Result<Self> {
        Self::from_legacy_gaussians(gaussians, splat_color_representation_for_config(config))
    }

    pub fn from_legacy_gaussians_inferred(gaussians: &[Gaussian3D]) -> candle_core::Result<Self> {
        let color_representation = infer_color_representation_from_legacy_gaussians(gaussians)?;
        Self::from_legacy_gaussians(gaussians, color_representation)
    }

    #[deprecated(note = "Use HostSplats::from_legacy_gaussians(...) instead.")]
    pub fn from_gaussian_map(
        map: &GaussianMap,
        color_representation: TrainableColorRepresentation,
    ) -> candle_core::Result<Self> {
        Self::from_legacy_gaussians(map.gaussians(), color_representation)
    }

    #[deprecated(note = "Use HostSplats::from_legacy_gaussians_for_config(...) instead.")]
    pub fn from_gaussian_map_for_config(
        map: &GaussianMap,
        config: &TrainingConfig,
    ) -> candle_core::Result<Self> {
        Self::from_legacy_gaussians_for_config(map.gaussians(), config)
    }

    #[deprecated(note = "Use HostSplats::from_legacy_gaussians_inferred(...) instead.")]
    pub fn from_gaussian_map_inferred(map: &GaussianMap) -> candle_core::Result<Self> {
        Self::from_legacy_gaussians_inferred(map.gaussians())
    }

    pub fn from_scene_gaussians(scene: &[Gaussian], sh_degree: usize) -> candle_core::Result<Self> {
        let mut splats = Self {
            positions: Vec::with_capacity(scene.len() * 3),
            log_scales: Vec::with_capacity(scene.len() * 3),
            rotations: Vec::with_capacity(scene.len() * 4),
            opacity_logits: Vec::with_capacity(scene.len()),
            sh_coeffs: Vec::with_capacity(scene.len() * sh_coeff_count_for_degree(sh_degree) * 3),
            sh_degree,
        };
        let sh_rest_row_width = splats.sh_rest_row_width();

        for (index, gaussian) in scene.iter().enumerate() {
            splats.positions.extend_from_slice(&gaussian.position);
            splats.log_scales.extend_from_slice(&[
                gaussian.scale[0].max(1e-6).ln(),
                gaussian.scale[1].max(1e-6).ln(),
                gaussian.scale[2].max(1e-6).ln(),
            ]);
            splats.rotations.extend_from_slice(&gaussian.rotation);
            splats
                .opacity_logits
                .push(opacity_to_logit(gaussian.opacity));

            let sh_rest = if sh_rest_row_width > 0 {
                match gaussian.sh_rest.as_ref() {
                    Some(values) => {
                        if values.len() != sh_rest_row_width {
                            candle_core::bail!(
                                "gaussian {index} has {} SH-rest values, expected {} for degree {}",
                                values.len(),
                                sh_rest_row_width,
                                sh_degree
                            );
                        }
                        values.as_slice()
                    }
                    None => &[],
                }
            } else {
                &[]
            };
            splats.push_sh_coeffs(gaussian.color.map(rgb_to_sh0_value), sh_rest);
            if sh_rest_row_width > 0 && gaussian.sh_rest.is_none() {
                // `push_sh_coeffs` already zero-fills missing SH rows.
            }
        }

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

    #[deprecated(note = "Use HostSplats::to_runtime(...) or HostSplats::upload(...) instead.")]
    pub fn to_trainable(&self, device: &Device) -> candle_core::Result<TrainableGaussians> {
        self.to_runtime(device)
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

    pub fn to_legacy_gaussians(&self) -> candle_core::Result<Vec<Gaussian3D>> {
        self.validate()?;
        let color_representation = if self.sh_degree == 0 {
            GaussianColorRepresentation::Rgb
        } else {
            GaussianColorRepresentation::SphericalHarmonics {
                degree: self.sh_degree,
            }
        };

        let mut output = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let scale = self.scale(idx);
            let rotation = self.rotation(idx);
            let sh_0 = self.sh_0(idx);
            let color = self.rgb_color(idx);
            output.push(Gaussian3D {
                position: glam::Vec3::new(
                    self.positions[idx * 3],
                    self.positions[idx * 3 + 1],
                    self.positions[idx * 3 + 2],
                ),
                scale: glam::Vec3::new(scale[0], scale[1], scale[2]),
                rotation: glam::Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]),
                opacity: sigmoid_scalar(self.opacity_logits[idx]).clamp(0.0, 1.0),
                color,
                color_representation,
                sh_dc: matches!(
                    color_representation,
                    GaussianColorRepresentation::SphericalHarmonics { .. }
                )
                .then_some(sh_0),
                sh_rest: matches!(
                    color_representation,
                    GaussianColorRepresentation::SphericalHarmonics { .. }
                )
                .then_some(self.sh_rest(idx).to_vec()),
                features: None,
                state: GaussianState::Stable,
            });
        }

        Ok(output)
    }

    #[deprecated(note = "Use HostSplats::to_legacy_gaussians(...) instead.")]
    pub fn to_gaussian_map(&self) -> candle_core::Result<GaussianMap> {
        let mut map = GaussianMap::from_gaussians(self.to_legacy_gaussians()?);
        map.update_states();
        Ok(map)
    }

    pub fn to_scene_gaussians(&self) -> candle_core::Result<Vec<Gaussian>> {
        self.validate()?;
        let mut gaussians = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let gaussian = match self.sh_degree {
                0 => Gaussian::new(
                    self.position(idx),
                    self.scale(idx),
                    self.rotation(idx),
                    sigmoid_scalar(self.opacity_logits[idx]).clamp(0.0, 1.0),
                    self.rgb_color(idx),
                ),
                _ => Gaussian::with_sh(
                    self.position(idx),
                    self.scale(idx),
                    self.rotation(idx),
                    sigmoid_scalar(self.opacity_logits[idx]).clamp(0.0, 1.0),
                    self.rgb_color(idx),
                    self.sh_rest(idx).to_vec(),
                ),
            };
            gaussians.push(gaussian);
        }
        Ok(gaussians)
    }

    pub fn to_scene_metadata(&self, iterations: usize, final_loss: f32) -> crate::SceneMetadata {
        crate::SceneMetadata {
            iterations,
            final_loss,
            gaussian_count: self.len(),
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

    pub(super) fn push_rgb(
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

    fn push_sh_coeffs(&mut self, sh_0: [f32; 3], sh_rest: &[f32]) {
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
) -> TrainableColorRepresentation {
    if config.training_profile == TrainingProfile::LiteGsMacV1 {
        TrainableColorRepresentation::SphericalHarmonics {
            degree: config.litegs.sh_degree,
        }
    } else {
        TrainableColorRepresentation::Rgb
    }
}

pub(super) fn infer_color_representation_from_legacy_gaussians(
    gaussians: &[Gaussian3D],
) -> candle_core::Result<TrainableColorRepresentation> {
    let mut inferred = TrainableColorRepresentation::Rgb;
    for gaussian in gaussians {
        match gaussian.color_representation {
            GaussianColorRepresentation::Rgb => {
                if inferred != TrainableColorRepresentation::Rgb {
                    candle_core::bail!(
                        "gaussian map mixes RGB and spherical harmonics color representations"
                    );
                }
            }
            GaussianColorRepresentation::SphericalHarmonics { degree } => {
                let next = TrainableColorRepresentation::SphericalHarmonics { degree };
                if inferred == TrainableColorRepresentation::Rgb {
                    inferred = next;
                } else if inferred != next {
                    candle_core::bail!(
                        "gaussian map mixes spherical harmonics degrees {} and {}",
                        inferred.sh_degree(),
                        degree
                    );
                }
            }
        }
    }
    Ok(inferred)
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

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::{
        infer_color_representation_from_legacy_gaussians, splat_color_representation_for_config,
        Splats,
    };
    use crate::core::{Gaussian3D, GaussianColorRepresentation};
    use crate::diff::diff_splat::{
        rgb_to_sh0_value, TrainableColorRepresentation, TrainableGaussians,
    };
    use crate::{LiteGsConfig, TrainingConfig, TrainingProfile};
    use candle_core::Device;
    use glam::{Quat, Vec3};

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
        let gaussians = TrainableGaussians::new(
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
        let gaussians = TrainableGaussians::new_with_sh(
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
    fn legacy_gaussians_round_trip_via_rgb_splats() {
        let device = Device::Cpu;
        let gaussians = vec![Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )];

        let splats =
            Splats::from_legacy_gaussians(&gaussians, TrainableColorRepresentation::Rgb).unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = Splats::from_runtime(&runtime)
            .unwrap()
            .to_legacy_gaussians()
            .unwrap();
        let gaussian = &rebuilt[0];

        assert_eq!(
            gaussian.color_representation,
            GaussianColorRepresentation::Rgb
        );
        assert!((gaussian.color[0] - 0.2).abs() < 1e-5);
        assert!((gaussian.color[1] - 0.4).abs() < 1e-5);
        assert!((gaussian.color[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn legacy_gaussians_round_trips_via_sh_splats() {
        let device = Device::Cpu;
        let gaussian = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 3 },
            Some([
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]),
            Some(vec![0.5; 15 * 3]),
        );
        let gaussians = vec![gaussian];

        let splats = Splats::from_legacy_gaussians(
            &gaussians,
            TrainableColorRepresentation::SphericalHarmonics { degree: 3 },
        )
        .unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = Splats::from_runtime(&runtime)
            .unwrap()
            .to_legacy_gaussians()
            .unwrap();
        let gaussian = &rebuilt[0];

        assert_eq!(
            gaussian.color_representation,
            GaussianColorRepresentation::SphericalHarmonics { degree: 3 }
        );
        assert_eq!(
            gaussian.sh_dc.unwrap(),
            [
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]
        );
        assert_eq!(gaussian.sh_rest.as_deref().unwrap(), &vec![0.5; 15 * 3]);
    }

    #[test]
    fn config_selects_internal_splat_color_representation() {
        let legacy = splat_color_representation_for_config(&TrainingConfig::default());
        assert_eq!(legacy, TrainableColorRepresentation::Rgb);

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
            TrainableColorRepresentation::SphericalHarmonics { degree: 3 }
        );
    }

    #[test]
    fn inferred_color_representation_tracks_sh_degree_from_legacy_gaussians() {
        let gaussian = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 3 },
            Some([
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]),
            Some(vec![0.5; 15 * 3]),
        );
        let gaussians = vec![gaussian];

        let representation = infer_color_representation_from_legacy_gaussians(&gaussians).unwrap();

        assert_eq!(
            representation,
            TrainableColorRepresentation::SphericalHarmonics { degree: 3 }
        );
    }

    #[test]
    fn inferred_color_representation_rejects_mixed_sh_degrees() {
        let sh_degree_2 = Gaussian3D::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 2 },
            Some([
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]),
            Some(vec![0.5; 8 * 3]),
        );
        let sh_degree_3 = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 3 },
            Some([
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]),
            Some(vec![0.5; 15 * 3]),
        );
        let gaussians = vec![sh_degree_2, sh_degree_3];

        let err = infer_color_representation_from_legacy_gaussians(&gaussians)
            .unwrap_err()
            .to_string();

        assert!(err.contains("mixes spherical harmonics degrees"));
    }

    #[test]
    fn inferred_splats_round_trip_to_scene_gaussians_with_sh() {
        let gaussian = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )
        .with_color_state(
            GaussianColorRepresentation::SphericalHarmonics { degree: 3 },
            Some([
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ]),
            Some(vec![0.5; 15 * 3]),
        );
        let gaussians = vec![gaussian];

        let splats = Splats::from_legacy_gaussians_inferred(&gaussians).unwrap();
        let scene = splats.to_scene_gaussians().unwrap();
        let metadata = splats.to_scene_metadata(7, 0.5);

        assert_eq!(metadata.iterations, 7);
        assert_eq!(metadata.final_loss, 0.5);
        assert_eq!(metadata.sh_degree, 3);
        assert_eq!(scene.len(), 1);
        assert_eq!(scene[0].sh_rest.as_deref().unwrap(), &vec![0.5; 15 * 3]);
        assert!((scene[0].color[0] - 0.2).abs() < 1e-5);
        assert!((scene[0].color[1] - 0.4).abs() < 1e-5);
        assert!((scene[0].color[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn scene_extent_tracks_radius_from_splats_positions() {
        let gaussians = vec![
            Gaussian3D::new(
                Vec3::new(-2.0, 0.0, 0.0),
                Vec3::new(0.1, 0.2, 0.3),
                Quat::IDENTITY,
                0.25,
                [0.2, 0.4, 0.6],
            ),
            Gaussian3D::new(
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.1, 0.2, 0.3),
                Quat::IDENTITY,
                0.25,
                [0.2, 0.4, 0.6],
            ),
        ];

        let splats = Splats::from_legacy_gaussians_inferred(&gaussians).unwrap();

        assert!((splats.scene_extent() - 2.0).abs() < 1e-6);
        assert_eq!(
            splats.positions_vec3(),
            vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        );
    }
}
