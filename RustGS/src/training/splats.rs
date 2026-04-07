use candle_core::Device;

use crate::core::{Gaussian3D, GaussianColorRepresentation, GaussianMap, GaussianState};
use crate::diff::diff_splat::{rgb_to_sh0_value, TrainableColorRepresentation, TrainableGaussians};
use crate::Gaussian;

use super::{TrainingConfig, TrainingProfile};

#[derive(Debug, Clone)]
pub(super) struct Splats {
    pub(super) positions: Vec<f32>,
    pub(super) log_scales: Vec<f32>,
    pub(super) rotations: Vec<f32>,
    pub(super) opacity_logits: Vec<f32>,
    pub(super) colors: Vec<f32>,
    pub(super) sh_rest: Vec<f32>,
    pub(super) color_representation: TrainableColorRepresentation,
}

impl Splats {
    pub(super) fn from_trainable(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        let splats = Self {
            positions: flatten_rows(gaussians.positions().to_vec2::<f32>()?),
            log_scales: flatten_rows(gaussians.scales.as_tensor().to_vec2::<f32>()?),
            rotations: flatten_rows(gaussians.rotations.as_tensor().to_vec2::<f32>()?),
            opacity_logits: gaussians.opacities.as_tensor().to_vec1::<f32>()?,
            colors: flatten_rows(gaussians.colors().to_vec2::<f32>()?),
            sh_rest: flatten_3d(gaussians.sh_rest().to_vec3::<f32>()?),
            color_representation: gaussians.color_representation(),
        };
        splats.validate()?;
        Ok(splats)
    }

    pub(super) fn from_gaussian_map(
        map: &GaussianMap,
        color_representation: TrainableColorRepresentation,
    ) -> candle_core::Result<Self> {
        let mut splats = Self {
            positions: Vec::with_capacity(map.len() * 3),
            log_scales: Vec::with_capacity(map.len() * 3),
            rotations: Vec::with_capacity(map.len() * 4),
            opacity_logits: Vec::with_capacity(map.len()),
            colors: Vec::with_capacity(map.len() * 3),
            sh_rest: Vec::with_capacity(map.len() * color_representation.sh_rest_coeff_count() * 3),
            color_representation,
        };

        for gaussian in map.gaussians() {
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
                    splats.colors.extend_from_slice(&gaussian.color);
                }
                TrainableColorRepresentation::SphericalHarmonics { degree } => {
                    match gaussian.color_representation {
                        GaussianColorRepresentation::Rgb => {
                            splats
                                .colors
                                .extend(gaussian.color.iter().copied().map(rgb_to_sh0_value));
                            splats
                                .sh_rest
                                .resize(splats.sh_rest.len() + splats.sh_rest_row_width(), 0.0);
                        }
                        GaussianColorRepresentation::SphericalHarmonics {
                            degree: stored_degree,
                        } => {
                            if stored_degree != degree {
                                candle_core::bail!(
                                    "gaussian map mixes SH degree {stored_degree} with requested degree {degree}"
                                );
                            }
                            splats.colors.extend_from_slice(
                                &gaussian
                                    .sh_dc
                                    .unwrap_or_else(|| gaussian.color.map(rgb_to_sh0_value)),
                            );
                            let source_rest = gaussian.sh_rest.as_deref().unwrap_or(&[]);
                            let expected = splats.sh_rest_row_width();
                            let copied = source_rest.len().min(expected);
                            splats.sh_rest.extend_from_slice(&source_rest[..copied]);
                            if copied < expected {
                                splats
                                    .sh_rest
                                    .resize(splats.sh_rest.len() + (expected - copied), 0.0);
                            }
                        }
                    }
                }
            }
        }

        splats.validate()?;
        Ok(splats)
    }

    pub(super) fn from_scene_gaussians(
        scene: &[Gaussian],
        sh_degree: usize,
    ) -> candle_core::Result<Self> {
        let color_representation = if sh_degree > 0 {
            TrainableColorRepresentation::SphericalHarmonics { degree: sh_degree }
        } else {
            TrainableColorRepresentation::Rgb
        };
        let sh_rest_row_width = color_representation.sh_rest_coeff_count() * 3;
        let mut splats = Self {
            positions: Vec::with_capacity(scene.len() * 3),
            log_scales: Vec::with_capacity(scene.len() * 3),
            rotations: Vec::with_capacity(scene.len() * 4),
            opacity_logits: Vec::with_capacity(scene.len()),
            colors: Vec::with_capacity(scene.len() * 3),
            sh_rest: Vec::with_capacity(scene.len() * sh_rest_row_width),
            color_representation,
        };

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
            splats.colors.extend_from_slice(&gaussian.color);

            if sh_rest_row_width > 0 {
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
                        splats.sh_rest.extend_from_slice(values);
                    }
                    None => splats
                        .sh_rest
                        .extend(std::iter::repeat_n(0.0, sh_rest_row_width)),
                }
            }
        }

        splats.validate()?;
        Ok(splats)
    }

    pub(super) fn to_trainable(&self, device: &Device) -> candle_core::Result<TrainableGaussians> {
        self.validate()?;
        match self.color_representation {
            TrainableColorRepresentation::Rgb => TrainableGaussians::new(
                &self.positions,
                &self.log_scales,
                &self.rotations,
                &self.opacity_logits,
                &self.colors,
                device,
            ),
            TrainableColorRepresentation::SphericalHarmonics { degree } => {
                TrainableGaussians::new_with_sh(
                    &self.positions,
                    &self.log_scales,
                    &self.rotations,
                    &self.opacity_logits,
                    &self.colors,
                    &self.sh_rest,
                    degree,
                    device,
                )
            }
        }
    }

    pub(super) fn to_gaussian_map(&self) -> candle_core::Result<GaussianMap> {
        self.validate()?;
        let color_representation = match self.color_representation {
            TrainableColorRepresentation::Rgb => GaussianColorRepresentation::Rgb,
            TrainableColorRepresentation::SphericalHarmonics { degree } => {
                GaussianColorRepresentation::SphericalHarmonics { degree }
            }
        };

        let mut output = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let scale = self.scale(idx);
            let rotation = self.rotation(idx);
            let color = self.color(idx);
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
                .then_some(color),
                sh_rest: matches!(
                    color_representation,
                    GaussianColorRepresentation::SphericalHarmonics { .. }
                )
                .then_some(self.sh_rest(idx).to_vec()),
                features: None,
                state: GaussianState::Stable,
            });
        }

        let mut map = GaussianMap::from_gaussians(output);
        map.update_states();
        Ok(map)
    }

    pub(super) fn validate(&self) -> candle_core::Result<()> {
        let row_count = self.opacity_logits.len();
        validate_component_len("positions", self.positions.len(), row_count, 3)?;
        validate_component_len("log_scales", self.log_scales.len(), row_count, 3)?;
        validate_component_len("rotations", self.rotations.len(), row_count, 4)?;
        validate_component_len("colors", self.colors.len(), row_count, 3)?;
        validate_component_len(
            "sh_rest",
            self.sh_rest.len(),
            row_count,
            self.sh_rest_row_width(),
        )?;
        Ok(())
    }

    pub(super) fn len(&self) -> usize {
        self.opacity_logits.len()
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

    pub(super) fn color(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.colors[base],
            self.colors[base + 1],
            self.colors[base + 2],
        ]
    }

    pub(super) fn sh_rest_row_width(&self) -> usize {
        self.color_representation.sh_rest_coeff_count() * 3
    }

    pub(super) fn sh_rest(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_rest, self.sh_rest_row_width(), idx)
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
        color: [f32; 3],
        sh_rest: &[f32],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.colors.extend_from_slice(&color);
        let sh_rest_row_width = self.sh_rest_row_width();
        if sh_rest_row_width > 0 {
            let copied = sh_rest.len().min(sh_rest_row_width);
            self.sh_rest.extend_from_slice(&sh_rest[..copied]);
            if copied < sh_rest_row_width {
                self.sh_rest
                    .resize(self.sh_rest.len() + (sh_rest_row_width - copied), 0.0);
            }
        }
    }

    #[allow(dead_code)]
    pub(super) fn retained_view(&self, row_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(row_count * 3),
            log_scales: Vec::with_capacity(row_count * 3),
            rotations: Vec::with_capacity(row_count * 4),
            opacity_logits: Vec::with_capacity(row_count),
            colors: Vec::with_capacity(row_count * 3),
            sh_rest: Vec::with_capacity(row_count * self.sh_rest_row_width()),
            color_representation: self.color_representation,
        }
    }
}

pub(super) fn trainable_color_representation_for_config(
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
    use super::{trainable_color_representation_for_config, Splats};
    use crate::core::{Gaussian3D, GaussianColorRepresentation, GaussianMap};
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
            colors: vec![0.0; 6],
            sh_rest: Vec::new(),
            color_representation: TrainableColorRepresentation::Rgb,
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

        let splats = Splats::from_trainable(&gaussians).unwrap();
        let rebuilt = splats.to_trainable(&device).unwrap();

        assert_eq!(
            rebuilt.positions().to_vec2::<f32>().unwrap(),
            gaussians.positions().to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            rebuilt.colors().to_vec2::<f32>().unwrap(),
            gaussians.colors().to_vec2::<f32>().unwrap()
        );
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

        let splats = Splats::from_trainable(&gaussians).unwrap();
        let rebuilt = splats.to_trainable(&device).unwrap();

        assert!(rebuilt.uses_spherical_harmonics());
        assert_eq!(rebuilt.sh_degree(), 3);
        assert_eq!(
            rebuilt.sh_rest().to_vec3::<f32>().unwrap(),
            gaussians.sh_rest().to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn gaussian_map_round_trips_via_rgb_splats() {
        let device = Device::Cpu;
        let map = GaussianMap::from_gaussians(vec![Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.25,
            [0.2, 0.4, 0.6],
        )]);

        let splats = Splats::from_gaussian_map(&map, TrainableColorRepresentation::Rgb).unwrap();
        let trainable = splats.to_trainable(&device).unwrap();
        let rebuilt = Splats::from_trainable(&trainable)
            .unwrap()
            .to_gaussian_map()
            .unwrap();
        let gaussian = &rebuilt.gaussians()[0];

        assert_eq!(
            gaussian.color_representation,
            GaussianColorRepresentation::Rgb
        );
        assert!((gaussian.color[0] - 0.2).abs() < 1e-5);
        assert!((gaussian.color[1] - 0.4).abs() < 1e-5);
        assert!((gaussian.color[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn gaussian_map_round_trips_via_sh_splats() {
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
        let map = GaussianMap::from_gaussians(vec![gaussian]);

        let splats = Splats::from_gaussian_map(
            &map,
            TrainableColorRepresentation::SphericalHarmonics { degree: 3 },
        )
        .unwrap();
        let trainable = splats.to_trainable(&device).unwrap();
        let rebuilt = Splats::from_trainable(&trainable)
            .unwrap()
            .to_gaussian_map()
            .unwrap();
        let gaussian = &rebuilt.gaussians()[0];

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
    fn config_selects_internal_trainable_color_representation() {
        let legacy = trainable_color_representation_for_config(&TrainingConfig::default());
        assert_eq!(legacy, TrainableColorRepresentation::Rgb);

        let litegs = trainable_color_representation_for_config(&TrainingConfig {
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
}
