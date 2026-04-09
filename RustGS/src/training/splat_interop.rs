use crate::diff::diff_splat::{rgb_to_sh0_value, SplatColorRepresentation};
use crate::legacy::{Gaussian3D, GaussianColorRepresentation, GaussianMap, GaussianState};
use crate::Gaussian;

use super::splats::{
    opacity_to_logit, sigmoid_scalar, splat_color_representation_for_config, HostSplats,
};
use super::TrainingConfig;

impl HostSplats {
    pub fn from_legacy_gaussians(
        gaussians: &[Gaussian3D],
        color_representation: SplatColorRepresentation,
    ) -> candle_core::Result<Self> {
        let sh_degree = color_representation.sh_degree();
        let mut splats = Self::with_sh_degree_capacity(sh_degree, gaussians.len());

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
                SplatColorRepresentation::Rgb => {
                    splats.push_sh_coeffs(gaussian.color.map(rgb_to_sh0_value), &[]);
                }
                SplatColorRepresentation::SphericalHarmonics { degree } => {
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
        color_representation: SplatColorRepresentation,
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
        let mut splats = Self::with_sh_degree_capacity(sh_degree, scene.len());
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
        }

        splats.validate()?;
        Ok(splats)
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
            sh_degree: self.sh_degree(),
        }
    }
}

pub(super) fn infer_color_representation_from_legacy_gaussians(
    gaussians: &[Gaussian3D],
) -> candle_core::Result<SplatColorRepresentation> {
    let mut inferred = SplatColorRepresentation::Rgb;
    for gaussian in gaussians {
        match gaussian.color_representation {
            GaussianColorRepresentation::Rgb => {
                if inferred != SplatColorRepresentation::Rgb {
                    candle_core::bail!(
                        "gaussian map mixes RGB and spherical harmonics color representations"
                    );
                }
            }
            GaussianColorRepresentation::SphericalHarmonics { degree } => {
                let next = SplatColorRepresentation::SphericalHarmonics { degree };
                if inferred == SplatColorRepresentation::Rgb {
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

#[cfg(test)]
mod tests {
    use super::{infer_color_representation_from_legacy_gaussians, HostSplats};
    use crate::diff::diff_splat::{rgb_to_sh0_value, SplatColorRepresentation};
    use crate::legacy::{Gaussian3D, GaussianColorRepresentation};
    use candle_core::Device;
    use glam::{Quat, Vec3};

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
            HostSplats::from_legacy_gaussians(&gaussians, SplatColorRepresentation::Rgb).unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = HostSplats::from_runtime(&runtime)
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

        let splats = HostSplats::from_legacy_gaussians(
            &gaussians,
            SplatColorRepresentation::SphericalHarmonics { degree: 3 },
        )
        .unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = HostSplats::from_runtime(&runtime)
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
            SplatColorRepresentation::SphericalHarmonics { degree: 3 }
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

        let splats = HostSplats::from_legacy_gaussians_inferred(&gaussians).unwrap();
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

        let splats = HostSplats::from_legacy_gaussians_inferred(&gaussians).unwrap();

        assert!((splats.scene_extent() - 2.0).abs() < 1e-6);
        assert_eq!(
            splats.positions_vec3(),
            vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        );
    }
}
