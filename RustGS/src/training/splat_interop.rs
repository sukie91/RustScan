use crate::diff::diff_splat::rgb_to_sh0_value;
use crate::Gaussian;

use super::splats::{opacity_to_logit, sigmoid_scalar, HostSplats};

impl HostSplats {
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

#[cfg(test)]
mod tests {
    use super::HostSplats;
    use crate::Gaussian;
    use candle_core::Device;

    #[test]
    fn scene_gaussians_round_trip_via_rgb_splats() {
        let device = Device::Cpu;
        let gaussians = vec![Gaussian::new(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            0.25,
            [0.2, 0.4, 0.6],
        )];

        let splats = HostSplats::from_scene_gaussians(&gaussians, 0).unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = HostSplats::from_runtime(&runtime)
            .unwrap()
            .to_scene_gaussians()
            .unwrap();
        let gaussian = &rebuilt[0];

        assert!((gaussian.color[0] - 0.2).abs() < 1e-5);
        assert!((gaussian.color[1] - 0.4).abs() < 1e-5);
        assert!((gaussian.color[2] - 0.6).abs() < 1e-5);
        assert!(gaussian.sh_rest.is_none());
    }

    #[test]
    fn scene_gaussians_round_trip_via_sh_splats() {
        let device = Device::Cpu;
        let gaussians = vec![Gaussian::with_sh(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            0.25,
            [0.2, 0.4, 0.6],
            vec![0.5; 15 * 3],
        )];

        let splats = HostSplats::from_scene_gaussians(&gaussians, 3).unwrap();
        let runtime = splats.to_runtime(&device).unwrap();
        let rebuilt = HostSplats::from_runtime(&runtime)
            .unwrap()
            .to_scene_gaussians()
            .unwrap();
        let gaussian = &rebuilt[0];

        assert_eq!(gaussian.sh_rest.as_deref().unwrap(), &vec![0.5; 15 * 3]);
        assert!((gaussian.color[0] - 0.2).abs() < 1e-5);
        assert!((gaussian.color[1] - 0.4).abs() < 1e-5);
        assert!((gaussian.color[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn scene_extent_tracks_radius_from_splats_positions() {
        let gaussians = vec![
            Gaussian::new(
                [-2.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [1.0, 0.0, 0.0, 0.0],
                0.25,
                [0.2, 0.4, 0.6],
            ),
            Gaussian::new(
                [2.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [1.0, 0.0, 0.0, 0.0],
                0.25,
                [0.2, 0.4, 0.6],
            ),
        ];

        let splats = HostSplats::from_scene_gaussians(&gaussians, 0).unwrap();

        assert!((splats.scene_extent() - 2.0).abs() < 1e-6);
        assert_eq!(
            splats.positions_vec3(),
            vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        );
    }
}
