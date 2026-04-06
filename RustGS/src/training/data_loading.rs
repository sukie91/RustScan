#[cfg(feature = "gpu")]
use super::frame_loader::{ordered_frame_indices, FrameLoaderOptions, PrefetchFrameLoader};
#[cfg(feature = "gpu")]
use super::init_map::build_initial_map;
#[cfg(feature = "gpu")]
use crate::core::{Gaussian3D, GaussianColorRepresentation, GaussianMap, GaussianState};
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::{
    rgb_to_sh0_value, DiffCamera, TrainableColorRepresentation, TrainableGaussians,
};
#[cfg(feature = "gpu")]
use crate::{TrainingDataset, TrainingError, SE3};
#[cfg(feature = "gpu")]
use candle_core::Device;

#[cfg(feature = "gpu")]
pub(super) struct FrameSample {
    pub(super) camera: DiffCamera,
    pub(super) color_u8: Vec<u8>,
    pub(super) color_f32: Vec<f32>,
    pub(super) depth: Vec<f32>,
    pub(super) rotation: [[f32; 3]; 3],
    pub(super) translation: [f32; 3],
}

#[cfg(feature = "gpu")]
pub(crate) struct LoadedTrainingData {
    pub cameras: Vec<DiffCamera>,
    pub colors: Vec<Vec<f32>>,
    pub depths: Vec<Vec<f32>>,
    pub initial_map: GaussianMap,
}

#[cfg(feature = "gpu")]
pub(crate) fn load_training_data(
    dataset: &TrainingDataset,
    config: &super::TrainingConfig,
    device: &Device,
) -> Result<LoadedTrainingData, TrainingError> {
    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training dataset does not contain any poses".to_string(),
        ));
    }

    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;

    let mut frames = Vec::with_capacity(dataset.poses.len());
    let mut loader = PrefetchFrameLoader::new(
        dataset,
        config,
        FrameLoaderOptions {
            cache_capacity: config.frame_cache_capacity,
            prefetch_ahead: config.frame_prefetch_ahead,
        },
    )?;
    let frame_order = ordered_frame_indices(dataset.poses.len(), 1, config.frame_shuffle_seed);
    let mut real_depth_frames = 0usize;
    for (cursor, &pose_idx) in frame_order.iter().enumerate() {
        loader.prefetch_order_window(&frame_order, cursor)?;
        let pose = &dataset.poses[pose_idx];
        let decoded = loader.get(pose_idx)?;
        real_depth_frames += usize::from(decoded.used_real_depth);

        let rotation = pose.pose.rotation();
        let translation = pose.pose.translation();
        let camera = diff_camera_from_scene_pose(
            &pose.pose,
            dataset.intrinsics.fx,
            dataset.intrinsics.fy,
            dataset.intrinsics.cx,
            dataset.intrinsics.cy,
            width,
            height,
            device,
        )?;

        frames.push(FrameSample {
            camera,
            color_f32: decoded.color_u8.iter().map(|v| *v as f32 / 255.0).collect(),
            color_u8: decoded.color_u8.clone(),
            depth: decoded.depth.clone(),
            rotation,
            translation,
        });
    }

    let initial_map = build_initial_map(dataset, &frames, config)?;

    let mut cameras = Vec::with_capacity(frames.len());
    let mut colors = Vec::with_capacity(frames.len());
    let mut depths = Vec::with_capacity(frames.len());
    for frame in frames {
        cameras.push(frame.camera);
        colors.push(frame.color_f32);
        depths.push(frame.depth);
    }

    if real_depth_frames == 0 && !config.use_synthetic_depth {
        log::info!(
            "Training dataset does not provide depth supervision; optimizing RGB loss only."
        );
    }

    Ok(LoadedTrainingData {
        cameras,
        colors,
        depths,
        initial_map,
    })
}

#[cfg(feature = "gpu")]
fn diff_camera_from_scene_pose(
    pose: &SE3,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: usize,
    height: usize,
    device: &Device,
) -> candle_core::Result<DiffCamera> {
    // ScenePose stores camera-to-world poses so they can be reused for
    // backprojection during Gaussian initialization. The differentiable
    // renderer, however, expects world-to-camera extrinsics.
    let view_pose = pose.inverse();
    let rotation = view_pose.rotation();
    let translation = view_pose.translation();

    DiffCamera::new(
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        &rotation,
        &translation,
        device,
    )
}

#[cfg(feature = "gpu")]
pub(crate) fn trainable_from_map(
    map: &GaussianMap,
    device: &Device,
    config: &super::TrainingConfig,
) -> candle_core::Result<TrainableGaussians> {
    let mut positions = Vec::with_capacity(map.len() * 3);
    let mut scales = Vec::with_capacity(map.len() * 3);
    let mut rotations = Vec::with_capacity(map.len() * 4);
    let mut opacities = Vec::with_capacity(map.len());
    let mut base_color_params = Vec::with_capacity(map.len() * 3);
    let use_litegs_sh = config.training_profile == super::TrainingProfile::LiteGsMacV1;
    let sh_degree = if use_litegs_sh {
        config.litegs.sh_degree
    } else {
        0
    };
    let sh_rest_coeff_count = if use_litegs_sh {
        super::super::diff::diff_splat::sh_coeff_count_for_degree(sh_degree).saturating_sub(1)
    } else {
        0
    };
    let mut sh_rest = if use_litegs_sh {
        Vec::with_capacity(map.len() * sh_rest_coeff_count * 3)
    } else {
        Vec::new()
    };

    for gaussian in map.gaussians() {
        positions.extend_from_slice(&[
            gaussian.position.x,
            gaussian.position.y,
            gaussian.position.z,
        ]);
        scales.extend_from_slice(&[
            gaussian.scale.x.max(1e-6).ln(),
            gaussian.scale.y.max(1e-6).ln(),
            gaussian.scale.z.max(1e-6).ln(),
        ]);
        rotations.extend_from_slice(&[
            gaussian.rotation.w,
            gaussian.rotation.x,
            gaussian.rotation.y,
            gaussian.rotation.z,
        ]);
        opacities.push(opacity_to_logit(gaussian.opacity));
        if use_litegs_sh {
            match gaussian.color_representation {
                GaussianColorRepresentation::Rgb => {
                    base_color_params.extend(gaussian.color.iter().copied().map(rgb_to_sh0_value));
                    sh_rest.resize(sh_rest.len() + sh_rest_coeff_count * 3, 0.0);
                }
                GaussianColorRepresentation::SphericalHarmonics { degree } => {
                    if degree != sh_degree {
                        candle_core::bail!(
                            "gaussian map mixes SH degree {degree} with requested degree {sh_degree}"
                        );
                    }
                    base_color_params.extend_from_slice(
                        &gaussian
                            .sh_dc
                            .unwrap_or_else(|| gaussian.color.map(rgb_to_sh0_value)),
                    );
                    let source_rest = gaussian.sh_rest.as_deref().unwrap_or(&[]);
                    let expected = sh_rest_coeff_count * 3;
                    let copied = source_rest.len().min(expected);
                    sh_rest.extend_from_slice(&source_rest[..copied]);
                    if copied < expected {
                        sh_rest.resize(sh_rest.len() + (expected - copied), 0.0);
                    }
                }
            }
        } else {
            base_color_params.extend_from_slice(&gaussian.color);
        }
    }

    if use_litegs_sh {
        TrainableGaussians::new_with_sh(
            &positions,
            &scales,
            &rotations,
            &opacities,
            &base_color_params,
            &sh_rest,
            sh_degree,
            device,
        )
    } else {
        TrainableGaussians::new(
            &positions,
            &scales,
            &rotations,
            &opacities,
            &base_color_params,
            device,
        )
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn map_from_trainable(
    gaussians: &TrainableGaussians,
) -> candle_core::Result<GaussianMap> {
    let positions = gaussians.positions().to_vec2::<f32>()?;
    let scales = gaussians.scales()?.to_vec2::<f32>()?;
    let rotations = gaussians.rotations()?.to_vec2::<f32>()?;
    let opacities = gaussians.opacities()?.to_vec1::<f32>()?;
    let colors = gaussians.render_colors()?.to_vec2::<f32>()?;
    let base_color_params = gaussians.colors().to_vec2::<f32>()?;
    let sh_rest = gaussians.sh_rest().to_vec3::<f32>()?;
    let color_representation = match gaussians.color_representation() {
        TrainableColorRepresentation::Rgb => GaussianColorRepresentation::Rgb,
        TrainableColorRepresentation::SphericalHarmonics { degree } => {
            GaussianColorRepresentation::SphericalHarmonics { degree }
        }
    };

    let mut output = Vec::with_capacity(gaussians.len());
    for idx in 0..gaussians.len() {
        output.push(Gaussian3D {
            position: glam::Vec3::new(positions[idx][0], positions[idx][1], positions[idx][2]),
            scale: glam::Vec3::new(scales[idx][0], scales[idx][1], scales[idx][2]),
            rotation: glam::Quat::from_xyzw(
                rotations[idx][1],
                rotations[idx][2],
                rotations[idx][3],
                rotations[idx][0],
            ),
            opacity: opacities[idx].clamp(0.0, 1.0),
            color: [colors[idx][0], colors[idx][1], colors[idx][2]],
            color_representation,
            sh_dc: matches!(
                color_representation,
                GaussianColorRepresentation::SphericalHarmonics { .. }
            )
            .then_some([
                base_color_params[idx][0],
                base_color_params[idx][1],
                base_color_params[idx][2],
            ]),
            sh_rest: matches!(
                color_representation,
                GaussianColorRepresentation::SphericalHarmonics { .. }
            )
            .then_some(sh_rest[idx].iter().flatten().copied().collect::<Vec<f32>>()),
            features: None,
            state: GaussianState::Stable,
        });
    }

    let mut map = GaussianMap::from_gaussians(output);
    map.update_states();
    Ok(map)
}

#[cfg(feature = "gpu")]
fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_diff_camera_uses_world_to_camera_view_pose() {
        let device = Device::Cpu;
        let pose = crate::SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, -2.0, 3.5]);

        let camera =
            diff_camera_from_scene_pose(&pose, 500.0, 510.0, 320.0, 240.0, 640, 480, &device)
                .unwrap();

        assert_eq!(
            camera.rotation,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert!((camera.translation[0] + 1.0).abs() < 1e-6);
        assert!((camera.translation[1] - 2.0).abs() < 1e-6);
        assert!((camera.translation[2] + 3.5).abs() < 1e-6);
    }

    #[test]
    fn litegs_profile_initializes_trainable_gaussians_with_sh_dc() {
        let device = Device::Cpu;
        let mut map = GaussianMap::new(1);
        let gaussian = Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [64, 128, 255]);
        let _ = map.add(gaussian);
        map.update_states();

        let config = super::super::TrainingConfig {
            training_profile: super::super::TrainingProfile::LiteGsMacV1,
            ..super::super::TrainingConfig::default()
        };
        let trainable = trainable_from_map(&map, &device, &config).unwrap();

        assert!(trainable.uses_spherical_harmonics());
        assert_eq!(trainable.sh_degree(), 3);
        assert_eq!(trainable.sh_rest().dims(), &[1, 15, 3]);

        let rendered_rgb = trainable.render_colors().unwrap().to_vec2::<f32>().unwrap();
        assert!((rendered_rgb[0][0] - (64.0 / 255.0)).abs() < 1e-5);
        assert!((rendered_rgb[0][1] - (128.0 / 255.0)).abs() < 1e-5);
        assert!((rendered_rgb[0][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn map_export_restores_rgb_from_sh_dc_trainables() {
        let device = Device::Cpu;
        let trainable = TrainableGaussians::new_with_sh(
            &[0.0, 0.0, 1.0],
            &[0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[opacity_to_logit(0.25)],
            &[
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.6),
            ],
            &vec![0.0; 15 * 3],
            3,
            &device,
        )
        .unwrap();

        let map = map_from_trainable(&trainable).unwrap();
        let gaussian = &map.gaussians()[0];
        assert!((gaussian.color[0] - 0.2).abs() < 1e-5);
        assert!((gaussian.color[1] - 0.4).abs() < 1e-5);
        assert!((gaussian.color[2] - 0.6).abs() < 1e-5);
    }
}
