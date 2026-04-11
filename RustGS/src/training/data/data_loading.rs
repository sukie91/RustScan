#[cfg(feature = "gpu")]
use super::frame_loader::{ordered_frame_indices, FrameLoaderOptions, PrefetchFrameLoader};
#[cfg(feature = "gpu")]
use super::frame_targets::{resize_depth, resize_rgb_u8_to_f32};
#[cfg(feature = "gpu")]
use super::init_map::build_initial_splats;
#[cfg(feature = "gpu")]
use super::splats::HostSplats;
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::DiffCamera;
#[cfg(feature = "gpu")]
use crate::training::pose_utils::se3_rotation_row_major;
#[cfg(feature = "gpu")]
use crate::{TrainingDataset, TrainingError, SE3};
#[cfg(feature = "gpu")]
use candle_core::Device;

#[cfg(feature = "gpu")]
pub(crate) struct LoadedTrainingData {
    pub cameras: Vec<DiffCamera>,
    pub colors: Vec<Vec<f32>>,
    pub depths: Vec<Vec<f32>>,
    pub target_width: usize,
    pub target_height: usize,
    pub initial_splats: HostSplats,
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
    let (target_width, target_height) =
        super::scaled_dimensions(width, height, config.metal_render_scale);

    let mut loader = PrefetchFrameLoader::new(
        dataset,
        config,
        FrameLoaderOptions {
            cache_capacity: config.frame_cache_capacity,
            prefetch_ahead: config.frame_prefetch_ahead,
        },
    )?;
    let mut cameras = Vec::with_capacity(dataset.poses.len());
    let mut colors = Vec::with_capacity(dataset.poses.len());
    let mut depths = Vec::with_capacity(dataset.poses.len());
    let frame_order = ordered_frame_indices(dataset.poses.len(), 1, config.frame_shuffle_seed);
    let mut real_depth_frames = 0usize;
    for (cursor, &pose_idx) in frame_order.iter().enumerate() {
        loader.prefetch_order_window(&frame_order, cursor)?;
        let pose = &dataset.poses[pose_idx];
        let decoded = loader.get(pose_idx)?;
        real_depth_frames += usize::from(decoded.used_real_depth);

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

        cameras.push(camera);
        colors.push(resize_rgb_u8_to_f32(
            &decoded.color_u8,
            width,
            height,
            target_width,
            target_height,
        ));
        depths.push(resize_depth(
            &decoded.depth,
            width,
            height,
            target_width,
            target_height,
        ));
    }

    log::info!(
        "Initializing gaussians from dataset sparse points | poses={} | sparse_points={} | max_initial_gaussians={}",
        dataset.poses.len(),
        dataset.initial_points.len(),
        config.max_initial_gaussians,
    );
    let initial_splats = build_initial_splats(dataset, config)?;

    if real_depth_frames == 0 && !config.use_synthetic_depth {
        log::info!(
            "Training dataset does not provide depth supervision; optimizing RGB loss only."
        );
    }

    Ok(LoadedTrainingData {
        cameras,
        colors,
        depths,
        target_width,
        target_height,
        initial_splats,
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
    let rotation = se3_rotation_row_major(&view_pose);
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

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::diff::diff_splat::{rgb_to_sh0_value, Splats};
    use crate::training::splats::HostSplats;
    use crate::{TrainingConfig, TrainingProfile};

    fn test_opacity_to_logit(opacity: f32) -> f32 {
        let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
        (clamped / (1.0 - clamped)).ln()
    }

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
    fn test_diff_camera_keeps_world_to_camera_rotation_row_major() {
        let device = Device::Cpu;
        let pose =
            crate::SE3::from_axis_angle(&[0.0, 0.0, std::f32::consts::FRAC_PI_2], &[0.0, 0.0, 0.0]);

        let camera =
            diff_camera_from_scene_pose(&pose, 500.0, 510.0, 320.0, 240.0, 640, 480, &device)
                .unwrap();

        let expected = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (camera.rotation[row][col] - expected[row][col]).abs() < 1e-6,
                    "row={row} col={col} actual={} expected={}",
                    camera.rotation[row][col],
                    expected[row][col]
                );
            }
        }
    }

    #[test]
    fn litegs_profile_initializes_trainable_gaussians_with_sh_dc() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        };
        let mut splats = HostSplats::with_sh_degree_capacity(
            super::super::splats::splat_color_representation_for_config(&config).sh_degree(),
            1,
        );
        splats.push_rgb(
            [0.0, 0.0, 1.0],
            [0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
            [1.0, 0.0, 0.0, 0.0],
            test_opacity_to_logit(0.5),
            [64.0 / 255.0, 128.0 / 255.0, 1.0],
        );
        let runtime = splats.upload(&device).unwrap();

        assert!(runtime.uses_spherical_harmonics());
        assert_eq!(runtime.sh_degree(), 3);
        assert_eq!(runtime.sh_rest().dims(), &[1, 15, 3]);

        let rendered_rgb = runtime.render_colors().unwrap().to_vec2::<f32>().unwrap();
        assert!((rendered_rgb[0][0] - (64.0 / 255.0)).abs() < 1e-5);
        assert!((rendered_rgb[0][1] - (128.0 / 255.0)).abs() < 1e-5);
        assert!((rendered_rgb[0][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn map_export_restores_rgb_from_sh_dc_trainables() {
        let device = Device::Cpu;
        let trainable = Splats::new_with_sh(
            &[0.0, 0.0, 1.0],
            &[0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[test_opacity_to_logit(0.25)],
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

        let splats = HostSplats::from_runtime(&trainable).unwrap();
        let color = splats.rgb_color(0);
        assert!((color[0] - 0.2).abs() < 1e-5);
        assert!((color[1] - 0.4).abs() < 1e-5);
        assert!((color[2] - 0.6).abs() < 1e-5);
    }
}
