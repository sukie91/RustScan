use crate::init::{initialize_gaussians_from_points, GaussianInitConfig};
use crate::{TrainingDataset, TrainingError};

use super::data_loading::FrameSample;
use super::splats::{splat_color_representation_for_config, HostSplats};

pub(super) fn build_initial_splats(
    dataset: &TrainingDataset,
    frames: &[FrameSample],
    config: &super::TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    if dataset.initial_points.is_empty() {
        if config.training_profile == super::TrainingProfile::LiteGsMacV1 {
            log::warn!(
                "LiteGsMacV1 did not receive sparse-point initialization; falling back to frame-based initialization for this dataset"
            );
        }
        return build_initial_splats_from_frames(dataset, frames, config);
    }

    let sh_degree = splat_color_representation_for_config(config).sh_degree();
    let init_config = gaussian_init_config_for_training(config);
    let mut gaussians = initialize_gaussians_from_points(&dataset.initial_points, &init_config);
    let max_initial = config.max_initial_gaussians.max(1);
    if gaussians.len() > max_initial {
        log::warn!(
            "Truncating point-initialized chunk from {} to {} gaussians to respect max_initial_gaussians",
            gaussians.len(),
            max_initial,
        );
        gaussians.truncate(max_initial);
    }

    let mut splats = HostSplats::with_sh_degree_capacity(sh_degree, gaussians.len());
    for gaussian in gaussians {
        splats.push_rgb(
            gaussian.position,
            [
                gaussian.scale[0].max(1e-6).ln(),
                gaussian.scale[1].max(1e-6).ln(),
                gaussian.scale[2].max(1e-6).ln(),
            ],
            gaussian.rotation,
            opacity_to_logit(gaussian.opacity),
            gaussian.color,
        );
    }

    splats
        .validate()
        .map_err(|err| TrainingError::TrainingFailed(err.to_string()))?;
    Ok(splats)
}

pub(super) fn gaussian_init_config_for_training(
    config: &super::TrainingConfig,
) -> GaussianInitConfig {
    let mut init = GaussianInitConfig::default();
    if config.training_profile == super::TrainingProfile::LiteGsMacV1 {
        init.min_scale = 0.000_316_227_76;
        init.max_scale = 10_000.0;
        init.scale_factor = 1.0;
        init.opacity = 0.1;
    }
    init
}

pub(super) fn initial_frame_sampling_step(
    dataset: &TrainingDataset,
    frame_count: usize,
    config: &super::TrainingConfig,
) -> usize {
    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;
    let frame_budget = config
        .max_initial_gaussians
        .checked_div(frame_count.max(1))
        .unwrap_or(1)
        .max(1);
    if config.sampling_step == 0 {
        compute_sampling_step(width, height, frame_budget)
    } else {
        config.sampling_step.max(1)
    }
}

pub(super) fn accumulate_frame_into_initial_splats(
    splats: &mut HostSplats,
    dataset: &TrainingDataset,
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
    color_u8: &[u8],
    depth: &[f32],
    sampling_step: usize,
    config: &super::TrainingConfig,
) {
    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;
    let max_initial = config.max_initial_gaussians.max(1);
    let rotation = glam::Mat3::from_cols(
        glam::Vec3::new(rotation[0][0], rotation[0][1], rotation[0][2]),
        glam::Vec3::new(rotation[1][0], rotation[1][1], rotation[1][2]),
        glam::Vec3::new(rotation[2][0], rotation[2][1], rotation[2][2]),
    );
    let translation = glam::Vec3::new(translation[0], translation[1], translation[2]);

    for y in (0..height).step_by(sampling_step) {
        for x in (0..width).step_by(sampling_step) {
            if splats.len() >= max_initial {
                return;
            }

            let idx = y * width + x;
            let depth = depth[idx];
            if depth < config.min_depth || depth > config.max_depth {
                continue;
            }

            let x_cam = (x as f32 - dataset.intrinsics.cx) * depth / dataset.intrinsics.fx;
            let y_cam = (y as f32 - dataset.intrinsics.cy) * depth / dataset.intrinsics.fy;
            let camera_point = glam::Vec3::new(x_cam, y_cam, depth);
            let world_point = rotation * camera_point + translation;
            let color_base = idx * 3;
            splats.push_rgb(
                [world_point.x, world_point.y, world_point.z],
                [0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
                [1.0, 0.0, 0.0, 0.0],
                opacity_to_logit(0.5),
                [
                    color_u8[color_base] as f32 / 255.0,
                    color_u8[color_base + 1] as f32 / 255.0,
                    color_u8[color_base + 2] as f32 / 255.0,
                ],
            );
        }
    }
}

pub(super) fn finalize_initial_splats(
    splats: &mut HostSplats,
    sampling_step: usize,
    config: &super::TrainingConfig,
) -> Result<(), TrainingError> {
    if splats.is_empty() {
        return Err(TrainingError::InvalidInput(
            "failed to build any initial gaussians from the training frames".to_string(),
        ));
    }

    if sampling_step > 1 && splats.len() >= config.max_initial_gaussians.max(1) {
        log::warn!(
            "initial splat set reached the max_initial_gaussians budget ({}) with sampling step {}",
            config.max_initial_gaussians,
            sampling_step,
        );
    }

    splats
        .validate()
        .map_err(|err| TrainingError::TrainingFailed(err.to_string()))?;
    log::info!(
        "Initialized {} gaussians from training frames",
        splats.len()
    );
    Ok(())
}

fn build_initial_splats_from_frames(
    dataset: &TrainingDataset,
    frames: &[FrameSample],
    config: &super::TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;
    let sampling_step = initial_frame_sampling_step(dataset, frames.len(), config);
    let sh_degree = splat_color_representation_for_config(config).sh_degree();
    let mut splats =
        HostSplats::with_sh_degree_capacity(sh_degree, config.max_initial_gaussians.max(1));

    log::info!(
        "Initializing gaussians from {} frames | resolution={}x{} | frame_budget={} | sampling_step={}",
        frames.len(),
        width,
        height,
        config
            .max_initial_gaussians
            .checked_div(frames.len().max(1))
            .unwrap_or(1)
            .max(1),
        sampling_step
    );

    for frame in frames {
        accumulate_frame_into_initial_splats(
            &mut splats,
            dataset,
            &frame.rotation,
            &frame.translation,
            &frame.color_u8,
            &frame.depth,
            sampling_step,
            config,
        );
        if splats.len() >= config.max_initial_gaussians.max(1) {
            break;
        }
    }

    finalize_initial_splats(&mut splats, sampling_step, config)?;
    Ok(splats)
}

fn compute_sampling_step(width: usize, height: usize, max_gaussians: usize) -> usize {
    let pixels = width.saturating_mul(height).max(1);
    let ratio = (pixels as f32 / max_gaussians.max(1) as f32).sqrt();
    ratio.ceil().max(2.0) as usize
}

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(test)]
mod tests {
    use super::gaussian_init_config_for_training;

    #[test]
    fn litegs_profile_uses_litegs_point_init_defaults() {
        let legacy = gaussian_init_config_for_training(&crate::TrainingConfig::default());
        assert_eq!(legacy.opacity, 0.5);
        assert_eq!(legacy.scale_factor, 0.5);

        let litegs = gaussian_init_config_for_training(&crate::TrainingConfig {
            training_profile: crate::TrainingProfile::LiteGsMacV1,
            ..crate::TrainingConfig::default()
        });
        assert!((litegs.min_scale - 0.000_316_227_76).abs() < 1e-9);
        assert_eq!(litegs.scale_factor, 1.0);
        assert_eq!(litegs.opacity, 0.1);
    }
}
