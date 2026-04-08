#[cfg(feature = "gpu")]
use super::data_loading::LoadedTrainingData;
#[cfg(feature = "gpu")]
use super::metal_trainer::{MetalStepProfile, MetalTrainer};
#[cfg(feature = "gpu")]
use super::splats::Splats;
#[cfg(feature = "gpu")]
use super::{TrainingConfig, TrainingProfile};
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::DiffCamera;
#[cfg(feature = "gpu")]
use crate::{Gaussian3D, GaussianMap, TrainingError};
#[cfg(feature = "gpu")]
use candle_core::Device;
#[cfg(feature = "gpu")]
use glam::{Quat, Vec3};
#[cfg(feature = "gpu")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalTrainingBenchmarkSpec {
    pub width: usize,
    pub height: usize,
    pub frame_count: usize,
    pub gaussian_count: usize,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub smoke_iterations: usize,
    pub training_profile: TrainingProfile,
}

#[cfg(feature = "gpu")]
impl Default for MetalTrainingBenchmarkSpec {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            frame_count: 3,
            gaussian_count: 128,
            warmup_steps: 2,
            measured_steps: 5,
            smoke_iterations: 8,
            training_profile: TrainingProfile::LegacyMetal,
        }
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalTrainingBenchmarkReport {
    pub spec: MetalTrainingBenchmarkSpec,
    pub average_forward_ms: f64,
    pub average_backward_ms: f64,
    pub average_loss_ms: f64,
    pub average_optimizer_ms: f64,
    pub average_step_ms: f64,
    pub smoke_training_ms: f64,
    pub final_loss: f32,
    pub final_gaussians: usize,
    pub active_sh_degree: Option<usize>,
    pub average_visible_gaussians: f64,
    pub average_active_tiles: f64,
}

#[cfg(feature = "gpu")]
pub fn run_metal_training_benchmark(
    spec: &MetalTrainingBenchmarkSpec,
) -> Result<MetalTrainingBenchmarkReport, TrainingError> {
    if spec.width == 0 || spec.height == 0 {
        return Err(TrainingError::InvalidInput(
            "benchmark width and height must both be > 0".to_string(),
        ));
    }
    if spec.frame_count == 0 {
        return Err(TrainingError::InvalidInput(
            "benchmark frame_count must be > 0".to_string(),
        ));
    }
    if spec.gaussian_count == 0 {
        return Err(TrainingError::InvalidInput(
            "benchmark gaussian_count must be > 0".to_string(),
        ));
    }
    if spec.measured_steps == 0 {
        return Err(TrainingError::InvalidInput(
            "benchmark measured_steps must be > 0".to_string(),
        ));
    }
    if spec.smoke_iterations == 0 {
        return Err(TrainingError::InvalidInput(
            "benchmark smoke_iterations must be > 0".to_string(),
        ));
    }

    let device = crate::require_metal_device()?;
    let config = benchmark_config(spec);

    let micro = run_step_microbenchmark(spec, &config, device.clone())?;
    let smoke = run_smoke_training_benchmark(spec, &config, device)?;

    Ok(MetalTrainingBenchmarkReport {
        spec: spec.clone(),
        average_forward_ms: micro.average_forward_ms,
        average_backward_ms: micro.average_backward_ms,
        average_loss_ms: micro.average_loss_ms,
        average_optimizer_ms: micro.average_optimizer_ms,
        average_step_ms: micro.average_step_ms,
        average_visible_gaussians: micro.average_visible_gaussians,
        average_active_tiles: micro.average_active_tiles,
        smoke_training_ms: smoke.smoke_training_ms,
        final_loss: smoke.final_loss,
        final_gaussians: smoke.final_gaussians,
        active_sh_degree: smoke.active_sh_degree,
    })
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
struct StepMicrobenchmark {
    average_forward_ms: f64,
    average_backward_ms: f64,
    average_loss_ms: f64,
    average_optimizer_ms: f64,
    average_step_ms: f64,
    average_visible_gaussians: f64,
    average_active_tiles: f64,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
struct SmokeBenchmark {
    smoke_training_ms: f64,
    final_loss: f32,
    final_gaussians: usize,
    active_sh_degree: Option<usize>,
}

#[cfg(feature = "gpu")]
fn benchmark_config(spec: &MetalTrainingBenchmarkSpec) -> TrainingConfig {
    let mut config = TrainingConfig::default();
    config.training_profile = spec.training_profile;
    config.iterations = spec
        .smoke_iterations
        .max(spec.warmup_steps.saturating_add(spec.measured_steps))
        .max(1);
    config.max_initial_gaussians = spec.gaussian_count.max(1);
    config.densify_interval = 0;
    config.prune_interval = 0;
    config.topology_warmup = config.iterations.saturating_add(1);
    config.metal_profile_steps = false;
    config.metal_profile_interval = 1;
    config.metal_use_native_forward = false;
    config
}

#[cfg(feature = "gpu")]
fn run_step_microbenchmark(
    spec: &MetalTrainingBenchmarkSpec,
    config: &TrainingConfig,
    device: Device,
) -> Result<StepMicrobenchmark, TrainingError> {
    let loaded = synthetic_loaded_training_data(spec, config, &device)?;
    let mut trainer = MetalTrainer::new(spec.width, spec.height, config, device.clone())?;
    let frames = trainer.prepare_frames(&loaded)?;
    let mut gaussians = loaded.initial_splats.to_trainable(&device)?;
    trainer.initialize_training_session(&mut gaussians, &frames)?;

    for step in 0..spec.warmup_steps {
        let frame_idx = step % frames.len();
        trainer.training_step(
            &mut gaussians,
            &frames[frame_idx],
            frame_idx,
            frames.len(),
            true,
        )?;
    }

    let mut forward_ms = 0.0;
    let mut backward_ms = 0.0;
    let mut loss_ms = 0.0;
    let mut optimizer_ms = 0.0;
    let mut step_ms = 0.0;
    let mut visible = 0.0;
    let mut active_tiles = 0.0;

    for step in 0..spec.measured_steps {
        let frame_idx = step % frames.len();
        let outcome = trainer.training_step(
            &mut gaussians,
            &frames[frame_idx],
            frame_idx,
            frames.len(),
            true,
        )?;
        let summary = outcome.profile_summary().ok_or_else(|| {
            TrainingError::TrainingFailed(
                "benchmark training_step did not produce a profile summary".to_string(),
            )
        })?;
        accumulate_profile(
            summary,
            &mut forward_ms,
            &mut backward_ms,
            &mut loss_ms,
            &mut optimizer_ms,
            &mut step_ms,
            &mut visible,
            &mut active_tiles,
        );
    }

    let divisor = spec.measured_steps as f64;
    Ok(StepMicrobenchmark {
        average_forward_ms: forward_ms / divisor,
        average_backward_ms: backward_ms / divisor,
        average_loss_ms: loss_ms / divisor,
        average_optimizer_ms: optimizer_ms / divisor,
        average_step_ms: step_ms / divisor,
        average_visible_gaussians: visible / divisor,
        average_active_tiles: active_tiles / divisor,
    })
}

#[cfg(feature = "gpu")]
fn run_smoke_training_benchmark(
    spec: &MetalTrainingBenchmarkSpec,
    config: &TrainingConfig,
    device: Device,
) -> Result<SmokeBenchmark, TrainingError> {
    let loaded = synthetic_loaded_training_data(spec, config, &device)?;
    let mut trainer = MetalTrainer::new(spec.width, spec.height, config, device.clone())?;
    let frames = trainer.prepare_frames(&loaded)?;
    let mut gaussians = loaded.initial_splats.to_trainable(&device)?;

    let smoke_start = Instant::now();
    let stats = trainer.train(&mut gaussians, &frames, spec.smoke_iterations)?;
    let smoke_training_ms = smoke_start.elapsed().as_secs_f64() * 1000.0;

    Ok(SmokeBenchmark {
        smoke_training_ms,
        final_loss: stats.final_loss,
        final_gaussians: stats
            .telemetry
            .topology
            .final_gaussians
            .unwrap_or(gaussians.len()),
        active_sh_degree: stats.telemetry.active_sh_degree,
    })
}

#[cfg(feature = "gpu")]
fn accumulate_profile(
    summary: MetalStepProfile,
    forward_ms: &mut f64,
    backward_ms: &mut f64,
    loss_ms: &mut f64,
    optimizer_ms: &mut f64,
    step_ms: &mut f64,
    visible: &mut f64,
    active_tiles: &mut f64,
) {
    *forward_ms += duration_ms(summary.projection)
        + duration_ms(summary.sorting)
        + duration_ms(summary.rasterization);
    *backward_ms += duration_ms(summary.backward);
    *loss_ms += duration_ms(summary.loss);
    *optimizer_ms += duration_ms(summary.optimizer);
    *step_ms += duration_ms(summary.total);
    *visible += summary.visible_gaussians as f64;
    *active_tiles += summary.active_tiles as f64;
}

#[cfg(feature = "gpu")]
fn synthetic_loaded_training_data(
    spec: &MetalTrainingBenchmarkSpec,
    config: &TrainingConfig,
    device: &Device,
) -> Result<LoadedTrainingData, TrainingError> {
    let mut cameras = Vec::with_capacity(spec.frame_count);
    let mut colors = Vec::with_capacity(spec.frame_count);
    let mut depths = Vec::with_capacity(spec.frame_count);
    for frame_idx in 0..spec.frame_count {
        let translation = [frame_idx as f32 * 0.02, 0.0, 0.0];
        cameras.push(
            DiffCamera::new(
                spec.width as f32,
                spec.height as f32,
                spec.width as f32 * 0.5,
                spec.height as f32 * 0.5,
                spec.width,
                spec.height,
                &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                &translation,
                device,
            )
            .map_err(TrainingError::from)?,
        );
        colors.push(synthetic_color_frame(spec.width, spec.height, frame_idx));
        depths.push(synthetic_depth_frame(spec.width, spec.height, frame_idx));
    }

    Ok(LoadedTrainingData {
        cameras,
        colors,
        depths,
        target_width: spec.width,
        target_height: spec.height,
        initial_splats: Splats::from_gaussian_map_for_config(
            &synthetic_initial_map(spec.gaussian_count),
            config,
        )?,
    })
}

#[cfg(feature = "gpu")]
fn synthetic_initial_map(gaussian_count: usize) -> GaussianMap {
    let cols = (gaussian_count as f32).sqrt().ceil().max(1.0) as usize;
    let mut gaussians = Vec::with_capacity(gaussian_count);
    for idx in 0..gaussian_count {
        let x_idx = idx % cols;
        let y_idx = idx / cols;
        let norm_x = if cols > 1 {
            x_idx as f32 / (cols - 1) as f32
        } else {
            0.5
        };
        let norm_y = if cols > 1 {
            y_idx as f32 / (cols - 1) as f32
        } else {
            0.5
        };
        let position = Vec3::new(
            (norm_x - 0.5) * 1.2,
            (norm_y - 0.5) * 0.8,
            1.5 + (idx % 7) as f32 * 0.05,
        );
        let scale = Vec3::new(0.06, 0.06, 0.08);
        let color = [
            0.2 + 0.6 * norm_x,
            0.15 + 0.7 * norm_y,
            0.3 + 0.5 * ((idx % 11) as f32 / 10.0),
        ];
        gaussians.push(Gaussian3D::new(
            position,
            scale,
            Quat::IDENTITY,
            0.55,
            color.map(|value| value.clamp(0.0, 1.0)),
        ));
    }
    GaussianMap::from_gaussians(gaussians)
}

#[cfg(feature = "gpu")]
fn synthetic_color_frame(width: usize, height: usize, frame_idx: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(width * height * 3);
    let phase = frame_idx as f32 * 0.17;
    for y in 0..height {
        let fy = if height > 1 {
            y as f32 / (height - 1) as f32
        } else {
            0.5
        };
        for x in 0..width {
            let fx = if width > 1 {
                x as f32 / (width - 1) as f32
            } else {
                0.5
            };
            output.push((0.5 + 0.5 * ((fx * 3.14159) + phase).sin()).clamp(0.0, 1.0));
            output.push((0.5 + 0.5 * ((fy * 2.35619) + phase * 0.5).cos()).clamp(0.0, 1.0));
            output.push((0.2 + 0.5 * fx + 0.3 * fy).clamp(0.0, 1.0));
        }
    }
    output
}

#[cfg(feature = "gpu")]
fn synthetic_depth_frame(width: usize, height: usize, frame_idx: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(width * height);
    let frame_bias = frame_idx as f32 * 0.03;
    for y in 0..height {
        let fy = if height > 1 {
            y as f32 / (height - 1) as f32
        } else {
            0.5
        };
        for x in 0..width {
            let fx = if width > 1 {
                x as f32 / (width - 1) as f32
            } else {
                0.5
            };
            output.push(1.2 + frame_bias + 0.35 * fy + 0.15 * fx);
        }
    }
    output
}

#[cfg(feature = "gpu")]
fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        benchmark_config, run_metal_training_benchmark, synthetic_loaded_training_data,
        MetalTrainingBenchmarkSpec,
    };
    use crate::{TrainingConfig, TrainingProfile};
    use candle_core::Device;

    #[test]
    fn benchmark_config_disables_topology_for_guardrail_runs() {
        let spec = MetalTrainingBenchmarkSpec {
            gaussian_count: 256,
            smoke_iterations: 9,
            warmup_steps: 2,
            measured_steps: 5,
            training_profile: TrainingProfile::LiteGsMacV1,
            ..MetalTrainingBenchmarkSpec::default()
        };

        let config = benchmark_config(&spec);

        assert_eq!(config.training_profile, TrainingProfile::LiteGsMacV1);
        assert_eq!(config.max_initial_gaussians, 256);
        assert_eq!(config.iterations, 9);
        assert_eq!(config.densify_interval, 0);
        assert_eq!(config.prune_interval, 0);
        assert!(config.topology_warmup > config.iterations);
        assert!(!config.metal_use_native_forward);
    }

    #[test]
    fn synthetic_loaded_training_data_matches_requested_fixture_shape() {
        let spec = MetalTrainingBenchmarkSpec {
            width: 20,
            height: 12,
            frame_count: 4,
            gaussian_count: 25,
            ..MetalTrainingBenchmarkSpec::default()
        };

        let loaded =
            synthetic_loaded_training_data(&spec, &TrainingConfig::default(), &Device::Cpu)
                .unwrap();

        assert_eq!(loaded.cameras.len(), 4);
        assert_eq!(loaded.colors.len(), 4);
        assert_eq!(loaded.depths.len(), 4);
        assert_eq!(loaded.colors[0].len(), 20 * 12 * 3);
        assert_eq!(loaded.depths[0].len(), 20 * 12);
        assert_eq!(loaded.initial_splats.len(), 25);
    }

    #[test]
    fn benchmark_rejects_invalid_spec_before_requesting_metal() {
        let err = run_metal_training_benchmark(&MetalTrainingBenchmarkSpec {
            width: 0,
            ..MetalTrainingBenchmarkSpec::default()
        })
        .expect_err("zero width should fail validation");
        assert!(err
            .to_string()
            .contains("width and height must both be > 0"));

        let err = run_metal_training_benchmark(&MetalTrainingBenchmarkSpec {
            gaussian_count: 0,
            ..MetalTrainingBenchmarkSpec::default()
        })
        .expect_err("zero gaussians should fail validation");
        assert!(err.to_string().contains("gaussian_count must be > 0"));
    }
}
