#[cfg(feature = "gpu")]
use super::{
    data_loading, metal_trainer, select_training_execution_plan, topology,
    train_chunked_sequentially, validate_litegs_mac_v1_config, TrainingConfig,
    TrainingExecutionPlan, TrainingExecutionRoute, TrainingProfile,
};
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::TrainableGaussians;
#[cfg(feature = "gpu")]
use crate::{GaussianMap, TrainingDataset, TrainingError};
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
pub(super) fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let plan = plan_training_route(dataset, config)?;
    execute_training_plan(dataset, config, plan)
}

#[cfg(feature = "gpu")]
pub(super) fn train_materialized_dataset(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let device = crate::require_metal_device()?;
    let loaded = data_loading::load_training_data(dataset, config, &device)?;
    metal_trainer::train_loaded(dataset, config, device, loaded)
}

#[cfg(feature = "gpu")]
pub(super) fn run_training_loop(
    trainer: &mut metal_trainer::MetalTrainer,
    gaussians: &mut TrainableGaussians,
    frames: &[metal_trainer::MetalTrainingFrame],
    max_iterations: usize,
) -> candle_core::Result<metal_trainer::MetalTrainingStats> {
    trainer.initialize_training_session(gaussians, frames)?;

    let train_start = Instant::now();
    for iter in 0..max_iterations {
        let frame_idx = iter % frames.len();
        let should_log = iter < 5 || iter % 25 == 0;
        let should_profile = trainer.should_profile_iteration(iter);
        let step_start = Instant::now();
        let policy = trainer.topology_policy();
        let schedule = topology::schedule_topology(
            &policy,
            topology::TopologyStepContext {
                iteration: trainer.next_iteration(),
                frame_idx,
                frame_count: frames.len(),
            },
        );
        let outcome = trainer.training_step(
            gaussians,
            &frames[frame_idx],
            frame_idx,
            frames.len(),
            should_profile,
            topology::should_collect_visible_indices(&policy, schedule),
        )?;
        trainer.apply_topology_schedule(gaussians, frames.len(), schedule)?;

        if should_log {
            log::info!(
                "Metal iter {:5}/{:5} | frame {:3}/{:3} | visible {:5}/{:5} | loss {:.6} | step_time={:.2}s | elapsed={:.2}s",
                iter,
                max_iterations,
                frame_idx + 1,
                frames.len(),
                outcome.visible_gaussians,
                outcome.total_gaussians,
                outcome.loss,
                step_start.elapsed().as_secs_f64(),
                train_start.elapsed().as_secs_f64()
            );
        }
        trainer.log_step_profile(&outcome, iter, max_iterations);
    }

    Ok(trainer.finalize_training_session(gaussians, frames.len()))
}

#[cfg(feature = "gpu")]
pub(super) fn plan_training_route(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingExecutionPlan, TrainingError> {
    match config.training_profile {
        TrainingProfile::LegacyMetal => select_training_execution_plan(dataset, config),
        TrainingProfile::LiteGsMacV1 => {
            validate_litegs_mac_v1_config(config)?;
            log::info!(
                "Training with LiteGS Mac V1 profile | sh_degree={} | cluster_size={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={}",
                config.litegs.sh_degree,
                config.litegs.cluster_size,
                config.litegs.tile_size,
                config.litegs.sparse_grad,
                config.litegs.reg_weight,
                config.litegs.enable_transmittance,
                config.litegs.enable_depth,
            );
            select_training_execution_plan(dataset, config)
        }
    }
}

#[cfg(feature = "gpu")]
fn execute_training_plan(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    plan: TrainingExecutionPlan,
) -> Result<GaussianMap, TrainingError> {
    match plan.route {
        TrainingExecutionRoute::Standard => train_materialized_dataset(dataset, config),
        TrainingExecutionRoute::ChunkedSingleChunk => {
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected single-chunk pass-through | requested_gaussians={} | affordable_gaussians={} | estimated_peak≈{:.1} GiB | effective_budget≈{:.1} GiB",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    estimate.estimated_peak_gib(),
                    estimate.effective_budget_gib(),
                );
            }
            train_materialized_dataset(dataset, config)
        }
        TrainingExecutionRoute::ChunkedSequential => {
            let chunk_plan = plan
                .chunk_plan
                .as_ref()
                .expect("sequential route requires chunk plan");
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected sequential chunk execution | requested_gaussians={} | affordable_gaussians={} | chunks={} | trainable_chunks={}",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    chunk_plan.chunks.len(),
                    chunk_plan.trainable_chunks().count(),
                );
            }
            train_chunked_sequentially(dataset, config, chunk_plan)
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{plan_training_route, TrainingExecutionRoute};
    use crate::{
        Intrinsics, LiteGsConfig, ScenePose, TrainingConfig, TrainingDataset, TrainingProfile, SE3,
    };
    use std::path::PathBuf;

    fn make_dataset(frame_count: usize, width: u32, height: u32) -> TrainingDataset {
        let intrinsics = Intrinsics::from_focal(500.0, width, height);
        let mut dataset = TrainingDataset::new(intrinsics);
        for idx in 0..frame_count {
            dataset.add_pose(ScenePose::new(
                idx as u64,
                PathBuf::from(format!("frame_{idx:04}.png")),
                SE3::identity(),
                idx as f64,
            ));
        }
        dataset
    }

    #[test]
    fn legacy_profile_route_smoke_uses_standard_path() {
        let dataset = make_dataset(3, 64, 64);
        let plan = plan_training_route(&dataset, &TrainingConfig::default()).unwrap();

        assert_eq!(plan.route, TrainingExecutionRoute::Standard);
        assert!(plan.chunk_estimate.is_none());
        assert!(plan.chunk_plan.is_none());
    }

    #[test]
    fn legacy_profile_route_smoke_uses_chunked_passthrough_when_affordable() {
        let dataset = make_dataset(3, 32, 32);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            max_initial_gaussians: 128,
            ..TrainingConfig::default()
        };

        let plan = plan_training_route(&dataset, &config).unwrap();

        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSingleChunk);
        assert!(plan.chunk_estimate.is_some());
        assert!(plan.chunk_plan.is_none());
    }

    #[test]
    fn legacy_profile_route_smoke_uses_sequential_chunk_execution_when_required() {
        let dataset = make_dataset(5, 1920, 1080);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            metal_render_scale: 1.0,
            max_initial_gaussians: 57_474,
            max_chunks: 4,
            ..TrainingConfig::default()
        };

        let plan = plan_training_route(&dataset, &config).unwrap();

        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSequential);
        assert!(plan.chunk_estimate.is_some());
        assert!(plan.chunk_plan.is_some());
    }

    #[test]
    fn litegs_profile_route_smoke_uses_standard_path_after_validation() {
        let dataset = make_dataset(3, 64, 64);
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig::default(),
            ..TrainingConfig::default()
        };

        let plan = plan_training_route(&dataset, &config).unwrap();

        assert_eq!(plan.route, TrainingExecutionRoute::Standard);
    }

    #[test]
    fn litegs_profile_route_smoke_surfaces_validation_failures_before_planning() {
        let dataset = make_dataset(3, 64, 64);
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sh_degree: 0,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        let err = plan_training_route(&dataset, &config).expect_err("invalid LiteGS config");
        assert!(
            err.to_string().contains("sh_degree=0"),
            "unexpected error: {err}"
        );
    }
}
