use super::metal_trainer::{self, ChunkCapacityEstimate};
use super::{plan_spatial_chunks, ChunkPlan, TrainingConfig};
use crate::{TrainingDataset, TrainingError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainingExecutionRoute {
    Standard,
    ChunkedSingleChunk,
    ChunkedSequential,
}

#[derive(Debug, Clone)]
pub(crate) struct TrainingExecutionPlan {
    pub(crate) route: TrainingExecutionRoute,
    pub(crate) chunk_estimate: Option<ChunkCapacityEstimate>,
    pub(crate) chunk_plan: Option<ChunkPlan>,
}

pub(crate) fn select_training_execution_plan(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingExecutionPlan, TrainingError> {
    if !config.chunked_training {
        return Ok(TrainingExecutionPlan {
            route: TrainingExecutionRoute::Standard,
            chunk_estimate: None,
            chunk_plan: None,
        });
    }

    let estimate = metal_trainer::estimate_chunk_capacity(dataset, config)?;
    if estimate.requires_subdivision_or_degradation() {
        let chunk_plan =
            plan_spatial_chunks(dataset, config, Some(estimate.affordable_initial_gaussians))?;
        if chunk_plan.trainable_chunks().count() == 0 {
            return Err(TrainingError::TrainingFailed(format!(
                "chunked training planned {} chunks under {:.1} GiB, but none met the minimum camera threshold of {}. Recommendations: {}",
                chunk_plan.chunks.len(),
                estimate.effective_budget_gib(),
                config.min_cameras_per_chunk,
                estimate.recommendations().join("; "),
            )));
        }

        return Ok(TrainingExecutionPlan {
            route: TrainingExecutionRoute::ChunkedSequential,
            chunk_estimate: Some(estimate),
            chunk_plan: Some(chunk_plan),
        });
    }

    Ok(TrainingExecutionPlan {
        route: TrainingExecutionRoute::ChunkedSingleChunk,
        chunk_estimate: Some(estimate),
        chunk_plan: None,
    })
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{select_training_execution_plan, TrainingExecutionRoute};
    use crate::{Intrinsics, ScenePose, TrainingDataset, SE3};
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
    fn non_chunked_execution_plan_uses_standard_route() {
        let dataset = make_dataset(3, 64, 64);
        let plan =
            select_training_execution_plan(&dataset, &crate::TrainingConfig::default()).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::Standard);
        assert!(plan.chunk_estimate.is_none());
    }

    #[test]
    fn chunked_execution_plan_selects_single_chunk_route_when_affordable() {
        let dataset = make_dataset(3, 32, 32);
        let config = crate::TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            max_initial_gaussians: 128,
            ..crate::TrainingConfig::default()
        };
        let plan = select_training_execution_plan(&dataset, &config).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSingleChunk);
        let estimate = plan
            .chunk_estimate
            .expect("chunk estimate should be present");
        assert!(!estimate.requires_subdivision_or_degradation());
        assert!(plan.chunk_plan.is_none());
    }

    #[test]
    fn chunked_execution_plan_uses_sequential_route_when_subdivision_is_required() {
        let dataset = make_dataset(5, 1920, 1080);
        let config = crate::TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            metal_render_scale: 1.0,
            max_initial_gaussians: 57_474,
            max_chunks: 4,
            ..crate::TrainingConfig::default()
        };
        let plan = select_training_execution_plan(&dataset, &config).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSequential);
        assert!(plan.chunk_estimate.is_some());
        assert!(plan.chunk_plan.is_some());
        assert!(plan.chunk_plan.unwrap().chunks.len() >= 2);
    }
}
