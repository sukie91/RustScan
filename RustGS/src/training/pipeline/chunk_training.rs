use super::events::{
    emit_training_event, TrainingChunkCompleted, TrainingChunkStarted, TrainingEvent,
    TrainingEventSink, TrainingRun, TrainingRunReport,
};
use super::export::ChunkPersistenceContext;
use super::metal::entry as metal_entry;
use super::metal::memory::{self as metal_memory, ChunkCapacityEstimate};
use super::splats::HostSplats;
use super::{
    materialize_chunk_dataset, ChunkBounds, ChunkDisposition, ChunkPlan, MaterializedChunkDataset,
    PlannedChunk, TrainingConfig, MIN_RENDER_SCALE,
};
use crate::{TrainingDataset, TrainingError};
use std::time::Instant;

#[derive(Debug, Clone)]
pub(crate) struct ChunkTrainingOverridePlan {
    pub(crate) effective_config: TrainingConfig,
    pub(crate) estimate: ChunkCapacityEstimate,
    pub(crate) lowered_gaussian_cap: bool,
    pub(crate) lowered_render_scale: bool,
}

pub(crate) fn train_chunked_sequentially_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    chunk_plan: &ChunkPlan,
    sink: &mut TrainingEventSink<'_>,
) -> Result<TrainingRun, TrainingError> {
    let start = Instant::now();
    let mut merged_scene = HostSplats::default();
    let total_chunks = chunk_plan.training_chunks().count();
    let mut persistence =
        ChunkPersistenceContext::new(config.chunk_artifact_dir.clone(), config, chunk_plan)?;

    for skipped in chunk_plan
        .chunks
        .iter()
        .filter(|chunk| chunk.disposition == ChunkDisposition::SkippedInsufficientCameras)
    {
        log::warn!(
            "Skipping chunk {} due to insufficient cameras | assigned_cameras={} | required_min={}",
            skipped.chunk_id,
            skipped.pose_indices.len(),
            config.min_cameras_per_chunk,
        );
        persistence.record_skipped(skipped, config)?;
    }

    execute_training_chunks_sequentially(dataset, chunk_plan, |chunk, materialized| {
        let chunk_index = chunk.chunk_id + 1;
        emit_training_event(
            sink,
            TrainingEvent::ChunkStarted(TrainingChunkStarted {
                chunk_index,
                total_chunks,
                chunk_id: chunk.chunk_id,
                pose_count: materialized.dataset.poses.len(),
                initial_point_count: materialized.dataset.initial_points.len(),
                used_frame_based_initialization: materialized.used_frame_based_initialization,
            }),
        );
        log::info!(
            "Training chunk {}/{} | chunk_id={} | poses={} | local_points={} | frame_init_fallback={}",
            chunk_index,
            total_chunks,
            chunk.chunk_id,
            materialized.dataset.poses.len(),
            materialized.dataset.initial_points.len(),
            materialized.used_frame_based_initialization,
        );

        let override_plan = match adapt_chunk_training_config(&materialized.dataset, config) {
            Ok(plan) => plan,
            Err(err) => {
                persistence.record_failure(chunk, &materialized, err.to_string(), None)?;
                return Err(err);
            }
        };
        log::info!(
            "Chunk {} effective config | max_initial_gaussians={} | metal_render_scale={:.3} | lowered_gaussian_cap={} | lowered_render_scale={} | estimated_peak≈{:.1} GiB",
            chunk.chunk_id,
            override_plan.effective_config.max_initial_gaussians,
            override_plan.effective_config.metal_render_scale,
            override_plan.lowered_gaussian_cap,
            override_plan.lowered_render_scale,
            override_plan.estimate.estimated_peak_gib(),
        );

        let chunk_run = match metal_entry::train_splats_with_report(
            &materialized.dataset,
            &override_plan.effective_config,
        ) {
            Ok(run) => run,
            Err(err) => {
                persistence.record_failure(
                    chunk,
                    &materialized,
                    err.to_string(),
                    Some(&override_plan),
                )?;
                return Err(err);
            }
        };
        let mut chunk_scene = chunk_run.splats;
        let core_filter_removed = merge_chunk_splats(
            &mut merged_scene,
            &mut chunk_scene,
            &chunk.core_bounds,
            config.merge_core_only,
        )?;
        persistence.record_success(
            chunk,
            &materialized,
            &override_plan,
            &chunk_scene,
            core_filter_removed,
        )?;
        log::info!(
            "Chunk {} complete | accumulated_gaussians={}",
            chunk.chunk_id,
            merged_scene.len(),
        );
        emit_training_event(
            sink,
            TrainingEvent::ChunkCompleted(TrainingChunkCompleted {
                chunk_index,
                total_chunks,
                chunk_id: chunk.chunk_id,
                chunk_gaussian_count: chunk_scene.len(),
                merged_gaussian_count: merged_scene.len(),
            }),
        );
        Ok(())
    })?;

    Ok(TrainingRun {
        report: TrainingRunReport {
            elapsed: start.elapsed(),
            final_loss: None,
            final_step_loss: None,
            gaussian_count: merged_scene.len(),
            sh_degree: merged_scene.sh_degree(),
            telemetry: None,
        },
        splats: merged_scene,
    })
}

pub(crate) fn execute_training_chunks_sequentially<F>(
    dataset: &TrainingDataset,
    chunk_plan: &ChunkPlan,
    mut execute: F,
) -> Result<(), TrainingError>
where
    F: FnMut(&PlannedChunk, MaterializedChunkDataset) -> Result<(), TrainingError>,
{
    for chunk in chunk_plan.training_chunks() {
        let materialized = materialize_chunk_dataset(dataset, chunk)?;
        execute(chunk, materialized)?;
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn retain_splats_in_bounds(scene: &mut HostSplats, bounds: &ChunkBounds) -> usize {
    let mut keep_mask = vec![false; scene.len()];
    let mut kept = 0usize;
    for (idx, keep) in keep_mask.iter_mut().enumerate() {
        let position = scene.position(idx);
        *keep = (0..3)
            .all(|axis| position[axis] >= bounds.min[axis] && position[axis] <= bounds.max[axis]);
        kept += usize::from(*keep);
    }
    let removed = scene.len().saturating_sub(kept);
    if removed == 0 {
        return 0;
    }
    let mut filtered = scene.retained_view(kept);
    for (idx, keep) in keep_mask.iter().copied().enumerate() {
        if keep {
            filtered.push(
                scene.position(idx),
                scene.log_scale(idx),
                scene.rotation(idx),
                scene.opacity_logits[idx],
                scene.sh_coeffs_row(idx),
            );
        }
    }
    *scene = filtered;
    removed
}

pub(crate) fn merge_chunk_splats(
    merged_scene: &mut HostSplats,
    chunk_scene: &mut HostSplats,
    core_bounds: &ChunkBounds,
    merge_core_only: bool,
) -> Result<usize, TrainingError> {
    let original_chunk = if merge_core_only {
        Some(chunk_scene.clone())
    } else {
        None
    };

    let removed = if merge_core_only {
        let original_len = chunk_scene.len();
        let removed = retain_splats_in_bounds(chunk_scene, core_bounds);
        if original_len > 0 && chunk_scene.is_empty() {
            log::warn!(
                "Core-only merge filter removed all {} gaussians for a chunk; falling back to unfiltered merge",
                original_len,
            );
            if let Some(original_chunk) = original_chunk {
                *chunk_scene = original_chunk;
            }
            0
        } else {
            log::info!(
                "Core-only merge filter removed {} gaussians before aggregation",
                removed,
            );
            removed
        }
    } else {
        0
    };
    if merged_scene.is_empty() {
        merged_scene.sh_degree = chunk_scene.sh_degree();
    } else if !chunk_scene.is_empty() && merged_scene.sh_degree() != chunk_scene.sh_degree() {
        return Err(TrainingError::TrainingFailed(format!(
            "chunk merge requires matching SH degree, got merged={} chunk={}",
            merged_scene.sh_degree(),
            chunk_scene.sh_degree()
        )));
    }
    merged_scene
        .positions
        .extend_from_slice(&chunk_scene.positions);
    merged_scene
        .log_scales
        .extend_from_slice(&chunk_scene.log_scales);
    merged_scene
        .rotations
        .extend_from_slice(&chunk_scene.rotations);
    merged_scene
        .opacity_logits
        .extend_from_slice(&chunk_scene.opacity_logits);
    merged_scene
        .sh_coeffs
        .extend_from_slice(&chunk_scene.sh_coeffs);
    merged_scene.validate().map_err(TrainingError::from)?;
    Ok(removed)
}

pub(crate) fn adapt_chunk_training_config(
    dataset: &TrainingDataset,
    base_config: &TrainingConfig,
) -> Result<ChunkTrainingOverridePlan, TrainingError> {
    let mut effective_config = base_config.clone();
    let mut estimate = metal_memory::estimate_chunk_capacity(dataset, &effective_config)?;
    let mut lowered_gaussian_cap = false;
    let mut lowered_render_scale = false;

    if estimate.requires_subdivision_or_degradation()
        && estimate.affordable_initial_gaussians < effective_config.max_initial_gaussians
    {
        effective_config.max_initial_gaussians = estimate.affordable_initial_gaussians.max(1);
        lowered_gaussian_cap = true;
        estimate = metal_memory::estimate_chunk_capacity(dataset, &effective_config)?;
    }

    while estimate.requires_subdivision_or_degradation()
        && effective_config.metal_render_scale > MIN_RENDER_SCALE
    {
        let next_scale = (effective_config.metal_render_scale * 0.75).max(MIN_RENDER_SCALE);
        if (next_scale - effective_config.metal_render_scale).abs() < f32::EPSILON {
            break;
        }
        effective_config.metal_render_scale = next_scale;
        lowered_render_scale = true;
        estimate = metal_memory::estimate_chunk_capacity(dataset, &effective_config)?;
    }

    if estimate.requires_subdivision_or_degradation() {
        return Err(TrainingError::TrainingFailed(format!(
            "chunk cannot be made safe under the configured budget after chunk-local overrides: chunk_budget_gb={:.2}, max_initial_gaussians={}, metal_render_scale={:.3}, safe_cap={}, estimated_peak≈{:.1} GiB. Recommendations: {}",
            base_config.chunk_budget_gb,
            effective_config.max_initial_gaussians,
            effective_config.metal_render_scale,
            estimate.affordable_initial_gaussians,
            estimate.estimated_peak_gib(),
            estimate.recommendations().join("; "),
        )));
    }

    Ok(ChunkTrainingOverridePlan {
        effective_config,
        estimate,
        lowered_gaussian_cap,
        lowered_render_scale,
    })
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        adapt_chunk_training_config, execute_training_chunks_sequentially, merge_chunk_splats,
    };
    use crate::training::splats::HostSplats;
    use crate::training::{plan_spatial_chunks, TrainingConfig};
    use crate::{ChunkBounds, Intrinsics, ScenePose, TrainingDataset, SE3};
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::rc::Rc;
    use tempfile::tempdir;

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

    fn make_test_splats(entries: &[([f32; 3], [f32; 3])]) -> HostSplats {
        let mut splats = HostSplats::with_sh_degree_capacity(0, entries.len());
        for (position, color) in entries {
            splats.push_rgb(
                *position,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                0.0,
                *color,
            );
        }
        splats
    }

    #[test]
    fn sequential_chunk_executor_runs_chunks_one_at_a_time() {
        let mut dataset = make_dataset(4, 64, 64);
        dataset.add_point([0.0, 0.0, 0.0], None);
        dataset.add_point([1.0, 0.0, 0.0], None);
        dataset.add_point([8.0, 0.0, 0.0], None);
        dataset.add_point([9.0, 0.0, 0.0], None);
        dataset.poses[0].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        dataset.poses[1].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        dataset.poses[2].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[8.0, 0.0, 0.0]);
        dataset.poses[3].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[9.0, 0.0, 0.0]);

        let config = TrainingConfig {
            min_cameras_per_chunk: 1,
            ..TrainingConfig::default()
        };
        let chunk_plan = plan_spatial_chunks(&dataset, &config, Some(2)).unwrap();
        let active = Rc::new(Cell::new(0usize));
        let max_active = Rc::new(Cell::new(0usize));
        let visited = Rc::new(Cell::new(0usize));

        execute_training_chunks_sequentially(&dataset, &chunk_plan, |chunk, materialized| {
            struct Guard {
                active: Rc<Cell<usize>>,
            }

            impl Drop for Guard {
                fn drop(&mut self) {
                    self.active.set(self.active.get().saturating_sub(1));
                }
            }

            active.set(active.get() + 1);
            max_active.set(max_active.get().max(active.get()));
            let _guard = Guard {
                active: Rc::clone(&active),
            };

            visited.set(visited.get() + 1);
            assert_eq!(materialized.dataset.poses.len(), chunk.pose_indices.len());
            assert_eq!(active.get(), 1);
            Ok(())
        })
        .unwrap();

        assert_eq!(visited.get(), chunk_plan.training_chunks().count());
        assert_eq!(max_active.get(), 1);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn adaptive_chunk_config_preserves_or_reduces_requested_budget() {
        let mut dataset = make_dataset(2, 128, 128);
        for idx in 0..64 {
            dataset.add_point([idx as f32 * 0.1, 0.0, 1.0], None);
        }
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 64,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };

        let override_plan = adapt_chunk_training_config(&dataset, &config).unwrap();
        assert!(override_plan.effective_config.max_initial_gaussians <= 64);
        assert!(override_plan.effective_config.metal_render_scale <= 1.0);
        assert!(
            override_plan.estimate.estimated_peak_gib()
                <= override_plan.estimate.effective_budget_gib() + 1e-6
        );
    }

    #[test]
    fn adaptive_chunk_config_reduces_render_scale_when_gaussian_cap_is_not_enough() {
        let dataset = make_dataset(40, 1920, 1080);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 8,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };

        let override_plan = adapt_chunk_training_config(&dataset, &config).unwrap();
        assert!(override_plan.lowered_render_scale);
        assert!(override_plan.effective_config.metal_render_scale < 1.0);
    }

    #[test]
    fn adaptive_chunk_configs_keep_each_trainable_chunk_within_budget_envelope() {
        let mut dataset = make_dataset(6, 1920, 1080);
        dataset.add_point([0.0, 0.0, 1.0], None);
        dataset.add_point([1.0, 0.0, 1.0], None);
        dataset.add_point([2.0, 0.0, 1.0], None);
        dataset.add_point([8.0, 0.0, 1.0], None);
        dataset.add_point([9.0, 0.0, 1.0], None);
        dataset.add_point([10.0, 0.0, 1.0], None);
        dataset.poses[0].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        dataset.poses[1].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        dataset.poses[2].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[2.0, 0.0, 0.0]);
        dataset.poses[3].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[8.0, 0.0, 0.0]);
        dataset.poses[4].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[9.0, 0.0, 0.0]);
        dataset.poses[5].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[10.0, 0.0, 0.0]);

        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 64,
            metal_render_scale: 1.0,
            min_cameras_per_chunk: 1,
            ..TrainingConfig::default()
        };
        let chunk_plan = plan_spatial_chunks(&dataset, &config, Some(3)).unwrap();

        for chunk in chunk_plan.training_chunks() {
            let materialized = crate::materialize_chunk_dataset(&dataset, chunk).unwrap();
            let override_plan =
                adapt_chunk_training_config(&materialized.dataset, &config).unwrap();
            assert!(
                override_plan.estimate.estimated_peak_gib()
                    <= override_plan.estimate.effective_budget_gib() + 1e-6,
                "chunk {} estimate {:.3} GiB exceeded effective budget {:.3} GiB",
                chunk.chunk_id,
                override_plan.estimate.estimated_peak_gib(),
                override_plan.estimate.effective_budget_gib(),
            );
        }
    }

    #[test]
    fn core_only_merge_filter_retains_only_gaussians_inside_core_bounds() {
        let mut merged = HostSplats::default();
        let mut chunk_scene = make_test_splats(&[
            ([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.5, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed =
            merge_chunk_splats(&mut merged, &mut chunk_scene, &core_bounds, true).unwrap();

        assert_eq!(removed, 1);
        assert_eq!(merged.len(), 1);
        assert!((merged.position(0)[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn merge_without_core_filter_keeps_overlap_gaussians() {
        let mut merged = HostSplats::default();
        let mut chunk_scene = make_test_splats(&[
            ([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.5, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed =
            merge_chunk_splats(&mut merged, &mut chunk_scene, &core_bounds, false).unwrap();

        assert_eq!(removed, 0);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn core_only_merge_filter_falls_back_when_it_would_empty_chunk() {
        let mut merged = HostSplats::default();
        let mut chunk_scene = make_test_splats(&[
            ([1.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([2.5, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed =
            merge_chunk_splats(&mut merged, &mut chunk_scene, &core_bounds, true).unwrap();

        assert_eq!(removed, 0);
        assert_eq!(merged.len(), 2);
        assert_eq!(chunk_scene.len(), 2);
    }

    #[test]
    fn merged_scene_output_contains_gaussians_from_multiple_chunks() {
        let mut merged = HostSplats::default();
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        let mut chunk_a = make_test_splats(&[([0.5, 0.0, 0.0], [1.0, 0.0, 0.0])]);
        let mut chunk_b = make_test_splats(&[([2.0, 0.0, 0.0], [0.0, 1.0, 0.0])]);

        merge_chunk_splats(&mut merged, &mut chunk_a, &core_bounds, true).unwrap();
        merge_chunk_splats(
            &mut merged,
            &mut chunk_b,
            &ChunkBounds {
                min: [1.5, -1.0, -1.0],
                max: [2.5, 1.0, 1.0],
            },
            true,
        )
        .unwrap();

        assert_eq!(merged.len(), 2);
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("merged-scene.ply");
        super::super::export::persist_host_splats_scene(&path, &merged, 10).unwrap();
        let (splats, _) = crate::load_splats_ply(&path).unwrap();
        assert_eq!(splats.len(), 2);
    }
}
