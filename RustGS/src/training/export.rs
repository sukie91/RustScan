use super::{
    ChunkPlan, ChunkTrainingOverridePlan, MaterializedChunkDataset, PlannedChunk, TrainingConfig,
};
use crate::{GaussianMap, TrainingError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChunkReportStatus {
    Success,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkReportEntry {
    chunk_id: usize,
    status: ChunkReportStatus,
    message: Option<String>,
    scene_path: Option<PathBuf>,
    pose_count: usize,
    local_points: usize,
    used_frame_based_initialization: bool,
    effective_max_initial_gaussians: Option<usize>,
    effective_render_scale: Option<f32>,
    estimated_peak_gib: Option<f64>,
    output_gaussians: Option<usize>,
    core_filter_removed: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkTrainingReport {
    merge_core_only: bool,
    requested_chunks: usize,
    trainable_chunks: usize,
    entries: Vec<ChunkReportEntry>,
}

pub(super) struct ChunkPersistenceContext {
    output_dir: Option<PathBuf>,
    report: ChunkTrainingReport,
}

impl ChunkPersistenceContext {
    pub(super) fn new(
        output_dir: Option<PathBuf>,
        config: &TrainingConfig,
        chunk_plan: &ChunkPlan,
    ) -> Result<Self, TrainingError> {
        if let Some(dir) = output_dir.as_ref() {
            fs::create_dir_all(dir).map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "failed to create chunk artifact directory {}: {}",
                    dir.display(),
                    err
                ))
            })?;
        }

        let mut this = Self {
            output_dir,
            report: ChunkTrainingReport {
                merge_core_only: config.merge_core_only,
                requested_chunks: chunk_plan.chunks.len(),
                trainable_chunks: chunk_plan.trainable_chunks().count(),
                entries: Vec::new(),
            },
        };
        this.flush_report()?;
        Ok(this)
    }

    fn report_path(&self) -> Option<PathBuf> {
        self.output_dir
            .as_ref()
            .map(|dir| dir.join("chunk-report.json"))
    }

    fn scene_path(&self, chunk_id: usize) -> Option<PathBuf> {
        self.output_dir
            .as_ref()
            .map(|dir| dir.join(format!("chunk-{chunk_id:03}.ply")))
    }

    pub(super) fn record_skipped(
        &mut self,
        chunk: &PlannedChunk,
        config: &TrainingConfig,
    ) -> Result<(), TrainingError> {
        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Skipped,
            message: Some(format!(
                "insufficient cameras: assigned {} < required {}",
                chunk.pose_indices.len(),
                config.min_cameras_per_chunk
            )),
            scene_path: None,
            pose_count: chunk.pose_indices.len(),
            local_points: chunk.point_indices.len(),
            used_frame_based_initialization: chunk.point_indices.is_empty(),
            effective_max_initial_gaussians: None,
            effective_render_scale: None,
            estimated_peak_gib: None,
            output_gaussians: None,
            core_filter_removed: None,
        });
        self.flush_report()
    }

    pub(super) fn record_success(
        &mut self,
        chunk: &PlannedChunk,
        materialized: &MaterializedChunkDataset,
        override_plan: &ChunkTrainingOverridePlan,
        chunk_scene: &GaussianMap,
        core_filter_removed: usize,
    ) -> Result<(), TrainingError> {
        let scene_path = if let Some(path) = self.scene_path(chunk.chunk_id) {
            persist_gaussian_map_scene(
                &path,
                chunk_scene,
                override_plan.effective_config.iterations,
            )?;
            Some(path)
        } else {
            None
        };

        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Success,
            message: None,
            scene_path,
            pose_count: materialized.dataset.poses.len(),
            local_points: materialized.dataset.initial_points.len(),
            used_frame_based_initialization: materialized.used_frame_based_initialization,
            effective_max_initial_gaussians: Some(
                override_plan.effective_config.max_initial_gaussians,
            ),
            effective_render_scale: Some(override_plan.effective_config.metal_render_scale),
            estimated_peak_gib: Some(override_plan.estimate.estimated_peak_gib()),
            output_gaussians: Some(chunk_scene.len()),
            core_filter_removed: Some(core_filter_removed),
        });
        self.flush_report()
    }

    pub(super) fn record_failure(
        &mut self,
        chunk: &PlannedChunk,
        materialized: &MaterializedChunkDataset,
        message: String,
        override_plan: Option<&ChunkTrainingOverridePlan>,
    ) -> Result<(), TrainingError> {
        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Failed,
            message: Some(message),
            scene_path: None,
            pose_count: materialized.dataset.poses.len(),
            local_points: materialized.dataset.initial_points.len(),
            used_frame_based_initialization: materialized.used_frame_based_initialization,
            effective_max_initial_gaussians: override_plan
                .map(|plan| plan.effective_config.max_initial_gaussians),
            effective_render_scale: override_plan
                .map(|plan| plan.effective_config.metal_render_scale),
            estimated_peak_gib: override_plan.map(|plan| plan.estimate.estimated_peak_gib()),
            output_gaussians: None,
            core_filter_removed: None,
        });
        self.flush_report()
    }

    fn flush_report(&mut self) -> Result<(), TrainingError> {
        let Some(report_path) = self.report_path() else {
            return Ok(());
        };
        let json = serde_json::to_string_pretty(&self.report).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to serialize chunk report {}: {}",
                report_path.display(),
                err
            ))
        })?;
        fs::write(&report_path, json).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to write chunk report {}: {}",
                report_path.display(),
                err
            ))
        })
    }
}

pub(super) fn persist_gaussian_map_scene(
    path: &Path,
    scene: &GaussianMap,
    iterations: usize,
) -> Result<(), TrainingError> {
    let gaussians = scene
        .gaussians()
        .iter()
        .map(crate::Gaussian::from_gaussian3d)
        .collect::<Vec<_>>();
    let color_representation = gaussians
        .first()
        .map(|gaussian| gaussian.color_representation)
        .unwrap_or_default();
    let metadata = crate::SceneMetadata {
        iterations,
        final_loss: 0.0,
        gaussian_count: gaussians.len(),
        color_representation,
        ..crate::SceneMetadata::default()
    };
    crate::save_scene_ply(path, &gaussians, &metadata).map_err(|err| {
        TrainingError::TrainingFailed(format!("failed to persist {}: {}", path.display(), err))
    })
}
