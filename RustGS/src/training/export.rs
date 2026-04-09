use super::chunk_training::ChunkTrainingOverridePlan;
use super::splats::HostSplats;
use super::{ChunkPlan, MaterializedChunkDataset, PlannedChunk, TrainingConfig};
use crate::TrainingError;
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
    #[serde(alias = "trainable_chunks")]
    training_chunks: usize,
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
                training_chunks: chunk_plan.training_chunks().count(),
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
        chunk_splats: &HostSplats,
        core_filter_removed: usize,
    ) -> Result<(), TrainingError> {
        let scene_path = if let Some(path) = self.scene_path(chunk.chunk_id) {
            persist_host_splats_scene(
                &path,
                chunk_splats,
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
            output_gaussians: Some(chunk_splats.len()),
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

pub(super) fn persist_host_splats_scene(
    path: &Path,
    splats: &HostSplats,
    iterations: usize,
) -> Result<(), TrainingError> {
    let metadata = splats.to_scene_metadata(iterations, 0.0);
    crate::save_splats_ply(path, splats, &metadata).map_err(|err| {
        TrainingError::TrainingFailed(format!("failed to persist {}: {}", path.display(), err))
    })
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{persist_host_splats_scene, ChunkPersistenceContext};
    use crate::diff::diff_splat::rgb_to_sh0_value;
    use crate::training::chunk_training::ChunkTrainingOverridePlan;
    use crate::training::splats::HostSplats;
    use crate::training::{PlannedChunk, TrainingConfig};
    use crate::{
        ChunkBounds, ChunkBoundsSource, ChunkDisposition, Intrinsics, MaterializedChunkDataset,
        ScenePose, TrainingDataset, SE3,
    };
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn persist_host_splats_scene_writes_chunk_ply() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("chunk-000.ply");
        let splats = HostSplats::from_raw_parts(
            vec![0.0, 0.0, 0.0],
            vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![
                rgb_to_sh0_value(0.5),
                rgb_to_sh0_value(0.5),
                rgb_to_sh0_value(0.5),
            ],
            0,
        )
        .unwrap();

        persist_host_splats_scene(&path, &splats, 42).unwrap();

        assert!(path.exists());
        let (loaded_splats, metadata) = crate::load_splats_ply(&path).unwrap();
        assert_eq!(loaded_splats.len(), 1);
        assert_eq!(metadata.iterations, 42);
    }

    #[test]
    fn persist_host_splats_scene_preserves_spherical_harmonics_metadata() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("chunk-sh.ply");
        let mut sh_coeffs = vec![
            rgb_to_sh0_value(0.2),
            rgb_to_sh0_value(0.4),
            rgb_to_sh0_value(0.6),
        ];
        sh_coeffs.extend(vec![0.5; 15 * 3]);
        let splats = HostSplats::from_raw_parts(
            vec![1.0, 2.0, 3.0],
            vec![0.1f32.ln(), 0.2f32.ln(), 0.3f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![(0.25f32 / (1.0 - 0.25)).ln()],
            sh_coeffs,
            3,
        )
        .unwrap();

        persist_host_splats_scene(&path, &splats, 42).unwrap();

        let (gaussians, metadata) = crate::load_splats_ply(&path).unwrap();
        let gaussians = gaussians.to_scene_gaussians().unwrap();
        assert_eq!(metadata.iterations, 42);
        assert_eq!(metadata.sh_degree, 3);
        assert_eq!(gaussians.len(), 1);
        assert_eq!(gaussians[0].sh_rest.as_deref().unwrap(), &vec![0.5; 15 * 3]);
        assert!((gaussians[0].color[0] - 0.2).abs() < 1e-5);
        assert!((gaussians[0].color[1] - 0.4).abs() < 1e-5);
        assert!((gaussians[0].color[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn chunk_persistence_writes_report_entries() {
        let tempdir = tempdir().unwrap();
        let chunk = PlannedChunk {
            chunk_id: 0,
            core_bounds: ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            overlap_bounds: ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            pose_indices: vec![0, 1],
            point_indices: vec![0],
            disposition: ChunkDisposition::Trainable,
        };
        let chunk_plan = crate::ChunkPlan {
            bounds_source: ChunkBoundsSource::SparsePoints,
            scene_bounds: chunk.core_bounds,
            chunk_axis: 0,
            requested_chunks: 1,
            chunks: vec![chunk.clone()],
        };
        let config = TrainingConfig {
            chunked_training: true,
            chunk_artifact_dir: Some(tempdir.path().join("artifacts")),
            ..TrainingConfig::default()
        };
        let dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 32, 32));
        let materialized = MaterializedChunkDataset {
            chunk_id: 0,
            dataset,
            used_frame_based_initialization: true,
        };
        let override_plan = ChunkTrainingOverridePlan {
            effective_config: config.clone(),
            estimate: crate::estimate_chunk_capacity(
                &TrainingDataset {
                    intrinsics: Intrinsics::from_focal(500.0, 32, 32),
                    depth_scale: 1000.0,
                    poses: vec![ScenePose::new(
                        0,
                        PathBuf::from("frame.png"),
                        SE3::identity(),
                        0.0,
                    )],
                    initial_points: vec![],
                },
                &config,
            )
            .unwrap(),
            lowered_gaussian_cap: false,
            lowered_render_scale: false,
        };
        let splats = HostSplats::from_raw_parts(
            vec![0.0, 0.0, 0.0],
            vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![
                rgb_to_sh0_value(0.5),
                rgb_to_sh0_value(0.5),
                rgb_to_sh0_value(0.5),
            ],
            0,
        )
        .unwrap();

        let mut persistence =
            ChunkPersistenceContext::new(config.chunk_artifact_dir.clone(), &config, &chunk_plan)
                .unwrap();
        persistence
            .record_success(&chunk, &materialized, &override_plan, &splats, 0)
            .unwrap();

        let report_path = tempdir.path().join("artifacts").join("chunk-report.json");
        assert!(report_path.exists());
        let report_json = std::fs::read_to_string(report_path).unwrap();
        let report: serde_json::Value = serde_json::from_str(&report_json).unwrap();
        assert_eq!(report["entries"][0]["status"], "success");
        assert_eq!(report["entries"][0]["output_gaussians"], 1);
        assert_eq!(
            report["entries"][0]["used_frame_based_initialization"],
            true
        );
    }
}
