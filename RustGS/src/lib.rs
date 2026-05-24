//! RustGS - 3D Gaussian Splatting Training Library
//!
//! This crate provides offline 3DGS training capabilities for RustScan.
//! It takes images and camera poses as input and outputs trained splat sets.
//!
//! # Architecture
//!
//! - `core`: shared training-neutral types such as cameras
//! - `render`: Rendering
//! - `training`: Training loops and optimizers
//! - `io`: Scene file I/O (.splat, PLY, checkpoints)
//! - `init`: Splat initialization from point clouds
//!
//! # Example
//!
//! ```no_run
//! use rustgs::{train_splats, TrainingConfig, TrainingDataset, TrainingOptions};
//! use std::path::PathBuf;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let path = PathBuf::from("dataset.json");
//!
//! // Load a prepared training dataset artifact.
//! let dataset = TrainingDataset::load(&path)?;
//!
//! // Train 3DGS splats.
//! let config = TrainingConfig::default();
//! let run = train_splats(&dataset, &config, TrainingOptions::default())?;
//!
//! // Save the trained splats.
//! rustgs::save_splats(
//!     "scene.splat".as_ref(),
//!     &run.splats,
//!     &rustgs::SplatMetadata::default(),
//! )?;
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod init;
pub mod io;
pub mod render;
mod sh;
pub mod training;

use std::path::Path;

pub use rustscan_types::{Intrinsics, MapPointData, ScenePose, TrainingDataset, SE3};

// Re-export core types
pub use crate::core::{GaussianCamera, HostSplats, SplatView};

// Re-export render types
pub use crate::render::{
    GaussianRenderer, ProjectedGaussian, RenderBuffer, RenderOutput, TiledRenderer,
};

// Re-export training types
#[cfg(feature = "gpu")]
pub use crate::training::SplatEvaluationRenderer;
pub use crate::training::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, LiteGsCameraConfig, LiteGsConfig, LiteGsFeatureConfig,
    LiteGsGrowthConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsPruningConfig,
    LiteGsRefineConfig, LiteGsRenderingConfig, LiteGsSplitScoreMode, LiteGsTileSize,
    LiteGsTopologyConfig, LiteGsTrainingProfile, ParityCheckOutcome, ParityCheckStatus,
    ParityFixtureKind, ParityFixtureSpec, ParityFloatDistribution, ParityGateEvaluation,
    ParityGateStatus, ParityHarnessReport, ParityLossCurveSample, ParityLossTerms,
    ParityMetricSnapshot, ParityReferenceComparison, ParityThresholds, ParityTimingMetrics,
    ParityTopologyMetrics, ParityTopologyStepSample, PsnrSummary, SplatEvaluationConfig,
    SplatEvaluationError, SplatEvaluationResult, SplatEvaluationSummary, TrainingDataConfig,
    TrainingInitializationConfig, TrainingLossConfig, TrainingOptimizerConfig,
    TrainingRasterConfig, DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_RASTER_COV_BLUR,
    DEFAULT_TINY_FIXTURE_ID,
};
pub use crate::training::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics,
};
#[cfg(feature = "gpu")]
pub use crate::training::{
    evaluate_splats, evaluation_device, last_training_telemetry, render_evaluation_frame,
    runtime_from_splats, LiteGsOptimizerLrs, LiteGsTrainingTelemetry, TrainingControl,
    TrainingEvent, TrainingEventCadence, TrainingEventRoute, TrainingIterationProgress,
    TrainingOptions, TrainingPlanSelected, TrainingRun, TrainingRunCancelled, TrainingRunCompleted,
    TrainingRunReport, TrainingRunStarted, TrainingSnapshotReady,
};
pub use crate::training::{TrainingBackend, TrainingConfig, TrainingResult};

// Re-export IO types
pub use crate::io::colmap_dataset::{load_colmap_dataset, ColmapConfig};
pub use crate::io::nerfstudio_dataset::{load_nerfstudio_dataset, NerfstudioConfig};
#[cfg(feature = "gpu")]
pub use crate::io::scene_io::{
    load_splats, load_splats_ply, load_splats_splat, save_splats, save_splats_ply,
    save_splats_splat,
};
pub use crate::io::scene_io::{SceneIoError, SplatMetadata};
pub use crate::io::tum_dataset::{load_tum_rgbd_dataset, TumRgbdConfig};
#[cfg(feature = "gpu")]
pub use crate::io::TrainingCheckpoint;

// Re-export initialization types
#[cfg(feature = "gpu")]
pub use crate::init::initialize_host_splats_from_points;
pub use crate::init::GaussianInitConfig;

#[cfg(not(feature = "gpu"))]
pub fn gpu_available() -> bool {
    false
}

#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    true
}

/// Load a training dataset from a TUM RGB-D directory, a COLMAP directory,
/// a Nerfstudio transforms directory, or a serialized `TrainingDataset` JSON file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingInputKind {
    TumRgbd,
    Colmap,
    Nerfstudio,
    TrainingDatasetJson,
}

impl std::fmt::Display for TrainingInputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TumRgbd => write!(f, "TUM RGB-D dataset"),
            Self::Colmap => write!(f, "COLMAP dataset"),
            Self::Nerfstudio => write!(f, "Nerfstudio dataset"),
            Self::TrainingDatasetJson => write!(f, "TrainingDataset JSON"),
        }
    }
}

/// Load a training dataset and report which input type was resolved.
pub fn load_training_dataset_with_source(
    input: &Path,
    tum_config: &TumRgbdConfig,
    colmap_config: &ColmapConfig,
) -> Result<(TrainingDataset, TrainingInputKind), TrainingError> {
    if input.is_dir() {
        if io::colmap_dataset::resolve_colmap_sparse_dir(input).is_ok() {
            return load_colmap_dataset(input, colmap_config)
                .map(|dataset| (dataset, TrainingInputKind::Colmap));
        }

        if io::nerfstudio_dataset::looks_like_nerfstudio_dataset(input) {
            return load_nerfstudio_dataset(
                input,
                &NerfstudioConfig {
                    max_frames: colmap_config.max_frames,
                    frame_stride: colmap_config.frame_stride,
                },
            )
            .map(|dataset| (dataset, TrainingInputKind::Nerfstudio));
        }

        return load_tum_rgbd_dataset(input, tum_config)
            .map(|dataset| (dataset, TrainingInputKind::TumRgbd));
    }

    let input_buf = input.to_path_buf();
    TrainingDataset::load(&input_buf)
        .map(|dataset| (dataset, TrainingInputKind::TrainingDatasetJson))
        .map_err(|err| {
            TrainingError::InvalidInput(format!(
                "failed to load {} as TrainingDataset JSON: {}",
                input.display(),
                err,
            ))
        })
}

/// Load a training dataset from a TUM RGB-D directory, a COLMAP directory,
/// a Nerfstudio transforms directory, or a serialized `TrainingDataset` JSON file.
pub fn load_training_dataset(
    input: &Path,
    tum_config: &TumRgbdConfig,
) -> Result<TrainingDataset, TrainingError> {
    let colmap_config = ColmapConfig {
        max_frames: tum_config.max_frames,
        frame_stride: tum_config.frame_stride,
        depth_scale: 1.0,
    };
    load_training_dataset_with_source(input, tum_config, &colmap_config).map(|(dataset, _)| dataset)
}

/// Train 3DGS splats from a prepared training dataset.
#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    options: TrainingOptions<'_>,
) -> Result<TrainingRun, TrainingError> {
    training::train_splats(dataset, config, options)
}

#[cfg(test)]
mod tests;
/// Training error type.
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),
}
