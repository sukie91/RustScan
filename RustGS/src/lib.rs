//! RustGS - 3D Gaussian Splatting Training Library
//!
//! This crate provides offline 3DGS training capabilities for RustScan.
//! It takes images and camera poses as input and outputs trained splat sets.
//!
//! # Architecture
//!
//! - `core`: shared training-neutral types such as cameras
//! - `render`: Rendering (forward and differentiable)
//! - `diff`: Differentiable rendering with Candle
//! - `training`: Training loops and optimizers
//! - `io`: Scene file I/O (PLY, checkpoints)
//! - `init`: Splat initialization from point clouds
//!
//! # Example
//!
//! ```ignore
//! use rustgs::{train_splats, TrainingConfig, TrainingDataset};
//!
//! // Load a prepared training dataset artifact.
//! let dataset = TrainingDataset::load(&path)?;
//!
//! // Train 3DGS splats.
//! let config = TrainingConfig::default();
//! let splats = train_splats(&dataset, &config)?;
//!
//! // Save the trained splats.
//! rustgs::save_splats_ply(
//!     "scene.ply".as_ref(),
//!     &splats,
//!     &rustgs::SplatMetadata::default(),
//! )?;
//! ```

pub mod core;
pub mod diff;
pub mod init;
pub mod io;
pub mod render;
pub mod training;

#[cfg(feature = "gpu")]
use candle_core::Device;
use std::path::Path;
#[cfg(feature = "gpu")]
use std::sync::{Mutex, OnceLock};

pub use rustscan_types::{Intrinsics, MapPointData, ScenePose, TrainingDataset, SE3};

// Re-export core types
pub use crate::core::GaussianCamera;

// Re-export render types
pub use crate::render::{
    GaussianRenderer, ProjectedGaussian, RenderBuffer, RenderOutput, TiledRenderer,
};

// Re-export training types
#[cfg(feature = "gpu")]
pub use crate::diff::Splats;
pub use crate::training::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize,
    ParityCheckOutcome, ParityCheckStatus, ParityFixtureKind, ParityFixtureSpec,
    ParityGateEvaluation, ParityGateStatus, ParityHarnessReport, ParityLossCurveSample,
    ParityLossTerms, ParityMetricSnapshot, ParityReferenceComparison, ParityThresholds,
    ParityTimingMetrics, ParityTopologyMetrics, PsnrSummary, SplatEvaluationConfig,
    SplatEvaluationError, SplatEvaluationResult, SplatEvaluationSummary, TrainingProfile,
    DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};
pub use crate::training::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics,
};
#[cfg(feature = "gpu")]
pub use crate::training::{
    evaluate_splats, evaluation_device, last_metal_training_telemetry, render_evaluation_frame,
    run_metal_training_benchmark, runtime_from_splats, LiteGsOptimizerLrs, LiteGsTrainingTelemetry,
    MetalTrainingBenchmarkReport, MetalTrainingBenchmarkSpec, TrainingEvent, TrainingEventRoute,
    TrainingPlanSelected, TrainingRun, TrainingRunCompleted, TrainingRunReport, TrainingRunStarted,
};
#[cfg(feature = "gpu")]
pub use crate::training::{HostSplats, SplatEvaluationRenderer, SplatView};
pub use crate::training::{TrainingBackend, TrainingConfig, TrainingResult};

// Re-export IO types
pub use crate::io::colmap_dataset::{load_colmap_dataset, ColmapConfig};
#[cfg(feature = "gpu")]
pub use crate::io::scene_io::{load_splats_ply, save_splats_ply};
pub use crate::io::scene_io::{SceneIoError, SplatMetadata};
pub use crate::io::tum_dataset::{load_tum_rgbd_dataset, TumRgbdConfig};
#[cfg(feature = "gpu")]
pub use crate::io::TrainingCheckpoint;

// Re-export initialization types
pub use crate::init::GaussianInitConfig;
#[cfg(feature = "gpu")]
pub use crate::init::{initialize_host_splats_from_points, initialize_runtime_splats_from_points};

#[cfg(feature = "gpu")]
pub(crate) fn preferred_device() -> Device {
    match try_metal_device() {
        Ok(device) => device,
        Err(err) => {
            log::warn!("{err}; falling back to CPU");
            Device::Cpu
        }
    }
}

#[cfg(feature = "gpu")]
pub fn metal_available() -> bool {
    try_metal_device().is_ok()
}

#[cfg(not(feature = "gpu"))]
pub fn metal_available() -> bool {
    false
}

#[cfg(feature = "gpu")]
pub(crate) fn require_metal_device() -> Result<Device, TrainingError> {
    try_metal_device().map_err(TrainingError::Gpu)
}

#[cfg(feature = "gpu")]
pub(crate) fn try_metal_device() -> Result<Device, String> {
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let hook_lock = PANIC_HOOK_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = hook_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| Device::new_metal(0));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(Ok(device)) => Ok(device),
        Ok(Err(err)) => Err(format!("Metal unavailable: {err}")),
        Err(_) => Err("Metal initialization panicked".to_string()),
    }
}

/// Load a training dataset from a TUM RGB-D directory, a COLMAP directory,
/// or a serialized `TrainingDataset` JSON file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingInputKind {
    TumRgbd,
    Colmap,
    TrainingDatasetJson,
}

impl std::fmt::Display for TrainingInputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TumRgbd => write!(f, "TUM RGB-D dataset"),
            Self::Colmap => write!(f, "COLMAP dataset"),
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
/// or a serialized `TrainingDataset` JSON file.
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

/// Train 3DGS splats directly from a prepared training dataset.
#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    training::train_splats(dataset, config)
}

/// Train 3DGS splats and return a structured report for downstream consumers.
#[cfg(feature = "gpu")]
pub fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    training::train_splats_with_report(dataset, config)
}

/// Train 3DGS splats while emitting structured training events.
#[cfg(feature = "gpu")]
pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    training::train_splats_with_events(dataset, config, on_event)
}

/// Compatibility adapter for path-based dataset loading that returns the host-side splat artifact.
#[cfg(feature = "gpu")]
pub fn train_splats_from_path(
    input: &Path,
    tum_config: &TumRgbdConfig,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    let dataset = load_training_dataset(input, tum_config)?;
    training::train_splats(&dataset, config)
}

/// Compatibility adapter for path-based dataset loading that returns training artifacts and report.
#[cfg(feature = "gpu")]
pub fn train_splats_from_path_with_report(
    input: &Path,
    tum_config: &TumRgbdConfig,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let dataset = load_training_dataset(input, tum_config)?;
    training::train_splats_with_report(&dataset, config)
}

/// Path-based training entry point with structured event emission.
#[cfg(feature = "gpu")]
pub fn train_splats_from_path_with_events<F>(
    input: &Path,
    tum_config: &TumRgbdConfig,
    config: &TrainingConfig,
    on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    let dataset = load_training_dataset(input, tum_config)?;
    training::train_splats_with_events(&dataset, config, on_event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_colmap_test_dataset(root: &Path) {
        let sparse = root.join("sparse").join("0");
        std::fs::create_dir_all(&sparse).unwrap();
        let images = root.join("images");
        std::fs::create_dir_all(&images).unwrap();

        std::fs::write(
            sparse.join("cameras.txt"),
            "# Camera list with one line of data per camera:\n1 PINHOLE 640 480 500 500 320 240\n",
        )
        .unwrap();
        std::fs::write(
            sparse.join("images.txt"),
            "# Image list with two lines of data per image:\n1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 frame_0001.jpg\n",
        )
        .unwrap();
        std::fs::write(
            sparse.join("points3D.txt"),
            "# 3D point list with one line of data per point:\n1 0.0 0.0 1.0 128 128 128 0.1\n",
        )
        .unwrap();
        std::fs::write(images.join("frame_0001.jpg"), vec![0u8; 640 * 480 * 3]).unwrap();
    }

    #[test]
    fn test_load_training_dataset_with_source_detects_colmap_directory() {
        let temp = tempdir().unwrap();
        write_colmap_test_dataset(temp.path());

        let (dataset, source) = load_training_dataset_with_source(
            temp.path(),
            &TumRgbdConfig::default(),
            &ColmapConfig::default(),
        )
        .unwrap();

        assert_eq!(source, TrainingInputKind::Colmap);
        assert_eq!(dataset.poses.len(), 1);
        assert_eq!(dataset.initial_points.len(), 1);
    }
}

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

#[cfg(feature = "gpu")]
impl From<candle_core::Error> for TrainingError {
    fn from(e: candle_core::Error) -> Self {
        TrainingError::Gpu(e.to_string())
    }
}
