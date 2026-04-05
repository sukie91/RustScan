//! RustGS - 3D Gaussian Splatting Training Library
//!
//! This crate provides offline 3DGS training capabilities for RustScan.
//! It takes images and camera poses as input and outputs trained Gaussian scenes.
//!
//! # Architecture
//!
//! - `core`: Gaussian data structures and camera types
//! - `render`: Rendering (forward and differentiable)
//! - `diff`: Differentiable rendering with Candle
//! - `training`: Training loops and optimizers
//! - `io`: Scene file I/O (PLY, checkpoints)
//! - `init`: Gaussian initialization from point clouds
//!
//! # Example
//!
//! ```ignore
//! use rustgs::{TrainingDataset, train_from_slam, TrainingConfig};
//! use rustscan_types::SlamOutput;
//!
//! // Load SLAM output
//! let slam_output = SlamOutput::load(&path)?;
//!
//! // Train 3DGS scene
//! let config = TrainingConfig::default();
//! let scene = train_from_slam(&slam_output, &config)?;
//!
//! // Save scene
//! scene.save("scene.ply")?;
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

// Re-export shared types from rustscan-types
pub use rustscan_types::{Intrinsics, MapPointData, ScenePose, SlamOutput, TrainingDataset, SE3};

// Re-export core types
pub use crate::core::{Gaussian3D, GaussianCamera, GaussianMap, GaussianState};

// Re-export render types
pub use crate::render::{
    densify, prune, Gaussian, GaussianRenderer, ProjectedGaussian, RenderBuffer, RenderOutput,
    TiledRenderer,
};

// Re-export training types
pub use crate::training::{
    default_litegs_parity_fixtures, default_parity_report_path, parity_fixture_id_for_input_path,
    LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize, ParityFixtureKind,
    ParityFixtureSpec, ParityHarnessReport, ParityLossTerms, ParityMetricSnapshot,
    ParityThresholds, ParityTimingMetrics, ParityTopologyMetrics, TrainingProfile,
    DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};
#[cfg(feature = "gpu")]
pub use crate::training::{
    estimate_chunk_capacity, last_metal_training_telemetry, ChunkCapacityDisposition,
    ChunkCapacityEstimate, LiteGsOptimizerLrs, LiteGsTrainingTelemetry,
};
pub use crate::training::{
    materialize_chunk_dataset, plan_spatial_chunks, ChunkBounds, ChunkBoundsSource,
    ChunkDisposition, ChunkPlan, MaterializedChunkDataset, PlannedChunk,
};
pub use crate::training::{TrainingBackend, TrainingConfig, TrainingResult};

// Re-export IO types
pub use crate::io::colmap_dataset::{load_colmap_dataset, ColmapConfig};
pub use crate::io::scene_io::{load_scene_ply, save_scene_ply, SceneIoError, SceneMetadata};
pub use crate::io::tum_dataset::{load_tum_rgbd_dataset, TumRgbdConfig};
pub use crate::io::TrainingCheckpoint;

// Re-export initialization types
pub use crate::init::GaussianInitConfig;
pub use crate::init::{initialize_gaussian3d_from_points, initialize_gaussians_from_points};

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

/// Initialize Gaussians from a point cloud (convenience wrapper).
pub fn initialize_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
) -> Vec<Gaussian3D> {
    initialize_gaussian3d_from_points(points, config)
}

/// Load a training dataset from a TUM RGB-D directory, a SLAM output JSON file,
/// a COLMAP directory, or a serialized `TrainingDataset` JSON file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingInputKind {
    TumRgbd,
    Colmap,
    SlamOutputJson,
    TrainingDatasetJson,
}

impl std::fmt::Display for TrainingInputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TumRgbd => write!(f, "TUM RGB-D dataset"),
            Self::Colmap => write!(f, "COLMAP dataset"),
            Self::SlamOutputJson => write!(f, "SlamOutput JSON"),
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
    match SlamOutput::load(&input_buf) {
        Ok(slam_output) => Ok((slam_output.to_dataset(), TrainingInputKind::SlamOutputJson)),
        Err(slam_err) => match TrainingDataset::load(&input_buf) {
            Ok(dataset) => Ok((dataset, TrainingInputKind::TrainingDatasetJson)),
            Err(dataset_err) => Err(TrainingError::InvalidInput(format!(
                "failed to load {} as SlamOutput JSON ({}) or TrainingDataset JSON ({})",
                input.display(),
                slam_err,
                dataset_err,
            ))),
        },
    }
}

/// Load a training dataset from a TUM RGB-D directory, a COLMAP directory, a
/// serialized `SlamOutput` JSON file, or a serialized `TrainingDataset` JSON file.
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

/// Train a 3DGS scene from a SLAM output.
///
/// This is the main entry point for offline 3DGS training.
///
/// # Arguments
/// * `slam_output` - SLAM output containing images, poses, and initial points
/// * `config` - Training configuration
///
/// # Returns
/// The trained Gaussian scene.
#[cfg(feature = "gpu")]
pub fn train_from_slam(
    slam_output: &SlamOutput,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let dataset = slam_output.to_dataset();
    training::train(&dataset, config)
}

/// Train a 3DGS scene directly from a dataset path.
///
/// `input` can be a TUM RGB-D dataset directory, a COLMAP directory, a
/// serialized `SlamOutput` JSON file, or a serialized `TrainingDataset` JSON file.
#[cfg(feature = "gpu")]
pub fn train_from_path(
    input: &Path,
    tum_config: &TumRgbdConfig,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let dataset = load_training_dataset(input, tum_config)?;
    training::train(&dataset, config)
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
