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
pub mod render;
pub mod diff;
pub mod training;
pub mod io;
pub mod init;

// Re-export shared types from rustscan-types
pub use rustscan_types::{SE3, Intrinsics, TrainingDataset, ScenePose, SlamOutput, MapPointData};

// Re-export core types
pub use crate::core::{Gaussian3D, GaussianMap, GaussianCamera, GaussianState};

// Re-export render types
pub use crate::render::{
    GaussianRenderer, RenderOutput,
    Gaussian, ProjectedGaussian, TiledRenderer, RenderBuffer,
    densify, prune,
};

// Re-export training types
pub use crate::training::{TrainingConfig, TrainingResult};

// Re-export IO types
pub use crate::io::TrainingCheckpoint;
pub use crate::io::scene_io::{save_scene_ply, load_scene_ply, SceneMetadata, SceneIoError};
#[cfg(feature = "gpu")]
pub use crate::io::training_checkpoint::{
    FullTrainingCheckpoint, CheckpointGaussian,
    save_checkpoint, load_checkpoint, load_latest_checkpoint, resume_latest_checkpoint,
    TrainingCheckpointError,
};

// Re-export initialization types
pub use crate::init::GaussianInitConfig;
pub use crate::init::{initialize_gaussians_from_points, initialize_gaussian3d_from_points};

/// Initialize Gaussians from a point cloud (convenience wrapper).
pub fn initialize_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
) -> Vec<Gaussian3D> {
    initialize_gaussian3d_from_points(points, config)
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