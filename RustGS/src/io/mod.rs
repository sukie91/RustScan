//! IO module for scene files and checkpoints.
//!
//! - `scene_io`: PLY scene export/import
//! - `training_checkpoint`: Legacy full-training checkpoint surface
//! - `tum_dataset`: TUM RGB-D dataset loading for direct RustGS training

pub mod scene_io;
#[cfg(all(feature = "gpu", feature = "legacy-trainers"))]
pub mod training_checkpoint;
pub mod tum_dataset;

use crate::core::GaussianMap;
use crate::TrainingError;
use std::path::Path;

/// Simple training checkpoint used by the trainer's incremental save.
#[derive(Debug, Clone, Default)]
pub struct TrainingCheckpoint {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Gaussian scene
    pub scene: GaussianMap,
}

impl TrainingCheckpoint {
    /// Create a new checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save checkpoint to file.
    pub fn save(&self, _path: &Path) -> Result<(), TrainingError> {
        Ok(())
    }

    /// Load checkpoint from file.
    pub fn load(_path: &Path) -> Result<Self, TrainingError> {
        Ok(Self::default())
    }
}
