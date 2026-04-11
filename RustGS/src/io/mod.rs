//! IO module for scene files and checkpoints.
//!
//! - `scene_io`: PLY scene export/import
//! - `tum_dataset`: TUM RGB-D dataset loading for direct RustGS training
//! - `colmap_dataset`: COLMAP dataset loading

pub mod colmap_dataset;
pub mod scene_io;
pub mod tum_dataset;

#[cfg(feature = "gpu")]
use crate::training::HostSplats;
use crate::TrainingError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Simple training checkpoint used by the trainer's incremental save.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Host-side splat artifact captured at the checkpoint boundary.
    pub splats: HostSplats,
}

#[cfg(feature = "gpu")]
impl Default for TrainingCheckpoint {
    fn default() -> Self {
        Self {
            iteration: 0,
            loss: 0.0,
            splats: HostSplats::default(),
        }
    }
}

#[cfg(feature = "gpu")]
impl TrainingCheckpoint {
    /// Create a new checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save checkpoint to file.
    pub fn save(&self, path: &Path) -> Result<(), TrainingError> {
        let serialized = serde_json::to_vec_pretty(self).map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to serialize checkpoint: {err}"))
        })?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load checkpoint from file.
    pub fn load(path: &Path) -> Result<Self, TrainingError> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to deserialize checkpoint {}: {err}",
                path.display()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TrainingCheckpoint;
    use crate::diff::diff_splat::rgb_to_sh0_value;
    use tempfile::tempdir;

    #[test]
    fn checkpoint_round_trips_host_splats() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("checkpoint.json");
        let splats = crate::training::HostSplats::from_raw_parts(
            vec![0.0, 0.0, 1.0],
            vec![0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.3),
                rgb_to_sh0_value(0.4),
            ],
            0,
        )
        .unwrap();
        let checkpoint = TrainingCheckpoint {
            iteration: 12,
            loss: 0.25,
            splats,
        };

        checkpoint.save(&path).unwrap();
        let loaded = TrainingCheckpoint::load(&path).unwrap();

        assert_eq!(loaded.iteration, 12);
        assert!((loaded.loss - 0.25).abs() < 1e-6);
        assert_eq!(loaded.splats.len(), 1);
    }
}
