//! IO module for scene files and checkpoints.
//!
//! - `scene_io`: PLY scene export/import
//! - `tum_dataset`: TUM RGB-D dataset loading for direct RustGS training
//! - `colmap_dataset`: COLMAP dataset loading

pub mod colmap_dataset;
pub mod scene_io;
pub mod tum_dataset;

#[cfg(not(feature = "gpu"))]
use crate::core::GaussianMap;
#[cfg(feature = "gpu")]
use crate::training::HostSplats;
use crate::TrainingError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Simple training checkpoint used by the trainer's incremental save.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Host-side splat artifact captured at the checkpoint boundary.
    #[cfg(feature = "gpu")]
    pub splats: HostSplats,
    /// Compatibility fallback for non-GPU builds.
    #[cfg(not(feature = "gpu"))]
    pub scene: GaussianMap,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyTrainingCheckpoint {
    iteration: usize,
    loss: f32,
    scene: crate::core::GaussianMap,
}

impl Default for TrainingCheckpoint {
    fn default() -> Self {
        Self {
            iteration: 0,
            loss: 0.0,
            #[cfg(feature = "gpu")]
            splats: HostSplats::default(),
            #[cfg(not(feature = "gpu"))]
            scene: GaussianMap::default(),
        }
    }
}

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
        match serde_json::from_slice(&bytes) {
            Ok(checkpoint) => Ok(checkpoint),
            Err(current_err) => {
                #[cfg(feature = "gpu")]
                {
                    if let Ok(legacy) = serde_json::from_slice::<LegacyTrainingCheckpoint>(&bytes) {
                        let splats = crate::training::HostSplats::from_legacy_gaussians_inferred(
                            legacy.scene.gaussians(),
                        )
                        .map_err(|err| {
                            TrainingError::TrainingFailed(format!(
                                "failed to convert legacy checkpoint {} into HostSplats: {err}",
                                path.display()
                            ))
                        })?;
                        return Ok(Self {
                            iteration: legacy.iteration,
                            loss: legacy.loss,
                            splats,
                        });
                    }
                }

                Err(TrainingError::TrainingFailed(format!(
                    "failed to deserialize checkpoint {}: {current_err}",
                    path.display()
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrainingCheckpoint;
    use tempfile::tempdir;

    #[cfg(feature = "gpu")]
    #[test]
    fn checkpoint_round_trips_host_splats() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("checkpoint.json");
        let splats = crate::training::HostSplats::from_scene_gaussians(
            &[crate::Gaussian::new(
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0, 0.0],
                0.5,
                [0.2, 0.3, 0.4],
            )],
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

    #[cfg(feature = "gpu")]
    #[test]
    fn checkpoint_load_accepts_legacy_scene_payloads() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("legacy-checkpoint.json");
        let legacy = super::LegacyTrainingCheckpoint {
            iteration: 3,
            loss: 0.75,
            scene: crate::core::GaussianMap::from_gaussians(vec![
                crate::Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]),
            ]),
        };

        std::fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();
        let loaded = TrainingCheckpoint::load(&path).unwrap();

        assert_eq!(loaded.iteration, 3);
        assert!((loaded.loss - 0.75).abs() < 1e-6);
        assert_eq!(loaded.splats.len(), 1);
    }
}
