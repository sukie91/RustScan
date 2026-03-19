//! Training checkpoint utilities for 3DGS optimization.
//!
//! Provides full checkpoint save/load with Adam optimizer state,
//! compatible with the CompleteTrainer.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrainingCheckpointError {
    #[error("failed to create checkpoint directory {path}: {source}")]
    CreateDir { path: PathBuf, source: std::io::Error },
    #[error("failed to write checkpoint {path}: {source}")]
    Write { path: PathBuf, source: std::io::Error },
    #[error("failed to read checkpoint {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("failed to serialize checkpoint {path}: {source}")]
    Serialize { path: PathBuf, source: serde_json::Error },
    #[error("failed to deserialize checkpoint {path}: {source}")]
    Deserialize { path: PathBuf, source: serde_json::Error },
    #[error("invalid checkpoint: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct TrainingCheckpointConfig {
    pub dir: PathBuf,
    pub interval: usize,
}

pub struct TrainingCheckpointManager {
    config: TrainingCheckpointConfig,
    last_saved_iter: usize,
}

impl TrainingCheckpointManager {
    pub fn with_output_dir(
        output_dir: &Path,
        interval: usize,
    ) -> Result<Self, TrainingCheckpointError> {
        let config = TrainingCheckpointConfig {
            dir: checkpoint_dir(output_dir),
            interval,
        };
        Self::new(config)
    }

    pub fn new(config: TrainingCheckpointConfig) -> Result<Self, TrainingCheckpointError> {
        if !config.dir.exists() {
            fs::create_dir_all(&config.dir).map_err(|source| TrainingCheckpointError::CreateDir {
                path: config.dir.clone(),
                source,
            })?;
        }

        Ok(Self {
            config,
            last_saved_iter: 0,
        })
    }

    #[cfg(feature = "gpu")]
    pub fn maybe_save(
        &mut self,
        iteration: usize,
        gaussians: &crate::diff::diff_splat::TrainableGaussians,
        trainer: &crate::training::complete_trainer::CompleteTrainer,
    ) -> Result<PathBuf, TrainingCheckpointError> {
        if self.config.interval == 0 {
            return Ok(PathBuf::new());
        }
        if iteration < self.last_saved_iter + self.config.interval {
            return Ok(PathBuf::new());
        }

        let checkpoint = FullTrainingCheckpoint::from_trainer(trainer, gaussians)?;
        let path = checkpoint_path(&self.config.dir, iteration);
        save_checkpoint(&checkpoint, &path)?;
        self.last_saved_iter = iteration;
        Ok(path)
    }

    pub fn set_last_saved_iter(&mut self, iteration: usize) {
        self.last_saved_iter = iteration;
    }
}

pub fn default_checkpoint_interval() -> usize {
    500
}

pub fn checkpoint_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("checkpoints")
}

pub fn checkpoint_path(dir: &Path, iteration: usize) -> PathBuf {
    dir.join(format!("3dgs_{iteration}.ckpt"))
}

fn parse_checkpoint_name(path: &Path) -> Option<usize> {
    let file_name = path.file_name()?.to_string_lossy();
    if !file_name.starts_with("3dgs_") || !file_name.ends_with(".ckpt") {
        return None;
    }
    let number = &file_name[5..file_name.len() - 5];
    number.parse::<usize>().ok()
}

// ─── GPU-gated types (depend on TrainerAdamState / TrainableGaussians) ───

#[cfg(feature = "gpu")]
mod gpu_checkpoint {
    use std::fs::{self, File};
    use std::io::{BufReader, BufWriter};
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};

    use candle_core::Device;
    use crate::training::complete_trainer::{CompleteTrainer, TrainerAdamState};
    use crate::diff::diff_splat::TrainableGaussians;

    use super::{TrainingCheckpointError, checkpoint_path, parse_checkpoint_name};

    const TRAINING_CHECKPOINT_VERSION: u32 = 1;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FullTrainingCheckpoint {
        pub version: u32,
        pub iteration: usize,
        pub gaussians: Vec<CheckpointGaussian>,
        pub optimizer: TrainerAdamState,
        pub loss_history: Vec<f32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CheckpointGaussian {
        pub position: [f32; 3],
        pub scale: [f32; 3],
        pub rotation: [f32; 4],
        pub opacity: f32,
        pub color: [f32; 3],
    }

    impl FullTrainingCheckpoint {
        pub fn from_trainer(
            trainer: &CompleteTrainer,
            gaussians: &TrainableGaussians,
        ) -> Result<Self, TrainingCheckpointError> {
            let gaussians = export_gaussians(gaussians)?;
            Ok(Self {
                version: TRAINING_CHECKPOINT_VERSION,
                iteration: trainer.iteration(),
                gaussians,
                optimizer: trainer.export_adam_state(),
                loss_history: trainer.loss_history().to_vec(),
            })
        }

        pub fn to_trainable_gaussians(&self, device: &Device) -> candle_core::Result<TrainableGaussians> {
            let mut positions = Vec::with_capacity(self.gaussians.len() * 3);
            let mut scales = Vec::with_capacity(self.gaussians.len() * 3);
            let mut rotations = Vec::with_capacity(self.gaussians.len() * 4);
            let mut opacities = Vec::with_capacity(self.gaussians.len());
            let mut colors = Vec::with_capacity(self.gaussians.len() * 3);

            for g in &self.gaussians {
                positions.extend_from_slice(&g.position);
                scales.extend_from_slice(&[
                    g.scale[0].ln(),
                    g.scale[1].ln(),
                    g.scale[2].ln(),
                ]);
                rotations.extend_from_slice(&g.rotation);
                opacities.push(opacity_to_logit(g.opacity));
                colors.extend_from_slice(&g.color);
            }

            TrainableGaussians::new(&positions, &scales, &rotations, &opacities, &colors, device)
        }

        pub fn apply_to_trainer(&self, trainer: &mut CompleteTrainer) {
            trainer.set_iteration(self.iteration);
            trainer.set_loss_history(self.loss_history.clone());
            trainer.import_adam_state(self.optimizer.clone());
        }
    }

    pub fn save_checkpoint(checkpoint: &FullTrainingCheckpoint, path: &Path) -> Result<(), TrainingCheckpointError> {
        let file = File::create(path).map_err(|source| TrainingCheckpointError::Write {
            path: path.to_path_buf(),
            source,
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, checkpoint).map_err(|source| TrainingCheckpointError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    pub fn load_checkpoint(path: &Path) -> Result<FullTrainingCheckpoint, TrainingCheckpointError> {
        let file = File::open(path).map_err(|source| TrainingCheckpointError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let reader = BufReader::new(file);
        let checkpoint: FullTrainingCheckpoint = serde_json::from_reader(reader).map_err(|source| TrainingCheckpointError::Deserialize {
            path: path.to_path_buf(),
            source,
        })?;
        if checkpoint.version != TRAINING_CHECKPOINT_VERSION {
            return Err(TrainingCheckpointError::Invalid(format!(
                "unsupported checkpoint version {}",
                checkpoint.version
            )));
        }
        Ok(checkpoint)
    }

    pub fn load_latest_checkpoint(dir: &Path) -> Result<Option<FullTrainingCheckpoint>, TrainingCheckpointError> {
        if !dir.exists() {
            return Ok(None);
        }

        let mut best: Option<(usize, PathBuf)> = None;
        for entry in fs::read_dir(dir).map_err(|source| TrainingCheckpointError::Read {
            path: dir.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| TrainingCheckpointError::Read {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if let Some(iteration) = parse_checkpoint_name(&path) {
                if best.as_ref().map(|(idx, _)| iteration > *idx).unwrap_or(true) {
                    best = Some((iteration, path));
                }
            }
        }

        let Some((_idx, path)) = best else { return Ok(None); };
        load_checkpoint(&path).map(Some)
    }

    pub fn resume_latest_checkpoint(
        dir: &Path,
        trainer: &mut CompleteTrainer,
        device: &Device,
    ) -> Result<Option<TrainableGaussians>, TrainingCheckpointError> {
        let Some(checkpoint) = load_latest_checkpoint(dir)? else {
            log::info!("No 3DGS checkpoint found in {}", dir.display());
            return Ok(None);
        };

        let gaussians = checkpoint
            .to_trainable_gaussians(device)
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;
        checkpoint.apply_to_trainer(trainer);

        log::info!(
            "Loaded 3DGS checkpoint at iteration {} from {}",
            checkpoint.iteration,
            dir.display()
        );

        Ok(Some(gaussians))
    }

    fn export_gaussians(gaussians: &TrainableGaussians) -> Result<Vec<CheckpointGaussian>, TrainingCheckpointError> {
        let positions = gaussians.positions().to_vec2::<f32>()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;
        let scales = gaussians.scales()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?
            .to_vec2::<f32>()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;
        let rotations = gaussians.rotations()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?
            .to_vec2::<f32>()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;
        let opacities = gaussians.opacities()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?
            .to_vec1::<f32>()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;
        let colors = gaussians.colors().to_vec2::<f32>()
            .map_err(|err| TrainingCheckpointError::Invalid(err.to_string()))?;

        let positions: Vec<f32> = positions.into_iter().flatten().collect();
        let scales: Vec<f32> = scales.into_iter().flatten().collect();
        let rotations: Vec<f32> = rotations.into_iter().flatten().collect();
        let colors: Vec<f32> = colors.into_iter().flatten().collect();

        let n = gaussians.len();
        let mut output = Vec::with_capacity(n);

        for i in 0..n {
            let p = i * 3;
            let r = i * 4;
            let c = i * 3;
            output.push(CheckpointGaussian {
                position: [positions[p], positions[p + 1], positions[p + 2]],
                scale: [scales[p], scales[p + 1], scales[p + 2]],
                rotation: [rotations[r], rotations[r + 1], rotations[r + 2], rotations[r + 3]],
                opacity: opacities[i],
                color: [colors[c], colors[c + 1], colors[c + 2]],
            });
        }

        Ok(output)
    }

    fn opacity_to_logit(opacity: f32) -> f32 {
        let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
        (clamped / (1.0 - clamped)).ln()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn test_training_checkpoint_roundtrip() {
            let device = Device::Cpu;
            let positions = vec![0.0f32, 0.0, 1.0];
            let scales = vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()];
            let rotations = vec![1.0f32, 0.0, 0.0, 0.0];
            let opacities = vec![opacity_to_logit(0.5)];
            let colors = vec![0.2f32, 0.3, 0.4];

            let gaussians = TrainableGaussians::new(
                &positions,
                &scales,
                &rotations,
                &opacities,
                &colors,
                &device,
            ).unwrap();

            let trainer = CompleteTrainer::new(4, 4, 0.001, 0.001, 0.001, 0.001, 0.001);

            let checkpoint = FullTrainingCheckpoint::from_trainer(&trainer, &gaussians).unwrap();

            let dir = tempdir().unwrap();
            let path = checkpoint_path(dir.path(), checkpoint.iteration);
            save_checkpoint(&checkpoint, &path).unwrap();

            let loaded = load_checkpoint(&path).unwrap();
            assert_eq!(loaded.iteration, 0);
            assert_eq!(loaded.gaussians.len(), 1);

            let restored = loaded.to_trainable_gaussians(&device).unwrap();
            assert_eq!(restored.len(), 1);
        }

        #[test]
        fn test_resume_latest_checkpoint() {
            let device = Device::Cpu;
            let positions = vec![0.0f32, 0.0, 1.0];
            let scales = vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()];
            let rotations = vec![1.0f32, 0.0, 0.0, 0.0];
            let opacities = vec![opacity_to_logit(0.5)];
            let colors = vec![0.2f32, 0.3, 0.4];

            let gaussians = TrainableGaussians::new(
                &positions,
                &scales,
                &rotations,
                &opacities,
                &colors,
                &device,
            ).unwrap();

            let mut trainer = CompleteTrainer::new(4, 4, 0.001, 0.001, 0.001, 0.001, 0.001);
            trainer.set_iteration(500);
            trainer.set_loss_history(vec![1.0, 0.5]);

            let checkpoint = FullTrainingCheckpoint::from_trainer(&trainer, &gaussians).unwrap();
            let dir = tempdir().unwrap();
            let path = checkpoint_path(dir.path(), checkpoint.iteration);
            save_checkpoint(&checkpoint, &path).unwrap();

            let mut new_trainer = CompleteTrainer::new(4, 4, 0.001, 0.001, 0.001, 0.001, 0.001);
            let restored = resume_latest_checkpoint(dir.path(), &mut new_trainer, &device).unwrap();

            assert!(restored.is_some());
            assert_eq!(new_trainer.iteration(), 500);
            assert_eq!(new_trainer.loss_history(), &[1.0, 0.5]);
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu_checkpoint::*;
