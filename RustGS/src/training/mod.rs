//! Training module for 3D Gaussian Splatting.
//!
//! This module provides various training strategies:
//! - `trainer` - Basic CPU trainer with finite-difference gradients
//! - `complete_trainer` - Full trainer with LR scheduler and SSIM loss
//! - `gpu_trainer` - GPU-optimized trainer with minimal CPU-GPU transfer
//! - `autodiff` - True autodiff using Candle's Var and backward()
//! - `autodiff_trainer` - Autodiff trainer using candle-core 0.9.x API
//! - `training_pipeline` - Training utilities, densify/prune, loss functions

pub mod training_pipeline;

#[cfg(feature = "gpu")]
pub mod trainer;

#[cfg(feature = "gpu")]
mod data_loading;

#[cfg(feature = "gpu")]
pub mod complete_trainer;

#[cfg(feature = "gpu")]
pub mod gpu_trainer;

#[cfg(feature = "gpu")]
pub mod autodiff;

#[cfg(feature = "gpu")]
pub mod autodiff_trainer;

// Re-export common types at module level
pub use training_pipeline::{
    compute_psnr, compute_ssim_loss, compute_training_loss, default_camera_intrinsics,
    densify_gaussians, prune_gaussians, reset_opacity, SceneIoError, SceneMetadata,
    TrainableGaussian, TrainingConfig as PipelineConfig, TrainingState,
};

#[cfg(feature = "gpu")]
pub use trainer::{TrainConfig, TrainState, Trainer};

#[cfg(feature = "gpu")]
pub use complete_trainer::{
    adam_update, CompleteTrainer, LrScheduler, TrainerAdamState,
    TrainingResult as DetailedTrainingResult,
};

#[cfg(feature = "gpu")]
pub use gpu_trainer::{
    GpuAdamState, GpuGaussianBuffer, GpuTrainer, GpuTrainerBuilder, GpuTrainerConfig, SyncData,
};

#[cfg(feature = "gpu")]
pub use autodiff::{TrueAutodiffTrainer, VarCamera, VarGaussian, VarOutput, VarRenderer};

#[cfg(feature = "gpu")]
pub use autodiff_trainer::{
    AutodiffTrainer, DiffGaussian, DiffLoss, DiffRenderCamera, DiffRendered, DiffSplat,
};

use crate::TrainingError;
use crate::{GaussianMap, TrainingDataset};

#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use data_loading::{load_training_data, map_from_trainable, trainable_from_map};

/// Legacy training configuration (kept for API compatibility).
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training iterations
    pub iterations: usize,
    /// Learning rate for positions
    pub lr_position: f32,
    /// Learning rate for scales
    pub lr_scale: f32,
    /// Learning rate for rotations
    pub lr_rotation: f32,
    /// Learning rate for opacities
    pub lr_opacity: f32,
    /// Learning rate for colors
    pub lr_color: f32,
    /// Densification interval
    pub densify_interval: usize,
    /// Pruning threshold
    pub prune_threshold: f32,
    /// Maximum number of Gaussians created during initialization
    pub max_initial_gaussians: usize,
    /// Sampling step for frame-to-Gaussian initialization (0 = auto)
    pub sampling_step: usize,
    /// Minimum valid depth in meters
    pub min_depth: f32,
    /// Maximum valid depth in meters
    pub max_depth: f32,
    /// Generate synthetic depth from image luminance when depth is unavailable
    pub use_synthetic_depth: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: 30000,
            lr_position: 0.00016,
            lr_scale: 0.005,
            lr_rotation: 0.001,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            densify_interval: 100,
            prune_threshold: 0.005,
            max_initial_gaussians: 100_000,
            sampling_step: 0,
            min_depth: 0.01,
            max_depth: 10.0,
            use_synthetic_depth: true,
        }
    }
}

/// Training result.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss
    pub final_loss: f32,
    /// Number of Gaussians
    pub num_gaussians: usize,
    /// Training time in seconds
    pub training_time: f64,
}

/// Train a 3DGS scene from a dataset.
///
/// This is a convenience function that uses CompleteTrainer internally.
#[cfg(feature = "gpu")]
pub fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let start = Instant::now();
    let device = crate::preferred_device();
    let loaded = load_training_data(dataset, config, &device)?;
    let mut gaussians = trainable_from_map(&loaded.initial_map, &device)?;

    if gaussians.len() == 0 {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }

    let mut trainer = CompleteTrainer::with_device(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        config.lr_position,
        config.lr_scale,
        config.lr_rotation,
        config.lr_opacity,
        config.lr_color,
        device.clone(),
    );

    let color_refs: Vec<&[f32]> = loaded.colors.iter().map(|color| color.as_slice()).collect();
    let depth_refs: Vec<&[f32]> = loaded.depths.iter().map(|depth| depth.as_slice()).collect();
    let result = trainer.train(
        &mut gaussians,
        &loaded.cameras,
        &color_refs,
        &depth_refs,
        config.iterations,
    )?;

    let trained_map = map_from_trainable(&gaussians)?;
    log::info!(
        "RustGS training complete in {:.2}s | frames={} | initial_gaussians={} | final_gaussians={} | final_loss={:.6}",
        start.elapsed().as_secs_f64(),
        dataset.poses.len(),
        loaded.initial_map.len(),
        trained_map.len(),
        result.final_loss,
    );

    Ok(trained_map)
}
