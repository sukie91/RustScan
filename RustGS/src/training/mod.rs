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
pub mod complete_trainer;

#[cfg(feature = "gpu")]
pub mod gpu_trainer;

#[cfg(feature = "gpu")]
pub mod autodiff;

#[cfg(feature = "gpu")]
pub mod autodiff_trainer;

// Re-export common types at module level
pub use training_pipeline::{
    TrainingConfig as PipelineConfig,
    TrainableGaussian,
    TrainingState,
    SceneMetadata,
    SceneIoError,
    densify_gaussians,
    prune_gaussians,
    reset_opacity,
    compute_ssim_loss,
    compute_training_loss,
    compute_psnr,
    default_camera_intrinsics,
};

#[cfg(feature = "gpu")]
pub use trainer::{
    TrainConfig,
    TrainState,
    Trainer,
};

#[cfg(feature = "gpu")]
pub use complete_trainer::{
    CompleteTrainer,
    LrScheduler,
    TrainingResult as DetailedTrainingResult,
    TrainerAdamState,
    adam_update,
};

#[cfg(feature = "gpu")]
pub use gpu_trainer::{
    GpuTrainerConfig,
    SyncData,
    GpuGaussianBuffer,
    GpuAdamState,
    GpuTrainer,
    GpuTrainerBuilder,
};

#[cfg(feature = "gpu")]
pub use autodiff::{
    VarGaussian,
    VarCamera,
    VarRenderer,
    VarOutput,
    TrueAutodiffTrainer,
};

#[cfg(feature = "gpu")]
pub use autodiff_trainer::{
    DiffGaussian,
    DiffRenderCamera,
    DiffSplat,
    DiffRendered,
    DiffLoss,
    AutodiffTrainer,
};

use crate::{TrainingDataset, GaussianMap};
use crate::TrainingError;

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
pub fn train(_dataset: &TrainingDataset, _config: &TrainingConfig) -> Result<GaussianMap, TrainingError> {
    // TODO: Implement using CompleteTrainer once TrainingDataset integration is complete
    Ok(GaussianMap::default())
}