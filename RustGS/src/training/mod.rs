//! Training module for 3D Gaussian Splatting.
//!
//! This module provides various training strategies:
//! - `trainer` - Basic CPU trainer with finite-difference gradients
//! - `complete_trainer` - Full trainer with LR scheduler and SSIM loss
//! - `gpu_trainer` - GPU-optimized trainer with minimal CPU-GPU transfer
//! - `metal_trainer` - Primary Metal-native training loop that keeps render/loss/backward on GPU
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
pub mod metal_trainer;

#[cfg(feature = "gpu")]
mod metal_runtime;

#[cfg(feature = "gpu")]
mod metal_loss;

#[cfg(feature = "gpu")]
mod metal_backward;

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
pub use metal_trainer::MetalTrainer;

#[cfg(feature = "gpu")]
pub use autodiff::{TrueAutodiffTrainer, VarCamera, VarGaussian, VarOutput, VarRenderer};

#[cfg(feature = "gpu")]
pub use autodiff_trainer::{
    AutodiffTrainer, DiffGaussian, DiffLoss, DiffRenderCamera, DiffRendered, DiffSplat,
};

use crate::TrainingError;
use crate::{GaussianMap, TrainingDataset};

/// Training backend selection.
///
/// RustGS training now standardizes on the Metal backend. The enum is kept so
/// existing config construction code does not break abruptly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingBackend {
    /// Metal-native training path that keeps render/loss/backward/optimizer on GPU.
    Metal,
}

impl Default for TrainingBackend {
    fn default() -> Self {
        Self::Metal
    }
}

impl std::fmt::Display for TrainingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "metal")
    }
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Training backend implementation to use.
    ///
    /// Only `Metal` is supported for the top-level training flow.
    pub backend: TrainingBackend,
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
    /// Pruning interval for Metal topology updates.
    pub prune_interval: usize,
    /// Delay topology updates until after this many training iterations.
    pub topology_warmup: usize,
    /// Emit topology scheduling/throughput logs every N scheduled checks.
    pub topology_log_interval: usize,
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
    /// Render scale used by the Metal backend (relative to input resolution).
    pub metal_render_scale: f32,
    /// Number of Gaussians processed per GPU chunk in the Metal backend.
    pub metal_gaussian_chunk_size: usize,
    /// Emit per-step timing breakdowns for the Metal backend.
    pub metal_profile_steps: bool,
    /// Log the Metal timing breakdown every N steps when profiling is enabled.
    pub metal_profile_interval: usize,
    /// Use the native Metal forward rasterizer during normal training.
    pub metal_use_native_forward: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::default(),
            iterations: 30000,
            lr_position: 0.00016,
            lr_scale: 0.005,
            lr_rotation: 0.001,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            densify_interval: 100,
            prune_interval: 100,
            topology_warmup: 100,
            topology_log_interval: 500,
            prune_threshold: 0.005,
            max_initial_gaussians: 100_000,
            sampling_step: 0,
            min_depth: 0.01,
            max_depth: 10.0,
            use_synthetic_depth: true,
            metal_render_scale: 0.25,
            metal_gaussian_chunk_size: 32,
            metal_profile_steps: false,
            metal_profile_interval: 25,
            metal_use_native_forward: true,
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
/// This is a convenience function that uses the Metal-native trainer.
#[cfg(feature = "gpu")]
pub fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    metal_trainer::train(dataset, config)
}

#[cfg(test)]
mod tests {
    use super::{TrainingBackend, TrainingConfig};

    #[test]
    fn default_training_backend_is_metal() {
        assert_eq!(TrainingBackend::default(), TrainingBackend::Metal);
        assert_eq!(TrainingConfig::default().backend, TrainingBackend::Metal);
        assert!(TrainingConfig::default().metal_use_native_forward);
        assert_eq!(TrainingConfig::default().prune_interval, 100);
    }
}
