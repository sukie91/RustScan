//! Dense Fusion Module
//!
//! This module provides dense reconstruction capabilities using 3D Gaussian Splatting.
//! Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al. SIGGRAPH 2023)
//! 
//! Architecture:
//! - gaussian.rs: Core 3D Gaussian data structures
//! - renderer.rs: Basic forward rendering
//! - diff_renderer.rs: Basic differentiable rendering (Candle)
//! - diff_splat.rs: Complete differentiable splatting (Candle + Metal)
//! - autodiff.rs: TRUE automatic differentiation with backward propagation
//! - tiled_renderer.rs: Complete tiled rasterization
//! - slam_integrator.rs: Sparse-Dense SLAM integration
//! - training_pipeline.rs: Complete training pipeline (NEW!)
//! - tracker.rs: Gaussian-based tracking
//! - mapper.rs: Incremental Gaussian mapping
//! - trainer.rs: Basic training pipeline
//! - complete_trainer.rs: Complete trainer

pub mod gaussian;
pub mod renderer;
pub mod diff_renderer;
pub mod diff_splat;
pub mod autodiff;
pub mod tiled_renderer;
pub mod slam_integrator;
pub mod training_pipeline;
pub mod tracker;
pub mod mapper;
pub mod trainer;
pub mod complete_trainer;

pub use gaussian::{Gaussian3D, GaussianMap, GaussianCamera, GaussianState};
pub use renderer::{GaussianRenderer, RenderOutput};
pub use diff_renderer::{DiffGaussianRenderer, GaussianTensors, CameraTensors, RenderLoss};
pub use diff_splat::{TrainableGaussians, DiffSplatRenderer, DiffCamera, DiffRenderOutput, DiffLoss};
pub use autodiff::{VarGaussian, VarCamera, VarRenderer, VarOutput, TrueAutodiffTrainer};
pub use tiled_renderer::{Gaussian, ProjectedGaussian, TiledRenderer, RenderBuffer, densify, prune};
pub use slam_integrator::{SparseDenseSlam, SlamIntegrator, SlamConfig, KeyFrame, SlamOutput};
pub use training_pipeline::{
    TrainingConfig, TrainableGaussian, TrainingState,
    densify_gaussians, prune_gaussians, reset_opacity,
    get_learning_rate, compute_ssim_loss, compute_training_loss,
};
pub use tracker::{GaussianTracker, TrackingResult};
pub use mapper::{GaussianMapper, MapperConfig, MapperUpdateResult};
pub use trainer::{Trainer, TrainConfig, TrainState};
pub use complete_trainer::{CompleteTrainer, LrScheduler, TrainingResult as CompleteTrainingResult};