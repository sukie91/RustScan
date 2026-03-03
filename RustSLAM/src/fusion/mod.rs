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
//! - gpu_trainer.rs: GPU-accelerated trainer with minimal CPU-GPU transfer
//! - slam_integrator.rs: Sparse-Dense SLAM integration
//! - training_pipeline.rs: Complete training pipeline
//! - tracker.rs: Gaussian-based tracking
//! - mapper.rs: Incremental Gaussian mapping
//! - trainer.rs: Basic training pipeline
//! - complete_trainer.rs: Complete trainer
//! - tsdf_volume.rs: TSDF volume for mesh extraction
//! - marching_cubes.rs: Marching cubes algorithm
//! - mesh_extractor.rs: High-level mesh extraction API

pub mod gaussian;
pub mod scene_io;
pub mod renderer;
pub mod analytical_backward;
pub mod tiled_renderer;
pub mod slam_integrator;
pub mod training_pipeline;
pub mod tracker;
pub mod mapper;
pub mod tsdf_volume;
pub mod marching_cubes;
pub mod mesh_extractor;
pub mod mesh_io;
pub mod mesh_metadata;

#[cfg(feature = "gpu")]
pub mod gaussian_init;
#[cfg(feature = "gpu")]
pub mod training_checkpoint;
#[cfg(feature = "gpu")]
pub mod diff_renderer;
#[cfg(feature = "gpu")]
pub mod diff_splat;
#[cfg(feature = "gpu")]
pub mod autodiff;
#[cfg(feature = "gpu")]
pub mod complete_trainer;
#[cfg(feature = "gpu")]
pub mod trainer;
#[cfg(feature = "gpu")]
pub mod gpu_trainer;
#[cfg(feature = "gpu")]
pub mod autodiff_trainer;

pub use gaussian::{Gaussian3D, GaussianMap, GaussianCamera, GaussianState};
pub use scene_io::{SceneMetadata, SceneIoError, save_scene_ply, load_scene_ply};
pub use renderer::{GaussianRenderer, RenderOutput};
pub use analytical_backward::{
    GaussianRenderRecord, ForwardIntermediate, AnalyticalGradients,
    backward as analytical_backward_pass,
};
pub use tiled_renderer::{Gaussian, ProjectedGaussian, TiledRenderer, RenderBuffer, densify, prune};
pub use slam_integrator::{SparseDenseSlam, SlamIntegrator, SlamConfig, KeyFrame, SlamOutput};
pub use training_pipeline::{
    TrainingConfig, TrainableGaussian, TrainingState,
    densify_gaussians, prune_gaussians, reset_opacity,
    get_learning_rate, compute_ssim_loss, compute_training_loss,
};
pub use tracker::{GaussianTracker, TrackingResult};
pub use mapper::{GaussianMapper, MapperConfig, MapperUpdateResult};
pub use tsdf_volume::{TsdfVolume, TsdfConfig, Voxel};
pub use marching_cubes::{Mesh, MeshVertex, MeshTriangle, MarchingCubes};
pub use mesh_extractor::{MeshExtractor, MeshExtractionConfig};
pub use mesh_io::{MeshIoError, save_mesh_obj, save_mesh_ply, export_mesh};
pub use mesh_metadata::{
    MeshMetadata,
    MeshMetadataError,
    MeshTimings,
    BoundingBox,
    TsdfMetadata,
    save_mesh_metadata,
    export_mesh_metadata,
};

#[cfg(feature = "gpu")]
pub use gaussian_init::{
    GaussianInitConfig,
    initialize_gaussians_from_map,
    initialize_trainable_gaussians_from_map,
};
#[cfg(feature = "gpu")]
pub use training_checkpoint::{
    TrainingCheckpoint,
    TrainingCheckpointConfig,
    TrainingCheckpointManager,
    TrainingCheckpointError,
    checkpoint_path as training_checkpoint_path,
    load_latest_checkpoint as load_latest_training_checkpoint,
};
#[cfg(feature = "gpu")]
pub use diff_renderer::{DiffGaussianRenderer, GaussianTensors, CameraTensors, RenderLoss};
#[cfg(feature = "gpu")]
pub use diff_splat::{TrainableGaussians, DiffSplatRenderer, DiffCamera, DiffRenderOutput, DiffLoss};
#[cfg(feature = "gpu")]
pub use autodiff::{VarGaussian, VarCamera, VarRenderer, VarOutput, TrueAutodiffTrainer};
#[cfg(feature = "gpu")]
pub use complete_trainer::{CompleteTrainer, LrScheduler, TrainingResult as CompleteTrainingResult};
#[cfg(feature = "gpu")]
pub use trainer::{Trainer, TrainConfig, TrainState};
#[cfg(feature = "gpu")]
pub use gpu_trainer::{GpuTrainer, GpuTrainerConfig, GpuGaussianBuffer, SyncData, GpuAdamState, GpuTrainerBuilder};
