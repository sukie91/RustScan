//! Training module for 3D Gaussian Splatting.
//!
//! Primary runtime path:
//! - `metal_trainer` - Metal-native training loop used by the top-level API.

#[cfg(feature = "gpu")]
mod benchmark;

pub mod chunk_planner;
pub mod clustering;
mod config;
pub mod density_controller;
pub mod eval;
pub mod morton;
mod orchestrator;
pub mod parity_harness;
pub mod pose_embedding;

#[cfg(feature = "gpu")]
mod data_loading;

#[cfg(feature = "gpu")]
mod chunk_training;

#[cfg(feature = "gpu")]
mod execution_plan;

#[cfg(feature = "gpu")]
mod export;

#[cfg(feature = "gpu")]
mod frame_loader;

#[cfg(feature = "gpu")]
mod frame_targets;

#[cfg(feature = "gpu")]
mod init_map;

#[cfg(feature = "gpu")]
pub mod metal_trainer;

#[cfg(feature = "gpu")]
mod metal_runtime;

#[cfg(feature = "gpu")]
mod metal_loss;

#[cfg(feature = "gpu")]
mod metal_backward;

#[cfg(feature = "gpu")]
mod metal_dispatch;

#[cfg(feature = "gpu")]
mod metal_forward;

#[cfg(feature = "gpu")]
mod metal_kernels;

#[cfg(feature = "gpu")]
mod metal_optimizer;

#[cfg(feature = "gpu")]
mod metal_pipelines;

#[cfg(feature = "gpu")]
mod metal_projection;

#[cfg(feature = "gpu")]
mod metal_raster;

#[cfg(feature = "gpu")]
mod metal_resources;

#[cfg(feature = "gpu")]
mod runtime_splats;

#[cfg(feature = "gpu")]
mod splats;

#[cfg(feature = "gpu")]
mod splat_interop;

#[cfg(feature = "gpu")]
mod topology;

#[cfg(feature = "gpu")]
pub use benchmark::{
    run_metal_training_benchmark, MetalTrainingBenchmarkReport, MetalTrainingBenchmarkSpec,
};
pub use chunk_planner::{
    materialize_chunk_dataset, plan_spatial_chunks, ChunkBounds, ChunkBoundsSource,
    ChunkDisposition, ChunkPlan, MaterializedChunkDataset, PlannedChunk,
};
#[cfg(feature = "gpu")]
pub use eval::SplatEvaluationRenderer;
pub use eval::MIN_RENDER_SCALE;
pub use eval::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, PsnrSummary, SceneEvaluationConfig, SceneEvaluationError,
    SplatEvaluationResult, SplatEvaluationSummary,
};
#[cfg(feature = "gpu")]
pub use eval::{
    evaluate_gaussians, evaluate_splats, evaluation_device, render_evaluation_frame,
    runtime_from_gaussians, runtime_from_splats,
};
pub use parity_harness::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, ParityCheckOutcome, ParityCheckStatus,
    ParityFixtureKind, ParityFixtureSpec, ParityGateEvaluation, ParityGateStatus,
    ParityHarnessReport, ParityLossCurveSample, ParityLossTerms, ParityMetricSnapshot,
    ParityReferenceComparison, ParityThresholds, ParityTimingMetrics, ParityTopologyMetrics,
    DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};
#[cfg(feature = "gpu")]
pub use splats::{HostSplats, SplatView};

pub use config::{
    LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize, TrainingBackend,
    TrainingConfig, TrainingProfile, TrainingResult,
};
#[cfg(feature = "gpu")]
pub use metal_trainer::{
    estimate_chunk_capacity, last_metal_training_telemetry, ChunkCapacityDisposition,
    ChunkCapacityEstimate, LiteGsOptimizerLrs, LiteGsTrainingTelemetry, MetalTrainer,
};

#[cfg(feature = "gpu")]
pub use orchestrator::train_splats;
