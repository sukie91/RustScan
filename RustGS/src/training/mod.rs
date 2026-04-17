//! Training module for 3D Gaussian Splatting.
//!
//! Primary runtime path:
//! - `wgpu` - Burn + wgpu backend used by the top-level API.

mod config;
#[path = "topology/density_controller.rs"]
pub mod density_controller;
pub mod eval;
pub mod parity_harness;
mod pose_utils;

#[cfg(feature = "gpu")]
mod data;

#[cfg(feature = "gpu")]
mod pipeline;

#[cfg(feature = "gpu")]
#[path = "state/runtime_splats.rs"]
mod runtime_splats;

#[cfg(feature = "gpu")]
mod telemetry;

#[cfg(feature = "gpu")]
mod topology;

#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(feature = "gpu")]
pub use eval::SplatEvaluationRenderer;
pub use eval::MIN_RENDER_SCALE;
pub use eval::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, PsnrSummary, SplatEvaluationConfig, SplatEvaluationError,
    SplatEvaluationResult, SplatEvaluationSummary,
};
#[cfg(feature = "gpu")]
pub use eval::{evaluate_splats, evaluation_device, render_evaluation_frame, runtime_from_splats};
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
pub use pipeline::events::{
    TrainingControl, TrainingEvent, TrainingEventCadence, TrainingEventRoute,
    TrainingIterationProgress, TrainingPlanSelected, TrainingRun, TrainingRunCancelled,
    TrainingRunCompleted, TrainingRunReport, TrainingRunStarted, TrainingSnapshotReady,
};

pub use config::{
    LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize, TrainingBackend,
    TrainingConfig, TrainingResult,
};
#[cfg(feature = "gpu")]
pub use telemetry::{last_training_telemetry, LiteGsOptimizerLrs, LiteGsTrainingTelemetry};

#[cfg(feature = "gpu")]
pub use pipeline::orchestrator::{
    train_splats, train_splats_with_controlled_events, train_splats_with_events,
    train_splats_with_report,
};
