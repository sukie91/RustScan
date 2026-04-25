//! Training module for 3D Gaussian Splatting.

mod config;
#[cfg(feature = "gpu")]
mod engine;
mod evaluation;
mod metrics;

#[cfg(test)]
mod tests;

#[cfg(feature = "gpu")]
mod backward;

#[cfg(feature = "gpu")]
mod data;

#[cfg(feature = "gpu")]
mod events;

#[cfg(feature = "gpu")]
mod forward;

#[cfg(feature = "gpu")]
mod gpu_primitives;

#[cfg(feature = "gpu")]
mod telemetry;

#[cfg(feature = "gpu")]
mod topology;

#[cfg(feature = "gpu")]
use crate::core::HostSplats;
#[cfg(feature = "gpu")]
use crate::{TrainingDataset, TrainingError};

#[cfg(feature = "gpu")]
pub use evaluation::SplatEvaluationRenderer;
pub use evaluation::MIN_RENDER_SCALE;
pub use evaluation::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, ParityCheckOutcome, ParityCheckStatus,
    ParityFixtureKind, ParityFixtureSpec, ParityGateEvaluation, ParityGateStatus,
    ParityHarnessReport, ParityMetricSnapshot, ParityReferenceComparison, ParityThresholds,
    ParityTimingMetrics, DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};
pub use evaluation::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, PsnrSummary, SplatEvaluationConfig, SplatEvaluationError,
    SplatEvaluationResult, SplatEvaluationSummary,
};
#[cfg(feature = "gpu")]
pub use evaluation::{
    evaluate_splats, evaluation_device, render_evaluation_frame, runtime_from_splats,
};
#[cfg(feature = "gpu")]
pub use events::{
    TrainingControl, TrainingEvent, TrainingEventCadence, TrainingEventRoute,
    TrainingIterationProgress, TrainingPlanSelected, TrainingRun, TrainingRunCancelled,
    TrainingRunCompleted, TrainingRunReport, TrainingRunStarted, TrainingSnapshotReady,
};
pub use metrics::{ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics};

pub use config::{
    LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize, TrainingBackend,
    TrainingConfig, TrainingResult,
};
#[cfg(feature = "gpu")]
pub use telemetry::{last_training_telemetry, LiteGsOptimizerLrs, LiteGsTrainingTelemetry};

#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    train_splats_with_report(dataset, config).map(TrainingRun::into_splats)
}

#[cfg(feature = "gpu")]
pub fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let mut sink = |_event| {};
    train_splats_with_events(dataset, config, &mut sink)
}

#[cfg(feature = "gpu")]
pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    train_splats_with_controlled_events(dataset, config, TrainingControl::default(), on_event)
}

#[cfg(feature = "gpu")]
pub fn train_splats_with_controlled_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    control: TrainingControl,
    on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    telemetry::store_last_training_telemetry(None);
    config.validate()?;
    validate_litegs_mac_v1_config(config)?;
    engine::train_splats_with_controlled_events(dataset, config, control, on_event)
}

#[cfg(feature = "gpu")]
fn validate_litegs_mac_v1_config(config: &TrainingConfig) -> Result<(), TrainingError> {
    let defaults = LiteGsConfig::default();
    let mut unsupported = Vec::new();

    if config.litegs.tile_size != defaults.tile_size {
        unsupported.push(format!(
            "tile_size={} overrides are reserved for later LiteGS parity work; bootstrap profile currently expects {}",
            config.litegs.tile_size, defaults.tile_size
        ));
    }
    if config.litegs.sh_degree == 0 {
        unsupported
            .push("sh_degree=0 is not supported for LiteGsMacV1; use degree >= 1".to_string());
    }
    if config.litegs.densification_interval == 0 {
        unsupported.push("densification_interval must be >= 1".to_string());
    }
    if config.litegs.refine_every == 0 {
        unsupported.push("refine_every must be >= 1".to_string());
    }
    if !config.litegs.growth_grad_threshold.is_finite() || config.litegs.growth_grad_threshold < 0.0
    {
        unsupported.push("growth_grad_threshold must be finite and >= 0".to_string());
    }
    if !config.litegs.growth_select_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.litegs.growth_select_fraction)
    {
        unsupported.push("growth_select_fraction must be in [0, 1]".to_string());
    }
    if config.litegs.growth_stop_iter == 0 {
        unsupported.push("growth_stop_iter must be >= 1".to_string());
    }
    if config.litegs.opacity_reset_interval == 0 {
        unsupported.push("opacity_reset_interval must be >= 1".to_string());
    }
    if config.litegs.target_primitives == 0 {
        unsupported.push("target_primitives must be >= 1".to_string());
    }
    if config.litegs.prune_min_age == 0 {
        unsupported.push("prune_min_age must be >= 1 to protect newly-added Gaussians".to_string());
    }
    if config.litegs.prune_invisible_epochs == 0 {
        unsupported.push("prune_invisible_epochs must be >= 1".to_string());
    }

    if unsupported.is_empty() {
        return Ok(());
    }

    Err(TrainingError::TrainingFailed(format!(
        "LiteGsMacV1 bootstrap profile rejected unsupported overrides: {}",
        unsupported.join("; ")
    )))
}
