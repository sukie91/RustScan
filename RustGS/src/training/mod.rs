//! Training module for 3D Gaussian Splatting.
//!
//! Primary runtime path:
//! - `metal_trainer` - Metal-native training loop used by the top-level API.
//!
//! - `training_pipeline` - Shared densify/prune heuristics and utility losses.

pub mod training_pipeline;

pub mod chunk_planner;
pub mod parity_harness;

#[cfg(feature = "gpu")]
mod data_loading;

#[cfg(feature = "gpu")]
pub mod metal_trainer;

#[cfg(feature = "gpu")]
mod metal_runtime;

#[cfg(feature = "gpu")]
mod metal_loss;

#[cfg(feature = "gpu")]
mod metal_backward;

// Re-export common types at module level
pub use training_pipeline::{
    compute_psnr, compute_ssim_loss, compute_training_loss, default_camera_intrinsics,
    densify_gaussians, prune_gaussians, reset_opacity, SceneIoError, SceneMetadata,
    TrainableGaussian, TrainingConfig as PipelineConfig, TrainingState,
};

pub use chunk_planner::{
    materialize_chunk_dataset, plan_spatial_chunks, ChunkBounds, ChunkBoundsSource,
    ChunkDisposition, ChunkPlan, MaterializedChunkDataset, PlannedChunk,
};
pub use parity_harness::{
    default_litegs_parity_fixtures, default_parity_report_path, parity_fixture_id_for_input_path,
    ParityFixtureKind, ParityFixtureSpec, ParityHarnessReport, ParityLossTerms,
    ParityMetricSnapshot, ParityThresholds, ParityTimingMetrics, ParityTopologyMetrics,
    DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};

#[cfg(feature = "gpu")]
pub use metal_trainer::{
    estimate_chunk_capacity, ChunkCapacityDisposition, ChunkCapacityEstimate, MetalTrainer,
};

use crate::TrainingError;
use crate::{GaussianMap, TrainingDataset};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingExecutionRoute {
    Standard,
    ChunkedSingleChunk,
    ChunkedSequential,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct TrainingExecutionPlan {
    route: TrainingExecutionRoute,
    chunk_estimate: Option<ChunkCapacityEstimate>,
    chunk_plan: Option<ChunkPlan>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct ChunkTrainingOverridePlan {
    effective_config: TrainingConfig,
    estimate: ChunkCapacityEstimate,
    lowered_gaussian_cap: bool,
    lowered_render_scale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChunkReportStatus {
    Success,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkReportEntry {
    chunk_id: usize,
    status: ChunkReportStatus,
    message: Option<String>,
    scene_path: Option<PathBuf>,
    pose_count: usize,
    local_points: usize,
    used_frame_based_initialization: bool,
    effective_max_initial_gaussians: Option<usize>,
    effective_render_scale: Option<f32>,
    estimated_peak_gib: Option<f64>,
    output_gaussians: Option<usize>,
    core_filter_removed: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkTrainingReport {
    merge_core_only: bool,
    requested_chunks: usize,
    trainable_chunks: usize,
    entries: Vec<ChunkReportEntry>,
}

struct ChunkPersistenceContext {
    output_dir: Option<PathBuf>,
    report: ChunkTrainingReport,
}

impl ChunkPersistenceContext {
    fn new(
        output_dir: Option<PathBuf>,
        config: &TrainingConfig,
        chunk_plan: &ChunkPlan,
    ) -> Result<Self, TrainingError> {
        if let Some(dir) = output_dir.as_ref() {
            fs::create_dir_all(dir).map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "failed to create chunk artifact directory {}: {}",
                    dir.display(),
                    err
                ))
            })?;
        }

        let mut this = Self {
            output_dir,
            report: ChunkTrainingReport {
                merge_core_only: config.merge_core_only,
                requested_chunks: chunk_plan.chunks.len(),
                trainable_chunks: chunk_plan.trainable_chunks().count(),
                entries: Vec::new(),
            },
        };
        this.flush_report()?;
        Ok(this)
    }

    fn report_path(&self) -> Option<PathBuf> {
        self.output_dir
            .as_ref()
            .map(|dir| dir.join("chunk-report.json"))
    }

    fn scene_path(&self, chunk_id: usize) -> Option<PathBuf> {
        self.output_dir
            .as_ref()
            .map(|dir| dir.join(format!("chunk-{chunk_id:03}.ply")))
    }

    fn record_skipped(
        &mut self,
        chunk: &PlannedChunk,
        config: &TrainingConfig,
    ) -> Result<(), TrainingError> {
        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Skipped,
            message: Some(format!(
                "insufficient cameras: assigned {} < required {}",
                chunk.pose_indices.len(),
                config.min_cameras_per_chunk
            )),
            scene_path: None,
            pose_count: chunk.pose_indices.len(),
            local_points: chunk.point_indices.len(),
            used_frame_based_initialization: chunk.point_indices.is_empty(),
            effective_max_initial_gaussians: None,
            effective_render_scale: None,
            estimated_peak_gib: None,
            output_gaussians: None,
            core_filter_removed: None,
        });
        self.flush_report()
    }

    fn record_success(
        &mut self,
        chunk: &PlannedChunk,
        materialized: &MaterializedChunkDataset,
        override_plan: &ChunkTrainingOverridePlan,
        chunk_scene: &GaussianMap,
        core_filter_removed: usize,
    ) -> Result<(), TrainingError> {
        let scene_path = if let Some(path) = self.scene_path(chunk.chunk_id) {
            persist_gaussian_map_scene(
                &path,
                chunk_scene,
                override_plan.effective_config.iterations,
            )?;
            Some(path)
        } else {
            None
        };

        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Success,
            message: None,
            scene_path,
            pose_count: materialized.dataset.poses.len(),
            local_points: materialized.dataset.initial_points.len(),
            used_frame_based_initialization: materialized.used_frame_based_initialization,
            effective_max_initial_gaussians: Some(
                override_plan.effective_config.max_initial_gaussians,
            ),
            effective_render_scale: Some(override_plan.effective_config.metal_render_scale),
            estimated_peak_gib: Some(override_plan.estimate.estimated_peak_gib()),
            output_gaussians: Some(chunk_scene.len()),
            core_filter_removed: Some(core_filter_removed),
        });
        self.flush_report()
    }

    fn record_failure(
        &mut self,
        chunk: &PlannedChunk,
        materialized: &MaterializedChunkDataset,
        message: String,
        override_plan: Option<&ChunkTrainingOverridePlan>,
    ) -> Result<(), TrainingError> {
        self.report.entries.push(ChunkReportEntry {
            chunk_id: chunk.chunk_id,
            status: ChunkReportStatus::Failed,
            message: Some(message),
            scene_path: None,
            pose_count: materialized.dataset.poses.len(),
            local_points: materialized.dataset.initial_points.len(),
            used_frame_based_initialization: materialized.used_frame_based_initialization,
            effective_max_initial_gaussians: override_plan
                .map(|plan| plan.effective_config.max_initial_gaussians),
            effective_render_scale: override_plan
                .map(|plan| plan.effective_config.metal_render_scale),
            estimated_peak_gib: override_plan.map(|plan| plan.estimate.estimated_peak_gib()),
            output_gaussians: None,
            core_filter_removed: None,
        });
        self.flush_report()
    }

    fn flush_report(&mut self) -> Result<(), TrainingError> {
        let Some(report_path) = self.report_path() else {
            return Ok(());
        };
        let json = serde_json::to_string_pretty(&self.report).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to serialize chunk report {}: {}",
                report_path.display(),
                err
            ))
        })?;
        fs::write(&report_path, json).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to write chunk report {}: {}",
                report_path.display(),
                err
            ))
        })
    }
}

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

/// Public training profile selection.
///
/// `LegacyMetal` preserves the existing RustGS behavior. `LiteGsMacV1` reserves
/// a dedicated LiteGS-compatible path for Apple Silicon parity work.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingProfile {
    LegacyMetal,
    LiteGsMacV1,
}

impl Default for TrainingProfile {
    fn default() -> Self {
        Self::LegacyMetal
    }
}

impl std::fmt::Display for TrainingProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LegacyMetal => write!(f, "legacy-metal"),
            Self::LiteGsMacV1 => write!(f, "litegs-mac-v1"),
        }
    }
}

impl FromStr for TrainingProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "legacy-metal" => Ok(Self::LegacyMetal),
            "litegs-mac-v1" => Ok(Self::LiteGsMacV1),
            other => Err(format!(
                "unsupported training profile '{other}'. Expected one of: legacy-metal, litegs-mac-v1"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiteGsTileSize {
    pub width: usize,
    pub height: usize,
}

impl LiteGsTileSize {
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl Default for LiteGsTileSize {
    fn default() -> Self {
        Self::new(8, 16)
    }
}

impl std::fmt::Display for LiteGsTileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl FromStr for LiteGsTileSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let token = value.trim();
        let parts: Vec<&str> = token
            .split(|ch| matches!(ch, 'x' | 'X' | ',' | ':'))
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid LiteGS tile size '{token}'. Expected formats like 8x16 or 8,16"
            ));
        }

        let width = parts[0]
            .parse::<usize>()
            .map_err(|_| format!("invalid tile width in '{token}'"))?;
        let height = parts[1]
            .parse::<usize>()
            .map_err(|_| format!("invalid tile height in '{token}'"))?;
        if width == 0 || height == 0 {
            return Err(format!(
                "invalid LiteGS tile size '{token}'. Width and height must both be > 0"
            ));
        }

        Ok(Self::new(width, height))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsOpacityResetMode {
    Decay,
    Reset,
}

impl Default for LiteGsOpacityResetMode {
    fn default() -> Self {
        Self::Decay
    }
}

impl std::fmt::Display for LiteGsOpacityResetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decay => write!(f, "decay"),
            Self::Reset => write!(f, "reset"),
        }
    }
}

impl FromStr for LiteGsOpacityResetMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "decay" => Ok(Self::Decay),
            "reset" => Ok(Self::Reset),
            other => Err(format!(
                "unsupported LiteGS opacity reset mode '{other}'. Expected one of: decay, reset"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsPruneMode {
    Threshold,
    Weight,
}

impl Default for LiteGsPruneMode {
    fn default() -> Self {
        Self::Weight
    }
}

impl std::fmt::Display for LiteGsPruneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Threshold => write!(f, "threshold"),
            Self::Weight => write!(f, "weight"),
        }
    }
}

impl FromStr for LiteGsPruneMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "threshold" => Ok(Self::Threshold),
            "weight" => Ok(Self::Weight),
            other => Err(format!(
                "unsupported LiteGS prune mode '{other}'. Expected one of: threshold, weight"
            )),
        }
    }
}

/// Nested LiteGS-compatible configuration surface.
///
/// The defaults are chosen for the phased Apple Silicon parity plan:
/// non-clustered by default, sparse-grad off, and camera optimization deferred.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiteGsConfig {
    pub sh_degree: usize,
    pub cluster_size: usize,
    pub tile_size: LiteGsTileSize,
    pub sparse_grad: bool,
    pub reg_weight: f32,
    pub enable_transmittance: bool,
    pub enable_depth: bool,
    pub densify_from: usize,
    pub densify_until: Option<usize>,
    pub densification_interval: usize,
    pub opacity_reset_interval: usize,
    pub opacity_reset_mode: LiteGsOpacityResetMode,
    pub prune_mode: LiteGsPruneMode,
    pub target_primitives: usize,
    pub learnable_viewproj: bool,
}

impl Default for LiteGsConfig {
    fn default() -> Self {
        Self {
            sh_degree: 3,
            cluster_size: 0,
            tile_size: LiteGsTileSize::default(),
            sparse_grad: false,
            reg_weight: 0.0,
            enable_transmittance: false,
            enable_depth: false,
            densify_from: 3,
            densify_until: None,
            densification_interval: 5,
            opacity_reset_interval: 10,
            opacity_reset_mode: LiteGsOpacityResetMode::Decay,
            prune_mode: LiteGsPruneMode::Weight,
            target_primitives: 1_000_000,
            learnable_viewproj: false,
        }
    }
}

fn normalize_config_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Training backend implementation to use.
    ///
    /// Only `Metal` is supported for the top-level training flow.
    pub backend: TrainingBackend,
    /// Public training profile used to route the top-level trainer.
    pub training_profile: TrainingProfile,
    /// Nested LiteGS-compatible configuration surface.
    pub litegs: LiteGsConfig,
    /// Number of training iterations
    pub iterations: usize,
    /// Learning rate for positions (initial)
    pub lr_position: f32,
    /// Learning rate for positions (final) – exponential decay target.
    /// Default is 1/100 of lr_position. Set to 0 to disable decay.
    pub lr_pos_final: f32,
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
    /// Generate synthetic depth from image luminance when depth is unavailable.
    /// Disabled by default for RGB-only datasets because pseudo-depth targets
    /// can destabilize geometric optimization.
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
    /// Enable budget-driven chunked training orchestration.
    pub chunked_training: bool,
    /// Per-chunk memory budget in GiB used by future chunk planning.
    pub chunk_budget_gb: f32,
    /// Fractional overlap applied when expanding chunk bounds.
    pub chunk_overlap_ratio: f32,
    /// Minimum camera count required for a chunk to remain trainable.
    pub min_cameras_per_chunk: usize,
    /// Maximum number of chunks to generate (0 = automatic).
    pub max_chunks: usize,
    /// Keep only chunk core-region Gaussians during merge by default.
    pub merge_core_only: bool,
    /// Optional directory for chunk artifacts and machine-readable reports.
    pub chunk_artifact_dir: Option<PathBuf>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::default(),
            training_profile: TrainingProfile::default(),
            litegs: LiteGsConfig::default(),
            iterations: 30000,
            lr_position: 0.00016,
            lr_pos_final: 0.0000016,
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
            use_synthetic_depth: false,
            metal_render_scale: 0.5,
            metal_gaussian_chunk_size: 32,
            metal_profile_steps: false,
            metal_profile_interval: 25,
            metal_use_native_forward: true,
            chunked_training: false,
            chunk_budget_gb: 12.0,
            chunk_overlap_ratio: 0.15,
            min_cameras_per_chunk: 3,
            max_chunks: 0,
            merge_core_only: true,
            chunk_artifact_dir: None,
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
    match config.training_profile {
        TrainingProfile::LegacyMetal => train_legacy_metal(dataset, config),
        TrainingProfile::LiteGsMacV1 => train_litegs_mac_v1(dataset, config),
    }
}

#[cfg(feature = "gpu")]
fn train_legacy_metal(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let plan = select_training_execution_plan(dataset, config)?;
    execute_training_plan(dataset, config, plan)
}

#[cfg(feature = "gpu")]
fn train_litegs_mac_v1(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    validate_litegs_mac_v1_config(config)?;
    log::info!(
        "Training with LiteGS Mac V1 profile | sh_degree={} | cluster_size={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={}",
        config.litegs.sh_degree,
        config.litegs.cluster_size,
        config.litegs.tile_size,
        config.litegs.sparse_grad,
        config.litegs.reg_weight,
        config.litegs.enable_transmittance,
        config.litegs.enable_depth,
    );

    let plan = select_training_execution_plan(dataset, config)?;
    execute_training_plan(dataset, config, plan)
}

#[cfg(feature = "gpu")]
fn execute_training_plan(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    plan: TrainingExecutionPlan,
) -> Result<GaussianMap, TrainingError> {
    match plan.route {
        TrainingExecutionRoute::Standard => metal_trainer::train(dataset, config),
        TrainingExecutionRoute::ChunkedSingleChunk => {
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected single-chunk pass-through | requested_gaussians={} | affordable_gaussians={} | estimated_peak≈{:.1} GiB | effective_budget≈{:.1} GiB",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    estimate.estimated_peak_gib(),
                    estimate.effective_budget_gib(),
                );
            }
            metal_trainer::train(dataset, config)
        }
        TrainingExecutionRoute::ChunkedSequential => {
            let chunk_plan = plan
                .chunk_plan
                .as_ref()
                .expect("sequential route requires chunk plan");
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected sequential chunk execution | requested_gaussians={} | affordable_gaussians={} | chunks={} | trainable_chunks={}",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    chunk_plan.chunks.len(),
                    chunk_plan.trainable_chunks().count(),
                );
            }
            train_chunked_sequentially(dataset, config, chunk_plan)
        }
    }
}

fn validate_litegs_mac_v1_config(config: &TrainingConfig) -> Result<(), TrainingError> {
    let defaults = LiteGsConfig::default();
    let mut unsupported = Vec::new();

    if config.litegs.learnable_viewproj {
        unsupported.push(
            "learnable_viewproj is reserved but deferred for LiteGsMacV1 on Mac (future story, currently unsupported)"
                .to_string(),
        );
    }
    if config.litegs.cluster_size != defaults.cluster_size {
        unsupported.push(format!(
            "cluster_size={} requires Epic 19 clustered parity; LiteGsMacV1 currently supports only non-clustered training",
            config.litegs.cluster_size
        ));
    }
    if config.litegs.sparse_grad != defaults.sparse_grad {
        unsupported.push(
            "sparse_grad=true requires Epic 19 sparse-gradient parity and is not available in LiteGsMacV1 yet"
                .to_string(),
        );
    }
    if config.litegs.tile_size != defaults.tile_size {
        unsupported.push(format!(
            "tile_size={} overrides are reserved for later LiteGS parity work; bootstrap profile currently expects {}",
            config.litegs.tile_size, defaults.tile_size
        ));
    }
    if (config.litegs.reg_weight - defaults.reg_weight).abs() > f32::EPSILON {
        unsupported.push(format!(
            "reg_weight={} requires LiteGS loss parity work (Epic 18.4) and is not wired yet",
            config.litegs.reg_weight
        ));
    }
    if config.litegs.enable_transmittance != defaults.enable_transmittance {
        unsupported.push(
            "enable_transmittance=true requires LiteGS transmittance loss parity (Epic 18.4)"
                .to_string(),
        );
    }
    if config.litegs.enable_depth != defaults.enable_depth {
        unsupported
            .push("enable_depth=true requires LiteGS depth loss parity (Epic 18.4)".to_string());
    }
    if config.litegs.densify_from != defaults.densify_from {
        unsupported.push(format!(
            "densify_from={} requires LiteGS densify parity (Epic 20)",
            config.litegs.densify_from
        ));
    }
    if config.litegs.densify_until != defaults.densify_until {
        unsupported.push(format!(
            "densify_until={:?} requires LiteGS densify parity (Epic 20)",
            config.litegs.densify_until
        ));
    }
    if config.litegs.densification_interval != defaults.densification_interval {
        unsupported.push(format!(
            "densification_interval={} requires LiteGS densify parity (Epic 20)",
            config.litegs.densification_interval
        ));
    }
    if config.litegs.opacity_reset_interval != defaults.opacity_reset_interval {
        unsupported.push(format!(
            "opacity_reset_interval={} requires LiteGS opacity reset parity (Epic 20)",
            config.litegs.opacity_reset_interval
        ));
    }
    if config.litegs.opacity_reset_mode != defaults.opacity_reset_mode {
        unsupported.push(format!(
            "opacity_reset_mode={} requires LiteGS opacity reset parity (Epic 20)",
            config.litegs.opacity_reset_mode
        ));
    }
    if config.litegs.prune_mode != defaults.prune_mode {
        unsupported.push(format!(
            "prune_mode={} requires LiteGS/TamingGS prune parity (Epic 20)",
            config.litegs.prune_mode
        ));
    }
    if config.litegs.target_primitives != defaults.target_primitives {
        unsupported.push(format!(
            "target_primitives={} requires TamingGS target schedule parity (Epic 20)",
            config.litegs.target_primitives
        ));
    }

    if unsupported.is_empty() {
        return Ok(());
    }

    Err(TrainingError::TrainingFailed(format!(
        "LiteGsMacV1 bootstrap profile rejected unsupported overrides: {}",
        unsupported.join("; ")
    )))
}

#[cfg(feature = "gpu")]
fn select_training_execution_plan(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingExecutionPlan, TrainingError> {
    if !config.chunked_training {
        return Ok(TrainingExecutionPlan {
            route: TrainingExecutionRoute::Standard,
            chunk_estimate: None,
            chunk_plan: None,
        });
    }

    let estimate = metal_trainer::estimate_chunk_capacity(dataset, config)?;
    if estimate.requires_subdivision_or_degradation() {
        let chunk_plan =
            plan_spatial_chunks(dataset, config, Some(estimate.affordable_initial_gaussians))?;
        if chunk_plan.trainable_chunks().count() == 0 {
            return Err(TrainingError::TrainingFailed(format!(
                "chunked training planned {} chunks under {:.1} GiB, but none met the minimum camera threshold of {}. Recommendations: {}",
                chunk_plan.chunks.len(),
                estimate.effective_budget_gib(),
                config.min_cameras_per_chunk,
                estimate.recommendations().join("; "),
            )));
        }

        return Ok(TrainingExecutionPlan {
            route: TrainingExecutionRoute::ChunkedSequential,
            chunk_estimate: Some(estimate),
            chunk_plan: Some(chunk_plan),
        });
    }

    Ok(TrainingExecutionPlan {
        route: TrainingExecutionRoute::ChunkedSingleChunk,
        chunk_estimate: Some(estimate),
        chunk_plan: None,
    })
}

#[cfg(feature = "gpu")]
fn train_chunked_sequentially(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    chunk_plan: &ChunkPlan,
) -> Result<GaussianMap, TrainingError> {
    let mut merged_scene = GaussianMap::default();
    let total_chunks = chunk_plan.trainable_chunks().count();
    let mut persistence =
        ChunkPersistenceContext::new(config.chunk_artifact_dir.clone(), config, chunk_plan)?;

    for skipped in chunk_plan
        .chunks
        .iter()
        .filter(|chunk| chunk.disposition == ChunkDisposition::SkippedInsufficientCameras)
    {
        log::warn!(
            "Skipping chunk {} due to insufficient cameras | assigned_cameras={} | required_min={}",
            skipped.chunk_id,
            skipped.pose_indices.len(),
            config.min_cameras_per_chunk,
        );
        persistence.record_skipped(skipped, config)?;
    }

    execute_trainable_chunks_sequentially(dataset, chunk_plan, |chunk, materialized| {
        log::info!(
            "Training chunk {}/{} | chunk_id={} | poses={} | local_points={} | frame_init_fallback={}",
            chunk.chunk_id + 1,
            total_chunks,
            chunk.chunk_id,
            materialized.dataset.poses.len(),
            materialized.dataset.initial_points.len(),
            materialized.used_frame_based_initialization,
        );

        let override_plan = match adapt_chunk_training_config(&materialized.dataset, config) {
            Ok(plan) => plan,
            Err(err) => {
                persistence.record_failure(chunk, &materialized, err.to_string(), None)?;
                return Err(err);
            }
        };
        log::info!(
            "Chunk {} effective config | max_initial_gaussians={} | metal_render_scale={:.3} | lowered_gaussian_cap={} | lowered_render_scale={} | estimated_peak≈{:.1} GiB",
            chunk.chunk_id,
            override_plan.effective_config.max_initial_gaussians,
            override_plan.effective_config.metal_render_scale,
            override_plan.lowered_gaussian_cap,
            override_plan.lowered_render_scale,
            override_plan.estimate.estimated_peak_gib(),
        );

        let mut chunk_scene =
            match metal_trainer::train(&materialized.dataset, &override_plan.effective_config) {
                Ok(scene) => scene,
                Err(err) => {
                    persistence.record_failure(
                        chunk,
                        &materialized,
                        err.to_string(),
                        Some(&override_plan),
                    )?;
                    return Err(err);
                }
            };
        let core_filter_removed = merge_chunk_scene(
            &mut merged_scene,
            &mut chunk_scene,
            &chunk.core_bounds,
            config.merge_core_only,
        );
        persistence.record_success(
            chunk,
            &materialized,
            &override_plan,
            &chunk_scene,
            core_filter_removed,
        )?;
        log::info!(
            "Chunk {} complete | accumulated_gaussians={}",
            chunk.chunk_id,
            merged_scene.len(),
        );
        Ok(())
    })?;

    Ok(merged_scene)
}

#[cfg(feature = "gpu")]
fn execute_trainable_chunks_sequentially<F>(
    dataset: &TrainingDataset,
    chunk_plan: &ChunkPlan,
    mut execute: F,
) -> Result<(), TrainingError>
where
    F: FnMut(&PlannedChunk, MaterializedChunkDataset) -> Result<(), TrainingError>,
{
    for chunk in chunk_plan.trainable_chunks() {
        let materialized = materialize_chunk_dataset(dataset, chunk)?;
        execute(chunk, materialized)?;
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn retain_gaussians_in_bounds(scene: &mut GaussianMap, bounds: &ChunkBounds) -> usize {
    scene.retain(|gaussian| {
        let position = gaussian.position.to_array();
        (0..3).all(|axis| position[axis] >= bounds.min[axis] && position[axis] <= bounds.max[axis])
    })
}

fn merge_chunk_scene(
    merged_scene: &mut GaussianMap,
    chunk_scene: &mut GaussianMap,
    core_bounds: &ChunkBounds,
    merge_core_only: bool,
) -> usize {
    let original_chunk = if merge_core_only {
        Some(chunk_scene.clone())
    } else {
        None
    };

    let removed = if merge_core_only {
        let original_len = chunk_scene.len();
        let removed = retain_gaussians_in_bounds(chunk_scene, core_bounds);
        if original_len > 0 && chunk_scene.is_empty() {
            log::warn!(
                "Core-only merge filter removed all {} gaussians for a chunk; falling back to unfiltered merge",
                original_len,
            );
            if let Some(original_chunk) = original_chunk {
                *chunk_scene = original_chunk;
            }
            0
        } else {
            log::info!(
                "Core-only merge filter removed {} gaussians before aggregation",
                removed,
            );
            removed
        }
    } else {
        0
    };
    merged_scene
        .gaussians_mut()
        .extend(chunk_scene.gaussians().iter().cloned());
    merged_scene.update_states();
    removed
}

fn persist_gaussian_map_scene(
    path: &Path,
    scene: &GaussianMap,
    iterations: usize,
) -> Result<(), TrainingError> {
    let gaussians = scene
        .gaussians()
        .iter()
        .map(|g| {
            crate::Gaussian::new(
                g.position.into(),
                g.scale.into(),
                [g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z],
                g.opacity,
                g.color,
            )
        })
        .collect::<Vec<_>>();
    let metadata = crate::SceneMetadata {
        iterations,
        final_loss: 0.0,
        gaussian_count: gaussians.len(),
    };
    crate::save_scene_ply(path, &gaussians, &metadata).map_err(|err| {
        TrainingError::TrainingFailed(format!("failed to persist {}: {}", path.display(), err))
    })
}

#[cfg(feature = "gpu")]
fn adapt_chunk_training_config(
    dataset: &TrainingDataset,
    base_config: &TrainingConfig,
) -> Result<ChunkTrainingOverridePlan, TrainingError> {
    let mut effective_config = base_config.clone();
    let mut estimate = metal_trainer::estimate_chunk_capacity(dataset, &effective_config)?;
    let mut lowered_gaussian_cap = false;
    let mut lowered_render_scale = false;

    if estimate.requires_subdivision_or_degradation()
        && estimate.affordable_initial_gaussians < effective_config.max_initial_gaussians
    {
        effective_config.max_initial_gaussians = estimate.affordable_initial_gaussians.max(1);
        lowered_gaussian_cap = true;
        estimate = metal_trainer::estimate_chunk_capacity(dataset, &effective_config)?;
    }

    while estimate.requires_subdivision_or_degradation()
        && effective_config.metal_render_scale > 0.125
    {
        let next_scale = (effective_config.metal_render_scale * 0.75).max(0.125);
        if (next_scale - effective_config.metal_render_scale).abs() < f32::EPSILON {
            break;
        }
        effective_config.metal_render_scale = next_scale;
        lowered_render_scale = true;
        estimate = metal_trainer::estimate_chunk_capacity(dataset, &effective_config)?;
    }

    if estimate.requires_subdivision_or_degradation() {
        return Err(TrainingError::TrainingFailed(format!(
            "chunk cannot be made safe under the configured budget after chunk-local overrides: chunk_budget_gb={:.2}, max_initial_gaussians={}, metal_render_scale={:.3}, safe_cap={}, estimated_peak≈{:.1} GiB. Recommendations: {}",
            base_config.chunk_budget_gb,
            effective_config.max_initial_gaussians,
            effective_config.metal_render_scale,
            estimate.affordable_initial_gaussians,
            estimate.estimated_peak_gib(),
            estimate.recommendations().join("; "),
        )));
    }

    Ok(ChunkTrainingOverridePlan {
        effective_config,
        estimate,
        lowered_gaussian_cap,
        lowered_render_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        validate_litegs_mac_v1_config, LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode,
        LiteGsTileSize, TrainingBackend, TrainingConfig, TrainingProfile,
    };
    use std::str::FromStr;

    #[test]
    fn default_training_backend_is_metal() {
        assert_eq!(TrainingBackend::default(), TrainingBackend::Metal);
        let config = TrainingConfig::default();
        assert_eq!(config.backend, TrainingBackend::Metal);
        assert_eq!(config.training_profile, TrainingProfile::LegacyMetal);
        assert!(config.metal_use_native_forward);
        assert_eq!(config.prune_interval, 100);
        assert!(!config.chunked_training);
        assert_eq!(config.chunk_budget_gb, 12.0);
        assert_eq!(config.chunk_overlap_ratio, 0.15);
        assert_eq!(config.min_cameras_per_chunk, 3);
        assert_eq!(config.max_chunks, 0);
        assert!(config.merge_core_only);
        assert!(config.chunk_artifact_dir.is_none());
        assert_eq!(config.litegs, LiteGsConfig::default());
    }

    #[test]
    fn litegs_config_defaults_match_mac_bootstrap_plan() {
        let litegs = LiteGsConfig::default();
        assert_eq!(litegs.sh_degree, 3);
        assert_eq!(litegs.cluster_size, 0);
        assert_eq!(litegs.tile_size, LiteGsTileSize::new(8, 16));
        assert!(!litegs.sparse_grad);
        assert_eq!(litegs.reg_weight, 0.0);
        assert!(!litegs.enable_transmittance);
        assert!(!litegs.enable_depth);
        assert_eq!(litegs.densify_from, 3);
        assert_eq!(litegs.densify_until, None);
        assert_eq!(litegs.densification_interval, 5);
        assert_eq!(litegs.opacity_reset_interval, 10);
        assert_eq!(litegs.opacity_reset_mode, LiteGsOpacityResetMode::Decay);
        assert_eq!(litegs.prune_mode, LiteGsPruneMode::Weight);
        assert_eq!(litegs.target_primitives, 1_000_000);
        assert!(!litegs.learnable_viewproj);
    }

    #[test]
    fn training_profile_and_litegs_enums_parse_cli_tokens() {
        assert_eq!(
            TrainingProfile::from_str("legacy-metal").unwrap(),
            TrainingProfile::LegacyMetal
        );
        assert_eq!(
            TrainingProfile::from_str("litegs_mac_v1").unwrap(),
            TrainingProfile::LiteGsMacV1
        );
        assert_eq!(
            LiteGsTileSize::from_str("16x8").unwrap(),
            LiteGsTileSize::new(16, 8)
        );
        assert_eq!(
            LiteGsTileSize::from_str("16,8").unwrap(),
            LiteGsTileSize::new(16, 8)
        );
        assert_eq!(
            LiteGsOpacityResetMode::from_str("reset").unwrap(),
            LiteGsOpacityResetMode::Reset
        );
        assert_eq!(
            LiteGsPruneMode::from_str("threshold").unwrap(),
            LiteGsPruneMode::Threshold
        );
    }

    #[test]
    fn litegs_mac_v1_accepts_bootstrap_defaults() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        };
        validate_litegs_mac_v1_config(&config).unwrap();
    }

    #[test]
    fn litegs_mac_v1_rejects_clustered_override_before_epic_19() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                cluster_size: 128,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        let err = validate_litegs_mac_v1_config(&config).unwrap_err();
        assert!(err.to_string().contains("cluster_size=128"));
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::{
        adapt_chunk_training_config, execute_trainable_chunks_sequentially, merge_chunk_scene,
        persist_gaussian_map_scene, plan_spatial_chunks, select_training_execution_plan,
        ChunkPersistenceContext, PlannedChunk, TrainingConfig, TrainingExecutionRoute,
    };
    use crate::{
        ChunkBounds, ChunkDisposition, Gaussian3D, GaussianMap, Intrinsics, ScenePose,
        TrainingDataset, SE3,
    };
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::rc::Rc;
    use tempfile::tempdir;

    fn make_dataset(frame_count: usize, width: u32, height: u32) -> TrainingDataset {
        let intrinsics = Intrinsics::from_focal(500.0, width, height);
        let mut dataset = TrainingDataset::new(intrinsics);
        for idx in 0..frame_count {
            dataset.add_pose(ScenePose::new(
                idx as u64,
                PathBuf::from(format!("frame_{idx:04}.png")),
                SE3::identity(),
                idx as f64,
            ));
        }
        dataset
    }

    #[test]
    fn non_chunked_execution_plan_uses_standard_route() {
        let dataset = make_dataset(3, 64, 64);
        let plan = select_training_execution_plan(&dataset, &TrainingConfig::default()).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::Standard);
        assert!(plan.chunk_estimate.is_none());
    }

    #[test]
    fn chunked_execution_plan_selects_single_chunk_route_when_affordable() {
        let dataset = make_dataset(3, 32, 32);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            max_initial_gaussians: 128,
            ..TrainingConfig::default()
        };
        let plan = select_training_execution_plan(&dataset, &config).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSingleChunk);
        let estimate = plan
            .chunk_estimate
            .expect("chunk estimate should be present");
        assert!(!estimate.requires_subdivision_or_degradation());
        assert!(plan.chunk_plan.is_none());
    }

    #[test]
    fn chunked_execution_plan_uses_sequential_route_when_subdivision_is_required() {
        let dataset = make_dataset(5, 1920, 1080);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            metal_render_scale: 1.0,
            max_initial_gaussians: 57_474,
            max_chunks: 4,
            ..TrainingConfig::default()
        };
        let plan = select_training_execution_plan(&dataset, &config).unwrap();
        assert_eq!(plan.route, TrainingExecutionRoute::ChunkedSequential);
        assert!(plan.chunk_estimate.is_some());
        assert!(plan.chunk_plan.is_some());
        assert!(plan.chunk_plan.unwrap().chunks.len() >= 2);
    }

    #[test]
    fn sequential_chunk_executor_runs_chunks_one_at_a_time() {
        let mut dataset = make_dataset(4, 64, 64);
        dataset.add_point([0.0, 0.0, 0.0], None);
        dataset.add_point([1.0, 0.0, 0.0], None);
        dataset.add_point([8.0, 0.0, 0.0], None);
        dataset.add_point([9.0, 0.0, 0.0], None);
        dataset.poses[0].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        dataset.poses[1].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        dataset.poses[2].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[8.0, 0.0, 0.0]);
        dataset.poses[3].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[9.0, 0.0, 0.0]);

        let config = TrainingConfig {
            min_cameras_per_chunk: 1,
            ..TrainingConfig::default()
        };
        let chunk_plan = plan_spatial_chunks(&dataset, &config, Some(2)).unwrap();
        let active = Rc::new(Cell::new(0usize));
        let max_active = Rc::new(Cell::new(0usize));
        let visited = Rc::new(Cell::new(0usize));

        execute_trainable_chunks_sequentially(&dataset, &chunk_plan, |chunk, materialized| {
            struct Guard {
                active: Rc<Cell<usize>>,
            }

            impl Drop for Guard {
                fn drop(&mut self) {
                    self.active.set(self.active.get().saturating_sub(1));
                }
            }

            active.set(active.get() + 1);
            max_active.set(max_active.get().max(active.get()));
            let _guard = Guard {
                active: Rc::clone(&active),
            };

            visited.set(visited.get() + 1);
            assert_eq!(materialized.dataset.poses.len(), chunk.pose_indices.len());
            assert!(active.get() == 1);
            Ok(())
        })
        .unwrap();

        assert_eq!(visited.get(), chunk_plan.trainable_chunks().count());
        assert_eq!(max_active.get(), 1);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn adaptive_chunk_config_lowers_gaussian_cap_before_render_scale() {
        let mut dataset = make_dataset(2, 128, 128);
        for idx in 0..64 {
            dataset.add_point([idx as f32 * 0.1, 0.0, 1.0], None);
        }
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 64,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };

        let override_plan = adapt_chunk_training_config(&dataset, &config).unwrap();
        assert!(override_plan.lowered_gaussian_cap);
        assert!(override_plan.effective_config.max_initial_gaussians < 64);
    }

    #[test]
    fn adaptive_chunk_config_reduces_render_scale_when_gaussian_cap_is_not_enough() {
        let dataset = make_dataset(40, 1920, 1080);
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 8,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };

        let override_plan = adapt_chunk_training_config(&dataset, &config).unwrap();
        assert!(override_plan.lowered_render_scale);
        assert!(override_plan.effective_config.metal_render_scale < 1.0);
    }

    #[test]
    fn persist_gaussian_map_scene_writes_chunk_ply() {
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("chunk-000.ply");
        let scene = GaussianMap::from_gaussians(vec![Gaussian3D::default()]);

        persist_gaussian_map_scene(&path, &scene, 42).unwrap();

        assert!(path.exists());
        let (gaussians, metadata) = crate::load_scene_ply(&path).unwrap();
        assert_eq!(gaussians.len(), 1);
        assert_eq!(metadata.iterations, 42);
    }

    #[test]
    fn chunk_persistence_writes_report_entries() {
        let tempdir = tempdir().unwrap();
        let chunk = PlannedChunk {
            chunk_id: 0,
            core_bounds: ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            overlap_bounds: ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            pose_indices: vec![0, 1],
            point_indices: vec![0],
            disposition: ChunkDisposition::Trainable,
        };
        let chunk_plan = super::ChunkPlan {
            bounds_source: crate::ChunkBoundsSource::SparsePoints,
            scene_bounds: chunk.core_bounds,
            chunk_axis: 0,
            requested_chunks: 1,
            chunks: vec![chunk.clone()],
        };
        let config = TrainingConfig {
            chunked_training: true,
            chunk_artifact_dir: Some(tempdir.path().join("artifacts")),
            ..TrainingConfig::default()
        };
        let dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 32, 32));
        let materialized = crate::MaterializedChunkDataset {
            chunk_id: 0,
            dataset,
            used_frame_based_initialization: true,
        };
        let override_plan = super::ChunkTrainingOverridePlan {
            effective_config: config.clone(),
            estimate: crate::estimate_chunk_capacity(
                &TrainingDataset {
                    intrinsics: Intrinsics::from_focal(500.0, 32, 32),
                    depth_scale: 1000.0,
                    poses: vec![ScenePose::new(
                        0,
                        PathBuf::from("frame.png"),
                        SE3::identity(),
                        0.0,
                    )],
                    initial_points: vec![],
                },
                &config,
            )
            .unwrap(),
            lowered_gaussian_cap: false,
            lowered_render_scale: false,
        };
        let scene = GaussianMap::from_gaussians(vec![Gaussian3D::default()]);

        let mut persistence =
            ChunkPersistenceContext::new(config.chunk_artifact_dir.clone(), &config, &chunk_plan)
                .unwrap();
        persistence
            .record_success(&chunk, &materialized, &override_plan, &scene, 0)
            .unwrap();

        let report_path = tempdir.path().join("artifacts").join("chunk-report.json");
        assert!(report_path.exists());
        let report_json = std::fs::read_to_string(report_path).unwrap();
        let report: serde_json::Value = serde_json::from_str(&report_json).unwrap();
        assert_eq!(report["entries"][0]["status"], "success");
        assert_eq!(report["entries"][0]["output_gaussians"], 1);
        assert_eq!(
            report["entries"][0]["used_frame_based_initialization"],
            true
        );
    }

    #[test]
    fn adaptive_chunk_configs_keep_each_trainable_chunk_within_budget_envelope() {
        let mut dataset = make_dataset(6, 1920, 1080);
        dataset.add_point([0.0, 0.0, 1.0], None);
        dataset.add_point([1.0, 0.0, 1.0], None);
        dataset.add_point([2.0, 0.0, 1.0], None);
        dataset.add_point([8.0, 0.0, 1.0], None);
        dataset.add_point([9.0, 0.0, 1.0], None);
        dataset.add_point([10.0, 0.0, 1.0], None);
        dataset.poses[0].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        dataset.poses[1].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        dataset.poses[2].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[2.0, 0.0, 0.0]);
        dataset.poses[3].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[8.0, 0.0, 0.0]);
        dataset.poses[4].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[9.0, 0.0, 0.0]);
        dataset.poses[5].pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[10.0, 0.0, 0.0]);

        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.35,
            max_initial_gaussians: 64,
            metal_render_scale: 1.0,
            min_cameras_per_chunk: 1,
            ..TrainingConfig::default()
        };
        let chunk_plan = plan_spatial_chunks(&dataset, &config, Some(3)).unwrap();

        for chunk in chunk_plan.trainable_chunks() {
            let materialized = crate::materialize_chunk_dataset(&dataset, chunk).unwrap();
            let override_plan =
                adapt_chunk_training_config(&materialized.dataset, &config).unwrap();
            assert!(
                override_plan.estimate.estimated_peak_gib()
                    <= override_plan.estimate.effective_budget_gib() + 1e-6,
                "chunk {} estimate {:.3} GiB exceeded effective budget {:.3} GiB",
                chunk.chunk_id,
                override_plan.estimate.estimated_peak_gib(),
                override_plan.estimate.effective_budget_gib(),
            );
        }
    }

    #[test]
    fn core_only_merge_filter_retains_only_gaussians_inside_core_bounds() {
        let mut merged = GaussianMap::default();
        let mut chunk_scene = GaussianMap::from_gaussians(vec![
            Gaussian3D::new(
                glam::Vec3::new(0.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [1.0, 0.0, 0.0],
            ),
            Gaussian3D::new(
                glam::Vec3::new(1.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [0.0, 1.0, 0.0],
            ),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed = merge_chunk_scene(&mut merged, &mut chunk_scene, &core_bounds, true);

        assert_eq!(removed, 1);
        assert_eq!(merged.len(), 1);
        assert!((merged.gaussians()[0].position.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn merge_without_core_filter_keeps_overlap_gaussians() {
        let mut merged = GaussianMap::default();
        let mut chunk_scene = GaussianMap::from_gaussians(vec![
            Gaussian3D::new(
                glam::Vec3::new(0.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [1.0, 0.0, 0.0],
            ),
            Gaussian3D::new(
                glam::Vec3::new(1.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [0.0, 1.0, 0.0],
            ),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed = merge_chunk_scene(&mut merged, &mut chunk_scene, &core_bounds, false);

        assert_eq!(removed, 0);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn core_only_merge_filter_falls_back_when_it_would_empty_chunk() {
        let mut merged = GaussianMap::default();
        let mut chunk_scene = GaussianMap::from_gaussians(vec![
            Gaussian3D::new(
                glam::Vec3::new(1.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [1.0, 0.0, 0.0],
            ),
            Gaussian3D::new(
                glam::Vec3::new(2.5, 0.0, 0.0),
                glam::Vec3::ONE,
                glam::Quat::IDENTITY,
                1.0,
                [0.0, 1.0, 0.0],
            ),
        ]);
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let removed = merge_chunk_scene(&mut merged, &mut chunk_scene, &core_bounds, true);

        assert_eq!(removed, 0);
        assert_eq!(merged.len(), 2);
        assert_eq!(chunk_scene.len(), 2);
    }

    #[test]
    fn merged_scene_output_contains_gaussians_from_multiple_chunks() {
        let mut merged = GaussianMap::default();
        let core_bounds = ChunkBounds {
            min: [0.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        let mut chunk_a = GaussianMap::from_gaussians(vec![Gaussian3D::new(
            glam::Vec3::new(0.5, 0.0, 0.0),
            glam::Vec3::ONE,
            glam::Quat::IDENTITY,
            1.0,
            [1.0, 0.0, 0.0],
        )]);
        let mut chunk_b = GaussianMap::from_gaussians(vec![Gaussian3D::new(
            glam::Vec3::new(2.0, 0.0, 0.0),
            glam::Vec3::ONE,
            glam::Quat::IDENTITY,
            1.0,
            [0.0, 1.0, 0.0],
        )]);

        merge_chunk_scene(&mut merged, &mut chunk_a, &core_bounds, true);
        merge_chunk_scene(
            &mut merged,
            &mut chunk_b,
            &ChunkBounds {
                min: [1.5, -1.0, -1.0],
                max: [2.5, 1.0, 1.0],
            },
            true,
        );

        assert_eq!(merged.len(), 2);
        let tempdir = tempdir().unwrap();
        let path = tempdir.path().join("merged-scene.ply");
        persist_gaussian_map_scene(&path, &merged, 10).unwrap();
        let (gaussians, _) = crate::load_scene_ply(&path).unwrap();
        assert_eq!(gaussians.len(), 2);
    }

    #[test]
    fn default_training_config_disables_synthetic_depth() {
        assert!(!TrainingConfig::default().use_synthetic_depth);
    }
}
