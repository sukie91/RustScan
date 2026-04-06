//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, and optimizer
//! updates on the Metal device, while replacing the generic autograd hot path
//! with a specialized analytical backward pass.

use std::cmp::Ordering;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor, Var};
use glam::{Mat3, Quat, Vec3};

use crate::diff::diff_splat::{
    DiffCamera, TrainableColorRepresentation, TrainableGaussians, SH_C0,
};
use crate::training::clustering::ClusterAssignment;
use crate::{GaussianMap, TrainingDataset, TrainingError};

use super::data_loading::{
    load_training_data, map_from_trainable, trainable_from_map, LoadedTrainingData,
};
use super::eval::summarize_training_metrics;
use super::metal_backward::MetalBackwardGrads;
use super::metal_loss::{masked_mean_abs_diff, mean_abs_diff, ssim_gradient};
use super::metal_runtime::{
    ChunkPixelWindow, MetalBufferSlot, MetalProjectionRecord, MetalRuntime, MetalTileBins,
    NativeForwardProfile, METAL_TILE_SIZE,
};
use super::parity_harness::{ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics};
use super::{LiteGsConfig, TrainingConfig, TrainingProfile};

#[cfg(test)]
use super::metal_runtime::ScreenRect;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_METAL_MEMORY_BUDGET_BYTES: u64 = 24 * GIB;
const METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR: u64 = 13;
const METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR: u64 = 20;
const METAL_WARN_BUDGET_NUMERATOR: u64 = 85;
const METAL_WARN_BUDGET_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_SAFETY_NUMERATOR: u64 = 15;
const METAL_ESTIMATE_SAFETY_DENOMINATOR: u64 = 100;
const METAL_ESTIMATE_MIN_SAFETY_BYTES: u64 = 256 * MIB;
const METAL_SOURCE_FRAME_BYTES_PER_PIXEL: u64 = 16;
const METAL_RESIZED_CPU_TARGET_BYTES_PER_PIXEL: u64 = 16;
const METAL_GPU_TARGET_BYTES_PER_PIXEL: u64 = 16;
const METAL_PIXEL_STATE_BYTES_PER_PIXEL: u64 = 40;
const METAL_GAUSSIAN_STATE_BYTES: u64 = 168;
const METAL_PROJECTED_BYTES_PER_GAUSSIAN: u64 = 64;
const METAL_CHUNK_WORKSPACE_BYTES_PER_GAUSSIAN_PIXEL: u64 = 64;
const METAL_RETAINED_GRAPH_BYTES_PER_GAUSSIAN_PIXEL: u64 = 224;
const LITEGS_LAMBDA_DSSIM: f32 = 0.2;
const LITEGS_DEPTH_LOSS_WEIGHT: f32 = 0.1;
const LITEGS_SH_ACTIVATION_EPOCH_INTERVAL: usize = 5;
const LITEGS_DENSIFY_GRAD_THRESHOLD: f32 = 0.00015;
const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_PERCENT_DENSE: f32 = 0.01;
const LITEGS_OPACITY_DECAY_RATE: f32 = 0.5;
const LITEGS_OPACITY_DECAY_MIN: f32 = 1.0 / 128.0;
const SH_C1: f32 = 0.488_602_52;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];
const SH_C4: [f32; 9] = [
    2.503_343,
    -1.770_130_8,
    0.946_174_7,
    -0.669_046_5,
    0.105_785_55,
    -0.669_046_5,
    0.473_087_34,
    -1.770_130_8,
    0.625_835_7,
];

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsOptimizerLrs {
    pub xyz: Option<f32>,
    pub sh_0: Option<f32>,
    pub sh_rest: Option<f32>,
    pub opacity: Option<f32>,
    pub scale: Option<f32>,
    pub rot: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsTrainingTelemetry {
    pub loss_terms: ParityLossTerms,
    pub loss_curve_samples: Vec<ParityLossCurveSample>,
    pub topology: ParityTopologyMetrics,
    pub active_sh_degree: Option<usize>,
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
    pub depth_grad_scale: Option<f32>,
    pub rotation_frozen: bool,
    pub learning_rates: LiteGsOptimizerLrs,
}

static LAST_METAL_TRAINING_TELEMETRY: OnceLock<Mutex<Option<LiteGsTrainingTelemetry>>> =
    OnceLock::new();

fn store_last_metal_training_telemetry(telemetry: Option<LiteGsTrainingTelemetry>) {
    let slot = LAST_METAL_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = telemetry;
}

pub fn last_metal_training_telemetry() -> Option<LiteGsTrainingTelemetry> {
    let slot = LAST_METAL_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

struct MetalTrainingFrame {
    camera: DiffCamera,
    target_color: Tensor,
    target_depth: Tensor,
    target_color_cpu: Vec<f32>,
    target_depth_cpu: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct TopologyCandidateInfo {
    max_scale: f32,
    opacity: f32,
    mean2d_grad: f32,
    fragment_weight: f32,
    fragment_err_score: f32,
    visible_count: usize,
    age_eligible: bool,
    invisible: bool,
    prune_candidate: bool,
    growth_candidate: bool,
}

#[derive(Debug, Default)]
struct TopologyAnalysis {
    infos: Vec<TopologyCandidateInfo>,
    clone_candidates: usize,
    split_candidates: usize,
    prune_candidates: usize,
    growth_candidates: usize,
    active_grad_stats: usize,
    small_scale_stats: usize,
    opacity_ready_stats: usize,
    max_grad: f32,
    mean_grad: f32,
}

#[derive(Debug, Default)]
struct LiteGsDensifySelection {
    selected_indices: Vec<usize>,
    replacement_count: usize,
    extra_growth_count: usize,
}

struct ProjectedGaussians {
    source_indices: Tensor,
    u: Tensor,
    v: Tensor,
    sigma_x: Tensor,
    sigma_y: Tensor,
    raw_sigma_x: Tensor,
    raw_sigma_y: Tensor,
    depth: Tensor,
    opacity: Tensor,
    opacity_logits: Tensor,
    scale3d: Tensor,
    colors: Tensor,
    min_x: Tensor,
    max_x: Tensor,
    min_y: Tensor,
    max_y: Tensor,
    visible_source_indices: Vec<u32>,
    visible_count: usize,
    tile_bins: MetalTileBins,
    staging_source: ProjectionStagingSource,
}

pub(crate) struct RenderedFrame {
    pub(crate) color: Tensor,
    pub(crate) depth: Tensor,
    pub(crate) alpha: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionStagingSource {
    TensorReadback,
    RuntimeBufferRead,
}

#[derive(Debug, Clone)]
pub(crate) struct CpuProjectedGaussian {
    pub(crate) source_idx: u32,
    pub(crate) u: f32,
    pub(crate) v: f32,
    pub(crate) sigma_x: f32,
    pub(crate) sigma_y: f32,
    pub(crate) raw_sigma_x: f32,
    pub(crate) raw_sigma_y: f32,
    pub(crate) depth: f32,
    pub(crate) opacity: f32,
    pub(crate) opacity_logit: f32,
    pub(crate) scale3d: [f32; 3],
    pub(crate) color: [f32; 3],
    pub(crate) min_x: f32,
    pub(crate) max_x: f32,
    pub(crate) min_y: f32,
    pub(crate) max_y: f32,
}

impl ProjectedGaussians {
    fn visible_source_indices(&self) -> &[u32] {
        &self.visible_source_indices
    }
}

#[derive(Debug, Clone)]
struct GaussianParameterSnapshot {
    positions: Vec<f32>,
    log_scales: Vec<f32>,
    rotations: Vec<f32>,
    opacity_logits: Vec<f32>,
    colors: Vec<f32>,
    sh_rest: Vec<f32>,
    color_representation: TrainableColorRepresentation,
}

impl GaussianParameterSnapshot {
    fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales[base],
            self.log_scales[base + 1],
            self.log_scales[base + 2],
        ]
    }

    fn rotation(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    fn color(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.colors[base],
            self.colors[base + 1],
            self.colors[base + 2],
        ]
    }

    fn sh_rest_row_width(&self) -> usize {
        self.color_representation.sh_rest_coeff_count() * 3
    }

    fn sh_rest(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_rest, self.sh_rest_row_width(), idx)
    }

    fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    fn push(
        &mut self,
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity_logit: f32,
        color: [f32; 3],
        sh_rest: &[f32],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.colors.extend_from_slice(&color);
        let sh_rest_row_width = self.sh_rest_row_width();
        if sh_rest_row_width > 0 {
            let copied = sh_rest.len().min(sh_rest_row_width);
            self.sh_rest.extend_from_slice(&sh_rest[..copied]);
            if copied < sh_rest_row_width {
                self.sh_rest
                    .resize(self.sh_rest.len() + (sh_rest_row_width - copied), 0.0);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TileBinningStats {
    active_tiles: usize,
    tile_gaussian_refs: usize,
    max_gaussians_per_tile: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct NativeParityProfile {
    setup: Duration,
    staging: Duration,
    kernel: Duration,
    total: Duration,
    color_max_abs: f32,
    depth_max_abs: f32,
    alpha_max_abs: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct MetalGaussianStats {
    mean2d_grad: RunningMoments,
    fragment_weight: RunningMoments,
    fragment_err: RunningMoments,
    visible_count: usize,
    age: usize,
    /// Number of consecutive epochs where this Gaussian was not visible.
    /// Reset to 0 when visible, incremented when not rendered.
    consecutive_invisible_epochs: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningMoments {
    mean: f32,
    m2: f32,
    count: usize,
}

impl RunningMoments {
    fn update(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.count = self.count.saturating_add(1);
        let count = self.count as f32;
        let delta = value - self.mean;
        self.mean += delta / count;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(self) -> f32 {
        if self.count <= 1 {
            0.0
        } else {
            self.m2 / self.count as f32
        }
    }
}

struct MetalAdamState {
    m_pos: Tensor,
    v_pos: Tensor,
    m_scale: Tensor,
    v_scale: Tensor,
    m_rot: Tensor,
    v_rot: Tensor,
    m_op: Tensor,
    v_op: Tensor,
    m_color: Tensor,
    v_color: Tensor,
    m_sh_rest: Tensor,
    v_sh_rest: Tensor,
}

impl MetalAdamState {
    fn new(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        Ok(Self {
            m_pos: gaussians.positions().zeros_like()?,
            v_pos: gaussians.positions().zeros_like()?,
            m_scale: gaussians.scales.as_tensor().zeros_like()?,
            v_scale: gaussians.scales.as_tensor().zeros_like()?,
            m_rot: gaussians.rotations.as_tensor().zeros_like()?,
            v_rot: gaussians.rotations.as_tensor().zeros_like()?,
            m_op: gaussians.opacities.as_tensor().zeros_like()?,
            v_op: gaussians.opacities.as_tensor().zeros_like()?,
            m_color: gaussians.colors().zeros_like()?,
            v_color: gaussians.colors().zeros_like()?,
            m_sh_rest: gaussians.sh_rest().zeros_like()?,
            v_sh_rest: gaussians.sh_rest().zeros_like()?,
        })
    }
}

pub struct MetalTrainer {
    training_profile: TrainingProfile,
    litegs: LiteGsConfig,
    device: Device,
    render_width: usize,
    render_height: usize,
    pixel_count: usize,
    source_pixel_count: usize,
    chunk_size: usize,
    densify_interval: usize,
    prune_interval: usize,
    topology_warmup: usize,
    topology_log_interval: usize,
    prune_threshold: f32,
    legacy_densify_grad_threshold: f32,
    legacy_clone_scale_threshold: f32,
    legacy_split_scale_threshold: f32,
    legacy_prune_scale_threshold: f32,
    legacy_max_densify_per_update: usize,
    max_gaussian_budget: usize,
    topology_memory_budget: Option<MetalMemoryBudget>,
    scene_extent: f32,
    lr_pos: f32,
    lr_pos_final: f32,
    max_iterations: usize,
    lr_scale: f32,
    lr_rotation: f32,
    lr_opacity: f32,
    lr_color: f32,
    lr_sh_rest: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    rotation_frozen: bool,
    active_sh_degree: usize,
    last_loss_terms: ParityLossTerms,
    topology_metrics: ParityTopologyMetrics,
    last_learning_rates: LiteGsOptimizerLrs,
    last_depth_valid_pixels: Option<usize>,
    last_depth_grad_scale: Option<f32>,
    loss_curve_samples: Vec<ParityLossCurveSample>,
    profile_steps: bool,
    profile_interval: usize,
    use_native_forward: bool,
    runtime: MetalRuntime,
    adam: Option<MetalAdamState>,
    gaussian_stats: Vec<MetalGaussianStats>,
    iteration: usize,
    loss_history: Vec<f32>,
    last_step_duration: Option<Duration>,
    cached_target_frame_idx: Option<usize>,
    /// Learnable camera poses (optional, enabled by learnable_viewproj)
    pose_embeddings: Option<crate::training::pose_embedding::PoseEmbeddings>,
    /// Cluster assignment for frustum culling (enabled by cluster_size > 0)
    cluster_assignment: Option<ClusterAssignment>,
}

struct MetalTrainingStats {
    final_loss: f32,
    final_step_loss: f32,
    telemetry: LiteGsTrainingTelemetry,
}

struct MetalStepOutcome {
    loss: f32,
    visible_gaussians: usize,
    total_gaussians: usize,
    profile: Option<MetalStepProfile>,
}

#[derive(Debug, Clone, Copy)]
struct MetalBackwardLossScales {
    color: f32,
    depth: f32,
    ssim: f32,
    alpha: f32,
}

fn should_record_loss_curve_sample(iter: usize, max_iterations: usize) -> bool {
    iter < 5 || iter % 25 == 0 || iter + 1 == max_iterations
}

fn record_topology_epoch(
    first: &mut Option<usize>,
    last: &mut Option<usize>,
    epoch: Option<usize>,
) {
    let Some(epoch) = epoch else {
        return;
    };
    *first = Some(first.map_or(epoch, |current| current.min(epoch)));
    *last = Some(last.map_or(epoch, |current| current.max(epoch)));
}

fn debug_training_step_probe_enabled() -> bool {
    std::env::var_os("RUSTGS_DEBUG_TRAIN_STEP").is_some()
}

fn abs_stats(values: &[f32]) -> (f32, f32, usize) {
    if values.is_empty() {
        return (0.0, 0.0, 0);
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut non_zero = 0usize;
    for value in values.iter().copied() {
        let abs = value.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        if abs > 0.0 {
            non_zero += 1;
        }
    }
    (max_abs, sum_abs / values.len() as f32, non_zero)
}

fn tensor_abs_stats(tensor: &Tensor) -> candle_core::Result<(f32, f32, usize)> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok(abs_stats(&values))
}

fn max_abs_delta(before: &[f32], after: &Tensor) -> candle_core::Result<f32> {
    let after_values = after.flatten_all()?.to_vec1::<f32>()?;
    let mut max_delta = 0.0f32;
    for (lhs, rhs) in before.iter().copied().zip(after_values.into_iter()) {
        max_delta = max_delta.max((lhs - rhs).abs());
    }
    Ok(max_delta)
}

#[derive(Debug, Clone, Copy)]
struct MetalMemoryEstimate {
    gaussian_state_bytes: u64,
    frame_bytes: u64,
    pixel_state_bytes: u64,
    projection_bytes: u64,
    chunk_workspace_bytes: u64,
    retained_graph_bytes: u64,
    safety_margin_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct MetalMemoryBudget {
    safe_bytes: u64,
    physical_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalMemoryDecision {
    Allow,
    Warn,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkCapacityDisposition {
    FitsBudget,
    NeedsSubdivisionOrDegradation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkCapacityEstimate {
    pub requested_initial_gaussians: usize,
    pub affordable_initial_gaussians: usize,
    pub frame_count: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub estimated_peak_bytes: u64,
    pub requested_budget_bytes: u64,
    pub effective_budget_bytes: u64,
    pub physical_memory_bytes: Option<u64>,
    pub disposition: ChunkCapacityDisposition,
    dominant_components: String,
    recommendations: Vec<String>,
}

impl ChunkCapacityEstimate {
    pub fn estimated_peak_gib(&self) -> f64 {
        bytes_to_gib(self.estimated_peak_bytes)
    }

    pub fn requested_budget_gib(&self) -> f64 {
        bytes_to_gib(self.requested_budget_bytes)
    }

    pub fn effective_budget_gib(&self) -> f64 {
        bytes_to_gib(self.effective_budget_bytes)
    }

    pub fn requires_subdivision_or_degradation(&self) -> bool {
        self.disposition == ChunkCapacityDisposition::NeedsSubdivisionOrDegradation
    }

    pub fn dominant_components(&self) -> &str {
        &self.dominant_components
    }

    pub fn recommendations(&self) -> &[String] {
        &self.recommendations
    }
}

impl MetalMemoryEstimate {
    fn subtotal_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.frame_bytes)
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
            .saturating_add(self.chunk_workspace_bytes)
            .saturating_add(self.retained_graph_bytes)
    }

    fn total_bytes(&self) -> u64 {
        self.subtotal_bytes()
            .saturating_add(self.safety_margin_bytes)
    }

    fn persistent_bytes(&self) -> u64 {
        self.gaussian_state_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_add(self.projection_bytes)
    }

    fn top_components_summary(&self, count: usize) -> String {
        let mut components = vec![
            ("graph", self.retained_graph_bytes),
            ("chunk", self.chunk_workspace_bytes),
            ("frames", self.frame_bytes),
            ("persistent", self.persistent_bytes()),
            ("safety", self.safety_margin_bytes),
        ];
        components.retain(|(_, bytes)| *bytes > 0);
        components.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1));
        components
            .into_iter()
            .take(count)
            .map(|(label, bytes)| format!("{label}≈{}", format_memory(bytes)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn recommendations(&self) -> Vec<&'static str> {
        let total = self.total_bytes().max(1);
        let mut recommendations = Vec::new();
        if self.retained_graph_bytes.saturating_mul(100) >= total.saturating_mul(45) {
            recommendations.push("lower --max-initial-gaussians or increase --sampling-step");
        }
        if self
            .frame_bytes
            .saturating_add(self.pixel_state_bytes)
            .saturating_mul(100)
            >= total.saturating_mul(20)
        {
            recommendations.push("lower --metal-render-scale");
        }
        if self.chunk_workspace_bytes.saturating_mul(100) >= total.saturating_mul(10) {
            recommendations.push("lower --metal-gaussian-chunk-size");
        }
        if recommendations.is_empty() {
            recommendations.push("lower --max-initial-gaussians or --metal-render-scale");
        }
        recommendations
    }
}

impl MetalMemoryBudget {
    fn describe(&self) -> String {
        match self.physical_bytes {
            Some(physical_bytes) => format!(
                "{:.1} GiB safe budget on {:.1} GiB system memory",
                bytes_to_gib(self.safe_bytes),
                bytes_to_gib(physical_bytes)
            ),
            None => format!(
                "{:.1} GiB safe budget (system memory unavailable)",
                bytes_to_gib(self.safe_bytes)
            ),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MetalRenderProfile {
    projection: Duration,
    sorting: Duration,
    rasterization: Duration,
    native_forward: Option<NativeParityProfile>,
    visible_gaussians: usize,
    total_gaussians: usize,
    active_tiles: usize,
    tile_gaussian_refs: usize,
    max_gaussians_per_tile: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct MetalStepProfile {
    projection: Duration,
    sorting: Duration,
    rasterization: Duration,
    native_forward: Option<NativeParityProfile>,
    loss: Duration,
    backward: Duration,
    optimizer: Duration,
    total: Duration,
    visible_gaussians: usize,
    total_gaussians: usize,
    active_tiles: usize,
    tile_gaussian_refs: usize,
    max_gaussians_per_tile: usize,
}

impl MetalStepProfile {
    fn from_render(render: MetalRenderProfile) -> Self {
        Self {
            projection: render.projection,
            sorting: render.sorting,
            rasterization: render.rasterization,
            native_forward: render.native_forward,
            visible_gaussians: render.visible_gaussians,
            total_gaussians: render.total_gaussians,
            active_tiles: render.active_tiles,
            tile_gaussian_refs: render.tile_gaussian_refs,
            max_gaussians_per_tile: render.max_gaussians_per_tile,
            ..Default::default()
        }
    }

    fn log(&self, iter: usize, max_iterations: usize) {
        let native_total_ms = self
            .native_forward
            .map(|profile| duration_ms(profile.total))
            .unwrap_or(0.0);
        let native_setup_ms = self
            .native_forward
            .map(|profile| duration_ms(profile.setup))
            .unwrap_or(0.0);
        let native_stage_ms = self
            .native_forward
            .map(|profile| duration_ms(profile.staging))
            .unwrap_or(0.0);
        let native_kernel_ms = self
            .native_forward
            .map(|profile| duration_ms(profile.kernel))
            .unwrap_or(0.0);
        let native_color_diff = self
            .native_forward
            .map(|profile| profile.color_max_abs)
            .unwrap_or(0.0);
        let native_depth_diff = self
            .native_forward
            .map(|profile| profile.depth_max_abs)
            .unwrap_or(0.0);
        let native_alpha_diff = self
            .native_forward
            .map(|profile| profile.alpha_max_abs)
            .unwrap_or(0.0);
        log::info!(
            "Metal profile iter {:5}/{:5} | visible={:5}/{:5} | tiles={:4} | tile_refs={:5} | max_tile={:4} | total={:.2}ms | project={:.2}ms | sort={:.2}ms | raster={:.2}ms | native={:.2}ms(setup={:.2} stage={:.2} kernel={:.2}) | diff(c={:.5} d={:.5} a={:.5}) | loss={:.2}ms | backward={:.2}ms | optimizer={:.2}ms",
            iter,
            max_iterations,
            self.visible_gaussians,
            self.total_gaussians,
            self.active_tiles,
            self.tile_gaussian_refs,
            self.max_gaussians_per_tile,
            duration_ms(self.total),
            duration_ms(self.projection),
            duration_ms(self.sorting),
            duration_ms(self.rasterization),
            native_total_ms,
            native_setup_ms,
            native_stage_ms,
            native_kernel_ms,
            native_color_diff,
            native_depth_diff,
            native_alpha_diff,
            duration_ms(self.loss),
            duration_ms(self.backward),
            duration_ms(self.optimizer),
        );
    }
}

impl MetalTrainer {
    fn is_litegs_mode(&self) -> bool {
        self.training_profile == TrainingProfile::LiteGsMacV1
    }

    fn current_learning_rates(&self) -> LiteGsOptimizerLrs {
        LiteGsOptimizerLrs {
            xyz: Some(self.compute_lr_pos()),
            sh_0: Some(self.lr_color),
            sh_rest: Some(self.lr_sh_rest),
            opacity: Some(self.lr_opacity),
            scale: Some(self.lr_scale),
            rot: Some(self.lr_rotation),
        }
    }

    fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        if frame_count == 0 {
            0
        } else {
            (self.max_iterations / frame_count).max(1)
        }
    }

    fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        if let Some(until) = self.litegs.densify_until {
            return until.max(self.litegs.densify_from.saturating_add(1));
        }

        let total_epochs = self.litegs_total_epochs(frame_count);
        let reset_interval = self.litegs.opacity_reset_interval.max(1);
        let scaled = ((total_epochs as f32) * 0.8).floor() as usize;
        let computed = (scaled / reset_interval) * reset_interval + 1;
        // Ensure densify_until > densify_from for valid active_window
        computed.max(self.litegs.densify_from.saturating_add(1))
    }

    fn litegs_late_stage_start_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        let densify_until = self.litegs_densify_until_epoch(frame_count);
        let derived = total_epochs.saturating_mul(2) / 3;
        derived
            .max(self.litegs.densify_from)
            .min(densify_until.saturating_sub(1))
    }

    fn refresh_litegs_topology_window_metrics(&mut self, frame_count: usize) {
        if !self.is_litegs_mode() {
            return;
        }

        self.topology_metrics.total_epochs = Some(self.litegs_total_epochs(frame_count));
        self.topology_metrics.densify_until_epoch =
            Some(self.litegs_densify_until_epoch(frame_count));
        self.topology_metrics.late_stage_start_epoch =
            Some(self.litegs_late_stage_start_epoch(frame_count));
        self.topology_metrics.topology_freeze_epoch = self.litegs.topology_freeze_after_epoch;
    }

    fn litegs_current_epoch(&self, frame_count: usize) -> Option<usize> {
        if frame_count == 0 || self.iteration == 0 {
            return None;
        }
        Some(self.iteration.saturating_sub(1) / frame_count)
    }

    fn is_litegs_late_stage_epoch(&self, epoch: usize, frame_count: usize) -> bool {
        self.is_litegs_mode() && epoch >= self.litegs_late_stage_start_epoch(frame_count)
    }

    fn litegs_active_sh_degree_for_iteration(&self, frame_count: usize) -> usize {
        if !self.is_litegs_mode() || self.litegs.sh_degree == 0 {
            return self.litegs.sh_degree;
        }
        let epoch = if frame_count == 0 {
            0
        } else {
            self.iteration.saturating_sub(1) / frame_count
        };
        (epoch / LITEGS_SH_ACTIVATION_EPOCH_INTERVAL).min(self.litegs.sh_degree)
    }

    fn litegs_clone_scale_threshold(&self) -> f32 {
        (self.scene_extent * LITEGS_PERCENT_DENSE).max(1e-4)
    }

    fn legacy_clone_scale_threshold(&self) -> f32 {
        self.legacy_clone_scale_threshold
    }

    fn reset_gaussian_stats(&mut self, gaussian_count: usize) {
        self.gaussian_stats = vec![MetalGaussianStats::default(); gaussian_count];
    }

    fn reset_litegs_refine_window_stats(&mut self) {
        if !self.is_litegs_mode() {
            return;
        }

        for stats in &mut self.gaussian_stats {
            stats.mean2d_grad = RunningMoments::default();
            stats.fragment_weight = RunningMoments::default();
            stats.fragment_err = RunningMoments::default();
            stats.visible_count = 0;
        }
    }

    fn record_loss_curve_sample(&mut self, iter: usize, frame_idx: usize, max_iterations: usize) {
        if !should_record_loss_curve_sample(iter, max_iterations) {
            return;
        }
        if self
            .loss_curve_samples
            .last()
            .map(|sample| sample.iteration == iter)
            .unwrap_or(false)
        {
            return;
        }

        self.loss_curve_samples.push(ParityLossCurveSample {
            iteration: iter,
            frame_idx,
            l1: self.last_loss_terms.l1,
            ssim: self.last_loss_terms.ssim,
            depth: self.last_loss_terms.depth,
            total: self.last_loss_terms.total,
            depth_valid_pixels: self.last_depth_valid_pixels,
        });
    }

    fn current_telemetry(&self, frame_count: usize) -> LiteGsTrainingTelemetry {
        let final_metrics = if self.loss_history.is_empty() {
            None
        } else {
            Some(summarize_training_metrics(&self.loss_history, frame_count))
        };
        LiteGsTrainingTelemetry {
            loss_terms: self.last_loss_terms.clone(),
            loss_curve_samples: self.loss_curve_samples.clone(),
            topology: self.topology_metrics.clone(),
            active_sh_degree: Some(self.active_sh_degree),
            final_loss: final_metrics.map(|metrics| metrics.final_loss),
            final_step_loss: final_metrics.map(|metrics| metrics.final_step_loss),
            depth_valid_pixels: self.last_depth_valid_pixels,
            depth_grad_scale: self.last_depth_grad_scale,
            rotation_frozen: self.rotation_frozen,
            learning_rates: self.last_learning_rates.clone(),
        }
    }

    fn empty_projected_gaussians(&self) -> candle_core::Result<ProjectedGaussians> {
        Ok(ProjectedGaussians {
            source_indices: Tensor::zeros((0,), DType::U32, &self.device)?,
            u: Tensor::zeros((0,), DType::F32, &self.device)?,
            v: Tensor::zeros((0,), DType::F32, &self.device)?,
            sigma_x: Tensor::zeros((0,), DType::F32, &self.device)?,
            sigma_y: Tensor::zeros((0,), DType::F32, &self.device)?,
            raw_sigma_x: Tensor::zeros((0,), DType::F32, &self.device)?,
            raw_sigma_y: Tensor::zeros((0,), DType::F32, &self.device)?,
            depth: Tensor::zeros((0,), DType::F32, &self.device)?,
            opacity: Tensor::zeros((0,), DType::F32, &self.device)?,
            opacity_logits: Tensor::zeros((0,), DType::F32, &self.device)?,
            scale3d: Tensor::zeros((0, 3), DType::F32, &self.device)?,
            colors: Tensor::zeros((0, 3), DType::F32, &self.device)?,
            min_x: Tensor::zeros((0,), DType::F32, &self.device)?,
            max_x: Tensor::zeros((0,), DType::F32, &self.device)?,
            min_y: Tensor::zeros((0,), DType::F32, &self.device)?,
            max_y: Tensor::zeros((0,), DType::F32, &self.device)?,
            visible_source_indices: Vec::new(),
            visible_count: 0,
            tile_bins: MetalTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        })
    }

    fn export_snapshot(
        &self,
        gaussians: &TrainableGaussians,
    ) -> candle_core::Result<GaussianParameterSnapshot> {
        Ok(GaussianParameterSnapshot {
            positions: flatten_rows(gaussians.positions().to_vec2::<f32>()?),
            log_scales: flatten_rows(gaussians.scales.as_tensor().to_vec2::<f32>()?),
            rotations: flatten_rows(gaussians.rotations.as_tensor().to_vec2::<f32>()?),
            opacity_logits: gaussians.opacities.as_tensor().to_vec1::<f32>()?,
            colors: flatten_rows(gaussians.colors().to_vec2::<f32>()?),
            sh_rest: flatten_3d(gaussians.sh_rest().to_vec3::<f32>()?),
            color_representation: gaussians.color_representation(),
        })
    }

    fn update_gaussian_stats(
        &mut self,
        param_grad_magnitudes: &[f32],
        projected_grad_magnitudes: &[f32],
        projected: &ProjectedGaussians,
        gaussian_count: usize,
    ) -> candle_core::Result<()> {
        if self.gaussian_stats.len() != gaussian_count {
            self.gaussian_stats
                .resize(gaussian_count, MetalGaussianStats::default());
        }

        for stats in &mut self.gaussian_stats {
            stats.age = stats.age.saturating_add(1);
            stats.consecutive_invisible_epochs =
                stats.consecutive_invisible_epochs.saturating_add(1);
        }

        if !self.is_litegs_mode() {
            for idx in 0..gaussian_count.min(param_grad_magnitudes.len()) {
                let grad_mag = param_grad_magnitudes[idx] * self.pixel_count.max(1) as f32;
                let stats = &mut self.gaussian_stats[idx];
                stats.mean2d_grad.update(grad_mag.min(10.0));
            }
            for source_idx in projected.visible_source_indices().iter().copied() {
                if let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) {
                    stats.visible_count = stats.visible_count.saturating_add(1);
                    stats.consecutive_invisible_epochs = 0; // Reset on visibility
                }
            }
            return Ok(());
        }

        let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
        let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
        let opacity = projected.opacity.to_vec1::<f32>()?;

        for (visible_idx, source_idx) in projected
            .visible_source_indices()
            .iter()
            .copied()
            .enumerate()
        {
            let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) else {
                continue;
            };
            // Reset consecutive invisibility when visible
            stats.consecutive_invisible_epochs = 0;

            let grad_mag = projected_grad_magnitudes
                .get(source_idx as usize)
                .copied()
                .unwrap_or_default()
                .max(0.0)
                * self.pixel_count.max(1) as f32;
            let sigma_x = sigma_x
                .get(visible_idx)
                .copied()
                .unwrap_or(0.0)
                .abs()
                .max(1e-4);
            let sigma_y = sigma_y
                .get(visible_idx)
                .copied()
                .unwrap_or(0.0)
                .abs()
                .max(1e-4);
            let opacity = opacity.get(visible_idx).copied().unwrap_or(0.0).max(0.0);
            let fragment_weight = opacity * sigma_x * sigma_y;
            let fragment_err = grad_mag * fragment_weight;

            stats.mean2d_grad.update(grad_mag);
            stats.fragment_weight.update(fragment_weight);
            stats.fragment_err.update(fragment_err);
            stats.visible_count = stats.visible_count.saturating_add(1);
        }

        if debug_training_step_probe_enabled()
            && (self.iteration <= 6 || self.iteration % self.litegs.refine_every.max(1) == 0)
        {
            let mut visible_raw_max = 0.0f32;
            let mut visible_scaled_max = 0.0f32;
            let mut visible_nonzero = 0usize;
            for source_idx in projected.visible_source_indices().iter().copied() {
                let raw = projected_grad_magnitudes
                    .get(source_idx as usize)
                    .copied()
                    .unwrap_or_default()
                    .max(0.0);
                if raw > 0.0 {
                    visible_nonzero += 1;
                    visible_raw_max = visible_raw_max.max(raw);
                    visible_scaled_max =
                        visible_scaled_max.max(raw * self.pixel_count.max(1) as f32);
                }
            }
            let (all_raw_max, all_raw_mean, all_raw_nonzero) = abs_stats(projected_grad_magnitudes);
            log::info!(
                "Growth stats step {} | visible={} | visible_nonzero={} | visible_raw_max={:.6e} | visible_scaled_max={:.6e} | all_raw_max={:.6e} | all_raw_mean={:.6e} | all_raw_nonzero={}",
                self.iteration,
                projected.visible_source_indices().len(),
                visible_nonzero,
                visible_raw_max,
                visible_scaled_max,
                all_raw_max,
                all_raw_mean,
                all_raw_nonzero,
            );
        }

        Ok(())
    }

    fn reset_opacity_state(&mut self, only_opacity: bool) -> candle_core::Result<()> {
        let Some(adam) = self.adam.as_mut() else {
            return Ok(());
        };

        adam.m_op = Tensor::zeros_like(&adam.m_op)?;
        adam.v_op = Tensor::zeros_like(&adam.v_op)?;
        if only_opacity {
            return Ok(());
        }

        adam.m_pos = Tensor::zeros_like(&adam.m_pos)?;
        adam.v_pos = Tensor::zeros_like(&adam.v_pos)?;
        adam.m_scale = Tensor::zeros_like(&adam.m_scale)?;
        adam.v_scale = Tensor::zeros_like(&adam.v_scale)?;
        adam.m_rot = Tensor::zeros_like(&adam.m_rot)?;
        adam.v_rot = Tensor::zeros_like(&adam.v_rot)?;
        adam.m_color = Tensor::zeros_like(&adam.m_color)?;
        adam.v_color = Tensor::zeros_like(&adam.v_color)?;
        adam.m_sh_rest = Tensor::zeros_like(&adam.m_sh_rest)?;
        adam.v_sh_rest = Tensor::zeros_like(&adam.v_sh_rest)?;
        Ok(())
    }

    fn apply_litegs_opacity_reset(
        &mut self,
        gaussians: &mut TrainableGaussians,
        event_epoch: Option<usize>,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        let opacities = gaussians.opacities()?;
        let new_opacity_values = match self.litegs.opacity_reset_mode {
            super::LiteGsOpacityResetMode::Decay => inverse_sigmoid_tensor(
                &opacities
                    .affine(LITEGS_OPACITY_DECAY_RATE as f64, 0.0)?
                    .clamp(LITEGS_OPACITY_DECAY_MIN, 1.0 - 1e-6)?,
            )?,
            super::LiteGsOpacityResetMode::Reset => {
                inverse_sigmoid_tensor(&opacities.clamp(1e-6, LITEGS_OPACITY_THRESHOLD)?)?
            }
        };
        gaussians.opacities.set(&new_opacity_values)?;
        self.reset_opacity_state(matches!(
            self.litegs.opacity_reset_mode,
            super::LiteGsOpacityResetMode::Reset
        ))?;
        self.topology_metrics.opacity_reset_events =
            self.topology_metrics.opacity_reset_events.saturating_add(1);
        record_topology_epoch(
            &mut self.topology_metrics.first_opacity_reset_epoch,
            &mut self.topology_metrics.last_opacity_reset_epoch,
            event_epoch,
        );
        if let Some(epoch) = event_epoch {
            if self.is_litegs_late_stage_epoch(epoch, frame_count) {
                self.topology_metrics.late_stage_opacity_reset_events = self
                    .topology_metrics
                    .late_stage_opacity_reset_events
                    .saturating_add(1);
            }
        }
        Ok(())
    }

    fn litegs_should_prune_candidate(&self, info: &TopologyCandidateInfo) -> bool {
        let prune_opacity = match self.litegs.prune_mode {
            super::LiteGsPruneMode::Weight => info.age_eligible && info.visible_count == 0,
            super::LiteGsPruneMode::Threshold => {
                (info.age_eligible && info.opacity < LITEGS_OPACITY_THRESHOLD) || info.invisible
            }
        };

        let prune_scale = self.litegs.prune_scale_threshold > 0.0
            && info.max_scale > self.litegs.prune_scale_threshold;

        prune_opacity || prune_scale
    }

    fn litegs_requested_additions(
        &self,
        infos: &[TopologyCandidateInfo],
        allow_extra_growth: bool,
    ) -> usize {
        let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
        if !allow_extra_growth {
            return prune_candidates;
        }

        let threshold_count = infos
            .iter()
            .filter(|info| !info.prune_candidate && info.growth_candidate)
            .count();
        let grow_count =
            (threshold_count as f32 * self.litegs.growth_select_fraction).round() as usize;

        prune_candidates.saturating_add(grow_count.saturating_sub(prune_candidates))
    }

    fn litegs_select_densify_candidates(
        &self,
        infos: &[TopologyCandidateInfo],
        max_new: usize,
        allow_extra_growth: bool,
    ) -> LiteGsDensifySelection {
        if max_new == 0 || infos.is_empty() {
            return LiteGsDensifySelection::default();
        }

        let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
        let mut replacement_sources: Vec<(usize, f32)> = infos
            .iter()
            .enumerate()
            .filter_map(|(idx, info)| {
                (!info.prune_candidate
                    && info.visible_count > 0
                    && info.opacity.is_finite()
                    && info.opacity > LITEGS_OPACITY_THRESHOLD)
                    .then_some((idx, info.opacity))
            })
            .collect();
        replacement_sources
            .sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

        let replacement_count = prune_candidates.min(max_new);
        let mut selection = LiteGsDensifySelection {
            selected_indices: Vec::with_capacity(max_new.min(infos.len())),
            replacement_count: 0,
            extra_growth_count: 0,
        };
        let mut used_sources = vec![false; infos.len()];

        if !replacement_sources.is_empty() && replacement_count > 0 {
            for offset in 0..replacement_count {
                let source_idx = replacement_sources[offset % replacement_sources.len()].0;
                selection.selected_indices.push(source_idx);
                selection.replacement_count += 1;
                used_sources[source_idx] = true;
            }
        }

        if allow_extra_growth && selection.selected_indices.len() < max_new {
            let threshold_count = infos
                .iter()
                .filter(|info| !info.prune_candidate && info.growth_candidate)
                .count();
            let grow_count =
                (threshold_count as f32 * self.litegs.growth_select_fraction).round() as usize;
            let extra_growth_limit = grow_count
                .saturating_sub(selection.replacement_count)
                .min(max_new.saturating_sub(selection.selected_indices.len()));

            if extra_growth_limit > 0 {
                let mut growth_sources: Vec<(usize, f32, f32)> = infos
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, info)| {
                        (!info.prune_candidate && info.growth_candidate && !used_sources[idx])
                            .then_some((idx, info.mean2d_grad, info.opacity))
                    })
                    .collect();
                growth_sources.sort_by(|lhs, rhs| {
                    rhs.1
                        .partial_cmp(&lhs.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| rhs.2.partial_cmp(&lhs.2).unwrap_or(Ordering::Equal))
                });

                for (source_idx, _, _) in growth_sources.into_iter().take(extra_growth_limit) {
                    selection.selected_indices.push(source_idx);
                    selection.extra_growth_count += 1;
                }
            }
        }

        selection
    }

    fn max_topology_gaussians(
        &self,
        requested_cap: usize,
        current_len: usize,
        frame_count: usize,
    ) -> usize {
        let min_cap = current_len.max(1);
        let requested_cap = requested_cap.max(min_cap);
        let Some(memory_budget) = self.topology_memory_budget else {
            return requested_cap;
        };
        if assess_memory_estimate(
            &estimate_peak_memory_with_source_pixels(
                requested_cap,
                self.pixel_count,
                self.source_pixel_count,
                frame_count,
                self.chunk_size,
            ),
            &memory_budget,
        ) != MetalMemoryDecision::Block
        {
            return requested_cap;
        }

        let mut low = min_cap;
        let mut high = requested_cap;
        while low < high {
            let mid = low + (high - low + 1) / 2;
            let decision = assess_memory_estimate(
                &estimate_peak_memory_with_source_pixels(
                    mid,
                    self.pixel_count,
                    self.source_pixel_count,
                    frame_count,
                    self.chunk_size,
                ),
                &memory_budget,
            );
            if decision == MetalMemoryDecision::Block {
                high = mid - 1;
            } else {
                low = mid;
            }
        }
        low
    }

    fn analyze_topology_candidates(
        &self,
        gaussians: &TrainableGaussians,
        stats: &[MetalGaussianStats],
    ) -> candle_core::Result<TopologyAnalysis> {
        let log_scales = gaussians.scales.as_tensor().to_vec2::<f32>()?;
        let opacity_logits = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let mut analysis = TopologyAnalysis {
            infos: Vec::with_capacity(opacity_logits.len()),
            ..TopologyAnalysis::default()
        };
        let mut grad_sum = 0.0f32;
        let clone_scale_threshold = if self.is_litegs_mode() {
            self.litegs_clone_scale_threshold()
        } else {
            self.legacy_clone_scale_threshold()
        };

        for idx in 0..opacity_logits.len() {
            let scale_row = log_scales.get(idx);
            let sx = scale_row
                .and_then(|row| row.first())
                .copied()
                .unwrap_or(0.0)
                .exp();
            let sy = scale_row
                .and_then(|row| row.get(1))
                .copied()
                .unwrap_or(0.0)
                .exp();
            let sz = scale_row
                .and_then(|row| row.get(2))
                .copied()
                .unwrap_or(0.0)
                .exp();
            let max_scale = sx.max(sy).max(sz);
            let opacity = sigmoid_scalar(opacity_logits[idx]);
            let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
            let mean2d_grad = gaussian_stats.mean2d_grad.mean;
            let fragment_weight = gaussian_stats.fragment_weight.mean;
            let fragment_err_score = if gaussian_stats.fragment_err.count <= 1 {
                mean2d_grad * opacity * opacity
            } else {
                gaussian_stats.fragment_err.variance()
                    * gaussian_stats.fragment_err.count as f32
                    * opacity
                    * opacity
            };

            // LiteGS mode: use age-aware and consecutive invisibility pruning
            let (invisible_for_prune, age_eligible) = if self.is_litegs_mode() {
                let min_age = self.litegs.prune_min_age.max(1);
                let min_invisible = self.litegs.prune_invisible_epochs.max(1);
                let age_ok = gaussian_stats.age >= min_age;
                let invisible_long_enough =
                    gaussian_stats.consecutive_invisible_epochs >= min_invisible;
                (age_ok && invisible_long_enough, age_ok)
            } else {
                // Legacy mode: simple invisibility check
                let invisible = gaussian_stats.visible_count == 0;
                (invisible, true)
            };

            let growth_threshold = if self.is_litegs_mode() {
                self.litegs.growth_grad_threshold
            } else {
                self.legacy_densify_grad_threshold
            };
            let growth_candidate = mean2d_grad.is_finite()
                && mean2d_grad >= growth_threshold
                && opacity > LITEGS_OPACITY_THRESHOLD;
            let candidate_info = TopologyCandidateInfo {
                max_scale,
                opacity,
                mean2d_grad,
                fragment_weight,
                fragment_err_score,
                visible_count: gaussian_stats.visible_count,
                age_eligible,
                invisible: invisible_for_prune,
                prune_candidate: false,
                growth_candidate,
            };
            let prune_candidate = if self.is_litegs_mode() {
                self.litegs_should_prune_candidate(&candidate_info)
            } else {
                false
            };

            analysis.infos.push(TopologyCandidateInfo {
                prune_candidate,
                ..candidate_info
            });
            if mean2d_grad > growth_threshold {
                analysis.active_grad_stats += 1;
            }
            if max_scale <= clone_scale_threshold {
                analysis.small_scale_stats += 1;
            }
            if opacity > LITEGS_OPACITY_THRESHOLD {
                analysis.opacity_ready_stats += 1;
            }
            if mean2d_grad.is_finite() {
                analysis.max_grad = analysis.max_grad.max(mean2d_grad);
                grad_sum += mean2d_grad;
            }
            if self.is_litegs_mode() {
                if growth_candidate {
                    analysis.growth_candidates += 1;
                    if max_scale <= clone_scale_threshold {
                        analysis.clone_candidates += 1;
                    } else {
                        analysis.split_candidates += 1;
                    }
                }
                if prune_candidate {
                    analysis.prune_candidates += 1;
                }
            } else {
                if mean2d_grad > self.legacy_densify_grad_threshold
                    && opacity > self.prune_threshold
                {
                    if max_scale < self.legacy_clone_scale_threshold {
                        analysis.clone_candidates += 1;
                    }
                    if max_scale > self.legacy_split_scale_threshold {
                        analysis.split_candidates += 1;
                    }
                }
                if opacity < self.prune_threshold || max_scale > self.legacy_prune_scale_threshold {
                    analysis.prune_candidates += 1;
                }
            }
        }

        if !analysis.infos.is_empty() {
            analysis.mean_grad = grad_sum / analysis.infos.len() as f32;
        }
        Ok(analysis)
    }

    fn densify_snapshot(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
        max_gaussians: usize,
    ) -> usize {
        if snapshot.len() >= max_gaussians {
            return 0;
        }

        let clone_opacity_threshold = self.prune_threshold;
        let original_len = snapshot.len();
        let mut added = 0usize;
        let mut clone_candidates = Vec::new();
        let mut split_candidates = Vec::new();

        for idx in 0..original_len {
            let info = infos.get(idx).copied().unwrap_or(TopologyCandidateInfo {
                max_scale: 0.0,
                opacity: 0.0,
                mean2d_grad: 0.0,
                fragment_weight: 0.0,
                fragment_err_score: 0.0,
                visible_count: 0,
                age_eligible: false,
                invisible: true,
                prune_candidate: false,
                growth_candidate: false,
            });
            let opacity = info.opacity;
            let max_scale = info.max_scale;
            let grad_accum = info.mean2d_grad;
            if !grad_accum.is_finite() || !opacity.is_finite() {
                continue;
            }
            if grad_accum <= self.legacy_densify_grad_threshold {
                continue;
            }
            if max_scale < self.legacy_clone_scale_threshold && opacity > clone_opacity_threshold {
                clone_candidates.push((idx, grad_accum));
            }
            if max_scale > self.legacy_split_scale_threshold && opacity > self.prune_threshold {
                split_candidates.push((idx, grad_accum * max_scale));
            }
        }

        clone_candidates.sort_by(|lhs, rhs| {
            rhs.1
                .partial_cmp(&lhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        split_candidates.sort_by(|lhs, rhs| {
            rhs.1
                .partial_cmp(&lhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut available = max_gaussians.saturating_sub(snapshot.len());
        let per_pass_limit = self
            .legacy_max_densify_per_update
            .min(available)
            .min((original_len / 32).max(32));
        let clone_limit = clone_candidates.len().min(per_pass_limit);
        for (rank, (idx, score)) in clone_candidates.into_iter().take(clone_limit).enumerate() {
            if score <= 0.0 {
                continue;
            }
            let position = snapshot.position(idx);
            let scale = snapshot.scale(idx);
            let log_scale = snapshot.log_scale(idx);
            let rotation = snapshot.rotation(idx);
            let color = snapshot.color(idx);
            let sh_rest = snapshot.sh_rest(idx).to_vec();
            let opacity_logit = snapshot.opacity_logits[idx];
            let axis = rank % 3;
            let mut cloned_position = position;
            cloned_position[axis] += scale[axis].max(0.01) * 0.5;
            snapshot.push(
                cloned_position,
                log_scale,
                rotation,
                opacity_logit,
                color,
                &sh_rest,
            );
            stats.push(MetalGaussianStats::default());
            origins.push(None);
            added += 1;
            available = available.saturating_sub(1);
            if available == 0 {
                return added;
            }
        }

        let split_limit = split_candidates
            .len()
            .min((per_pass_limit / 4).max(1))
            .min(available / 2);
        for (idx, score) in split_candidates.into_iter().take(split_limit) {
            if score <= self.legacy_densify_grad_threshold {
                continue;
            }
            let position = snapshot.position(idx);
            let max_scale = snapshot.scale(idx).into_iter().fold(0.0f32, f32::max);
            let mut split_scale = snapshot.log_scale(idx);
            split_scale[0] = (max_scale * 0.5).max(1e-6).ln();
            let rotation = snapshot.rotation(idx);
            let color = snapshot.color(idx);
            let sh_rest = snapshot.sh_rest(idx).to_vec();
            let opacity_logit = snapshot.opacity_logits[idx];
            for direction in [1.0f32, -1.0] {
                if available == 0 {
                    break;
                }
                let mut split_position = position;
                split_position[0] += direction * max_scale * 0.1;
                snapshot.push(
                    split_position,
                    split_scale,
                    rotation,
                    opacity_logit,
                    color,
                    &sh_rest,
                );
                stats.push(MetalGaussianStats::default());
                origins.push(None);
                added += 1;
                available = available.saturating_sub(1);
            }
        }

        added
    }

    fn prune_snapshot(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
    ) -> usize {
        if snapshot.len() <= 1 {
            return 0;
        }

        let mut keep_mask = vec![false; snapshot.len()];
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for idx in 0..snapshot.len() {
            let position = snapshot.position(idx);
            let rotation = snapshot.rotation(idx);
            let color = snapshot.color(idx);
            let info = infos.get(idx).copied().unwrap_or_else(|| {
                let scale = snapshot.scale(idx);
                TopologyCandidateInfo {
                    max_scale: scale[0].max(scale[1]).max(scale[2]),
                    opacity: sigmoid_scalar(snapshot.opacity_logits[idx]),
                    mean2d_grad: stats.get(idx).copied().unwrap_or_default().mean2d_grad.mean,
                    fragment_weight: 0.0,
                    fragment_err_score: 0.0,
                    visible_count: 0,
                    age_eligible: false,
                    invisible: true,
                    prune_candidate: false,
                    growth_candidate: false,
                }
            });
            let opacity = info.opacity;
            let max_scale = info.max_scale;
            let valid = opacity.is_finite()
                && opacity >= self.prune_threshold
                && max_scale.is_finite()
                && max_scale <= self.legacy_prune_scale_threshold
                && position.iter().all(|value| value.is_finite())
                && rotation.iter().all(|value| value.is_finite())
                && color.iter().all(|value| value.is_finite());
            if valid {
                keep_mask[idx] = true;
            }
            let score = if opacity.is_finite() {
                opacity
            } else {
                f32::NEG_INFINITY
            };
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        if !keep_mask.iter().any(|keep| *keep) {
            keep_mask[best_idx] = true;
        }

        let pruned = keep_mask.iter().filter(|keep| !**keep).count();
        if pruned == 0 {
            return 0;
        }

        let mut kept_snapshot = GaussianParameterSnapshot {
            positions: Vec::with_capacity((snapshot.len() - pruned) * 3),
            log_scales: Vec::with_capacity((snapshot.len() - pruned) * 3),
            rotations: Vec::with_capacity((snapshot.len() - pruned) * 4),
            opacity_logits: Vec::with_capacity(snapshot.len() - pruned),
            colors: Vec::with_capacity((snapshot.len() - pruned) * 3),
            sh_rest: Vec::with_capacity((snapshot.len() - pruned) * snapshot.sh_rest_row_width()),
            color_representation: snapshot.color_representation,
        };
        let mut kept_stats = Vec::with_capacity(snapshot.len() - pruned);
        let mut kept_origins = Vec::with_capacity(snapshot.len() - pruned);

        for idx in 0..snapshot.len() {
            if keep_mask[idx] {
                kept_snapshot.push(
                    snapshot.position(idx),
                    snapshot.log_scale(idx),
                    snapshot.rotation(idx),
                    snapshot.opacity_logits[idx],
                    snapshot.color(idx),
                    snapshot.sh_rest(idx),
                );
                kept_stats.push(stats[idx]);
                kept_origins.push(origins[idx]);
            }
        }

        *snapshot = kept_snapshot;
        *stats = kept_stats;
        *origins = kept_origins;
        pruned
    }

    fn densify_snapshot_litegs(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        max_gaussians: usize,
        selected_indices: &[usize],
    ) -> usize {
        if snapshot.len() >= max_gaussians || selected_indices.is_empty() {
            return 0;
        }

        let clone_scale_threshold = self.litegs_clone_scale_threshold();

        // Separate into clone and split candidates (LiteGS semantics)
        let mut clone_indices = Vec::new();
        let mut split_indices = Vec::new();

        for &idx in selected_indices
            .iter()
            .take(max_gaussians.saturating_sub(snapshot.len()))
        {
            let scale = snapshot.scale(idx);
            let max_scale = scale.into_iter().fold(0.0f32, f32::max);
            if max_scale <= clone_scale_threshold {
                clone_indices.push(idx);
            } else {
                split_indices.push(idx);
            }
        }

        let mut added = 0usize;

        // Process split candidates (LiteGS: sample offset from Gaussian distribution)
        // split_xyz = xyz + random_shift ~ N(0, scale) @ rotation_matrix
        // split_scale = scale / (0.8 * 2) = scale / 1.6
        for idx in &split_indices {
            if snapshot.len() >= max_gaussians {
                break;
            }

            let mut position = snapshot.position(*idx);
            let mut log_scale = snapshot.log_scale(*idx);
            let rotation = snapshot.rotation(*idx);
            let opacity_logit = snapshot.opacity_logits[*idx];
            let color = snapshot.color(*idx);
            let sh_rest = snapshot.sh_rest(*idx).to_vec();
            let scale = snapshot.scale(*idx);

            // LiteGS: find the axis with maximum scale
            let (max_axis, max_axis_scale) = scale
                .into_iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .unwrap_or((0, 0.0f32));

            // LiteGS: position offset along the max scale axis
            // This approximates the random sampling from N(0, scale) @ rotation
            // by placing the new Gaussian at +offset along the dominant axis
            position[max_axis] += max_axis_scale * 0.5;

            // LiteGS: split_scale = scale / (0.8 * 2) = scale / 1.6
            // log_scale = ln(scale / 1.6) = ln(scale) - ln(1.6)
            log_scale[max_axis] = (max_axis_scale / 1.6).max(1e-6).ln();

            snapshot.push(
                position,
                log_scale,
                rotation,
                opacity_logit,
                color,
                &sh_rest,
            );
            stats.push(MetalGaussianStats::default());
            origins.push(None);
            added = added.saturating_add(1);
        }

        // Process clone candidates (LiteGS: copy directly)
        for idx in &clone_indices {
            if snapshot.len() >= max_gaussians {
                break;
            }

            let position = snapshot.position(*idx);
            let log_scale = snapshot.log_scale(*idx);
            let rotation = snapshot.rotation(*idx);
            let opacity_logit = snapshot.opacity_logits[*idx];
            let color = snapshot.color(*idx);
            let sh_rest = snapshot.sh_rest(*idx).to_vec();

            snapshot.push(
                position,
                log_scale,
                rotation,
                opacity_logit,
                color,
                &sh_rest,
            );
            stats.push(MetalGaussianStats::default());
            origins.push(None);
            added = added.saturating_add(1);
        }

        added
    }

    fn prune_snapshot_litegs(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
    ) -> usize {
        if snapshot.len() <= 1 {
            return 0;
        }

        let mut keep_mask = vec![true; snapshot.len()];
        for (idx, info) in infos.iter().enumerate() {
            if info.prune_candidate {
                keep_mask[idx] = false;
            }
        }

        // Safety: keep at least one Gaussian
        if !keep_mask.iter().any(|keep| *keep) {
            if let Some((best_idx, _)) = infos.iter().enumerate().max_by(|lhs, rhs| {
                lhs.1
                    .opacity
                    .partial_cmp(&rhs.1.opacity)
                    .unwrap_or(Ordering::Equal)
            }) {
                keep_mask[best_idx] = true;
            }
        }

        let pruned = keep_mask.iter().filter(|keep| !**keep).count();
        if pruned == 0 {
            return 0;
        }

        let mut kept_snapshot = GaussianParameterSnapshot {
            positions: Vec::with_capacity((snapshot.len() - pruned) * 3),
            log_scales: Vec::with_capacity((snapshot.len() - pruned) * 3),
            rotations: Vec::with_capacity((snapshot.len() - pruned) * 4),
            opacity_logits: Vec::with_capacity(snapshot.len() - pruned),
            colors: Vec::with_capacity((snapshot.len() - pruned) * 3),
            sh_rest: Vec::with_capacity((snapshot.len() - pruned) * snapshot.sh_rest_row_width()),
            color_representation: snapshot.color_representation,
        };
        let mut kept_stats = Vec::with_capacity(snapshot.len() - pruned);
        let mut kept_origins = Vec::with_capacity(snapshot.len() - pruned);

        for idx in 0..snapshot.len() {
            if keep_mask[idx] {
                kept_snapshot.push(
                    snapshot.position(idx),
                    snapshot.log_scale(idx),
                    snapshot.rotation(idx),
                    snapshot.opacity_logits[idx],
                    snapshot.color(idx),
                    snapshot.sh_rest(idx),
                );
                kept_stats.push(stats[idx]);
                kept_origins.push(origins[idx]);
            }
        }

        *snapshot = kept_snapshot;
        *stats = kept_stats;
        *origins = kept_origins;
        pruned
    }

    fn rebuild_adam_state(
        &self,
        old_state: &MetalAdamState,
        origins: &[Option<usize>],
    ) -> candle_core::Result<MetalAdamState> {
        let row_count = origins.len();
        let m_pos = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.m_pos.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            &self.device,
        )?;
        let v_pos = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.v_pos.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            &self.device,
        )?;
        let m_scale = Tensor::from_slice(
            &gather_rows(
                &flatten_rows(old_state.m_scale.to_vec2::<f32>()?),
                3,
                origins,
            ),
            (row_count, 3),
            &self.device,
        )?;
        let v_scale = Tensor::from_slice(
            &gather_rows(
                &flatten_rows(old_state.v_scale.to_vec2::<f32>()?),
                3,
                origins,
            ),
            (row_count, 3),
            &self.device,
        )?;
        let m_rot = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.m_rot.to_vec2::<f32>()?), 4, origins),
            (row_count, 4),
            &self.device,
        )?;
        let v_rot = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.v_rot.to_vec2::<f32>()?), 4, origins),
            (row_count, 4),
            &self.device,
        )?;
        let m_op = Tensor::from_slice(
            &gather_rows(&old_state.m_op.to_vec1::<f32>()?, 1, origins),
            row_count,
            &self.device,
        )?;
        let v_op = Tensor::from_slice(
            &gather_rows(&old_state.v_op.to_vec1::<f32>()?, 1, origins),
            row_count,
            &self.device,
        )?;
        let m_color = Tensor::from_slice(
            &gather_rows(
                &flatten_rows(old_state.m_color.to_vec2::<f32>()?),
                3,
                origins,
            ),
            (row_count, 3),
            &self.device,
        )?;
        let v_color = Tensor::from_slice(
            &gather_rows(
                &flatten_rows(old_state.v_color.to_vec2::<f32>()?),
                3,
                origins,
            ),
            (row_count, 3),
            &self.device,
        )?;
        let sh_rest_dims = old_state.m_sh_rest.dims();
        let sh_rest_coeff_count = sh_rest_dims.get(1).copied().unwrap_or(0);
        let sh_rest_shape = (row_count, sh_rest_coeff_count, 3usize);
        let m_sh_rest = Tensor::from_slice(
            &gather_rows(
                &flatten_3d(old_state.m_sh_rest.to_vec3::<f32>()?),
                sh_rest_coeff_count.saturating_mul(3),
                origins,
            ),
            sh_rest_shape,
            &self.device,
        )?;
        let v_sh_rest = Tensor::from_slice(
            &gather_rows(
                &flatten_3d(old_state.v_sh_rest.to_vec3::<f32>()?),
                sh_rest_coeff_count.saturating_mul(3),
                origins,
            ),
            sh_rest_shape,
            &self.device,
        )?;

        Ok(MetalAdamState {
            m_pos,
            v_pos,
            m_scale,
            v_scale,
            m_rot,
            v_rot,
            m_op,
            v_op,
            m_color,
            v_color,
            m_sh_rest,
            v_sh_rest,
        })
    }

    fn should_densify_at(&self, iteration: usize) -> bool {
        self.densify_interval > 0
            && iteration > self.topology_warmup
            && iteration % self.densify_interval == 0
    }

    fn should_prune_at(&self, iteration: usize) -> bool {
        self.prune_interval > 0
            && iteration > self.topology_warmup
            && iteration % self.prune_interval == 0
    }

    fn should_log_topology_at(&self, iteration: usize) -> bool {
        iteration % self.topology_log_interval == 0
    }

    fn clustering_positions(
        &self,
        gaussians: &TrainableGaussians,
    ) -> candle_core::Result<Vec<[f32; 3]>> {
        gaussians
            .positions()
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| {
                if row.len() != 3 {
                    candle_core::bail!("expected gaussian positions with row width 3");
                }
                Ok([row[0], row[1], row[2]])
            })
            .collect()
    }

    fn sync_cluster_assignment(
        &mut self,
        gaussians: &TrainableGaussians,
        topology_changed: bool,
    ) -> candle_core::Result<()> {
        if self.litegs.cluster_size == 0 {
            self.cluster_assignment = None;
            return Ok(());
        }

        let positions = self.clustering_positions(gaussians)?;
        if positions.is_empty() {
            self.cluster_assignment = None;
            return Ok(());
        }

        match self.cluster_assignment.as_mut() {
            Some(assignment)
                if !topology_changed && assignment.cluster_indices.len() == positions.len() =>
            {
                assignment.update_aabbs(&positions);
            }
            Some(assignment) => {
                assignment.reassign(&positions, self.litegs.cluster_size, self.scene_extent);
            }
            None => {
                self.cluster_assignment = Some(ClusterAssignment::assign_spatial_hash(
                    &positions,
                    self.litegs.cluster_size,
                    self.scene_extent,
                ));
            }
        }

        Ok(())
    }

    fn cluster_visible_mask_for_camera(
        &self,
        gaussians_len: usize,
        camera: &DiffCamera,
    ) -> Option<Vec<bool>> {
        let assignment = self.cluster_assignment.as_ref()?;
        let view_proj = camera.view_projection_mat4();
        let visible_clusters = assignment.get_visible_clusters(&view_proj);

        let mut mask = vec![false; gaussians_len];
        for &cluster in &visible_clusters {
            for (i, &c) in assignment.cluster_indices.iter().enumerate() {
                if c == cluster {
                    mask[i] = true;
                }
            }
        }

        let visible_count = mask.iter().filter(|&v| *v).count();
        if visible_count < gaussians_len {
            log::debug!(
                "Cluster culling: {} / {} Gaussians visible ({} / {} clusters)",
                visible_count,
                gaussians_len,
                visible_clusters.len(),
                assignment.num_clusters
            );
        }

        Some(mask)
    }

    fn loss_weights(&self) -> (f32, f32, f32) {
        let color_weight = if self.is_litegs_mode() {
            1.0 - LITEGS_LAMBDA_DSSIM
        } else {
            0.8
        };
        let ssim_weight = if self.is_litegs_mode() {
            LITEGS_LAMBDA_DSSIM
        } else {
            0.2
        };
        let depth_weight = if self.is_litegs_mode() {
            if self.litegs.enable_depth {
                LITEGS_DEPTH_LOSS_WEIGHT
            } else {
                0.0
            }
        } else {
            LITEGS_DEPTH_LOSS_WEIGHT
        };
        (color_weight, ssim_weight, depth_weight)
    }

    fn total_loss_for_render_result(
        &self,
        gaussians: &TrainableGaussians,
        rendered: &RenderedFrame,
        projected: &ProjectedGaussians,
        frame: &MetalTrainingFrame,
    ) -> candle_core::Result<f32> {
        let color_loss = mean_abs_diff(&rendered.color, &frame.target_color)?;
        let depth_loss =
            masked_mean_abs_diff(&rendered.depth, &frame.target_depth, &frame.target_depth)?;
        let rendered_color_cpu = rendered.color.flatten_all()?.to_vec1::<f32>()?;
        let (ssim_value, _) = ssim_gradient(
            &rendered_color_cpu,
            &frame.target_color_cpu,
            self.render_width,
            self.render_height,
        );
        let ssim_loss_term = 1.0 - ssim_value;
        let (color_weight, ssim_weight, depth_weight) = self.loss_weights();
        let scale_reg_term =
            if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 && projected.visible_count > 0
            {
                self.litegs_scale_regularization_term(
                    &gaussians
                        .scales
                        .as_tensor()
                        .index_select(&projected.source_indices, 0)?,
                )?
            } else {
                Tensor::new(0.0f32, color_loss.device())?
            };
        let transmittance_term = if self.is_litegs_mode() && self.litegs.enable_transmittance {
            rendered.alpha.mean_all()?
        } else {
            Tensor::new(0.0f32, color_loss.device())?
        };

        let mut total =
            color_loss
                .affine(color_weight as f64, 0.0)?
                .broadcast_add(&Tensor::new(
                    ssim_weight * ssim_loss_term,
                    color_loss.device(),
                )?)?;
        if depth_weight > 0.0 {
            total = total.broadcast_add(&depth_loss.affine(depth_weight as f64, 0.0)?)?;
        }
        if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 {
            total =
                total.broadcast_add(&scale_reg_term.affine(self.litegs.reg_weight as f64, 0.0)?)?;
        }
        if self.is_litegs_mode() && self.litegs.enable_transmittance {
            total = total.broadcast_add(&transmittance_term)?;
        }

        total.to_vec0::<f32>()
    }

    fn loss_for_camera(
        &mut self,
        gaussians: &TrainableGaussians,
        frame: &MetalTrainingFrame,
        camera: &DiffCamera,
    ) -> candle_core::Result<f32> {
        let collect_visible_indices = self.is_litegs_mode() && self.litegs.reg_weight > 0.0;
        let cluster_visible_mask = self.cluster_visible_mask_for_camera(gaussians.len(), camera);
        let (rendered, projected, _) = self.render(
            gaussians,
            camera,
            false,
            collect_visible_indices,
            cluster_visible_mask.as_deref(),
        )?;
        self.total_loss_for_render_result(gaussians, &rendered, &projected, frame)
    }

    fn pose_parameter_grads(
        &mut self,
        gaussians: &TrainableGaussians,
        frame: &MetalTrainingFrame,
        frame_idx: usize,
    ) -> candle_core::Result<Option<(Tensor, Tensor)>> {
        let Some(embedding) = self
            .pose_embeddings
            .as_ref()
            .and_then(|pose_embeddings| pose_embeddings.get(frame_idx))
            .cloned()
        else {
            return Ok(None);
        };
        let device = self.device.clone();

        let grads = crate::training::pose_embedding::compute_pose_gradients_fd(
            &embedding,
            frame.camera.fx,
            frame.camera.fy,
            frame.camera.cx,
            frame.camera.cy,
            frame.camera.width,
            frame.camera.height,
            |camera| self.loss_for_camera(gaussians, frame, camera),
            &device,
        )?;
        Ok(Some(grads))
    }

    fn maybe_apply_topology_updates(
        &mut self,
        gaussians: &mut TrainableGaussians,
        _frame_idx: usize,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        if gaussians.len() == 0 {
            return Ok(());
        }

        self.refresh_litegs_topology_window_metrics(frame_count);

        let mut completed_epoch = None;
        let mut should_reset_opacity = false;
        let mut allow_extra_growth = false;
        let (mut should_densify, mut should_prune) = if self.is_litegs_mode() {
            let Some(epoch) = self.litegs_current_epoch(frame_count) else {
                return Ok(());
            };
            completed_epoch = Some(epoch);
            let passed_warmup = self.iteration > self.topology_warmup;
            let densify_until = self.litegs_densify_until_epoch(frame_count);
            let active_window = epoch >= self.litegs.densify_from && epoch < densify_until;
            let frozen = self
                .litegs
                .topology_freeze_after_epoch
                .map(|freeze_epoch| epoch >= freeze_epoch)
                .unwrap_or(false);
            let refine_every = self.litegs.refine_every.max(1);
            let should_refine =
                passed_warmup && !frozen && active_window && self.iteration % refine_every == 0;
            let opacity_reset_period =
                refine_every.saturating_mul(self.litegs.opacity_reset_interval.max(1));
            should_reset_opacity = passed_warmup
                && !frozen
                && active_window
                && self.iteration % opacity_reset_period.max(1) == 0;
            allow_extra_growth = should_refine && self.iteration < self.litegs.growth_stop_iter;

            (should_refine, should_refine)
        } else {
            (
                self.should_densify_at(self.iteration),
                self.should_prune_at(self.iteration),
            )
        };
        if (!should_densify && !should_prune && !should_reset_opacity) || gaussians.len() == 0 {
            return Ok(());
        }

        let topology_start = Instant::now();
        let old_len = gaussians.len();
        let mut stats = self.gaussian_stats.clone();
        if stats.len() != old_len {
            stats.resize(old_len, MetalGaussianStats::default());
        }

        let analysis = self.analyze_topology_candidates(gaussians, &stats)?;
        let litegs_requested_additions = if self.is_litegs_mode() && should_densify {
            self.litegs_requested_additions(&analysis.infos, allow_extra_growth)
        } else {
            0
        };
        let requested_cap = if self.is_litegs_mode() {
            self.max_gaussian_budget
                .max(old_len.saturating_add(litegs_requested_additions))
        } else {
            self.max_gaussian_budget
                .max(old_len.saturating_add(self.legacy_max_densify_per_update))
        };
        let max_gaussians = self.max_topology_gaussians(requested_cap, old_len, frame_count);
        let litegs_selection = if self.is_litegs_mode() && should_densify {
            self.litegs_select_densify_candidates(
                &analysis.infos,
                max_gaussians.saturating_sub(old_len),
                allow_extra_growth,
            )
        } else {
            LiteGsDensifySelection::default()
        };
        if self.is_litegs_mode()
            && (should_densify || should_prune || should_reset_opacity)
            && litegs_selection.selected_indices.is_empty()
            && analysis.prune_candidates > 0
        {
            if self.should_log_topology_at(self.iteration) {
                log::info!(
                    "Metal topology check at iter {} skipped destructive LiteGS prune/reset because no replacement or growth sources were available | epoch={:?} | prune_candidates={} | growth_candidates={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
                    self.iteration,
                    completed_epoch,
                    analysis.prune_candidates,
                    analysis.growth_candidates,
                    analysis.max_grad,
                    analysis.mean_grad,
                );
            }
            should_densify = false;
            should_prune = false;
            should_reset_opacity = false;
        }

        if analysis.clone_candidates == 0
            && analysis.split_candidates == 0
            && analysis.prune_candidates == 0
            && !should_reset_opacity
        {
            if self.should_log_topology_at(self.iteration) {
                log::info!(
                    "Metal topology check at iter {} found no eligible candidates | densify={} | prune={} | reset_opacity={} | gaussians={} | budget_cap={} | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={}",
                    self.iteration,
                    should_densify,
                    should_prune,
                    should_reset_opacity,
                    old_len,
                    max_gaussians,
                    analysis.max_grad,
                    analysis.mean_grad,
                    analysis.active_grad_stats,
                    analysis.small_scale_stats,
                    analysis.opacity_ready_stats,
                    analysis.clone_candidates,
                    analysis.split_candidates,
                    analysis.prune_candidates,
                );
            }
            if self.is_litegs_mode() {
                self.reset_litegs_refine_window_stats();
            }
            return Ok(());
        }

        let mut snapshot = self.export_snapshot(gaussians)?;
        let mut origins: Vec<Option<usize>> = (0..snapshot.len()).map(Some).collect();

        let added = if should_densify {
            if self.is_litegs_mode() {
                self.densify_snapshot_litegs(
                    &mut snapshot,
                    &mut stats,
                    &mut origins,
                    max_gaussians,
                    &litegs_selection.selected_indices,
                )
            } else {
                self.densify_snapshot(
                    &mut snapshot,
                    &mut stats,
                    &mut origins,
                    &analysis.infos,
                    max_gaussians,
                )
            }
        } else {
            0
        };
        let pruned = if should_prune {
            if self.is_litegs_mode() {
                self.prune_snapshot_litegs(&mut snapshot, &mut stats, &mut origins, &analysis.infos)
            } else {
                self.prune_snapshot(&mut snapshot, &mut stats, &mut origins, &analysis.infos)
            }
        } else {
            0
        };

        // Apply Morton code spatial sorting after densification for better memory coherence
        let morton_sorted = if self.is_litegs_mode()
            && self.litegs.morton_sort_on_densify
            && added > 0
            && snapshot.len() > 1
        {
            let morton_perm = super::morton::morton_sort_permutation(
                &snapshot.positions,
                super::morton::DEFAULT_MORTON_BITS,
            );

            if morton_perm.len() == snapshot.len() {
                // Apply permutation to all snapshot arrays
                snapshot.positions = super::morton::permute_vec3(&snapshot.positions, &morton_perm);
                snapshot.log_scales =
                    super::morton::permute_vec3(&snapshot.log_scales, &morton_perm);
                snapshot.rotations = super::morton::permute_vec4(&snapshot.rotations, &morton_perm);
                snapshot.opacity_logits =
                    super::morton::permute_scalar(&snapshot.opacity_logits, &morton_perm);
                snapshot.colors = super::morton::permute_vec3(&snapshot.colors, &morton_perm);

                // Apply permutation to sh_rest if present
                let sh_rest_row_width = snapshot.sh_rest_row_width();
                if sh_rest_row_width > 0 && !snapshot.sh_rest.is_empty() {
                    snapshot.sh_rest = super::morton::permute_rows(
                        &snapshot.sh_rest,
                        sh_rest_row_width,
                        &morton_perm,
                    );
                }

                // Apply permutation to stats and origins
                let mut new_stats = vec![MetalGaussianStats::default(); morton_perm.len()];
                let mut new_origins = vec![None; morton_perm.len()];
                for (new_idx, &old_idx) in morton_perm.iter().enumerate() {
                    new_stats[new_idx] = stats[old_idx].clone();
                    new_origins[new_idx] = origins[old_idx];
                }
                stats = new_stats;
                origins = new_origins;

                true
            } else {
                false
            }
        } else {
            false
        };

        let topology_duration = topology_start.elapsed();
        let topology_ms = duration_ms(topology_duration);
        let topology_ratio = self
            .last_step_duration
            .map(|step| {
                let step_ms = duration_ms(step);
                if step_ms > 0.0 {
                    topology_ms / step_ms
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let guardrail_triggered = topology_ms >= 50.0 || topology_ratio >= 0.35;
        let late_stage = completed_epoch
            .map(|epoch| self.is_litegs_late_stage_epoch(epoch, frame_count))
            .unwrap_or(false);

        if added == 0 && pruned == 0 {
            if should_reset_opacity {
                self.apply_litegs_opacity_reset(gaussians, completed_epoch, frame_count)?;
                if self.is_litegs_mode() {
                    self.reset_litegs_refine_window_stats();
                } else {
                    self.reset_gaussian_stats(gaussians.len());
                }
                self.topology_metrics.final_gaussians = Some(gaussians.len());
            }
            if self.should_log_topology_at(self.iteration) || guardrail_triggered {
                log::info!(
                    "Metal topology check at iter {} | epoch={:?} | late_stage={} | made no changes | densify={} | prune={} | reset_opacity={} | gaussians={} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={}",
                    self.iteration,
                    completed_epoch,
                    late_stage,
                    should_densify,
                    should_prune,
                    should_reset_opacity,
                    old_len,
                    max_gaussians,
                    topology_ms,
                    topology_ratio * 100.0,
                    analysis.max_grad,
                    analysis.mean_grad,
                    analysis.active_grad_stats,
                    analysis.small_scale_stats,
                    analysis.opacity_ready_stats,
                    analysis.clone_candidates,
                    analysis.split_candidates,
                    analysis.prune_candidates,
                );
            }
            if guardrail_triggered {
                log::warn!(
                    "Metal topology guardrail triggered at iter {} | topology={:.2}ms | previous_step={:.2}ms | share={:.0}% | consider increasing --topology-warmup, --densify-interval, or --prune-interval",
                    self.iteration,
                    topology_ms,
                    self.last_step_duration.map(duration_ms).unwrap_or(0.0),
                    topology_ratio * 100.0,
                );
            }
            if self.is_litegs_mode() {
                self.reset_litegs_refine_window_stats();
            }
            return Ok(());
        }

        let rebuilt = match snapshot.color_representation {
            TrainableColorRepresentation::Rgb => TrainableGaussians::new(
                &snapshot.positions,
                &snapshot.log_scales,
                &snapshot.rotations,
                &snapshot.opacity_logits,
                &snapshot.colors,
                &self.device,
            )?,
            TrainableColorRepresentation::SphericalHarmonics { degree } => {
                TrainableGaussians::new_with_sh(
                    &snapshot.positions,
                    &snapshot.log_scales,
                    &snapshot.rotations,
                    &snapshot.opacity_logits,
                    &snapshot.colors,
                    &snapshot.sh_rest,
                    degree,
                    &self.device,
                )?
            }
        };
        let new_adam = match self.adam.take() {
            Some(old_state) => self.rebuild_adam_state(&old_state, &origins)?,
            None => MetalAdamState::new(&rebuilt)?,
        };

        *gaussians = rebuilt;
        self.adam = Some(new_adam);
        self.gaussian_stats = stats;
        self.sync_cluster_assignment(gaussians, true)?;
        self.runtime.reserve_core_buffers(gaussians.len())?;
        if should_reset_opacity {
            self.apply_litegs_opacity_reset(gaussians, completed_epoch, frame_count)?;
        }
        if self.is_litegs_mode() {
            // Resize stats array to match new gaussian count
            // Newly-added Gaussians get default stats (age=0, visible_count=0)
            // Existing Gaussians retain their accumulated stats (age, visibility, etc.)
            self.gaussian_stats
                .resize(gaussians.len(), MetalGaussianStats::default());
            self.reset_litegs_refine_window_stats();
        }
        self.topology_metrics.final_gaussians = Some(gaussians.len());
        if added > 0 {
            self.topology_metrics.densify_events =
                self.topology_metrics.densify_events.saturating_add(1);
            self.topology_metrics.densify_added =
                self.topology_metrics.densify_added.saturating_add(added);
            record_topology_epoch(
                &mut self.topology_metrics.first_densify_epoch,
                &mut self.topology_metrics.last_densify_epoch,
                completed_epoch,
            );
            if late_stage {
                self.topology_metrics.late_stage_densify_events = self
                    .topology_metrics
                    .late_stage_densify_events
                    .saturating_add(1);
                self.topology_metrics.late_stage_densify_added = self
                    .topology_metrics
                    .late_stage_densify_added
                    .saturating_add(added);
            }
        }
        if pruned > 0 {
            self.topology_metrics.prune_events =
                self.topology_metrics.prune_events.saturating_add(1);
            self.topology_metrics.prune_removed =
                self.topology_metrics.prune_removed.saturating_add(pruned);
            record_topology_epoch(
                &mut self.topology_metrics.first_prune_epoch,
                &mut self.topology_metrics.last_prune_epoch,
                completed_epoch,
            );
            if late_stage {
                self.topology_metrics.late_stage_prune_events = self
                    .topology_metrics
                    .late_stage_prune_events
                    .saturating_add(1);
                self.topology_metrics.late_stage_prune_removed = self
                    .topology_metrics
                    .late_stage_prune_removed
                    .saturating_add(pruned);
            }
        }

        log::info!(
            "Metal topology update at iter {} | epoch={:?} | late_stage={} | densify={} | prune={} | reset_opacity={} | added {} | pruned {} | morton={} | gaussians {} -> {} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
            self.iteration,
            completed_epoch,
            late_stage,
            should_densify,
            should_prune,
            should_reset_opacity,
            added,
            pruned,
            morton_sorted,
            old_len,
            gaussians.len(),
            max_gaussians,
            topology_ms,
            topology_ratio * 100.0,
            analysis.active_grad_stats,
            analysis.small_scale_stats,
            analysis.opacity_ready_stats,
            analysis.clone_candidates,
            analysis.split_candidates,
            analysis.prune_candidates,
            analysis.max_grad,
            analysis.mean_grad,
        );
        if guardrail_triggered {
            log::warn!(
                "Metal topology guardrail triggered at iter {} | topology={:.2}ms | previous_step={:.2}ms | share={:.0}% | added={} | pruned={}",
                self.iteration,
                topology_ms,
                self.last_step_duration.map(duration_ms).unwrap_or(0.0),
                topology_ratio * 100.0,
                added,
                pruned,
            );
        }
        Ok(())
    }

    pub fn new(
        input_width: usize,
        input_height: usize,
        config: &TrainingConfig,
        device: Device,
    ) -> candle_core::Result<Self> {
        let (render_width, render_height) =
            scaled_dimensions(input_width, input_height, config.metal_render_scale);
        let pixel_count = render_width * render_height;
        let source_pixel_count = input_width.saturating_mul(input_height);
        let runtime = MetalRuntime::new(render_width, render_height, device.clone())?;
        let use_native_forward = config.metal_use_native_forward && device.is_metal();

        Ok(Self {
            training_profile: config.training_profile,
            litegs: config.litegs.clone(),
            device,
            render_width,
            render_height,
            pixel_count,
            source_pixel_count,
            chunk_size: config.metal_gaussian_chunk_size.max(1),
            densify_interval: config.densify_interval,
            prune_interval: config.prune_interval,
            topology_warmup: config.topology_warmup,
            topology_log_interval: config.topology_log_interval.max(1),
            prune_threshold: config.prune_threshold,
            legacy_densify_grad_threshold: config.legacy_densify_grad_threshold,
            legacy_clone_scale_threshold: config.legacy_clone_scale_threshold,
            legacy_split_scale_threshold: config.legacy_split_scale_threshold,
            legacy_prune_scale_threshold: config.legacy_prune_scale_threshold,
            legacy_max_densify_per_update: config.legacy_max_densify_per_update.max(1),
            max_gaussian_budget: config.max_initial_gaussians.max(1),
            topology_memory_budget: Some(training_memory_budget(config)),
            scene_extent: 1.0,
            lr_pos: config.lr_position,
            lr_pos_final: config.lr_pos_final,
            max_iterations: config.iterations,
            lr_scale: config.lr_scale,
            lr_rotation: config.lr_rotation,
            lr_opacity: config.lr_opacity,
            lr_color: config.lr_color,
            lr_sh_rest: config.lr_color / 10.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            rotation_frozen: config.lr_rotation == 0.0,
            active_sh_degree: 0,
            last_loss_terms: ParityLossTerms::default(),
            topology_metrics: ParityTopologyMetrics::default(),
            last_learning_rates: LiteGsOptimizerLrs::default(),
            last_depth_valid_pixels: None,
            last_depth_grad_scale: None,
            loss_curve_samples: Vec::new(),
            profile_steps: config.metal_profile_steps,
            profile_interval: config.metal_profile_interval.max(1),
            use_native_forward,
            runtime,
            adam: None,
            gaussian_stats: Vec::new(),
            iteration: 0,
            loss_history: Vec::new(),
            last_step_duration: None,
            cached_target_frame_idx: None,
            pose_embeddings: None,
            cluster_assignment: None,
        })
    }

    fn prepare_frames(
        &self,
        loaded: &LoadedTrainingData,
    ) -> Result<Vec<MetalTrainingFrame>, TrainingError> {
        let mut frames = Vec::with_capacity(loaded.cameras.len());
        for idx in 0..loaded.cameras.len() {
            let src_camera = &loaded.cameras[idx];
            let target_color_cpu = resize_rgb(
                &loaded.colors[idx],
                src_camera.width,
                src_camera.height,
                self.render_width,
                self.render_height,
            );
            let target_depth_cpu = resize_depth(
                &loaded.depths[idx],
                src_camera.width,
                src_camera.height,
                self.render_width,
                self.render_height,
            );
            let scaled_camera = scale_camera(
                src_camera,
                self.render_width,
                self.render_height,
                &self.device,
            )?;
            frames.push(MetalTrainingFrame {
                camera: scaled_camera,
                target_color: Tensor::from_slice(
                    &target_color_cpu,
                    (self.pixel_count, 3),
                    &self.device,
                )?,
                target_depth: Tensor::from_slice(
                    &target_depth_cpu,
                    (self.pixel_count,),
                    &self.device,
                )?,
                target_color_cpu,
                target_depth_cpu,
            });
        }
        Ok(frames)
    }

    fn train(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frames: &[MetalTrainingFrame],
        max_iterations: usize,
    ) -> candle_core::Result<MetalTrainingStats> {
        if frames.is_empty() {
            candle_core::bail!("metal backend received zero training frames");
        }
        self.loss_curve_samples.clear();
        self.adam = Some(MetalAdamState::new(gaussians)?);
        self.reset_gaussian_stats(gaussians.len());
        self.topology_metrics.initialization_gaussians = Some(gaussians.len());
        self.topology_metrics.final_gaussians = Some(gaussians.len());
        self.refresh_litegs_topology_window_metrics(frames.len());
        self.runtime.reserve_core_buffers(gaussians.len())?;
        if self.device.is_metal() {
            self.runtime
                .dispatch_fill_u32(MetalBufferSlot::TileIndices, 0, 1)?;
        }
        let runtime_stats = self.runtime.stats();

        log::info!(
            "MetalTrainer running at {}x{} | chunk_size={} | native_forward={} | topology(densify={} prune={} warmup={} log={}) | frames={} | initial_gaussians={} | tiles={} | runtime_buffers={} | pipeline_warmups={} | tile_index_capacity={}B",
            self.render_width,
            self.render_height,
            self.chunk_size,
            self.use_native_forward,
            self.densify_interval,
            self.prune_interval,
            self.topology_warmup,
            self.topology_log_interval,
            frames.len(),
            gaussians.len(),
            runtime_stats.tile_windows,
            runtime_stats.buffer_allocations,
            runtime_stats.pipeline_compilations,
            self.runtime.buffer_capacity(MetalBufferSlot::TileIndices),
        );

        let train_start = Instant::now();
        for iter in 0..max_iterations {
            let frame_idx = iter % frames.len();
            let should_log = iter < 5 || iter % 25 == 0;
            let should_profile =
                should_profile_iteration(self.profile_steps, self.profile_interval, iter);
            let step_start = Instant::now();
            let outcome = self.training_step(
                gaussians,
                &frames[frame_idx],
                frame_idx,
                frames.len(),
                should_profile,
            )?;
            self.record_loss_curve_sample(iter, frame_idx, max_iterations);
            self.maybe_apply_topology_updates(gaussians, frame_idx, frames.len())?;
            if should_log {
                log::info!(
                    "Metal iter {:5}/{:5} | frame {:3}/{:3} | visible {:5}/{:5} | loss {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    frames.len(),
                    outcome.visible_gaussians,
                    outcome.total_gaussians,
                    outcome.loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }
            if let Some(profile) = outcome.profile {
                profile.log(iter, max_iterations);
            }
        }
        self.topology_metrics.final_gaussians = Some(gaussians.len());

        let final_metrics = summarize_training_metrics(&self.loss_history, frames.len());
        Ok(MetalTrainingStats {
            final_loss: final_metrics.final_loss,
            final_step_loss: final_metrics.final_step_loss,
            telemetry: self.current_telemetry(frames.len()),
        })
    }

    fn training_step(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frame: &MetalTrainingFrame,
        frame_idx: usize,
        frame_count: usize,
        should_profile: bool,
    ) -> candle_core::Result<MetalStepOutcome> {
        self.iteration += 1;
        self.active_sh_degree = self.litegs_active_sh_degree_for_iteration(frame_count);
        self.last_learning_rates = self.current_learning_rates();
        let total_start = Instant::now();
        let collect_visible_indices = self.is_litegs_mode()
            || self.should_densify_at(self.iteration)
            || self.should_prune_at(self.iteration);

        if self.litegs.cluster_size > 0 {
            self.sync_cluster_assignment(gaussians, false)?;
        }

        // Get the camera to use for rendering
        // If pose embeddings exist, use the learnable camera for this frame
        let render_camera = if let Some(ref pose_embeddings) = self.pose_embeddings {
            if let Some(embedding) = pose_embeddings.get(frame_idx) {
                embedding.to_diff_camera(
                    frame.camera.fx,
                    frame.camera.fy,
                    frame.camera.cx,
                    frame.camera.cy,
                    frame.camera.width,
                    frame.camera.height,
                    &self.device,
                )?
            } else {
                frame.camera.clone()
            }
        } else {
            frame.camera.clone()
        };

        let cluster_visible_mask =
            self.cluster_visible_mask_for_camera(gaussians.len(), &render_camera);

        let (rendered, projected, render_profile) = self.render(
            gaussians,
            &render_camera,
            should_profile,
            collect_visible_indices,
            cluster_visible_mask.as_deref(),
        )?;
        let mut profile = MetalStepProfile::from_render(render_profile);

        let loss_start = Instant::now();
        let color_loss = mean_abs_diff(&rendered.color, &frame.target_color)?;
        let depth_loss =
            masked_mean_abs_diff(&rendered.depth, &frame.target_depth, &frame.target_depth)?;

        // Compute SSIM gradient on CPU using the rendered output.
        let rendered_color_cpu = rendered.color.flatten_all()?.to_vec1::<f32>()?;
        let (ssim_value, ssim_grads) = ssim_gradient(
            &rendered_color_cpu,
            &frame.target_color_cpu,
            self.render_width,
            self.render_height,
        );

        let ssim_loss_term = 1.0 - ssim_value;
        let (color_weight, ssim_weight, depth_weight) = self.loss_weights();
        let valid_depth_pixels = if depth_weight > 0.0 {
            Some(valid_depth_sample_count(&frame.target_depth_cpu))
        } else {
            None
        };
        let depth_grad_scale = if depth_weight > 0.0 {
            Some(depth_backward_scale(depth_weight, &frame.target_depth_cpu))
        } else {
            None
        };
        let scale_reg_term =
            if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 && projected.visible_count > 0
            {
                self.litegs_scale_regularization_term(
                    &gaussians
                        .scales
                        .as_tensor()
                        .index_select(&projected.source_indices, 0)?,
                )?
            } else {
                Tensor::new(0.0f32, color_loss.device())?
            };
        let transmittance_term = if self.is_litegs_mode() && self.litegs.enable_transmittance {
            rendered.alpha.mean_all()?
        } else {
            Tensor::new(0.0f32, color_loss.device())?
        };

        let mut total =
            color_loss
                .affine(color_weight as f64, 0.0)?
                .broadcast_add(&Tensor::new(
                    ssim_weight * ssim_loss_term,
                    color_loss.device(),
                )?)?;
        if depth_weight > 0.0 {
            total = total.broadcast_add(&depth_loss.affine(depth_weight as f64, 0.0)?)?;
        }
        if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 {
            total =
                total.broadcast_add(&scale_reg_term.affine(self.litegs.reg_weight as f64, 0.0)?)?;
        }
        if self.is_litegs_mode() && self.litegs.enable_transmittance {
            total = total.broadcast_add(&transmittance_term)?;
        }
        let loss_value = total.to_vec0::<f32>()?;
        self.last_loss_terms = ParityLossTerms {
            l1: Some(color_loss.to_vec0::<f32>()?),
            ssim: Some(ssim_loss_term),
            scale_regularization: if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 {
                Some(scale_reg_term.to_vec0::<f32>()?)
            } else {
                None
            },
            transmittance: if self.is_litegs_mode() && self.litegs.enable_transmittance {
                Some(transmittance_term.to_vec0::<f32>()?)
            } else {
                None
            },
            depth: if depth_weight > 0.0 {
                Some(depth_loss.to_vec0::<f32>()?)
            } else {
                None
            },
            total: Some(loss_value),
        };
        self.last_depth_valid_pixels = valid_depth_pixels;
        self.last_depth_grad_scale = depth_grad_scale;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        // Write SSIM gradient every step (depends on rendered output which changes each step).
        self.runtime.write_ssim_grad(&ssim_grads)?;

        let backward_loss_scales = MetalBackwardLossScales {
            color: color_weight / frame.target_color_cpu.len().max(1) as f32,
            depth: depth_grad_scale.unwrap_or(0.0),
            ssim: ssim_weight / frame.target_color_cpu.len().max(1) as f32,
            alpha: if self.is_litegs_mode() && self.litegs.enable_transmittance {
                1.0 / self.pixel_count.max(1) as f32
            } else {
                0.0
            },
        };
        if self.cached_target_frame_idx != Some(frame_idx) {
            // color_scale weights the L1 contribution in the backward pass (0.8 factor).
            // ssim_scale weights the per-pixel SSIM gradient contribution.
            self.runtime.write_target_data(
                &frame.target_color_cpu,
                &frame.target_depth_cpu,
                backward_loss_scales.color,
                backward_loss_scales.depth,
                backward_loss_scales.ssim,
                backward_loss_scales.alpha,
            )?;
            self.cached_target_frame_idx = Some(frame_idx);
        }
        let backward = super::metal_backward::backward_weighted_l1(
            &mut self.runtime,
            &projected.tile_bins,
            gaussians.len(),
            &render_camera,
        )?;
        profile.backward = backward_start.elapsed();

        if debug_training_step_probe_enabled()
            && (self.iteration <= 6 || self.iteration % self.litegs.refine_every.max(1) == 0)
        {
            let param_grad_stats = abs_stats(&backward.grad_magnitudes);
            let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
            log::info!(
                "Backward stats step {} | visible={} | param_grad_max={:.6e} | param_grad_mean={:.6e} | param_grad_nonzero={} | projected_grad_max={:.6e} | projected_grad_mean={:.6e} | projected_grad_nonzero={}",
                self.iteration,
                projected.visible_count,
                param_grad_stats.0,
                param_grad_stats.1,
                param_grad_stats.2,
                projected_grad_stats.0,
                projected_grad_stats.1,
                projected_grad_stats.2,
            );
        }

        let debug_probe = debug_training_step_probe_enabled() && self.iteration <= 2;
        let debug_param_snapshots = if debug_probe {
            Some((
                gaussians.positions().flatten_all()?.to_vec1::<f32>()?,
                gaussians
                    .scales
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                gaussians
                    .opacities
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                gaussians.colors().flatten_all()?.to_vec1::<f32>()?,
            ))
        } else {
            None
        };
        if debug_probe {
            let position_stats = tensor_abs_stats(&backward.grads.positions)?;
            let scale_stats = tensor_abs_stats(&backward.grads.log_scales)?;
            let opacity_stats = tensor_abs_stats(&backward.grads.opacity_logits)?;
            let color_stats = tensor_abs_stats(&backward.grads.colors)?;
            let param_grad_stats = abs_stats(&backward.grad_magnitudes);
            let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
            log::info!(
                "Metal debug backward step {} | frame={} | grad_positions(max={:.6e} mean={:.6e} nz={}) | grad_scales(max={:.6e} mean={:.6e} nz={}) | grad_opacity(max={:.6e} mean={:.6e} nz={}) | grad_colors(max={:.6e} mean={:.6e} nz={}) | param_grad_mag(max={:.6e} mean={:.6e} nz={}) | projected_grad_mag(max={:.6e} mean={:.6e} nz={})",
                self.iteration,
                frame_idx,
                position_stats.0,
                position_stats.1,
                position_stats.2,
                scale_stats.0,
                scale_stats.1,
                scale_stats.2,
                opacity_stats.0,
                opacity_stats.1,
                opacity_stats.2,
                color_stats.0,
                color_stats.1,
                color_stats.2,
                param_grad_stats.0,
                param_grad_stats.1,
                param_grad_stats.2,
                projected_grad_stats.0,
                projected_grad_stats.1,
                projected_grad_stats.2,
            );
        }

        let optimizer_start = Instant::now();
        let effective_lr_pos = self.compute_lr_pos();
        let scale_reg_grad = if self.is_litegs_mode()
            && self.litegs.reg_weight > 0.0
            && projected.visible_count > 0
        {
            let visible_log_scales = gaussians
                .scales
                .as_tensor()
                .index_select(&projected.source_indices, 0)?;
            let visible_reg_grad = self.litegs_scale_regularization_grad(&visible_log_scales)?;
            Some(Tensor::zeros_like(gaussians.scales.as_tensor())?.index_add(
                &projected.source_indices,
                &visible_reg_grad,
                0,
            )?)
        } else {
            None
        };
        let rotation_parameter_grads = if self.lr_rotation > 0.0 && projected.visible_count > 0 {
            Some(self.rotation_parameter_grads(
                gaussians,
                &projected,
                &rendered,
                &rendered_color_cpu,
                &frame.target_color_cpu,
                &frame.target_depth_cpu,
                &ssim_grads,
                backward_loss_scales,
                &render_camera,
            )?)
        } else {
            None
        };
        let pose_parameter_grads = self.pose_parameter_grads(gaussians, frame, frame_idx)?;
        self.apply_backward_grads(
            gaussians,
            &backward.grads,
            &projected,
            &render_camera,
            effective_lr_pos,
            scale_reg_grad.as_ref(),
            rotation_parameter_grads.as_ref(),
        )?;
        if let Some((before_positions, before_scales, before_opacities, before_colors)) =
            debug_param_snapshots.as_ref()
        {
            let position_delta = max_abs_delta(before_positions, gaussians.positions())?;
            let scale_delta = max_abs_delta(before_scales, gaussians.scales.as_tensor())?;
            let opacity_delta = max_abs_delta(before_opacities, gaussians.opacities.as_tensor())?;
            let color_delta = max_abs_delta(before_colors, &gaussians.colors())?;
            log::info!(
                "Metal debug optimizer step {} | frame={} | delta_positions={:.6e} | delta_scales={:.6e} | delta_opacity={:.6e} | delta_colors={:.6e}",
                self.iteration,
                frame_idx,
                position_delta,
                scale_delta,
                opacity_delta,
                color_delta,
            );
        }
        if let Some((quaternion_grad, translation_grad)) = pose_parameter_grads {
            if let Some(pose_embeddings) = self.pose_embeddings.as_mut() {
                pose_embeddings.adam_step(&[frame_idx], &[quaternion_grad], &[translation_grad])?;
            }
        }

        self.update_gaussian_stats(
            &backward.grad_magnitudes,
            &backward.projected_grad_magnitudes,
            &projected,
            gaussians.len(),
        )?;
        self.synchronize_if_needed(should_profile)?;
        profile.optimizer = optimizer_start.elapsed();
        profile.total = total_start.elapsed();
        self.last_step_duration = Some(profile.total);
        self.loss_history.push(loss_value);

        Ok(MetalStepOutcome {
            loss: loss_value,
            visible_gaussians: profile.visible_gaussians,
            total_gaussians: profile.total_gaussians,
            profile: if should_profile { Some(profile) } else { None },
        })
    }

    /// Compute the current effective position learning rate using exponential decay.
    ///
    /// η(t) = η₀ × (η_end / η₀)^(t / T)
    fn compute_lr_pos(&self) -> f32 {
        let lr0 = self.lr_pos;
        let lr_end = self.lr_pos_final;
        let t = self.iteration as f32;
        let total = self.max_iterations as f32;
        if total <= 0.0 || lr0 <= 0.0 || lr_end <= 0.0 {
            return lr0;
        }
        lr0 * (lr_end / lr0).powf(t / total)
    }

    fn litegs_scale_regularization_term(
        &self,
        visible_log_scales: &Tensor,
    ) -> candle_core::Result<Tensor> {
        visible_log_scales.exp()?.sqr()?.mean_all()
    }

    fn litegs_scale_regularization_grad(
        &self,
        visible_log_scales: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let visible_elem_count = visible_log_scales.elem_count().max(1) as f32;
        visible_log_scales.exp()?.sqr()?.affine(
            ((2.0 * self.litegs.reg_weight) / visible_elem_count) as f64,
            0.0,
        )
    }

    fn render_colors_for_camera(
        &self,
        gaussians: &TrainableGaussians,
        positions: &Tensor,
        camera: &DiffCamera,
    ) -> candle_core::Result<Tensor> {
        if !gaussians.uses_spherical_harmonics() {
            return Ok(gaussians.render_colors()?.detach());
        }

        let active_degree = self.active_sh_degree.min(gaussians.sh_degree());
        let row_count = gaussians.len();
        if row_count == 0 {
            return Tensor::zeros((0, 3), DType::F32, &self.device);
        }

        let dirs = view_directions_for_camera(positions, camera, &self.device)?;
        let x = dirs.narrow(1, 0, 1)?.squeeze(1)?;
        let y = dirs.narrow(1, 1, 1)?.squeeze(1)?;
        let z = dirs.narrow(1, 2, 1)?.squeeze(1)?;
        let xx = x.sqr()?;
        let yy = y.sqr()?;
        let zz = z.sqr()?;
        let xy = x.broadcast_mul(&y)?;
        let yz = y.broadcast_mul(&z)?;
        let xz = x.broadcast_mul(&z)?;

        let sh_0 = gaussians.sh_0().detach();
        let sh_rest = gaussians.sh_rest().detach();
        let mut color = sh_0.affine(SH_C0 as f64, 0.5)?;

        macro_rules! add_sh_term {
            ($coeff_idx:expr, $basis:expr) => {{
                let coeff = sh_rest.narrow(1, $coeff_idx, 1)?.squeeze(1)?;
                let basis = ($basis)?.reshape((row_count, 1))?;
                color = color.broadcast_add(&coeff.broadcast_mul(&basis)?)?;
            }};
        }

        if active_degree > 0 {
            add_sh_term!(0, y.affine((-SH_C1) as f64, 0.0));
            add_sh_term!(1, z.affine(SH_C1 as f64, 0.0));
            add_sh_term!(2, x.affine((-SH_C1) as f64, 0.0));
        }

        if active_degree > 1 {
            add_sh_term!(3, xy.affine(SH_C2[0] as f64, 0.0));
            add_sh_term!(4, yz.affine(SH_C2[1] as f64, 0.0));
            add_sh_term!(
                5,
                zz.affine((2.0 * SH_C2[2]) as f64, 0.0)?
                    .broadcast_sub(&xx.affine(SH_C2[2] as f64, 0.0)?)?
                    .broadcast_sub(&yy.affine(SH_C2[2] as f64, 0.0)?)
            );
            add_sh_term!(6, xz.affine(SH_C2[3] as f64, 0.0));
            add_sh_term!(
                7,
                xx.affine(SH_C2[4] as f64, 0.0)?
                    .broadcast_sub(&yy.affine(SH_C2[4] as f64, 0.0,)?)
            );
        }

        if active_degree > 2 {
            add_sh_term!(
                8,
                y.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?
                    .affine(SH_C3[0] as f64, 0.0)
            );
            add_sh_term!(9, xy.broadcast_mul(&z)?.affine(SH_C3[1] as f64, 0.0));
            add_sh_term!(
                10,
                y.broadcast_mul(
                    &zz.affine(4.0, 0.0)?
                        .broadcast_sub(&xx)?
                        .broadcast_sub(&yy)?,
                )?
                .affine(SH_C3[2] as f64, 0.0)
            );
            add_sh_term!(
                11,
                z.broadcast_mul(
                    &zz.affine(2.0, 0.0)?
                        .broadcast_sub(&xx.affine(3.0, 0.0)?)?
                        .broadcast_sub(&yy.affine(3.0, 0.0)?)?,
                )?
                .affine(SH_C3[3] as f64, 0.0)
            );
            add_sh_term!(
                12,
                x.broadcast_mul(
                    &zz.affine(4.0, 0.0)?
                        .broadcast_sub(&xx)?
                        .broadcast_sub(&yy)?,
                )?
                .affine(SH_C3[4] as f64, 0.0)
            );
            add_sh_term!(
                13,
                z.broadcast_mul(&xx.broadcast_sub(&yy)?)?
                    .affine(SH_C3[5] as f64, 0.0)
            );
            add_sh_term!(
                14,
                x.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                    .affine(SH_C3[6] as f64, 0.0)
            );
        }

        if active_degree > 3 {
            let zz7_minus_1 = zz.affine(7.0, -1.0)?;
            let zz7_minus_3 = zz.affine(7.0, -3.0)?;
            add_sh_term!(
                15,
                xy.broadcast_mul(&xx.broadcast_sub(&yy)?)?
                    .affine(SH_C4[0] as f64, 0.0)
            );
            add_sh_term!(
                16,
                yz.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?
                    .affine(SH_C4[1] as f64, 0.0)
            );
            add_sh_term!(
                17,
                xy.broadcast_mul(&zz.affine(7.0, -1.0)?)?
                    .affine(SH_C4[2] as f64, 0.0)
            );
            add_sh_term!(
                18,
                yz.broadcast_mul(&zz7_minus_3)?.affine(SH_C4[3] as f64, 0.0)
            );
            add_sh_term!(
                19,
                zz.broadcast_mul(&zz.affine(35.0, -30.0)?)?
                    .affine(SH_C4[4] as f64, 3.0 * SH_C4[4] as f64)
            );
            add_sh_term!(
                20,
                xz.broadcast_mul(&zz7_minus_3)?.affine(SH_C4[5] as f64, 0.0)
            );
            add_sh_term!(
                21,
                xx.broadcast_sub(&yy)?
                    .broadcast_mul(&zz7_minus_1)?
                    .affine(SH_C4[6] as f64, 0.0)
            );
            add_sh_term!(
                22,
                xz.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                    .affine(SH_C4[7] as f64, 0.0)
            );
            add_sh_term!(
                23,
                xx.broadcast_mul(&xx.broadcast_sub(&yy.affine(3.0, 0.0)?)?)?
                    .broadcast_sub(&yy.broadcast_mul(&xx.affine(3.0, 0.0)?.broadcast_sub(&yy)?)?,)?
                    .affine(SH_C4[8] as f64, 0.0)
            );
        }

        color.clamp(0.0, f32::MAX)
    }

    fn rotation_parameter_grads(
        &self,
        gaussians: &TrainableGaussians,
        projected: &ProjectedGaussians,
        rendered: &RenderedFrame,
        rendered_color_cpu: &[f32],
        target_color_cpu: &[f32],
        target_depth_cpu: &[f32],
        ssim_grads: &[f32],
        loss_scales: MetalBackwardLossScales,
        camera: &DiffCamera,
    ) -> candle_core::Result<Tensor> {
        let row_count = gaussians.len();
        if row_count == 0 || projected.visible_count == 0 || self.lr_rotation == 0.0 {
            return Tensor::zeros((row_count, 4), DType::F32, &self.device);
        }

        let projected_cpu = projected_rows_to_cpu(projected)?;
        if projected_cpu.is_empty() || projected.tile_bins.total_assignments() == 0 {
            return Tensor::zeros((row_count, 4), DType::F32, &self.device);
        }

        let rendered_depth_cpu = rendered.depth.flatten_all()?.to_vec1::<f32>()?;
        let rendered_alpha_cpu = rendered.alpha.flatten_all()?.to_vec1::<f32>()?;
        let raw_rotation_rows = gaussians.rotations.as_tensor().to_vec2::<f32>()?;
        let color_grads = pixel_color_grads(
            rendered_color_cpu,
            target_color_cpu,
            ssim_grads,
            loss_scales,
        );
        let depth_grads = pixel_depth_grads(&rendered_depth_cpu, target_depth_cpu, loss_scales);
        let mut dl_dsigma_x = vec![0.0f32; row_count];
        let mut dl_dsigma_y = vec![0.0f32; row_count];
        let tile_bins = &projected.tile_bins;
        let packed_indices = tile_bins.packed_indices();
        let num_tiles_x = self.render_width.div_ceil(METAL_TILE_SIZE);

        for &tile_idx in tile_bins.active_tiles() {
            let Some(record) = tile_bins.record(tile_idx) else {
                continue;
            };
            if record.count() == 0 {
                continue;
            }

            let tile_x = tile_idx % num_tiles_x;
            let tile_y = tile_idx / num_tiles_x;
            let min_x = tile_x * METAL_TILE_SIZE;
            let min_y = tile_y * METAL_TILE_SIZE;
            let max_x = (min_x + METAL_TILE_SIZE)
                .min(self.render_width)
                .saturating_sub(1);
            let max_y = (min_y + METAL_TILE_SIZE)
                .min(self.render_height)
                .saturating_sub(1);
            let tile_width = max_x.saturating_sub(min_x) + 1;
            let tile_height = max_y.saturating_sub(min_y) + 1;
            let tile_pixel_count = tile_width * tile_height;
            let mut running_s = vec![0.0f32; tile_pixel_count * 3];
            let mut running_alpha = vec![0.0f32; tile_pixel_count];
            let mut running_depth_num = vec![0.0f32; tile_pixel_count];

            for offset in 0..record.count() {
                let Some(&packed_idx) = packed_indices.get(record.start() + offset) else {
                    continue;
                };
                let Some(g) = projected_cpu.get(packed_idx as usize) else {
                    continue;
                };
                let source_idx = g.source_idx as usize;
                if source_idx >= row_count
                    || !g.sigma_x.is_finite()
                    || !g.sigma_y.is_finite()
                    || g.sigma_x <= 0.0
                    || g.sigma_y <= 0.0
                {
                    continue;
                }

                for py in min_y..=max_y {
                    for px in min_x..=max_x {
                        let local_idx = (py - min_y) * tile_width + (px - min_x);
                        if (1.0 - running_alpha[local_idx]) <= 1e-4 {
                            continue;
                        }

                        let pixel_idx = py * self.render_width + px;
                        let color_idx = pixel_idx * 3;
                        let px_center = px as f32 + 0.5;
                        let py_center = py as f32 + 0.5;
                        let dx = (px_center - g.u) / g.sigma_x;
                        let dy = (py_center - g.v) / g.sigma_y;
                        let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                        let alpha_raw = kernel * g.opacity;
                        let alpha = alpha_raw.clamp(0.0, 0.99);
                        let contrib = alpha * (1.0 - running_alpha[local_idx]);
                        if contrib <= 1e-8 {
                            continue;
                        }

                        let final_depth = rendered_depth_cpu.get(pixel_idx).copied().unwrap_or(0.0);
                        let final_alpha = rendered_alpha_cpu.get(pixel_idx).copied().unwrap_or(0.0);
                        let depth_denom = final_alpha + 1e-6;
                        let final_color_r =
                            rendered_color_cpu.get(color_idx).copied().unwrap_or(0.0);
                        let final_color_g = rendered_color_cpu
                            .get(color_idx + 1)
                            .copied()
                            .unwrap_or(0.0);
                        let final_color_b = rendered_color_cpu
                            .get(color_idx + 2)
                            .copied()
                            .unwrap_or(0.0);
                        let r_r = final_color_r - running_s[local_idx * 3] - contrib * g.color[0];
                        let r_g =
                            final_color_g - running_s[local_idx * 3 + 1] - contrib * g.color[1];
                        let r_b =
                            final_color_b - running_s[local_idx * 3 + 2] - contrib * g.color[2];

                        let transmittance = 1.0 - running_alpha[local_idx];
                        let inv_one_minus_alpha = 1.0 / (1.0 - alpha).max(1e-6);
                        let dc_r = color_grads.get(color_idx).copied().unwrap_or(0.0);
                        let dc_g = color_grads.get(color_idx + 1).copied().unwrap_or(0.0);
                        let dc_b = color_grads.get(color_idx + 2).copied().unwrap_or(0.0);
                        let dl_dalpha_color =
                            (transmittance * g.color[0] - r_r * inv_one_minus_alpha) * dc_r
                                + (transmittance * g.color[1] - r_g * inv_one_minus_alpha) * dc_g
                                + (transmittance * g.color[2] - r_b * inv_one_minus_alpha) * dc_b;
                        let dd_depth = depth_grads.get(pixel_idx).copied().unwrap_or(0.0);
                        let tail_alpha = final_alpha - running_alpha[local_idx] - contrib;
                        let tail_depth_num = final_depth * depth_denom
                            - running_depth_num[local_idx]
                            - contrib * g.depth;
                        let mut dl_dalpha_depth = 0.0f32;
                        let mut dl_dalpha_alpha = 0.0f32;
                        if dd_depth != 0.0 {
                            let dnum_dalpha =
                                transmittance * g.depth - tail_depth_num * inv_one_minus_alpha;
                            let dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
                            let ddepth_dalpha = (dnum_dalpha * depth_denom
                                - final_depth * depth_denom * dalpha_dalpha)
                                / (depth_denom * depth_denom);
                            dl_dalpha_depth = dd_depth * ddepth_dalpha;
                            if loss_scales.alpha > 0.0 {
                                dl_dalpha_alpha = loss_scales.alpha * dalpha_dalpha;
                            }
                        } else if loss_scales.alpha > 0.0 {
                            let dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
                            dl_dalpha_alpha = loss_scales.alpha * dalpha_dalpha;
                        }

                        running_s[local_idx * 3] += contrib * g.color[0];
                        running_s[local_idx * 3 + 1] += contrib * g.color[1];
                        running_s[local_idx * 3 + 2] += contrib * g.color[2];
                        running_alpha[local_idx] += contrib;
                        running_depth_num[local_idx] += contrib * g.depth;

                        if alpha_raw <= 0.0 || alpha_raw >= 0.99 {
                            continue;
                        }

                        let dl_dalpha_total = dl_dalpha_color + dl_dalpha_depth + dl_dalpha_alpha;
                        let dl_dkernel = dl_dalpha_total * g.opacity;
                        let dk_ddx = kernel * (-dx);
                        let dk_ddy = kernel * (-dy);
                        if g.sigma_x.abs() >= 0.5 {
                            dl_dsigma_x[source_idx] += dl_dkernel * dk_ddx * (-dx / g.sigma_x);
                        }
                        if g.sigma_y.abs() >= 0.5 {
                            dl_dsigma_y[source_idx] += dl_dkernel * dk_ddy * (-dy / g.sigma_y);
                        }
                    }
                }
            }
        }

        let mut rotation_grads = vec![0.0f32; row_count * 4];

        // Use finite-difference gradients through the full projection chain.
        // This correctly accounts for camera rotation and projection Jacobian.
        // TODO: Implement analytical chain rule for performance (currently ~4x slower).
        for g in &projected_cpu {
            let source_idx = g.source_idx as usize;
            if source_idx >= row_count
                || (dl_dsigma_x[source_idx].abs() + dl_dsigma_y[source_idx].abs()) <= 1e-12
            {
                continue;
            }

            let raw_rotation = row_to_quaternion(
                raw_rotation_rows
                    .get(source_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );

            // Recover camera-space position from projected coordinates
            let x = (g.u - camera.cx) * g.depth / camera.fx.max(1e-6);
            let y = (g.v - camera.cy) * g.depth / camera.fy.max(1e-6);

            for component in 0..4 {
                let (d_sigma_x, d_sigma_y) = finite_difference_sigma_wrt_rotation_component(
                    x,
                    y,
                    g.depth,
                    g.scale3d,
                    raw_rotation,
                    component,
                    camera,
                );
                rotation_grads[source_idx * 4 + component] +=
                    dl_dsigma_x[source_idx] * d_sigma_x + dl_dsigma_y[source_idx] * d_sigma_y;
            }
        }

        Tensor::from_slice(&rotation_grads, (row_count, 4), &self.device)
    }

    fn parameter_grads_from_render_color_grads(
        &self,
        gaussians: &TrainableGaussians,
        projected: &ProjectedGaussians,
        render_color_grads: &Tensor,
        camera: &DiffCamera,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        if !gaussians.uses_spherical_harmonics() {
            return Ok((
                gaussians.render_color_grads_to_parameter_grads(render_color_grads)?,
                Tensor::zeros_like(gaussians.sh_rest())?,
            ));
        }

        let row_count = gaussians.len();
        let sh_rest_coeff_count = gaussians.sh_rest().dims().get(1).copied().unwrap_or(0);
        let mut sh_0_grads = vec![0.0f32; row_count * 3];
        let mut sh_rest_grads = vec![0.0f32; row_count * sh_rest_coeff_count * 3];
        let source_indices = projected.source_indices.to_vec1::<u32>()?;
        if source_indices.is_empty() {
            return Ok((
                Tensor::from_slice(&sh_0_grads, (row_count, 3), &self.device)?,
                Tensor::from_slice(
                    &sh_rest_grads,
                    (row_count, sh_rest_coeff_count, 3usize),
                    &self.device,
                )?,
            ));
        }

        let visible_grads = render_color_grads
            .index_select(&projected.source_indices, 0)?
            .to_vec2::<f32>()?;
        let visible_positions = gaussians
            .positions()
            .index_select(&projected.source_indices, 0)?
            .to_vec2::<f32>()?;
        let visible_sh_0 = gaussians
            .sh_0()
            .index_select(&projected.source_indices, 0)?
            .to_vec2::<f32>()?;
        let visible_sh_rest = if sh_rest_coeff_count > 0 {
            gaussians
                .sh_rest()
                .index_select(&projected.source_indices, 0)?
                .to_vec3::<f32>()?
        } else {
            Vec::new()
        };
        let active_degree = self.active_sh_degree.min(gaussians.sh_degree());
        let camera_center = camera_center_world(camera);

        for (visible_idx, &source_idx) in source_indices.iter().enumerate() {
            let source_idx = source_idx as usize;
            if source_idx >= row_count {
                continue;
            }
            let position = row_to_vec3(
                visible_positions
                    .get(visible_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );
            let direction = normalized_view_direction(position, camera_center);
            let basis = sh_basis_values(direction, active_degree);
            let sh_0 = row_to_vec3(
                visible_sh_0
                    .get(visible_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );
            let unclamped_rgb = sh_rgb_from_basis(
                sh_0,
                visible_sh_rest
                    .get(visible_idx)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                &basis,
            );
            let render_grad = visible_grads
                .get(visible_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);

            for channel in 0..3 {
                if unclamped_rgb[channel] <= 0.0 {
                    continue;
                }
                let grad_value = render_grad.get(channel).copied().unwrap_or(0.0);
                sh_0_grads[source_idx * 3 + channel] += basis[0] * grad_value;
                for coeff_idx in 0..sh_rest_coeff_count.min(basis.len().saturating_sub(1)) {
                    let flat_idx = (source_idx * sh_rest_coeff_count + coeff_idx) * 3 + channel;
                    sh_rest_grads[flat_idx] += basis[coeff_idx + 1] * grad_value;
                }
            }
        }

        Ok((
            Tensor::from_slice(&sh_0_grads, (row_count, 3), &self.device)?,
            Tensor::from_slice(
                &sh_rest_grads,
                (row_count, sh_rest_coeff_count, 3usize),
                &self.device,
            )?,
        ))
    }

    fn apply_backward_grads(
        &mut self,
        gaussians: &mut TrainableGaussians,
        grads: &MetalBackwardGrads,
        projected: &ProjectedGaussians,
        camera: &DiffCamera,
        effective_lr_pos: f32,
        scale_reg_grad: Option<&Tensor>,
        rotation_parameter_grads: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let (color_parameter_grads, sh_rest_parameter_grads) = self
            .parameter_grads_from_render_color_grads(gaussians, projected, &grads.colors, camera)?;
        let use_sparse_updates = self.is_litegs_mode() && self.litegs.sparse_grad;
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;

        let (beta1, beta2, eps, step) = (self.beta1, self.beta2, self.eps, self.iteration);
        if use_sparse_updates && projected.visible_count == 0 {
            return Ok(());
        }
        let sparse_row_indices = use_sparse_updates.then_some(&projected.source_indices);
        let scale_grads = if let Some(extra) = scale_reg_grad {
            grads.log_scales.broadcast_add(extra)?
        } else {
            grads.log_scales.clone()
        };

        if let Some(row_indices) = sparse_row_indices {
            adam_step_var_sparse(
                &gaussians.positions,
                &grads.positions,
                &mut adam.m_pos,
                &mut adam.v_pos,
                row_indices,
                effective_lr_pos,
                beta1,
                beta2,
                eps,
                step,
            )?;
            adam_step_var_sparse(
                &gaussians.scales,
                &scale_grads,
                &mut adam.m_scale,
                &mut adam.v_scale,
                row_indices,
                self.lr_scale,
                beta1,
                beta2,
                eps,
                step,
            )?;
        } else {
            // Use fused Adam kernel on Metal device to eliminate ~48 temp Tensor allocs per step.
            adam_step_var_fused(
                &gaussians.positions,
                &grads.positions,
                &mut adam.m_pos,
                &mut adam.v_pos,
                &mut self.runtime,
                effective_lr_pos,
                beta1,
                beta2,
                eps,
                step,
                MetalBufferSlot::AdamGradPos,
                MetalBufferSlot::AdamMPos,
                MetalBufferSlot::AdamVPos,
                MetalBufferSlot::AdamParamPos,
            )?;
            adam_step_var_fused(
                &gaussians.scales,
                &scale_grads,
                &mut adam.m_scale,
                &mut adam.v_scale,
                &mut self.runtime,
                self.lr_scale,
                beta1,
                beta2,
                eps,
                step,
                MetalBufferSlot::AdamGradScale,
                MetalBufferSlot::AdamMScale,
                MetalBufferSlot::AdamVScale,
                MetalBufferSlot::AdamParamScale,
            )?;
        }
        if let Some(rotation_grads) = rotation_parameter_grads {
            if let Some(row_indices) = sparse_row_indices {
                adam_step_var_sparse(
                    &gaussians.rotations,
                    rotation_grads,
                    &mut adam.m_rot,
                    &mut adam.v_rot,
                    row_indices,
                    self.lr_rotation,
                    beta1,
                    beta2,
                    eps,
                    step,
                )?;
            } else {
                adam_step_var(
                    &gaussians.rotations,
                    rotation_grads,
                    &mut adam.m_rot,
                    &mut adam.v_rot,
                    self.lr_rotation,
                    beta1,
                    beta2,
                    eps,
                    step,
                )?;
            }
        }
        if let Some(row_indices) = sparse_row_indices {
            adam_step_var_sparse(
                &gaussians.opacities,
                &grads.opacity_logits,
                &mut adam.m_op,
                &mut adam.v_op,
                row_indices,
                self.lr_opacity,
                beta1,
                beta2,
                eps,
                step,
            )?;
            adam_step_var_sparse(
                &gaussians.colors,
                &color_parameter_grads,
                &mut adam.m_color,
                &mut adam.v_color,
                row_indices,
                self.lr_color,
                beta1,
                beta2,
                eps,
                step,
            )?;
        } else {
            adam_step_var_fused(
                &gaussians.opacities,
                &grads.opacity_logits,
                &mut adam.m_op,
                &mut adam.v_op,
                &mut self.runtime,
                self.lr_opacity,
                beta1,
                beta2,
                eps,
                step,
                MetalBufferSlot::AdamGradOpacity,
                MetalBufferSlot::AdamMOpacity,
                MetalBufferSlot::AdamVOpacity,
                MetalBufferSlot::AdamParamOpacity,
            )?;
            adam_step_var_fused(
                &gaussians.colors,
                &color_parameter_grads,
                &mut adam.m_color,
                &mut adam.v_color,
                &mut self.runtime,
                self.lr_color,
                beta1,
                beta2,
                eps,
                step,
                MetalBufferSlot::AdamGradColor,
                MetalBufferSlot::AdamMColor,
                MetalBufferSlot::AdamVColor,
                MetalBufferSlot::AdamParamColor,
            )?;
        }
        if gaussians.uses_spherical_harmonics() {
            if let Some(row_indices) = sparse_row_indices {
                adam_step_var_sparse(
                    &gaussians.sh_rest,
                    &sh_rest_parameter_grads,
                    &mut adam.m_sh_rest,
                    &mut adam.v_sh_rest,
                    row_indices,
                    self.lr_sh_rest,
                    beta1,
                    beta2,
                    eps,
                    step,
                )?;
            } else {
                adam_step_var_fused(
                    &gaussians.sh_rest,
                    &sh_rest_parameter_grads,
                    &mut adam.m_sh_rest,
                    &mut adam.v_sh_rest,
                    &mut self.runtime,
                    self.lr_sh_rest,
                    beta1,
                    beta2,
                    eps,
                    step,
                    MetalBufferSlot::AdamGradColor,
                    MetalBufferSlot::AdamMColor,
                    MetalBufferSlot::AdamVColor,
                    MetalBufferSlot::AdamParamColor,
                )?;
            }
        }

        Ok(())
    }

    fn render(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(RenderedFrame, ProjectedGaussians, MetalRenderProfile)> {
        if gaussians.len() == 0 {
            return Ok((
                RenderedFrame {
                    color: Tensor::zeros((self.pixel_count, 3), DType::F32, &self.device)?,
                    depth: Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?,
                    alpha: Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?,
                },
                self.empty_projected_gaussians()?,
                MetalRenderProfile::default(),
            ));
        }

        // Camera will be staged inside project_gaussians, no need to stage here
        if should_profile && self.device.is_metal() {
            let render_positions = gaussians.positions().detach();
            let render_colors =
                self.render_colors_for_camera(gaussians, &render_positions, camera)?;
            let gaussian_bindings = self.runtime.bind_gaussians(gaussians, &render_colors)?;
            let _ = (
                gaussian_bindings.positions.byte_offset(),
                gaussian_bindings.positions.element_count(),
                gaussian_bindings.positions.dtype(),
                gaussian_bindings.positions.buffer()?,
                gaussian_bindings.scales.byte_offset(),
                gaussian_bindings.rotations.byte_offset(),
                gaussian_bindings.opacities.byte_offset(),
                gaussian_bindings.colors.byte_offset(),
            );
        }

        let (projected, mut profile) = self.project_gaussians(
            gaussians,
            camera,
            should_profile,
            collect_visible_indices,
            cluster_visible_mask,
        )?;
        let raster_start = Instant::now();
        let tile_bins = self.build_tile_bins(&projected)?;
        let (rendered, tile_stats, native_profile) = if self.use_native_forward {
            let (rendered, tile_stats, native_profile) =
                self.rasterize_native(&projected, &tile_bins)?;
            (rendered, tile_stats, Some(native_profile))
        } else {
            let (rendered, tile_stats) = self.rasterize(&projected, &tile_bins)?;
            (rendered, tile_stats, None)
        };
        self.synchronize_if_needed(should_profile)?;

        // Store tile_bins in projected for backward pass
        let mut projected = projected;
        projected.tile_bins = tile_bins;

        profile.rasterization = raster_start.elapsed();
        profile.active_tiles = tile_stats.active_tiles;
        profile.tile_gaussian_refs = tile_stats.tile_gaussian_refs;
        profile.max_gaussians_per_tile = tile_stats.max_gaussians_per_tile;
        if should_profile && self.device.is_metal() {
            profile.native_forward = if let Some(native_profile) = native_profile {
                let (baseline, _) = self.rasterize(&projected, &projected.tile_bins)?;
                Some(self.build_native_parity_profile(&baseline, &rendered, native_profile)?)
            } else {
                Some(self.profile_native_forward(&projected, &projected.tile_bins, &rendered)?)
            };
        }
        Ok((rendered, projected, profile))
    }

    fn project_gaussians(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        self.runtime.stage_camera(camera)?;
        let mut profile = MetalRenderProfile::default();
        profile.total_gaussians = gaussians.len();
        let projection_start = Instant::now();
        let pos = gaussians.positions().detach();
        let scales = gaussians.scales.as_tensor().detach().exp()?;
        let opacity_logits = gaussians.opacities.as_tensor().detach();
        let colors = self.render_colors_for_camera(gaussians, &pos, camera)?;
        let rotations = gaussians.rotations()?.detach();

        let px = pos.narrow(1, 0, 1)?.squeeze(1)?;
        let py = pos.narrow(1, 1, 1)?.squeeze(1)?;
        let pz = pos.narrow(1, 2, 1)?.squeeze(1)?;

        let x = px
            .affine(camera.rotation[0][0] as f64, camera.translation[0] as f64)?
            .broadcast_add(&py.affine(camera.rotation[0][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[0][2] as f64, 0.0)?)?;
        let y = px
            .affine(camera.rotation[1][0] as f64, camera.translation[1] as f64)?
            .broadcast_add(&py.affine(camera.rotation[1][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[1][2] as f64, 0.0)?)?;
        let z = px
            .affine(camera.rotation[2][0] as f64, camera.translation[2] as f64)?
            .broadcast_add(&py.affine(camera.rotation[2][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[2][2] as f64, 0.0)?)?;

        let staging_source = if self.device.is_metal() {
            ProjectionStagingSource::RuntimeBufferRead
        } else {
            ProjectionStagingSource::TensorReadback
        };
        let max_x_bound = camera.width.saturating_sub(1) as f32;
        let max_y_bound = camera.height.saturating_sub(1) as f32;

        let mut projected_cpu = Vec::with_capacity(gaussians.len());
        let mut visible_source_indices = Vec::new();
        if self.device.is_metal() {
            // Metal projection still runs on the GPU, but LiteGS mode reads the compacted
            // projection records back to the CPU for tiling and backward. Filter those rows
            // against the cluster mask so clustered rendering and sparse updates share the
            // same visible primitive set.
            let gpu_batch =
                self.runtime
                    .project_gaussians(gaussians, &colors, collect_visible_indices)?;
            visible_source_indices = gpu_batch.visible_source_indices;
            profile.visible_gaussians = gpu_batch.visible_count;
            if !self.use_native_forward || should_profile || self.is_litegs_mode() {
                let records = self.runtime.read_buffer_structs::<MetalProjectionRecord>(
                    MetalBufferSlot::ProjectionRecords,
                    gpu_batch.visible_count,
                )?;
                projected_cpu.extend(records.into_iter().map(cpu_projected_from_record));
            }
        } else {
            let x_values = self.runtime.read_tensor_flat::<f32>(&x)?;
            let y_values = self.runtime.read_tensor_flat::<f32>(&y)?;
            let z_values = self.runtime.read_tensor_flat::<f32>(&z)?;
            let scale_values = self.runtime.read_tensor_flat::<f32>(&scales)?;
            let rotation_values = self.runtime.read_tensor_flat::<f32>(&rotations)?;
            let opacity_logit_values = self.runtime.read_tensor_flat::<f32>(&opacity_logits)?;
            let color_values = self.runtime.read_tensor_flat::<f32>(&colors)?;
            for idx in 0..gaussians.len() {
                // Cluster visibility check: skip if cluster is not visible
                if let Some(mask) = cluster_visible_mask {
                    if !mask.get(idx).copied().unwrap_or(true) {
                        continue;
                    }
                }

                let z_value = x_values
                    .get(idx)
                    .zip(y_values.get(idx))
                    .zip(z_values.get(idx))
                    .map(|((_, _), z)| *z)
                    .unwrap_or(0.0);
                if !z_value.is_finite() || z_value < 1e-4 {
                    continue;
                }
                let x_value = x_values[idx];
                let y_value = y_values[idx];
                let scale3d = row_to_vec3(row_slice(&scale_values, 3, idx));
                let rotation = row_to_quaternion(row_slice(&rotation_values, 4, idx));
                let u_value = camera.fx * x_value / z_value + camera.cx;
                let v_value = camera.fy * y_value / z_value + camera.cy;
                let (raw_sigma_x, raw_sigma_y) = projected_axis_aligned_sigmas(
                    x_value,
                    y_value,
                    z_value,
                    scale3d,
                    rotation,
                    &camera.rotation,
                    camera.fx,
                    camera.fy,
                );
                if !raw_sigma_x.is_finite() || !raw_sigma_y.is_finite() {
                    continue;
                }
                let sigma_x = raw_sigma_x.clamp(0.5, 256.0);
                let sigma_y = raw_sigma_y.clamp(0.5, 256.0);
                let support_x = sigma_x * 3.0;
                let support_y = sigma_y * 3.0;
                if u_value + support_x < 0.0
                    || u_value - support_x > camera.width as f32
                    || v_value + support_y < 0.0
                    || v_value - support_y > camera.height as f32
                {
                    continue;
                }

                let opacity_logit = opacity_logit_values.get(idx).copied().unwrap_or(0.0);
                let color = row_to_vec3(row_slice(&color_values, 3, idx));
                projected_cpu.push(CpuProjectedGaussian {
                    source_idx: idx as u32,
                    u: u_value,
                    v: v_value,
                    sigma_x,
                    sigma_y,
                    raw_sigma_x,
                    raw_sigma_y,
                    depth: z_value,
                    opacity: sigmoid_scalar(opacity_logit),
                    opacity_logit,
                    scale3d,
                    color,
                    min_x: (u_value - support_x).clamp(0.0, max_x_bound),
                    max_x: (u_value + support_x).clamp(0.0, max_x_bound),
                    min_y: (v_value - support_y).clamp(0.0, max_y_bound),
                    max_y: (v_value + support_y).clamp(0.0, max_y_bound),
                });
            }
        }

        let had_projected_cpu_rows = !projected_cpu.is_empty();
        if had_projected_cpu_rows {
            filter_projected_gaussians_by_cluster_visibility(
                &mut projected_cpu,
                cluster_visible_mask,
            );
        }

        if !self.device.is_metal() || had_projected_cpu_rows {
            visible_source_indices = projected_cpu.iter().map(|g| g.source_idx).collect();
            profile.visible_gaussians = projected_cpu.len();
        }
        self.synchronize_if_needed(should_profile)?;
        profile.projection = projection_start.elapsed();

        if profile.visible_gaussians == 0 {
            let mut empty = self.empty_projected_gaussians()?;
            empty.visible_source_indices = visible_source_indices;
            empty.visible_count = profile.visible_gaussians;
            empty.staging_source = staging_source;
            return Ok((empty, profile));
        }

        let sort_start = Instant::now();
        if !self.device.is_metal() {
            projected_cpu.sort_unstable_by(|lhs, rhs| {
                lhs.depth.partial_cmp(&rhs.depth).unwrap_or(Ordering::Equal)
            });
            visible_source_indices = projected_cpu.iter().map(|g| g.source_idx).collect();
        }
        let source_indices: Vec<u32> = if self.device.is_metal() && projected_cpu.is_empty() {
            visible_source_indices.clone()
        } else {
            projected_cpu.iter().map(|g| g.source_idx).collect()
        };

        let effective_count =
            if matches!(staging_source, ProjectionStagingSource::RuntimeBufferRead) {
                profile.visible_gaussians
            } else {
                source_indices.len()
            };

        let u: Vec<f32> = projected_cpu.iter().map(|g| g.u).collect();
        let v: Vec<f32> = projected_cpu.iter().map(|g| g.v).collect();
        let sigma_x: Vec<f32> = projected_cpu.iter().map(|g| g.sigma_x).collect();
        let sigma_y: Vec<f32> = projected_cpu.iter().map(|g| g.sigma_y).collect();
        let raw_sigma_x: Vec<f32> = projected_cpu.iter().map(|g| g.raw_sigma_x).collect();
        let raw_sigma_y: Vec<f32> = projected_cpu.iter().map(|g| g.raw_sigma_y).collect();
        let depth: Vec<f32> = projected_cpu.iter().map(|g| g.depth).collect();
        let opacity: Vec<f32> = projected_cpu.iter().map(|g| g.opacity).collect();
        let opacity_logits: Vec<f32> = projected_cpu.iter().map(|g| g.opacity_logit).collect();
        let scale3d: Vec<f32> = projected_cpu.iter().flat_map(|g| g.scale3d).collect();
        let colors: Vec<f32> = projected_cpu.iter().flat_map(|g| g.color).collect();
        let min_x: Vec<f32> = projected_cpu.iter().map(|g| g.min_x).collect();
        let max_x: Vec<f32> = projected_cpu.iter().map(|g| g.max_x).collect();
        let min_y: Vec<f32> = projected_cpu.iter().map(|g| g.min_y).collect();
        let max_y: Vec<f32> = projected_cpu.iter().map(|g| g.max_y).collect();
        let projected = ProjectedGaussians {
            source_indices: if effective_count == 0 {
                Tensor::zeros((0,), DType::U32, &self.device)?
            } else if source_indices.is_empty() {
                Tensor::zeros((effective_count,), DType::U32, &self.device)?
            } else {
                Tensor::from_slice(&source_indices, source_indices.len(), &self.device)?
            },
            u: if u.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&u, u.len(), &self.device)?
            },
            v: if v.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&v, v.len(), &self.device)?
            },
            sigma_x: if sigma_x.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&sigma_x, sigma_x.len(), &self.device)?
            },
            sigma_y: if sigma_y.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&sigma_y, sigma_y.len(), &self.device)?
            },
            raw_sigma_x: if raw_sigma_x.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&raw_sigma_x, raw_sigma_x.len(), &self.device)?
            },
            raw_sigma_y: if raw_sigma_y.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&raw_sigma_y, raw_sigma_y.len(), &self.device)?
            },
            depth: if depth.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&depth, depth.len(), &self.device)?
            },
            opacity: if opacity.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&opacity, opacity.len(), &self.device)?
            },
            opacity_logits: if opacity_logits.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&opacity_logits, opacity_logits.len(), &self.device)?
            },
            scale3d: if scale3d.is_empty() {
                Tensor::zeros((effective_count, 3), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&scale3d, (effective_count, 3), &self.device)?
            },
            colors: if colors.is_empty() {
                Tensor::zeros((effective_count, 3), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&colors, (effective_count, 3), &self.device)?
            },
            min_x: if min_x.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&min_x, min_x.len(), &self.device)?
            },
            max_x: if max_x.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&max_x, max_x.len(), &self.device)?
            },
            min_y: if min_y.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&min_y, min_y.len(), &self.device)?
            },
            max_y: if max_y.is_empty() {
                Tensor::zeros((effective_count,), DType::F32, &self.device)?
            } else {
                Tensor::from_slice(&max_y, max_y.len(), &self.device)?
            },
            visible_source_indices,
            visible_count: profile.visible_gaussians,
            tile_bins: MetalTileBins::default(),
            staging_source,
        };
        self.synchronize_if_needed(should_profile)?;
        profile.sorting = sort_start.elapsed();

        Ok((projected, profile))
    }

    fn rasterize(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &MetalTileBins,
    ) -> candle_core::Result<(RenderedFrame, TileBinningStats)> {
        let mut color_acc = Tensor::zeros((self.pixel_count, 3), DType::F32, &self.device)?;
        let mut depth_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        let mut alpha_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        let tile_stats = self.tile_binning_stats(tile_bins);
        let tile_index_tensor = if tile_bins.total_assignments() == 0 {
            Tensor::zeros((0,), DType::U32, &self.device)?
        } else {
            Tensor::from_slice(
                tile_bins.packed_indices(),
                tile_bins.total_assignments(),
                &self.device,
            )?
        };

        for &tile_idx in tile_bins.active_tiles() {
            let Some(record) = tile_bins.record(tile_idx) else {
                continue;
            };
            if record.count() == 0 {
                continue;
            }

            let window = self.runtime.tile_window(tile_idx)?;
            let mut tile_color_acc =
                Tensor::zeros((window.pixel_count, 3), DType::F32, &self.device)?;
            let mut tile_depth_acc =
                Tensor::zeros((window.pixel_count,), DType::F32, &self.device)?;
            let mut tile_alpha_acc =
                Tensor::zeros((window.pixel_count,), DType::F32, &self.device)?;
            let mut tile_trans = Tensor::ones((window.pixel_count,), DType::F32, &self.device)?;

            for start in (0..record.count()).step_by(self.chunk_size) {
                let len = (record.count() - start).min(self.chunk_size);
                let chunk_indices = tile_index_tensor.narrow(0, record.start() + start, len)?;
                let alpha = self.chunk_alpha(
                    &window,
                    &projected.u.index_select(&chunk_indices, 0)?,
                    &projected.v.index_select(&chunk_indices, 0)?,
                    &projected.sigma_x.index_select(&chunk_indices, 0)?,
                    &projected.sigma_y.index_select(&chunk_indices, 0)?,
                    &projected.opacity.index_select(&chunk_indices, 0)?,
                )?;
                let (chunk_color, chunk_depth, chunk_alpha, tail_trans) = self.integrate_chunk(
                    &alpha,
                    &projected.colors.index_select(&chunk_indices, 0)?,
                    &projected.depth.index_select(&chunk_indices, 0)?,
                )?;
                let tile_trans_col = tile_trans.reshape((window.pixel_count, 1))?;
                tile_color_acc =
                    tile_color_acc.broadcast_add(&chunk_color.broadcast_mul(&tile_trans_col)?)?;
                tile_depth_acc =
                    tile_depth_acc.broadcast_add(&chunk_depth.broadcast_mul(&tile_trans)?)?;
                tile_alpha_acc =
                    tile_alpha_acc.broadcast_add(&chunk_alpha.broadcast_mul(&tile_trans)?)?;
                tile_trans = tile_trans.broadcast_mul(&tail_trans)?;
            }

            color_acc = color_acc.index_add(&window.indices, &tile_color_acc, 0)?;
            depth_acc = depth_acc.index_add(&window.indices, &tile_depth_acc, 0)?;
            alpha_acc = alpha_acc.index_add(&window.indices, &tile_alpha_acc, 0)?;
        }

        let denom = alpha_acc.broadcast_add(&Tensor::new(1e-6f32, &self.device)?)?;
        Ok((
            RenderedFrame {
                color: color_acc.clamp(0.0, 1.0)?,
                depth: depth_acc.broadcast_div(&denom)?,
                alpha: alpha_acc,
            },
            tile_stats,
        ))
    }

    fn rasterize_native(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &MetalTileBins,
    ) -> candle_core::Result<(RenderedFrame, TileBinningStats, NativeForwardProfile)> {
        if !matches!(
            projected.staging_source,
            ProjectionStagingSource::RuntimeBufferRead
        ) {
            self.stage_projected_records_from_tensors(projected)?;
        }
        let (native_frame, native_profile) = self.runtime.rasterize_forward(
            projected.visible_count,
            tile_bins,
            self.render_width,
            self.render_height,
        )?;
        let tile_stats = self.tile_binning_stats(tile_bins);
        Ok((
            RenderedFrame {
                color: native_frame.color,
                depth: native_frame.depth,
                alpha: native_frame.alpha,
            },
            tile_stats,
            native_profile,
        ))
    }

    fn chunk_alpha(
        &self,
        window: &ChunkPixelWindow,
        u: &Tensor,
        v: &Tensor,
        sigma_x: &Tensor,
        sigma_y: &Tensor,
        opacity: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let len = u.dim(0)?;
        let dx = window
            .pixel_x
            .broadcast_sub(&u.reshape((len, 1))?)?
            .broadcast_div(&sigma_x.reshape((len, 1))?)?;
        let dy = window
            .pixel_y
            .broadcast_sub(&v.reshape((len, 1))?)?
            .broadcast_div(&sigma_y.reshape((len, 1))?)?;
        let exponent = dx.sqr()?.broadcast_add(&dy.sqr()?)?.affine(-0.5, 0.0)?;
        exponent
            .exp()?
            .broadcast_mul(&opacity.reshape((len, 1))?)?
            .clamp(0.0, 0.99)
    }

    fn build_tile_bins(
        &mut self,
        projected: &ProjectedGaussians,
    ) -> candle_core::Result<MetalTileBins> {
        if self.device.is_metal() {
            if !matches!(
                projected.staging_source,
                ProjectionStagingSource::RuntimeBufferRead
            ) {
                self.stage_projected_records_from_tensors(projected)?;
            }
            return self.runtime.build_tile_bins_gpu(projected.visible_count);
        }
        let min_x_values = projected.min_x.to_vec1::<f32>()?;
        let max_x_values = projected.max_x.to_vec1::<f32>()?;
        let min_y_values = projected.min_y.to_vec1::<f32>()?;
        let max_y_values = projected.max_y.to_vec1::<f32>()?;
        self.runtime
            .build_tile_bins(&min_x_values, &max_x_values, &min_y_values, &max_y_values)
    }

    fn profile_native_forward(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &MetalTileBins,
        baseline: &RenderedFrame,
    ) -> candle_core::Result<NativeParityProfile> {
        if !matches!(
            projected.staging_source,
            ProjectionStagingSource::RuntimeBufferRead
        ) {
            self.stage_projected_records_from_tensors(projected)?;
        }
        let (native_frame, native_profile) = self.runtime.rasterize_forward(
            projected.visible_count,
            tile_bins,
            self.render_width,
            self.render_height,
        )?;
        self.build_native_parity_profile(
            baseline,
            &RenderedFrame {
                color: native_frame.color,
                depth: native_frame.depth,
                alpha: native_frame.alpha,
            },
            native_profile,
        )
    }

    fn build_native_parity_profile(
        &self,
        baseline: &RenderedFrame,
        native: &RenderedFrame,
        native_profile: NativeForwardProfile,
    ) -> candle_core::Result<NativeParityProfile> {
        let color_max_abs = baseline
            .color
            .sub(&native.color)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        let depth_max_abs = baseline
            .depth
            .sub(&native.depth)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        let alpha_max_abs = baseline
            .alpha
            .sub(&native.alpha)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        Ok(NativeParityProfile {
            setup: native_profile.setup,
            staging: native_profile.staging,
            kernel: native_profile.kernel,
            total: native_profile.total,
            color_max_abs,
            depth_max_abs,
            alpha_max_abs,
        })
    }

    fn stage_projected_records_from_tensors(
        &mut self,
        projected: &ProjectedGaussians,
    ) -> candle_core::Result<()> {
        let source_indices = projected.source_indices.to_vec1::<u32>()?;
        let u = projected.u.to_vec1::<f32>()?;
        let v = projected.v.to_vec1::<f32>()?;
        let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
        let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
        let raw_sigma_x = projected.raw_sigma_x.to_vec1::<f32>()?;
        let raw_sigma_y = projected.raw_sigma_y.to_vec1::<f32>()?;
        let depth = projected.depth.to_vec1::<f32>()?;
        let opacity = projected.opacity.to_vec1::<f32>()?;
        let opacity_logits = projected.opacity_logits.to_vec1::<f32>()?;
        let scale3d = projected.scale3d.to_vec2::<f32>()?;
        let colors = projected.colors.to_vec2::<f32>()?;
        let min_x = projected.min_x.to_vec1::<f32>()?;
        let max_x = projected.max_x.to_vec1::<f32>()?;
        let min_y = projected.min_y.to_vec1::<f32>()?;
        let max_y = projected.max_y.to_vec1::<f32>()?;
        let mut records = Vec::with_capacity(source_indices.len());
        for idx in 0..source_indices.len() {
            records.push(MetalProjectionRecord {
                source_idx: source_indices[idx],
                visible: 1,
                u: u[idx],
                v: v[idx],
                sigma_x: sigma_x[idx],
                sigma_y: sigma_y[idx],
                raw_sigma_x: raw_sigma_x[idx],
                raw_sigma_y: raw_sigma_y[idx],
                depth: depth[idx],
                opacity: opacity[idx],
                opacity_logit: opacity_logits[idx],
                scale_x: scale3d[idx][0],
                scale_y: scale3d[idx][1],
                scale_z: scale3d[idx][2],
                color_r: colors[idx][0],
                color_g: colors[idx][1],
                color_b: colors[idx][2],
                min_x: min_x[idx],
                max_x: max_x[idx],
                min_y: min_y[idx],
                max_y: max_y[idx],
            });
        }
        self.runtime
            .ensure_projection_record_buffer(records.len())?;
        self.runtime.write_projection_records(&records)
    }

    fn tile_binning_stats(&self, tile_bins: &MetalTileBins) -> TileBinningStats {
        TileBinningStats {
            active_tiles: tile_bins.active_tile_count(),
            tile_gaussian_refs: tile_bins.total_assignments(),
            max_gaussians_per_tile: tile_bins.max_gaussians_per_tile(),
        }
    }

    fn integrate_chunk(
        &self,
        alpha: &Tensor,
        colors: &Tensor,
        depth: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let len = alpha.dim(0)?;
        let pixel_count = alpha.dim(1)?;
        let zero_row = Tensor::zeros((1, pixel_count), DType::F32, &self.device)?;
        let inclusive = Tensor::ones_like(alpha)?
            .broadcast_sub(alpha)?
            .clamp(1e-4, 1.0)?
            .log()?
            .cumsum(0)?;
        let exclusive = if len == 1 {
            zero_row
        } else {
            Tensor::cat(&[&zero_row, &inclusive.narrow(0, 0, len - 1)?], 0)?
        };
        let local_contrib = alpha.broadcast_mul(&exclusive.exp()?)?;
        let local_contrib_t = local_contrib.t()?;
        let chunk_color = local_contrib_t.matmul(colors)?;
        let chunk_depth = local_contrib_t
            .matmul(&depth.reshape((len, 1))?)?
            .squeeze(1)?;
        let chunk_alpha = local_contrib.sum(0)?;
        let tail_trans = inclusive.get_on_dim(0, len - 1)?.exp()?;
        Ok((chunk_color, chunk_depth, chunk_alpha, tail_trans))
    }

    fn synchronize_if_needed(&self, should_profile: bool) -> candle_core::Result<()> {
        if should_profile {
            self.device.synchronize()?;
        }
        Ok(())
    }
}

pub fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let start = Instant::now();
    store_last_metal_training_telemetry(None);
    let device = crate::require_metal_device()?;
    let mut effective_config = effective_metal_config(config);
    let mut loaded = load_training_data(dataset, &effective_config, &device)?;

    if loaded.initial_map.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }

    let mut trainer = MetalTrainer::new(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        &effective_config,
        device,
    )?;
    let memory_budget = training_memory_budget(config);
    let frame_count = loaded.cameras.len();
    log::info!(
        "MetalTrainer preflight | gaussians={} | frames={} | pixels={} | chunk_size={} | estimated_peak≈{:.1} GiB | budget={} | dominant={}",
        loaded.initial_map.len(),
        frame_count,
        trainer.pixel_count,
        trainer.chunk_size,
        bytes_to_gib(
            estimate_peak_memory_with_source_pixels(
                loaded.initial_map.len(),
                trainer.pixel_count,
                trainer.source_pixel_count,
                frame_count,
                trainer.chunk_size,
            )
            .total_bytes()
        ),
        memory_budget.describe(),
        estimate_peak_memory_with_source_pixels(
            loaded.initial_map.len(),
            trainer.pixel_count,
            trainer.source_pixel_count,
            frame_count,
            trainer.chunk_size,
        )
        .top_components_summary(3),
    );
    let skip_memory_guard = std::env::var_os("RUSTGS_SKIP_METAL_MEMORY_GUARD").is_some();
    trainer.topology_memory_budget = if skip_memory_guard {
        None
    } else {
        Some(memory_budget)
    };
    let affordable_cap = affordable_initial_gaussian_cap(
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_map.len()),
        trainer.pixel_count,
        trainer.source_pixel_count,
        frame_count,
        trainer.chunk_size,
        &memory_budget,
    );
    if !skip_memory_guard && affordable_cap > 0 && loaded.initial_map.len() > affordable_cap {
        let initial_cap =
            preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
        log::warn!(
            "MetalTrainer preflight lowered initial_gaussians from {} to {} for this run to fit the safe memory budget using even coverage downsampling. Growth budget remains capped at {}. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to keep the larger initialization.",
            loaded.initial_map.len(),
            initial_cap,
            affordable_cap,
        );
        downsample_gaussian_map_evenly(&mut loaded.initial_map, initial_cap);
        effective_config.max_initial_gaussians = initial_cap;
    }
    trainer.max_gaussian_budget = if skip_memory_guard {
        if effective_config.training_profile == TrainingProfile::LiteGsMacV1 {
            effective_config
                .litegs
                .target_primitives
                .max(loaded.initial_map.len())
        } else {
            effective_config
                .max_initial_gaussians
                .max(loaded.initial_map.len())
        }
    } else {
        let profile_cap = if effective_config.training_profile == TrainingProfile::LiteGsMacV1 {
            effective_config
                .litegs
                .target_primitives
                .max(loaded.initial_map.len())
        } else {
            affordable_cap.max(loaded.initial_map.len())
        };
        profile_cap.min(affordable_cap.max(loaded.initial_map.len()))
    };
    let estimated_peak = estimate_peak_memory_with_source_pixels(
        loaded.initial_map.len(),
        trainer.pixel_count,
        trainer.source_pixel_count,
        frame_count,
        trainer.chunk_size,
    );
    match assess_memory_estimate(&estimated_peak, &memory_budget) {
        MetalMemoryDecision::Allow => {
            let headroom = memory_budget
                .safe_bytes
                .saturating_sub(estimated_peak.total_bytes());
            log::info!(
                "MetalTrainer preflight passed with {:.1} GiB headroom",
                bytes_to_gib(headroom)
            );
        }
        MetalMemoryDecision::Warn => {
            log::warn!(
                "MetalTrainer preflight is close to the safe memory budget; recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block if skip_memory_guard => {
            log::warn!(
                "MetalTrainer preflight exceeds the safe memory budget but RUSTGS_SKIP_METAL_MEMORY_GUARD=1 is set; continuing anyway. Recommendations: {}",
                estimated_peak.recommendations().join("; ")
            );
        }
        MetalMemoryDecision::Block => {
            return Err(TrainingError::TrainingFailed(format!(
                "metal backend is estimated to need about {:.1} GiB for a single training step, above the safe budget of {}. Dominant terms: {}. Recommendations: {}. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to bypass this guard.",
                bytes_to_gib(estimated_peak.total_bytes()),
                memory_budget.describe(),
                estimated_peak.top_components_summary(3),
                estimated_peak.recommendations().join("; ")
            )));
        }
    }
    let mut gaussians = trainable_from_map(&loaded.initial_map, &trainer.device, config)?;
    trainer.scene_extent = estimate_scene_extent(&loaded.initial_map);

    // Initialize pose embeddings if learnable_viewproj is enabled
    if config.litegs.learnable_viewproj && config.litegs.lr_pose > 0.0 {
        use crate::training::pose_embedding::PoseEmbeddings;
        let pose_embeddings =
            PoseEmbeddings::from_dataset(&dataset.poses, config.litegs.lr_pose, &trainer.device)?;
        log::info!(
            "MetalTrainer initialized {} learnable camera poses with lr={}",
            pose_embeddings.len(),
            config.litegs.lr_pose
        );
        trainer.pose_embeddings = Some(pose_embeddings);
    }

    // Initialize cluster assignment if cluster_size > 0
    if config.litegs.cluster_size > 0 {
        use crate::training::clustering::ClusterAssignment;
        let positions: Vec<[f32; 3]> = loaded
            .initial_map
            .gaussians()
            .iter()
            .map(|g| g.position.into())
            .collect();
        let assignment = ClusterAssignment::assign_spatial_hash(
            &positions,
            config.litegs.cluster_size,
            trainer.scene_extent,
        );
        log::info!(
            "MetalTrainer initialized {} spatial clusters with target size {}",
            assignment.num_clusters,
            config.litegs.cluster_size
        );
        trainer.cluster_assignment = Some(assignment);
    }

    if gaussians.len() == 0 {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }
    let frames = trainer.prepare_frames(&loaded)?;
    let stats = trainer.train(&mut gaussians, &frames, config.iterations)?;
    let trained_map = map_from_trainable(&gaussians)?;

    log::info!(
        "Metal backend complete in {:.2}s | frames={} | render={}x{} | initial_gaussians={} | final_gaussians={} | final_loss_mean={:.6} | last_step_loss={:.6}",
        start.elapsed().as_secs_f64(),
        dataset.poses.len(),
        trainer.render_width,
        trainer.render_height,
        loaded.initial_map.len(),
        trained_map.len(),
        stats.final_loss,
        stats.final_step_loss,
    );
    store_last_metal_training_telemetry(Some(stats.telemetry.clone()));

    Ok(trained_map)
}

fn effective_metal_config(config: &TrainingConfig) -> TrainingConfig {
    let mut effective = config.clone();
    if effective.training_profile == TrainingProfile::LiteGsMacV1
        && (effective.lr_opacity - TrainingConfig::default().lr_opacity).abs() < f32::EPSILON
    {
        effective.lr_opacity = 0.025;
    }
    if effective.training_profile != TrainingProfile::LiteGsMacV1 && effective.lr_rotation != 0.0 {
        log::warn!(
            "Legacy Metal training still freezes Gaussian rotations; overriding lr_rotation from {} to 0.0 for this run.",
            effective.lr_rotation
        );
        effective.lr_rotation = 0.0;
    }
    effective
}

pub fn estimate_chunk_capacity(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<ChunkCapacityEstimate, TrainingError> {
    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training dataset does not contain any poses".to_string(),
        ));
    }

    let effective_config = effective_metal_config(config);
    let requested_initial_gaussians = if dataset.initial_points.is_empty() {
        effective_config.max_initial_gaussians.max(1)
    } else {
        dataset
            .initial_points
            .len()
            .min(effective_config.max_initial_gaussians.max(1))
            .max(1)
    };
    let frame_count = dataset.poses.len();
    let (render_width, render_height) = scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        effective_config.metal_render_scale,
    );
    let pixel_count = render_width.saturating_mul(render_height);
    let source_pixel_count =
        (dataset.intrinsics.width as usize).saturating_mul(dataset.intrinsics.height as usize);
    let requested_budget_bytes = gib_to_bytes(config.chunk_budget_gb);
    let effective_budget =
        resolve_chunk_memory_budget(requested_budget_bytes, detect_metal_memory_budget());
    let estimate = estimate_peak_memory_with_source_pixels(
        requested_initial_gaussians,
        pixel_count,
        source_pixel_count,
        frame_count,
        effective_config.metal_gaussian_chunk_size,
    );
    let affordable_initial_gaussians = affordable_initial_gaussian_cap(
        requested_initial_gaussians,
        pixel_count,
        source_pixel_count,
        frame_count,
        effective_config.metal_gaussian_chunk_size,
        &effective_budget,
    );
    let disposition = if affordable_initial_gaussians < requested_initial_gaussians {
        ChunkCapacityDisposition::NeedsSubdivisionOrDegradation
    } else {
        ChunkCapacityDisposition::FitsBudget
    };
    let mut recommendations = estimate
        .recommendations()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if disposition == ChunkCapacityDisposition::NeedsSubdivisionOrDegradation {
        recommendations.insert(
            0,
            format!(
                "subdivide the chunk or degrade it to at most {} initial gaussians",
                affordable_initial_gaussians.max(1)
            ),
        );
    }

    Ok(ChunkCapacityEstimate {
        requested_initial_gaussians,
        affordable_initial_gaussians,
        frame_count,
        render_width,
        render_height,
        estimated_peak_bytes: estimate.total_bytes(),
        requested_budget_bytes,
        effective_budget_bytes: effective_budget.safe_bytes,
        physical_memory_bytes: effective_budget.physical_bytes,
        disposition,
        dominant_components: estimate.top_components_summary(3),
        recommendations,
    })
}

fn training_memory_budget(config: &TrainingConfig) -> MetalMemoryBudget {
    if config.chunked_training {
        resolve_chunk_memory_budget(
            gib_to_bytes(config.chunk_budget_gb),
            detect_metal_memory_budget(),
        )
    } else {
        detect_metal_memory_budget()
    }
}

fn affordable_initial_gaussian_cap(
    requested_cap: usize,
    pixel_count: usize,
    source_pixel_count: usize,
    frame_count: usize,
    chunk_size: usize,
    memory_budget: &MetalMemoryBudget,
) -> usize {
    let requested_cap = requested_cap.max(1);
    if assess_memory_estimate(
        &estimate_peak_memory_with_source_pixels(
            requested_cap,
            pixel_count,
            source_pixel_count,
            frame_count,
            chunk_size,
        ),
        memory_budget,
    ) != MetalMemoryDecision::Block
    {
        return requested_cap;
    }

    let mut low = 0usize;
    let mut high = requested_cap;
    while low < high {
        let mid = low + (high - low + 1) / 2;
        let decision = assess_memory_estimate(
            &estimate_peak_memory_with_source_pixels(
                mid,
                pixel_count,
                source_pixel_count,
                frame_count,
                chunk_size,
            ),
            memory_budget,
        );
        if decision == MetalMemoryDecision::Block {
            high = mid - 1;
        } else {
            low = mid;
        }
    }
    low
}

fn preflight_initial_gaussian_cap(
    training_profile: TrainingProfile,
    affordable_cap: usize,
) -> usize {
    if training_profile != TrainingProfile::LiteGsMacV1 || affordable_cap == 0 {
        return affordable_cap;
    }

    // Keep a small densification budget without discarding a large fraction of
    // the even-sampled initialization. The previous 20% reserve collapsed the
    // TUM fallback init from 552 to 442, which preserved growth but hurt PSNR.
    let reserved_headroom = affordable_cap
        .saturating_div(20)
        .clamp(16, 64)
        .min(affordable_cap.saturating_sub(1));
    affordable_cap.saturating_sub(reserved_headroom).max(1)
}

#[cfg(test)]
fn estimate_peak_memory(
    num_gaussians: usize,
    pixel_count: usize,
    frame_count: usize,
    chunk_size: usize,
) -> MetalMemoryEstimate {
    estimate_peak_memory_with_source_pixels(
        num_gaussians,
        pixel_count,
        pixel_count,
        frame_count,
        chunk_size,
    )
}

fn estimate_peak_memory_with_source_pixels(
    num_gaussians: usize,
    pixel_count: usize,
    source_pixel_count: usize,
    frame_count: usize,
    chunk_size: usize,
) -> MetalMemoryEstimate {
    let num_gaussians = num_gaussians as u64;
    let pixel_count = pixel_count as u64;
    let source_pixel_count = source_pixel_count as u64;
    let frame_count = frame_count as u64;
    let chunk_size = chunk_size.max(1) as u64;
    let gaussian_state_bytes = num_gaussians.saturating_mul(METAL_GAUSSIAN_STATE_BYTES);
    let full_res_frame_bytes = frame_count
        .saturating_mul(source_pixel_count)
        .saturating_mul(METAL_SOURCE_FRAME_BYTES_PER_PIXEL);
    let resized_cpu_target_bytes = frame_count
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_RESIZED_CPU_TARGET_BYTES_PER_PIXEL);
    let gpu_target_bytes = frame_count
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_GPU_TARGET_BYTES_PER_PIXEL);
    let frame_bytes = full_res_frame_bytes
        .saturating_add(resized_cpu_target_bytes)
        .saturating_add(gpu_target_bytes);
    let pixel_state_bytes = pixel_count.saturating_mul(METAL_PIXEL_STATE_BYTES_PER_PIXEL);
    let projection_bytes = num_gaussians.saturating_mul(METAL_PROJECTED_BYTES_PER_GAUSSIAN);
    let chunk_workspace_bytes = chunk_size
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_CHUNK_WORKSPACE_BYTES_PER_GAUSSIAN_PIXEL);
    let retained_graph_bytes = num_gaussians
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_RETAINED_GRAPH_BYTES_PER_GAUSSIAN_PIXEL);
    let subtotal = gaussian_state_bytes
        .saturating_add(frame_bytes)
        .saturating_add(pixel_state_bytes)
        .saturating_add(projection_bytes)
        .saturating_add(chunk_workspace_bytes)
        .saturating_add(retained_graph_bytes);
    let safety_margin_bytes = apply_ratio(
        subtotal,
        METAL_ESTIMATE_SAFETY_NUMERATOR,
        METAL_ESTIMATE_SAFETY_DENOMINATOR,
    )
    .max(METAL_ESTIMATE_MIN_SAFETY_BYTES);

    MetalMemoryEstimate {
        gaussian_state_bytes,
        frame_bytes,
        pixel_state_bytes,
        projection_bytes,
        chunk_workspace_bytes,
        retained_graph_bytes,
        safety_margin_bytes,
    }
}

fn assess_memory_estimate(
    estimate: &MetalMemoryEstimate,
    budget: &MetalMemoryBudget,
) -> MetalMemoryDecision {
    let total = estimate.total_bytes();
    if total > budget.safe_bytes {
        return MetalMemoryDecision::Block;
    }
    if total.saturating_mul(METAL_WARN_BUDGET_DENOMINATOR)
        >= budget
            .safe_bytes
            .saturating_mul(METAL_WARN_BUDGET_NUMERATOR)
    {
        return MetalMemoryDecision::Warn;
    }
    MetalMemoryDecision::Allow
}

fn detect_metal_memory_budget() -> MetalMemoryBudget {
    let physical_bytes = detect_physical_memory_bytes();
    let safe_bytes = physical_bytes
        .map(|bytes| {
            apply_ratio(
                bytes,
                METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
                METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
            )
        })
        .map(|bytes| bytes.min(DEFAULT_METAL_MEMORY_BUDGET_BYTES))
        .unwrap_or(DEFAULT_METAL_MEMORY_BUDGET_BYTES);

    MetalMemoryBudget {
        safe_bytes,
        physical_bytes,
    }
}

fn resolve_chunk_memory_budget(
    requested_budget_bytes: u64,
    system_budget: MetalMemoryBudget,
) -> MetalMemoryBudget {
    let safe_bytes = requested_budget_bytes
        .max(1)
        .min(system_budget.safe_bytes.max(1));
    MetalMemoryBudget {
        safe_bytes,
        physical_bytes: system_budget.physical_bytes,
    }
}

fn detect_physical_memory_bytes() -> Option<u64> {
    #[allow(unsafe_code)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages <= 0 || page_size <= 0 {
            return None;
        }
        Some(((pages as u128).saturating_mul(page_size as u128)).min(u64::MAX as u128) as u64)
    }
}

fn apply_ratio(bytes: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return bytes;
    }
    ((bytes as u128)
        .saturating_mul(numerator as u128)
        .checked_div(denominator as u128)
        .unwrap_or(u128::MAX))
    .min(u64::MAX as u128) as u64
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024f64 / 1024f64 / 1024f64
}

fn gib_to_bytes(gib: f32) -> u64 {
    if !gib.is_finite() || gib <= 0.0 {
        return 0;
    }
    ((gib as f64) * 1024f64 * 1024f64 * 1024f64)
        .round()
        .clamp(0.0, u64::MAX as f64) as u64
}

fn format_memory(bytes: u64) -> String {
    if bytes >= GIB {
        format!("{:.1} GiB", bytes_to_gib(bytes))
    } else {
        format!("{:.0} MiB", bytes as f64 / 1024f64 / 1024f64)
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn should_profile_iteration(profile_steps: bool, profile_interval: usize, iter: usize) -> bool {
    profile_steps && (iter < 5 || iter % profile_interval.max(1) == 0)
}

fn downsample_gaussian_map_evenly(map: &mut GaussianMap, target_count: usize) {
    let gaussians = map.gaussians();
    if target_count == 0 || gaussians.len() <= target_count {
        return;
    }

    let len = gaussians.len();
    let mut sampled = Vec::with_capacity(target_count);
    for out_idx in 0..target_count {
        let src_idx = out_idx.saturating_mul(len) / target_count;
        sampled.push(gaussians[src_idx.min(len.saturating_sub(1))].clone());
    }

    *map = GaussianMap::from_gaussians(sampled);
    map.update_states();
}

/// Fused Adam update using a single Metal compute kernel.
/// Eliminates ~12 intermediate Tensor allocations per parameter group.
/// Falls back to the tensor-op path on CPU device.
fn adam_step_var_fused(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    runtime: &mut MetalRuntime,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
    grad_slot: MetalBufferSlot,
    m_slot: MetalBufferSlot,
    v_slot: MetalBufferSlot,
    param_slot: MetalBufferSlot,
) -> candle_core::Result<()> {
    if !var.as_tensor().device().is_metal() {
        return adam_step_var(var, grad, m, v, lr, beta1, beta2, eps, step);
    }

    let element_count = grad.elem_count();

    // Stage current m, v, param and grad into named Metal buffers
    let grad_flat = runtime.read_tensor_flat::<f32>(grad)?;
    let m_flat = runtime.read_tensor_flat::<f32>(m)?;
    let v_flat = runtime.read_tensor_flat::<f32>(v)?;
    let param_flat = runtime.read_tensor_flat::<f32>(var.as_tensor())?;

    let shape = grad.shape().clone();
    runtime.ensure_buffer(grad_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(m_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(v_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.ensure_buffer(param_slot, element_count * std::mem::size_of::<f32>())?;
    runtime.write_slice(grad_slot, &grad_flat)?;
    runtime.write_slice(m_slot, &m_flat)?;
    runtime.write_slice(v_slot, &v_flat)?;
    runtime.write_slice(param_slot, &param_flat)?;

    // Dispatch fused kernel — no synchronize, will flush on read
    runtime.adam_step_fused(
        param_slot,
        grad_slot,
        m_slot,
        v_slot,
        element_count,
        lr,
        beta1,
        beta2,
        eps,
        step,
    )?;

    // Read back updated m, v, params from GPU
    runtime.device.synchronize()?;
    let new_m_flat = runtime.read_buffer_structs::<f32>(m_slot, element_count)?;
    let new_v_flat = runtime.read_buffer_structs::<f32>(v_slot, element_count)?;
    let new_param_flat = runtime.read_buffer_structs::<f32>(param_slot, element_count)?;

    *m = Tensor::from_slice(&new_m_flat, shape.clone(), var.as_tensor().device())?;
    *v = Tensor::from_slice(&new_v_flat, shape.clone(), var.as_tensor().device())?;
    var.set(&Tensor::from_slice(
        &new_param_flat,
        shape,
        var.as_tensor().device(),
    )?)?;
    Ok(())
}

fn adam_step_var(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<()> {
    let (new_param, new_m, new_v) =
        adam_updated_tensors(var.as_tensor(), grad, m, v, lr, beta1, beta2, eps, step)?;
    *m = new_m;
    *v = new_v;
    var.set(&new_param)?;
    Ok(())
}

fn adam_step_var_sparse(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    row_indices: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<()> {
    let grad_rows = grad.index_select(row_indices, 0)?;
    if grad_rows.dim(0)? == 0 {
        return Ok(());
    }

    let m_rows = m.index_select(row_indices, 0)?;
    let v_rows = v.index_select(row_indices, 0)?;
    let param_rows = var.as_tensor().index_select(row_indices, 0)?;
    let (new_param_rows, new_m_rows, new_v_rows) = adam_updated_tensors(
        &param_rows,
        &grad_rows,
        &m_rows,
        &v_rows,
        lr,
        beta1,
        beta2,
        eps,
        step,
    )?;

    *m = m.index_add(row_indices, &new_m_rows.broadcast_sub(&m_rows)?, 0)?;
    *v = v.index_add(row_indices, &new_v_rows.broadcast_sub(&v_rows)?, 0)?;
    let updated_params =
        var.as_tensor()
            .index_add(row_indices, &new_param_rows.broadcast_sub(&param_rows)?, 0)?;
    var.set(&updated_params)?;
    Ok(())
}

fn adam_updated_tensors(
    param: &Tensor,
    grad: &Tensor,
    m: &Tensor,
    v: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let new_m = m
        .affine(beta1 as f64, 0.0)?
        .broadcast_add(&grad.affine((1.0 - beta1) as f64, 0.0)?)?;
    let new_v = v
        .affine(beta2 as f64, 0.0)?
        .broadcast_add(&grad.sqr()?.affine((1.0 - beta2) as f64, 0.0)?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let m_hat = new_m.affine(1.0 / bc1 as f64, 0.0)?;
    let v_hat = new_v.affine(1.0 / bc2 as f64, 0.0)?;
    let denom = v_hat
        .sqrt()?
        .broadcast_add(&Tensor::new(eps, param.device())?)?;
    let update = m_hat.broadcast_div(&denom)?.affine(lr as f64, 0.0)?;
    let new_param = param.sub(&update)?;
    Ok((new_param, new_m, new_v))
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
}

fn flatten_3d(rows: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    rows.into_iter().flatten().flatten().collect()
}

fn row_slice(values: &[f32], width: usize, idx: usize) -> &[f32] {
    let start = idx.saturating_mul(width);
    let end = start.saturating_add(width);
    values.get(start..end).unwrap_or(&[])
}

fn gather_rows(source: &[f32], row_width: usize, origins: &[Option<usize>]) -> Vec<f32> {
    let mut gathered = Vec::with_capacity(origins.len() * row_width);
    for origin in origins {
        match origin {
            Some(idx) => {
                let start = idx.saturating_mul(row_width);
                let end = start.saturating_add(row_width).min(source.len());
                gathered.extend_from_slice(&source[start..end]);
                if end - start < row_width {
                    gathered.resize(gathered.len() + (row_width - (end - start)), 0.0);
                }
            }
            None => gathered.resize(gathered.len() + row_width, 0.0),
        }
    }
    gathered
}

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn inverse_sigmoid_tensor(values: &Tensor) -> candle_core::Result<Tensor> {
    let one = Tensor::ones_like(values)?;
    values.broadcast_div(&one.sub(values)?)?.log()
}

fn camera_center_world(camera: &DiffCamera) -> [f32; 3] {
    [
        -(camera.rotation[0][0] * camera.translation[0]
            + camera.rotation[1][0] * camera.translation[1]
            + camera.rotation[2][0] * camera.translation[2]),
        -(camera.rotation[0][1] * camera.translation[0]
            + camera.rotation[1][1] * camera.translation[1]
            + camera.rotation[2][1] * camera.translation[2]),
        -(camera.rotation[0][2] * camera.translation[0]
            + camera.rotation[1][2] * camera.translation[1]
            + camera.rotation[2][2] * camera.translation[2]),
    ]
}

fn view_directions_for_camera(
    positions: &Tensor,
    camera: &DiffCamera,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let camera_center = Tensor::from_slice(&camera_center_world(camera), (1, 3), device)?;
    let dirs = positions.broadcast_sub(&camera_center)?;
    let norms = dirs
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-6, f32::MAX)?
        .reshape((positions.dim(0)?, 1))?;
    dirs.broadcast_div(&norms)
}

fn normalized_view_direction(position: [f32; 3], camera_center: [f32; 3]) -> [f32; 3] {
    let dx = position[0] - camera_center[0];
    let dy = position[1] - camera_center[1];
    let dz = position[2] - camera_center[2];
    let norm = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
    [dx / norm, dy / norm, dz / norm]
}

fn sh_basis_values(direction: [f32; 3], degree: usize) -> Vec<f32> {
    let x = direction[0];
    let y = direction[1];
    let z = direction[2];
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    let mut basis = vec![SH_C0];
    if degree > 0 {
        basis.push(-SH_C1 * y);
        basis.push(SH_C1 * z);
        basis.push(-SH_C1 * x);
    }
    if degree > 1 {
        basis.push(SH_C2[0] * xy);
        basis.push(SH_C2[1] * yz);
        basis.push(SH_C2[2] * (2.0 * zz - xx - yy));
        basis.push(SH_C2[3] * xz);
        basis.push(SH_C2[4] * (xx - yy));
    }
    if degree > 2 {
        basis.push(SH_C3[0] * y * (3.0 * xx - yy));
        basis.push(SH_C3[1] * xy * z);
        basis.push(SH_C3[2] * y * (4.0 * zz - xx - yy));
        basis.push(SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
        basis.push(SH_C3[4] * x * (4.0 * zz - xx - yy));
        basis.push(SH_C3[5] * z * (xx - yy));
        basis.push(SH_C3[6] * x * (xx - 3.0 * yy));
    }
    if degree > 3 {
        basis.push(SH_C4[0] * xy * (xx - yy));
        basis.push(SH_C4[1] * yz * (3.0 * xx - yy));
        basis.push(SH_C4[2] * xy * (7.0 * zz - 1.0));
        basis.push(SH_C4[3] * yz * (7.0 * zz - 3.0));
        basis.push(SH_C4[4] * (zz * (35.0 * zz - 30.0) + 3.0));
        basis.push(SH_C4[5] * xz * (7.0 * zz - 3.0));
        basis.push(SH_C4[6] * (xx - yy) * (7.0 * zz - 1.0));
        basis.push(SH_C4[7] * xz * (xx - 3.0 * yy));
        basis.push(SH_C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)));
    }
    basis
}

fn sh_rgb_from_basis(sh_0: [f32; 3], sh_rest: &[Vec<f32>], basis: &[f32]) -> [f32; 3] {
    let mut rgb = [
        0.5 + basis[0] * sh_0[0],
        0.5 + basis[0] * sh_0[1],
        0.5 + basis[0] * sh_0[2],
    ];
    for (coeff_idx, basis_value) in basis.iter().enumerate().skip(1) {
        let coeff_row = sh_rest.get(coeff_idx - 1).map(Vec::as_slice).unwrap_or(&[]);
        rgb[0] += basis_value * coeff_row.first().copied().unwrap_or(0.0);
        rgb[1] += basis_value * coeff_row.get(1).copied().unwrap_or(0.0);
        rgb[2] += basis_value * coeff_row.get(2).copied().unwrap_or(0.0);
    }
    rgb
}

fn estimate_scene_extent(map: &GaussianMap) -> f32 {
    if map.is_empty() {
        return 1.0;
    }

    let mut center = [0.0f32; 3];
    for gaussian in map.gaussians() {
        center[0] += gaussian.position.x;
        center[1] += gaussian.position.y;
        center[2] += gaussian.position.z;
    }
    let inv = 1.0 / map.len().max(1) as f32;
    center[0] *= inv;
    center[1] *= inv;
    center[2] *= inv;

    let mut max_dist = 0.0f32;
    for gaussian in map.gaussians() {
        let dx = gaussian.position.x - center[0];
        let dy = gaussian.position.y - center[1];
        let dz = gaussian.position.z - center[2];
        max_dist = max_dist.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    max_dist.max(1e-3)
}

fn cpu_projected_from_record(record: MetalProjectionRecord) -> CpuProjectedGaussian {
    CpuProjectedGaussian {
        source_idx: record.source_idx,
        u: record.u,
        v: record.v,
        sigma_x: record.sigma_x,
        sigma_y: record.sigma_y,
        raw_sigma_x: record.raw_sigma_x,
        raw_sigma_y: record.raw_sigma_y,
        depth: record.depth,
        opacity: record.opacity,
        opacity_logit: record.opacity_logit,
        scale3d: [record.scale_x, record.scale_y, record.scale_z],
        color: [record.color_r, record.color_g, record.color_b],
        min_x: record.min_x,
        max_x: record.max_x,
        min_y: record.min_y,
        max_y: record.max_y,
    }
}

fn projected_rows_to_cpu(
    projected: &ProjectedGaussians,
) -> candle_core::Result<Vec<CpuProjectedGaussian>> {
    let source_indices = projected.source_indices.to_vec1::<u32>()?;
    let u = projected.u.to_vec1::<f32>()?;
    let v = projected.v.to_vec1::<f32>()?;
    let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
    let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
    let raw_sigma_x = projected.raw_sigma_x.to_vec1::<f32>()?;
    let raw_sigma_y = projected.raw_sigma_y.to_vec1::<f32>()?;
    let depth = projected.depth.to_vec1::<f32>()?;
    let opacity = projected.opacity.to_vec1::<f32>()?;
    let opacity_logits = projected.opacity_logits.to_vec1::<f32>()?;
    let scale3d = projected.scale3d.to_vec2::<f32>()?;
    let colors = projected.colors.to_vec2::<f32>()?;
    let min_x = projected.min_x.to_vec1::<f32>()?;
    let max_x = projected.max_x.to_vec1::<f32>()?;
    let min_y = projected.min_y.to_vec1::<f32>()?;
    let max_y = projected.max_y.to_vec1::<f32>()?;
    let mut projected_cpu = Vec::with_capacity(source_indices.len());

    for idx in 0..source_indices.len() {
        projected_cpu.push(CpuProjectedGaussian {
            source_idx: source_indices[idx],
            u: u.get(idx).copied().unwrap_or(0.0),
            v: v.get(idx).copied().unwrap_or(0.0),
            sigma_x: sigma_x.get(idx).copied().unwrap_or(0.0),
            sigma_y: sigma_y.get(idx).copied().unwrap_or(0.0),
            raw_sigma_x: raw_sigma_x.get(idx).copied().unwrap_or(0.0),
            raw_sigma_y: raw_sigma_y.get(idx).copied().unwrap_or(0.0),
            depth: depth.get(idx).copied().unwrap_or(0.0),
            opacity: opacity.get(idx).copied().unwrap_or(0.0),
            opacity_logit: opacity_logits.get(idx).copied().unwrap_or(0.0),
            scale3d: row_to_vec3(scale3d.get(idx).map(Vec::as_slice).unwrap_or(&[])),
            color: row_to_vec3(colors.get(idx).map(Vec::as_slice).unwrap_or(&[])),
            min_x: min_x.get(idx).copied().unwrap_or(0.0),
            max_x: max_x.get(idx).copied().unwrap_or(0.0),
            min_y: min_y.get(idx).copied().unwrap_or(0.0),
            max_y: max_y.get(idx).copied().unwrap_or(0.0),
        });
    }

    Ok(projected_cpu)
}

fn pixel_color_grads(
    rendered_color: &[f32],
    target_color: &[f32],
    ssim_grads: &[f32],
    loss_scales: MetalBackwardLossScales,
) -> Vec<f32> {
    rendered_color
        .iter()
        .enumerate()
        .map(|(idx, &rendered)| {
            let target = target_color.get(idx).copied().unwrap_or(0.0);
            let l1 = if rendered > target {
                loss_scales.color
            } else if rendered < target {
                -loss_scales.color
            } else {
                0.0
            };
            l1 + ssim_grads.get(idx).copied().unwrap_or(0.0) * loss_scales.ssim
        })
        .collect()
}

fn pixel_depth_grads(
    rendered_depth: &[f32],
    target_depth: &[f32],
    loss_scales: MetalBackwardLossScales,
) -> Vec<f32> {
    rendered_depth
        .iter()
        .enumerate()
        .map(|(idx, &rendered)| {
            if loss_scales.depth <= 0.0
                || !is_valid_depth_sample(target_depth.get(idx).copied().unwrap_or(0.0))
            {
                return 0.0;
            }
            let target = target_depth[idx];
            if rendered > target {
                loss_scales.depth
            } else if rendered < target {
                -loss_scales.depth
            } else {
                0.0
            }
        })
        .collect()
}

fn is_valid_depth_sample(depth: f32) -> bool {
    depth.is_finite() && depth > 0.0
}

fn valid_depth_sample_count(depth: &[f32]) -> usize {
    depth
        .iter()
        .copied()
        .filter(|depth| is_valid_depth_sample(*depth))
        .count()
}

fn depth_backward_scale(depth_weight: f32, target_depth: &[f32]) -> f32 {
    if depth_weight <= 0.0 {
        return 0.0;
    }
    let valid_count = valid_depth_sample_count(target_depth);
    if valid_count == 0 {
        0.0
    } else {
        depth_weight / valid_count as f32
    }
}

fn row_to_vec3(row: &[f32]) -> [f32; 3] {
    [
        row.first().copied().unwrap_or(0.0),
        row.get(1).copied().unwrap_or(0.0),
        row.get(2).copied().unwrap_or(0.0),
    ]
}

fn row_to_quaternion(row: &[f32]) -> [f32; 4] {
    [
        row.first().copied().unwrap_or(1.0),
        row.get(1).copied().unwrap_or(0.0),
        row.get(2).copied().unwrap_or(0.0),
        row.get(3).copied().unwrap_or(0.0),
    ]
}

fn mat3_from_row_major(rotation: &[[f32; 3]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(rotation[0][0], rotation[1][0], rotation[2][0]),
        Vec3::new(rotation[0][1], rotation[1][1], rotation[2][1]),
        Vec3::new(rotation[0][2], rotation[1][2], rotation[2][2]),
    )
}

fn quat_from_wxyz(rotation: [f32; 4]) -> Quat {
    let length_sq = rotation.iter().map(|value| value * value).sum::<f32>();
    if !length_sq.is_finite() || length_sq <= 1e-12 {
        return Quat::IDENTITY;
    }
    Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]).normalize()
}

fn projected_axis_aligned_sigmas(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
    camera_rotation: &[[f32; 3]; 3],
    fx: f32,
    fy: f32,
) -> (f32, f32) {
    let inv_z = 1.0 / z.max(1e-4);
    let object_rotation = Mat3::from_quat(quat_from_wxyz(rotation));
    let scale_cov = Mat3::from_diagonal(Vec3::new(
        scale[0] * scale[0],
        scale[1] * scale[1],
        scale[2] * scale[2],
    ));
    let covariance_world = object_rotation * scale_cov * object_rotation.transpose();
    let camera_rotation = mat3_from_row_major(camera_rotation);
    let covariance_camera = camera_rotation * covariance_world * camera_rotation.transpose();

    let projection_row_x = Vec3::new(fx * inv_z, 0.0, -fx * x * inv_z * inv_z);
    let projection_row_y = Vec3::new(0.0, fy * inv_z, -fy * y * inv_z * inv_z);
    let covariance_x = projection_row_x.dot(covariance_camera * projection_row_x);
    let covariance_y = projection_row_y.dot(covariance_camera * projection_row_y);

    (covariance_x.max(1e-6).sqrt(), covariance_y.max(1e-6).sqrt())
}

fn finite_difference_sigma_wrt_rotation_component(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    raw_rotation: [f32; 4],
    component: usize,
    camera: &DiffCamera,
) -> (f32, f32) {
    const ROTATION_FD_EPS: f32 = 1e-3;
    if component >= 4 {
        return (0.0, 0.0);
    }

    let mut plus = raw_rotation;
    let mut minus = raw_rotation;
    plus[component] += ROTATION_FD_EPS;
    minus[component] -= ROTATION_FD_EPS;

    let (plus_sigma_x, plus_sigma_y) =
        projected_axis_aligned_sigmas(x, y, z, scale, plus, &camera.rotation, camera.fx, camera.fy);
    let (minus_sigma_x, minus_sigma_y) = projected_axis_aligned_sigmas(
        x,
        y,
        z,
        scale,
        minus,
        &camera.rotation,
        camera.fx,
        camera.fy,
    );
    (
        (plus_sigma_x.clamp(0.5, 256.0) - minus_sigma_x.clamp(0.5, 256.0))
            / (2.0 * ROTATION_FD_EPS),
        (plus_sigma_y.clamp(0.5, 256.0) - minus_sigma_y.clamp(0.5, 256.0))
            / (2.0 * ROTATION_FD_EPS),
    )
}

fn scale_camera(
    src: &DiffCamera,
    width: usize,
    height: usize,
    device: &Device,
) -> candle_core::Result<DiffCamera> {
    let sx = width as f32 / src.width as f32;
    let sy = height as f32 / src.height as f32;
    DiffCamera::new(
        src.fx * sx,
        src.fy * sy,
        src.cx * sx,
        src.cy * sy,
        width,
        height,
        &src.rotation,
        &src.translation,
        device,
    )
}

fn scaled_dimensions(width: usize, height: usize, render_scale: f32) -> (usize, usize) {
    let scale = render_scale.clamp(0.0625, 1.0);
    let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
    let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
    (scaled_width, scaled_height)
}

fn resize_rgb(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height * 3];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = [0.0f32; 3];
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src_idx = (sy * src_width + sx) * 3;
                    acc[0] += src[src_idx];
                    acc[1] += src[src_idx + 1];
                    acc[2] += src[src_idx + 2];
                    count += 1;
                }
            }
            let dst_idx = (dy * dst_width + dx) * 3;
            let inv = 1.0 / count.max(1) as f32;
            dst[dst_idx] = acc[0] * inv;
            dst[dst_idx + 1] = acc[1] * inv;
            dst[dst_idx + 2] = acc[2] * inv;
        }
    }
    dst
}

fn resize_depth(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = 0.0f32;
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let depth = src[sy * src_width + sx];
                    if is_valid_depth_sample(depth) {
                        acc += depth;
                        count += 1;
                    }
                }
            }
            dst[dy * dst_width + dx] = if count == 0 { 0.0 } else { acc / count as f32 };
        }
    }
    dst
}

fn filter_projected_gaussians_by_cluster_visibility(
    projected_cpu: &mut Vec<CpuProjectedGaussian>,
    cluster_visible_mask: Option<&[bool]>,
) {
    let Some(mask) = cluster_visible_mask else {
        return;
    };

    projected_cpu.retain(|gaussian| {
        mask.get(gaussian.source_idx as usize)
            .copied()
            .unwrap_or(true)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::metal_backward;

    fn max_slice_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max)
    }

    fn make_test_camera(device: &Device) -> DiffCamera {
        DiffCamera::new(
            64.0,
            64.0,
            32.0,
            32.0,
            64,
            64,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            device,
        )
        .unwrap()
    }

    fn projected_with_visible_sources(
        device: &Device,
        visible_source_indices: &[u32],
    ) -> ProjectedGaussians {
        let visible_count = visible_source_indices.len();
        ProjectedGaussians {
            source_indices: if visible_count == 0 {
                Tensor::zeros((0,), DType::U32, device).unwrap()
            } else {
                Tensor::from_slice(visible_source_indices, visible_count, device).unwrap()
            },
            u: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            v: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            sigma_x: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            sigma_y: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            raw_sigma_x: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            raw_sigma_y: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            depth: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            opacity: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
            opacity_logits: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            scale3d: Tensor::ones((visible_count, 3), DType::F32, device).unwrap(),
            colors: Tensor::zeros((visible_count, 3), DType::F32, device).unwrap(),
            min_x: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            max_x: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            min_y: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            max_y: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
            visible_source_indices: visible_source_indices.to_vec(),
            visible_count,
            tile_bins: MetalTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        }
    }

    #[test]
    fn update_gaussian_stats_uses_projected_grad_magnitudes_for_litegs() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &config, device.clone()).unwrap();
        let projected = projected_with_visible_sources(&device, &[1]);

        trainer
            .update_gaussian_stats(&[1e-9, 2e-9], &[0.0, 0.0035], &projected, 2)
            .unwrap();

        let expected = 0.0035 * trainer.pixel_count as f32;
        assert_eq!(trainer.gaussian_stats.len(), 2);
        assert!(trainer.gaussian_stats[0].mean2d_grad.mean.abs() < 1e-12);
        assert!((trainer.gaussian_stats[1].mean2d_grad.mean - expected).abs() < 1e-6);
        assert_eq!(trainer.gaussian_stats[1].visible_count, 1);
        assert_eq!(trainer.gaussian_stats[1].consecutive_invisible_epochs, 0);
        assert!((trainer.gaussian_stats[1].fragment_weight.mean - 1.0).abs() < 1e-6);
        assert!((trainer.gaussian_stats[1].fragment_err.mean - expected).abs() < 1e-6);
    }

    #[test]
    fn update_gaussian_stats_keeps_legacy_param_gradient_path() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LegacyMetal,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &config, device.clone()).unwrap();
        let projected = projected_with_visible_sources(&device, &[0]);

        trainer
            .update_gaussian_stats(&[0.001], &[9.0], &projected, 1)
            .unwrap();

        let expected = (0.001 * trainer.pixel_count as f32).min(10.0);
        assert!((trainer.gaussian_stats[0].mean2d_grad.mean - expected).abs() < 1e-6);
        assert_eq!(trainer.gaussian_stats[0].visible_count, 1);
        assert_eq!(trainer.gaussian_stats[0].consecutive_invisible_epochs, 0);
    }

    #[test]
    fn scaled_dimensions_keep_minimum_size() {
        assert_eq!(scaled_dimensions(640, 480, 0.25), (160, 120));
        assert_eq!(scaled_dimensions(1, 1, 0.0), (1, 1));
    }

    #[test]
    fn resize_depth_ignores_invalid_values() {
        let src = vec![1.0, 0.0, 3.0, f32::NAN];
        let resized = resize_depth(&src, 2, 2, 1, 1);
        assert_eq!(resized.len(), 1);
        assert!((resized[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn depth_backward_scale_uses_only_valid_depth_samples() {
        let target_depth = [1.0f32, 0.0, f32::NAN, 2.0];
        let scale = depth_backward_scale(LITEGS_DEPTH_LOSS_WEIGHT, &target_depth);
        assert!((scale - (LITEGS_DEPTH_LOSS_WEIGHT / 2.0)).abs() < 1e-6);
        assert_eq!(
            depth_backward_scale(LITEGS_DEPTH_LOSS_WEIGHT, &[0.0, f32::NAN]),
            0.0
        );
    }

    #[test]
    fn loss_curve_sampling_captures_bootstrap_interval_and_final_step() {
        assert!(should_record_loss_curve_sample(0, 100));
        assert!(should_record_loss_curve_sample(4, 100));
        assert!(!should_record_loss_curve_sample(5, 100));
        assert!(should_record_loss_curve_sample(25, 100));
        assert!(should_record_loss_curve_sample(99, 100));
    }

    #[test]
    fn metal_config_uses_safer_default_budget() {
        let effective = effective_metal_config(&TrainingConfig::default());
        assert_eq!(
            effective.max_initial_gaussians,
            TrainingConfig::default().max_initial_gaussians
        );
        assert_eq!(effective.lr_rotation, 0.0);
    }

    #[test]
    fn metal_config_freezes_rotation_learning_for_legacy_profile() {
        let effective = effective_metal_config(&TrainingConfig {
            lr_rotation: 0.25,
            ..TrainingConfig::default()
        });
        assert_eq!(effective.lr_rotation, 0.0);
    }

    #[test]
    fn litegs_profile_preserves_rotation_learning_rate() {
        let effective = effective_metal_config(&TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            lr_rotation: 0.25,
            ..TrainingConfig::default()
        });
        assert_eq!(effective.lr_rotation, 0.25);
    }

    #[test]
    fn litegs_profile_uses_litegs_opacity_lr_default() {
        let effective = effective_metal_config(&TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        });
        assert_eq!(effective.lr_opacity, 0.025);
    }

    #[test]
    fn trainer_disables_native_forward_on_cpu() {
        let trainer = MetalTrainer::new(32, 16, &TrainingConfig::default(), Device::Cpu).unwrap();
        assert!(!trainer.use_native_forward);
    }

    #[test]
    fn trainer_respects_explicit_native_forward_disable() {
        let config = TrainingConfig {
            metal_use_native_forward: false,
            ..TrainingConfig::default()
        };
        let trainer = MetalTrainer::new(32, 16, &config, Device::Cpu).unwrap();
        assert!(!trainer.use_native_forward);
    }

    #[test]
    fn trainer_uses_explicit_legacy_topology_thresholds_from_config() {
        let config = TrainingConfig {
            legacy_densify_grad_threshold: 0.0125,
            legacy_clone_scale_threshold: 0.22,
            legacy_split_scale_threshold: 0.44,
            legacy_prune_scale_threshold: 0.66,
            legacy_max_densify_per_update: 77,
            ..TrainingConfig::default()
        };

        let trainer = MetalTrainer::new(32, 16, &config, Device::Cpu).unwrap();

        assert_eq!(trainer.legacy_densify_grad_threshold, 0.0125);
        assert_eq!(trainer.legacy_clone_scale_threshold, 0.22);
        assert_eq!(trainer.legacy_split_scale_threshold, 0.44);
        assert_eq!(trainer.legacy_prune_scale_threshold, 0.66);
        assert_eq!(trainer.legacy_max_densify_per_update, 77);
    }

    #[test]
    fn litegs_late_stage_start_epoch_clamps_to_topology_window() {
        let trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                iterations: 1_200,
                litegs: LiteGsConfig {
                    densify_from: 3,
                    densify_until: Some(11),
                    ..LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            Device::Cpu,
        )
        .unwrap();

        assert_eq!(trainer.litegs_total_epochs(90), 13);
        assert_eq!(trainer.litegs_densify_until_epoch(90), 11);
        assert_eq!(trainer.litegs_late_stage_start_epoch(90), 8);
    }

    #[test]
    fn litegs_topology_metrics_capture_late_stage_activity() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 3,
            topology_warmup: 0,
            max_initial_gaussians: 4,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(3),
                refine_every: 1,
                densification_interval: 1,
                opacity_reset_interval: 1,
                prune_min_age: 1,
                prune_invisible_epochs: 1,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        trainer.iteration = 3;

        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
            ],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[2.0, -10.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                visible_count: 1,
                age: 1,
                ..Default::default()
            },
            MetalGaussianStats {
                age: 1,
                ..Default::default()
            },
        ];

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert_eq!(trainer.topology_metrics.total_epochs, Some(3));
        assert_eq!(trainer.topology_metrics.densify_until_epoch, Some(3));
        assert_eq!(trainer.topology_metrics.late_stage_start_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.first_densify_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.last_densify_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.late_stage_densify_events, 1);
        assert_eq!(trainer.topology_metrics.late_stage_densify_added, 1);
        assert_eq!(trainer.topology_metrics.first_prune_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.last_prune_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.late_stage_prune_events, 1);
        assert_eq!(trainer.topology_metrics.late_stage_prune_removed, 1);
        assert_eq!(trainer.topology_metrics.first_opacity_reset_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.last_opacity_reset_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.late_stage_opacity_reset_events, 1);
    }

    #[test]
    fn litegs_topology_freeze_after_epoch_skips_late_stage_updates() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 3,
            topology_warmup: 0,
            max_initial_gaussians: 4,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(3),
                topology_freeze_after_epoch: Some(2),
                refine_every: 1,
                densification_interval: 1,
                opacity_reset_interval: 1,
                prune_min_age: 1,
                prune_invisible_epochs: 1,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        trainer.iteration = 3;

        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0],
            &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[2.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 1,
            ..Default::default()
        }];

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert_eq!(gaussians.len(), 1);
        assert_eq!(trainer.topology_metrics.topology_freeze_epoch, Some(2));
        assert_eq!(trainer.topology_metrics.densify_events, 0);
        assert_eq!(trainer.topology_metrics.prune_events, 0);
        assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
    }

    #[test]
    fn litegs_topology_warmup_blocks_epoch_based_updates() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 20,
            topology_warmup: 10,
            max_initial_gaussians: 4,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(6),
                refine_every: 1,
                densification_interval: 1,
                opacity_reset_interval: 1,
                prune_min_age: 1,
                prune_invisible_epochs: 1,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        trainer.iteration = 6;

        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0],
            &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[2.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 1,
            ..Default::default()
        }];

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert_eq!(gaussians.len(), 1);
        assert_eq!(trainer.topology_metrics.densify_events, 0);
        assert_eq!(trainer.topology_metrics.prune_events, 0);
        assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
    }

    #[test]
    fn litegs_topology_skips_prune_without_growth_candidates() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 20,
            topology_warmup: 0,
            max_initial_gaussians: 4,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(6),
                refine_every: 1,
                densification_interval: 1,
                opacity_reset_interval: 1,
                prune_min_age: 1,
                prune_invisible_epochs: 1,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        trainer.iteration = 6;

        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0],
            &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[-10.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![MetalGaussianStats {
            age: 1,
            ..Default::default()
        }];

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert_eq!(gaussians.len(), 1);
        assert_eq!(trainer.topology_metrics.densify_events, 0);
        assert_eq!(trainer.topology_metrics.prune_events, 0);
        assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
    }

    #[test]
    fn projected_axis_aligned_sigmas_change_with_rotation() {
        let camera_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let identity = projected_axis_aligned_sigmas(
            0.0,
            0.0,
            2.0,
            [0.4, 0.05, 0.05],
            [1.0, 0.0, 0.0, 0.0],
            &camera_rotation,
            64.0,
            64.0,
        );
        let rotated = projected_axis_aligned_sigmas(
            0.0,
            0.0,
            2.0,
            [0.4, 0.05, 0.05],
            [
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            &camera_rotation,
            64.0,
            64.0,
        );

        assert!(identity.0 > identity.1 * 5.0, "{identity:?}");
        assert!(rotated.1 > rotated.0 * 5.0, "{rotated:?}");
        assert!(
            (identity.0 - rotated.0).abs() > 1.0,
            "{identity:?} vs {rotated:?}"
        );
    }

    #[test]
    fn project_gaussians_uses_rotation_aware_footprints() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            64.0,
            64.0,
            32.0,
            32.0,
            64,
            64,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let identity = TrainableGaussians::new(
            &[0.0, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let rotated = TrainableGaussians::new(
            &[0.0, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let (identity_projected, _) = trainer
            .project_gaussians(&identity, &camera, false, true, None)
            .unwrap();
        let (rotated_projected, _) = trainer
            .project_gaussians(&rotated, &camera, false, true, None)
            .unwrap();
        let identity_sigma_x = identity_projected.sigma_x.to_vec1::<f32>().unwrap()[0];
        let identity_sigma_y = identity_projected.sigma_y.to_vec1::<f32>().unwrap()[0];
        let rotated_sigma_x = rotated_projected.sigma_x.to_vec1::<f32>().unwrap()[0];
        let rotated_sigma_y = rotated_projected.sigma_y.to_vec1::<f32>().unwrap()[0];

        assert!(identity_sigma_x > identity_sigma_y * 5.0);
        assert!(rotated_sigma_y > rotated_sigma_x * 5.0);
        assert!((identity_sigma_x - rotated_sigma_x).abs() > 1.0);
    }

    #[test]
    fn rotation_parameter_grads_become_nonzero_for_asymmetric_color_error() {
        let device = Device::Cpu;
        let base_z = 0.25f32;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            lr_rotation: 0.1,
            metal_render_scale: 1.0,
            metal_use_native_forward: false,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            96.0,
            48.0,
            32.0,
            32.0,
            64,
            64,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[0.25, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, base_z],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let (rendered, projected, _) = trainer
            .render(&gaussians, &camera, false, true, None)
            .unwrap();
        let rendered_color_cpu = rendered
            .color
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut target_color = rendered_color_cpu.clone();
        let u = projected.u.to_vec1::<f32>().unwrap()[0];
        let v = projected.v.to_vec1::<f32>().unwrap()[0];
        let target_px = (u + 2.0).round().clamp(0.0, 63.0) as usize;
        let target_py = v.round().clamp(0.0, 63.0) as usize;
        let target_idx = (target_py * 64 + target_px) * 3;
        target_color[target_idx] = (target_color[target_idx] - 0.2).max(0.0);
        let target_depth = vec![0.0; trainer.pixel_count];
        let ssim_grads = vec![0.0; rendered_color_cpu.len()];
        let rotation_grads = trainer
            .rotation_parameter_grads(
                &gaussians,
                &projected,
                &rendered,
                &rendered_color_cpu,
                &target_color,
                &target_depth,
                &ssim_grads,
                MetalBackwardLossScales {
                    color: 1.0,
                    depth: 0.0,
                    ssim: 0.0,
                    alpha: 0.0,
                },
                &camera,
            )
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let analytic = rotation_grads[0][3];

        assert!(analytic.abs() > 1e-4, "analytic={analytic}");
    }

    #[test]
    fn apply_backward_grads_updates_rotations_when_rotation_grad_is_present() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            lr_rotation: 0.1,
            metal_render_scale: 1.0,
            metal_use_native_forward: false,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.iteration = 1;
        let camera = DiffCamera::new(
            64.0,
            64.0,
            32.0,
            32.0,
            64,
            64,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        let (_, projected, _) = trainer
            .render(&gaussians, &camera, false, true, None)
            .unwrap();
        let zero_grads = MetalBackwardGrads {
            positions: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
            log_scales: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
            opacity_logits: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            colors: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
        };
        let rotation_grads = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], (1, 4), &device).unwrap();
        let before = gaussians.rotations.as_tensor().to_vec2::<f32>().unwrap();

        trainer
            .apply_backward_grads(
                &mut gaussians,
                &zero_grads,
                &projected,
                &camera,
                0.0,
                None,
                Some(&rotation_grads),
            )
            .unwrap();

        let after = gaussians.rotations.as_tensor().to_vec2::<f32>().unwrap();
        assert!(
            after[0][3] < before[0][3],
            "before={before:?} after={after:?}"
        );
    }

    #[test]
    fn apply_backward_grads_sparse_grad_preserves_invisible_rows_and_moments() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sparse_grad: true,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.iteration = 3;
        let camera = make_test_camera(&device);
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        let adam = trainer.adam.as_mut().unwrap();
        adam.m_pos =
            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.5, -0.25, 0.75], (2, 3), &device).unwrap();
        adam.v_pos =
            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.4, 0.2, 0.6], (2, 3), &device).unwrap();
        let projected = projected_with_visible_sources(&device, &[0]);
        let grads = MetalBackwardGrads {
            positions: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &device)
                .unwrap(),
            log_scales: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            opacity_logits: Tensor::zeros((2,), DType::F32, &device).unwrap(),
            colors: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        };
        let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();

        trainer
            .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
            .unwrap();

        let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
        let after_m_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .m_pos
            .to_vec2::<f32>()
            .unwrap();
        let after_v_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .v_pos
            .to_vec2::<f32>()
            .unwrap();

        assert!(after_positions[0][0] < before_positions[0][0]);
        assert_eq!(after_positions[1], before_positions[1]);
        assert!(after_m_pos[0][0].abs() > 1e-6);
        assert_eq!(after_m_pos[1], vec![0.5, -0.25, 0.75]);
        assert_eq!(after_v_pos[1], vec![0.4, 0.2, 0.6]);
    }

    #[test]
    fn apply_backward_grads_sparse_grad_noops_when_no_gaussians_are_visible() {
        let device = Device::Cpu;
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sparse_grad: true,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.iteration = 4;
        let camera = make_test_camera(&device);
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        let adam = trainer.adam.as_mut().unwrap();
        adam.m_pos =
            Tensor::from_slice(&[0.3f32, -0.1, 0.2, 0.5, -0.25, 0.75], (2, 3), &device).unwrap();
        adam.v_pos =
            Tensor::from_slice(&[0.4f32, 0.2, 0.1, 0.4, 0.2, 0.6], (2, 3), &device).unwrap();
        let projected = projected_with_visible_sources(&device, &[]);
        let grads = MetalBackwardGrads {
            positions: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            log_scales: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
            opacity_logits: Tensor::zeros((2,), DType::F32, &device).unwrap(),
            colors: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        };
        let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();
        let before_m_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .m_pos
            .to_vec2::<f32>()
            .unwrap();
        let before_v_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .v_pos
            .to_vec2::<f32>()
            .unwrap();

        trainer
            .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
            .unwrap();

        let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
        let after_m_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .m_pos
            .to_vec2::<f32>()
            .unwrap();
        let after_v_pos = trainer
            .adam
            .as_ref()
            .unwrap()
            .v_pos
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(after_positions, before_positions);
        assert_eq!(after_m_pos, before_m_pos);
        assert_eq!(after_v_pos, before_v_pos);
    }

    #[test]
    fn litegs_loss_weights_only_enable_depth_when_requested() {
        let disabled = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: LiteGsConfig {
                    enable_depth: false,
                    ..LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            Device::Cpu,
        )
        .unwrap();
        let enabled = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: LiteGsConfig {
                    enable_depth: true,
                    ..LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            Device::Cpu,
        )
        .unwrap();

        assert_eq!(disabled.loss_weights(), (0.8, 0.2, 0.0));
        assert_eq!(enabled.loss_weights(), (0.8, 0.2, LITEGS_DEPTH_LOSS_WEIGHT));
    }

    #[test]
    fn training_step_records_depth_telemetry_with_clustered_sparse_grad() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                cluster_size: 1,
                sparse_grad: true,
                enable_depth: true,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.scene_extent = 16.0;
        let camera = make_test_camera(&device);
        let mut gaussians = TrainableGaussians::new(
            &[
                0.0, 0.0, 2.0, //
                0.0, 0.0, -2.0,
            ],
            &[
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            &[
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0,
            ],
            &[0.0, 0.0],
            &[
                1.0, 0.25, 0.25, //
                0.1, 1.0, 0.1,
            ],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.sync_cluster_assignment(&gaussians, false).unwrap();
        let cluster_visible_mask =
            trainer.cluster_visible_mask_for_camera(gaussians.len(), &camera);
        let (rendered, _, _) = trainer
            .render(
                &gaussians,
                &camera,
                false,
                true,
                cluster_visible_mask.as_deref(),
            )
            .unwrap();
        let target_depth_cpu = rendered
            .depth
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let target_color_cpu = vec![0.0f32; trainer.pixel_count * 3];
        let frame = MetalTrainingFrame {
            camera: camera.clone(),
            target_color: Tensor::from_slice(&target_color_cpu, (trainer.pixel_count, 3), &device)
                .unwrap(),
            target_depth: rendered.depth.clone(),
            target_color_cpu,
            target_depth_cpu,
        };

        let outcome = trainer
            .training_step(&mut gaussians, &frame, 0, 1, false)
            .unwrap();
        let telemetry = trainer.current_telemetry(1);
        let color_moments = trainer
            .adam
            .as_ref()
            .unwrap()
            .m_color
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(outcome.visible_gaussians, 1);
        assert!(telemetry.loss_terms.total.unwrap_or(0.0) > 0.0);
        assert!(telemetry.loss_terms.depth.is_some());
        assert_eq!(telemetry.depth_valid_pixels, Some(trainer.pixel_count));
        assert_eq!(
            telemetry.depth_grad_scale,
            Some(LITEGS_DEPTH_LOSS_WEIGHT / trainer.pixel_count as f32)
        );
        assert_eq!(telemetry.learning_rates.xyz, Some(trainer.compute_lr_pos()));
        assert!(color_moments[0].iter().any(|value| value.abs() > 1e-8));
        assert!(color_moments[1].iter().all(|value| value.abs() < 1e-8));
        assert!(trainer.cluster_assignment.is_some());
    }

    #[test]
    fn training_step_updates_dense_litegs_params_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                cluster_size: 1,
                sparse_grad: false,
                enable_depth: true,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.scene_extent = 16.0;
        let camera = make_test_camera(&device);
        let mut gaussians = TrainableGaussians::new(
            &[
                0.0, 0.0, 2.0, //
                0.0, 0.0, -2.0,
            ],
            &[
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            &[
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0,
            ],
            &[0.0, 0.0],
            &[
                1.0, 0.25, 0.25, //
                0.1, 1.0, 0.1,
            ],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.sync_cluster_assignment(&gaussians, false).unwrap();
        let cluster_visible_mask =
            trainer.cluster_visible_mask_for_camera(gaussians.len(), &camera);
        let (rendered, _, _) = trainer
            .render(
                &gaussians,
                &camera,
                false,
                true,
                cluster_visible_mask.as_deref(),
            )
            .unwrap();
        let target_depth_cpu = rendered
            .depth
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let target_color_cpu = vec![0.0f32; trainer.pixel_count * 3];
        let frame = MetalTrainingFrame {
            camera: camera.clone(),
            target_color: Tensor::from_slice(&target_color_cpu, (trainer.pixel_count, 3), &device)
                .unwrap(),
            target_depth: rendered.depth.clone(),
            target_color_cpu,
            target_depth_cpu,
        };
        let before_positions = gaussians
            .positions()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_scales = gaussians
            .scales
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_opacities = gaussians
            .opacities
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_colors = gaussians
            .colors()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let outcome = trainer
            .training_step(&mut gaussians, &frame, 0, 1, false)
            .unwrap();
        let telemetry = trainer.current_telemetry(1);
        let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
        let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
        let opacity_delta =
            max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
        let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();
        let color_moments = trainer
            .adam
            .as_ref()
            .unwrap()
            .m_color
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(outcome.visible_gaussians, 1);
        assert!(telemetry.loss_terms.total.unwrap_or(0.0) > 0.0);
        assert!(
            position_delta > 1e-8
                || scale_delta > 1e-8
                || opacity_delta > 1e-8
                || color_delta > 1e-8,
            "position_delta={position_delta:.6e} scale_delta={scale_delta:.6e} opacity_delta={opacity_delta:.6e} color_delta={color_delta:.6e}"
        );
        assert!(
            color_moments.iter().any(|value| value.abs() > 1e-8),
            "color_moments={color_moments:?}"
        );
    }

    #[test]
    fn tum_frame_initialized_backward_probe_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test_data/tum/rgbd_dataset_freiburg1_xyz");
        let dataset = crate::load_tum_rgbd_dataset(
            &dataset_path,
            &crate::TumRgbdConfig {
                max_frames: 10,
                frame_stride: 30,
                ..crate::TumRgbdConfig::default()
            },
        )
        .unwrap();
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 40,
            max_initial_gaussians: 100000,
            topology_warmup: 0,
            metal_render_scale: 0.5,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(6),
                refine_every: 10,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let effective_config = effective_metal_config(&config);
        let mut loaded = load_training_data(&dataset, &effective_config, &device).unwrap();
        let mut trainer = MetalTrainer::new(
            dataset.intrinsics.width as usize,
            dataset.intrinsics.height as usize,
            &effective_config,
            device.clone(),
        )
        .unwrap();
        let memory_budget = training_memory_budget(&config);
        let affordable_cap = affordable_initial_gaussian_cap(
            effective_config
                .max_initial_gaussians
                .max(loaded.initial_map.len()),
            trainer.pixel_count,
            trainer.source_pixel_count,
            loaded.cameras.len(),
            trainer.chunk_size,
            &memory_budget,
        );
        if affordable_cap > 0 && loaded.initial_map.len() > affordable_cap {
            let initial_cap =
                preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
            downsample_gaussian_map_evenly(&mut loaded.initial_map, initial_cap);
        }
        trainer.scene_extent = estimate_scene_extent(&loaded.initial_map);
        let mut gaussians =
            trainable_from_map(&loaded.initial_map, &device, &effective_config).unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.iteration = 1;
        let frames = trainer.prepare_frames(&loaded).unwrap();
        let frame = &frames[0];
        let (rendered, projected, _) = trainer
            .render(&gaussians, &frame.camera, false, true, None)
            .unwrap();
        let rendered_color_cpu = rendered
            .color
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (ssim_value, ssim_grads) = ssim_gradient(
            &rendered_color_cpu,
            &frame.target_color_cpu,
            trainer.render_width,
            trainer.render_height,
        );
        trainer.runtime.write_ssim_grad(&ssim_grads).unwrap();
        let backward_loss_scales = MetalBackwardLossScales {
            color: 0.8 / frame.target_color_cpu.len().max(1) as f32,
            depth: 0.0,
            ssim: 0.2 / frame.target_color_cpu.len().max(1) as f32,
            alpha: 0.0,
        };
        trainer
            .runtime
            .write_target_data(
                &frame.target_color_cpu,
                &frame.target_depth_cpu,
                backward_loss_scales.color,
                backward_loss_scales.depth,
                backward_loss_scales.ssim,
                backward_loss_scales.alpha,
            )
            .unwrap();
        let backward = metal_backward::backward_weighted_l1(
            &mut trainer.runtime,
            &projected.tile_bins,
            gaussians.len(),
            &frame.camera,
        )
        .unwrap();
        let position_stats = tensor_abs_stats(&backward.grads.positions).unwrap();
        let scale_stats = tensor_abs_stats(&backward.grads.log_scales).unwrap();
        let opacity_stats = tensor_abs_stats(&backward.grads.opacity_logits).unwrap();
        let color_stats = tensor_abs_stats(&backward.grads.colors).unwrap();
        let param_grad_stats = abs_stats(&backward.grad_magnitudes);
        let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
        let before_positions = gaussians
            .positions()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_scales = gaussians
            .scales
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_opacities = gaussians
            .opacities
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_colors = gaussians
            .colors()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        trainer
            .apply_backward_grads(
                &mut gaussians,
                &backward.grads,
                &projected,
                &frame.camera,
                trainer.compute_lr_pos(),
                None,
                None,
            )
            .unwrap();
        let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
        let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
        let opacity_delta =
            max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
        let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();
        let trained_map = map_from_trainable(&gaussians).unwrap();
        let mut map_position_delta = 0.0f32;
        let mut map_scale_delta = 0.0f32;
        let mut map_opacity_delta = 0.0f32;
        let mut map_color_delta = 0.0f32;
        for (before, after) in loaded
            .initial_map
            .gaussians()
            .iter()
            .zip(trained_map.gaussians().iter())
        {
            map_position_delta =
                map_position_delta.max((before.position - after.position).abs().max_element());
            map_scale_delta = map_scale_delta.max((before.scale - after.scale).abs().max_element());
            map_opacity_delta = map_opacity_delta.max((before.opacity - after.opacity).abs());
            for channel in 0..3 {
                map_color_delta =
                    map_color_delta.max((before.color[channel] - after.color[channel]).abs());
            }
        }

        assert!(
            position_delta > 1e-8
                || scale_delta > 1e-8
                || opacity_delta > 1e-8
                || color_delta > 1e-8,
            "tum probe no-op | gaussians={} visible={} ssim={:.6} | grad_positions={:?} grad_scales={:?} grad_opacity={:?} grad_colors={:?} | param_grad={:?} projected_grad={:?} | deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
            gaussians.len(),
            projected.visible_count,
            ssim_value,
            position_stats,
            scale_stats,
            opacity_stats,
            color_stats,
            param_grad_stats,
            projected_grad_stats,
            position_delta,
            scale_delta,
            opacity_delta,
            color_delta,
        );
        assert!(
            map_position_delta > 1e-8
                || map_scale_delta > 1e-8
                || map_opacity_delta > 1e-8
                || map_color_delta > 1e-8,
            "tum train-loop export no-op | tensor_deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e}) | map_deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
            position_delta,
            scale_delta,
            opacity_delta,
            color_delta,
            map_position_delta,
            map_scale_delta,
            map_opacity_delta,
            map_color_delta,
        );
    }

    #[test]
    fn tum_frame_initialized_train_loop_updates_params_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test_data/tum/rgbd_dataset_freiburg1_xyz");
        let dataset = crate::load_tum_rgbd_dataset(
            &dataset_path,
            &crate::TumRgbdConfig {
                max_frames: 10,
                frame_stride: 30,
                ..crate::TumRgbdConfig::default()
            },
        )
        .unwrap();
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 1,
            max_initial_gaussians: 100000,
            topology_warmup: 0,
            metal_render_scale: 0.5,
            litegs: LiteGsConfig {
                densify_from: 0,
                densify_until: Some(6),
                refine_every: 10,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let effective_config = effective_metal_config(&config);
        let mut loaded = load_training_data(&dataset, &effective_config, &device).unwrap();
        let mut trainer = MetalTrainer::new(
            dataset.intrinsics.width as usize,
            dataset.intrinsics.height as usize,
            &effective_config,
            device.clone(),
        )
        .unwrap();
        let memory_budget = training_memory_budget(&config);
        let affordable_cap = affordable_initial_gaussian_cap(
            effective_config
                .max_initial_gaussians
                .max(loaded.initial_map.len()),
            trainer.pixel_count,
            trainer.source_pixel_count,
            loaded.cameras.len(),
            trainer.chunk_size,
            &memory_budget,
        );
        if affordable_cap > 0 && loaded.initial_map.len() > affordable_cap {
            let initial_cap =
                preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
            downsample_gaussian_map_evenly(&mut loaded.initial_map, initial_cap);
        }
        trainer.scene_extent = estimate_scene_extent(&loaded.initial_map);
        let mut gaussians =
            trainable_from_map(&loaded.initial_map, &device, &effective_config).unwrap();
        let frames = trainer.prepare_frames(&loaded).unwrap();
        let before_positions = gaussians
            .positions()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_scales = gaussians
            .scales
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_opacities = gaussians
            .opacities
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let before_colors = gaussians
            .colors()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        trainer.train(&mut gaussians, &frames, 1).unwrap();

        let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
        let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
        let opacity_delta =
            max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
        let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();

        assert!(
            position_delta > 1e-8
                || scale_delta > 1e-8
                || opacity_delta > 1e-8
                || color_delta > 1e-8,
            "tum train-loop no-op | deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
            position_delta,
            scale_delta,
            opacity_delta,
            color_delta,
        );
    }

    #[test]
    fn adam_step_var_sparse_preserves_invisible_rows_for_tensor3_params() {
        let device = Device::Cpu;
        let var = Var::from_tensor(
            &Tensor::from_slice(
                &[
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                (2, 2, 3),
                &device,
            )
            .unwrap(),
        )
        .unwrap();
        let grad = Tensor::from_slice(
            &[
                0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75,
            ],
            (2, 2, 3),
            &device,
        )
        .unwrap();
        let mut m = Tensor::from_slice(
            &[
                0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            (2, 2, 3),
            &device,
        )
        .unwrap();
        let mut v = Tensor::from_slice(
            &[
                0.25f32, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            (2, 2, 3),
            &device,
        )
        .unwrap();
        let indices = Tensor::from_slice(&[1u32], 1, &device).unwrap();
        let before_param = var.as_tensor().to_vec3::<f32>().unwrap();
        let before_m = m.to_vec3::<f32>().unwrap();
        let before_v = v.to_vec3::<f32>().unwrap();

        adam_step_var_sparse(
            &var, &grad, &mut m, &mut v, &indices, 0.1, 0.9, 0.999, 1e-8, 2,
        )
        .unwrap();

        let after_param = var.as_tensor().to_vec3::<f32>().unwrap();
        let after_m = m.to_vec3::<f32>().unwrap();
        let after_v = v.to_vec3::<f32>().unwrap();

        assert_eq!(after_param[0], before_param[0]);
        assert_eq!(after_m[0], before_m[0]);
        assert_eq!(after_v[0], before_v[0]);
        assert_ne!(after_param[1], before_param[1]);
        assert_ne!(after_m[1], before_m[1]);
        assert_ne!(after_v[1], before_v[1]);
    }

    #[test]
    fn adam_step_var_fused_matches_cpu_update_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let mut runtime = MetalRuntime::new(1, 1, device.clone()).unwrap();
        let shape = (2, 3);
        let initial = [1.0f32, -2.0, 0.25, 3.0, -4.0, 5.0];
        let grads = [0.5f32, -1.5, 2.0, -0.25, 0.75, -3.0];
        let var = Var::from_tensor(&Tensor::from_slice(&initial, shape, &device).unwrap()).unwrap();
        let grad = Tensor::from_slice(&grads, shape, &device).unwrap();
        let mut m = Tensor::zeros(shape, DType::F32, &device).unwrap();
        let mut v = Tensor::zeros(shape, DType::F32, &device).unwrap();

        let cpu = Device::Cpu;
        let (expected_param, expected_m, expected_v) = adam_updated_tensors(
            &Tensor::from_slice(&initial, shape, &cpu).unwrap(),
            &Tensor::from_slice(&grads, shape, &cpu).unwrap(),
            &Tensor::zeros(shape, DType::F32, &cpu).unwrap(),
            &Tensor::zeros(shape, DType::F32, &cpu).unwrap(),
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
        )
        .unwrap();

        adam_step_var_fused(
            &var,
            &grad,
            &mut m,
            &mut v,
            &mut runtime,
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
            MetalBufferSlot::AdamGradPos,
            MetalBufferSlot::AdamMPos,
            MetalBufferSlot::AdamVPos,
            MetalBufferSlot::AdamParamPos,
        )
        .unwrap();

        let actual_param = var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual_m = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_v = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_param = expected_param
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_m = expected_m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_v = expected_v.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert!(
            max_slice_delta(&actual_param, &expected_param) < 1e-6,
            "actual_param={actual_param:?} expected_param={expected_param:?}"
        );
        assert!(
            max_slice_delta(&actual_m, &expected_m) < 1e-6,
            "actual_m={actual_m:?} expected_m={expected_m:?}"
        );
        assert!(
            max_slice_delta(&actual_v, &expected_v) < 1e-6,
            "actual_v={actual_v:?} expected_v={expected_v:?}"
        );
    }

    #[test]
    fn apply_backward_grads_dense_updates_metal_params() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sparse_grad: false,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
        trainer.iteration = 3;
        let camera = make_test_camera(&device);
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[0.5, 0.1, 0.2, 0.9, 0.2, 0.1],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        let projected = projected_with_visible_sources(&device, &[0]);
        let grads = MetalBackwardGrads {
            positions: Tensor::from_slice(&[1.0f32, -0.5, 0.25, 0.0, 0.0, 0.0], (2, 3), &device)
                .unwrap(),
            log_scales: Tensor::from_slice(&[0.2f32, -0.1, 0.05, 0.0, 0.0, 0.0], (2, 3), &device)
                .unwrap(),
            opacity_logits: Tensor::from_slice(&[0.3f32, 0.0], 2, &device).unwrap(),
            colors: Tensor::from_slice(&[0.4f32, -0.2, 0.1, 0.0, 0.0, 0.0], (2, 3), &device)
                .unwrap(),
        };
        let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();
        let before_scales = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap();
        let before_opacity = gaussians.opacities.as_tensor().to_vec1::<f32>().unwrap();
        let before_colors = gaussians.colors().to_vec2::<f32>().unwrap();

        trainer
            .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
            .unwrap();

        let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
        let after_scales = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap();
        let after_opacity = gaussians.opacities.as_tensor().to_vec1::<f32>().unwrap();
        let after_colors = gaussians.colors().to_vec2::<f32>().unwrap();
        let adam = trainer.adam.as_ref().unwrap();
        let m_pos = adam.m_pos.to_vec2::<f32>().unwrap();
        let m_color = adam.m_color.to_vec2::<f32>().unwrap();

        assert_ne!(after_positions[0], before_positions[0]);
        assert_eq!(after_positions[1], before_positions[1]);
        assert_ne!(after_scales[0], before_scales[0]);
        assert_eq!(after_scales[1], before_scales[1]);
        assert_ne!(after_opacity[0], before_opacity[0]);
        assert_eq!(after_opacity[1], before_opacity[1]);
        assert_ne!(after_colors[0], before_colors[0]);
        assert_eq!(after_colors[1], before_colors[1]);
        assert!(m_pos[0].iter().any(|value| value.abs() > 1e-8));
        assert!(m_pos[1].iter().all(|value| value.abs() < 1e-8));
        assert!(m_color[0].iter().any(|value| value.abs() > 1e-8));
        assert!(m_color[1].iter().all(|value| value.abs() < 1e-8));
    }

    #[test]
    fn peak_estimate_scales_with_problem_size() {
        let small = estimate_peak_memory(4_096, 4_800, 5, 32);
        let large = estimate_peak_memory(57_474, 4_800, 5, 32);
        assert!(large.total_bytes() > small.total_bytes());
        assert!(bytes_to_gib(large.total_bytes()) > 10.0);
    }

    #[test]
    fn peak_estimate_accounts_for_frames_and_chunk_size() {
        let baseline = estimate_peak_memory(4_096, 4_800, 5, 32);
        let more_frames = estimate_peak_memory(4_096, 4_800, 25, 32);
        let larger_chunk = estimate_peak_memory(4_096, 4_800, 5, 128);
        assert!(more_frames.total_bytes() > baseline.total_bytes());
        assert!(larger_chunk.total_bytes() > baseline.total_bytes());
    }

    #[test]
    fn peak_estimate_accounts_for_retained_source_resolution_staging() {
        let render_pixels = 320 * 180;
        let source_pixels = 1920 * 1080;
        let baseline = estimate_peak_memory(4_096, render_pixels, 5, 32);
        let staged =
            estimate_peak_memory_with_source_pixels(4_096, render_pixels, source_pixels, 5, 32);
        assert!(staged.frame_bytes > baseline.frame_bytes);
        assert!(staged.total_bytes() > baseline.total_bytes());
    }

    #[test]
    fn detected_budget_prefers_fraction_of_system_memory() {
        let physical = 16 * GIB;
        let safe = apply_ratio(
            physical,
            METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
            METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
        )
        .min(DEFAULT_METAL_MEMORY_BUDGET_BYTES);
        assert!((bytes_to_gib(safe) - 10.4).abs() < 0.05);
    }

    #[test]
    fn preflight_warns_when_close_to_budget() {
        let estimate = estimate_peak_memory(8_000, 4_800, 5, 32);
        let budget = MetalMemoryBudget {
            safe_bytes: estimate.total_bytes().saturating_add(512 * MIB),
            physical_bytes: Some(16 * GIB),
        };
        assert_eq!(
            assess_memory_estimate(&estimate, &budget),
            MetalMemoryDecision::Warn
        );
    }

    #[test]
    fn preflight_blocks_when_estimate_exceeds_budget() {
        let estimate = estimate_peak_memory(57_474, 4_800, 1, 32);
        let budget = MetalMemoryBudget {
            safe_bytes: 16 * GIB,
            physical_bytes: Some(24 * GIB),
        };
        assert_eq!(
            assess_memory_estimate(&estimate, &budget),
            MetalMemoryDecision::Block
        );
    }

    #[test]
    fn affordable_initial_gaussian_cap_finds_non_blocking_limit() {
        let budget = MetalMemoryBudget {
            safe_bytes: 16 * GIB,
            physical_bytes: Some(24 * GIB),
        };
        let cap = affordable_initial_gaussian_cap(57_474, 4_800, 4_800, 1, 32, &budget);
        assert!(cap > 0);
        assert!(cap < 57_474);
        assert_ne!(
            assess_memory_estimate(&estimate_peak_memory(cap, 4_800, 1, 32), &budget),
            MetalMemoryDecision::Block
        );
        assert_eq!(
            assess_memory_estimate(&estimate_peak_memory(cap + 1, 4_800, 1, 32), &budget),
            MetalMemoryDecision::Block
        );
    }

    #[test]
    fn preflight_initial_gaussian_cap_reserves_litegs_headroom() {
        assert_eq!(
            preflight_initial_gaussian_cap(TrainingProfile::LiteGsMacV1, 552),
            525
        );
        assert_eq!(
            preflight_initial_gaussian_cap(TrainingProfile::LegacyMetal, 552),
            552
        );
    }

    #[test]
    fn downsample_gaussian_map_evenly_spreads_samples_across_map() {
        let mut map = GaussianMap::from_gaussians(
            (0..10)
                .map(|idx| {
                    let mut gaussian = crate::Gaussian3D::default();
                    gaussian.position.x = idx as f32;
                    gaussian
                })
                .collect(),
        );

        downsample_gaussian_map_evenly(&mut map, 4);

        let sampled_positions: Vec<f32> = map
            .gaussians()
            .iter()
            .map(|gaussian| gaussian.position.x)
            .collect();
        assert_eq!(sampled_positions, vec![0.0, 2.0, 5.0, 7.0]);
    }

    #[test]
    fn resolve_chunk_memory_budget_caps_requested_budget_to_system_limit() {
        let system_budget = MetalMemoryBudget {
            safe_bytes: 10 * GIB,
            physical_bytes: Some(16 * GIB),
        };
        let resolved = resolve_chunk_memory_budget(12 * GIB, system_budget);
        assert_eq!(resolved.safe_bytes, 10 * GIB);
        assert_eq!(resolved.physical_bytes, Some(16 * GIB));
    }

    #[test]
    fn gib_to_bytes_rejects_non_positive_values() {
        assert_eq!(gib_to_bytes(0.0), 0);
        assert_eq!(gib_to_bytes(-1.0), 0);
    }

    #[test]
    fn chunk_capacity_marks_over_budget_requests_for_subdivision() {
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            metal_render_scale: 1.0,
            max_initial_gaussians: 57_474,
            ..TrainingConfig::default()
        };
        let dataset = TrainingDataset {
            intrinsics: crate::Intrinsics::from_focal(500.0, 1920, 1080),
            depth_scale: 1000.0,
            poses: vec![crate::ScenePose::new(
                0,
                std::path::PathBuf::from("frame.png"),
                crate::SE3::identity(),
                0.0,
            )],
            initial_points: Vec::new(),
        };
        let estimate = estimate_chunk_capacity(&dataset, &config).unwrap();
        assert!(estimate.requires_subdivision_or_degradation());
        assert!(estimate.affordable_initial_gaussians < estimate.requested_initial_gaussians);
        assert!(estimate
            .recommendations()
            .first()
            .expect("recommendations should not be empty")
            .contains("subdivide the chunk"));
    }

    #[test]
    fn chunk_capacity_uses_existing_initial_points_as_requested_scale() {
        let config = TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 1.0,
            max_initial_gaussians: 16,
            ..TrainingConfig::default()
        };
        let dataset = TrainingDataset {
            intrinsics: crate::Intrinsics::from_focal(500.0, 32, 32),
            depth_scale: 1000.0,
            poses: vec![crate::ScenePose::new(
                0,
                std::path::PathBuf::from("frame.png"),
                crate::SE3::identity(),
                0.0,
            )],
            initial_points: vec![([0.0, 0.0, 1.0], None); 64],
        };
        let estimate = estimate_chunk_capacity(&dataset, &config).unwrap();
        assert_eq!(estimate.requested_initial_gaussians, 16);
    }

    #[test]
    fn profile_tracks_visible_gaussians() {
        let render = MetalRenderProfile {
            visible_gaussians: 12,
            total_gaussians: 40,
            ..Default::default()
        };
        let profile = MetalStepProfile::from_render(render);
        assert_eq!(profile.visible_gaussians, 12);
        assert_eq!(profile.total_gaussians, 40);
    }

    #[test]
    fn chunk_rect_area_matches_bounds() {
        let rect = ScreenRect {
            min_x: 2,
            max_x: 5,
            min_y: 3,
            max_y: 4,
        };
        assert_eq!(rect.max_x - rect.min_x + 1, 4);
        assert_eq!(rect.max_y - rect.min_y + 1, 2);
    }

    #[test]
    fn tile_bins_only_include_overlapping_gaussians() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let projected = ProjectedGaussians {
            source_indices: Tensor::from_slice(&[0u32, 1], 2, &device).unwrap(),
            u: Tensor::from_slice(&[8.0f32, 18.0], 2, &device).unwrap(),
            v: Tensor::from_slice(&[8.0f32, 8.0], 2, &device).unwrap(),
            sigma_x: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
            sigma_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
            raw_sigma_x: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
            raw_sigma_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
            depth: Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap(),
            opacity: Tensor::from_slice(&[0.5f32, 0.5], 2, &device).unwrap(),
            opacity_logits: Tensor::from_slice(&[0.0f32, 0.0], 2, &device).unwrap(),
            scale3d: Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &device)
                .unwrap(),
            colors: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device)
                .unwrap(),
            min_x: Tensor::from_slice(&[2.0f32, 14.0], 2, &device).unwrap(),
            max_x: Tensor::from_slice(&[15.0f32, 18.0], 2, &device).unwrap(),
            min_y: Tensor::from_slice(&[1.0f32, 1.0], 2, &device).unwrap(),
            max_y: Tensor::from_slice(&[14.0f32, 14.0], 2, &device).unwrap(),
            visible_source_indices: vec![0, 1],
            visible_count: 2,
            tile_bins: MetalTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        };

        let bins = trainer.build_tile_bins(&projected).unwrap();
        assert_eq!(bins.active_tile_count(), 2);
        assert_eq!(bins.total_assignments(), 3);
        assert_eq!(bins.max_gaussians_per_tile(), 2);
        assert_eq!(bins.indices_for_tile(0), &[0, 1]);
        assert_eq!(bins.indices_for_tile(1), &[1]);
    }

    #[test]
    fn native_forward_matches_baseline_on_tiny_scene() {
        let Ok(device) = crate::try_metal_device() else {
            return;
        };
        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            1.0,
            1.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.runtime.stage_camera(&camera).unwrap();
        let projected = ProjectedGaussians {
            source_indices: Tensor::from_slice(&[0u32, 1], 2, &device).unwrap(),
            u: Tensor::from_slice(&[8.0f32, 10.0], 2, &device).unwrap(),
            v: Tensor::from_slice(&[8.0f32, 8.5], 2, &device).unwrap(),
            sigma_x: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
            sigma_y: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
            raw_sigma_x: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
            raw_sigma_y: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
            depth: Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap(),
            opacity: Tensor::from_slice(&[0.6f32, 0.4], 2, &device).unwrap(),
            opacity_logits: Tensor::from_slice(&[0.0f32, 0.0], 2, &device).unwrap(),
            scale3d: Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &device)
                .unwrap(),
            colors: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device)
                .unwrap(),
            min_x: Tensor::from_slice(&[2.0f32, 3.0], 2, &device).unwrap(),
            max_x: Tensor::from_slice(&[14.0f32, 17.0], 2, &device).unwrap(),
            min_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
            max_y: Tensor::from_slice(&[14.0f32, 15.0], 2, &device).unwrap(),
            visible_source_indices: vec![0, 1],
            visible_count: 2,
            tile_bins: MetalTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        };

        let tile_bins = trainer.build_tile_bins(&projected).unwrap();
        let (baseline, _) = trainer.rasterize(&projected, &tile_bins).unwrap();
        let parity = trainer
            .profile_native_forward(&projected, &tile_bins, &baseline)
            .unwrap();

        assert!(
            parity.color_max_abs < 5e-4,
            "color diff={}",
            parity.color_max_abs
        );
        assert!(
            parity.depth_max_abs < 5e-4,
            "depth diff={}",
            parity.depth_max_abs
        );
        assert!(
            parity.alpha_max_abs < 5e-4,
            "alpha diff={}",
            parity.alpha_max_abs
        );
    }

    #[test]
    fn metal_visible_set_stays_on_device() {
        let Ok(device) = crate::try_metal_device() else {
            return;
        };
        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            1.0,
            1.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0, 1.0, 0.0, 3.0],
            &[
                0.1f32.ln(),
                0.1f32.ln(),
                0.1f32.ln(),
                0.1f32.ln(),
                0.1f32.ln(),
                0.1f32.ln(),
            ],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();

        let (projected, _) = trainer
            .project_gaussians(&gaussians, &camera, false, true, None)
            .unwrap();

        assert!(matches!(
            projected.staging_source,
            ProjectionStagingSource::RuntimeBufferRead
        ));
        assert_eq!(projected.visible_count, 2);
        assert_eq!(projected.visible_source_indices, vec![0, 1]);
    }

    #[test]
    fn prune_interval_is_independent_from_densify_interval() {
        let trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                densify_interval: 128,
                prune_interval: 200,
                topology_warmup: 0,
                ..TrainingConfig::default()
            },
            Device::Cpu,
        )
        .unwrap();

        assert!(trainer.should_prune_at(200));
        assert!(!trainer.should_densify_at(200));
        assert!(trainer.should_densify_at(128));
        assert!(!trainer.should_prune_at(128));
    }

    #[test]
    fn profile_schedule_honors_interval() {
        assert!(should_profile_iteration(true, 25, 0));
        assert!(should_profile_iteration(true, 25, 4));
        assert!(!should_profile_iteration(true, 25, 5));
        assert!(should_profile_iteration(true, 25, 25));
        assert!(!should_profile_iteration(false, 25, 25));
    }

    #[test]
    fn summarized_final_loss_uses_last_epoch_mean() {
        let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
        let metrics = summarize_training_metrics(&history, 2);
        assert!((metrics.final_loss - 0.55).abs() < 1e-6);
        assert!((metrics.final_step_loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn project_gaussians_handles_zero_visible_without_index_select() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            1.0,
            1.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, -1.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let (projected, profile) = trainer
            .project_gaussians(&gaussians, &camera, false, true, None)
            .unwrap();

        assert_eq!(profile.total_gaussians, 1);
        assert_eq!(profile.visible_gaussians, 0);
        assert_eq!(projected.source_indices.dim(0).unwrap(), 0);
        assert_eq!(projected.u.dim(0).unwrap(), 0);
        assert_eq!(projected.colors.dim(0).unwrap(), 0);
    }

    #[test]
    fn project_gaussians_keeps_distinct_visible_indices_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let trainer_config = TrainingConfig {
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            16.0,
            16.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[
                0.0, 0.0, 1.0, //
                0.1, 0.0, 0.5, //
                -0.1, 0.0, 2.0,
            ],
            &[
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            &[
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0,
            ],
            &[0.0, 0.0, 0.0],
            &[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            &device,
        )
        .unwrap();

        let (projected, profile) = trainer
            .project_gaussians(&gaussians, &camera, false, true, None)
            .unwrap();
        let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

        assert_eq!(profile.visible_gaussians, 3);
        assert_eq!(source_indices, vec![1, 0, 2]);
    }

    #[test]
    fn project_gaussians_applies_cluster_visible_mask_on_metal() {
        let device = crate::preferred_device();
        if !device.is_metal() {
            return;
        }

        let trainer_config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                cluster_size: 1,
                ..LiteGsConfig::default()
            },
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let camera = DiffCamera::new(
            16.0,
            16.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[
                0.0, 0.0, 1.0, //
                0.1, 0.0, 0.5, //
                -0.1, 0.0, 2.0,
            ],
            &[
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            &[
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0,
            ],
            &[0.0, 0.0, 0.0],
            &[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            &device,
        )
        .unwrap();
        let cluster_visible_mask = [true, false, true];

        let (projected, profile) = trainer
            .project_gaussians(
                &gaussians,
                &camera,
                false,
                true,
                Some(cluster_visible_mask.as_slice()),
            )
            .unwrap();
        let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

        assert_eq!(profile.visible_gaussians, 2);
        assert_eq!(projected.visible_source_indices, vec![0, 2]);
        assert_eq!(source_indices, vec![0, 2]);
    }

    #[test]
    fn topology_updates_can_grow_beyond_initial_gaussian_count_limit() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            densify_interval: 1,
            prune_interval: 0,
            topology_warmup: 0,
            prune_threshold: 0.05,
            max_initial_gaussians: 2,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
            ],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[2.0, 2.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                age: 1,
                ..Default::default()
            },
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                age: 1,
                ..Default::default()
            },
        ];
        trainer.iteration = 1;

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert!(gaussians.len() > 2);
    }

    #[test]
    fn topology_updates_preserve_sh_representation_for_litegs_trainables() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            densify_interval: 1,
            prune_interval: 0,
            topology_warmup: 0,
            prune_threshold: 0.05,
            max_initial_gaussians: 2,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        trainer.topology_memory_budget = None;
        let mut gaussians = TrainableGaussians::new_with_sh(
            &[0.0, 0.0, 1.0],
            &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[2.0],
            &[
                crate::diff::diff_splat::rgb_to_sh0_value(0.2),
                crate::diff::diff_splat::rgb_to_sh0_value(0.4),
                crate::diff::diff_splat::rgb_to_sh0_value(0.6),
            ],
            &vec![0.0; 15 * 3],
            3,
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 1,
            ..Default::default()
        }];
        trainer.iteration = 1;

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert!(gaussians.len() > 1);
        assert!(gaussians.uses_spherical_harmonics());
        assert_eq!(gaussians.sh_degree(), 3);
        assert_eq!(gaussians.sh_rest().dims()[0], gaussians.len());
    }

    #[test]
    fn topology_update_densifies_and_prunes_with_matching_adam_state() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            densify_interval: 1,
            prune_interval: 1,
            topology_warmup: 0,
            prune_threshold: 0.05,
            max_initial_gaussians: 4,
            metal_render_scale: 1.0,
            ..TrainingConfig::default()
        };
        let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
        let mut gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
            ],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[2.0, -10.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();
        trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
        trainer.gaussian_stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                age: 5,
                ..Default::default()
            },
            MetalGaussianStats {
                mean2d_grad: RunningMoments::default(),
                age: 7,
                ..Default::default()
            },
        ];
        trainer.iteration = 1;

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 0, 1)
            .unwrap();

        assert_eq!(gaussians.len(), 2);
        assert_eq!(trainer.gaussian_stats.len(), 2);
        assert!(trainer.gaussian_stats.iter().any(|stats| stats.age == 0));
        let opacities = gaussians.opacities().unwrap().to_vec1::<f32>().unwrap();
        assert!(opacities
            .iter()
            .all(|opacity| *opacity >= trainer_config.prune_threshold));
        let positions = gaussians.positions().to_vec2::<f32>().unwrap();
        assert!((positions[1][0] - positions[0][0]).abs() > 1e-6);

        let adam = trainer.adam.as_ref().unwrap();
        assert_eq!(adam.m_pos.dim(0).unwrap(), 2);
        assert_eq!(adam.v_pos.dim(0).unwrap(), 2);
        let m_pos = adam.m_pos.to_vec2::<f32>().unwrap();
        assert!(m_pos[1].iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn rebuild_adam_state_preserves_reordered_rows() {
        let device = Device::Cpu;
        let trainer =
            MetalTrainer::new(32, 16, &TrainingConfig::default(), device.clone()).unwrap();
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0, 2.0],
            &[
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.06f32.ln(),
                0.06f32.ln(),
                0.06f32.ln(),
            ],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();
        let mut adam = MetalAdamState::new(&gaussians).unwrap();
        adam.m_pos =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();
        adam.v_pos =
            Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (2, 3), &device).unwrap();

        let reordered = trainer
            .rebuild_adam_state(&adam, &[Some(1), Some(0)])
            .unwrap();
        assert_eq!(
            reordered.m_pos.to_vec2::<f32>().unwrap(),
            vec![vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0]]
        );
        assert_eq!(
            reordered.v_pos.to_vec2::<f32>().unwrap(),
            vec![vec![10.0, 11.0, 12.0], vec![7.0, 8.0, 9.0]]
        );
    }

    #[test]
    fn sync_cluster_assignment_updates_aabbs_from_current_positions() {
        let device = Device::Cpu;
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: super::LiteGsConfig {
                    cluster_size: 2,
                    ..super::LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        trainer.scene_extent = 16.0;
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();

        trainer.sync_cluster_assignment(&gaussians, false).unwrap();
        let initial_aabb = trainer.cluster_assignment.as_ref().unwrap().aabbs[0];

        gaussians
            .positions
            .set(
                &Tensor::from_slice(&[10.0f32, 0.0, 0.0, 11.0, 0.0, 0.0], (2, 3), &device).unwrap(),
            )
            .unwrap();
        trainer.sync_cluster_assignment(&gaussians, false).unwrap();

        let updated_aabb = trainer.cluster_assignment.as_ref().unwrap().aabbs[0];
        assert!(initial_aabb.min[0] < 1.0);
        assert!((updated_aabb.min[0] - 10.0).abs() < 1e-6);
        assert!((updated_aabb.max[0] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn sync_cluster_assignment_reassigns_after_topology_change() {
        let device = Device::Cpu;
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: super::LiteGsConfig {
                    cluster_size: 1,
                    ..super::LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        trainer.scene_extent = 16.0;
        let gaussians_one = TrainableGaussians::new(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        trainer
            .sync_cluster_assignment(&gaussians_one, false)
            .unwrap();
        assert_eq!(
            trainer
                .cluster_assignment
                .as_ref()
                .unwrap()
                .cluster_indices
                .len(),
            1
        );

        let gaussians_two = TrainableGaussians::new(
            &[0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &device,
        )
        .unwrap();
        trainer
            .sync_cluster_assignment(&gaussians_two, true)
            .unwrap();

        let assignment = trainer.cluster_assignment.as_ref().unwrap();
        assert_eq!(assignment.cluster_indices.len(), 2);
        assert_eq!(assignment.cluster_sizes.iter().sum::<usize>(), 2);
        assert_eq!(assignment.aabbs.len(), assignment.num_clusters);
    }

    #[test]
    fn lr_pos_exponential_decay_is_correct() {
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                lr_position: 0.001,
                lr_pos_final: 0.00001,
                iterations: 1000,
                ..TrainingConfig::default()
            },
            Device::Cpu,
        )
        .unwrap();

        // At iteration 0, effective LR equals initial LR.
        let lr_init = trainer.compute_lr_pos();
        assert!(
            (lr_init - 0.001).abs() < 1e-7,
            "at t=0 expected 0.001, got {lr_init}"
        );

        // At iteration = max, effective LR equals final LR.
        trainer.iteration = 1000;
        let lr_end = trainer.compute_lr_pos();
        assert!(
            (lr_end - 0.00001).abs() < 1e-9,
            "at t=T expected 0.00001, got {lr_end}"
        );

        // At t = T/2, effective LR is geometric mean of init and final.
        trainer.iteration = 500;
        let lr_mid = trainer.compute_lr_pos();
        let expected_mid = (0.001f32 * 0.00001f32).sqrt();
        assert!(
            (lr_mid - expected_mid).abs() < expected_mid * 1e-4,
            "at t=T/2 expected {expected_mid}, got {lr_mid}"
        );

        // LR must strictly decrease over time.
        trainer.iteration = 100;
        let lr_100 = trainer.compute_lr_pos();
        trainer.iteration = 900;
        let lr_900 = trainer.compute_lr_pos();
        assert!(lr_100 > lr_900, "LR should decrease: {lr_100} > {lr_900}");
    }

    #[test]
    fn sh_render_colors_follow_view_direction_terms() {
        let device = Device::Cpu;
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        trainer.active_sh_degree = 1;
        let camera = DiffCamera::new(
            16.0,
            16.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let mut sh_rest = vec![0.0f32; 3 * 3];
        sh_rest[0] = -0.5;
        let gaussians = TrainableGaussians::new_with_sh(
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[0.0, 0.0, 0.0],
            &sh_rest,
            1,
            &device,
        )
        .unwrap();

        let positions = gaussians.positions().detach();
        let colors = trainer
            .render_colors_for_camera(&gaussians, &positions, &camera)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected_red = 0.5 + (-SH_C1) * -0.5;

        assert!((colors[0][0] - expected_red).abs() < 1e-5);
        assert!((colors[0][1] - 0.5).abs() < 1e-6);
        assert!((colors[0][2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn litegs_scale_regularization_uses_activated_scales() {
        let device = Device::Cpu;
        let trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: super::LiteGsConfig {
                    reg_weight: 0.5,
                    ..super::LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        let visible_log_scales =
            Tensor::from_slice(&[2.0f32.ln(), 1.0f32.ln(), 0.5f32.ln()], (1, 3), &device).unwrap();

        let term = trainer
            .litegs_scale_regularization_term(&visible_log_scales)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let grad = trainer
            .litegs_scale_regularization_grad(&visible_log_scales)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert!((term - 1.75).abs() < 1e-6);
        assert!((grad[0][0] - (4.0 / 3.0)).abs() < 1e-6);
        assert!((grad[0][1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((grad[0][2] - (1.0 / 12.0)).abs() < 1e-6);
    }

    #[test]
    fn pose_parameter_grads_returns_tensor_pair() {
        let device = Device::Cpu;
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                litegs: super::LiteGsConfig {
                    learnable_viewproj: true,
                    lr_pose: 1e-4,
                    ..super::LiteGsConfig::default()
                },
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        let camera = DiffCamera::new(
            16.0,
            16.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 3.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.5, 0.25],
            &device,
        )
        .unwrap();
        let (rendered, _, _) = trainer
            .render(&gaussians, &camera, false, true, None)
            .unwrap();
        let target_color_cpu = rendered
            .color
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let target_depth_cpu = rendered
            .depth
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let frame = MetalTrainingFrame {
            camera: camera.clone(),
            target_color: rendered.color.clone(),
            target_depth: rendered.depth.clone(),
            target_color_cpu,
            target_depth_cpu,
        };
        trainer.pose_embeddings = Some(
            crate::training::pose_embedding::PoseEmbeddings::from_dataset(
                &[crate::ScenePose::new(
                    0,
                    std::path::PathBuf::from("frame.png"),
                    crate::SE3::identity(),
                    0.0,
                )],
                1e-4,
                &device,
            )
            .unwrap(),
        );

        let (quaternion_grad, translation_grad) = trainer
            .pose_parameter_grads(&gaussians, &frame, 0)
            .unwrap()
            .unwrap();
        let quaternion_grad = quaternion_grad.to_vec1::<f32>().unwrap();
        let translation_grad = translation_grad.to_vec1::<f32>().unwrap();

        assert_eq!(quaternion_grad.len(), 4);
        assert_eq!(translation_grad.len(), 3);
        assert!(quaternion_grad.iter().all(|value| value.is_finite()));
        assert!(translation_grad.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sh_parameter_grads_populate_sh_rest_terms() {
        let device = Device::Cpu;
        let mut trainer = MetalTrainer::new(
            32,
            16,
            &TrainingConfig {
                training_profile: TrainingProfile::LiteGsMacV1,
                ..TrainingConfig::default()
            },
            device.clone(),
        )
        .unwrap();
        trainer.active_sh_degree = 1;
        let camera = DiffCamera::new(
            16.0,
            16.0,
            16.0,
            8.0,
            32,
            16,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let gaussians = TrainableGaussians::new_with_sh(
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[
                crate::diff::diff_splat::rgb_to_sh0_value(0.5),
                crate::diff::diff_splat::rgb_to_sh0_value(0.5),
                crate::diff::diff_splat::rgb_to_sh0_value(0.5),
            ],
            &vec![0.0; 3 * 3],
            1,
            &device,
        )
        .unwrap();
        let projected = ProjectedGaussians {
            source_indices: Tensor::from_slice(&[0u32], 1, &device).unwrap(),
            u: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            v: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
            sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
            raw_sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
            raw_sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
            depth: Tensor::ones((1,), DType::F32, &device).unwrap(),
            opacity: Tensor::ones((1,), DType::F32, &device).unwrap(),
            opacity_logits: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            scale3d: Tensor::ones((1, 3), DType::F32, &device).unwrap(),
            colors: Tensor::from_slice(&[0.5f32, 0.5, 0.5], (1, 3), &device).unwrap(),
            min_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            max_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            min_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            max_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
            visible_source_indices: vec![0],
            visible_count: 1,
            tile_bins: MetalTileBins::default(),
            staging_source: ProjectionStagingSource::TensorReadback,
        };
        let render_grads = Tensor::from_slice(&[1.0f32, 2.0, -0.5], (1, 3), &device).unwrap();

        let (sh_0_grads, sh_rest_grads) = trainer
            .parameter_grads_from_render_color_grads(&gaussians, &projected, &render_grads, &camera)
            .unwrap();
        let sh_0_grads = sh_0_grads.to_vec2::<f32>().unwrap();
        let sh_rest_grads = sh_rest_grads.to_vec3::<f32>().unwrap();
        let expected_basis = -SH_C1;

        assert!((sh_0_grads[0][0] - SH_C0).abs() < 1e-6);
        assert!((sh_0_grads[0][1] - 2.0 * SH_C0).abs() < 1e-6);
        assert!((sh_0_grads[0][2] + 0.5 * SH_C0).abs() < 1e-6);
        assert!((sh_rest_grads[0][0][0] - expected_basis).abs() < 1e-6);
        assert!((sh_rest_grads[0][0][1] - 2.0 * expected_basis).abs() < 1e-6);
        assert!((sh_rest_grads[0][0][2] + 0.5 * expected_basis).abs() < 1e-6);
        assert!(sh_rest_grads[0][1].iter().all(|value| value.abs() < 1e-6));
        assert!(sh_rest_grads[0][2].iter().all(|value| value.abs() < 1e-6));
    }
}
