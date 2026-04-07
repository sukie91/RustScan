//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, and optimizer
//! updates on the Metal device, while replacing the generic autograd hot path
//! with a specialized analytical backward pass.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[cfg(test)]
use candle_core::Var;
use candle_core::{DType, Device, Tensor};

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians, SH_C0};
use crate::training::clustering::ClusterAssignment;
use crate::{GaussianMap, TrainingDataset, TrainingError};

use super::data_loading::{
    load_training_data, map_from_trainable, trainable_from_map, LoadedTrainingData,
};
use super::eval::{scaled_dimensions, summarize_training_metrics};
use super::frame_targets::{resize_depth, resize_rgb};
use super::metal_backward::{
    self as metal_backward, MetalBackwardRequest, MetalParameterGradInputs, MetalParameterGrads,
};
#[cfg(test)]
use super::metal_backward::{MetalBackwardGrads, MetalBackwardLossScales};
use super::metal_forward::{
    self as metal_forward, scale_camera, MetalForwardContext, MetalForwardInputs,
    MetalForwardSettings, MetalRenderProfile, NativeParityProfile, ProjectedGaussians,
    RenderedFrame,
};
#[cfg(test)]
use super::metal_loss::{
    depth_backward_scale, scale_regularization_grad as test_scale_regularization_grad,
    scale_regularization_term, ssim_gradient,
};
use super::metal_loss::{evaluate_training_step_loss, scale_regularization_grad, MetalLossConfig};
use super::metal_optimizer::{self as metal_optimizer, MetalAdamState, MetalOptimizerConfig};
use super::metal_runtime::{MetalBufferSlot, MetalRuntime};
use super::parity_harness::{ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics};
use super::splats::Splats;
use super::topology::{
    self, LiteGsDensifySelection, MetalGaussianStats, RunningMoments, TopologyAnalysis,
    TopologyCandidateInfo, TopologyPolicy, TopologyStepContext,
};
use super::{LiteGsConfig, TrainingConfig, TrainingProfile};

#[cfg(test)]
use super::metal_forward::projected_axis_aligned_sigmas;
#[cfg(test)]
use super::metal_forward::{ProjectionStagingSource, TileBinningStats};
#[cfg(test)]
use super::metal_runtime::MetalTileBins;
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
const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
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

pub(crate) struct MetalTrainingFrame {
    camera: DiffCamera,
    target_color: Tensor,
    target_depth: Tensor,
    target_color_cpu: Vec<f32>,
    target_depth_cpu: Vec<f32>,
}

type GaussianParameterSnapshot = Splats;

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

pub(crate) struct MetalTrainingStats {
    pub(crate) final_loss: f32,
    pub(crate) final_step_loss: f32,
    pub(crate) telemetry: LiteGsTrainingTelemetry,
}

pub(crate) struct MetalStepOutcome {
    loss: f32,
    visible_gaussians: usize,
    total_gaussians: usize,
    profile: Option<MetalStepProfile>,
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
pub(crate) struct MetalStepProfile {
    pub(crate) projection: Duration,
    pub(crate) sorting: Duration,
    pub(crate) rasterization: Duration,
    native_forward: Option<NativeParityProfile>,
    pub(crate) loss: Duration,
    pub(crate) backward: Duration,
    pub(crate) optimizer: Duration,
    pub(crate) total: Duration,
    pub(crate) visible_gaussians: usize,
    pub(crate) total_gaussians: usize,
    pub(crate) active_tiles: usize,
    pub(crate) tile_gaussian_refs: usize,
    pub(crate) max_gaussians_per_tile: usize,
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

impl MetalStepOutcome {
    pub(crate) fn profile_summary(&self) -> Option<MetalStepProfile> {
        self.profile
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

    fn topology_policy(&self) -> TopologyPolicy {
        TopologyPolicy {
            training_profile: self.training_profile,
            litegs: self.litegs.clone(),
            prune_threshold: self.prune_threshold,
            densify_interval: self.densify_interval,
            prune_interval: self.prune_interval,
            topology_warmup: self.topology_warmup,
            topology_log_interval: self.topology_log_interval,
            legacy_densify_grad_threshold: self.legacy_densify_grad_threshold,
            legacy_clone_scale_threshold: self.legacy_clone_scale_threshold,
            legacy_split_scale_threshold: self.legacy_split_scale_threshold,
            legacy_prune_scale_threshold: self.legacy_prune_scale_threshold,
            legacy_max_densify_per_update: self.legacy_max_densify_per_update,
            max_gaussian_budget: self.max_gaussian_budget,
            scene_extent: self.scene_extent,
            max_iterations: self.max_iterations,
        }
    }

    fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        self.topology_policy().litegs_total_epochs(frame_count)
    }

    fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        self.topology_policy()
            .litegs_densify_until_epoch(frame_count)
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

    fn export_snapshot(
        &self,
        gaussians: &TrainableGaussians,
    ) -> candle_core::Result<GaussianParameterSnapshot> {
        GaussianParameterSnapshot::from_trainable(gaussians)
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
        adam.reset_moments(only_opacity)
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

    fn litegs_requested_additions(
        &self,
        infos: &[TopologyCandidateInfo],
        allow_extra_growth: bool,
    ) -> usize {
        topology::litegs_requested_additions(
            infos,
            self.litegs.growth_select_fraction,
            allow_extra_growth,
        )
    }

    fn litegs_select_densify_candidates(
        &self,
        infos: &[TopologyCandidateInfo],
        max_new: usize,
        allow_extra_growth: bool,
    ) -> LiteGsDensifySelection {
        topology::litegs_select_densify_candidates(
            infos,
            max_new,
            self.litegs.growth_select_fraction,
            allow_extra_growth,
        )
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
        Ok(topology::analyze_topology_candidates(
            &self.topology_policy(),
            &Splats::from_trainable(gaussians)?,
            stats,
        ))
    }

    fn densify_snapshot(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
        max_gaussians: usize,
    ) -> usize {
        topology::densify_snapshot(
            &self.topology_policy(),
            snapshot,
            stats,
            origins,
            infos,
            max_gaussians,
        )
    }

    fn prune_snapshot(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
    ) -> usize {
        topology::prune_snapshot(&self.topology_policy(), snapshot, stats, origins, infos)
    }

    fn densify_snapshot_litegs(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        max_gaussians: usize,
        selected_indices: &[usize],
    ) -> usize {
        topology::densify_snapshot_litegs(
            &self.topology_policy(),
            snapshot,
            stats,
            origins,
            max_gaussians,
            selected_indices,
        )
    }

    fn prune_snapshot_litegs(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
        infos: &[TopologyCandidateInfo],
    ) -> usize {
        topology::prune_snapshot_litegs(snapshot, stats, origins, infos)
    }

    fn rebuild_adam_state(
        &self,
        old_state: &MetalAdamState,
        origins: &[Option<usize>],
    ) -> candle_core::Result<MetalAdamState> {
        metal_optimizer::rebuild_adam_state(&self.device, old_state, origins)
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

    fn loss_config(&self) -> MetalLossConfig {
        let (color_weight, ssim_weight, depth_weight) = self.loss_weights();
        MetalLossConfig {
            color_weight,
            ssim_weight,
            depth_weight,
            scale_regularization_weight: if self.is_litegs_mode() {
                self.litegs.reg_weight
            } else {
                0.0
            },
            enable_transmittance_loss: self.is_litegs_mode() && self.litegs.enable_transmittance,
            render_width: self.render_width,
            render_height: self.render_height,
        }
    }

    fn total_loss_for_render_result(
        &self,
        gaussians: &TrainableGaussians,
        rendered: &RenderedFrame,
        projected: &ProjectedGaussians,
        frame: &MetalTrainingFrame,
    ) -> candle_core::Result<f32> {
        evaluate_training_step_loss(
            gaussians,
            rendered,
            projected,
            &frame.target_color,
            &frame.target_depth,
            &frame.target_color_cpu,
            &frame.target_depth_cpu,
            self.loss_config(),
        )
        .map(|context| context.total_loss)
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
        let policy = self.topology_policy();
        let schedule = topology::schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: self.iteration,
                frame_count,
            },
        );
        let should_log_topology = policy.should_log_topology(self.iteration);

        let completed_epoch = schedule.completed_epoch;
        let mut should_reset_opacity = schedule.reset_opacity;
        let allow_extra_growth = schedule.allow_extra_growth;
        let (mut should_densify, mut should_prune) = (schedule.densify, schedule.prune);
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
        let requested_cap =
            topology::requested_gaussian_cap(&policy, old_len, litegs_requested_additions);
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
            if should_log_topology {
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
            if should_log_topology {
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
            if should_log_topology || guardrail_triggered {
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

        let rebuilt = snapshot.to_trainable(&self.device)?;
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
        let source_pixel_count = pixel_count;
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

    pub(crate) fn prepare_frames(
        &self,
        loaded: &LoadedTrainingData,
    ) -> Result<Vec<MetalTrainingFrame>, TrainingError> {
        let mut frames = Vec::with_capacity(loaded.cameras.len());
        for idx in 0..loaded.cameras.len() {
            let src_camera = &loaded.cameras[idx];
            let target_color_cpu = if loaded.target_width == self.render_width
                && loaded.target_height == self.render_height
            {
                loaded.colors[idx].clone()
            } else {
                resize_rgb(
                    &loaded.colors[idx],
                    loaded.target_width,
                    loaded.target_height,
                    self.render_width,
                    self.render_height,
                )
            };
            let target_depth_cpu = if loaded.target_width == self.render_width
                && loaded.target_height == self.render_height
            {
                loaded.depths[idx].clone()
            } else {
                resize_depth(
                    &loaded.depths[idx],
                    loaded.target_width,
                    loaded.target_height,
                    self.render_width,
                    self.render_height,
                )
            };
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

    pub(crate) fn initialize_training_session(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frames: &[MetalTrainingFrame],
    ) -> candle_core::Result<()> {
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
        Ok(())
    }

    pub(crate) fn train(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frames: &[MetalTrainingFrame],
        max_iterations: usize,
    ) -> candle_core::Result<MetalTrainingStats> {
        self.initialize_training_session(gaussians, frames)?;
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

    pub(crate) fn training_step(
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
        let collect_visible_indices = topology::should_collect_visible_indices(
            &self.topology_policy(),
            topology::schedule_topology(
                &self.topology_policy(),
                TopologyStepContext {
                    iteration: self.iteration,
                    frame_count,
                },
            ),
        );

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
        let step_loss = evaluate_training_step_loss(
            gaussians,
            &rendered,
            &projected,
            &frame.target_color,
            &frame.target_depth,
            &frame.target_color_cpu,
            &frame.target_depth_cpu,
            self.loss_config(),
        )?;
        self.last_loss_terms = step_loss.telemetry.loss_terms.clone();
        self.last_depth_valid_pixels = step_loss.telemetry.depth_valid_pixels;
        self.last_depth_grad_scale = step_loss.telemetry.depth_grad_scale;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        let refresh_target_buffers = self.cached_target_frame_idx != Some(frame_idx);
        let backward = metal_backward::execute_backward_pass(
            &mut self.runtime,
            MetalBackwardRequest {
                tile_bins: &projected.tile_bins,
                n_gaussians: gaussians.len(),
                camera: &render_camera,
                target_color_cpu: &frame.target_color_cpu,
                target_depth_cpu: &frame.target_depth_cpu,
                ssim_grads: &step_loss.ssim_grads,
                loss_scales: step_loss.backward_loss_scales,
                refresh_target_buffers,
            },
        )?;
        if refresh_target_buffers {
            self.cached_target_frame_idx = Some(frame_idx);
        }
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
        let scale_reg_grad =
            if self.is_litegs_mode() && self.litegs.reg_weight > 0.0 && projected.visible_count > 0
            {
                let visible_log_scales = gaussians
                    .scales
                    .as_tensor()
                    .index_select(&projected.source_indices, 0)?;
                let visible_reg_grad =
                    scale_regularization_grad(&visible_log_scales, self.litegs.reg_weight)?;
                Some(Tensor::zeros_like(gaussians.scales.as_tensor())?.index_add(
                    &projected.source_indices,
                    &visible_reg_grad,
                    0,
                )?)
            } else {
                None
            };
        let parameter_grads = metal_backward::assemble_parameter_grads(
            &self.device,
            MetalParameterGradInputs {
                gaussians,
                raw_grads: &backward.grads,
                projected: &projected,
                rendered: &rendered,
                rendered_color_cpu: &step_loss.rendered_color_cpu,
                target_color_cpu: &frame.target_color_cpu,
                target_depth_cpu: &frame.target_depth_cpu,
                ssim_grads: &step_loss.ssim_grads,
                loss_scales: step_loss.backward_loss_scales,
                camera: &render_camera,
                active_sh_degree: self.active_sh_degree,
                render_width: self.render_width,
                render_height: self.render_height,
                include_rotation_grads: self.lr_rotation > 0.0 && projected.visible_count > 0,
            },
        )?;
        let pose_parameter_grads = self.pose_parameter_grads(gaussians, frame, frame_idx)?;
        self.apply_parameter_grads(
            gaussians,
            &parameter_grads,
            &projected,
            effective_lr_pos,
            scale_reg_grad.as_ref(),
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
        self.loss_history.push(step_loss.total_loss);

        Ok(MetalStepOutcome {
            loss: step_loss.total_loss,
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

    #[cfg(test)]
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
        metal_backward::rotation_parameter_grads(
            &self.device,
            gaussians,
            projected,
            rendered,
            rendered_color_cpu,
            target_color_cpu,
            target_depth_cpu,
            ssim_grads,
            loss_scales,
            camera,
            self.render_width,
            self.render_height,
        )
    }

    #[cfg(test)]
    fn parameter_grads_from_render_color_grads(
        &self,
        gaussians: &TrainableGaussians,
        projected: &ProjectedGaussians,
        render_color_grads: &Tensor,
        camera: &DiffCamera,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        metal_backward::parameter_grads_from_render_color_grads(
            &self.device,
            self.active_sh_degree,
            gaussians,
            projected,
            render_color_grads,
            camera,
        )
    }

    fn apply_parameter_grads(
        &mut self,
        gaussians: &mut TrainableGaussians,
        parameter_grads: &MetalParameterGrads,
        projected: &ProjectedGaussians,
        effective_lr_pos: f32,
        scale_reg_grad: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let optimizer_config = MetalOptimizerConfig {
            effective_lr_pos,
            lr_scale: self.lr_scale,
            lr_rotation: self.lr_rotation,
            lr_opacity: self.lr_opacity,
            lr_color: self.lr_color,
            lr_sh_rest: self.lr_sh_rest,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            step: self.iteration,
            use_sparse_updates: self.is_litegs_mode() && self.litegs.sparse_grad,
        };
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;
        let scale_grads = if let Some(extra) = scale_reg_grad {
            parameter_grads.log_scales.broadcast_add(extra)?
        } else {
            parameter_grads.log_scales.clone()
        };
        let parameter_grads = MetalParameterGrads {
            positions: parameter_grads.positions.clone(),
            log_scales: scale_grads,
            rotations: parameter_grads.rotations.clone(),
            opacity_logits: parameter_grads.opacity_logits.clone(),
            colors: parameter_grads.colors.clone(),
            sh_rest: parameter_grads.sh_rest.clone(),
        };
        metal_optimizer::apply_optimizer_step(
            gaussians,
            &mut self.runtime,
            adam,
            &parameter_grads,
            Some(&projected.source_indices),
            optimizer_config,
        )
    }

    #[cfg(test)]
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
        let parameter_grads = MetalParameterGrads {
            positions: grads.positions.clone(),
            log_scales: grads.log_scales.clone(),
            rotations: rotation_parameter_grads.cloned(),
            opacity_logits: grads.opacity_logits.clone(),
            colors: color_parameter_grads,
            sh_rest: sh_rest_parameter_grads,
        };
        self.apply_parameter_grads(
            gaussians,
            &parameter_grads,
            projected,
            effective_lr_pos,
            scale_reg_grad,
        )
    }

    fn forward_settings(&self) -> MetalForwardSettings {
        MetalForwardSettings {
            pixel_count: self.pixel_count,
            render_width: self.render_width,
            render_height: self.render_height,
            chunk_size: self.chunk_size,
            use_native_forward: self.use_native_forward,
            litegs_mode: self.is_litegs_mode(),
        }
    }

    fn render(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(RenderedFrame, ProjectedGaussians, MetalRenderProfile)> {
        let positions = gaussians.positions().detach();
        let render_colors = self.render_colors_for_camera(gaussians, &positions, camera)?;

        if should_profile && self.device.is_metal() && gaussians.len() > 0 {
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

        let settings = self.forward_settings();
        let mut forward = MetalForwardContext {
            runtime: &mut self.runtime,
            device: &self.device,
            settings,
        };
        metal_forward::execute_forward_pass(
            &mut forward,
            MetalForwardInputs {
                gaussians,
                positions: &positions,
                colors: &render_colors,
                camera,
                should_profile,
                collect_visible_indices,
                cluster_visible_mask,
            },
        )
    }

    #[cfg(test)]
    fn project_gaussians(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        let positions = gaussians.positions().detach();
        let render_colors = self.render_colors_for_camera(gaussians, &positions, camera)?;
        let settings = self.forward_settings();
        let mut forward = MetalForwardContext {
            runtime: &mut self.runtime,
            device: &self.device,
            settings,
        };
        metal_forward::project_gaussians(
            &mut forward,
            MetalForwardInputs {
                gaussians,
                positions: &positions,
                colors: &render_colors,
                camera,
                should_profile,
                collect_visible_indices,
                cluster_visible_mask,
            },
        )
    }

    #[cfg(test)]
    fn rasterize(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &MetalTileBins,
    ) -> candle_core::Result<(RenderedFrame, TileBinningStats)> {
        metal_forward::rasterize(
            &mut self.runtime,
            &self.device,
            self.pixel_count,
            self.chunk_size,
            projected,
            tile_bins,
        )
    }

    #[cfg(test)]
    fn build_tile_bins(
        &mut self,
        projected: &ProjectedGaussians,
    ) -> candle_core::Result<MetalTileBins> {
        metal_forward::build_tile_bins(&mut self.runtime, &self.device, projected)
    }

    #[cfg(test)]
    fn profile_native_forward(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &MetalTileBins,
        baseline: &RenderedFrame,
    ) -> candle_core::Result<NativeParityProfile> {
        metal_forward::profile_native_forward(
            &mut self.runtime,
            self.render_width,
            self.render_height,
            projected,
            tile_bins,
            baseline,
        )
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
    let source_pixel_count = pixel_count;
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

#[cfg(test)]
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
    metal_optimizer::adam_step_var_fused(
        var, grad, m, v, runtime, lr, beta1, beta2, eps, step, grad_slot, m_slot, v_slot,
        param_slot,
    )
}

#[cfg(test)]
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
    metal_optimizer::adam_step_var_sparse(var, grad, m, v, row_indices, lr, beta1, beta2, eps, step)
}

#[cfg(test)]
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
    metal_optimizer::adam_updated_tensors(param, grad, m, v, lr, beta1, beta2, eps, step)
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
        let backward_loss_scales = metal_backward::backward_loss_scales(
            0.8,
            0.2,
            frame.target_color_cpu.len(),
            0.0,
            false,
            trainer.pixel_count,
        );
        let backward = metal_backward::execute_backward_pass(
            &mut trainer.runtime,
            MetalBackwardRequest {
                tile_bins: &projected.tile_bins,
                n_gaussians: gaussians.len(),
                camera: &frame.camera,
                target_color_cpu: &frame.target_color_cpu,
                target_depth_cpu: &frame.target_depth_cpu,
                ssim_grads: &ssim_grads,
                loss_scales: backward_loss_scales,
                refresh_target_buffers: true,
            },
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

        let policy = trainer.topology_policy();
        let prune = topology::schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 200,
                frame_count: 1,
            },
        );
        let densify = topology::schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 128,
                frame_count: 1,
            },
        );

        assert!(prune.prune);
        assert!(!prune.densify);
        assert!(densify.densify);
        assert!(!densify.prune);
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

        let term = scale_regularization_term(&visible_log_scales)
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let grad = test_scale_regularization_grad(&visible_log_scales, trainer.litegs.reg_weight)
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
