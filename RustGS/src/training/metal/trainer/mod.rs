//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, and optimizer
//! updates on the Metal device, while replacing the generic autograd hot path
//! with a specialized analytical backward pass.

use std::time::{Duration, Instant};

#[cfg(test)]
use candle_core::DType;
#[cfg(test)]
use candle_core::Var;
use candle_core::{Device, Tensor};

#[cfg(test)]
use crate::diff::diff_splat::SH_C0;
use crate::diff::diff_splat::{DiffCamera, Splats};
use crate::training::clustering::ClusterAssignment;
#[cfg(test)]
use crate::TrainingDataset;
use crate::TrainingError;

use super::backward::{
    self as metal_backward, MetalBackwardRequest, MetalParameterGradInputs, MetalParameterGrads,
};
#[cfg(test)]
use super::backward::{MetalBackwardGrads, MetalBackwardLossScales};
#[cfg(test)]
use super::data_loading::load_training_data;
use super::data_loading::LoadedTrainingData;
use super::eval::{scaled_dimensions, summarize_training_metrics};
use super::forward::{
    self as metal_forward, scale_camera, MetalForwardInputs, MetalForwardSettings,
    MetalRenderProfile, NativeParityProfile, ProjectedGaussians, RenderedFrame,
};
use super::frame_targets::{resize_depth, resize_rgb};
#[cfg(test)]
use super::loss::optional_full_scale_regularization_grad as test_optional_full_scale_regularization_grad;
#[cfg(test)]
use super::loss::{
    depth_backward_scale, scale_regularization_grad as test_scale_regularization_grad,
    scale_regularization_term, ssim_gradient,
};
use super::loss::{
    evaluate_training_step_loss, optional_full_scale_regularization_grad, MetalLossConfig,
};
use super::memory::{
    assess_memory_estimate, estimate_peak_memory_with_source_pixels, training_memory_budget,
    MetalMemoryBudget, MetalMemoryDecision,
};
use super::optimizer::{self as metal_optimizer, MetalAdamState, MetalOptimizerConfig};
use super::parity_harness::{ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics};
use super::pose_embedding;
#[cfg(test)]
use super::runtime::MetalBufferSlot;
use super::runtime::MetalRuntime;
use super::runtime_splats::{apply_topology_plan, TopologySplatMetrics};
#[cfg(test)]
use super::splats::HostSplats;
use super::telemetry::{LiteGsOptimizerLrs, LiteGsTrainingTelemetry};
use super::topology::{
    self, MetalGaussianStats, RunningMoments, TopologyExecutionDisposition,
    TopologyMutationRequest, TopologyPolicy, TopologyStatsAction, TopologyStepContext,
};
use super::{LiteGsConfig, TrainingConfig, TrainingProfile};

mod session;
mod step;
mod support;
mod topology_impl;

use self::session::MetalTrainingFrame;

#[cfg(test)]
use super::entry::effective_metal_config;
#[cfg(test)]
use super::forward::projected_axis_aligned_sigmas;
#[cfg(test)]
use super::forward::ProjectedTileBins;
#[cfg(test)]
use super::forward::{ProjectionStagingSource, TileBinningStats};
#[cfg(test)]
use super::memory::{
    affordable_initial_gaussian_cap, apply_ratio, bytes_to_gib, estimate_chunk_capacity,
    estimate_peak_memory, gib_to_bytes, preflight_initial_gaussian_cap,
    resolve_chunk_memory_budget, DEFAULT_METAL_MEMORY_BUDGET_BYTES, GIB,
    METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR, METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
};
#[cfg(test)]
use super::runtime::ScreenRect;

const LITEGS_LAMBDA_DSSIM: f32 = 0.2;
const LITEGS_DEPTH_LOSS_WEIGHT: f32 = 0.1;
const LITEGS_SH_ACTIVATION_EPOCH_INTERVAL: usize = 5;
const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_OPACITY_DECAY_RATE: f32 = 0.5;
const LITEGS_OPACITY_DECAY_MIN: f32 = 1.0 / 128.0;
const LITEGS_REFINE_OPACITY_DECAY: f32 = 0.004;
const LITEGS_REFINE_SCALE_DECAY: f32 = 0.002;

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

    pub(crate) fn render_dimensions(&self) -> (usize, usize) {
        (self.render_width, self.render_height)
    }

    pub(crate) fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    pub(crate) fn source_pixel_count(&self) -> usize {
        self.source_pixel_count
    }

    pub(crate) fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn set_topology_memory_budget(&mut self, budget: Option<MetalMemoryBudget>) {
        self.topology_memory_budget = budget;
    }

    pub(crate) fn set_max_gaussian_budget(&mut self, budget: usize) {
        self.max_gaussian_budget = budget.max(1);
    }

    pub(crate) fn set_scene_extent(&mut self, scene_extent: f32) {
        self.scene_extent = scene_extent;
    }

    pub(crate) fn set_pose_embeddings(
        &mut self,
        pose_embeddings: crate::training::pose_embedding::PoseEmbeddings,
    ) {
        self.pose_embeddings = Some(pose_embeddings);
    }

    pub(crate) fn set_cluster_assignment(&mut self, cluster_assignment: ClusterAssignment) {
        self.cluster_assignment = Some(cluster_assignment);
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn should_profile_iteration(profile_steps: bool, profile_interval: usize, iter: usize) -> bool {
    profile_steps && (iter < 5 || iter % profile_interval.max(1) == 0)
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

#[cfg(test)]
mod tests;
