//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, and optimizer
//! updates on the Metal device, while replacing the generic autograd hot path
//! with a specialized analytical backward pass.

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor, Var};
use glam::{Mat3, Quat, Vec3};

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};
use crate::{GaussianMap, TrainingDataset, TrainingError};

use super::data_loading::{
    load_training_data, map_from_trainable, trainable_from_map, LoadedTrainingData,
};
use super::metal_backward::MetalBackwardGrads;
use super::metal_loss::{masked_mean_abs_diff, mean_abs_diff};
use super::metal_runtime::{
    ChunkPixelWindow, MetalBufferSlot, MetalProjectionRecord, MetalRuntime, MetalTileBins,
    NativeForwardProfile,
};
use super::training_pipeline::TrainingConfig as TopologyConfig;
use super::TrainingConfig;

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
const METAL_FRAME_BYTES_PER_PIXEL: u64 = 16;
const METAL_PIXEL_STATE_BYTES_PER_PIXEL: u64 = 40;
const METAL_GAUSSIAN_STATE_BYTES: u64 = 168;
const METAL_PROJECTED_BYTES_PER_GAUSSIAN: u64 = 64;
const METAL_CHUNK_WORKSPACE_BYTES_PER_GAUSSIAN_PIXEL: u64 = 64;
const METAL_RETAINED_GRAPH_BYTES_PER_GAUSSIAN_PIXEL: u64 = 224;

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
    grad_accum: f32,
}

#[derive(Debug, Default)]
struct TopologyAnalysis {
    infos: Vec<TopologyCandidateInfo>,
    clone_candidates: usize,
    split_candidates: usize,
    prune_candidates: usize,
    active_grad_stats: usize,
    small_scale_stats: usize,
    opacity_ready_stats: usize,
    max_grad: f32,
    mean_grad: f32,
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
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.colors.extend_from_slice(&color);
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
    grad_accum: f32,
    age: usize,
}

struct MetalAdamState {
    m_pos: Tensor,
    v_pos: Tensor,
    m_scale: Tensor,
    v_scale: Tensor,
    m_op: Tensor,
    v_op: Tensor,
    m_color: Tensor,
    v_color: Tensor,
}

impl MetalAdamState {
    fn new(gaussians: &TrainableGaussians) -> candle_core::Result<Self> {
        Ok(Self {
            m_pos: gaussians.positions().zeros_like()?,
            v_pos: gaussians.positions().zeros_like()?,
            m_scale: gaussians.scales.as_tensor().zeros_like()?,
            v_scale: gaussians.scales.as_tensor().zeros_like()?,
            m_op: gaussians.opacities.as_tensor().zeros_like()?,
            v_op: gaussians.opacities.as_tensor().zeros_like()?,
            m_color: gaussians.colors().zeros_like()?,
            v_color: gaussians.colors().zeros_like()?,
        })
    }
}

pub struct MetalTrainer {
    device: Device,
    render_width: usize,
    render_height: usize,
    pixel_count: usize,
    chunk_size: usize,
    densify_interval: usize,
    prune_interval: usize,
    topology_warmup: usize,
    topology_log_interval: usize,
    prune_threshold: f32,
    max_gaussian_budget: usize,
    lr_pos: f32,
    lr_scale: f32,
    lr_opacity: f32,
    lr_color: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
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
}

struct MetalTrainingStats {
    final_loss: f32,
}

struct MetalStepOutcome {
    loss: f32,
    visible_gaussians: usize,
    total_gaussians: usize,
    profile: Option<MetalStepProfile>,
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
        })
    }

    fn update_gaussian_stats(
        &mut self,
        grad_magnitudes: &[f32],
        projected: &ProjectedGaussians,
        loss_value: f32,
        gaussian_count: usize,
    ) -> candle_core::Result<()> {
        if self.gaussian_stats.len() != gaussian_count {
            self.gaussian_stats
                .resize(gaussian_count, MetalGaussianStats::default());
        }

        for stats in &mut self.gaussian_stats {
            stats.grad_accum *= 0.9;
            stats.age = stats.age.saturating_add(1);
        }

        for idx in 0..gaussian_count.min(grad_magnitudes.len()) {
            let grad_mag = grad_magnitudes[idx] * self.pixel_count.max(1) as f32;
            let stats = &mut self.gaussian_stats[idx];
            stats.grad_accum = (stats.grad_accum + grad_mag).min(10.0);
        }

        let visibility_bonus = loss_value.max(0.01);
        for source_idx in projected.visible_source_indices().iter().copied() {
            if let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) {
                stats.grad_accum = (stats.grad_accum + visibility_bonus).min(10.0);
            }
        }

        Ok(())
    }

    fn max_topology_gaussians(
        &self,
        requested_cap: usize,
        current_len: usize,
        frame_count: usize,
    ) -> usize {
        let min_cap = current_len.max(1);
        let requested_cap = requested_cap.max(min_cap);
        let memory_budget = detect_metal_memory_budget();
        if assess_memory_estimate(
            &estimate_peak_memory(
                requested_cap,
                self.pixel_count,
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
                &estimate_peak_memory(mid, self.pixel_count, frame_count, self.chunk_size),
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
        let defaults = TopologyConfig::default();
        let clone_opacity_threshold = self.prune_threshold;
        let log_scales = gaussians.scales.as_tensor().to_vec2::<f32>()?;
        let opacity_logits = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let mut analysis = TopologyAnalysis {
            infos: Vec::with_capacity(opacity_logits.len()),
            ..TopologyAnalysis::default()
        };
        let mut grad_sum = 0.0f32;

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
            let grad_accum = stats.get(idx).copied().unwrap_or_default().grad_accum;
            analysis.infos.push(TopologyCandidateInfo {
                max_scale,
                opacity,
                grad_accum,
            });
            if grad_accum > defaults.densify_grad_threshold {
                analysis.active_grad_stats += 1;
            }
            if max_scale < 0.1 {
                analysis.small_scale_stats += 1;
            }
            if opacity > clone_opacity_threshold {
                analysis.opacity_ready_stats += 1;
            }
            if grad_accum.is_finite() {
                analysis.max_grad = analysis.max_grad.max(grad_accum);
                grad_sum += grad_accum;
            }
            if grad_accum > defaults.densify_grad_threshold && opacity > clone_opacity_threshold {
                if max_scale < 0.1 {
                    analysis.clone_candidates += 1;
                }
                if max_scale > 0.3 {
                    analysis.split_candidates += 1;
                }
            }
            if opacity < self.prune_threshold || max_scale > defaults.prune_scale_threshold {
                analysis.prune_candidates += 1;
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

        let defaults = TopologyConfig::default();
        let clone_opacity_threshold = self.prune_threshold;
        let original_len = snapshot.len();
        let mut added = 0usize;
        let mut clone_candidates = Vec::new();
        let mut split_candidates = Vec::new();

        for idx in 0..original_len {
            let info = infos.get(idx).copied().unwrap_or(TopologyCandidateInfo {
                max_scale: 0.0,
                opacity: 0.0,
                grad_accum: 0.0,
            });
            let opacity = info.opacity;
            let max_scale = info.max_scale;
            let grad_accum = info.grad_accum;
            if !grad_accum.is_finite() || !opacity.is_finite() {
                continue;
            }
            if grad_accum <= defaults.densify_grad_threshold {
                continue;
            }
            if max_scale < 0.1 && opacity > clone_opacity_threshold {
                clone_candidates.push((idx, grad_accum));
            }
            if max_scale > 0.3 && opacity > self.prune_threshold {
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
        let per_pass_limit = defaults
            .max_densify
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
            let opacity_logit = snapshot.opacity_logits[idx];
            let axis = rank % 3;
            let mut cloned_position = position;
            cloned_position[axis] += scale[axis].max(0.01) * 0.5;
            snapshot.push(cloned_position, log_scale, rotation, opacity_logit, color);
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
            if score <= defaults.densify_grad_threshold {
                continue;
            }
            let position = snapshot.position(idx);
            let max_scale = snapshot.scale(idx).into_iter().fold(0.0f32, f32::max);
            let mut split_scale = snapshot.log_scale(idx);
            split_scale[0] = (max_scale * 0.5).max(1e-6).ln();
            let rotation = snapshot.rotation(idx);
            let color = snapshot.color(idx);
            let opacity_logit = snapshot.opacity_logits[idx];
            for direction in [1.0f32, -1.0] {
                if available == 0 {
                    break;
                }
                let mut split_position = position;
                split_position[0] += direction * max_scale * 0.1;
                snapshot.push(split_position, split_scale, rotation, opacity_logit, color);
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

        let defaults = TopologyConfig::default();
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
                    grad_accum: stats.get(idx).copied().unwrap_or_default().grad_accum,
                }
            });
            let opacity = info.opacity;
            let max_scale = info.max_scale;
            let valid = opacity.is_finite()
                && opacity >= self.prune_threshold
                && max_scale.is_finite()
                && max_scale <= defaults.prune_scale_threshold
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

        Ok(MetalAdamState {
            m_pos,
            v_pos,
            m_scale,
            v_scale,
            m_op,
            v_op,
            m_color,
            v_color,
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

    fn maybe_apply_topology_updates(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        let should_densify = self.should_densify_at(self.iteration);
        let should_prune = self.should_prune_at(self.iteration);
        if (!should_densify && !should_prune) || gaussians.len() == 0 {
            return Ok(());
        }

        let topology_start = Instant::now();
        let old_len = gaussians.len();
        let requested_cap = self.max_gaussian_budget.max(old_len);
        let max_gaussians = self.max_topology_gaussians(requested_cap, old_len, frame_count);
        let mut stats = self.gaussian_stats.clone();
        if stats.len() != old_len {
            stats.resize(old_len, MetalGaussianStats::default());
        }

        let analysis = self.analyze_topology_candidates(gaussians, &stats)?;

        if analysis.clone_candidates == 0
            && analysis.split_candidates == 0
            && analysis.prune_candidates == 0
        {
            if self.should_log_topology_at(self.iteration) {
                log::info!(
                    "Metal topology check at iter {} found no eligible candidates | densify={} | prune={} | gaussians={} | budget_cap={} | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={}",
                    self.iteration,
                    should_densify,
                    should_prune,
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
            return Ok(());
        }

        let mut snapshot = self.export_snapshot(gaussians)?;
        let mut origins: Vec<Option<usize>> = (0..snapshot.len()).map(Some).collect();

        let added = if should_densify {
            self.densify_snapshot(
                &mut snapshot,
                &mut stats,
                &mut origins,
                &analysis.infos,
                max_gaussians,
            )
        } else {
            0
        };
        let pruned = if should_prune {
            self.prune_snapshot(&mut snapshot, &mut stats, &mut origins, &analysis.infos)
        } else {
            0
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

        if added == 0 && pruned == 0 {
            if self.should_log_topology_at(self.iteration) || guardrail_triggered {
                log::info!(
                    "Metal topology check at iter {} made no changes | densify={} | prune={} | gaussians={} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={}",
                    self.iteration,
                    should_densify,
                    should_prune,
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
            return Ok(());
        }

        let rebuilt = TrainableGaussians::new(
            &snapshot.positions,
            &snapshot.log_scales,
            &snapshot.rotations,
            &snapshot.opacity_logits,
            &snapshot.colors,
            &self.device,
        )?;
        let new_adam = match self.adam.take() {
            Some(old_state) => self.rebuild_adam_state(&old_state, &origins)?,
            None => MetalAdamState::new(&rebuilt)?,
        };

        *gaussians = rebuilt;
        self.adam = Some(new_adam);
        self.gaussian_stats = stats;
        self.runtime.reserve_core_buffers(gaussians.len())?;

        log::info!(
            "Metal topology update at iter {} | densify={} | prune={} | added {} | pruned {} | gaussians {} -> {} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
            self.iteration,
            should_densify,
            should_prune,
            added,
            pruned,
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
        let runtime = MetalRuntime::new(render_width, render_height, device.clone())?;
        let use_native_forward = config.metal_use_native_forward && device.is_metal();

        Ok(Self {
            device,
            render_width,
            render_height,
            pixel_count,
            chunk_size: config.metal_gaussian_chunk_size.max(1),
            densify_interval: config.densify_interval,
            prune_interval: config.prune_interval,
            topology_warmup: config.topology_warmup,
            topology_log_interval: config.topology_log_interval.max(1),
            prune_threshold: config.prune_threshold,
            max_gaussian_budget: config.max_initial_gaussians.max(1),
            lr_pos: config.lr_position,
            lr_scale: config.lr_scale,
            lr_opacity: config.lr_opacity,
            lr_color: config.lr_color,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
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
        self.adam = Some(MetalAdamState::new(gaussians)?);
        self.gaussian_stats = vec![MetalGaussianStats::default(); gaussians.len()];
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
            let outcome =
                self.training_step(gaussians, &frames[frame_idx], frame_idx, should_profile)?;
            self.maybe_apply_topology_updates(gaussians, frames.len())?;
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

        Ok(MetalTrainingStats {
            final_loss: self.loss_history.last().copied().unwrap_or(0.0),
        })
    }

    fn training_step(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frame: &MetalTrainingFrame,
        frame_idx: usize,
        should_profile: bool,
    ) -> candle_core::Result<MetalStepOutcome> {
        self.iteration += 1;
        let total_start = Instant::now();
        let collect_visible_indices =
            self.should_densify_at(self.iteration) || self.should_prune_at(self.iteration);
        let (rendered, projected, render_profile) = self.render(
            gaussians,
            &frame.camera,
            should_profile,
            collect_visible_indices,
        )?;
        let mut profile = MetalStepProfile::from_render(render_profile);

        let loss_start = Instant::now();
        let color_loss = mean_abs_diff(&rendered.color, &frame.target_color)?;
        let depth_loss =
            masked_mean_abs_diff(&rendered.depth, &frame.target_depth, &frame.target_depth)?;
        let total = color_loss.broadcast_add(&depth_loss.affine(0.1, 0.0)?)?;
        let loss_value = total.to_vec0::<f32>()?;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        if self.cached_target_frame_idx != Some(frame_idx) {
            self.runtime.write_target_data(
                &frame.target_color_cpu,
                &frame.target_depth_cpu,
                1.0 / frame.target_color_cpu.len().max(1) as f32,
                0.1 / frame.target_depth_cpu.len().max(1) as f32,
            )?;
            self.cached_target_frame_idx = Some(frame_idx);
        }
        let backward = super::metal_backward::backward_weighted_l1(
            &mut self.runtime,
            &projected.tile_bins,
            gaussians.len(),
            &frame.camera,
        )?;
        profile.backward = backward_start.elapsed();

        let optimizer_start = Instant::now();
        self.apply_backward_grads(gaussians, &backward.grads)?;
        self.update_gaussian_stats(
            &backward.grad_magnitudes,
            &projected,
            loss_value,
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

    fn apply_backward_grads(
        &mut self,
        gaussians: &mut TrainableGaussians,
        grads: &MetalBackwardGrads,
    ) -> candle_core::Result<()> {
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;

        let (beta1, beta2, eps, step) = (self.beta1, self.beta2, self.eps, self.iteration);

        // Use fused Adam kernel on Metal device to eliminate ~48 temp Tensor allocs per step.
        adam_step_var_fused(
            &gaussians.positions,
            &grads.positions,
            &mut adam.m_pos,
            &mut adam.v_pos,
            &mut self.runtime,
            self.lr_pos,
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
            &grads.log_scales,
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
            &grads.colors,
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

        Ok(())
    }

    fn render(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
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
            let gaussian_bindings = self.runtime.bind_gaussians(gaussians)?;
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

        let (projected, mut profile) =
            self.project_gaussians(gaussians, camera, should_profile, collect_visible_indices)?;
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
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        self.runtime.stage_camera(camera)?;
        let mut profile = MetalRenderProfile::default();
        profile.total_gaussians = gaussians.len();
        let projection_start = Instant::now();
        let pos = gaussians.positions().detach();
        let scales = gaussians.scales.as_tensor().detach().exp()?;
        let opacity_logits = gaussians.opacities.as_tensor().detach();
        let colors = gaussians.colors().detach();
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
            let gpu_batch = self
                .runtime
                .project_gaussians(gaussians, collect_visible_indices)?;
            visible_source_indices = gpu_batch.visible_source_indices;
            profile.visible_gaussians = gpu_batch.visible_count;
            if !self.use_native_forward || should_profile {
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

        if !self.device.is_metal() {
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
            estimate_peak_memory(
                loaded.initial_map.len(),
                trainer.pixel_count,
                frame_count,
                trainer.chunk_size,
            )
            .total_bytes()
        ),
        memory_budget.describe(),
        estimate_peak_memory(
            loaded.initial_map.len(),
            trainer.pixel_count,
            frame_count,
            trainer.chunk_size,
        )
        .top_components_summary(3),
    );
    let skip_memory_guard = std::env::var_os("RUSTGS_SKIP_METAL_MEMORY_GUARD").is_some();
    let affordable_cap = affordable_initial_gaussian_cap(
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_map.len()),
        trainer.pixel_count,
        frame_count,
        trainer.chunk_size,
        &memory_budget,
    );
    if !skip_memory_guard && affordable_cap > 0 && loaded.initial_map.len() > affordable_cap {
        log::warn!(
            "MetalTrainer preflight lowered initial_gaussians from {} to {} for this run to fit the safe memory budget. Set RUSTGS_SKIP_METAL_MEMORY_GUARD=1 to keep the larger initialization.",
            loaded.initial_map.len(),
            affordable_cap,
        );
        loaded.initial_map.gaussians_mut().truncate(affordable_cap);
        loaded.initial_map.update_states();
        effective_config.max_initial_gaussians = affordable_cap;
    }
    trainer.max_gaussian_budget = if skip_memory_guard {
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_map.len())
    } else {
        affordable_cap.max(loaded.initial_map.len())
    };
    let estimated_peak = estimate_peak_memory(
        loaded.initial_map.len(),
        trainer.pixel_count,
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
    let mut gaussians = trainable_from_map(&loaded.initial_map, &trainer.device)?;

    if gaussians.len() == 0 {
        return Err(TrainingError::InvalidInput(
            "training initialization produced zero Gaussians".to_string(),
        ));
    }
    let frames = trainer.prepare_frames(&loaded)?;
    let stats = trainer.train(&mut gaussians, &frames, config.iterations)?;
    let trained_map = map_from_trainable(&gaussians)?;

    log::info!(
        "Metal backend complete in {:.2}s | frames={} | render={}x{} | initial_gaussians={} | final_gaussians={} | final_loss={:.6}",
        start.elapsed().as_secs_f64(),
        dataset.poses.len(),
        trainer.render_width,
        trainer.render_height,
        loaded.initial_map.len(),
        trained_map.len(),
        stats.final_loss,
    );

    Ok(trained_map)
}

fn effective_metal_config(config: &TrainingConfig) -> TrainingConfig {
    let mut effective = config.clone();
    if effective.lr_rotation != 0.0 {
        log::warn!(
            "Metal backend currently freezes Gaussian rotations because the backward path does not propagate rotation-aware projection gradients yet; overriding lr_rotation from {} to 0.0 for this run.",
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
    let requested_budget_bytes = gib_to_bytes(config.chunk_budget_gb);
    let effective_budget =
        resolve_chunk_memory_budget(requested_budget_bytes, detect_metal_memory_budget());
    let estimate = estimate_peak_memory(
        requested_initial_gaussians,
        pixel_count,
        frame_count,
        effective_config.metal_gaussian_chunk_size,
    );
    let affordable_initial_gaussians = affordable_initial_gaussian_cap(
        requested_initial_gaussians,
        pixel_count,
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
    frame_count: usize,
    chunk_size: usize,
    memory_budget: &MetalMemoryBudget,
) -> usize {
    let requested_cap = requested_cap.max(1);
    if assess_memory_estimate(
        &estimate_peak_memory(requested_cap, pixel_count, frame_count, chunk_size),
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
            &estimate_peak_memory(mid, pixel_count, frame_count, chunk_size),
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

fn estimate_peak_memory(
    num_gaussians: usize,
    pixel_count: usize,
    frame_count: usize,
    chunk_size: usize,
) -> MetalMemoryEstimate {
    let num_gaussians = num_gaussians as u64;
    let pixel_count = pixel_count as u64;
    let frame_count = frame_count as u64;
    let chunk_size = chunk_size.max(1) as u64;
    let gaussian_state_bytes = num_gaussians.saturating_mul(METAL_GAUSSIAN_STATE_BYTES);
    let frame_bytes = frame_count
        .saturating_mul(pixel_count)
        .saturating_mul(METAL_FRAME_BYTES_PER_PIXEL);
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
    *m = m
        .affine(beta1 as f64, 0.0)?
        .broadcast_add(&grad.affine((1.0 - beta1) as f64, 0.0)?)?;
    *v = v
        .affine(beta2 as f64, 0.0)?
        .broadcast_add(&grad.sqr()?.affine((1.0 - beta2) as f64, 0.0)?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let m_hat = m.affine(1.0 / bc1 as f64, 0.0)?;
    let v_hat = v.affine(1.0 / bc2 as f64, 0.0)?;
    let denom = v_hat
        .sqrt()?
        .broadcast_add(&Tensor::new(eps, var.as_tensor().device())?)?;
    let update = m_hat.broadcast_div(&denom)?.affine(lr as f64, 0.0)?;
    var.set(&var.as_tensor().sub(&update)?)?;
    Ok(())
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
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
                    if depth > 0.0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_dimensions_keep_minimum_size() {
        assert_eq!(scaled_dimensions(640, 480, 0.25), (160, 120));
        assert_eq!(scaled_dimensions(1, 1, 0.0), (1, 1));
    }

    #[test]
    fn resize_depth_ignores_invalid_values() {
        let src = vec![1.0, 0.0, 3.0, 5.0];
        let resized = resize_depth(&src, 2, 2, 1, 1);
        assert_eq!(resized.len(), 1);
        assert!((resized[0] - 3.0).abs() < 1e-6);
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
    fn metal_config_freezes_rotation_learning_until_projection_supports_it() {
        let effective = effective_metal_config(&TrainingConfig {
            lr_rotation: 0.25,
            ..TrainingConfig::default()
        });
        assert_eq!(effective.lr_rotation, 0.0);
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
            .project_gaussians(&identity, &camera, false, true)
            .unwrap();
        let (rotated_projected, _) = trainer
            .project_gaussians(&rotated, &camera, false, true)
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
        let cap = affordable_initial_gaussian_cap(57_474, 4_800, 1, 32, &budget);
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
            .project_gaussians(&gaussians, &camera, false, true)
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
            .project_gaussians(&gaussians, &camera, false, true)
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
            .project_gaussians(&gaussians, &camera, false, true)
            .unwrap();
        let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

        assert_eq!(profile.visible_gaussians, 3);
        assert_eq!(source_indices, vec![1, 0, 2]);
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
                grad_accum: 1.0,
                age: 5,
            },
            MetalGaussianStats {
                grad_accum: 0.0,
                age: 7,
            },
        ];
        trainer.iteration = 1;

        trainer
            .maybe_apply_topology_updates(&mut gaussians, 1)
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
}
