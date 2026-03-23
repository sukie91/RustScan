//! Metal-native 3DGS training backend.
//!
//! This path keeps projection, rasterization, loss computation, and optimizer
//! updates on the Metal device, while replacing the generic autograd hot path
//! with a specialized analytical backward pass.

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor, Var};

use crate::diff::analytical_backward::{self, ForwardIntermediate, GaussianRenderRecord};
use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};
use crate::{GaussianMap, TrainingDataset, TrainingError};

use super::data_loading::{
    load_training_data, map_from_trainable, trainable_from_map, LoadedTrainingData,
};
use super::metal_loss::mean_abs_diff;
use super::metal_runtime::{
    ChunkPixelWindow, MetalBufferSlot, MetalProjectedGaussian, MetalRuntime,
    MetalTileDispatchRecord, METAL_TILE_SIZE,
};
use super::training_pipeline::TrainingConfig as TopologyConfig;
use super::TrainingConfig;

#[cfg(test)]
use super::metal_runtime::ScreenRect;

const DEFAULT_METAL_MAX_INITIAL_GAUSSIANS: usize = 4_096;
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
}

struct RenderedFrame {
    color: Tensor,
    depth: Tensor,
    alpha: Tensor,
}

struct CpuRenderedFrame {
    color: Vec<f32>,
    depth: Vec<f32>,
    alpha: Vec<f32>,
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
        [self.colors[base], self.colors[base + 1], self.colors[base + 2]]
    }

    fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    fn opacity(&self, idx: usize) -> f32 {
        sigmoid_scalar(self.opacity_logits[idx])
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

#[derive(Debug, Clone, Copy)]
struct TileMetadata {
    tile_idx: usize,
    gaussian_count: usize,
}

struct TileBins {
    metadata: Vec<TileMetadata>,
    gaussian_indices: Vec<Vec<u32>>,
    total_assignments: usize,
    max_gaussians_per_tile: usize,
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
    m_rot: Tensor,
    v_rot: Tensor,
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
            m_rot: gaussians.rotations.as_tensor().zeros_like()?,
            v_rot: gaussians.rotations.as_tensor().zeros_like()?,
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
    prune_threshold: f32,
    max_gaussian_budget: usize,
    lr_pos: f32,
    lr_scale: f32,
    lr_rot: f32,
    lr_opacity: f32,
    lr_color: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    profile_steps: bool,
    profile_interval: usize,
    runtime: MetalRuntime,
    adam: Option<MetalAdamState>,
    gaussian_stats: Vec<MetalGaussianStats>,
    iteration: usize,
    loss_history: Vec<f32>,
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
        grads: &analytical_backward::AnalyticalGradients,
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

        for idx in 0..gaussian_count {
            let p = idx * 3;
            let grad_mag = (grads.positions.get(p).copied().unwrap_or(0.0).abs()
                + grads.positions.get(p + 1).copied().unwrap_or(0.0).abs()
                + grads.positions.get(p + 2).copied().unwrap_or(0.0).abs()
                + grads.log_scales.get(p).copied().unwrap_or(0.0).abs()
                + grads.log_scales.get(p + 1).copied().unwrap_or(0.0).abs()
                + grads.log_scales.get(p + 2).copied().unwrap_or(0.0).abs()
                + grads.colors.get(p).copied().unwrap_or(0.0).abs()
                + grads.colors.get(p + 1).copied().unwrap_or(0.0).abs()
                + grads.colors.get(p + 2).copied().unwrap_or(0.0).abs()
                + grads.opacity_logits.get(idx).copied().unwrap_or(0.0).abs())
                * self.pixel_count.max(1) as f32;
            let stats = &mut self.gaussian_stats[idx];
            stats.grad_accum = (stats.grad_accum + grad_mag).min(10.0);
        }

        let visible_sources = projected.source_indices.to_vec1::<u32>()?;
        let visibility_bonus = loss_value.max(0.01);
        for source_idx in visible_sources {
            if let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) {
                stats.grad_accum = (stats.grad_accum + visibility_bonus).min(10.0);
            }
        }

        Ok(())
    }

    fn max_topology_gaussians(&self, requested_cap: usize, current_len: usize, frame_count: usize) -> usize {
        let min_cap = current_len.max(1);
        let requested_cap = requested_cap.max(min_cap);
        let memory_budget = detect_metal_memory_budget();
        if assess_memory_estimate(
            &estimate_peak_memory(requested_cap, self.pixel_count, frame_count, self.chunk_size),
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

    fn densify_snapshot(
        &self,
        snapshot: &mut GaussianParameterSnapshot,
        stats: &mut Vec<MetalGaussianStats>,
        origins: &mut Vec<Option<usize>>,
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
            let opacity = snapshot.opacity(idx);
            let scale = snapshot.scale(idx);
            let max_scale = scale[0].max(scale[1]).max(scale[2]);
            let grad_accum = stats.get(idx).copied().unwrap_or_default().grad_accum;
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

        clone_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(std::cmp::Ordering::Equal));
        split_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut available = max_gaussians.saturating_sub(snapshot.len());
        let per_pass_limit = defaults
            .max_densify
            .min(available)
            .min((original_len / 32).max(32));
        let clone_limit = clone_candidates
            .len()
            .min(per_pass_limit);
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
            let opacity = snapshot.opacity(idx);
            let scale = snapshot.scale(idx);
            let max_scale = scale[0].max(scale[1]).max(scale[2]);
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
            &gather_rows(&flatten_rows(old_state.m_scale.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            &self.device,
        )?;
        let v_scale = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.v_scale.to_vec2::<f32>()?), 3, origins),
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
            &gather_rows(&flatten_rows(old_state.m_color.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
            &self.device,
        )?;
        let v_color = Tensor::from_slice(
            &gather_rows(&flatten_rows(old_state.v_color.to_vec2::<f32>()?), 3, origins),
            (row_count, 3),
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
        })
    }

    fn maybe_apply_topology_updates(
        &mut self,
        gaussians: &mut TrainableGaussians,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        if self.densify_interval == 0
            || self.iteration == 0
            || self.iteration % self.densify_interval != 0
            || gaussians.len() == 0
        {
            return Ok(());
        }

        let old_len = gaussians.len();
        let requested_cap = self.max_gaussian_budget.max(old_len);
        let max_gaussians = self.max_topology_gaussians(requested_cap, old_len, frame_count);
        let mut snapshot = self.export_snapshot(gaussians)?;
        let mut stats = self.gaussian_stats.clone();
        if stats.len() != snapshot.len() {
            stats.resize(snapshot.len(), MetalGaussianStats::default());
        }
        let mut origins: Vec<Option<usize>> = (0..snapshot.len()).map(Some).collect();

        let defaults = TopologyConfig::default();
        let clone_opacity_threshold = self.prune_threshold;
        let clone_candidates = (0..snapshot.len())
            .filter(|&idx| {
                let scale = snapshot.scale(idx);
                let grad_accum = stats.get(idx).copied().unwrap_or_default().grad_accum;
                scale[0].max(scale[1]).max(scale[2]) < 0.1
                    && snapshot.opacity(idx) > clone_opacity_threshold
                    && grad_accum > defaults.densify_grad_threshold
            })
            .count();
        let split_candidates = (0..snapshot.len())
            .filter(|&idx| {
                let scale = snapshot.scale(idx);
                let grad_accum = stats.get(idx).copied().unwrap_or_default().grad_accum;
                scale[0].max(scale[1]).max(scale[2]) > 0.3
                    && snapshot.opacity(idx) > self.prune_threshold
                    && grad_accum > defaults.densify_grad_threshold
            })
            .count();
        let prune_candidates = (0..snapshot.len())
            .filter(|&idx| {
                let scale = snapshot.scale(idx);
                let max_scale = scale[0].max(scale[1]).max(scale[2]);
                snapshot.opacity(idx) < self.prune_threshold || max_scale > defaults.prune_scale_threshold
            })
            .count();
        let active_grad_stats = stats
            .iter()
            .filter(|stat| stat.grad_accum > defaults.densify_grad_threshold)
            .count();
        let small_scale_stats = (0..snapshot.len())
            .filter(|&idx| {
                let scale = snapshot.scale(idx);
                scale[0].max(scale[1]).max(scale[2]) < 0.1
            })
            .count();
        let opacity_ready_stats = (0..snapshot.len())
            .filter(|&idx| snapshot.opacity(idx) > clone_opacity_threshold)
            .count();
        let max_grad = stats
            .iter()
            .map(|stat| stat.grad_accum)
            .fold(0.0f32, f32::max);
        let mean_grad = if stats.is_empty() {
            0.0
        } else {
            stats.iter().map(|stat| stat.grad_accum).sum::<f32>() / stats.len() as f32
        };

        let added = self.densify_snapshot(&mut snapshot, &mut stats, &mut origins, max_gaussians);
        let pruned = self.prune_snapshot(&mut snapshot, &mut stats, &mut origins);

        if added == 0 && pruned == 0 {
            log::info!(
                "Metal topology check at iter {} made no changes | gaussians={} | budget_cap={} | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={}",
                self.iteration,
                old_len,
                max_gaussians,
                max_grad,
                mean_grad,
                active_grad_stats,
                small_scale_stats,
                opacity_ready_stats,
                clone_candidates,
                split_candidates,
                prune_candidates,
            );
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
            "Metal topology update at iter {} | added {} | pruned {} | gaussians {} -> {} | budget_cap={} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
            self.iteration,
            added,
            pruned,
            old_len,
            gaussians.len(),
            max_gaussians,
            active_grad_stats,
            small_scale_stats,
            opacity_ready_stats,
            clone_candidates,
            split_candidates,
            prune_candidates,
            max_grad,
            mean_grad,
        );
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

        Ok(Self {
            device,
            render_width,
            render_height,
            pixel_count,
            chunk_size: config.metal_gaussian_chunk_size.max(1),
            densify_interval: config.densify_interval,
            prune_threshold: config.prune_threshold,
            max_gaussian_budget: config.max_initial_gaussians.max(1),
            lr_pos: config.lr_position,
            lr_scale: config.lr_scale,
            lr_rot: config.lr_rotation,
            lr_opacity: config.lr_opacity,
            lr_color: config.lr_color,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            profile_steps: config.metal_profile_steps,
            profile_interval: config.metal_profile_interval.max(1),
            runtime,
            adam: None,
            gaussian_stats: Vec::new(),
            iteration: 0,
            loss_history: Vec::new(),
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
            "MetalTrainer running at {}x{} | chunk_size={} | frames={} | initial_gaussians={} | tiles={} | runtime_buffers={} | pipeline_warmups={} | tile_index_capacity={}B",
            self.render_width,
            self.render_height,
            self.chunk_size,
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
            let outcome = self.training_step(gaussians, &frames[frame_idx], should_profile)?;
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
        should_profile: bool,
    ) -> candle_core::Result<MetalStepOutcome> {
        self.iteration += 1;
        let total_start = Instant::now();
        let (rendered, projected, render_profile) =
            self.render(gaussians, &frame.camera, should_profile)?;
        let mut profile = MetalStepProfile::from_render(render_profile);

        let loss_start = Instant::now();
        let color_loss = mean_abs_diff(&rendered.color, &frame.target_color)?;
        let depth_loss = mean_abs_diff(&rendered.depth, &frame.target_depth)?;
        let total = color_loss.broadcast_add(&depth_loss.affine(0.1, 0.0)?)?;
        let loss_value = total.to_vec0::<f32>()?;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        let intermediate = self.build_forward_intermediate(&projected, &rendered)?;
        let analytical_grads = analytical_backward::backward_weighted_l1(
            &intermediate,
            &frame.target_color_cpu,
            &frame.target_depth_cpu,
            gaussians.len(),
            frame.camera.fx,
            frame.camera.fy,
            frame.camera.cx,
            frame.camera.cy,
            1.0 / frame.target_color_cpu.len().max(1) as f32,
            0.1 / frame.target_depth_cpu.len().max(1) as f32,
        );
        profile.backward = backward_start.elapsed();

        let optimizer_start = Instant::now();
        self.apply_analytical_gradients(gaussians, &analytical_grads)?;
        self.update_gaussian_stats(&analytical_grads, &projected, loss_value, gaussians.len())?;
        self.synchronize_if_needed(should_profile)?;
        profile.optimizer = optimizer_start.elapsed();
        profile.total = total_start.elapsed();
        self.loss_history.push(loss_value);

        Ok(MetalStepOutcome {
            loss: loss_value,
            visible_gaussians: profile.visible_gaussians,
            total_gaussians: profile.total_gaussians,
            profile: if should_profile { Some(profile) } else { None },
        })
    }

    fn apply_analytical_gradients(
        &mut self,
        gaussians: &mut TrainableGaussians,
        grads: &analytical_backward::AnalyticalGradients,
    ) -> candle_core::Result<()> {
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;

        let n = gaussians.len();
        let pos_grad =
            self.runtime
                .stage_tensor_from_slice(MetalBufferSlot::GradPositions, &grads.positions, (n, 3))?;
        let scale_grad =
            self.runtime
                .stage_tensor_from_slice(MetalBufferSlot::GradScales, &grads.log_scales, (n, 3))?;
        let opacity_grad =
            self.runtime
                .stage_tensor_from_slice(MetalBufferSlot::GradOpacity, &grads.opacity_logits, (n,))?;
        let color_grad =
            self.runtime
                .stage_tensor_from_slice(MetalBufferSlot::GradColors, &grads.colors, (n, 3))?;

        adam_step_var(
            &gaussians.positions,
            &pos_grad,
            &mut adam.m_pos,
            &mut adam.v_pos,
            self.lr_pos,
            self.beta1,
            self.beta2,
            self.eps,
            self.iteration,
        )?;
        adam_step_var(
            &gaussians.scales,
            &scale_grad,
            &mut adam.m_scale,
            &mut adam.v_scale,
            self.lr_scale,
            self.beta1,
            self.beta2,
            self.eps,
            self.iteration,
        )?;
        adam_step_var(
            &gaussians.opacities,
            &opacity_grad,
            &mut adam.m_op,
            &mut adam.v_op,
            self.lr_opacity,
            self.beta1,
            self.beta2,
            self.eps,
            self.iteration,
        )?;
        adam_step_var(
            &gaussians.colors,
            &color_grad,
            &mut adam.m_color,
            &mut adam.v_color,
            self.lr_color,
            self.beta1,
            self.beta2,
            self.eps,
            self.iteration,
        )?;

        normalize_rotations(&gaussians.rotations)?;
        Ok(())
    }

    fn render(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
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

        self.runtime.stage_camera(camera)?;
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

        let (projected, mut profile) = self.project_gaussians(gaussians, camera, should_profile)?;
        let raster_start = Instant::now();
        let tile_bins = self.build_tile_bins(&projected)?;
        let (rendered, tile_stats) = self.rasterize(&projected, &tile_bins)?;
        self.synchronize_if_needed(should_profile)?;
        profile.rasterization = raster_start.elapsed();
        profile.active_tiles = tile_stats.active_tiles;
        profile.tile_gaussian_refs = tile_stats.tile_gaussian_refs;
        profile.max_gaussians_per_tile = tile_stats.max_gaussians_per_tile;
        if should_profile && self.device.is_metal() {
            profile.native_forward =
                Some(self.profile_native_forward(&projected, &tile_bins, &rendered)?);
        }
        Ok((rendered, projected, profile))
    }

    fn project_gaussians(
        &self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
        should_profile: bool,
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        let mut profile = MetalRenderProfile::default();
        profile.total_gaussians = gaussians.len();
        let projection_start = Instant::now();
        let pos = gaussians.positions().detach();
        let scales = gaussians.scales.as_tensor().detach().exp()?;
        let opacity_logits = gaussians.opacities.as_tensor().detach();
        let opacity = sigmoid_tensor(&opacity_logits)?;
        let colors = gaussians.colors().detach();

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

        let z_safe = z.clamp(1e-4, f32::MAX)?;
        let fx = Tensor::new(camera.fx, &self.device)?;
        let fy = Tensor::new(camera.fy, &self.device)?;
        let cx = Tensor::new(camera.cx, &self.device)?;
        let cy = Tensor::new(camera.cy, &self.device)?;

        let u = x
            .broadcast_mul(&fx)?
            .broadcast_div(&z_safe)?
            .broadcast_add(&cx)?;
        let v = y
            .broadcast_mul(&fy)?
            .broadcast_div(&z_safe)?
            .broadcast_add(&cy)?;

        let raw_sx = scales
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .broadcast_mul(&fx)?
            .broadcast_div(&z_safe)?;
        let raw_sy = scales
            .narrow(1, 1, 1)?
            .squeeze(1)?
            .broadcast_mul(&fy)?
            .broadcast_div(&z_safe)?;
        let sx = raw_sx.clamp(0.5, 256.0)?;
        let sy = raw_sy.clamp(0.5, 256.0)?;

        let valid = z.ge(1e-4f64)?.to_dtype(DType::F32)?;
        let support_x = sx.affine(3.0, 0.0)?;
        let support_y = sy.affine(3.0, 0.0)?;
        let min_x = u
            .broadcast_sub(&support_x)?
            .clamp(0.0f64, (camera.width.saturating_sub(1)) as f64)?;
        let max_x = u
            .broadcast_add(&support_x)?
            .clamp(0.0f64, (camera.width.saturating_sub(1)) as f64)?;
        let min_y = v
            .broadcast_sub(&support_y)?
            .clamp(0.0f64, (camera.height.saturating_sub(1)) as f64)?;
        let max_y = v
            .broadcast_add(&support_y)?
            .clamp(0.0f64, (camera.height.saturating_sub(1)) as f64)?;
        let left_ok = u
            .broadcast_add(&support_x)?
            .ge(0.0f64)?
            .to_dtype(DType::F32)?;
        let right_ok = u
            .broadcast_sub(&support_x)?
            .le(camera.width as f64)?
            .to_dtype(DType::F32)?;
        let top_ok = v
            .broadcast_add(&support_y)?
            .ge(0.0f64)?
            .to_dtype(DType::F32)?;
        let bottom_ok = v
            .broadcast_sub(&support_y)?
            .le(camera.height as f64)?
            .to_dtype(DType::F32)?;
        let visible = valid
            .broadcast_mul(&left_ok)?
            .broadcast_mul(&right_ok)?
            .broadcast_mul(&top_ok)?
            .broadcast_mul(&bottom_ok)?;
        let opacity = opacity.broadcast_mul(&visible)?;
        profile.visible_gaussians = visible.sum_all()?.to_vec0::<f32>()?.round() as usize;
        self.synchronize_if_needed(should_profile)?;
        profile.projection = projection_start.elapsed();

        if profile.visible_gaussians == 0 {
            return Ok((self.empty_projected_gaussians()?, profile));
        }

        let sort_start = Instant::now();
        // Candle's Metal arg-sort path has been observed to duplicate the first
        // visible index here, which collapses training statistics onto a single
        // Gaussian. Build and sort the visible set on CPU, then upload it back.
        let visible_values = visible.to_vec1::<f32>()?;
        let depth_values = z.to_vec1::<f32>()?;
        let mut visible_indices = Vec::with_capacity(profile.visible_gaussians);
        for (idx, visible_value) in visible_values.iter().enumerate() {
            if *visible_value >= 0.5 {
                visible_indices.push(idx as u32);
            }
        }
        visible_indices.sort_unstable_by(|lhs, rhs| {
            depth_values[*lhs as usize]
                .partial_cmp(&depth_values[*rhs as usize])
                .unwrap_or(Ordering::Equal)
        });
        let visible_idx = Tensor::from_slice(&visible_indices, visible_indices.len(), &self.device)?;
        let projected = ProjectedGaussians {
            source_indices: visible_idx.clone(),
            u: u.index_select(&visible_idx, 0)?,
            v: v.index_select(&visible_idx, 0)?,
            sigma_x: sx.index_select(&visible_idx, 0)?,
            sigma_y: sy.index_select(&visible_idx, 0)?,
            raw_sigma_x: raw_sx.index_select(&visible_idx, 0)?,
            raw_sigma_y: raw_sy.index_select(&visible_idx, 0)?,
            depth: z.index_select(&visible_idx, 0)?,
            opacity: opacity.index_select(&visible_idx, 0)?,
            opacity_logits: opacity_logits.index_select(&visible_idx, 0)?,
            scale3d: scales.index_select(&visible_idx, 0)?,
            colors: colors.index_select(&visible_idx, 0)?,
            min_x: min_x.index_select(&visible_idx, 0)?,
            max_x: max_x.index_select(&visible_idx, 0)?,
            min_y: min_y.index_select(&visible_idx, 0)?,
            max_y: max_y.index_select(&visible_idx, 0)?,
        };
        self.synchronize_if_needed(should_profile)?;
        profile.sorting = sort_start.elapsed();

        Ok((projected, profile))
    }

    fn rasterize(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &TileBins,
    ) -> candle_core::Result<(RenderedFrame, TileBinningStats)> {
        let mut color_acc = Tensor::zeros((self.pixel_count, 3), DType::F32, &self.device)?;
        let mut depth_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        let mut alpha_acc = Tensor::zeros((self.pixel_count,), DType::F32, &self.device)?;
        self.runtime
            .reserve_tile_index_capacity(tile_bins.total_assignments)?;
        let tile_stats = TileBinningStats {
            active_tiles: tile_bins.metadata.len(),
            tile_gaussian_refs: tile_bins.total_assignments,
            max_gaussians_per_tile: tile_bins.max_gaussians_per_tile,
        };

        for tile in &tile_bins.metadata {
            let indices = &tile_bins.gaussian_indices[tile.tile_idx];
            if indices.is_empty() {
                continue;
            }

            let tile_index_tensor = Tensor::from_slice(indices, indices.len(), &self.device)?;
            let window = self.runtime.tile_window(tile.tile_idx)?;
            let mut tile_color_acc =
                Tensor::zeros((window.pixel_count, 3), DType::F32, &self.device)?;
            let mut tile_depth_acc =
                Tensor::zeros((window.pixel_count,), DType::F32, &self.device)?;
            let mut tile_alpha_acc =
                Tensor::zeros((window.pixel_count,), DType::F32, &self.device)?;
            let mut tile_trans = Tensor::ones((window.pixel_count,), DType::F32, &self.device)?;

            for start in (0..tile.gaussian_count).step_by(self.chunk_size) {
                let len = (tile.gaussian_count - start).min(self.chunk_size);
                let chunk_indices = tile_index_tensor.narrow(0, start, len)?;
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

    fn build_tile_bins(&self, projected: &ProjectedGaussians) -> candle_core::Result<TileBins> {
        let total = projected.depth.dim(0)?;
        let (num_tiles_x, num_tiles_y) = self.runtime.tile_grid();
        let tile_count = num_tiles_x * num_tiles_y;
        let mut gaussian_indices = vec![Vec::new(); tile_count];

        let min_x_values = projected.min_x.to_vec1::<f32>()?;
        let max_x_values = projected.max_x.to_vec1::<f32>()?;
        let min_y_values = projected.min_y.to_vec1::<f32>()?;
        let max_y_values = projected.max_y.to_vec1::<f32>()?;
        let mut total_assignments = 0usize;
        let max_x_bound = self.render_width.saturating_sub(1);
        let max_y_bound = self.render_height.saturating_sub(1);

        for idx in 0..total {
            let g_min_x = min_x_values[idx].floor().max(0.0) as usize;
            let g_max_x = max_x_values[idx].ceil().max(0.0) as usize;
            let g_min_y = min_y_values[idx].floor().max(0.0) as usize;
            let g_max_y = max_y_values[idx].ceil().max(0.0) as usize;

            let tile_x_min = g_min_x.min(max_x_bound) / METAL_TILE_SIZE;
            let tile_x_max = g_max_x.min(max_x_bound) / METAL_TILE_SIZE;
            let tile_y_min = g_min_y.min(max_y_bound) / METAL_TILE_SIZE;
            let tile_y_max = g_max_y.min(max_y_bound) / METAL_TILE_SIZE;

            for ty in tile_y_min..=tile_y_max.min(num_tiles_y.saturating_sub(1)) {
                for tx in tile_x_min..=tile_x_max.min(num_tiles_x.saturating_sub(1)) {
                    gaussian_indices[ty * num_tiles_x + tx].push(idx as u32);
                    total_assignments += 1;
                }
            }
        }

        let mut metadata = Vec::new();
        let mut max_gaussians_per_tile = 0usize;
        for (tile_idx, indices) in gaussian_indices.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }
            max_gaussians_per_tile = max_gaussians_per_tile.max(indices.len());
            metadata.push(TileMetadata {
                tile_idx,
                gaussian_count: indices.len(),
            });
        }

        Ok(TileBins {
            metadata,
            gaussian_indices,
            total_assignments,
            max_gaussians_per_tile,
        })
    }

    fn profile_native_forward(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &TileBins,
        baseline: &RenderedFrame,
    ) -> candle_core::Result<NativeParityProfile> {
        let packed_gaussians = self.pack_projected_gaussians(projected)?;
        let (tile_records, tile_indices) = self.pack_native_tile_data(tile_bins);
        let (native_frame, native_profile) = self.runtime.rasterize_forward(
            &packed_gaussians,
            &tile_records,
            &tile_indices,
            self.render_width,
            self.render_height,
        )?;
        let color_max_abs = baseline
            .color
            .sub(&native_frame.color)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        let depth_max_abs = baseline
            .depth
            .sub(&native_frame.depth)?
            .abs()?
            .max_all()?
            .to_vec0::<f32>()?;
        let alpha_max_abs = baseline
            .alpha
            .sub(&native_frame.alpha)?
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

    fn build_forward_intermediate(
        &self,
        projected: &ProjectedGaussians,
        rendered: &RenderedFrame,
    ) -> candle_core::Result<ForwardIntermediate> {
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
        let colors = projected.colors.to_vec2::<f32>()?;
        let scale3d = projected.scale3d.to_vec2::<f32>()?;
        let min_x = projected.min_x.to_vec1::<f32>()?;
        let max_x = projected.max_x.to_vec1::<f32>()?;
        let min_y = projected.min_y.to_vec1::<f32>()?;
        let max_y = projected.max_y.to_vec1::<f32>()?;

        let cpu_rendered = CpuRenderedFrame {
            color: rendered.color.flatten_all()?.to_vec1::<f32>()?,
            depth: rendered.depth.to_vec1::<f32>()?,
            alpha: rendered.alpha.to_vec1::<f32>()?,
        };

        let mut records = Vec::with_capacity(source_indices.len());
        for idx in 0..source_indices.len() {
            let rgb = if colors[idx].len() >= 3 {
                [colors[idx][0], colors[idx][1], colors[idx][2]]
            } else {
                [0.0, 0.0, 0.0]
            };
            let scale = if scale3d[idx].len() >= 3 {
                [scale3d[idx][0], scale3d[idx][1], scale3d[idx][2]]
            } else {
                [0.0, 0.0, 0.0]
            };
            records.push(GaussianRenderRecord {
                gaussian_idx: source_indices[idx] as usize,
                u: u[idx],
                v: v[idx],
                sigma_x: sigma_x[idx],
                sigma_y: sigma_y[idx],
                z: depth[idx],
                base_alpha: opacity[idx].clamp(0.0, 1.0),
                color: rgb,
                min_x: min_x[idx].floor().max(0.0) as usize,
                max_x: max_x[idx].ceil().max(0.0) as usize,
                min_y: min_y[idx].floor().max(0.0) as usize,
                max_y: max_y[idx].ceil().max(0.0) as usize,
                raw_scale_2d_x: raw_sigma_x[idx],
                raw_scale_2d_y: raw_sigma_y[idx],
                raw_opacity: opacity[idx],
                scale_3d: scale,
                opacity_logit: opacity_logits[idx],
            });
        }

        Ok(ForwardIntermediate {
            records,
            rendered_color: cpu_rendered.color,
            alpha_acc: cpu_rendered.alpha,
            rendered_depth: cpu_rendered.depth,
            width: self.render_width,
            height: self.render_height,
        })
    }

    fn pack_projected_gaussians(
        &self,
        projected: &ProjectedGaussians,
    ) -> candle_core::Result<Vec<MetalProjectedGaussian>> {
        let u = projected.u.to_vec1::<f32>()?;
        let v = projected.v.to_vec1::<f32>()?;
        let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
        let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
        let depth = projected.depth.to_vec1::<f32>()?;
        let opacity = projected.opacity.to_vec1::<f32>()?;
        let colors = projected.colors.to_vec2::<f32>()?;

        let mut packed = Vec::with_capacity(u.len());
        for idx in 0..u.len() {
            packed.push(MetalProjectedGaussian::new(
                u[idx],
                v[idx],
                sigma_x[idx],
                sigma_y[idx],
                depth[idx],
                opacity[idx],
                colors[idx][0],
                colors[idx][1],
                colors[idx][2],
            ));
        }
        Ok(packed)
    }

    fn pack_native_tile_data(
        &self,
        tile_bins: &TileBins,
    ) -> (Vec<MetalTileDispatchRecord>, Vec<u32>) {
        let mut records = Vec::with_capacity(tile_bins.gaussian_indices.len());
        let mut indices = Vec::with_capacity(tile_bins.total_assignments);
        let mut start = 0usize;
        for gaussian_indices in &tile_bins.gaussian_indices {
            records.push(MetalTileDispatchRecord::new(
                start as u32,
                gaussian_indices.len() as u32,
            ));
            indices.extend_from_slice(gaussian_indices);
            start += gaussian_indices.len();
        }
        (records, indices)
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
    let effective_config = effective_metal_config(config);
    let loaded = load_training_data(dataset, &effective_config, &device)?;
    let mut gaussians = trainable_from_map(&loaded.initial_map, &device)?;

    if gaussians.len() == 0 {
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
    let memory_budget = detect_metal_memory_budget();
    let estimated_peak = estimate_peak_memory(
        gaussians.len(),
        trainer.pixel_count,
        loaded.cameras.len(),
        trainer.chunk_size,
    );
    log::info!(
        "MetalTrainer preflight | gaussians={} | frames={} | pixels={} | chunk_size={} | estimated_peak≈{:.1} GiB | budget={} | dominant={}",
        gaussians.len(),
        loaded.cameras.len(),
        trainer.pixel_count,
        trainer.chunk_size,
        bytes_to_gib(estimated_peak.total_bytes()),
        memory_budget.describe(),
        estimated_peak.top_components_summary(3),
    );
    let skip_memory_guard = std::env::var_os("RUSTGS_SKIP_METAL_MEMORY_GUARD").is_some();
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
    let defaults = TrainingConfig::default();
    if effective.max_initial_gaussians == defaults.max_initial_gaussians {
        effective.max_initial_gaussians = DEFAULT_METAL_MAX_INITIAL_GAUSSIANS;
        log::warn!(
            "Metal backend lowered max_initial_gaussians from {} to {} to avoid OOM. Override with --max-initial-gaussians if you want a different budget.",
            defaults.max_initial_gaussians,
            effective.max_initial_gaussians
        );
    }
    effective
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

fn normalize_rotations(rotations: &Var) -> candle_core::Result<()> {
    let rot = rotations.as_tensor();
    let norm = rot
        .sqr()?
        .sum(1)?
        .sqrt()?
        .clamp(1e-6, f32::MAX)?
        .unsqueeze(1)?;
    rotations.set(&rot.broadcast_div(&norm)?)?;
    Ok(())
}

fn flatten_rows(rows: Vec<Vec<f32>>) -> Vec<f32> {
    rows.into_iter().flatten().collect()
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

fn sigmoid_tensor(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let neg = tensor.neg()?;
    let exp_neg = neg.exp()?;
    let one = Tensor::ones_like(tensor)?;
    one.broadcast_div(&one.broadcast_add(&exp_neg)?)
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
            DEFAULT_METAL_MAX_INITIAL_GAUSSIANS
        );
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
        let trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
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
        };

        let bins = trainer.build_tile_bins(&projected).unwrap();
        assert_eq!(bins.metadata.len(), 2);
        assert_eq!(bins.total_assignments, 3);
        assert_eq!(bins.max_gaussians_per_tile, 2);
        assert_eq!(bins.gaussian_indices[0], vec![0, 1]);
        assert_eq!(bins.gaussian_indices[1], vec![1]);
    }

    #[test]
    fn native_forward_matches_baseline_on_tiny_scene() {
        let Ok(device) = Device::new_metal(0) else {
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
        let trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
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

        let (projected, profile) = trainer.project_gaussians(&gaussians, &camera, false).unwrap();

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
        let trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
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

        let (projected, profile) = trainer.project_gaussians(&gaussians, &camera, false).unwrap();
        let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

        assert_eq!(profile.visible_gaussians, 3);
        assert_eq!(source_indices, vec![1, 0, 2]);
    }

    #[test]
    fn topology_update_densifies_and_prunes_with_matching_adam_state() {
        let device = Device::Cpu;
        let trainer_config = TrainingConfig {
            densify_interval: 1,
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

        trainer.maybe_apply_topology_updates(&mut gaussians, 1).unwrap();

        assert_eq!(gaussians.len(), 2);
        assert_eq!(trainer.gaussian_stats.len(), 2);
        assert!(trainer.gaussian_stats.iter().any(|stats| stats.age == 0));
        let opacities = gaussians.opacities().unwrap().to_vec1::<f32>().unwrap();
        assert!(opacities.iter().all(|opacity| *opacity >= trainer_config.prune_threshold));
        let positions = gaussians.positions().to_vec2::<f32>().unwrap();
        assert!((positions[1][0] - positions[0][0]).abs() > 1e-6);

        let adam = trainer.adam.as_ref().unwrap();
        assert_eq!(adam.m_pos.dim(0).unwrap(), 2);
        assert_eq!(adam.v_pos.dim(0).unwrap(), 2);
        let m_pos = adam.m_pos.to_vec2::<f32>().unwrap();
        assert!(m_pos[1].iter().all(|value| value.abs() < 1e-6));
    }
}
