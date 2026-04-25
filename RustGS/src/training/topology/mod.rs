#[cfg(test)]
use std::cmp::Ordering;

mod apply;
mod bridge;
mod density_controller;
mod reference;
mod schedule;
mod splat_metrics;

#[cfg(test)]
use glam::{Quat, Vec3};
use rand::seq::index::sample_weighted;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub(crate) use self::schedule::should_apply_topology_step;
pub(crate) use apply::apply_mutations;
pub(crate) use bridge::{plan_mutations, snapshot_for_topology};

use self::density_controller::{DensityControllerConfig, PruneMode};
use self::schedule::{plan_topology_execution, schedule_topology, TopologyStepContext};
use self::splat_metrics::TopologySplatMetrics;
use super::metrics::ParityTopologyMetrics;
use super::{LiteGsConfig, LiteGsPruneMode, TrainingConfig};
use crate::core::HostSplats;
#[cfg(test)]
use crate::core::HostSplats as Splats;

const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_PERCENT_DENSE: f32 = 0.01;
const BRUSH_MIN_OPACITY: f32 = 1.0 / 255.0;
const BRUSH_MIN_SCALE: f32 = 1e-10;
const BRUSH_REFINE_PROGRESS_LIMIT: f32 = 0.95;
const TOPOLOGY_SELECTION_SALT: u64 = 0x6c69_7465_6773_7365;
const TOPOLOGY_REFINE_SALT: u64 = 0x6272_7573_685f_7266;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RunningMoments {
    pub(super) mean: f32,
    pub(super) m2: f32,
    pub(super) count: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl RunningMoments {
    #[allow(dead_code)]
    pub(super) fn update(&mut self, value: f32) {
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
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct MetalGaussianStats {
    pub(super) mean2d_grad: RunningMoments,
    pub(super) fragment_weight: RunningMoments,
    pub(super) fragment_err: RunningMoments,
    pub(super) refine_weight_max: f32,
    pub(super) visible_count: usize,
    #[allow(dead_code)]
    pub(super) age: usize,
    #[allow(dead_code)]
    pub(super) consecutive_invisible_epochs: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyCandidateInfo {
    pub(super) opacity: f32,
    pub(super) mean2d_grad: f32,
    pub(super) visible_count: usize,
    pub(super) prune_candidate: bool,
    pub(super) growth_candidate: bool,
}

#[derive(Debug, Default)]
pub(super) struct TopologyAnalysis {
    pub(super) infos: Vec<TopologyCandidateInfo>,
    pub(super) clone_candidates: usize,
    pub(super) split_candidates: usize,
    pub(super) prune_candidates: usize,
    pub(super) growth_candidates: usize,
    pub(super) active_grad_stats: usize,
    pub(super) small_scale_stats: usize,
    pub(super) opacity_ready_stats: usize,
    pub(super) max_grad: f32,
    pub(super) mean_grad: f32,
}

#[derive(Debug, Default, Clone)]
pub(super) struct LiteGsDensifySelection {
    pub(super) selected_indices: Vec<usize>,
    pub(super) replacement_count: usize,
    pub(super) extra_growth_count: usize,
}

#[derive(Debug, Clone)]
pub(super) struct TopologyPolicy {
    pub(super) litegs: LiteGsConfig,
    pub(super) max_gaussian_budget: usize,
    pub(super) scene_extent: f32,
    pub(super) max_iterations: usize,
}

impl TopologyPolicy {
    pub(crate) fn from_training_config(config: &TrainingConfig, scene_extent: f32) -> Self {
        Self {
            litegs: config.litegs.clone(),
            max_gaussian_budget: config.max_initial_gaussians.max(1),
            scene_extent,
            max_iterations: config.iterations,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        if frame_count == 0 {
            0
        } else {
            (self.max_iterations / frame_count).max(1)
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_effective_densify_from_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        if total_epochs == 0 || self.litegs.densify_from >= total_epochs {
            total_epochs
        } else {
            self.litegs.densify_from
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        let densify_from = self.litegs_effective_densify_from_epoch(frame_count);
        if densify_from >= total_epochs {
            return total_epochs;
        }
        if let Some(until) = self.litegs.densify_until {
            return until.max(densify_from.saturating_add(1)).min(total_epochs);
        }

        let reset_interval = self.litegs.opacity_reset_interval.max(1);
        let scaled = ((total_epochs as f32) * 0.8).floor() as usize;
        let computed = (scaled / reset_interval) * reset_interval + 1;
        computed
            .max(densify_from.saturating_add(1))
            .min(total_epochs)
    }

    pub(super) fn litegs_clone_scale_threshold(&self) -> f32 {
        (self.scene_extent * LITEGS_PERCENT_DENSE).max(1e-4)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn density_controller_reference_config(
        &self,
        current_gaussians: usize,
    ) -> DensityControllerConfig {
        DensityControllerConfig {
            densify_grad_threshold: self.litegs.growth_grad_threshold,
            opacity_threshold: LITEGS_OPACITY_THRESHOLD,
            percent_dense: LITEGS_PERCENT_DENSE,
            screen_extent: self.scene_extent.max(1e-6),
            init_points_num: current_gaussians.max(1),
            target_primitives: self.litegs.target_primitives.max(current_gaussians.max(1)),
            densify_from: self.litegs.densify_from,
            densify_until: self.litegs.densify_until.unwrap_or(self.max_iterations),
            densification_interval: self.litegs.densification_interval.max(1),
            prune_mode: density_controller_prune_mode(self.litegs.prune_mode),
        }
    }
}

pub(crate) fn plan_topology_from_host_snapshot(
    config: &TrainingConfig,
    splats: &HostSplats,
    grad_2d_accum: &[f32],
    grad_color_accum: &[f32],
    num_observations: &[u32],
    iteration: usize,
    frame_count: usize,
) -> TopologyMutationPlan {
    let metrics = TopologySplatMetrics::from_snapshot(splats);
    let policy = TopologyPolicy::from_training_config(config, splats.scene_extent());
    let schedule = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration,
            frame_count,
        },
    );

    let stats = build_host_snapshot_stats(
        metrics.len(),
        grad_2d_accum,
        grad_color_accum,
        num_observations,
    );
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats);

    let requested_additions = litegs_requested_additions(
        &analysis.infos,
        policy.litegs.growth_select_fraction,
        schedule.allow_extra_growth,
    );
    let max_gaussians = requested_gaussian_cap(&policy, metrics.len(), requested_additions);
    let max_new = max_gaussians.saturating_sub(metrics.len());
    let topology_seed = topology_rng_seed(config.frame_shuffle_seed, iteration);
    let litegs_selection = litegs_select_densify_candidates_seeded(
        &analysis.infos,
        max_new,
        policy.litegs.growth_select_fraction,
        schedule.allow_extra_growth,
        topology_seed,
    );

    let execution = plan_topology_execution(&policy, schedule, &analysis, &litegs_selection);
    let request = TopologyMutationRequest {
        should_densify: execution.should_densify,
        should_reset_opacity: execution.should_reset_opacity,
        completed_epoch: execution.completed_epoch,
        late_stage: iteration >= config.iterations.saturating_mul(9) / 10,
        infos: &analysis.infos,
        litegs_selection: &litegs_selection,
        random_seed: topology_seed,
    };

    plan_topology_mutation(&metrics, request)
}

pub(super) fn analyze_topology_candidates(
    policy: &TopologyPolicy,
    splats: &TopologySplatMetrics,
    stats: &[MetalGaussianStats],
) -> TopologyAnalysis {
    let mut analysis = TopologyAnalysis {
        infos: Vec::with_capacity(splats.len()),
        ..TopologyAnalysis::default()
    };
    let mut grad_sum = 0.0f32;
    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let (brush_center, brush_extent) = splats.brush_bounds_center_extent();
    let brush_max_allowed_bounds = brush_extent.max(policy.scene_extent.max(1e-6)) * 100.0;

    for idx in 0..splats.len() {
        let position = splats.position(idx);
        let scale = splats.scale(idx);
        let max_scale = splats.max_scale(idx);
        let opacity = splats.opacity(idx);
        let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
        let mean2d_grad = gaussian_stats.refine_weight_max;
        let growth_threshold = policy.litegs.growth_grad_threshold;
        let growth_candidate = mean2d_grad.is_finite()
            && mean2d_grad >= growth_threshold
            && gaussian_stats.visible_count > 0;
        let candidate_info = TopologyCandidateInfo {
            opacity,
            mean2d_grad,
            visible_count: gaussian_stats.visible_count,
            prune_candidate: false,
            growth_candidate,
        };
        let prune_candidate = litegs_should_prune_candidate(
            policy,
            &candidate_info,
            position,
            scale,
            brush_center,
            brush_max_allowed_bounds,
            splats.retainable(idx),
        );

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
    }

    if !analysis.infos.is_empty() {
        analysis.mean_grad = grad_sum / analysis.infos.len() as f32;
    }
    analysis
}

pub(super) fn litegs_requested_additions(
    infos: &[TopologyCandidateInfo],
    growth_select_fraction: f32,
    allow_extra_growth: bool,
) -> usize {
    let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
    if !allow_extra_growth {
        return prune_candidates;
    }

    let threshold_count = infos.iter().filter(|info| info.growth_candidate).count();
    let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;

    prune_candidates.saturating_add(grow_count.saturating_sub(prune_candidates))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn litegs_select_densify_candidates(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
) -> LiteGsDensifySelection {
    litegs_select_densify_candidates_seeded(
        infos,
        max_new,
        growth_select_fraction,
        allow_extra_growth,
        0,
    )
}

fn litegs_select_densify_candidates_seeded(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
    seed: u64,
) -> LiteGsDensifySelection {
    let mut rng = StdRng::seed_from_u64(seed ^ TOPOLOGY_SELECTION_SALT);
    litegs_select_densify_candidates_with_rng(
        infos,
        max_new,
        growth_select_fraction,
        allow_extra_growth,
        &mut rng,
    )
}

fn litegs_select_densify_candidates_with_rng<R: Rng + ?Sized>(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
    rng: &mut R,
) -> LiteGsDensifySelection {
    if max_new == 0 || infos.is_empty() {
        return LiteGsDensifySelection::default();
    }

    let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
    let replacement_sources: Vec<usize> = infos
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            (!info.prune_candidate
                && info.visible_count > 0
                && info.opacity.is_finite()
                && info.opacity > 0.0)
                .then_some(idx)
        })
        .collect();
    let replacement_weights: Vec<f32> = replacement_sources
        .iter()
        .map(|&idx| infos[idx].opacity.max(0.0))
        .collect();

    let replacement_count = prune_candidates.min(max_new).min(replacement_sources.len());
    let mut selection = LiteGsDensifySelection {
        selected_indices: Vec::with_capacity(max_new.min(infos.len())),
        replacement_count: 0,
        extra_growth_count: 0,
    };
    let mut used_sources = vec![false; infos.len()];

    if replacement_count > 0 {
        for sampled in sample_weighted_indices(&replacement_weights, replacement_count, rng) {
            let source_idx = replacement_sources[sampled];
            selection.selected_indices.push(source_idx);
            selection.replacement_count += 1;
            used_sources[source_idx] = true;
        }
    }

    if allow_extra_growth && selection.selected_indices.len() < max_new {
        let threshold_count = infos.iter().filter(|info| info.growth_candidate).count();
        let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;
        let sample_high_grad = grow_count.saturating_sub(prune_candidates);
        let extra_growth_limit =
            sample_high_grad.min(max_new.saturating_sub(selection.selected_indices.len()));

        if extra_growth_limit > 0 {
            let growth_sources: Vec<usize> = infos
                .iter()
                .enumerate()
                .filter_map(|(idx, info)| {
                    (!info.prune_candidate && info.growth_candidate && !used_sources[idx])
                        .then_some(idx)
                })
                .collect();
            let growth_weights: Vec<f32> = growth_sources
                .iter()
                .map(|&idx| infos[idx].mean2d_grad.max(0.0))
                .collect();
            for sampled in sample_weighted_indices(
                &growth_weights,
                extra_growth_limit.min(growth_sources.len()),
                rng,
            ) {
                let source_idx = growth_sources[sampled];
                selection.selected_indices.push(source_idx);
                selection.extra_growth_count += 1;
            }
        }
    }

    selection
}

pub(super) fn requested_gaussian_cap(
    policy: &TopologyPolicy,
    current_len: usize,
    litegs_requested_additions: usize,
) -> usize {
    policy
        .max_gaussian_budget
        .max(current_len.saturating_add(litegs_requested_additions))
}

#[cfg(test)]
pub(super) fn densify_snapshot_litegs(
    policy: &TopologyPolicy,
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    max_gaussians: usize,
    selected_indices: &[usize],
) -> usize {
    if snapshot.len() >= max_gaussians || selected_indices.is_empty() {
        return 0;
    }

    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let mut clone_indices = Vec::new();
    let mut split_indices = Vec::new();

    for &idx in selected_indices
        .iter()
        .take(max_gaussians.saturating_sub(snapshot.len()))
    {
        let max_scale = snapshot.scale(idx).into_iter().fold(0.0f32, f32::max);
        if max_scale <= clone_scale_threshold {
            clone_indices.push(idx);
        } else {
            split_indices.push(idx);
        }
    }

    let mut added = 0usize;

    for idx in &split_indices {
        if snapshot.len() >= max_gaussians {
            break;
        }

        let mut position = snapshot.position(*idx);
        let mut log_scale = snapshot.log_scale(*idx);
        let rotation = snapshot.rotation(*idx);
        let opacity_logit = snapshot.opacity_logits[*idx];
        let sh_coeffs = snapshot.sh_coeffs_row(*idx).to_vec();
        let scale = snapshot.scale(*idx);
        let (max_axis, max_axis_scale) = scale
            .into_iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
            .unwrap_or((0, 0.0));
        position[max_axis] += max_axis_scale * 0.5;
        log_scale[max_axis] = (max_axis_scale / 1.6f32).max(1e-6f32).ln();

        snapshot.push(position, log_scale, rotation, opacity_logit, &sh_coeffs);
        stats.push(MetalGaussianStats::default());
        origins.push(None);
        added = added.saturating_add(1);
    }

    for idx in &clone_indices {
        if snapshot.len() >= max_gaussians {
            break;
        }

        let position = snapshot.position(*idx);
        let log_scale = snapshot.log_scale(*idx);
        let rotation = snapshot.rotation(*idx);
        let opacity_logit = snapshot.opacity_logits[*idx];
        let sh_coeffs = snapshot.sh_coeffs_row(*idx).to_vec();
        snapshot.push(position, log_scale, rotation, opacity_logit, &sh_coeffs);
        stats.push(MetalGaussianStats::default());
        origins.push(None);
        added = added.saturating_add(1);
    }

    added
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg(test)]
pub(super) struct TopologyMutationResult {
    pub(super) added: usize,
    pub(super) pruned: usize,
    pub(super) aftermath: TopologyMutationAftermath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyStatsAction {
    KeepCurrent,
    UseMutated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TopologyMetricsDelta {
    pub(super) final_gaussians: usize,
    pub(super) added: usize,
    pub(super) pruned: usize,
    pub(super) opacity_reset: bool,
    pub(super) completed_epoch: Option<usize>,
    pub(super) late_stage: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TopologyMutationAftermath {
    pub(super) requires_runtime_rebuild: bool,
    pub(super) requires_adam_rebuild: bool,
    pub(super) requires_cluster_resync: bool,
    pub(super) requires_runtime_reserve: bool,
    pub(super) apply_opacity_reset: bool,
    pub(super) reset_refine_window_stats: bool,
    pub(super) gaussian_stats_action: TopologyStatsAction,
    pub(super) metrics_delta: TopologyMetricsDelta,
}

impl Default for TopologyMutationAftermath {
    fn default() -> Self {
        Self {
            requires_runtime_rebuild: false,
            requires_adam_rebuild: false,
            requires_cluster_resync: false,
            requires_runtime_reserve: false,
            apply_opacity_reset: false,
            reset_refine_window_stats: false,
            gaussian_stats_action: TopologyStatsAction::KeepCurrent,
            metrics_delta: TopologyMetricsDelta::default(),
        }
    }
}

pub(super) struct TopologyMutationRequest<'a> {
    pub(super) should_densify: bool,
    pub(super) should_reset_opacity: bool,
    pub(super) completed_epoch: Option<usize>,
    pub(super) late_stage: bool,
    pub(super) infos: &'a [TopologyCandidateInfo],
    pub(super) litegs_selection: &'a LiteGsDensifySelection,
    pub(super) random_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TopologyPlanRow {
    Existing {
        source_idx: usize,
    },
    BrushRefineExisting {
        source_idx: usize,
        sample_scalar: f32,
    },
    BrushRefineNew {
        source_idx: usize,
        sample_scalar: f32,
    },
}

impl TopologyPlanRow {
    pub(crate) fn source_idx(&self) -> usize {
        match *self {
            Self::Existing { source_idx }
            | Self::BrushRefineExisting { source_idx, .. }
            | Self::BrushRefineNew { source_idx, .. } => source_idx,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_existing(&self) -> bool {
        matches!(
            self,
            Self::Existing { .. } | Self::BrushRefineExisting { .. }
        )
    }

    #[cfg(test)]
    fn position(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut position = metrics.position(source_idx);
        match *self {
            Self::Existing { .. } => position,
            Self::BrushRefineExisting { sample_scalar, .. } => {
                let offset = brush_refine_offset(metrics, source_idx, sample_scalar);
                position[0] -= offset[0];
                position[1] -= offset[1];
                position[2] -= offset[2];
                position
            }
            Self::BrushRefineNew { sample_scalar, .. } => {
                let offset = brush_refine_offset(metrics, source_idx, sample_scalar);
                position[0] += offset[0];
                position[1] += offset[1];
                position[2] += offset[2];
                position
            }
        }
    }

    #[cfg(test)]
    fn scale(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut scale = metrics.scale(source_idx);
        match *self {
            Self::Existing { .. } => scale,
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                scale = brush_refine_scale(scale);
                scale
            }
        }
    }

    #[cfg(test)]
    fn opacity(&self, metrics: &TopologySplatMetrics) -> f32 {
        match *self {
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                brush_refine_opacity(metrics.opacity(self.source_idx()))
            }
            _ => metrics.opacity(self.source_idx()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TopologyMutationPlan {
    pub(crate) rows: Vec<TopologyPlanRow>,
    pub(crate) added: usize,
    pub(crate) pruned: usize,
    pub(super) aftermath: TopologyMutationAftermath,
}

impl TopologyMutationPlan {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn origins(&self) -> Vec<Option<usize>> {
        self.rows
            .iter()
            .map(|row| row.is_existing().then_some(row.source_idx()))
            .collect()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn remap_stats(&self, current: &[MetalGaussianStats]) -> Vec<MetalGaussianStats> {
        self.rows
            .iter()
            .map(|row| {
                if row.is_existing() {
                    current.get(row.source_idx()).copied().unwrap_or_default()
                } else {
                    MetalGaussianStats::default()
                }
            })
            .collect()
    }
}

pub(crate) fn plan_topology_mutation(
    metrics: &TopologySplatMetrics,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationPlan {
    plan_brush_refine_mutation(metrics, request)
}

fn plan_brush_refine_mutation(
    metrics: &TopologySplatMetrics,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationPlan {
    let mut surviving_indices: Vec<usize> = (0..metrics.len())
        .filter(|&idx| {
            !request
                .infos
                .get(idx)
                .map(|info| info.prune_candidate)
                .unwrap_or(false)
        })
        .collect();
    if surviving_indices.is_empty() {
        surviving_indices = (0..metrics.len()).collect();
    }

    let mut rows: Vec<TopologyPlanRow> = surviving_indices
        .iter()
        .copied()
        .map(|source_idx| TopologyPlanRow::Existing { source_idx })
        .collect();
    let pruned = metrics.len().saturating_sub(surviving_indices.len());

    let mut added = 0usize;
    if request.should_densify && !request.litegs_selection.selected_indices.is_empty() {
        let mut rng = StdRng::seed_from_u64(request.random_seed ^ TOPOLOGY_REFINE_SALT);
        for &source_idx in &request.litegs_selection.selected_indices {
            let Some(row_idx) = surviving_indices.iter().position(|&idx| idx == source_idx) else {
                continue;
            };
            let sample_scalar = brush_refine_sample_scalar(&mut rng);
            rows[row_idx] = TopologyPlanRow::BrushRefineExisting {
                source_idx,
                sample_scalar,
            };
            rows.push(TopologyPlanRow::BrushRefineNew {
                source_idx,
                sample_scalar,
            });
            added = added.saturating_add(1);
        }
    }

    let aftermath = topology_mutation_aftermath(&request, rows.len(), added, pruned);
    TopologyMutationPlan {
        rows,
        added,
        pruned,
        aftermath,
    }
}

#[cfg(test)]
pub(super) fn apply_snapshot_mutations(
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationResult {
    let metrics = TopologySplatMetrics::from_snapshot(snapshot);
    let plan = plan_topology_mutation(&metrics, request);
    if plan.added > 0 || plan.pruned > 0 || plan.rows.len() != snapshot.len() {
        *snapshot = rebuild_snapshot_from_plan(snapshot, &metrics, &plan);
        *stats = plan.remap_stats(stats);
        *origins = plan.origins();
    }

    TopologyMutationResult {
        added: plan.added,
        pruned: plan.pruned,
        aftermath: plan.aftermath,
    }
}

fn topology_mutation_aftermath(
    request: &TopologyMutationRequest<'_>,
    final_gaussians: usize,
    added: usize,
    pruned: usize,
) -> TopologyMutationAftermath {
    let topology_changed = added > 0 || pruned > 0;
    let gaussian_stats_action = if topology_changed {
        TopologyStatsAction::UseMutated
    } else {
        TopologyStatsAction::KeepCurrent
    };

    TopologyMutationAftermath {
        requires_runtime_rebuild: topology_changed,
        requires_adam_rebuild: topology_changed,
        requires_cluster_resync: topology_changed,
        requires_runtime_reserve: topology_changed,
        apply_opacity_reset: request.should_reset_opacity,
        reset_refine_window_stats: true,
        gaussian_stats_action,
        metrics_delta: TopologyMetricsDelta {
            final_gaussians,
            added,
            pruned,
            opacity_reset: request.should_reset_opacity,
            completed_epoch: request.completed_epoch,
            late_stage: request.late_stage,
        },
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_topology_metrics_delta(
    metrics: &mut ParityTopologyMetrics,
    delta: TopologyMetricsDelta,
) {
    metrics.final_gaussians = Some(delta.final_gaussians);
    if delta.added > 0 {
        metrics.densify_events = metrics.densify_events.saturating_add(1);
        metrics.densify_added = metrics.densify_added.saturating_add(delta.added);
        record_topology_epoch(
            &mut metrics.first_densify_epoch,
            &mut metrics.last_densify_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_densify_events = metrics.late_stage_densify_events.saturating_add(1);
            metrics.late_stage_densify_added =
                metrics.late_stage_densify_added.saturating_add(delta.added);
        }
    }
    if delta.pruned > 0 {
        metrics.prune_events = metrics.prune_events.saturating_add(1);
        metrics.prune_removed = metrics.prune_removed.saturating_add(delta.pruned);
        record_topology_epoch(
            &mut metrics.first_prune_epoch,
            &mut metrics.last_prune_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_prune_events = metrics.late_stage_prune_events.saturating_add(1);
            metrics.late_stage_prune_removed = metrics
                .late_stage_prune_removed
                .saturating_add(delta.pruned);
        }
    }
    if delta.opacity_reset {
        metrics.opacity_reset_events = metrics.opacity_reset_events.saturating_add(1);
        record_topology_epoch(
            &mut metrics.first_opacity_reset_epoch,
            &mut metrics.last_opacity_reset_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_opacity_reset_events =
                metrics.late_stage_opacity_reset_events.saturating_add(1);
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn density_controller_prune_mode(mode: LiteGsPruneMode) -> PruneMode {
    match mode {
        LiteGsPruneMode::Threshold => PruneMode::Threshold,
        LiteGsPruneMode::Weight => PruneMode::Weight,
    }
}

fn sample_weighted_indices<R: Rng + ?Sized>(
    weights: &[f32],
    count: usize,
    rng: &mut R,
) -> Vec<usize> {
    if count == 0 || weights.is_empty() {
        return Vec::new();
    }

    let sanitized: Vec<f32> = weights
        .iter()
        .map(|weight| {
            if weight.is_finite() && *weight > 0.0 {
                *weight
            } else {
                0.0
            }
        })
        .collect();
    sample_weighted(
        rng,
        sanitized.len(),
        |idx| sanitized[idx],
        count.min(sanitized.len()),
    )
    .map(|sample| sample.into_iter().collect())
    .unwrap_or_default()
}

fn brush_refine_sample_scalar<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-6, 1.0 - 1e-6);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn topology_rng_seed(base_seed: u64, iteration: usize) -> u64 {
    base_seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(iteration as u64)
        .rotate_left(17)
}

#[cfg(test)]
fn brush_refine_scale(scale: [f32; 3]) -> [f32; 3] {
    let mut refined = scale;
    let mut max_axis = 0usize;
    for axis in 1..3 {
        if refined[axis] > refined[max_axis] {
            max_axis = axis;
        }
    }
    refined[max_axis] *= 0.5;
    refined
}

#[cfg(test)]
fn brush_refine_opacity(opacity: f32) -> f32 {
    let clamped = opacity.clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY);
    let inverted = (1.0 - clamped).max(0.0);
    (1.0 - inverted.sqrt()).clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY)
}

#[cfg(test)]
fn brush_refine_offset(
    metrics: &TopologySplatMetrics,
    source_idx: usize,
    sample_scalar: f32,
) -> [f32; 3] {
    let rotation = metrics.rotation(source_idx);
    let quat = Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    let quat = if quat.length_squared() > 0.0 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    };
    (quat * (Vec3::from_array(metrics.scale(source_idx)) * sample_scalar)).to_array()
}

#[cfg_attr(not(test), allow(dead_code))]
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

fn litegs_should_prune_candidate(
    _policy: &TopologyPolicy,
    info: &TopologyCandidateInfo,
    position: [f32; 3],
    scale: [f32; 3],
    center: [f32; 3],
    max_allowed_bounds: f32,
    retainable: bool,
) -> bool {
    let opacity_prune = !info.opacity.is_finite() || info.opacity < BRUSH_MIN_OPACITY;
    let scale_small = scale
        .iter()
        .any(|value| !value.is_finite() || *value < BRUSH_MIN_SCALE);
    let scale_large = scale
        .iter()
        .any(|value| value.is_finite() && *value > max_allowed_bounds);
    let out_of_bounds = position
        .iter()
        .zip(center.iter())
        .any(|(position, center)| (*position - *center).abs() > max_allowed_bounds);
    opacity_prune || scale_small || scale_large || out_of_bounds || !retainable
}

fn build_host_snapshot_stats(
    len: usize,
    grad_2d_accum: &[f32],
    grad_color_accum: &[f32],
    num_observations: &[u32],
) -> Vec<MetalGaussianStats> {
    (0..len)
        .map(|idx| {
            let observations = num_observations.get(idx).copied().unwrap_or_default() as usize;
            let denom = observations.max(1) as f32;
            let grad_2d = grad_2d_accum.get(idx).copied().unwrap_or_default();
            let grad_color = grad_color_accum.get(idx).copied().unwrap_or_default();

            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: grad_2d / denom,
                    m2: 0.0,
                    count: observations,
                },
                fragment_weight: RunningMoments {
                    mean: grad_color / denom,
                    m2: 0.0,
                    count: observations,
                },
                refine_weight_max: grad_2d / denom,
                visible_count: observations,
                ..MetalGaussianStats::default()
            }
        })
        .collect()
}

#[cfg(test)]
fn rebuild_snapshot_from_plan(
    snapshot: &Splats,
    metrics: &TopologySplatMetrics,
    plan: &TopologyMutationPlan,
) -> Splats {
    let mut rebuilt = snapshot.retained_view(plan.rows.len());
    for row in &plan.rows {
        let source_idx = row.source_idx();
        let position = row.position(metrics);
        let scale = row.scale(metrics);
        let log_scale = scale.map(|value| value.max(1e-6).ln());
        let rotation = snapshot.rotation(source_idx);
        let opacity = row.opacity(metrics).clamp(1e-6, 1.0 - 1e-6);
        let opacity_logit = (opacity / (1.0 - opacity)).ln();
        rebuilt.push(
            position,
            log_scale,
            rotation,
            opacity_logit,
            snapshot.sh_coeffs_row(source_idx),
        );
    }
    rebuilt
}

#[cfg(test)]
mod tests {
    use super::reference::density_controller_reference_summary;
    use super::schedule::{schedule_topology, TopologyStepContext};
    use super::splat_metrics::TopologySplatMetrics;
    use super::{
        analyze_topology_candidates, apply_snapshot_mutations, apply_topology_metrics_delta,
        densify_snapshot_litegs, litegs_requested_additions, litegs_select_densify_candidates,
        litegs_select_densify_candidates_seeded, should_apply_topology_step,
        LiteGsDensifySelection, MetalGaussianStats, RunningMoments, TopologyMutationRequest,
        TopologyPolicy, TopologyStatsAction,
    };
    use crate::core::HostSplats as Splats;
    use crate::sh::{rgb_to_sh0_value, sh_coeff_count_for_degree, SplatColorRepresentation};
    use crate::training::metrics::ParityTopologyMetrics;
    use crate::training::{LiteGsConfig, TrainingConfig};

    fn test_snapshot(
        positions: Vec<f32>,
        log_scales: Vec<f32>,
        rotations: Vec<f32>,
        opacity_logits: Vec<f32>,
        colors: Vec<f32>,
        sh_rest: Vec<f32>,
        color_representation: SplatColorRepresentation,
    ) -> Splats {
        let sh_degree = color_representation.sh_degree();
        let row_count = opacity_logits.len();
        let sh_rest_row_width = sh_coeff_count_for_degree(sh_degree).saturating_sub(1) * 3;
        let mut sh_coeffs =
            Vec::with_capacity(row_count * sh_coeff_count_for_degree(sh_degree) * 3);
        for idx in 0..row_count {
            let color_base = idx * 3;
            match color_representation {
                SplatColorRepresentation::Rgb => {
                    sh_coeffs.extend(
                        colors[color_base..color_base + 3]
                            .iter()
                            .copied()
                            .map(rgb_to_sh0_value),
                    );
                }
                SplatColorRepresentation::SphericalHarmonics { .. } => {
                    sh_coeffs.extend_from_slice(&colors[color_base..color_base + 3]);
                    let rest = crate::core::splats::row_slice(&sh_rest, sh_rest_row_width, idx);
                    sh_coeffs.extend_from_slice(rest);
                }
            }
        }
        Splats {
            positions,
            log_scales,
            rotations,
            opacity_logits,
            sh_coeffs,
            sh_degree,
        }
    }

    #[test]
    fn litegs_schedule_respects_refine_cadence_and_freeze() {
        let config = TrainingConfig {
            iterations: 30,
            litegs: LiteGsConfig {
                densify_from: 1,
                densify_until: Some(6),
                refine_every: 2,
                opacity_reset_interval: 2,
                topology_freeze_after_epoch: Some(4),
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);

        let epoch2 = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 7,
                frame_count: 2,
            },
        );
        let epoch4 = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 11,
                frame_count: 2,
            },
        );

        assert_eq!(epoch2.completed_epoch, Some(3));
        assert!(epoch2.densify);
        assert!(epoch2.prune);
        assert!(!epoch2.reset_opacity);
        assert!(epoch2.allow_extra_growth);

        assert_eq!(epoch4.completed_epoch, Some(5));
        assert!(!epoch4.densify);
        assert!(!epoch4.prune);
        assert!(!epoch4.reset_opacity);
    }

    #[test]
    fn litegs_short_run_schedule_disables_densify_window() {
        let config = TrainingConfig {
            iterations: 500,
            litegs: LiteGsConfig {
                densify_from: 3,
                refine_every: 200,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);

        assert_eq!(policy.litegs_total_epochs(638), 1);
        assert_eq!(policy.litegs_effective_densify_from_epoch(638), 1);
        assert_eq!(policy.litegs_densify_until_epoch(638), 1);

        let early = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 100,
                frame_count: 638,
            },
        );
        let first_refine = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 201,
                frame_count: 638,
            },
        );
        let second_refine = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 401,
                frame_count: 638,
            },
        );

        assert_eq!(early.completed_epoch, Some(0));
        assert!(!early.densify);
        assert!(!early.prune);

        assert_eq!(first_refine.completed_epoch, Some(0));
        assert!(first_refine.densify);
        assert!(first_refine.prune);
        assert!(first_refine.allow_extra_growth);
        assert!(!first_refine.reset_opacity);

        assert_eq!(second_refine.completed_epoch, Some(0));
        assert!(second_refine.densify);
        assert!(second_refine.prune);
        assert!(second_refine.allow_extra_growth);
        assert!(!second_refine.reset_opacity);
    }

    #[test]
    fn should_apply_topology_step_uses_litegs_refine_cadence() {
        let config = TrainingConfig {
            iterations: 500,
            litegs: LiteGsConfig {
                refine_every: 160,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        assert!(
            !should_apply_topology_step(&config, 100, 638),
            "legacy cadence checkpoint should not trigger LiteGS refine scheduling"
        );
        assert!(
            should_apply_topology_step(&config, 161, 638),
            "LiteGS refine should trigger at phase_iter=160"
        );
        assert!(
            !should_apply_topology_step(&config, 201, 638),
            "non-refine iterations should not trigger topology work"
        );
    }

    #[test]
    fn litegs_requested_additions_only_adds_extra_growth_when_enabled() {
        let infos = vec![
            candidate(true, false, 0, 0.0, 0.0),
            candidate(false, true, 1, 0.2, 1.0),
            candidate(false, true, 1, 0.3, 2.0),
            candidate(false, false, 1, 0.4, 0.0),
        ];

        assert_eq!(litegs_requested_additions(&infos, 0.5, false), 1);
        assert_eq!(litegs_requested_additions(&infos, 0.5, true), 1);
        assert_eq!(litegs_requested_additions(&infos, 1.0, true), 2);
    }

    #[test]
    fn litegs_selection_prefers_replacements_before_extra_growth() {
        let infos = vec![
            candidate(true, false, 0, 0.0, 0.0),
            candidate(false, true, 1, 0.9, 2.0),
            candidate(false, true, 1, 0.8, 3.0),
            candidate(false, true, 1, 0.7, 1.0),
        ];

        let selection = litegs_select_densify_candidates(&infos, 3, 1.0, true);

        assert_eq!(selection.replacement_count, 1);
        assert_eq!(selection.extra_growth_count, 2);
        assert_eq!(selection.selected_indices.len(), 3);
        assert!(selection.selected_indices.iter().all(|idx| *idx > 0));
    }

    #[test]
    fn litegs_selection_is_repeatable_for_same_seed() {
        let infos = vec![
            candidate(true, false, 0, 0.0, 0.0),
            candidate(false, true, 1, 0.9, 2.0),
            candidate(false, true, 1, 0.8, 3.0),
            candidate(false, true, 1, 0.7, 1.0),
            candidate(false, true, 1, 0.6, 4.0),
        ];

        let lhs = litegs_select_densify_candidates_seeded(&infos, 3, 1.0, true, 42);
        let rhs = litegs_select_densify_candidates_seeded(&infos, 3, 1.0, true, 42);

        assert_eq!(lhs.selected_indices, rhs.selected_indices);
        assert_eq!(lhs.replacement_count, rhs.replacement_count);
        assert_eq!(lhs.extra_growth_count, rhs.extra_growth_count);
    }

    #[test]
    fn litegs_densify_preserves_sh_layout() {
        let config = TrainingConfig {
            litegs: LiteGsConfig {
                sh_degree: 3,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);
        let mut snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0],
            vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5],
            vec![0.2, 0.4, 0.6],
            vec![0.1; 15 * 3],
            SplatColorRepresentation::SphericalHarmonics { degree: 3 },
        );
        let mut stats = vec![MetalGaussianStats::default()];
        let mut origins = vec![Some(0)];

        let added =
            densify_snapshot_litegs(&policy, &mut snapshot, &mut stats, &mut origins, 2, &[0]);

        assert_eq!(added, 1);
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.sh_coeffs.len(), 2 * 16 * 3);
    }

    #[test]
    fn analyze_topology_candidates_marks_litegs_growth_and_prune() {
        let config = TrainingConfig {
            litegs: LiteGsConfig {
                prune_min_age: 1,
                prune_invisible_epochs: 1,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);
        let snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![
                0.005f32.ln(),
                0.005f32.ln(),
                0.005f32.ln(),
                0.5f32.ln(),
                0.5f32.ln(),
                0.5f32.ln(),
            ],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![2.0, -10.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0; 2 * 15 * 3],
            SplatColorRepresentation::SphericalHarmonics { degree: 3 },
        );
        let stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                refine_weight_max: 1.0,
                visible_count: 1,
                age: 1,
                ..Default::default()
            },
            MetalGaussianStats {
                age: 1,
                consecutive_invisible_epochs: 2,
                ..Default::default()
            },
        ];

        let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
        let analysis = analyze_topology_candidates(&policy, &metrics, &stats);

        assert_eq!(analysis.clone_candidates, 1);
        assert_eq!(analysis.prune_candidates, 1);
        assert_eq!(analysis.growth_candidates, 1);
        assert!(analysis.infos[0].growth_candidate);
        assert!(analysis.infos[1].prune_candidate);
    }

    #[test]
    fn density_controller_reference_summary_tracks_threshold_masks() {
        let config = TrainingConfig {
            iterations: 12,
            litegs: LiteGsConfig {
                densify_from: 1,
                densify_until: Some(8),
                densification_interval: 1,
                growth_grad_threshold: 0.001,
                prune_mode: crate::training::LiteGsPruneMode::Threshold,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 2.0);
        let snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            vec![
                0.005f32.ln(),
                0.005f32.ln(),
                0.005f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.01f32.ln(),
                0.01f32.ln(),
                0.01f32.ln(),
            ],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![2.0, 2.0, -10.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0; 3 * 15 * 3],
            SplatColorRepresentation::SphericalHarmonics { degree: 3 },
        );
        let stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 0.01,
                    m2: 0.0,
                    count: 1,
                },
                visible_count: 1,
                ..Default::default()
            },
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 0.02,
                    m2: 0.0,
                    count: 1,
                },
                visible_count: 1,
                ..Default::default()
            },
            MetalGaussianStats::default(),
        ];

        let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
        let summary = density_controller_reference_summary(&policy, &metrics, &stats, Some(2));

        assert_eq!(summary.clone_candidates(), 1);
        assert_eq!(summary.split_candidates(), 1);
        assert_eq!(summary.prune_candidates(), 1);
        assert_eq!(summary.densify_budget, Some(3));
    }

    #[test]
    fn density_controller_reference_summary_tracks_weight_budget() {
        let config = TrainingConfig {
            iterations: 12,
            litegs: LiteGsConfig {
                densify_from: 1,
                densify_until: Some(5),
                densification_interval: 1,
                growth_grad_threshold: 0.001,
                prune_mode: crate::training::LiteGsPruneMode::Weight,
                target_primitives: 6,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);
        let snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            vec![0.01f32.ln(); 9],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![2.0, 2.0, 2.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0; 3 * 15 * 3],
            SplatColorRepresentation::SphericalHarmonics { degree: 3 },
        );
        let stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 0.01,
                    m2: 0.0,
                    count: 1,
                },
                fragment_weight: RunningMoments {
                    mean: 0.5,
                    m2: 0.0,
                    count: 1,
                },
                fragment_err: RunningMoments {
                    mean: 0.6,
                    m2: 0.1,
                    count: 2,
                },
                visible_count: 2,
                ..Default::default()
            },
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 0.01,
                    m2: 0.0,
                    count: 1,
                },
                fragment_weight: RunningMoments {
                    mean: 0.4,
                    m2: 0.0,
                    count: 1,
                },
                fragment_err: RunningMoments {
                    mean: 0.5,
                    m2: 0.05,
                    count: 2,
                },
                visible_count: 2,
                ..Default::default()
            },
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 0.01,
                    m2: 0.0,
                    count: 1,
                },
                visible_count: 3,
                ..Default::default()
            },
        ];

        let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
        let summary = density_controller_reference_summary(&policy, &metrics, &stats, Some(3));

        assert_eq!(summary.prune_candidates(), 1);
        assert_eq!(summary.clone_candidates(), 3);
        assert_eq!(summary.split_candidates(), 0);
        assert_eq!(summary.densify_budget, Some(3));
    }

    #[test]
    fn mutation_aftermath_requests_structural_rebuild_and_metrics() {
        let mut snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
                0.05f32.ln(),
            ],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![2.0, -10.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            Vec::new(),
            SplatColorRepresentation::Rgb,
        );
        let mut stats = vec![
            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: 1.0,
                    m2: 0.0,
                    count: 1,
                },
                age: 5,
                ..Default::default()
            },
            MetalGaussianStats::default(),
        ];
        let mut origins = vec![Some(0), Some(1)];
        let infos = vec![
            candidate(false, true, 1, 0.9, 1.0),
            candidate(true, false, 0, 0.0, 0.0),
        ];
        let selection = LiteGsDensifySelection {
            selected_indices: vec![0],
            replacement_count: 1,
            extra_growth_count: 0,
        };

        let mutation = apply_snapshot_mutations(
            &mut snapshot,
            &mut stats,
            &mut origins,
            TopologyMutationRequest {
                should_densify: true,
                should_reset_opacity: false,
                completed_epoch: Some(2),
                late_stage: true,
                infos: &infos,
                litegs_selection: &selection,
                random_seed: 0,
            },
        );

        assert_eq!(mutation.added, 1);
        assert_eq!(mutation.pruned, 1);
        assert_eq!(snapshot.len(), 2);
        assert_eq!(
            mutation.aftermath.gaussian_stats_action,
            TopologyStatsAction::UseMutated
        );
        assert!(mutation.aftermath.requires_runtime_rebuild);
        assert!(mutation.aftermath.requires_adam_rebuild);
        assert!(mutation.aftermath.requires_cluster_resync);
        assert!(mutation.aftermath.requires_runtime_reserve);
        assert!(!mutation.aftermath.apply_opacity_reset);

        let mut metrics = ParityTopologyMetrics::default();
        apply_topology_metrics_delta(&mut metrics, mutation.aftermath.metrics_delta);
        assert_eq!(metrics.final_gaussians, Some(2));
        assert_eq!(metrics.densify_events, 1);
        assert_eq!(metrics.densify_added, 1);
        assert_eq!(metrics.prune_events, 1);
        assert_eq!(metrics.prune_removed, 1);
        assert_eq!(metrics.first_densify_epoch, Some(2));
        assert_eq!(metrics.last_prune_epoch, Some(2));
        assert_eq!(metrics.late_stage_densify_events, 1);
        assert_eq!(metrics.late_stage_prune_events, 1);
    }

    #[test]
    fn mutation_aftermath_marks_litegs_opacity_reset_without_structural_change() {
        let mut snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0],
            vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![2.0],
            vec![1.0, 0.0, 0.0],
            Vec::new(),
            SplatColorRepresentation::Rgb,
        );
        let mut stats = vec![MetalGaussianStats::default()];
        let mut origins = vec![Some(0)];
        let infos = vec![candidate(false, false, 1, 0.9, 0.0)];

        let mutation = apply_snapshot_mutations(
            &mut snapshot,
            &mut stats,
            &mut origins,
            TopologyMutationRequest {
                should_densify: false,
                should_reset_opacity: true,
                completed_epoch: Some(3),
                late_stage: true,
                infos: &infos,
                litegs_selection: &LiteGsDensifySelection::default(),
                random_seed: 0,
            },
        );

        assert_eq!(mutation.added, 0);
        assert_eq!(mutation.pruned, 0);
        assert_eq!(
            mutation.aftermath.gaussian_stats_action,
            TopologyStatsAction::KeepCurrent
        );
        assert!(!mutation.aftermath.requires_runtime_rebuild);
        assert!(mutation.aftermath.apply_opacity_reset);
        assert!(mutation.aftermath.reset_refine_window_stats);

        let mut metrics = ParityTopologyMetrics::default();
        apply_topology_metrics_delta(&mut metrics, mutation.aftermath.metrics_delta);
        assert_eq!(metrics.final_gaussians, Some(1));
        assert_eq!(metrics.opacity_reset_events, 1);
        assert_eq!(metrics.first_opacity_reset_epoch, Some(3));
        assert_eq!(metrics.late_stage_opacity_reset_events, 1);
    }

    fn candidate(
        prune_candidate: bool,
        growth_candidate: bool,
        visible_count: usize,
        opacity: f32,
        mean2d_grad: f32,
    ) -> super::TopologyCandidateInfo {
        super::TopologyCandidateInfo {
            opacity,
            mean2d_grad,
            visible_count,
            prune_candidate,
            growth_candidate,
        }
    }
}
