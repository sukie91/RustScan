mod apply;
mod bridge;
mod density_controller;
mod reference;
mod schedule;
mod splat_metrics;

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
    pub(super) age: usize,
    pub(super) consecutive_invisible_epochs: usize,
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
    num_observations: &[f32],
    visible_observations: &[f32],
    splat_ages: &[usize],
    invisible_windows: &[usize],
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
        visible_observations,
        splat_ages,
        invisible_windows,
    );
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, schedule.densify);

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
        refine_decay: refine_decay_for_schedule(
            config,
            schedule.densify || schedule.prune,
            iteration,
        ),
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
    opacity_prune_enabled: bool,
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
        let growth_candidate = mean2d_grad.is_finite() && mean2d_grad >= growth_threshold;
        let candidate_info = TopologyCandidateInfo {
            opacity,
            mean2d_grad,
            visible_count: gaussian_stats.visible_count,
            age: gaussian_stats.age,
            consecutive_invisible_epochs: gaussian_stats.consecutive_invisible_epochs,
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
            opacity_prune_enabled,
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
            (!info.prune_candidate && info.opacity.is_finite() && info.opacity > 0.0).then_some(idx)
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

fn refine_decay_for_schedule(
    config: &TrainingConfig,
    should_refine: bool,
    iteration: usize,
) -> Option<TopologyRefineDecay> {
    if !should_refine || (config.litegs.opacity_decay == 0.0 && config.litegs.scale_decay == 0.0) {
        return None;
    }

    let train_t = if config.iterations == 0 {
        1.0
    } else {
        iteration as f32 / config.iterations as f32
    };
    Some(TopologyRefineDecay {
        opacity_decay: config.litegs.opacity_decay,
        scale_decay: config.litegs.scale_decay,
        train_t: train_t.clamp(0.0, 1.0),
    })
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyStatsAction {
    KeepCurrent,
    UseMutated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TopologyRefineDecay {
    pub(crate) opacity_decay: f32,
    pub(crate) scale_decay: f32,
    pub(crate) train_t: f32,
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
    pub(super) refine_decay: Option<TopologyRefineDecay>,
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

    pub(crate) fn is_existing(&self) -> bool {
        matches!(
            self,
            Self::Existing { .. } | Self::BrushRefineExisting { .. }
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TopologyMutationPlan {
    pub(crate) rows: Vec<TopologyPlanRow>,
    pub(crate) added: usize,
    pub(crate) pruned: usize,
    pub(crate) refine_decay: Option<TopologyRefineDecay>,
    pub(super) aftermath: TopologyMutationAftermath,
}

impl TopologyMutationPlan {
    pub(crate) fn mutates_splats(&self) -> bool {
        self.added > 0 || self.pruned > 0 || self.refine_decay.is_some()
    }

    pub(crate) fn origins(&self) -> Vec<Option<usize>> {
        self.rows
            .iter()
            .map(|row| row.is_existing().then_some(row.source_idx()))
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
        refine_decay: request.refine_decay,
        aftermath,
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
    policy: &TopologyPolicy,
    info: &TopologyCandidateInfo,
    position: [f32; 3],
    scale: [f32; 3],
    center: [f32; 3],
    max_allowed_bounds: f32,
    retainable: bool,
    opacity_prune_enabled: bool,
) -> bool {
    let opacity_threshold = policy.litegs.prune_opacity_threshold.max(BRUSH_MIN_OPACITY);
    let old_enough = info.age >= policy.litegs.prune_min_age;
    let opacity_prune =
        old_enough && (!info.opacity.is_finite() || info.opacity < opacity_threshold);
    let history_invisible_prune = old_enough
        && info.visible_count == 0
        && info.consecutive_invisible_epochs >= policy.litegs.prune_invisible_epochs;
    let contribution_prune = match policy.litegs.prune_mode {
        LiteGsPruneMode::Threshold => opacity_prune || history_invisible_prune,
        LiteGsPruneMode::Weight => {
            (opacity_prune_enabled && opacity_prune)
                || (!opacity_prune_enabled && history_invisible_prune)
        }
    };
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
    contribution_prune || scale_small || scale_large || out_of_bounds || !retainable
}

fn build_host_snapshot_stats(
    len: usize,
    grad_2d_accum: &[f32],
    grad_color_accum: &[f32],
    num_observations: &[f32],
    visible_observations: &[f32],
    splat_ages: &[usize],
    invisible_windows: &[usize],
) -> Vec<MetalGaussianStats> {
    (0..len)
        .map(|idx| {
            let observations = num_observations
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let denom = observations.max(1.0);
            let visible_observations = visible_observations
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let visible_count = visible_observations.round().min(usize::MAX as f32) as usize;
            let grad_2d = grad_2d_accum.get(idx).copied().unwrap_or_default();
            let grad_color = grad_color_accum.get(idx).copied().unwrap_or_default();
            let age = splat_ages.get(idx).copied().unwrap_or_default();
            let consecutive_invisible_epochs =
                invisible_windows.get(idx).copied().unwrap_or_default();

            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: grad_2d / denom,
                    m2: 0.0,
                    count: visible_count,
                },
                fragment_weight: RunningMoments {
                    mean: grad_color / denom,
                    m2: 0.0,
                    count: visible_count,
                },
                refine_weight_max: grad_2d / denom,
                visible_count,
                age,
                consecutive_invisible_epochs,
                ..MetalGaussianStats::default()
            }
        })
        .collect()
}



#[cfg(test)]
mod tests;
