use std::cmp::Ordering;

use glam::{Quat, Vec3};
use rand::seq::index::sample_weighted;
use rand::Rng;

use super::density_controller::{
    DensityController, DensityControllerConfig, OpacityResetMode, PruneMode,
};
use super::parity_harness::ParityTopologyMetrics;
use super::runtime_splats::TopologySplatMetrics;
#[cfg(test)]
use super::splats::sigmoid_scalar;
#[cfg(test)]
use super::splats::Splats;
use super::{LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, TrainingProfile};

#[cfg(test)]
use super::TrainingConfig;

const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_PERCENT_DENSE: f32 = 0.01;
const BRUSH_MIN_OPACITY: f32 = 1.0 / 255.0;
const BRUSH_MIN_SCALE: f32 = 1e-10;
const BRUSH_REFINE_PROGRESS_LIMIT: f32 = 0.95;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RunningMoments {
    pub(super) mean: f32,
    pub(super) m2: f32,
    pub(super) count: usize,
}

impl RunningMoments {
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

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct MetalGaussianStats {
    pub(super) mean2d_grad: RunningMoments,
    pub(super) fragment_weight: RunningMoments,
    pub(super) fragment_err: RunningMoments,
    pub(super) refine_weight_max: f32,
    pub(super) visible_count: usize,
    pub(super) age: usize,
    pub(super) consecutive_invisible_epochs: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyCandidateInfo {
    pub(super) max_scale: f32,
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
    pub(super) training_profile: TrainingProfile,
    pub(super) litegs: LiteGsConfig,
    pub(super) prune_threshold: f32,
    pub(super) densify_interval: usize,
    pub(super) prune_interval: usize,
    pub(super) topology_warmup: usize,
    pub(super) topology_log_interval: usize,
    pub(super) legacy_densify_grad_threshold: f32,
    pub(super) legacy_clone_scale_threshold: f32,
    pub(super) legacy_split_scale_threshold: f32,
    pub(super) legacy_prune_scale_threshold: f32,
    pub(super) legacy_max_densify_per_update: usize,
    pub(super) max_gaussian_budget: usize,
    pub(super) scene_extent: f32,
    pub(super) max_iterations: usize,
}

impl TopologyPolicy {
    #[cfg(test)]
    pub(super) fn from_training_config(config: &TrainingConfig, scene_extent: f32) -> Self {
        Self {
            training_profile: config.training_profile,
            litegs: config.litegs.clone(),
            prune_threshold: config.prune_threshold,
            densify_interval: config.densify_interval,
            prune_interval: config.prune_interval,
            topology_warmup: config.topology_warmup,
            topology_log_interval: config.topology_log_interval.max(1),
            legacy_densify_grad_threshold: config.legacy_densify_grad_threshold,
            legacy_clone_scale_threshold: config.legacy_clone_scale_threshold,
            legacy_split_scale_threshold: config.legacy_split_scale_threshold,
            legacy_prune_scale_threshold: config.legacy_prune_scale_threshold,
            legacy_max_densify_per_update: config.legacy_max_densify_per_update.max(1),
            max_gaussian_budget: config.max_initial_gaussians.max(1),
            scene_extent,
            max_iterations: config.iterations,
        }
    }

    pub(super) fn is_litegs_mode(&self) -> bool {
        self.training_profile == TrainingProfile::LiteGsMacV1
    }

    pub(super) fn should_log_topology(&self, iteration: usize) -> bool {
        iteration % self.topology_log_interval.max(1) == 0
    }

    pub(super) fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        if frame_count == 0 {
            0
        } else {
            (self.max_iterations / frame_count).max(1)
        }
    }

    pub(super) fn litegs_effective_densify_from_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        if total_epochs == 0 || self.litegs.densify_from >= total_epochs {
            total_epochs
        } else {
            self.litegs.densify_from
        }
    }

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

    pub(super) fn density_controller_reference_config(
        &self,
        current_gaussians: usize,
    ) -> DensityControllerConfig {
        DensityControllerConfig {
            densify_grad_threshold: if self.is_litegs_mode() {
                self.litegs.growth_grad_threshold
            } else {
                self.legacy_densify_grad_threshold
            },
            opacity_threshold: if self.is_litegs_mode() {
                LITEGS_OPACITY_THRESHOLD
            } else {
                self.prune_threshold
            },
            percent_dense: LITEGS_PERCENT_DENSE,
            screen_extent: self.scene_extent.max(1e-6),
            screen_size_threshold: if self.is_litegs_mode() {
                self.litegs.prune_scale_threshold
            } else {
                self.legacy_prune_scale_threshold
            },
            init_points_num: current_gaussians.max(1),
            target_primitives: if self.is_litegs_mode() {
                self.litegs.target_primitives.max(current_gaussians.max(1))
            } else {
                self.max_gaussian_budget.max(current_gaussians.max(1))
            },
            densify_from: if self.is_litegs_mode() {
                self.litegs.densify_from
            } else {
                self.topology_warmup
            },
            densify_until: if self.is_litegs_mode() {
                self.litegs.densify_until.unwrap_or(self.max_iterations)
            } else {
                self.max_iterations
            },
            densification_interval: if self.is_litegs_mode() {
                self.litegs.densification_interval.max(1)
            } else {
                self.densify_interval.max(1)
            },
            opacity_reset_interval: if self.is_litegs_mode() {
                self.litegs.opacity_reset_interval.max(1)
            } else {
                self.prune_interval.max(1)
            },
            opacity_reset_mode: density_controller_opacity_reset_mode(
                self.litegs.opacity_reset_mode,
            ),
            prune_mode: density_controller_prune_mode(self.litegs.prune_mode),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DensityControllerReferenceSummary {
    pub(super) prune_mask: Vec<bool>,
    pub(super) clone_mask: Vec<bool>,
    pub(super) split_mask: Vec<bool>,
    pub(super) densify_budget: Option<usize>,
}

impl DensityControllerReferenceSummary {
    pub(super) fn prune_candidates(&self) -> usize {
        self.prune_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }

    pub(super) fn clone_candidates(&self) -> usize {
        self.clone_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }

    pub(super) fn split_candidates(&self) -> usize {
        self.split_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct DensityControllerReferenceAdapter {
    controller: DensityController,
}

impl DensityControllerReferenceAdapter {
    pub(super) fn from_topology_state(
        policy: &TopologyPolicy,
        splats: &TopologySplatMetrics,
        stats: &[MetalGaussianStats],
    ) -> Self {
        let mut controller = DensityController::new(
            policy.density_controller_reference_config(splats.len()),
            density_controller_taming_mode(policy),
        );
        controller.stats.resize(splats.len());

        for idx in 0..splats.len() {
            let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
            let mean2d_grad_count = gaussian_stats.mean2d_grad.count as f32;
            let fragment_weight_count = gaussian_stats.fragment_weight.count as f32;
            let fragment_err_count = gaussian_stats.fragment_err.count as f32;

            controller.stats.visible_count[idx] =
                gaussian_stats.visible_count.min(u32::MAX as usize) as u32;
            controller.stats.mean2d_grad[idx] = (
                gaussian_stats.mean2d_grad.mean * mean2d_grad_count,
                mean2d_grad_count,
            );
            controller.stats.fragment_weight[idx] = (
                gaussian_stats.fragment_weight.mean * fragment_weight_count,
                fragment_weight_count,
            );
            controller.stats.fragment_err[idx] = (
                gaussian_stats.fragment_err.mean * fragment_err_count,
                gaussian_stats.fragment_err.m2
                    + fragment_err_count
                        * gaussian_stats.fragment_err.mean
                        * gaussian_stats.fragment_err.mean,
                fragment_err_count,
            );
            controller.stats.opacity[idx] = splats.opacity(idx);
            controller.stats.max_scale[idx] = splats.max_scale(idx);
        }

        Self { controller }
    }

    pub(super) fn summary(
        &self,
        completed_epoch: Option<usize>,
    ) -> DensityControllerReferenceSummary {
        let prune_mask = self.controller.get_prune_mask();
        let clone_mask = self.controller.get_clone_mask();
        let split_mask = self.controller.get_split_mask();
        let prune_candidates = prune_mask.iter().filter(|candidate| **candidate).count();
        let densify_budget = completed_epoch
            .filter(|epoch| self.controller.is_densify_active(*epoch))
            .map(|epoch| {
                self.controller.compute_densify_budget(
                    self.controller.stats.len(),
                    prune_candidates,
                    epoch,
                )
            });

        DensityControllerReferenceSummary {
            prune_mask,
            clone_mask,
            split_mask,
            densify_budget,
        }
    }
}

pub(super) fn density_controller_reference_summary(
    policy: &TopologyPolicy,
    splats: &TopologySplatMetrics,
    stats: &[MetalGaussianStats],
    completed_epoch: Option<usize>,
) -> DensityControllerReferenceSummary {
    DensityControllerReferenceAdapter::from_topology_state(policy, splats, stats)
        .summary(completed_epoch)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyExecutionDisposition {
    Apply,
    SkipDestructiveLiteGs,
    SkipNoEligibleCandidates,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyExecutionPlan {
    pub(super) completed_epoch: Option<usize>,
    pub(super) should_densify: bool,
    pub(super) should_prune: bool,
    pub(super) should_reset_opacity: bool,
    pub(super) disposition: TopologyExecutionDisposition,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TopologySchedule {
    pub(super) completed_epoch: Option<usize>,
    pub(super) densify: bool,
    pub(super) prune: bool,
    pub(super) reset_opacity: bool,
    pub(super) allow_extra_growth: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyStepContext {
    pub(super) iteration: usize,
    pub(super) frame_count: usize,
}

pub(super) fn schedule_topology(
    policy: &TopologyPolicy,
    step: TopologyStepContext,
) -> TopologySchedule {
    if policy.is_litegs_mode() {
        let Some(epoch) = litegs_current_epoch(step.iteration, step.frame_count) else {
            return TopologySchedule::default();
        };
        let phase_iter = step.iteration.saturating_sub(1);
        let frozen = policy
            .litegs
            .topology_freeze_after_epoch
            .map(|freeze_epoch| epoch >= freeze_epoch)
            .unwrap_or(false);
        let refine_every = policy.litegs.refine_every.max(1);
        let progress = if policy.max_iterations == 0 {
            1.0
        } else {
            phase_iter as f32 / policy.max_iterations as f32
        };
        let refine = phase_iter > 0
            && !frozen
            && progress <= BRUSH_REFINE_PROGRESS_LIMIT
            && phase_iter % refine_every == 0;
        return TopologySchedule {
            completed_epoch: Some(epoch),
            densify: refine,
            prune: refine,
            reset_opacity: false,
            allow_extra_growth: refine && phase_iter < policy.litegs.growth_stop_iter,
        };
    }

    TopologySchedule {
        completed_epoch: None,
        densify: should_densify_at(
            policy.densify_interval,
            policy.topology_warmup,
            step.iteration,
        ),
        prune: should_prune_at(
            policy.prune_interval,
            policy.topology_warmup,
            step.iteration,
        ),
        reset_opacity: false,
        allow_extra_growth: false,
    }
}

pub(super) fn should_collect_visible_indices(
    policy: &TopologyPolicy,
    schedule: TopologySchedule,
) -> bool {
    policy.is_litegs_mode() || schedule.densify || schedule.prune
}

pub(super) fn plan_topology_execution(
    policy: &TopologyPolicy,
    schedule: TopologySchedule,
    analysis: &TopologyAnalysis,
    litegs_selection: &LiteGsDensifySelection,
) -> TopologyExecutionPlan {
    let mut plan = TopologyExecutionPlan {
        completed_epoch: schedule.completed_epoch,
        should_densify: schedule.densify,
        should_prune: schedule.prune,
        should_reset_opacity: schedule.reset_opacity,
        disposition: TopologyExecutionDisposition::Apply,
    };

    let has_candidates = if policy.is_litegs_mode() {
        !litegs_selection.selected_indices.is_empty() || analysis.prune_candidates > 0
    } else {
        analysis.clone_candidates > 0
            || analysis.split_candidates > 0
            || analysis.prune_candidates > 0
    };
    if !has_candidates && !plan.should_reset_opacity {
        plan.disposition = TopologyExecutionDisposition::SkipNoEligibleCandidates;
    }

    plan
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
    let clone_scale_threshold = if policy.is_litegs_mode() {
        policy.litegs_clone_scale_threshold()
    } else {
        policy.legacy_clone_scale_threshold
    };
    let (brush_center, brush_extent) = splats.brush_bounds_center_extent();
    let brush_max_allowed_bounds = brush_extent.max(policy.scene_extent.max(1e-6)) * 100.0;

    for idx in 0..splats.len() {
        let position = splats.position(idx);
        let scale = splats.scale(idx);
        let max_scale = splats.max_scale(idx);
        let opacity = splats.opacity(idx);
        let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
        let mean2d_grad = if policy.is_litegs_mode() {
            gaussian_stats.refine_weight_max
        } else {
            gaussian_stats.mean2d_grad.mean
        };

        let growth_threshold = if policy.is_litegs_mode() {
            policy.litegs.growth_grad_threshold
        } else {
            policy.legacy_densify_grad_threshold
        };
        let growth_candidate = mean2d_grad.is_finite()
            && mean2d_grad >= growth_threshold
            && gaussian_stats.visible_count > 0;
        let candidate_info = TopologyCandidateInfo {
            max_scale,
            opacity,
            mean2d_grad,
            visible_count: gaussian_stats.visible_count,
            prune_candidate: false,
            growth_candidate,
        };
        let prune_candidate = if policy.is_litegs_mode() {
            litegs_should_prune_candidate(
                policy,
                &candidate_info,
                position,
                scale,
                brush_center,
                brush_max_allowed_bounds,
                splats.retainable(idx),
            )
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

        if policy.is_litegs_mode() {
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
            if mean2d_grad > policy.legacy_densify_grad_threshold
                && opacity > policy.prune_threshold
            {
                if max_scale < policy.legacy_clone_scale_threshold {
                    analysis.clone_candidates += 1;
                }
                if max_scale > policy.legacy_split_scale_threshold {
                    analysis.split_candidates += 1;
                }
            }
            if opacity < policy.prune_threshold || max_scale > policy.legacy_prune_scale_threshold {
                analysis.prune_candidates += 1;
            }
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

pub(super) fn litegs_select_densify_candidates(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
) -> LiteGsDensifySelection {
    let mut rng = rand::thread_rng();
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
        let extra_growth_limit = sample_high_grad
            .min(max_new.saturating_sub(selection.selected_indices.len()));

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
    if policy.is_litegs_mode() {
        policy
            .max_gaussian_budget
            .max(current_len.saturating_add(litegs_requested_additions))
    } else {
        policy
            .max_gaussian_budget
            .max(current_len.saturating_add(policy.legacy_max_densify_per_update))
    }
}

#[cfg(test)]
pub(super) fn prune_snapshot(
    policy: &TopologyPolicy,
    snapshot: &mut Splats,
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
        let sh_coeffs = snapshot.sh_coeffs_row(idx);
        let info = infos.get(idx).copied().unwrap_or_else(|| {
            let scale = snapshot.scale(idx);
            TopologyCandidateInfo {
                max_scale: scale[0].max(scale[1]).max(scale[2]),
                opacity: sigmoid_scalar(snapshot.opacity_logits[idx]),
                mean2d_grad: stats.get(idx).copied().unwrap_or_default().mean2d_grad.mean,
                visible_count: 0,
                prune_candidate: false,
                growth_candidate: false,
            }
        });
        let opacity = info.opacity;
        let max_scale = info.max_scale;
        let valid = opacity.is_finite()
            && opacity >= policy.prune_threshold
            && max_scale.is_finite()
            && max_scale <= policy.legacy_prune_scale_threshold
            && position.iter().all(|value| value.is_finite())
            && rotation.iter().all(|value| value.is_finite())
            && sh_coeffs.iter().all(|value| value.is_finite());
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

    filter_snapshot(snapshot, stats, origins, &keep_mask)
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
        log_scale[max_axis] = (max_axis_scale / 1.6).max(1e-6).ln();

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
    #[allow(dead_code)]
    pub(super) morton_sorted: bool,
    pub(super) aftermath: TopologyMutationAftermath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyStatsAction {
    KeepCurrent,
    UseMutated,
    ResetAll,
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
    pub(super) policy: &'a TopologyPolicy,
    pub(super) should_densify: bool,
    pub(super) should_prune: bool,
    pub(super) should_reset_opacity: bool,
    pub(super) completed_epoch: Option<usize>,
    pub(super) late_stage: bool,
    pub(super) max_gaussians: usize,
    pub(super) infos: &'a [TopologyCandidateInfo],
    pub(super) litegs_selection: &'a LiteGsDensifySelection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TopologyPlanRow {
    Existing {
        source_idx: usize,
    },
    LegacyOffsetClone {
        source_idx: usize,
        axis: usize,
    },
    LegacySplit {
        source_idx: usize,
        direction: i8,
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
    pub(super) fn source_idx(&self) -> usize {
        match *self {
            Self::Existing { source_idx }
            | Self::LegacyOffsetClone { source_idx, .. }
            | Self::LegacySplit { source_idx, .. }
            | Self::BrushRefineExisting { source_idx, .. }
            | Self::BrushRefineNew { source_idx, .. } => source_idx,
        }
    }

    pub(super) fn is_existing(&self) -> bool {
        matches!(
            self,
            Self::Existing { .. } | Self::BrushRefineExisting { .. }
        )
    }

    fn position(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut position = metrics.position(source_idx);
        match *self {
            Self::Existing { .. } => position,
            Self::LegacyOffsetClone { axis, .. } => {
                let scale = metrics.scale(source_idx);
                if axis < 3 {
                    position[axis] += scale[axis].max(0.01) * 0.5;
                }
                position
            }
            Self::LegacySplit { direction, .. } => {
                position[0] += (direction as f32) * metrics.max_scale(source_idx) * 0.1;
                position
            }
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

    fn scale(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut scale = metrics.scale(source_idx);
        match *self {
            Self::Existing { .. } | Self::LegacyOffsetClone { .. } => scale,
            Self::LegacySplit { .. } => {
                scale[0] = (metrics.max_scale(source_idx) * 0.5).max(1e-6);
                scale
            }
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                scale = brush_refine_scale(scale);
                scale
            }
        }
    }

    fn max_scale(&self, metrics: &TopologySplatMetrics) -> f32 {
        let scale = self.scale(metrics);
        scale[0].max(scale[1]).max(scale[2])
    }

    fn opacity(&self, metrics: &TopologySplatMetrics) -> f32 {
        match *self {
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                brush_refine_opacity(metrics.opacity(self.source_idx()))
            }
            _ => metrics.opacity(self.source_idx()),
        }
    }

    fn retainable(&self, metrics: &TopologySplatMetrics) -> bool {
        metrics.retainable(self.source_idx())
            && self.position(metrics).iter().all(|value| value.is_finite())
    }

    fn keep_for_legacy_prune(
        &self,
        metrics: &TopologySplatMetrics,
        policy: &TopologyPolicy,
    ) -> bool {
        let opacity = self.opacity(metrics);
        let max_scale = self.max_scale(metrics);
        opacity.is_finite()
            && opacity >= policy.prune_threshold
            && max_scale.is_finite()
            && max_scale <= policy.legacy_prune_scale_threshold
            && self.retainable(metrics)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct TopologyMutationPlan {
    pub(super) rows: Vec<TopologyPlanRow>,
    pub(super) added: usize,
    pub(super) pruned: usize,
    pub(super) morton_sorted: bool,
    pub(super) aftermath: TopologyMutationAftermath,
}

impl TopologyMutationPlan {
    pub(super) fn origins(&self) -> Vec<Option<usize>> {
        self.rows
            .iter()
            .map(|row| row.is_existing().then_some(row.source_idx()))
            .collect()
    }

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

pub(super) fn plan_topology_mutation(
    metrics: &TopologySplatMetrics,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationPlan {
    if request.policy.is_litegs_mode() {
        return plan_brush_refine_mutation(metrics, request);
    }

    let mut rows: Vec<TopologyPlanRow> = (0..metrics.len())
        .map(|source_idx| TopologyPlanRow::Existing { source_idx })
        .collect();
    let additions = if request.should_densify {
        plan_legacy_densify_rows(
            request.policy,
            metrics,
            request.infos,
            request.max_gaussians,
        )
    } else {
        Vec::new()
    };
    let added = additions.len();
    rows.extend(additions);

    let mut pruned = 0usize;
    if request.should_prune && rows.len() > 1 {
        let mut keep_mask = rows
            .iter()
            .map(|row| row.keep_for_legacy_prune(metrics, request.policy))
            .collect::<Vec<_>>();

        if !keep_mask.iter().any(|keep| *keep) {
            if let Some((best_idx, _)) = rows.iter().enumerate().max_by(|lhs, rhs| {
                lhs.1
                    .opacity(metrics)
                    .partial_cmp(&rhs.1.opacity(metrics))
                    .unwrap_or(Ordering::Equal)
            }) {
                keep_mask[best_idx] = true;
            }
        }

        pruned = keep_mask.iter().filter(|keep| !**keep).count();
        if pruned > 0 {
            rows = rows
                .into_iter()
                .zip(keep_mask)
                .filter_map(|(row, keep)| keep.then_some(row))
                .collect();
        }
    }

    let morton_sorted = false;

    let aftermath = topology_mutation_aftermath(&request, rows.len(), added, pruned);
    TopologyMutationPlan {
        rows,
        added,
        pruned,
        morton_sorted,
        aftermath,
    }
}

fn plan_legacy_densify_rows(
    policy: &TopologyPolicy,
    metrics: &TopologySplatMetrics,
    infos: &[TopologyCandidateInfo],
    max_gaussians: usize,
) -> Vec<TopologyPlanRow> {
    if metrics.len() >= max_gaussians {
        return Vec::new();
    }

    let clone_opacity_threshold = policy.prune_threshold;
    let original_len = metrics.len();
    let mut clone_candidates = Vec::new();
    let mut split_candidates = Vec::new();

    for idx in 0..original_len {
        let info = infos.get(idx).copied().unwrap_or(TopologyCandidateInfo {
            max_scale: 0.0,
            opacity: 0.0,
            mean2d_grad: 0.0,
            visible_count: 0,
            prune_candidate: false,
            growth_candidate: false,
        });
        let opacity = info.opacity;
        let max_scale = info.max_scale;
        let grad_accum = info.mean2d_grad;
        if !grad_accum.is_finite() || !opacity.is_finite() {
            continue;
        }
        if grad_accum <= policy.legacy_densify_grad_threshold {
            continue;
        }
        if max_scale < policy.legacy_clone_scale_threshold && opacity > clone_opacity_threshold {
            clone_candidates.push((idx, grad_accum));
        }
        if max_scale > policy.legacy_split_scale_threshold && opacity > policy.prune_threshold {
            split_candidates.push((idx, grad_accum * max_scale));
        }
    }

    clone_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));
    split_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

    let mut rows = Vec::new();
    let mut available = max_gaussians.saturating_sub(metrics.len());
    let per_pass_limit = policy
        .legacy_max_densify_per_update
        .min(available)
        .min((original_len / 32).max(32));
    let clone_limit = clone_candidates.len().min(per_pass_limit);
    for (rank, (idx, score)) in clone_candidates.into_iter().take(clone_limit).enumerate() {
        if score <= 0.0 {
            continue;
        }
        rows.push(TopologyPlanRow::LegacyOffsetClone {
            source_idx: idx,
            axis: rank % 3,
        });
        available = available.saturating_sub(1);
        if available == 0 {
            return rows;
        }
    }

    let split_limit = split_candidates
        .len()
        .min((per_pass_limit / 4).max(1))
        .min(available / 2);
    for (idx, score) in split_candidates.into_iter().take(split_limit) {
        if score <= policy.legacy_densify_grad_threshold {
            continue;
        }
        for direction in [1i8, -1i8] {
            if available == 0 {
                break;
            }
            rows.push(TopologyPlanRow::LegacySplit {
                source_idx: idx,
                direction,
            });
            available = available.saturating_sub(1);
        }
    }

    rows
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
        let mut rng = rand::thread_rng();
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
        morton_sorted: false,
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
        morton_sorted: plan.morton_sorted,
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
    } else if request.should_reset_opacity && !request.policy.is_litegs_mode() {
        TopologyStatsAction::ResetAll
    } else {
        TopologyStatsAction::KeepCurrent
    };

    TopologyMutationAftermath {
        requires_runtime_rebuild: topology_changed,
        requires_adam_rebuild: topology_changed,
        requires_cluster_resync: topology_changed,
        requires_runtime_reserve: topology_changed,
        apply_opacity_reset: request.should_reset_opacity,
        reset_refine_window_stats: request.policy.is_litegs_mode(),
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

pub(super) fn apply_topology_metrics_delta(
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

fn density_controller_opacity_reset_mode(mode: LiteGsOpacityResetMode) -> OpacityResetMode {
    match mode {
        LiteGsOpacityResetMode::Decay => OpacityResetMode::Decay,
        LiteGsOpacityResetMode::Reset => OpacityResetMode::Reset,
    }
}

fn density_controller_prune_mode(mode: LiteGsPruneMode) -> PruneMode {
    match mode {
        LiteGsPruneMode::Threshold => PruneMode::Threshold,
        LiteGsPruneMode::Weight => PruneMode::Weight,
    }
}

fn density_controller_taming_mode(policy: &TopologyPolicy) -> bool {
    policy.is_litegs_mode() && matches!(policy.litegs.prune_mode, LiteGsPruneMode::Weight)
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
    sample_weighted(rng, sanitized.len(), |idx| sanitized[idx], count.min(sanitized.len()))
        .map(|sample| sample.into_iter().collect())
        .unwrap_or_default()
}

fn brush_refine_sample_scalar<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-6, 1.0 - 1e-6);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

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

fn brush_refine_opacity(opacity: f32) -> f32 {
    let clamped = opacity.clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY);
    let inverted = (1.0 - clamped).max(0.0);
    (1.0 - inverted.sqrt()).clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY)
}

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

fn should_densify_at(densify_interval: usize, topology_warmup: usize, iteration: usize) -> bool {
    densify_interval > 0 && iteration > topology_warmup && iteration % densify_interval == 0
}

fn should_prune_at(prune_interval: usize, topology_warmup: usize, iteration: usize) -> bool {
    prune_interval > 0 && iteration > topology_warmup && iteration % prune_interval == 0
}

fn litegs_current_epoch(iteration: usize, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || iteration == 0 {
        return None;
    }
    Some(iteration.saturating_sub(1) / frame_count)
}

#[cfg(test)]
fn filter_snapshot(
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    keep_mask: &[bool],
) -> usize {
    let pruned = keep_mask.iter().filter(|keep| !**keep).count();
    if pruned == 0 {
        return 0;
    }

    let mut kept_snapshot = snapshot.retained_view(snapshot.len() - pruned);
    let mut kept_stats = Vec::with_capacity(snapshot.len() - pruned);
    let mut kept_origins = Vec::with_capacity(snapshot.len() - pruned);

    for idx in 0..snapshot.len() {
        if keep_mask[idx] {
            kept_snapshot.push(
                snapshot.position(idx),
                snapshot.log_scale(idx),
                snapshot.rotation(idx),
                snapshot.opacity_logits[idx],
                snapshot.sh_coeffs_row(idx),
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
    use super::{
        analyze_topology_candidates, apply_snapshot_mutations, apply_topology_metrics_delta,
        densify_snapshot_litegs, density_controller_reference_summary, litegs_requested_additions,
        litegs_select_densify_candidates, prune_snapshot, schedule_topology,
        should_collect_visible_indices, LiteGsDensifySelection, MetalGaussianStats, RunningMoments,
        TopologyAnalysis, TopologyMutationRequest, TopologyPolicy, TopologySchedule,
        TopologyStatsAction, TopologyStepContext,
    };
    use crate::diff::diff_splat::{
        rgb_to_sh0_value, sh_coeff_count_for_degree, SplatColorRepresentation,
    };
    use crate::training::parity_harness::ParityTopologyMetrics;
    use crate::training::runtime_splats::TopologySplatMetrics;
    use crate::training::splats::Splats;
    use crate::training::{LiteGsConfig, TrainingConfig, TrainingProfile};

    fn legacy_policy() -> TopologyPolicy {
        let config = TrainingConfig {
            densify_interval: 128,
            prune_interval: 200,
            topology_warmup: 0,
            prune_threshold: 0.05,
            max_initial_gaussians: 128,
            iterations: 1000,
            ..TrainingConfig::default()
        };
        TopologyPolicy::from_training_config(&config, 1.0)
    }

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
                    let rest = super::super::splats::row_slice(&sh_rest, sh_rest_row_width, idx);
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
    fn legacy_schedule_keeps_densify_and_prune_independent() {
        let policy = legacy_policy();

        let densify = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 128,
                frame_count: 1,
            },
        );
        let prune = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 200,
                frame_count: 1,
            },
        );

        assert_eq!(
            densify,
            TopologySchedule {
                completed_epoch: None,
                densify: true,
                prune: false,
                reset_opacity: false,
                allow_extra_growth: false,
            }
        );
        assert_eq!(
            prune,
            TopologySchedule {
                completed_epoch: None,
                densify: false,
                prune: true,
                reset_opacity: false,
                allow_extra_growth: false,
            }
        );
    }

    #[test]
    fn litegs_schedule_respects_refine_cadence_and_freeze() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 30,
            topology_warmup: 0,
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
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 500,
            topology_warmup: 100,
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
    fn prune_snapshot_keeps_best_gaussian_when_all_candidates_fail() {
        let policy = legacy_policy();
        let mut snapshot = test_snapshot(
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 2.0],
            vec![10.0; 6],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![-10.0, -1.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            Vec::new(),
            SplatColorRepresentation::Rgb,
        );
        let mut stats = vec![MetalGaussianStats::default(), MetalGaussianStats::default()];
        let mut origins = vec![Some(0), Some(1)];
        let infos = TopologyAnalysis::default().infos;

        let pruned = prune_snapshot(&policy, &mut snapshot, &mut stats, &mut origins, &infos);

        assert_eq!(pruned, 1);
        assert_eq!(snapshot.len(), 1);
        assert_eq!(origins, vec![Some(1)]);
    }

    #[test]
    fn litegs_densify_preserves_sh_layout() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
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
            training_profile: TrainingProfile::LiteGsMacV1,
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
            training_profile: TrainingProfile::LiteGsMacV1,
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
            training_profile: TrainingProfile::LiteGsMacV1,
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
    fn collect_visible_indices_only_when_legacy_topology_work_is_scheduled() {
        let policy = legacy_policy();

        assert!(!should_collect_visible_indices(
            &policy,
            TopologySchedule::default(),
        ));
        assert!(should_collect_visible_indices(
            &policy,
            TopologySchedule {
                densify: true,
                ..TopologySchedule::default()
            },
        ));
        assert!(should_collect_visible_indices(
            &policy,
            TopologySchedule {
                prune: true,
                ..TopologySchedule::default()
            },
        ));
    }

    #[test]
    fn mutation_aftermath_requests_structural_rebuild_and_metrics() {
        let policy = legacy_policy();
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

        let mutation = apply_snapshot_mutations(
            &mut snapshot,
            &mut stats,
            &mut origins,
            TopologyMutationRequest {
                policy: &policy,
                should_densify: true,
                should_prune: true,
                should_reset_opacity: false,
                completed_epoch: Some(2),
                late_stage: true,
                max_gaussians: 4,
                infos: &infos,
                litegs_selection: &LiteGsDensifySelection::default(),
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
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);
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
                policy: &policy,
                should_densify: false,
                should_prune: false,
                should_reset_opacity: true,
                completed_epoch: Some(3),
                late_stage: true,
                max_gaussians: 1,
                infos: &infos,
                litegs_selection: &LiteGsDensifySelection::default(),
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
            max_scale: 0.01,
            opacity,
            mean2d_grad,
            visible_count,
            prune_candidate,
            growth_candidate,
        }
    }
}
