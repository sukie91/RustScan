use std::cmp::Ordering;

use super::splat_params::SplatParameterView;
use super::training_pipeline::TrainingConfig as TopologyConfig;
use super::{LiteGsConfig, LiteGsPruneMode, TrainingProfile};

const LITEGS_DENSIFY_GRAD_THRESHOLD: f32 = 0.00015;
const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_PERCENT_DENSE: f32 = 0.01;

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

    pub(super) fn variance(self) -> f32 {
        if self.count <= 1 {
            0.0
        } else {
            self.m2 / self.count as f32
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct MetalGaussianStats {
    pub(super) mean2d_grad: RunningMoments,
    pub(super) fragment_weight: RunningMoments,
    pub(super) fragment_err: RunningMoments,
    pub(super) visible_count: usize,
    pub(super) age: usize,
    pub(super) consecutive_invisible_epochs: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyCandidateInfo {
    pub(super) max_scale: f32,
    pub(super) opacity: f32,
    pub(super) mean2d_grad: f32,
    pub(super) fragment_weight: f32,
    pub(super) fragment_err_score: f32,
    pub(super) invisible: bool,
}

#[derive(Debug, Default)]
pub(super) struct TopologyAnalysis {
    pub(super) infos: Vec<TopologyCandidateInfo>,
    pub(super) clone_candidates: usize,
    pub(super) split_candidates: usize,
    pub(super) prune_candidates: usize,
    pub(super) active_grad_stats: usize,
    pub(super) small_scale_stats: usize,
    pub(super) opacity_ready_stats: usize,
    pub(super) max_grad: f32,
    pub(super) mean_grad: f32,
}

impl TopologyAnalysis {
    pub(super) fn has_work(&self) -> bool {
        self.clone_candidates > 0 || self.split_candidates > 0 || self.prune_candidates > 0
    }
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
    pub(super) max_gaussian_budget: usize,
    pub(super) scene_extent: f32,
    pub(super) max_iterations: usize,
}

impl TopologyPolicy {
    pub(super) fn is_litegs_mode(&self) -> bool {
        self.training_profile == TrainingProfile::LiteGsMacV1
    }

    pub(super) fn should_log_topology(&self, iteration: usize) -> bool {
        iteration % self.topology_log_interval.max(1) == 0
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
        (scaled / reset_interval) * reset_interval + 1
    }

    fn litegs_clone_scale_threshold(&self) -> f32 {
        (self.scene_extent * LITEGS_PERCENT_DENSE).max(1e-4)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TopologySchedule {
    pub(super) completed_epoch: Option<usize>,
    pub(super) densify: bool,
    pub(super) prune: bool,
    pub(super) reset_opacity: bool,
}

impl TopologySchedule {
    pub(super) fn any(self) -> bool {
        self.densify || self.prune || self.reset_opacity
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyStepContext {
    pub(super) iteration: usize,
    pub(super) frame_idx: usize,
    pub(super) frame_count: usize,
}

pub(super) fn schedule_topology(
    policy: &TopologyPolicy,
    step: TopologyStepContext,
) -> TopologySchedule {
    if policy.is_litegs_mode() {
        let Some(epoch) = litegs_completed_epoch(step.iteration, step.frame_idx, step.frame_count)
        else {
            return TopologySchedule::default();
        };
        let densify_until = policy.litegs_densify_until_epoch(step.frame_count);
        let active_window = epoch >= policy.litegs.densify_from && epoch < densify_until;
        let reset_opacity =
            active_window && epoch % policy.litegs.opacity_reset_interval.max(1) == 0;
        let densify = active_window && epoch % policy.litegs.densification_interval.max(1) == 0;
        let prune_epoch = epoch.saturating_sub(policy.litegs.prune_offset_epochs);
        let prune = active_window
            && prune_epoch % policy.litegs.densification_interval.max(1) == 0
            && epoch
                >= policy
                    .litegs
                    .densify_from
                    .saturating_add(policy.litegs.prune_offset_epochs);
        return TopologySchedule {
            completed_epoch: Some(epoch),
            densify,
            prune,
            reset_opacity,
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
    }
}

pub(super) fn should_collect_visible_indices(
    policy: &TopologyPolicy,
    schedule: TopologySchedule,
) -> bool {
    policy.is_litegs_mode() || schedule.densify || schedule.prune
}

pub(super) fn analyze_topology_candidates(
    policy: &TopologyPolicy,
    splats: &SplatParameterView,
    stats: &[MetalGaussianStats],
) -> TopologyAnalysis {
    let mut analysis = TopologyAnalysis {
        infos: Vec::with_capacity(splats.len()),
        ..TopologyAnalysis::default()
    };
    let mut grad_sum = 0.0f32;
    let defaults = TopologyConfig::default();
    let clone_scale_threshold = policy.litegs_clone_scale_threshold();

    for idx in 0..splats.len() {
        let scale = splats.scale(idx);
        let max_scale = scale.into_iter().fold(0.0f32, f32::max);
        let opacity = sigmoid_scalar(splats.opacity_logits[idx]);
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

        let (invisible_for_prune, age_eligible) = if policy.is_litegs_mode() {
            let min_age = policy.litegs.prune_min_age.max(1);
            let min_invisible = policy.litegs.prune_invisible_epochs.max(1);
            let age_ok = gaussian_stats.age >= min_age;
            let invisible_long_enough =
                gaussian_stats.consecutive_invisible_epochs >= min_invisible;
            (age_ok && invisible_long_enough, age_ok)
        } else {
            (gaussian_stats.visible_count == 0, true)
        };

        analysis.infos.push(TopologyCandidateInfo {
            max_scale,
            opacity,
            mean2d_grad,
            fragment_weight,
            fragment_err_score,
            invisible: invisible_for_prune,
        });
        if mean2d_grad > LITEGS_DENSIFY_GRAD_THRESHOLD {
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
            if fragment_err_score > 0.0 && opacity > LITEGS_OPACITY_THRESHOLD {
                if max_scale <= clone_scale_threshold {
                    analysis.clone_candidates += 1;
                } else {
                    analysis.split_candidates += 1;
                }
            }
            let should_prune = match policy.litegs.prune_mode {
                LiteGsPruneMode::Weight => {
                    invisible_for_prune || (age_eligible && fragment_weight <= 0.0)
                }
                LiteGsPruneMode::Threshold => {
                    invisible_for_prune || (age_eligible && opacity < LITEGS_OPACITY_THRESHOLD)
                }
            };
            if should_prune {
                analysis.prune_candidates += 1;
            }
        } else {
            if mean2d_grad > defaults.densify_grad_threshold && opacity > policy.prune_threshold {
                if max_scale < 0.1 {
                    analysis.clone_candidates += 1;
                }
                if max_scale > 0.3 {
                    analysis.split_candidates += 1;
                }
            }
            if opacity < policy.prune_threshold || max_scale > defaults.prune_scale_threshold {
                analysis.prune_candidates += 1;
            }
        }
    }

    if !analysis.infos.is_empty() {
        analysis.mean_grad = grad_sum / analysis.infos.len() as f32;
    }
    analysis
}

pub(super) fn litegs_densify_budget(
    policy: &TopologyPolicy,
    initial_len: usize,
    current_len: usize,
    prune_candidates: usize,
    epoch: usize,
    frame_count: usize,
) -> usize {
    let init = initial_len.max(1);
    let densify_until = policy.litegs_densify_until_epoch(frame_count);
    let span = densify_until
        .saturating_sub(policy.litegs.densify_from)
        .max(1);
    let progressed = epoch.saturating_sub(policy.litegs.densify_from);
    let target = init as f32
        + ((policy.litegs.target_primitives.saturating_sub(init)) as f32 / span as f32)
            * progressed as f32;
    let target = target.round() as usize;
    let deficit = target.saturating_sub(current_len).max(1);
    deficit.saturating_add(prune_candidates).min(current_len)
}

pub(super) fn requested_gaussian_cap(
    policy: &TopologyPolicy,
    current_len: usize,
    litegs_budget: usize,
) -> usize {
    if policy.is_litegs_mode() {
        policy
            .max_gaussian_budget
            .max(current_len.saturating_add(litegs_budget))
    } else {
        policy
            .max_gaussian_budget
            .max(current_len.saturating_add(TopologyConfig::default().max_densify))
    }
}

pub(super) fn densify_snapshot(
    policy: &TopologyPolicy,
    snapshot: &mut SplatParameterView,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    infos: &[TopologyCandidateInfo],
    max_gaussians: usize,
) -> usize {
    if snapshot.len() >= max_gaussians {
        return 0;
    }

    let defaults = TopologyConfig::default();
    let clone_opacity_threshold = policy.prune_threshold;
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
            invisible: true,
        });
        if !info.mean2d_grad.is_finite() || !info.opacity.is_finite() {
            continue;
        }
        if info.mean2d_grad <= defaults.densify_grad_threshold {
            continue;
        }
        if info.max_scale < 0.1 && info.opacity > clone_opacity_threshold {
            clone_candidates.push((idx, info.mean2d_grad));
        }
        if info.max_scale > 0.3 && info.opacity > policy.prune_threshold {
            split_candidates.push((idx, info.mean2d_grad * info.max_scale));
        }
    }

    clone_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));
    split_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

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
        if score <= defaults.densify_grad_threshold {
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

pub(super) fn prune_snapshot(
    policy: &TopologyPolicy,
    snapshot: &mut SplatParameterView,
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
                mean2d_grad: stats.get(idx).copied().unwrap_or_default().mean2d_grad.mean,
                fragment_weight: 0.0,
                fragment_err_score: 0.0,
                invisible: true,
            }
        });
        let valid = info.opacity.is_finite()
            && info.opacity >= policy.prune_threshold
            && info.max_scale.is_finite()
            && info.max_scale <= defaults.prune_scale_threshold
            && position.iter().all(|value| value.is_finite())
            && rotation.iter().all(|value| value.is_finite())
            && color.iter().all(|value| value.is_finite());
        if valid {
            keep_mask[idx] = true;
        }
        let score = if info.opacity.is_finite() {
            info.opacity
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

pub(super) fn densify_snapshot_litegs(
    policy: &TopologyPolicy,
    snapshot: &mut SplatParameterView,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    infos: &[TopologyCandidateInfo],
    max_gaussians: usize,
    budget: usize,
) -> usize {
    if snapshot.len() >= max_gaussians || budget == 0 {
        return 0;
    }

    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let mut candidates: Vec<(usize, f32)> = infos
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            let score = info.fragment_err_score;
            (score.is_finite() && score > 0.0).then_some((idx, score))
        })
        .collect();
    candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

    let available = max_gaussians.saturating_sub(snapshot.len());
    let limit = budget.min(available).min(candidates.len());
    let mut added = 0usize;

    for (idx, _) in candidates.into_iter().take(limit) {
        let info = infos[idx];
        let mut position = snapshot.position(idx);
        let mut log_scale = snapshot.log_scale(idx);
        let rotation = snapshot.rotation(idx);
        let opacity_logit = snapshot.opacity_logits[idx];
        let color = snapshot.color(idx);
        let sh_rest = snapshot.sh_rest(idx).to_vec();

        if info.max_scale > clone_scale_threshold {
            let scale = snapshot.scale(idx);
            let (axis, axis_scale) = scale
                .into_iter()
                .enumerate()
                .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
                .unwrap_or((0, 0.0));
            position[axis] += axis_scale * 0.5;
            log_scale[axis] = (axis_scale / (0.8 * 2.0)).max(1e-6).ln();
        }

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

pub(super) fn prune_snapshot_litegs(
    policy: &TopologyPolicy,
    snapshot: &mut SplatParameterView,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    infos: &[TopologyCandidateInfo],
) -> usize {
    if snapshot.len() <= 1 {
        return 0;
    }

    let mut keep_mask = vec![true; snapshot.len()];
    for (idx, info) in infos.iter().enumerate() {
        let prune = match policy.litegs.prune_mode {
            LiteGsPruneMode::Weight => info.invisible || info.fragment_weight <= 0.0,
            LiteGsPruneMode::Threshold => info.invisible || info.opacity < LITEGS_OPACITY_THRESHOLD,
        };
        if prune {
            keep_mask[idx] = false;
        }
    }

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

    filter_snapshot(snapshot, stats, origins, &keep_mask)
}

fn should_densify_at(densify_interval: usize, topology_warmup: usize, iteration: usize) -> bool {
    densify_interval > 0 && iteration > topology_warmup && iteration % densify_interval == 0
}

fn should_prune_at(prune_interval: usize, topology_warmup: usize, iteration: usize) -> bool {
    prune_interval > 0 && iteration > topology_warmup && iteration % prune_interval == 0
}

fn litegs_completed_epoch(iteration: usize, frame_idx: usize, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || frame_idx + 1 != frame_count || iteration < frame_count {
        return None;
    }
    Some(iteration / frame_count - 1)
}

fn filter_snapshot(
    snapshot: &mut SplatParameterView,
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

fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::{
        prune_snapshot, schedule_topology, MetalGaussianStats, TopologyAnalysis, TopologyPolicy,
        TopologySchedule, TopologyStepContext,
    };
    use crate::diff::diff_splat::TrainableColorRepresentation;
    use crate::training::splat_params::SplatParameterView;
    use crate::training::{LiteGsConfig, TrainingProfile};

    fn legacy_policy() -> TopologyPolicy {
        TopologyPolicy {
            training_profile: TrainingProfile::LegacyMetal,
            litegs: LiteGsConfig::default(),
            prune_threshold: 0.05,
            densify_interval: 128,
            prune_interval: 200,
            topology_warmup: 0,
            topology_log_interval: 10,
            max_gaussian_budget: 128,
            scene_extent: 1.0,
            max_iterations: 1000,
        }
    }

    #[test]
    fn legacy_schedule_keeps_densify_and_prune_independent() {
        let policy = legacy_policy();

        let densify = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 128,
                frame_idx: 0,
                frame_count: 1,
            },
        );
        let prune = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 200,
                frame_idx: 0,
                frame_count: 1,
            },
        );

        assert_eq!(
            densify,
            TopologySchedule {
                completed_epoch: None,
                densify: true,
                prune: false,
                reset_opacity: false
            }
        );
        assert_eq!(
            prune,
            TopologySchedule {
                completed_epoch: None,
                densify: false,
                prune: true,
                reset_opacity: false
            }
        );
    }

    #[test]
    fn litegs_schedule_respects_prune_offset_and_reset_interval() {
        let mut policy = legacy_policy();
        policy.training_profile = TrainingProfile::LiteGsMacV1;
        policy.max_iterations = 30;
        policy.litegs.densify_from = 1;
        policy.litegs.densify_until = Some(6);
        policy.litegs.densification_interval = 2;
        policy.litegs.prune_offset_epochs = 1;
        policy.litegs.opacity_reset_interval = 2;

        let epoch2 = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 6,
                frame_idx: 1,
                frame_count: 2,
            },
        );
        let epoch3 = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 8,
                frame_idx: 1,
                frame_count: 2,
            },
        );

        assert!(epoch2.densify);
        assert!(!epoch2.prune);
        assert!(epoch2.reset_opacity);
        assert_eq!(epoch2.completed_epoch, Some(2));

        assert!(epoch3.prune);
        assert!(!epoch3.reset_opacity);
        assert_eq!(epoch3.completed_epoch, Some(3));
    }

    #[test]
    fn prune_snapshot_keeps_best_gaussian_when_all_candidates_fail() {
        let policy = legacy_policy();
        let mut snapshot = SplatParameterView {
            positions: vec![0.0, 0.0, 1.0, 1.0, 0.0, 2.0],
            log_scales: vec![10.0; 6],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            opacity_logits: vec![-10.0, -1.0],
            colors: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            sh_rest: Vec::new(),
            color_representation: TrainableColorRepresentation::Rgb,
        };
        let mut stats = vec![MetalGaussianStats::default(), MetalGaussianStats::default()];
        let mut origins = vec![Some(0), Some(1)];
        let infos = TopologyAnalysis::default().infos;

        let pruned = prune_snapshot(&policy, &mut snapshot, &mut stats, &mut origins, &infos);

        assert_eq!(pruned, 1);
        assert_eq!(snapshot.len(), 1);
        assert_eq!(origins, vec![Some(1)]);
    }

    #[test]
    fn collect_visible_indices_only_when_topology_work_is_scheduled() {
        let policy = legacy_policy();

        assert!(!super::should_collect_visible_indices(
            &policy,
            TopologySchedule::default(),
        ));
        assert!(super::should_collect_visible_indices(
            &policy,
            TopologySchedule {
                densify: true,
                ..TopologySchedule::default()
            },
        ));
        assert!(super::should_collect_visible_indices(
            &policy,
            TopologySchedule {
                prune: true,
                ..TopologySchedule::default()
            },
        ));
    }
}
