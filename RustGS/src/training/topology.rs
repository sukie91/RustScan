use std::cmp::Ordering;

use super::splats::Splats;
use super::{LiteGsConfig, LiteGsPruneMode, TrainingProfile};

#[cfg(test)]
use super::TrainingConfig;

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
    pub(super) visible_count: usize,
    pub(super) age_eligible: bool,
    pub(super) invisible: bool,
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

    pub(super) fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        if let Some(until) = self.litegs.densify_until {
            return until.max(self.litegs.densify_from.saturating_add(1));
        }

        let total_epochs = self.litegs_total_epochs(frame_count);
        let reset_interval = self.litegs.opacity_reset_interval.max(1);
        let scaled = ((total_epochs as f32) * 0.8).floor() as usize;
        let computed = (scaled / reset_interval) * reset_interval + 1;
        computed.max(self.litegs.densify_from.saturating_add(1))
    }

    pub(super) fn litegs_clone_scale_threshold(&self) -> f32 {
        (self.scene_extent * LITEGS_PERCENT_DENSE).max(1e-4)
    }
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
        let passed_warmup = step.iteration > policy.topology_warmup;
        let densify_until = policy.litegs_densify_until_epoch(step.frame_count);
        let active_window = epoch >= policy.litegs.densify_from && epoch < densify_until;
        let frozen = policy
            .litegs
            .topology_freeze_after_epoch
            .map(|freeze_epoch| epoch >= freeze_epoch)
            .unwrap_or(false);
        let refine_every = policy.litegs.refine_every.max(1);
        let refine =
            passed_warmup && !frozen && active_window && step.iteration % refine_every == 0;
        let opacity_reset_period =
            refine_every.saturating_mul(policy.litegs.opacity_reset_interval.max(1));
        let reset_opacity = passed_warmup
            && !frozen
            && active_window
            && step.iteration % opacity_reset_period.max(1) == 0;
        return TopologySchedule {
            completed_epoch: Some(epoch),
            densify: refine,
            prune: refine,
            reset_opacity,
            allow_extra_growth: refine && step.iteration < policy.litegs.growth_stop_iter,
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

pub(super) fn analyze_topology_candidates(
    policy: &TopologyPolicy,
    splats: &Splats,
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

    for idx in 0..splats.len() {
        let max_scale = splats.scale(idx).into_iter().fold(0.0f32, f32::max);
        let opacity = sigmoid_scalar(splats.opacity_logits[idx]);
        let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
        let mean2d_grad = gaussian_stats.mean2d_grad.mean;

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

        let growth_threshold = if policy.is_litegs_mode() {
            policy.litegs.growth_grad_threshold
        } else {
            policy.legacy_densify_grad_threshold
        };
        let growth_candidate = mean2d_grad.is_finite()
            && mean2d_grad >= growth_threshold
            && opacity > LITEGS_OPACITY_THRESHOLD;
        let candidate_info = TopologyCandidateInfo {
            max_scale,
            opacity,
            mean2d_grad,
            visible_count: gaussian_stats.visible_count,
            age_eligible,
            invisible: invisible_for_prune,
            prune_candidate: false,
            growth_candidate,
        };
        let prune_candidate = if policy.is_litegs_mode() {
            litegs_should_prune_candidate(policy, &candidate_info)
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

    let threshold_count = infos
        .iter()
        .filter(|info| !info.prune_candidate && info.growth_candidate)
        .count();
    let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;

    prune_candidates.saturating_add(grow_count.saturating_sub(prune_candidates))
}

pub(super) fn litegs_select_densify_candidates(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
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
    replacement_sources.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

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
        let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;
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

pub(super) fn densify_snapshot(
    policy: &TopologyPolicy,
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    infos: &[TopologyCandidateInfo],
    max_gaussians: usize,
) -> usize {
    if snapshot.len() >= max_gaussians {
        return 0;
    }

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

    let mut available = max_gaussians.saturating_sub(snapshot.len());
    let per_pass_limit = policy
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
        if score <= policy.legacy_densify_grad_threshold {
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
        let color = snapshot.color(idx);
        let info = infos.get(idx).copied().unwrap_or_else(|| {
            let scale = snapshot.scale(idx);
            TopologyCandidateInfo {
                max_scale: scale[0].max(scale[1]).max(scale[2]),
                opacity: sigmoid_scalar(snapshot.opacity_logits[idx]),
                mean2d_grad: stats.get(idx).copied().unwrap_or_default().mean2d_grad.mean,
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
            && opacity >= policy.prune_threshold
            && max_scale.is_finite()
            && max_scale <= policy.legacy_prune_scale_threshold
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

    filter_snapshot(snapshot, stats, origins, &keep_mask)
}

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
        let color = snapshot.color(*idx);
        let sh_rest = snapshot.sh_rest(*idx).to_vec();
        let scale = snapshot.scale(*idx);

        let (max_axis, max_axis_scale) = scale
            .into_iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
            .unwrap_or((0, 0.0));
        position[max_axis] += max_axis_scale * 0.5;
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

pub(super) fn prune_snapshot_litegs(
    snapshot: &mut Splats,
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

fn litegs_should_prune_candidate(policy: &TopologyPolicy, info: &TopologyCandidateInfo) -> bool {
    let prune_opacity = match policy.litegs.prune_mode {
        LiteGsPruneMode::Weight => info.age_eligible && info.visible_count == 0,
        LiteGsPruneMode::Threshold => {
            (info.age_eligible && info.opacity < LITEGS_OPACITY_THRESHOLD) || info.invisible
        }
    };

    let prune_scale = policy.litegs.prune_scale_threshold > 0.0
        && info.max_scale > policy.litegs.prune_scale_threshold;

    prune_opacity || prune_scale
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
        analyze_topology_candidates, densify_snapshot_litegs, litegs_requested_additions,
        litegs_select_densify_candidates, prune_snapshot, schedule_topology,
        should_collect_visible_indices, MetalGaussianStats, RunningMoments, TopologyAnalysis,
        TopologyPolicy, TopologySchedule, TopologyStepContext,
    };
    use crate::diff::diff_splat::TrainableColorRepresentation;
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
                iteration: 6,
                frame_count: 2,
            },
        );
        let epoch4 = schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: 10,
                frame_count: 2,
            },
        );

        assert_eq!(epoch2.completed_epoch, Some(2));
        assert!(epoch2.densify);
        assert!(epoch2.prune);
        assert!(!epoch2.reset_opacity);
        assert!(epoch2.allow_extra_growth);

        assert_eq!(epoch4.completed_epoch, Some(4));
        assert!(!epoch4.densify);
        assert!(!epoch4.prune);
        assert!(!epoch4.reset_opacity);
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
        assert_eq!(selection.selected_indices[0], 1);
        assert_eq!(selection.selected_indices.len(), 3);
    }

    #[test]
    fn prune_snapshot_keeps_best_gaussian_when_all_candidates_fail() {
        let policy = legacy_policy();
        let mut snapshot = Splats {
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
        let mut snapshot = Splats {
            positions: vec![0.0, 0.0, 1.0],
            log_scales: vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            rotations: vec![1.0, 0.0, 0.0, 0.0],
            opacity_logits: vec![0.5],
            colors: vec![0.2, 0.4, 0.6],
            sh_rest: vec![0.1; 15 * 3],
            color_representation: TrainableColorRepresentation::SphericalHarmonics { degree: 3 },
        };
        let mut stats = vec![MetalGaussianStats::default()];
        let mut origins = vec![Some(0)];

        let added =
            densify_snapshot_litegs(&policy, &mut snapshot, &mut stats, &mut origins, 2, &[0]);

        assert_eq!(added, 1);
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.sh_rest.len(), 2 * 15 * 3);
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
        let snapshot = Splats {
            positions: vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            log_scales: vec![
                0.005f32.ln(),
                0.005f32.ln(),
                0.005f32.ln(),
                0.5f32.ln(),
                0.5f32.ln(),
                0.5f32.ln(),
            ],
            rotations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            opacity_logits: vec![2.0, -10.0],
            colors: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            sh_rest: vec![0.0; 2 * 15 * 3],
            color_representation: TrainableColorRepresentation::SphericalHarmonics { degree: 3 },
        };
        let stats = vec![
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
                consecutive_invisible_epochs: 2,
                ..Default::default()
            },
        ];

        let analysis = analyze_topology_candidates(&policy, &snapshot, &stats);

        assert_eq!(analysis.clone_candidates, 1);
        assert_eq!(analysis.prune_candidates, 1);
        assert_eq!(analysis.growth_candidates, 1);
        assert!(analysis.infos[0].growth_candidate);
        assert!(analysis.infos[1].prune_candidate);
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
            age_eligible: true,
            invisible: false,
            prune_candidate,
            growth_candidate,
        }
    }
}
