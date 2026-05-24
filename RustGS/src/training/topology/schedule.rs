use super::{
    LiteGsDensifySelection, TopologyAnalysis, TopologyPolicy, BRUSH_REFINE_PROGRESS_LIMIT,
};
use crate::training::{LiteGsOpacityResetMode, TrainingConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyExecutionDisposition {
    Apply,
    SkipNoEligibleCandidates,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyExecutionPlan {
    pub(super) completed_epoch: Option<usize>,
    pub(super) should_densify: bool,
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
    let Some(epoch) = litegs_current_epoch(step.iteration, step.frame_count) else {
        return TopologySchedule::default();
    };
    let phase_iter = step.iteration.saturating_sub(1);
    let topology_frozen = policy
        .litegs
        .topology
        .topology_freeze_after_epoch
        .map(|freeze_epoch| epoch >= freeze_epoch)
        .unwrap_or(false);
    let growth_frozen = topology_frozen
        || policy
            .litegs
            .topology
            .growth_freeze_after_epoch
            .map(|freeze_epoch| epoch >= freeze_epoch)
            .unwrap_or(false);
    let refine_every = policy.litegs.topology.refine_every.max(1);
    let progress = if policy.max_iterations == 0 {
        1.0
    } else {
        phase_iter as f32 / policy.max_iterations as f32
    };
    let on_refine_cadence = phase_iter > 0 && phase_iter.is_multiple_of(refine_every);
    let growth_window = progress <= BRUSH_REFINE_PROGRESS_LIMIT;
    let prune_window = policy
        .litegs
        .pruning
        .prune_until_epoch
        .map(|until_epoch| epoch < until_epoch)
        .unwrap_or(growth_window);
    let densify = on_refine_cadence && !growth_frozen && growth_window;
    let prune = on_refine_cadence && !topology_frozen && prune_window;
    let reset_opacity = on_refine_cadence
        && !topology_frozen
        && matches!(
            policy.litegs.topology.opacity_reset_mode,
            LiteGsOpacityResetMode::Reset
        )
        && policy.litegs.topology.opacity_reset_interval > 0
        && epoch > 0
        && epoch.is_multiple_of(policy.litegs.topology.opacity_reset_interval);
    TopologySchedule {
        completed_epoch: Some(epoch),
        densify,
        prune,
        reset_opacity,
        allow_extra_growth: densify && phase_iter < policy.litegs.growth.growth_stop_iter,
    }
}

pub(crate) fn should_apply_topology_step(
    config: &TrainingConfig,
    iteration: usize,
    frame_count: usize,
) -> bool {
    let policy = TopologyPolicy::from_training_config(config, 1.0);
    let schedule = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration,
            frame_count,
        },
    );
    schedule.densify || schedule.prune || schedule.reset_opacity
}

pub(super) fn plan_topology_execution(
    _policy: &TopologyPolicy,
    schedule: TopologySchedule,
    analysis: &TopologyAnalysis,
    litegs_selection: &LiteGsDensifySelection,
) -> TopologyExecutionPlan {
    let mut plan = TopologyExecutionPlan {
        completed_epoch: schedule.completed_epoch,
        should_densify: schedule.densify,
        should_reset_opacity: schedule.reset_opacity,
        disposition: TopologyExecutionDisposition::Apply,
    };

    let has_candidates =
        !litegs_selection.selected_indices.is_empty() || analysis.prune_candidates > 0;
    if !has_candidates && !plan.should_reset_opacity {
        plan.disposition = TopologyExecutionDisposition::SkipNoEligibleCandidates;
    }

    plan
}

fn litegs_current_epoch(iteration: usize, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || iteration == 0 {
        return None;
    }
    Some(iteration.saturating_sub(1) / frame_count)
}
