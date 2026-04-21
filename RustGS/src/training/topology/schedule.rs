use super::{
    LiteGsDensifySelection, TopologyAnalysis, TopologyPolicy, BRUSH_REFINE_PROGRESS_LIMIT,
};
use crate::training::TrainingConfig;

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
    TopologySchedule {
        completed_epoch: Some(epoch),
        densify: refine,
        prune: refine,
        reset_opacity: false,
        allow_extra_growth: refine && phase_iter < policy.litegs.growth_stop_iter,
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
