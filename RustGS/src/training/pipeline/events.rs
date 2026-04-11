use super::splats::HostSplats;
use super::telemetry::LiteGsTrainingTelemetry;
use super::TrainingProfile;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingEventRoute {
    Standard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRunStarted {
    pub profile: TrainingProfile,
    pub iterations: usize,
    pub frame_count: usize,
    pub input_point_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingPlanSelected {
    pub route: TrainingEventRoute,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunCompleted {
    pub report: TrainingRunReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingEvent {
    RunStarted(TrainingRunStarted),
    PlanSelected(TrainingPlanSelected),
    RunCompleted(TrainingRunCompleted),
}

pub type TrainingEventSink<'a> = dyn FnMut(TrainingEvent) + 'a;

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunReport {
    pub elapsed: Duration,
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub gaussian_count: usize,
    pub sh_degree: usize,
    pub telemetry: Option<LiteGsTrainingTelemetry>,
}

impl TrainingRunReport {
    pub fn metadata_final_loss_or(&self, default: f32) -> f32 {
        self.final_loss.unwrap_or(default)
    }
}

#[derive(Debug)]
pub struct TrainingRun {
    pub splats: HostSplats,
    pub report: TrainingRunReport,
}

impl TrainingRun {
    pub fn into_splats(self) -> HostSplats {
        self.splats
    }
}

pub(crate) fn emit_training_event(sink: &mut TrainingEventSink<'_>, event: TrainingEvent) {
    sink(event);
}
