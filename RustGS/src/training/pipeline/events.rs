use super::splats::HostSplats;
use super::telemetry::LiteGsTrainingTelemetry;
use super::TrainingProfile;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingEventRoute {
    Standard,
    ChunkedSingleChunk,
    ChunkedSequential,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingPlanEstimate {
    pub requested_initial_gaussians: usize,
    pub affordable_initial_gaussians: usize,
    pub estimated_peak_gib: f64,
    pub effective_budget_gib: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRunStarted {
    pub profile: TrainingProfile,
    pub iterations: usize,
    pub frame_count: usize,
    pub input_point_count: usize,
    pub chunked: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingPlanSelected {
    pub route: TrainingEventRoute,
    pub training_chunks: Option<usize>,
    pub estimate: Option<TrainingPlanEstimate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingChunkStarted {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub chunk_id: usize,
    pub pose_count: usize,
    pub initial_point_count: usize,
    pub used_frame_based_initialization: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingChunkCompleted {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub chunk_id: usize,
    pub chunk_gaussian_count: usize,
    pub merged_gaussian_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunCompleted {
    pub report: TrainingRunReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingEvent {
    RunStarted(TrainingRunStarted),
    PlanSelected(TrainingPlanSelected),
    ChunkStarted(TrainingChunkStarted),
    ChunkCompleted(TrainingChunkCompleted),
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
