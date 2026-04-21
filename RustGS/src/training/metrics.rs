use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossTerms {
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub scale_regularization: Option<f32>,
    pub transmittance: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityTopologyMetrics {
    pub initialization_gaussians: Option<usize>,
    pub final_gaussians: Option<usize>,
    pub total_epochs: Option<usize>,
    pub densify_until_epoch: Option<usize>,
    pub late_stage_start_epoch: Option<usize>,
    pub topology_freeze_epoch: Option<usize>,
    pub densify_events: usize,
    pub densify_added: usize,
    pub first_densify_epoch: Option<usize>,
    pub last_densify_epoch: Option<usize>,
    pub late_stage_densify_events: usize,
    pub late_stage_densify_added: usize,
    pub prune_events: usize,
    pub prune_removed: usize,
    pub first_prune_epoch: Option<usize>,
    pub last_prune_epoch: Option<usize>,
    pub late_stage_prune_events: usize,
    pub late_stage_prune_removed: usize,
    pub opacity_reset_events: usize,
    pub first_opacity_reset_epoch: Option<usize>,
    pub last_opacity_reset_epoch: Option<usize>,
    pub late_stage_opacity_reset_events: usize,
    pub export_outputs: usize,
    pub checkpoint_roundtrips: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossCurveSample {
    pub iteration: usize,
    pub frame_idx: usize,
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
}
