use super::parity_harness::{ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsOptimizerLrs {
    pub xyz: Option<f32>,
    pub sh_0: Option<f32>,
    pub sh_rest: Option<f32>,
    pub opacity: Option<f32>,
    pub scale: Option<f32>,
    pub rot: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsTrainingTelemetry {
    pub loss_terms: ParityLossTerms,
    pub loss_curve_samples: Vec<ParityLossCurveSample>,
    pub topology: ParityTopologyMetrics,
    pub active_sh_degree: Option<usize>,
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
    pub depth_grad_scale: Option<f32>,
    pub rotation_frozen: bool,
    pub learning_rates: LiteGsOptimizerLrs,
}

static LAST_TRAINING_TELEMETRY: OnceLock<Mutex<Option<LiteGsTrainingTelemetry>>> = OnceLock::new();

pub(super) fn store_last_training_telemetry(telemetry: Option<LiteGsTrainingTelemetry>) {
    let slot = LAST_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = telemetry;
}

pub fn last_training_telemetry() -> Option<LiteGsTrainingTelemetry> {
    let slot = LAST_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

#[cfg(test)]
mod tests {
    use super::{
        last_training_telemetry, store_last_training_telemetry, LiteGsOptimizerLrs,
        LiteGsTrainingTelemetry,
    };
    use crate::training::parity_harness::{ParityLossTerms, ParityTopologyMetrics};

    #[test]
    fn telemetry_store_round_trips_latest_snapshot() {
        store_last_training_telemetry(None);
        let snapshot = LiteGsTrainingTelemetry {
            loss_terms: ParityLossTerms::default(),
            loss_curve_samples: Vec::new(),
            topology: ParityTopologyMetrics::default(),
            active_sh_degree: Some(3),
            final_loss: Some(0.25),
            final_step_loss: Some(0.2),
            depth_valid_pixels: Some(128),
            depth_grad_scale: Some(0.5),
            rotation_frozen: true,
            learning_rates: LiteGsOptimizerLrs::default(),
        };
        store_last_training_telemetry(Some(snapshot.clone()));

        assert_eq!(last_training_telemetry(), Some(snapshot));
    }
}
