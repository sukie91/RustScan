use super::parity_harness::{ParityLossTerms, ParityTopologyMetrics};
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
    pub topology: ParityTopologyMetrics,
    pub active_sh_degree: Option<usize>,
    pub rotation_frozen: bool,
    pub learning_rates: LiteGsOptimizerLrs,
}

static LAST_METAL_TRAINING_TELEMETRY: OnceLock<Mutex<Option<LiteGsTrainingTelemetry>>> =
    OnceLock::new();

pub(super) fn assemble_training_telemetry(
    loss_terms: ParityLossTerms,
    topology: ParityTopologyMetrics,
    active_sh_degree: Option<usize>,
    rotation_frozen: bool,
    learning_rates: LiteGsOptimizerLrs,
) -> LiteGsTrainingTelemetry {
    LiteGsTrainingTelemetry {
        loss_terms,
        topology,
        active_sh_degree,
        rotation_frozen,
        learning_rates,
    }
}

pub(super) fn store_last_metal_training_telemetry(telemetry: Option<LiteGsTrainingTelemetry>) {
    let slot = LAST_METAL_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = telemetry;
}

pub fn last_metal_training_telemetry() -> Option<LiteGsTrainingTelemetry> {
    let slot = LAST_METAL_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_training_telemetry, last_metal_training_telemetry,
        store_last_metal_training_telemetry, LiteGsOptimizerLrs,
    };
    use crate::training::parity_harness::{ParityLossTerms, ParityTopologyMetrics};

    #[test]
    fn telemetry_store_round_trips_latest_snapshot() {
        store_last_metal_training_telemetry(None);
        let snapshot = assemble_training_telemetry(
            ParityLossTerms::default(),
            ParityTopologyMetrics::default(),
            Some(3),
            true,
            LiteGsOptimizerLrs::default(),
        );
        store_last_metal_training_telemetry(Some(snapshot.clone()));

        assert_eq!(last_metal_training_telemetry(), Some(snapshot));
    }
}
