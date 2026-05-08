use super::{
    last_training_telemetry, store_last_training_telemetry, LiteGsOptimizerLrs,
    LiteGsTrainingTelemetry,
};
use crate::training::reporting::metrics::{ParityLossTerms, ParityTopologyMetrics};

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
