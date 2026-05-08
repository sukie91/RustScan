use super::{TrainingControl, TrainingEventCadence};

#[test]
fn cadence_checks_progress_and_snapshot_intervals() {
    let cadence = TrainingEventCadence {
        progress_every: 2,
        snapshot_every: Some(3),
    };

    assert!(!cadence.should_emit_progress(1));
    assert!(cadence.should_emit_progress(2));
    assert!(!cadence.should_emit_snapshot(2));
    assert!(cadence.should_emit_snapshot(3));
}

#[test]
fn control_cancellation_is_cooperative_and_clone_safe() {
    let control = TrainingControl::new(TrainingEventCadence::default());
    let control_clone = control.clone();
    assert!(!control.is_cancel_requested());
    control_clone.request_cancel();
    assert!(control.is_cancel_requested());
}
