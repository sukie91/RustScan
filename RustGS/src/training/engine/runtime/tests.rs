use super::super::trainer::TrainingLoopObserver;
use super::*;
use crate::training::reporting::telemetry::{last_training_telemetry, LiteGsTrainingTelemetry};

#[test]
fn observer_emits_progress_and_snapshot_on_cadence() {
    let control = TrainingControl::new(TrainingEventCadence {
        progress_every: 2,
        snapshot_every: Some(3),
    });
    let mut events = Vec::new();
    let started = Instant::now();
    let mut sink = |event| events.push(event);
    let mut observer = TrainingEventObserver {
        control: &control,
        cadence: control.cadence(),
        started_at: started,
        on_event: &mut sink,
    };

    observer.on_iteration(TrainingIterationMetrics {
        iteration: 1,
        loss: 1.0,
        gaussian_count: 8,
    });
    observer.on_iteration(TrainingIterationMetrics {
        iteration: 2,
        loss: 0.8,
        gaussian_count: 8,
    });
    assert!(observer.should_emit_snapshot(3));
    observer.on_snapshot(
        TrainingIterationMetrics {
            iteration: 3,
            loss: 0.7,
            gaussian_count: 9,
        },
        HostSplats::default(),
    );

    assert!(events.iter().any(|event| matches!(
        event,
        TrainingEvent::IterationProgress(TrainingIterationProgress { iteration: 2, .. })
    )));
    assert!(events
        .iter()
        .any(|event| matches!(event, TrainingEvent::SnapshotReady(_))));
}

#[test]
fn observer_honors_cooperative_cancellation() {
    let control = TrainingControl::default();
    let mut sink = |_event| {};
    let observer = TrainingEventObserver {
        control: &control,
        cadence: control.cadence(),
        started_at: Instant::now(),
        on_event: &mut sink,
    };
    assert!(!observer.should_cancel());
    control.request_cancel();
    assert!(observer.should_cancel());
}

#[test]
fn build_training_run_publishes_report_telemetry() {
    let telemetry = LiteGsTrainingTelemetry {
        final_loss: Some(0.25),
        final_step_loss: Some(0.25),
        ..LiteGsTrainingTelemetry::default()
    };
    let run = build_training_run(
        HostSplats::default(),
        WgpuTrainingReport {
            final_loss: Some(0.25),
            final_step_loss: Some(0.25),
            final_gaussian_count: 0,
            completed_iterations: 3,
            cancelled: false,
            telemetry: telemetry.clone(),
        },
        std::time::Duration::from_millis(12),
    );

    assert_eq!(run.report.telemetry, Some(telemetry.clone()));
    assert_eq!(last_training_telemetry(), Some(telemetry));
}
