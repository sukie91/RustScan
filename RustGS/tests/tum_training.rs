use std::path::PathBuf;

use rustgs::{load_training_dataset, TrainingConfig, TumRgbdConfig};

fn tum_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/tum")
}

#[test]
fn loads_workspace_tum_directory_as_training_dataset() {
    let dataset = load_training_dataset(
        &tum_root(),
        &TumRgbdConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(dataset.poses.len() >= 3);
    assert_eq!(dataset.depth_scale, 5000.0);
    assert!(dataset.poses.iter().all(|pose| pose.depth_path.is_some()));
}

#[cfg(feature = "gpu")]
#[test]
fn trains_directly_from_workspace_tum_directory() {
    let mut config = TrainingConfig::default();
    config.iterations = 1;
    config.max_initial_gaussians = 10_000;

    let scene = rustgs::train_from_path(
        &tum_root(),
        &TumRgbdConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        },
        &config,
    )
    .unwrap();

    assert!(scene.len() > 0);
}
