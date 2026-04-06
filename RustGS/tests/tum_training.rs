use std::path::PathBuf;

use rustgs::{
    evaluate_scene, evaluation_device, load_training_dataset, select_evaluation_frames,
    EvaluationDevice, Gaussian, SceneEvaluationConfig, SceneMetadata, TrainingConfig,
    TumRgbdConfig,
};

fn tum_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/tum")
}

fn tum_root_if_available() -> Option<PathBuf> {
    let root = tum_root();
    root.exists().then_some(root)
}

#[test]
fn loads_workspace_tum_directory_as_training_dataset() {
    let Some(root) = tum_root_if_available() else {
        eprintln!(
            "skipping test: missing TUM fixture at {}",
            tum_root().display()
        );
        return;
    };
    let dataset = load_training_dataset(
        &root,
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

#[test]
fn selects_stable_tum_eval_subset_with_stride() {
    let Some(root) = tum_root_if_available() else {
        eprintln!(
            "skipping test: missing TUM fixture at {}",
            tum_root().display()
        );
        return;
    };
    let dataset = load_training_dataset(
        &root,
        &TumRgbdConfig {
            max_frames: 180,
            frame_stride: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let selected = select_evaluation_frames(&dataset, 180, 30);
    assert!(!selected.poses.is_empty());
    assert!(selected.poses.len() <= dataset.poses.len());
}

#[cfg(feature = "gpu")]
#[test]
fn trains_directly_from_workspace_tum_directory() {
    let Some(root) = tum_root_if_available() else {
        eprintln!(
            "skipping test: missing TUM fixture at {}",
            tum_root().display()
        );
        return;
    };
    if !rustgs::metal_available() {
        eprintln!("skipping test: Metal unavailable in current environment");
        return;
    }
    let mut config = TrainingConfig::default();
    config.iterations = 1;
    config.max_initial_gaussians = 10_000;

    let scene = rustgs::train_from_path(
        &root,
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

#[cfg(feature = "gpu")]
#[test]
fn tum_training_smoke_produces_post_train_evaluation_summary() {
    let Some(root) = tum_root_if_available() else {
        eprintln!(
            "skipping test: missing TUM fixture at {}",
            tum_root().display()
        );
        return;
    };
    if !rustgs::metal_available() {
        eprintln!("skipping test: Metal unavailable in current environment");
        return;
    }

    let tum_config = TumRgbdConfig {
        max_frames: 90,
        frame_stride: 30,
        ..Default::default()
    };
    let dataset = load_training_dataset(&root, &tum_config).unwrap();

    let mut config = TrainingConfig::default();
    config.iterations = 1;
    config.max_initial_gaussians = 2_000;

    let scene = rustgs::train_from_path(&root, &tum_config, &config).unwrap();
    let gaussians: Vec<Gaussian> = scene
        .gaussians()
        .iter()
        .map(|g| {
            Gaussian::new(
                g.position.into(),
                g.scale.into(),
                [g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z],
                g.opacity,
                g.color,
            )
        })
        .collect();
    let metadata = SceneMetadata {
        iterations: config.iterations,
        final_loss: 0.0,
        gaussian_count: gaussians.len(),
        sh_degree: config.litegs.sh_degree,
    };
    let device = evaluation_device(EvaluationDevice::Cpu).unwrap();
    let evaluation = evaluate_scene(
        &dataset,
        &gaussians,
        &metadata,
        &SceneEvaluationConfig {
            render_scale: 0.25,
            frame_stride: 30,
            max_frames: 90,
            worst_frame_count: 2,
        },
        &device,
        None,
    )
    .unwrap();

    assert_eq!(evaluation.summary.device, EvaluationDevice::Cpu);
    assert!(evaluation.summary.frame_count > 0);
    assert_eq!(evaluation.summary.scene_gaussian_count, gaussians.len());
    assert!(evaluation.summary.psnr_mean_db.is_finite());
    assert!(evaluation.summary.elapsed_seconds >= 0.0);
}
