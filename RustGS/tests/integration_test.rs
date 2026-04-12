#![cfg(feature = "gpu")]

use rustgs::{train_splats_with_report, Intrinsics, ScenePose, TrainingConfig, TrainingDataset, SE3};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tempfile::tempdir;

fn write_raw_rgb(path: &Path, width: usize, height: usize) {
    let mut bytes = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            bytes.push(((x * 4) % 256) as u8);
            bytes.push(((y * 4) % 256) as u8);
            bytes.push(160);
        }
    }
    fs::write(path, bytes).expect("write raw rgb frame");
}

fn tiny_dataset(root: &Path) -> TrainingDataset {
    let width = 64;
    let height = 64;
    let image_path = root.join("frame_000.rgb");
    write_raw_rgb(&image_path, width as usize, height as usize);

    let mut dataset = TrainingDataset::new(Intrinsics::new(64.0, 64.0, 32.0, 32.0, width, height));
    dataset.add_pose(ScenePose::new(0, image_path, SE3::identity(), 0.0));

    for idx in 0..10 {
        let x = (idx % 5) as f32 * 0.05 - 0.1;
        let y = (idx / 5) as f32 * 0.05 - 0.05;
        let z = 2.0 + idx as f32 * 0.01;
        dataset.add_point([x, y, z], Some([0.8, 0.7, 0.6]));
    }

    dataset
}

fn tiny_config() -> TrainingConfig {
    TrainingConfig {
        iterations: 100,
        max_initial_gaussians: 10,
        use_synthetic_depth: true,
        metal_render_scale: 1.0,
        densify_interval: 0,
        prune_interval: 0,
        frame_shuffle_seed: 0,
        ..TrainingConfig::default()
    }
}

fn run_tiny_training(root: &Path) -> rustgs::TrainingRun {
    let dataset = tiny_dataset(root);
    train_splats_with_report(&dataset, &tiny_config()).expect("tiny wgpu training run")
}

#[test]
#[ignore]
fn test_wgpu_training_tiny_dataset() {
    let dir = tempdir().expect("tempdir");
    let run = run_tiny_training(dir.path());

    assert!(!run.splats.is_empty());
    assert!((10..=1000).contains(&run.splats.len()));
}

#[test]
#[ignore]
fn test_wgpu_training_with_report() {
    let dir = tempdir().expect("tempdir");
    let run = run_tiny_training(dir.path());

    assert!(run.report.final_loss.is_some());
    assert!(run.report.elapsed < Duration::from_secs(300));
    assert!((10..=1000).contains(&run.report.gaussian_count));
}
