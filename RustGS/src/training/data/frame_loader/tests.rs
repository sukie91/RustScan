use super::{
    decode_frame, load_color_image, load_depth_image, synthetic_depth,
    DeterministicFrameBatchIter, FrameLoaderOptions, PrefetchFrameLoader, TrainingConfig,
};
use crate::{Intrinsics, ScenePose, TrainingDataset, SE3};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn make_dataset(root: &Path, frame_count: usize) -> TrainingDataset {
    let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 2, 1));
    dataset.depth_scale = 1.0;
    for idx in 0..frame_count {
        let image_path = root.join(format!("frame_{idx:03}.rgb"));
        let depth_path = root.join(format!("frame_{idx:03}.depth"));
        fs::write(
            &image_path,
            [idx as u8, idx as u8 + 1, idx as u8 + 2, 9, 8, 7],
        )
        .unwrap();
        let mut depth_bytes = Vec::new();
        depth_bytes.extend_from_slice(&(idx as f32 + 1.0).to_le_bytes());
        depth_bytes.extend_from_slice(&(idx as f32 + 2.0).to_le_bytes());
        fs::write(&depth_path, depth_bytes).unwrap();
        let pose = ScenePose::new(idx as u64, image_path, SE3::identity(), idx as f64)
            .with_depth_path(depth_path);
        dataset.add_pose(pose);
    }
    dataset
}

#[test]
fn test_load_raw_rgb() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame.rgb");
    fs::write(&path, [1u8, 2, 3, 4, 5, 6]).unwrap();

    let loaded = load_color_image(&path, 2, 1).unwrap();
    assert_eq!(loaded, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_load_raw_depth() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame.depth");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1.25f32.to_le_bytes());
    bytes.extend_from_slice(&2.5f32.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let loaded = load_depth_image(&path, 2, 1, 1.0).unwrap();
    assert_eq!(loaded, vec![1.25, 2.5]);
}

#[test]
fn test_synthetic_depth_range() {
    let color = vec![0u8, 0, 0, 255, 255, 255];
    let depth = synthetic_depth(&color, 2, 1, 0.1, 2.0);
    assert_eq!(depth.len(), 2);
    assert!(depth[0] >= 0.1 && depth[0] <= 2.0);
    assert!(depth[1] >= 0.1 && depth[1] <= 2.0);
    assert!(depth[0] > depth[1]);
}

#[test]
fn decode_frame_uses_synthetic_depth_fallback() {
    let dir = tempdir().unwrap();
    let image_path = dir.path().join("frame.rgb");
    fs::write(&image_path, [0u8, 0, 0, 255, 255, 255]).unwrap();

    let frame = decode_frame(&image_path, None, 2, 1, None, 1.0, true, 0.1, 2.0).unwrap();

    assert_eq!(frame.color_u8.len(), 6);
    assert!(frame.target_rgb.is_none());
    assert_eq!(frame.depth.len(), 2);
    assert!(!frame.used_real_depth);
    assert!(frame.depth[0] > frame.depth[1]);
}

#[test]
fn deterministic_batch_iterator_repeats_for_same_seed() {
    let order_a = DeterministicFrameBatchIter::new(6, 2, 7)
        .flat_map(|batch| batch.into_iter())
        .collect::<Vec<_>>();
    let order_b = DeterministicFrameBatchIter::new(6, 2, 7)
        .flat_map(|batch| batch.into_iter())
        .collect::<Vec<_>>();

    assert_eq!(order_a, order_b);
    let mut sorted = order_a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn deterministic_batch_iterator_changes_order_with_seed() {
    let order_a = DeterministicFrameBatchIter::new(6, 2, 7)
        .flat_map(|batch| batch.into_iter())
        .collect::<Vec<_>>();
    let order_b = DeterministicFrameBatchIter::new(6, 2, 11)
        .flat_map(|batch| batch.into_iter())
        .collect::<Vec<_>>();

    assert_ne!(order_a, order_b);
}

#[test]
fn ordered_frame_indices_matches_batch_iterator_shuffle() {
    let iter_order = DeterministicFrameBatchIter::new(8, 3, 19)
        .flat_map(|batch| batch.into_iter())
        .collect::<Vec<_>>();
    let helper_order = super::ordered_frame_indices(8, 3, 19);

    assert_eq!(helper_order, iter_order);
}

#[test]
fn prefetch_cache_stays_within_capacity_and_evicts_old_frames() {
    let dir = tempdir().unwrap();
    let dataset = make_dataset(dir.path(), 4);
    let config = TrainingConfig {
        initialization: crate::training::TrainingInitializationConfig {
            use_synthetic_depth: false,
            ..crate::training::TrainingInitializationConfig::default()
        },
        ..TrainingConfig::default()
    };
    let mut loader = PrefetchFrameLoader::new(
        &dataset,
        &config,
        FrameLoaderOptions {
            cache_capacity: 2,
            prefetch_ahead: 3,
            rgb_target_size: None,
        },
    )
    .unwrap();
    let order = DeterministicFrameBatchIter::new(dataset.poses.len(), 1, 3)
        .order()
        .to_vec();

    loader.prefetch_order_window(&order, 0).unwrap();
    let first = order[0];
    let second = order[1];
    let third = order[2];

    let frame0 = loader.get(first).unwrap();
    assert_eq!(frame0.depth[0], first as f32 + 1.0);
    loader.prefetch_order_window(&order, 1).unwrap();
    loader.get(second).unwrap();
    loader.get(third).unwrap();

    assert!(loader.cache_len() <= 2);
    assert!(!loader.is_cached(first));
    assert!(loader.is_cached(second) || loader.is_cached(third));
}

#[test]
fn worker_count_respects_prefetch_ahead_and_frame_count() {
    assert_eq!(
        super::frame_loader_worker_count(
            1,
            FrameLoaderOptions {
                cache_capacity: 8,
                prefetch_ahead: 8,
                rgb_target_size: None,
            }
        ),
        1
    );
    assert_eq!(
        super::frame_loader_worker_count(
            3,
            FrameLoaderOptions {
                cache_capacity: 8,
                prefetch_ahead: 2,
                rgb_target_size: None,
            }
        ),
        2
    );
}

#[test]
fn decode_frame_prepares_target_rgb_when_requested() {
    let dir = tempdir().unwrap();
    let image_path = dir.path().join("frame.rgb");
    fs::write(&image_path, [0u8, 64, 255, 255, 128, 0]).unwrap();

    let frame =
        decode_frame(&image_path, None, 2, 1, Some((2, 1)), 1.0, false, 0.1, 2.0).unwrap();

    let target_rgb = frame.target_rgb.expect("prepared target rgb");
    assert_eq!(target_rgb.len(), 6);
    assert!((target_rgb[1] - (64.0 / 255.0)).abs() < 1e-6);
}
