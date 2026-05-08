use super::*;
use image::{ImageBuffer, Luma, Rgb, RgbImage};
use tempfile::tempdir;

fn write_test_dataset(root: &Path) {
    write_test_dataset_with_frame_count(root, 2);
}

fn write_test_dataset_with_frame_count(root: &Path, frame_count: usize) {
    std::fs::create_dir_all(root.join("rgb")).unwrap();
    std::fs::create_dir_all(root.join("depth")).unwrap();

    let rgb = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
    let depth: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_pixel(2, 2, Luma([5000]));
    let mut rgb_lines = String::from("# timestamp filename\n");
    let mut depth_lines = String::from("# timestamp filename\n");
    let mut groundtruth_lines = String::from("# timestamp tx ty tz qx qy qz qw\n");

    for idx in 1..=frame_count {
        let timestamp = format!("{idx}.0");
        let rgb_name = format!("rgb/{timestamp}.png");
        let depth_name = format!("depth/{timestamp}.png");
        rgb.save(root.join(&rgb_name)).unwrap();
        depth.save(root.join(&depth_name)).unwrap();
        rgb_lines.push_str(&format!("{timestamp} {rgb_name}\n"));
        depth_lines.push_str(&format!("{timestamp} {depth_name}\n"));
        groundtruth_lines.push_str(&format!(
            "{timestamp} {} {} {} 0 0 0 1\n",
            idx - 1,
            idx,
            idx + 1
        ));
    }

    std::fs::write(root.join("rgb.txt"), rgb_lines).unwrap();
    std::fs::write(root.join("depth.txt"), depth_lines).unwrap();
    std::fs::write(root.join("groundtruth.txt"), groundtruth_lines).unwrap();
}

#[test]
fn test_load_tum_rgbd_dataset_from_dataset_root() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("rgbd_dataset_freiburg1_xyz");
    write_test_dataset(&root);

    let dataset = load_tum_rgbd_dataset(&root, &TumRgbdConfig::default()).unwrap();
    assert_eq!(dataset.poses.len(), 2);
    assert!(dataset.poses[0].depth_path.is_some());
    assert_eq!(dataset.intrinsics.width, 2);
    assert_eq!(dataset.intrinsics.height, 2);
    assert!((dataset.intrinsics.fx - 517.3).abs() < 1e-3);
    assert_eq!(dataset.depth_scale, 5000.0);
}

#[test]
fn test_load_tum_rgbd_dataset_from_parent_directory() {
    let temp = tempdir().unwrap();
    let tum_root = temp.path().join("tum");
    let dataset_root = tum_root.join("rgbd_dataset_freiburg1_xyz");
    write_test_dataset(&dataset_root);

    let dataset = load_tum_rgbd_dataset(&tum_root, &TumRgbdConfig::default()).unwrap();
    assert_eq!(dataset.poses.len(), 2);
    assert_eq!(dataset.depth_scale, 5000.0);
}

#[test]
fn test_frame_selection_summary_applies_max_frames_before_stride() {
    let summary = summarize_frame_selection(30, 25, 5);
    assert_eq!(
        summary,
        FrameSelectionSummary {
            total_rgb_frames: 30,
            considered_rgb_frames: 25,
            frame_stride: 5,
            selected_rgb_frames: 5,
        }
    );
}

#[test]
fn test_load_tum_rgbd_dataset_uses_stride_within_considered_prefix() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("rgbd_dataset_freiburg1_xyz");
    write_test_dataset_with_frame_count(&root, 30);

    let dataset = load_tum_rgbd_dataset(
        &root,
        &TumRgbdConfig {
            max_frames: 25,
            frame_stride: 5,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(dataset.poses.len(), 5);
    let timestamps = dataset
        .poses
        .iter()
        .map(|pose| pose.timestamp)
        .collect::<Vec<_>>();
    assert_eq!(timestamps, vec![1.0, 6.0, 11.0, 16.0, 21.0]);
}
