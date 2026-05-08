use super::*;
use tempfile::tempdir;

fn write_colmap_test_dataset(root: &Path) {
    // Create sparse directory
    let sparse = root.join("sparse").join("0");
    std::fs::create_dir_all(&sparse).unwrap();

    // Create images directory
    let images = root.join("images");
    std::fs::create_dir_all(&images).unwrap();

    // Write cameras.txt
    std::fs::write(
        sparse.join("cameras.txt"),
        "# Camera list with one line of data per camera:\n1 PINHOLE 1920 1080 1500 1500 960 540\n",
    )
    .unwrap();

    // Write images.txt
    std::fs::write(
        sparse.join("images.txt"),
        "# Image list with two lines of data per image:\n1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 frame_0001.jpg\n2 1.0 0.0 0.0 0.0 1.0 0.0 2.0 1 frame_0002.jpg\n",
    )
    .unwrap();

    // Write points3D.txt
    std::fs::write(
        sparse.join("points3D.txt"),
        "# 3D point list with one line of data per point:\n1 0.0 0.0 1.0 128 128 128 0.1\n2 1.0 0.0 1.0 64 64 64 0.1\n",
    )
    .unwrap();

    // Create placeholder images
    let img_data: Vec<u8> = vec![0u8; 1920 * 1080 * 3];
    for name in ["frame_0001.jpg", "frame_0002.jpg"] {
        std::fs::write(images.join(name), &img_data).unwrap();
    }
}

#[test]
fn test_load_colmap_dataset_text_format() {
    let temp = tempdir().unwrap();
    write_colmap_test_dataset(temp.path());

    let dataset = load_colmap_dataset(temp.path(), &ColmapConfig::default()).unwrap();
    assert_eq!(dataset.poses.len(), 2);
    assert_eq!(dataset.initial_points.len(), 2);
    assert_eq!(dataset.intrinsics.width, 1920);
    assert_eq!(dataset.intrinsics.height, 1080);
    assert!((dataset.intrinsics.fx - 1500.0).abs() < 1e-3);
    assert_eq!(dataset.poses[0].pose.translation(), [0.0, 0.0, -1.0]);
    assert_eq!(dataset.poses[1].pose.translation(), [-1.0, 0.0, -2.0]);
}

#[test]
fn test_load_colmap_dataset_requires_sparse_points() {
    let temp = tempdir().unwrap();
    write_colmap_test_dataset(temp.path());
    std::fs::remove_file(temp.path().join("sparse").join("0").join("points3D.txt")).unwrap();

    let err = load_colmap_dataset(temp.path(), &ColmapConfig::default()).unwrap_err();
    assert!(
        err.to_string().contains("missing COLMAP sparse points"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_load_colmap_with_stride() {
    let temp = tempdir().unwrap();
    write_colmap_test_dataset(temp.path());

    let dataset = load_colmap_dataset(
        temp.path(),
        &ColmapConfig {
            max_frames: 0,
            frame_stride: 2,
            depth_scale: 1.0,
        },
    )
    .unwrap();
    assert_eq!(dataset.poses.len(), 1); // Only first frame with stride 2
}

#[test]
fn test_camera_model_intrinsics() {
    let pinhole = CameraModel::Pinhole;
    let intrinsics = pinhole.intrinsics(1920, 1080, &[1500.0, 1500.0, 960.0, 540.0]);
    assert!(intrinsics.is_some());
    let intrinsics = intrinsics.unwrap();
    assert!((intrinsics.fx - 1500.0).abs() < 1e-3);
}
