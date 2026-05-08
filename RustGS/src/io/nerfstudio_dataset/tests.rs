use super::{load_nerfstudio_dataset, looks_like_nerfstudio_dataset, NerfstudioConfig};
use std::fs;
use tempfile::tempdir;

#[test]
fn loads_nerfstudio_transforms_json() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("frame.png"), []).unwrap();
    fs::write(
        dir.path().join("transforms.json"),
        r#"{
            "fl_x": 500.0,
            "fl_y": 505.0,
            "cx": 320.0,
            "cy": 240.0,
            "w": 640,
            "h": 480,
            "frames": [
                {
                    "file_path": "frame.png",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            ]
        }"#,
    )
    .unwrap();

    assert!(looks_like_nerfstudio_dataset(dir.path()));
    let dataset = load_nerfstudio_dataset(dir.path(), &NerfstudioConfig::default()).unwrap();
    assert_eq!(dataset.poses.len(), 1);
    assert_eq!(dataset.intrinsics.width, 640);
    assert_eq!(dataset.poses[0].pose.translation(), [1.0, 2.0, 3.0]);
}

#[test]
fn loads_ascii_sparse_point_cloud_for_initialization() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("frame0.png"), []).unwrap();
    fs::write(dir.path().join("frame1.png"), []).unwrap();
    fs::write(
        dir.path().join("sparse_pc.ply"),
        "ply\n\
format ascii 1.0\n\
element vertex 2\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
end_header\n\
0.0 0.0 1.0 255 128 0\n\
1.0 0.0 2.0 0 64 255\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("transforms.json"),
        r#"{
            "fl_x": 500.0,
            "w": 640,
            "h": 480,
            "frames": [
                {
                    "file_path": "frame0.png",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                },
                {
                    "file_path": "frame1.png",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            ]
        }"#,
    )
    .unwrap();

    let dataset = load_nerfstudio_dataset(
        dir.path(),
        &NerfstudioConfig {
            max_frames: 1,
            frame_stride: 1,
        },
    )
    .unwrap();

    assert_eq!(dataset.poses.len(), 1);
    assert_eq!(dataset.initial_points.len(), 2);
    assert_eq!(dataset.initial_points[0].0, [0.0, 0.0, 1.0]);
    assert_eq!(dataset.initial_points[0].1, Some([1.0, 128.0 / 255.0, 0.0]));
    assert_eq!(dataset.initial_points[1].1, Some([0.0, 64.0 / 255.0, 1.0]));
}
