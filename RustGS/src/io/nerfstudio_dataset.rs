use crate::{Intrinsics, ScenePose, TrainingDataset, TrainingError, SE3};
use glam::{Mat3, Quat, Vec3};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "gpu")]
use crate::sh::{sh0_to_rgb_value, sh_coeff_count_for_degree};

#[derive(Debug, Deserialize)]
struct NerfstudioScene {
    #[serde(default)]
    fl_x: Option<f32>,
    #[serde(default)]
    fl_y: Option<f32>,
    #[serde(default)]
    cx: Option<f32>,
    #[serde(default)]
    cy: Option<f32>,
    #[serde(default)]
    w: Option<u32>,
    #[serde(default)]
    h: Option<u32>,
    #[serde(default)]
    ply_file_path: Option<String>,
    frames: Vec<NerfstudioFrame>,
}

#[derive(Debug, Deserialize)]
struct NerfstudioFrame {
    file_path: String,
    transform_matrix: [[f32; 4]; 4],
    #[serde(default)]
    fl_x: Option<f32>,
    #[serde(default)]
    fl_y: Option<f32>,
    #[serde(default)]
    cx: Option<f32>,
    #[serde(default)]
    cy: Option<f32>,
    #[serde(default)]
    w: Option<u32>,
    #[serde(default)]
    h: Option<u32>,
}

pub fn looks_like_nerfstudio_dataset(path: &Path) -> bool {
    discover_transforms_path(path).is_some()
}

pub fn load_nerfstudio_dataset(root: &Path) -> Result<TrainingDataset, TrainingError> {
    let transforms_path = discover_transforms_path(root).ok_or_else(|| {
        TrainingError::InvalidInput(format!(
            "{} does not look like a Nerfstudio dataset",
            root.display()
        ))
    })?;
    let scene: NerfstudioScene = serde_json::from_str(&fs::read_to_string(&transforms_path)?)
        .map_err(|err| {
            TrainingError::InvalidInput(format!(
                "failed to parse Nerfstudio transforms {}: {}",
                transforms_path.display(),
                err
            ))
        })?;
    if scene.frames.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "Nerfstudio transforms file {} contains no frames",
            transforms_path.display()
        )));
    }

    let first_frame = &scene.frames[0];
    let width = first_frame.w.or(scene.w).ok_or_else(|| {
        TrainingError::InvalidInput("Nerfstudio dataset is missing image width".to_string())
    })?;
    let height = first_frame.h.or(scene.h).ok_or_else(|| {
        TrainingError::InvalidInput("Nerfstudio dataset is missing image height".to_string())
    })?;
    let fx = first_frame.fl_x.or(scene.fl_x).ok_or_else(|| {
        TrainingError::InvalidInput("Nerfstudio dataset is missing fl_x".to_string())
    })?;
    let fy = first_frame.fl_y.or(scene.fl_y).unwrap_or(fx);
    let cx = first_frame.cx.or(scene.cx).unwrap_or(width as f32 * 0.5);
    let cy = first_frame.cy.or(scene.cy).unwrap_or(height as f32 * 0.5);
    let intrinsics = Intrinsics::new(fx, fy, cx, cy, width, height);
    let mut dataset = TrainingDataset::new(intrinsics);

    let transforms_dir = transforms_path.parent().unwrap_or(root);
    for (frame_idx, frame) in scene.frames.iter().enumerate() {
        let pose = pose_from_transform_matrix(&frame.transform_matrix);
        let image_path = resolve_frame_path(transforms_dir, &frame.file_path);
        dataset.add_pose(ScenePose::new(
            frame_idx as u64,
            image_path,
            pose,
            frame_idx as f64,
        ));
    }

    if let Some(ply_file_path) = scene.ply_file_path.as_ref() {
        let ply_path = transforms_dir.join(ply_file_path);
        if ply_path.exists() {
            #[cfg(feature = "gpu")]
            match crate::load_splats_ply(&ply_path) {
                Ok((splats, _)) => {
                    let view = splats.as_view();
                    let sh_row_width = sh_coeff_count_for_degree(view.sh_degree) * 3;
                    dataset.initial_points = (0..splats.len())
                        .map(|idx| {
                            let pos_base = idx * 3;
                            let sh_base = idx * sh_row_width;
                            (
                                [
                                    view.positions[pos_base],
                                    view.positions[pos_base + 1],
                                    view.positions[pos_base + 2],
                                ],
                                Some([
                                    sh0_to_rgb_value(view.sh_coeffs[sh_base]),
                                    sh0_to_rgb_value(view.sh_coeffs[sh_base + 1]),
                                    sh0_to_rgb_value(view.sh_coeffs[sh_base + 2]),
                                ]),
                            )
                        })
                        .collect();
                }
                Err(err) => {
                    log::warn!(
                        "Ignoring Nerfstudio init splat {} because it could not be parsed as a RustGS splat payload: {}",
                        ply_path.display(),
                        err
                    );
                }
            }

            #[cfg(not(feature = "gpu"))]
            log::warn!(
                "Ignoring Nerfstudio init splat {} because RustGS without GPU support cannot load host splat payloads",
                ply_path.display(),
            );
        }
    }

    Ok(dataset)
}

fn discover_transforms_path(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("transforms.json"),
        root.join("transforms_train.json"),
        root.join("transforms_val.json"),
        root.join("transforms_test.json"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    find_transforms_recursive(root)
}

fn find_transforms_recursive(root: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|name| {
                    name == "transforms.json"
                        || name == "transforms_train.json"
                        || name == "transforms_val.json"
                        || name == "transforms_test.json"
                })
                .unwrap_or(false)
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_transforms_recursive(&path) {
                return Some(found);
            }
        }
    }
    None
}

fn resolve_frame_path(root: &Path, file_path: &str) -> PathBuf {
    let path = Path::new(file_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn pose_from_transform_matrix(matrix: &[[f32; 4]; 4]) -> SE3 {
    let rotation = Mat3::from_cols(
        Vec3::new(matrix[0][0], matrix[1][0], matrix[2][0]),
        Vec3::new(matrix[0][1], matrix[1][1], matrix[2][1]),
        Vec3::new(matrix[0][2], matrix[1][2], matrix[2][2]),
    );
    let translation = Vec3::new(matrix[0][3], matrix[1][3], matrix[2][3]);
    SE3::from_quat_translation(Quat::from_mat3(&rotation), translation)
}

#[cfg(test)]
mod tests {
    use super::{load_nerfstudio_dataset, looks_like_nerfstudio_dataset};
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
        let dataset = load_nerfstudio_dataset(dir.path()).unwrap();
        assert_eq!(dataset.poses.len(), 1);
        assert_eq!(dataset.intrinsics.width, 640);
        assert_eq!(dataset.poses[0].pose.translation(), [1.0, 2.0, 3.0]);
    }
}
