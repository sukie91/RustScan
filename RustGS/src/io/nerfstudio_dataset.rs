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

/// Configuration for loading a Nerfstudio-style dataset.
#[derive(Debug, Clone)]
pub struct NerfstudioConfig {
    /// Maximum number of frames to load (0 = all).
    pub max_frames: usize,
    /// Keep every Nth frame.
    pub frame_stride: usize,
}

impl Default for NerfstudioConfig {
    fn default() -> Self {
        Self {
            max_frames: 0,
            frame_stride: 1,
        }
    }
}

pub fn looks_like_nerfstudio_dataset(path: &Path) -> bool {
    discover_transforms_path(path).is_some()
}

pub fn load_nerfstudio_dataset(
    root: &Path,
    config: &NerfstudioConfig,
) -> Result<TrainingDataset, TrainingError> {
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
    let frame_stride = config.frame_stride.max(1);
    let considered_frames = if config.max_frames > 0 {
        config.max_frames.min(scene.frames.len())
    } else {
        scene.frames.len()
    };
    for (frame_idx, frame) in scene
        .frames
        .iter()
        .take(considered_frames)
        .step_by(frame_stride)
        .enumerate()
    {
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
            if let Some(points) = load_nerfstudio_init_points(&ply_path) {
                dataset.initial_points = points;
            }
        }
    }

    if dataset.initial_points.is_empty() {
        for candidate in discover_point_cloud_candidates(transforms_dir) {
            if let Some(points) = load_nerfstudio_init_points(&candidate) {
                dataset.initial_points = points;
                break;
            }
        }
    }

    log::info!(
        "Loaded Nerfstudio dataset {} | transforms={} | frames_total={} | considered_frames={} | frame_stride={} | frames={} | init_points={} | resolution={}x{}",
        root.display(),
        transforms_path.display(),
        scene.frames.len(),
        considered_frames,
        frame_stride,
        dataset.poses.len(),
        dataset.initial_points.len(),
        intrinsics.width,
        intrinsics.height,
    );

    Ok(dataset)
}

fn load_nerfstudio_init_points(path: &Path) -> Option<Vec<([f32; 3], Option<[f32; 3]>)>> {
    match load_ascii_point_cloud_ply(path) {
        Ok(points) if !points.is_empty() => return Some(points),
        Ok(_) => {
            log::warn!("Ignoring empty Nerfstudio point cloud {}", path.display());
            return None;
        }
        Err(err) => {
            log::debug!(
                "Could not parse {} as an ASCII xyz/rgb point cloud: {}",
                path.display(),
                err
            );
        }
    }

    #[cfg(feature = "gpu")]
    match crate::load_splats_ply(path) {
        Ok((splats, _)) => {
            let view = splats.as_view();
            let sh_row_width = sh_coeff_count_for_degree(view.sh_degree) * 3;
            return Some(
                (0..splats.len())
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
                    .collect(),
            );
        }
        Err(err) => {
            log::warn!(
                "Ignoring Nerfstudio init splat {} because it could not be parsed as a RustGS splat payload: {}",
                path.display(),
                err
            );
        }
    }

    None
}

fn discover_point_cloud_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "sparse_pc.ply",
        "point_cloud.ply",
        "points3D.ply",
        "points.ply",
        "colmap/sparse_pc.ply",
    ]
    .into_iter()
    .map(|relative| root.join(relative))
    .filter(|path| path.exists())
    .collect()
}

fn load_ascii_point_cloud_ply(
    path: &Path,
) -> Result<Vec<([f32; 3], Option<[f32; 3]>)>, TrainingError> {
    let text = fs::read_to_string(path)?;
    let mut vertex_count = None;
    let mut vertex_properties = Vec::new();
    let mut in_vertex_element = false;
    let mut body_start_line = None;

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if line_idx == 0 && trimmed != "ply" {
            return Err(TrainingError::InvalidInput(format!(
                "{} is not a PLY file",
                path.display()
            )));
        }
        if trimmed == "format binary_little_endian 1.0" || trimmed == "format binary_big_endian 1.0"
        {
            return Err(TrainingError::InvalidInput(format!(
                "{} is not an ASCII PLY point cloud",
                path.display()
            )));
        }
        if let Some(rest) = trimmed.strip_prefix("element ") {
            let parts = rest.split_whitespace().collect::<Vec<_>>();
            in_vertex_element = parts.first() == Some(&"vertex");
            if in_vertex_element && parts.len() >= 2 {
                vertex_count = parts[1].parse::<usize>().ok();
            }
            continue;
        }
        if trimmed.starts_with("property ") && in_vertex_element {
            if let Some(name) = trimmed.split_whitespace().last() {
                vertex_properties.push(name.to_string());
            }
            continue;
        }
        if trimmed == "end_header" {
            body_start_line = Some(line_idx + 1);
            break;
        }
    }

    let body_start_line = body_start_line.ok_or_else(|| {
        TrainingError::InvalidInput(format!("{} is missing end_header", path.display()))
    })?;
    let vertex_count = vertex_count.ok_or_else(|| {
        TrainingError::InvalidInput(format!("{} is missing vertex count", path.display()))
    })?;
    let x_idx = property_index(&vertex_properties, &["x"])?;
    let y_idx = property_index(&vertex_properties, &["y"])?;
    let z_idx = property_index(&vertex_properties, &["z"])?;
    let r_idx = property_index(&vertex_properties, &["red", "r"]).ok();
    let g_idx = property_index(&vertex_properties, &["green", "g"]).ok();
    let b_idx = property_index(&vertex_properties, &["blue", "b"]).ok();

    let mut points = Vec::with_capacity(vertex_count);
    for line in text.lines().skip(body_start_line).take(vertex_count) {
        let values = line.split_whitespace().collect::<Vec<_>>();
        if values.len() < vertex_properties.len() {
            return Err(TrainingError::InvalidInput(format!(
                "{} has a vertex row with {} values, expected {}",
                path.display(),
                values.len(),
                vertex_properties.len()
            )));
        }

        let position = [
            parse_f32(values[x_idx], path)?,
            parse_f32(values[y_idx], path)?,
            parse_f32(values[z_idx], path)?,
        ];
        let color = match (r_idx, g_idx, b_idx) {
            (Some(r_idx), Some(g_idx), Some(b_idx)) => Some([
                normalize_color(parse_f32(values[r_idx], path)?),
                normalize_color(parse_f32(values[g_idx], path)?),
                normalize_color(parse_f32(values[b_idx], path)?),
            ]),
            _ => None,
        };
        points.push((position, color));
    }

    if points.len() != vertex_count {
        return Err(TrainingError::InvalidInput(format!(
            "{} vertex count mismatch: header {}, parsed {}",
            path.display(),
            vertex_count,
            points.len()
        )));
    }

    Ok(points)
}

fn property_index(properties: &[String], names: &[&str]) -> Result<usize, TrainingError> {
    properties
        .iter()
        .position(|property| names.iter().any(|name| property == name))
        .ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "missing required PLY property {}",
                names.join("/")
            ))
        })
}

fn parse_f32(value: &str, path: &Path) -> Result<f32, TrainingError> {
    value.parse::<f32>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "failed to parse numeric value in {}: {}",
            path.display(),
            err
        ))
    })
}

fn normalize_color(value: f32) -> f32 {
    if value > 1.0 {
        (value / 255.0).clamp(0.0, 1.0)
    } else {
        value.clamp(0.0, 1.0)
    }
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
mod tests;
