use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::{Intrinsics, ScenePose, TrainingDataset, TrainingError, SE3};

/// Configuration for loading a TUM RGB-D dataset directly into RustGS.
#[derive(Debug, Clone)]
pub struct TumRgbdConfig {
    /// Maximum number of RGB frames to consider before stride is applied (0 = all).
    pub max_frames: usize,
    /// Keep every Nth frame.
    pub frame_stride: usize,
    /// Scale factor used to convert 16-bit PNG depth values into meters.
    pub depth_scale: f32,
    /// Maximum timestamp delta when pairing RGB and depth frames.
    pub depth_tolerance_seconds: f64,
    /// Maximum timestamp delta when pairing RGB frames and poses.
    pub pose_tolerance_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameSelectionSummary {
    total_rgb_frames: usize,
    considered_rgb_frames: usize,
    frame_stride: usize,
    selected_rgb_frames: usize,
}

impl Default for TumRgbdConfig {
    fn default() -> Self {
        Self {
            max_frames: 0,
            frame_stride: 1,
            depth_scale: 5000.0,
            depth_tolerance_seconds: 0.1,
            pose_tolerance_seconds: 0.1,
        }
    }
}

/// Load a TUM RGB-D dataset directory as a RustGS training dataset.
pub fn load_tum_rgbd_dataset(
    input: &Path,
    config: &TumRgbdConfig,
) -> Result<TrainingDataset, TrainingError> {
    let root = resolve_tum_root(input)?;
    let rgb_file = root.join("rgb.txt");
    let depth_file = root.join("depth.txt");
    let gt_file = root.join("groundtruth.txt");

    if !rgb_file.exists() {
        return Err(TrainingError::InvalidInput(format!(
            "rgb.txt not found in {}",
            root.display(),
        )));
    }
    if !gt_file.exists() {
        return Err(TrainingError::InvalidInput(format!(
            "groundtruth.txt not found in {}",
            root.display(),
        )));
    }

    let rgb_entries = parse_association_file(&rgb_file, &root)?;
    if rgb_entries.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "no RGB frames listed in {}",
            rgb_file.display(),
        )));
    }
    let depth_entries = if depth_file.exists() {
        parse_association_file(&depth_file, &root)?
    } else {
        Vec::new()
    };
    let ground_truth = parse_ground_truth(&gt_file)?;
    if ground_truth.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "no poses listed in {}",
            gt_file.display(),
        )));
    }

    let (width, height) = image::image_dimensions(&rgb_entries[0].1).map_err(|err| {
        TrainingError::InvalidInput(format!(
            "failed to inspect RGB frame {}: {err}",
            rgb_entries[0].1.display(),
        ))
    })?;
    let intrinsics = load_intrinsics(&root, width, height);
    let mut dataset = TrainingDataset::new(intrinsics).with_depth_scale(config.depth_scale);

    let selection =
        summarize_frame_selection(rgb_entries.len(), config.max_frames, config.frame_stride);

    let mut skipped_pose = 0usize;
    let mut matched_depth = 0usize;
    for (frame_idx, (timestamp, image_path)) in rgb_entries
        .into_iter()
        .take(selection.considered_rgb_frames)
        .step_by(selection.frame_stride)
        .enumerate()
    {
        let Some(pose) = find_closest_pose(&ground_truth, timestamp, config.pose_tolerance_seconds)
        else {
            skipped_pose += 1;
            continue;
        };

        let mut scene_pose = ScenePose::new(frame_idx as u64, image_path, pose, timestamp);
        if let Some(depth_path) =
            find_closest_path(&depth_entries, timestamp, config.depth_tolerance_seconds)
        {
            matched_depth += 1;
            scene_pose = scene_pose.with_depth_path(depth_path);
        }
        dataset.add_pose(scene_pose);
    }

    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "no trainable frames found in {} after pose association",
            root.display(),
        )));
    }

    log::info!(
        "Loaded TUM RGB-D dataset {} | rgb_total={} | considered_rgb={} | frame_stride={} | selected_rgb={} | frames={} | depth_pairs={} | skipped_without_pose={} | resolution={}x{}",
        root.display(),
        selection.total_rgb_frames,
        selection.considered_rgb_frames,
        selection.frame_stride,
        selection.selected_rgb_frames,
        dataset.poses.len(),
        matched_depth,
        skipped_pose,
        intrinsics.width,
        intrinsics.height,
    );

    Ok(dataset)
}

pub fn looks_like_tum_dataset(input: &Path) -> bool {
    if is_tum_root(input) {
        return true;
    }
    input.is_dir()
        && std::fs::read_dir(input)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .any(|entry| is_tum_root(&entry.path()))
            })
            .unwrap_or(false)
}

fn summarize_frame_selection(
    total_rgb_frames: usize,
    max_frames: usize,
    frame_stride: usize,
) -> FrameSelectionSummary {
    let considered_rgb_frames = if max_frames > 0 {
        max_frames.min(total_rgb_frames)
    } else {
        total_rgb_frames
    };
    let frame_stride = frame_stride.max(1);
    let selected_rgb_frames = if considered_rgb_frames == 0 {
        0
    } else {
        considered_rgb_frames.div_ceil(frame_stride)
    };

    FrameSelectionSummary {
        total_rgb_frames,
        considered_rgb_frames,
        frame_stride,
        selected_rgb_frames,
    }
}

fn resolve_tum_root(input: &Path) -> Result<PathBuf, TrainingError> {
    if is_tum_root(input) {
        return Ok(input.to_path_buf());
    }
    if !input.is_dir() {
        return Err(TrainingError::InvalidInput(format!(
            "{} is not a TUM RGB-D dataset directory",
            input.display(),
        )));
    }

    let mut matches = std::fs::read_dir(input)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_tum_root(path))
        .collect::<Vec<_>>();
    matches.sort();

    match matches.len() {
        1 => Ok(matches.swap_remove(0)),
        0 => Err(TrainingError::InvalidInput(format!(
            "could not find a TUM RGB-D dataset under {}",
            input.display(),
        ))),
        _ => Err(TrainingError::InvalidInput(format!(
            "multiple TUM RGB-D datasets found under {}; pass the concrete dataset directory instead",
            input.display(),
        ))),
    }
}

fn is_tum_root(path: &Path) -> bool {
    path.is_dir()
        && path.join("rgb.txt").exists()
        && path.join("groundtruth.txt").exists()
        && (path.join("depth.txt").exists() || path.join("depth").is_dir())
}

fn parse_association_file(path: &Path, root: &Path) -> Result<Vec<(f64, PathBuf)>, TrainingError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: expected 2 columns, got {}",
                path.display(),
                line_num + 1,
                parts.len(),
            )));
        }

        let timestamp = parts[0].parse::<f64>().map_err(|err| {
            TrainingError::InvalidInput(format!(
                "{} line {}: invalid timestamp '{}': {}",
                path.display(),
                line_num + 1,
                parts[0],
                err,
            ))
        })?;
        entries.push((timestamp, root.join(parts[1])));
    }

    Ok(entries)
}

fn parse_ground_truth(path: &Path) -> Result<Vec<(f64, SE3)>, TrainingError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut poses = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 8 {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: expected 8 columns, got {}",
                path.display(),
                line_num + 1,
                parts.len(),
            )));
        }

        let timestamp = parts[0].parse::<f64>().map_err(|err| {
            TrainingError::InvalidInput(format!(
                "{} line {}: invalid timestamp '{}': {}",
                path.display(),
                line_num + 1,
                parts[0],
                err,
            ))
        })?;
        let tx = parse_pose_value(path, line_num, "tx", parts[1])?;
        let ty = parse_pose_value(path, line_num, "ty", parts[2])?;
        let tz = parse_pose_value(path, line_num, "tz", parts[3])?;
        let qx = parse_pose_value(path, line_num, "qx", parts[4])?;
        let qy = parse_pose_value(path, line_num, "qy", parts[5])?;
        let qz = parse_pose_value(path, line_num, "qz", parts[6])?;
        let qw = parse_pose_value(path, line_num, "qw", parts[7])?;

        poses.push((timestamp, SE3::new(&[qx, qy, qz, qw], &[tx, ty, tz])));
    }

    poses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(poses)
}

fn parse_pose_value(
    path: &Path,
    line_num: usize,
    label: &str,
    value: &str,
) -> Result<f32, TrainingError> {
    value.parse::<f32>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num + 1,
            label,
            value,
            err,
        ))
    })
}

fn find_closest_path(
    entries: &[(f64, PathBuf)],
    timestamp: f64,
    tolerance_seconds: f64,
) -> Option<PathBuf> {
    entries
        .iter()
        .filter_map(|(candidate_ts, path)| {
            let diff = (candidate_ts - timestamp).abs();
            (diff <= tolerance_seconds).then_some((diff, path))
        })
        .min_by(|(lhs, _), (rhs, _)| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, path)| path.clone())
}

fn find_closest_pose(poses: &[(f64, SE3)], timestamp: f64, tolerance_seconds: f64) -> Option<SE3> {
    if poses.is_empty() {
        return None;
    }

    match poses.binary_search_by(|(candidate_ts, _)| {
        candidate_ts
            .partial_cmp(&timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(index) => Some(poses[index].1),
        Err(index) => {
            let mut best: Option<(f64, SE3)> = None;
            for candidate in [index.checked_sub(1), Some(index)].into_iter().flatten() {
                if let Some((candidate_ts, pose)) = poses.get(candidate) {
                    let diff = (candidate_ts - timestamp).abs();
                    if diff <= tolerance_seconds
                        && best.map(|(best_diff, _)| diff < best_diff).unwrap_or(true)
                    {
                        best = Some((diff, *pose));
                    }
                }
            }
            best.map(|(_, pose)| pose)
        }
    }
}

fn load_intrinsics(root: &Path, width: u32, height: u32) -> Intrinsics {
    let calib_file = root.join("calibration.txt");
    if let Ok(content) = std::fs::read_to_string(&calib_file) {
        let values = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| line.parse::<f32>().ok())
            .collect::<Vec<_>>();
        if values.len() >= 4 {
            return Intrinsics::new(values[0], values[1], values[2], values[3], width, height);
        }
    }

    let sequence = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (fx, fy, cx, cy) = if sequence.contains("freiburg1") {
        (517.3, 516.5, 318.6, 255.3)
    } else if sequence.contains("freiburg2") {
        (520.9, 521.0, 325.1, 249.7)
    } else if sequence.contains("freiburg3") {
        (535.4, 539.2, 320.1, 247.6)
    } else {
        (525.0, 525.0, 319.5, 239.5)
    };

    Intrinsics::new(fx, fy, cx, cy, width, height)
}

#[cfg(test)]
mod tests {
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
}
