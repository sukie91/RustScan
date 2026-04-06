use crate::{Intrinsics, ScenePose, TrainingDataset, TrainingError, SE3};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct ColmapCamera {
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

#[derive(Debug, Clone)]
struct ColmapImage {
    camera_id: u32,
    name: String,
    pose: SE3,
}

pub fn looks_like_colmap_dataset(path: &Path) -> bool {
    discover_colmap_model_dir(path).is_some()
}

pub fn load_colmap_dataset(root: &Path) -> Result<TrainingDataset, TrainingError> {
    let model_dir = discover_colmap_model_dir(root).ok_or_else(|| {
        TrainingError::InvalidInput(format!(
            "{} does not look like a COLMAP reconstruction",
            root.display()
        ))
    })?;

    let cameras = if model_dir.join("cameras.bin").exists() {
        read_cameras_bin(&model_dir.join("cameras.bin"))?
    } else {
        read_cameras_txt(&model_dir.join("cameras.txt"))?
    };
    let images = if model_dir.join("images.bin").exists() {
        read_images_bin(&model_dir.join("images.bin"))?
    } else {
        read_images_txt(&model_dir.join("images.txt"))?
    };
    if images.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "COLMAP model {} does not contain any images",
            model_dir.display()
        )));
    }

    let first_camera = cameras.get(&images[0].camera_id).ok_or_else(|| {
        TrainingError::InvalidInput(format!(
            "COLMAP image {} references missing camera {}",
            images[0].name, images[0].camera_id
        ))
    })?;
    let intrinsics = Intrinsics::new(
        first_camera.fx,
        first_camera.fy,
        first_camera.cx,
        first_camera.cy,
        first_camera.width,
        first_camera.height,
    );
    let mut dataset = TrainingDataset::new(intrinsics);

    for (frame_idx, image) in images.iter().enumerate() {
        let camera = cameras.get(&image.camera_id).ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "COLMAP image {} references missing camera {}",
                image.name, image.camera_id
            ))
        })?;
        if camera.width != dataset.intrinsics.width
            || camera.height != dataset.intrinsics.height
            || (camera.fx - dataset.intrinsics.fx).abs() > 1e-5
            || (camera.fy - dataset.intrinsics.fy).abs() > 1e-5
            || (camera.cx - dataset.intrinsics.cx).abs() > 1e-5
            || (camera.cy - dataset.intrinsics.cy).abs() > 1e-5
        {
            return Err(TrainingError::InvalidInput(
                "COLMAP loader currently expects a single shared camera intrinsics set".to_string(),
            ));
        }

        let image_path = resolve_colmap_image_path(root, &model_dir, &image.name)?;
        dataset.add_pose(ScenePose::new(
            frame_idx as u64,
            image_path,
            image.pose,
            frame_idx as f64,
        ));
    }

    if model_dir.join("points3D.bin").exists() {
        dataset.initial_points = read_points_bin(&model_dir.join("points3D.bin"))?;
    } else if model_dir.join("points3D.txt").exists() {
        dataset.initial_points = read_points_txt(&model_dir.join("points3D.txt"))?;
    }

    Ok(dataset)
}

fn discover_colmap_model_dir(root: &Path) -> Option<PathBuf> {
    if has_colmap_model_files(root) {
        return Some(root.to_path_buf());
    }
    let candidates = [root.join("sparse"), root.join("sparse/0")];
    for candidate in candidates {
        if has_colmap_model_files(&candidate) {
            return Some(candidate);
        }
    }
    find_nested_dir(root, &|path| has_colmap_model_files(path))
}

fn has_colmap_model_files(path: &Path) -> bool {
    path.is_dir()
        && ((path.join("cameras.txt").exists() && path.join("images.txt").exists())
            || (path.join("cameras.bin").exists() && path.join("images.bin").exists()))
}

fn find_nested_dir(root: &Path, predicate: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if predicate(&path) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_nested_dir(&path, predicate) {
                return Some(found);
            }
        }
    }
    None
}

fn resolve_colmap_image_path(
    root: &Path,
    model_dir: &Path,
    name: &str,
) -> Result<PathBuf, TrainingError> {
    let candidates = [
        root.join(name),
        model_dir.join(name),
        root.join("images").join(name),
        model_dir.parent().unwrap_or(root).join(name),
        model_dir.parent().unwrap_or(root).join("images").join(name),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let basename = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(name);
    find_file_recursive(root, basename).ok_or_else(|| {
        TrainingError::InvalidInput(format!(
            "failed to resolve COLMAP image {} under {}",
            name,
            root.display()
        ))
    })
}

fn find_file_recursive(root: &Path, basename: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|value| value == basename)
                .unwrap_or(false)
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, basename) {
                return Some(found);
            }
        }
    }
    None
}

fn read_cameras_txt(path: &Path) -> Result<HashMap<u32, ColmapCamera>, TrainingError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut cameras = HashMap::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 5 {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: expected at least 5 columns",
                path.display(),
                line_num + 1
            )));
        }
        let camera_id = parse_u32(parts[0], path, line_num + 1, "camera id")?;
        let model = parts[1];
        let width = parse_u32(parts[2], path, line_num + 1, "width")?;
        let height = parse_u32(parts[3], path, line_num + 1, "height")?;
        let params = parts[4..]
            .iter()
            .map(|value| parse_f64(value, path, line_num + 1, "camera parameter"))
            .collect::<Result<Vec<_>, _>>()?;
        cameras.insert(camera_id, camera_from_model(model, width, height, &params)?);
    }

    Ok(cameras)
}

fn read_images_txt(path: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut expecting_header = true;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !expecting_header {
            expecting_header = true;
            continue;
        }

        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 10 {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: expected image header with at least 10 columns",
                path.display(),
                line_num + 1
            )));
        }
        let qw = parse_f32(parts[1], path, line_num + 1, "qw")?;
        let qx = parse_f32(parts[2], path, line_num + 1, "qx")?;
        let qy = parse_f32(parts[3], path, line_num + 1, "qy")?;
        let qz = parse_f32(parts[4], path, line_num + 1, "qz")?;
        let tx = parse_f32(parts[5], path, line_num + 1, "tx")?;
        let ty = parse_f32(parts[6], path, line_num + 1, "ty")?;
        let tz = parse_f32(parts[7], path, line_num + 1, "tz")?;
        let camera_id = parse_u32(parts[8], path, line_num + 1, "camera id")?;
        let world_to_camera =
            SE3::from_quat_translation(Quat::from_xyzw(qx, qy, qz, qw), Vec3::new(tx, ty, tz));
        images.push(ColmapImage {
            camera_id,
            name: parts[9].to_string(),
            pose: world_to_camera.inverse(),
        });
        expecting_header = false;
    }

    images.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    Ok(images)
}

fn read_points_txt(path: &Path) -> Result<Vec<([f32; 3], Option<[f32; 3]>)>, TrainingError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut points = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 7 {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: expected at least 7 columns",
                path.display(),
                line_num + 1
            )));
        }
        let x = parse_f32(parts[1], path, line_num + 1, "x")?;
        let y = parse_f32(parts[2], path, line_num + 1, "y")?;
        let z = parse_f32(parts[3], path, line_num + 1, "z")?;
        let r = parse_f32(parts[4], path, line_num + 1, "r")? / 255.0;
        let g = parse_f32(parts[5], path, line_num + 1, "g")? / 255.0;
        let b = parse_f32(parts[6], path, line_num + 1, "b")? / 255.0;
        points.push(([x, y, z], Some([r, g, b])));
    }

    Ok(points)
}

fn read_cameras_bin(path: &Path) -> Result<HashMap<u32, ColmapCamera>, TrainingError> {
    let mut file = File::open(path)?;
    let count = read_u64(&mut file)? as usize;
    let mut cameras = HashMap::with_capacity(count);
    for _ in 0..count {
        let camera_id = read_u32(&mut file)?;
        let model_id = read_u32(&mut file)?;
        let width = read_u64(&mut file)? as u32;
        let height = read_u64(&mut file)? as u32;
        let model_name = camera_model_name(model_id)?;
        let param_count = camera_param_count(model_id)?;
        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(read_f64(&mut file)?);
        }
        cameras.insert(
            camera_id,
            camera_from_model(model_name, width, height, &params)?,
        );
    }
    Ok(cameras)
}

fn read_images_bin(path: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let mut file = File::open(path)?;
    let count = read_u64(&mut file)? as usize;
    let mut images = Vec::with_capacity(count);
    for _ in 0..count {
        let _image_id = read_u32(&mut file)?;
        let qw = read_f64(&mut file)? as f32;
        let qx = read_f64(&mut file)? as f32;
        let qy = read_f64(&mut file)? as f32;
        let qz = read_f64(&mut file)? as f32;
        let tx = read_f64(&mut file)? as f32;
        let ty = read_f64(&mut file)? as f32;
        let tz = read_f64(&mut file)? as f32;
        let camera_id = read_u32(&mut file)?;
        let name = read_c_string(&mut file)?;
        let point_count = read_u64(&mut file)? as usize;
        skip_bytes(&mut file, point_count * (8 + 8 + 8))?;
        let world_to_camera =
            SE3::from_quat_translation(Quat::from_xyzw(qx, qy, qz, qw), Vec3::new(tx, ty, tz));
        images.push(ColmapImage {
            camera_id,
            name,
            pose: world_to_camera.inverse(),
        });
    }
    images.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    Ok(images)
}

fn read_points_bin(path: &Path) -> Result<Vec<([f32; 3], Option<[f32; 3]>)>, TrainingError> {
    let mut file = File::open(path)?;
    let count = read_u64(&mut file)? as usize;
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        let _point_id = read_u64(&mut file)?;
        let x = read_f64(&mut file)? as f32;
        let y = read_f64(&mut file)? as f32;
        let z = read_f64(&mut file)? as f32;
        let mut rgb = [0u8; 3];
        file.read_exact(&mut rgb)?;
        let _error = read_f64(&mut file)?;
        let track_length = read_u64(&mut file)? as usize;
        skip_bytes(&mut file, track_length * (4 + 4))?;
        points.push((
            [x, y, z],
            Some([
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]),
        ));
    }
    Ok(points)
}

fn camera_from_model(
    model: &str,
    width: u32,
    height: u32,
    params: &[f64],
) -> Result<ColmapCamera, TrainingError> {
    let (fx, fy, cx, cy) = match model {
        "SIMPLE_PINHOLE"
        | "SIMPLE_RADIAL"
        | "RADIAL"
        | "SIMPLE_RADIAL_FISHEYE"
        | "RADIAL_FISHEYE" => {
            if params.len() < 3 {
                return Err(TrainingError::InvalidInput(format!(
                    "COLMAP camera model {} requires at least 3 parameters",
                    model
                )));
            }
            (params[0], params[0], params[1], params[2])
        }
        "PINHOLE" | "OPENCV" | "OPENCV_FISHEYE" | "FULL_OPENCV" | "THIN_PRISM_FISHEYE" | "FOV" => {
            if params.len() < 4 {
                return Err(TrainingError::InvalidInput(format!(
                    "COLMAP camera model {} requires at least 4 parameters",
                    model
                )));
            }
            (params[0], params[1], params[2], params[3])
        }
        other => {
            return Err(TrainingError::InvalidInput(format!(
                "unsupported COLMAP camera model {}",
                other
            )))
        }
    };
    Ok(ColmapCamera {
        width,
        height,
        fx: fx as f32,
        fy: fy as f32,
        cx: cx as f32,
        cy: cy as f32,
    })
}

fn camera_model_name(model_id: u32) -> Result<&'static str, TrainingError> {
    match model_id {
        0 => Ok("SIMPLE_PINHOLE"),
        1 => Ok("PINHOLE"),
        2 => Ok("SIMPLE_RADIAL"),
        3 => Ok("RADIAL"),
        4 => Ok("OPENCV"),
        5 => Ok("OPENCV_FISHEYE"),
        6 => Ok("FULL_OPENCV"),
        7 => Ok("FOV"),
        8 => Ok("SIMPLE_RADIAL_FISHEYE"),
        9 => Ok("RADIAL_FISHEYE"),
        10 => Ok("THIN_PRISM_FISHEYE"),
        other => Err(TrainingError::InvalidInput(format!(
            "unsupported COLMAP camera model id {}",
            other
        ))),
    }
}

fn camera_param_count(model_id: u32) -> Result<usize, TrainingError> {
    match model_id {
        0 => Ok(3),
        1 => Ok(4),
        2 => Ok(4),
        3 => Ok(5),
        4 => Ok(8),
        5 => Ok(8),
        6 => Ok(12),
        7 => Ok(5),
        8 => Ok(4),
        9 => Ok(5),
        10 => Ok(12),
        other => Err(TrainingError::InvalidInput(format!(
            "unsupported COLMAP camera model id {}",
            other
        ))),
    }
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, TrainingError> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, TrainingError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64, TrainingError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_c_string<R: Read>(reader: &mut R) -> Result<String, TrainingError> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        if byte[0] == 0 {
            break;
        }
        bytes.push(byte[0]);
    }
    String::from_utf8(bytes)
        .map_err(|err| TrainingError::InvalidInput(format!("invalid UTF-8 in image name: {}", err)))
}

fn skip_bytes<R: Read>(reader: &mut R, count: usize) -> Result<(), TrainingError> {
    let mut remaining = count;
    let mut buffer = [0u8; 4096];
    while remaining > 0 {
        let chunk = remaining.min(buffer.len());
        reader.read_exact(&mut buffer[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn parse_u32(value: &str, path: &Path, line_num: usize, label: &str) -> Result<u32, TrainingError> {
    value.parse::<u32>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num,
            label,
            value,
            err
        ))
    })
}

fn parse_f32(value: &str, path: &Path, line_num: usize, label: &str) -> Result<f32, TrainingError> {
    value.parse::<f32>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num,
            label,
            value,
            err
        ))
    })
}

fn parse_f64(value: &str, path: &Path, line_num: usize, label: &str) -> Result<f64, TrainingError> {
    value.parse::<f64>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num,
            label,
            value,
            err
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{load_colmap_dataset, looks_like_colmap_dataset};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn loads_text_colmap_dataset_with_sparse_points() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("images")).unwrap();
        fs::write(dir.path().join("images/frame_000.png"), []).unwrap();
        fs::write(
            dir.path().join("cameras.txt"),
            "1 PINHOLE 640 480 500 500 320 240\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("images.txt"),
            "1 1 0 0 0 0 0 0 1 frame_000.png\n0 0 -1\n",
        )
        .unwrap();
        fs::write(dir.path().join("points3D.txt"), "1 0 0 1 255 128 0 0\n").unwrap();

        assert!(looks_like_colmap_dataset(dir.path()));
        let dataset = load_colmap_dataset(dir.path()).unwrap();
        assert_eq!(dataset.poses.len(), 1);
        assert_eq!(dataset.initial_points.len(), 1);
        assert_eq!(dataset.intrinsics.width, 640);
    }
}
