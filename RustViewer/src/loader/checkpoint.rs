//! Loader for SLAM checkpoint JSON files.

use std::path::Path;
use thiserror::Error;

use crate::renderer::scene::Scene;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("OBJ parse error: {0}")]
    ObjParse(String),
    #[error("PLY parse error: {0}")]
    PlyParse(String),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
}

/// Load a SLAM checkpoint JSON and populate trajectory + map points in the scene.
pub fn load_checkpoint(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let checkpoint = rustslam::pipeline::checkpoint::load_checkpoint(path)
        .map_err(|e| LoadError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

    // Extract camera trajectory from keyframes with valid poses
    for kf in &checkpoint.keyframes {
        if let Some(pose) = &kf.pose {
            scene.trajectory.push(pose.translation);
            scene.bounds.extend(pose.translation);
        }
    }

    // Extract map points (skip outliers)
    let ys: Vec<f32> = checkpoint
        .map_points
        .iter()
        .filter(|mp| !mp.is_outlier)
        .map(|mp| mp.position[1])
        .collect();

    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_range = (y_max - y_min).max(1e-6);

    for mp in checkpoint.map_points.iter().filter(|mp| !mp.is_outlier) {
        let pos = mp.position;
        scene.map_points.push(pos);
        scene.bounds.extend(pos);

        // Depth-shaded color: green (near) → red (far), based on Y
        let t = ((pos[1] - y_min) / y_range).clamp(0.0, 1.0);
        scene.map_point_colors.push([t, 1.0 - t, 0.2]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_checkpoint_empty() {
        let json = r#"{"version":1,"frame_index":0,"keyframes":[],"map_points":[]}"#;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(json.as_bytes()).unwrap();
        let path = tmpfile.path().to_path_buf();

        let mut scene = Scene::default();
        let result = load_checkpoint(&path, &mut scene);
        assert!(result.is_ok(), "empty checkpoint should succeed: {:?}", result);
        assert!(scene.trajectory.is_empty());
        assert!(scene.map_points.is_empty());
        assert!(!scene.bounds.is_valid());
    }
}
