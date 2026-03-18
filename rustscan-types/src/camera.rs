//! Camera types for RustScan.
//!
//! This module provides camera intrinsic parameters and dataset types
//! for 3DGS offline training.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::SE3;

/// Camera intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    /// Focal length in x (pixels)
    pub fx: f32,
    /// Focal length in y (pixels)
    pub fy: f32,
    /// Principal point x (pixels)
    pub cx: f32,
    /// Principal point y (pixels)
    pub cy: f32,
    /// Image width (pixels)
    pub width: u32,
    /// Image height (pixels)
    pub height: u32,
}

impl Intrinsics {
    /// Create new intrinsic parameters.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self { fx, fy, cx, cy, width, height }
    }

    /// Create from a single focal length (assumes square pixels).
    pub fn from_focal(f: f32, width: u32, height: u32) -> Self {
        Self {
            fx: f,
            fy: f,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
        }
    }

    /// Get the field of view in radians (horizontal).
    pub fn fov_x(&self) -> f32 {
        2.0 * (self.width as f32 / (2.0 * self.fx)).atan()
    }

    /// Get the field of view in radians (vertical).
    pub fn fov_y(&self) -> f32 {
        2.0 * (self.height as f32 / (2.0 * self.fy)).atan()
    }

    /// Convert normalized coordinates to pixel coordinates.
    pub fn normalized_to_pixel(&self, x: f32, y: f32) -> (f32, f32) {
        (x * self.fx + self.cx, y * self.fy + self.cy)
    }

    /// Convert pixel coordinates to normalized coordinates.
    pub fn pixel_to_normalized(&self, u: f32, v: f32) -> (f32, f32) {
        ((u - self.cx) / self.fx, (v - self.cy) / self.fy)
    }
}

impl Default for Intrinsics {
    fn default() -> Self {
        // Default iPhone camera parameters
        Self {
            fx: 1500.0,
            fy: 1500.0,
            cx: 960.0,
            cy: 540.0,
            width: 1920,
            height: 1080,
        }
    }
}

/// A single pose in a training dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenePose {
    /// Frame identifier
    pub frame_id: u64,
    /// Path to the image file
    pub image_path: PathBuf,
    /// Camera pose (world-to-camera transform)
    pub pose: SE3,
    /// Timestamp in seconds
    pub timestamp: f64,
}

impl ScenePose {
    /// Create a new scene pose.
    pub fn new(frame_id: u64, image_path: PathBuf, pose: SE3, timestamp: f64) -> Self {
        Self { frame_id, image_path, pose, timestamp }
    }
}

/// Training dataset for 3DGS offline training.
///
/// Contains all the data needed to train a 3DGS scene:
/// - Camera intrinsics (shared across all frames)
/// - Per-frame poses with image paths
/// - Optional initial point cloud for Gaussian initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Camera intrinsics (shared across all frames)
    pub intrinsics: Intrinsics,
    /// Per-frame poses with image paths
    pub poses: Vec<ScenePose>,
    /// Initial point cloud for Gaussian initialization
    /// Each point: (position [x, y, z], optional color [r, g, b])
    pub initial_points: Vec<([f32; 3], Option<[f32; 3]>)>,
}

impl TrainingDataset {
    /// Create a new empty training dataset.
    pub fn new(intrinsics: Intrinsics) -> Self {
        Self {
            intrinsics,
            poses: Vec::new(),
            initial_points: Vec::new(),
        }
    }

    /// Add a pose to the dataset.
    pub fn add_pose(&mut self, pose: ScenePose) {
        self.poses.push(pose);
    }

    /// Add an initial point.
    pub fn add_point(&mut self, position: [f32; 3], color: Option<[f32; 3]>) {
        self.initial_points.push((position, color));
    }

    /// Get the number of poses.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Save the dataset to a JSON file.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a dataset from a JSON file.
    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let dataset: TrainingDataset = serde_json::from_str(&json)?;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsics_default() {
        let intrinsics = Intrinsics::default();
        assert_eq!(intrinsics.width, 1920);
        assert_eq!(intrinsics.height, 1080);
        // Default has fx == fy, but different width/height so FOVs are different
        assert!((intrinsics.fx - intrinsics.fy).abs() < 1e-6);
    }

    #[test]
    fn test_intrinsics_pixel_conversion() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let (x, y) = intrinsics.pixel_to_normalized(960.0, 540.0);
        assert!((x - 0.0).abs() < 1e-6);
        assert!((y - 0.0).abs() < 1e-6);

        let (u, v) = intrinsics.normalized_to_pixel(x, y);
        assert!((u - 960.0).abs() < 1e-6);
        assert!((v - 540.0).abs() < 1e-6);
    }

    #[test]
    fn test_training_dataset() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let mut dataset = TrainingDataset::new(intrinsics);

        let pose = ScenePose::new(
            0,
            PathBuf::from("frame_0000.jpg"),
            SE3::identity(),
            0.0,
        );
        dataset.add_pose(pose);
        dataset.add_point([0.0, 0.0, 1.0], Some([0.5, 0.5, 0.5]));

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.initial_points.len(), 1);
    }

    #[test]
    fn test_training_dataset_serde() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let mut dataset = TrainingDataset::new(intrinsics);

        let pose = ScenePose::new(
            0,
            PathBuf::from("frame_0000.jpg"),
            SE3::identity(),
            0.0,
        );
        dataset.add_pose(pose);

        let json = serde_json::to_string(&dataset).unwrap();
        let decoded: TrainingDataset = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.len(), 1);
    }
}