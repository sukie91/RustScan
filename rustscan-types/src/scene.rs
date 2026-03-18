//! Scene data types for RustScan.
//!
//! This module provides types for SLAM output and point cloud data.

use serde::{Deserialize, Serialize};
use crate::{Intrinsics, ScenePose};

/// Map point data for SLAM output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MapPointData {
    /// 3D position [x, y, z] in world coordinates
    pub position: [f32; 3],
    /// Optional RGB color [r, g, b] in 0-1 range
    pub color: Option<[f32; 3]>,
}

impl MapPointData {
    /// Create a new map point.
    pub fn new(position: [f32; 3], color: Option<[f32; 3]>) -> Self {
        Self { position, color }
    }
}

/// SLAM output for downstream processing.
///
/// Contains all the data produced by SLAM that is needed for
/// 3DGS training and mesh extraction:
/// - Camera intrinsics
/// - Per-frame poses with image paths
/// - Sparse 3D point cloud
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamOutput {
    /// Camera intrinsics (shared across all frames)
    pub intrinsics: Intrinsics,
    /// Per-frame poses with image paths
    pub poses: Vec<ScenePose>,
    /// Sparse 3D point cloud from SLAM
    pub map_points: Vec<MapPointData>,
}

impl SlamOutput {
    /// Create a new empty SLAM output.
    pub fn new(intrinsics: Intrinsics) -> Self {
        Self {
            intrinsics,
            poses: Vec::new(),
            map_points: Vec::new(),
        }
    } 

    /// Create from a training dataset (without map points).
    pub fn from_dataset(dataset: crate::TrainingDataset) -> Self {
        Self {
            intrinsics: dataset.intrinsics,
            poses: dataset.poses,
            map_points: dataset.initial_points.into_iter()
                .map(|(pos, color)| MapPointData::new(pos, color))
                .collect(),
        }
    }

    /// Convert to a training dataset.
    pub fn to_dataset(&self) -> crate::TrainingDataset {
        crate::TrainingDataset {
            intrinsics: self.intrinsics,
            poses: self.poses.clone(),
            initial_points: self.map_points.iter()
                .map(|mp| (mp.position, mp.color))
                .collect(),
        }
    }

    /// Add a pose to the output.
    pub fn add_pose(&mut self, pose: ScenePose) {
        self.poses.push(pose);
    }

    /// Add a map point to the output.
    pub fn add_map_point(&mut self, point: MapPointData) {
        self.map_points.push(point);
    }

    /// Get the number of poses.
    pub fn num_poses(&self) -> usize {
        self.poses.len()
    }

    /// Get the number of map points.
    pub fn num_points(&self) -> usize {
        self.map_points.len()
    }

    /// Save the SLAM output to a JSON file.
    pub fn save(&self, path: &std::path::PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a SLAM output from a JSON file.
    pub fn load(path: &std::path::PathBuf) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let output: SlamOutput = serde_json::from_str(&json)?;
        Ok(output)
    }

    /// Get all unique frame IDs.
    pub fn frame_ids(&self) -> Vec<u64> {
        self.poses.iter().map(|p| p.frame_id).collect()
    }

    /// Get all image paths.
    pub fn image_paths(&self) -> Vec<&std::path::Path> {
        self.poses.iter().map(|p| p.image_path.as_path()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SE3;
    use std::path::PathBuf;

    #[test]
    fn test_map_point_data() {
        let point = MapPointData::new([1.0, 2.0, 3.0], Some([0.5, 0.5, 0.5]));
        assert_eq!(point.position, [1.0, 2.0, 3.0]);
        assert_eq!(point.color, Some([0.5, 0.5, 0.5]));
    }

    #[test]
    fn test_slam_output() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let mut output = SlamOutput::new(intrinsics);

        let pose = ScenePose::new(
            0,
            PathBuf::from("frame_0000.jpg"),
            SE3::identity(),
            0.0,
        );
        output.add_pose(pose);
        output.add_map_point(MapPointData::new([0.0, 0.0, 1.0], None));

        assert_eq!(output.num_poses(), 1);
        assert_eq!(output.num_points(), 1);
    }

    #[test]
    fn test_slam_output_to_dataset() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let mut output = SlamOutput::new(intrinsics);

        let pose = ScenePose::new(
            0,
            PathBuf::from("frame_0000.jpg"),
            SE3::identity(),
            0.0,
        );
        output.add_pose(pose);
        output.add_map_point(MapPointData::new([0.0, 0.0, 1.0], Some([0.5, 0.5, 0.5])));

        let dataset = output.to_dataset();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.initial_points.len(), 1);
    }

    #[test]
    fn test_slam_output_serde() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let mut output = SlamOutput::new(intrinsics);

        let pose = ScenePose::new(
            0,
            PathBuf::from("frame_0000.jpg"),
            SE3::identity(),
            0.0,
        );
        output.add_pose(pose);

        let json = serde_json::to_string(&output).unwrap();
        let decoded: SlamOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.num_poses(), 1);
    }
}