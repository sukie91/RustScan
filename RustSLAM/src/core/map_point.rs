//! MapPoint representation

use glam::Vec3;

/// A 3D map point
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// Unique ID
    pub id: u64,
    /// 3D position in world frame
    pub position: Vec3,
    /// Normal direction (optional)
    pub normal: Option<Vec3>,
    /// Reference keyframe ID
    pub reference_kf: u64,
    /// Number of observed from keyframes
    pub observations: u32,
    /// Is outlier
    pub is_outlier: bool,
}

impl MapPoint {
    /// Create a new map point
    pub fn new(id: u64, position: Vec3, reference_kf: u64) -> Self {
        Self {
            id,
            position,
            normal: None,
            reference_kf,
            observations: 1,
            is_outlier: false,
        }
    }

    /// Add an observation
    pub fn add_observation(&mut self) {
        self.observations += 1;
    }

    /// Mark as outlier
    pub fn mark_outlier(&mut self) {
        self.is_outlier = true;
    }

    /// Mark as inlier
    pub fn mark_inlier(&mut self) {
        self.is_outlier = false;
    }

    /// Set normal direction
    pub fn set_normal(&mut self, normal: Vec3) {
        self.normal = Some(normal);
    }
}
