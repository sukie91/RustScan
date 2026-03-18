//! Gaussian 3D data structures.
//!
//! This module will be populated from RustSLAM/src/fusion/gaussian.rs in Story 9-3.

use glam::{Vec3, Quat};
use serde::{Deserialize, Serialize};

/// A single 3D Gaussian primitive.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Gaussian3D {
    /// Position in world space
    pub position: Vec3,
    /// Scale along each axis
    pub scale: Vec3,
    /// Rotation as quaternion (w, x, y, z)
    pub rotation: Quat,
    /// Opacity [0, 1]
    pub opacity: f32,
    /// RGB color [0, 1]^3
    pub color: [f32; 3],
    /// State for tracking
    pub state: GaussianState,
}

impl Gaussian3D {
    /// Create a new Gaussian with default state.
    pub fn new(position: Vec3, scale: Vec3, rotation: Quat, opacity: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            scale,
            rotation,
            opacity,
            color,
            state: GaussianState::default(),
        }
    }
}

/// State of a Gaussian for densification/pruning.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GaussianState {
    /// Number of frames this Gaussian is visible in
    pub visibility_count: u32,
    /// Accumulated gradient for densification
    pub gradient_accum: f32,
    /// Whether this Gaussian is marked for splitting
    pub needs_split: bool,
    /// Whether this Gaussian is marked for pruning
    pub needs_prune: bool,
}

/// A collection of Gaussians representing a scene.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GaussianMap {
    /// All Gaussians in the scene
    pub gaussians: Vec<Gaussian3D>,
    /// Bounding box of the scene
    pub bounding_box: ([f32; 3], [f32; 3]),
}

impl GaussianMap {
    /// Create an empty Gaussian map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a vector of Gaussians.
    pub fn from_gaussians(gaussians: Vec<Gaussian3D>) -> Self {
        let bounding_box = Self::compute_bounding_box(&gaussians);
        Self { gaussians, bounding_box }
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    fn compute_bounding_box(gaussians: &[Gaussian3D]) -> ([f32; 3], [f32; 3]) {
        if gaussians.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for g in gaussians {
            let p = g.position;
            min[0] = min[0].min(p.x);
            min[1] = min[1].min(p.y);
            min[2] = min[2].min(p.z);
            max[0] = max[0].max(p.x);
            max[1] = max[1].max(p.y);
            max[2] = max[2].max(p.z);
        }

        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_3d_creation() {
        let g = Gaussian3D::new(
            Vec3::ZERO,
            Vec3::ONE,
            Quat::IDENTITY,
            0.5,
            [1.0, 0.0, 0.0],
        );
        assert_eq!(g.position, Vec3::ZERO);
        assert_eq!(g.opacity, 0.5);
        assert_eq!(g.color, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gaussian_map() {
        let mut map = GaussianMap::new();
        assert!(map.is_empty());

        let g = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ONE,
            Quat::IDENTITY,
            0.5,
            [1.0, 0.0, 0.0],
        );
        map.gaussians.push(g);

        assert_eq!(map.len(), 1);
    }
}