//! Clustered Gaussian management for efficient rendering.
//!
//! Implements LiteGS-style spatial clustering for:
//! - Frustum culling of invisible clusters
//! - Reduced memory bandwidth during rendering
//! - Better cache locality for large scenes

use glam::{Vec3, Mat4};
use std::collections::HashMap;

/// Axis-aligned bounding box for a cluster.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterAABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl ClusterAABB {
    pub fn new() -> Self {
        Self {
            min: [f32::MAX, f32::MAX, f32::MAX],
            max: [f32::MIN, f32::MIN, f32::MIN],
        }
    }

    /// Expand the AABB to include a point.
    pub fn expand(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(point[i]);
            self.max[i] = self.max[i].max(point[i]);
        }
    }

    /// Check if the AABB intersects a frustum.
    ///
    /// Uses the separating axis theorem for frustum-AABB intersection.
    pub fn intersects_frustum(&self, view_proj: &Mat4) -> bool {
        // Transform AABB corners to clip space and check if any are visible
        let corners = [
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
        ];

        let mut all_left = true;
        let mut all_right = true;
        let mut all_bottom = true;
        let mut all_top = true;
        let mut all_near = true;
        let mut all_far = true;

        for corner in &corners {
            let p = Vec3::from(*corner);
            let clip = *view_proj * p.extend(1.0);
            let w = clip.w.abs().max(1e-6);
            let x = clip.x / w;
            let y = clip.y / w;
            let z = clip.z / w;

            if x > -1.0 { all_left = false; }
            if x < 1.0 { all_right = false; }
            if y > -1.0 { all_bottom = false; }
            if y < 1.0 { all_top = false; }
            if z > 0.0 { all_near = false; }
            if z < 1.0 { all_far = false; }
        }

        // If all corners are outside any single plane, the AABB is not visible
        !(all_left || all_right || all_bottom || all_top || all_near || all_far)
    }

    /// Get the center of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get the half-extent of the AABB.
    pub fn half_extent(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }
}

/// Cluster assignment for a set of Gaussians.
#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    /// Cluster index for each Gaussian.
    pub cluster_indices: Vec<u32>,
    /// Number of clusters.
    pub num_clusters: usize,
    /// AABB for each cluster.
    pub aabbs: Vec<ClusterAABB>,
    /// Gaussians per cluster (for allocation).
    pub cluster_sizes: Vec<usize>,
}

impl ClusterAssignment {
    pub fn new(num_gaussians: usize, num_clusters: usize) -> Self {
        Self {
            cluster_indices: vec![0; num_gaussians],
            num_clusters,
            aabbs: vec![ClusterAABB::new(); num_clusters],
            cluster_sizes: vec![0; num_clusters],
        }
    }

    /// Assign Gaussians to clusters using spatial hashing.
    ///
    /// This is a simple O(n) clustering approach suitable for real-time updates.
    /// Uses a 3D grid-based spatial hash.
    pub fn assign_spatial_hash(
        positions: &[[f32; 3]],
        cluster_size: usize,
        scene_extent: f32,
    ) -> Self {
        let n = positions.len();
        if n == 0 {
            return Self::new(0, 0);
        }

        // Compute grid cell size based on desired cluster size
        // Target: cluster_size Gaussians per cluster
        // Volume per Gaussian ≈ (scene_extent)^3 / n
        // Cluster volume ≈ cluster_size * volume_per_gaussian
        // Cell size ≈ (cluster_volume)^(1/3)
        let volume_per_gaussian = (scene_extent * scene_extent * scene_extent) / (n as f32);
        let cluster_volume = cluster_size as f32 * volume_per_gaussian;
        let cell_size = cluster_volume.powf(1.0 / 3.0).max(0.01);

        // Use spatial hash to assign clusters
        let mut hash_to_cluster: HashMap<(i32, i32, i32), u32> = HashMap::new();
        let mut cluster_indices = vec![0u32; n];
        let mut next_cluster = 0u32;

        for (i, pos) in positions.iter().enumerate() {
            let cell = (
                (pos[0] / cell_size).floor() as i32,
                (pos[1] / cell_size).floor() as i32,
                (pos[2] / cell_size).floor() as i32,
            );

            let cluster = *hash_to_cluster.entry(cell).or_insert_with(|| {
                let c = next_cluster;
                next_cluster += 1;
                c
            });
            cluster_indices[i] = cluster;
        }

        let num_clusters = next_cluster as usize;
        let mut aabbs = vec![ClusterAABB::new(); num_clusters];
        let mut cluster_sizes = vec![0usize; num_clusters];

        // Compute AABBs and sizes
        for (i, pos) in positions.iter().enumerate() {
            let cluster = cluster_indices[i] as usize;
            aabbs[cluster].expand(*pos);
            cluster_sizes[cluster] += 1;
        }

        Self {
            cluster_indices,
            num_clusters,
            aabbs,
            cluster_sizes,
        }
    }

    /// Get visible clusters for a given view-projection matrix.
    pub fn get_visible_clusters(&self, view_proj: &Mat4) -> Vec<u32> {
        (0..self.num_clusters)
            .filter(|&c| self.aabbs[c].intersects_frustum(view_proj))
            .map(|c| c as u32)
            .collect()
    }

    /// Get Gaussian indices for a specific cluster.
    pub fn get_cluster_gaussians(&self, cluster: u32) -> Vec<usize> {
        self.cluster_indices
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == cluster)
            .map(|(i, _)| i)
            .collect()
    }

    /// Update AABBs after Gaussian positions change.
    pub fn update_aabbs(&mut self, positions: &[[f32; 3]]) {
        // Reset AABBs
        for aabb in &mut self.aabbs {
            *aabb = ClusterAABB::new();
        }

        // Recompute from positions
        for (i, pos) in positions.iter().enumerate() {
            let cluster = self.cluster_indices[i] as usize;
            if cluster < self.aabbs.len() {
                self.aabbs[cluster].expand(*pos);
            }
        }
    }

    /// Reassign clusters (called after significant topology changes).
    pub fn reassign(
        &mut self,
        positions: &[[f32; 3]],
        cluster_size: usize,
        scene_extent: f32,
    ) {
        let new_assignment = Self::assign_spatial_hash(positions, cluster_size, scene_extent);
        *self = new_assignment;
    }
}

/// Morton code computation for Z-order sorting.
pub mod morton {
    /// Compute 30-bit Morton code from 3D coordinates.
    ///
    /// Uses 10 bits per dimension for a total of 30 bits.
    /// Z-order: x bits at positions 0, 3, 6, ...; y at 1, 4, 7, ...; z at 2, 5, 8, ...
    pub fn encode(x: u32, y: u32, z: u32) -> u32 {
        // Spread bits for each dimension
        let x = spread_bits(x & 0x3FF);
        let y = spread_bits(y & 0x3FF);
        let z = spread_bits(z & 0x3FF);
        // Interleave: x at 0, 3, 6, ...; y at 1, 4, 7, ...; z at 2, 5, 8, ...
        (x << 2) | (y << 1) | z
    }

    /// Spread bits: 10-bit value becomes 30-bit with zeros between.
    fn spread_bits(mut v: u32) -> u32 {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        v
    }

    /// Compute Morton code from a position within a bounding box.
    ///
    /// Returns a 30-bit Morton code representing the Z-order position.
    pub fn from_position(pos: [f32; 3], min: [f32; 3], max: [f32; 3]) -> u32 {
        // Normalize to [0, 1] within the AABB
        let x = ((pos[0] - min[0]) / (max[0] - min[0]).max(1e-6)).clamp(0.0, 1.0);
        let y = ((pos[1] - min[1]) / (max[1] - min[1]).max(1e-6)).clamp(0.0, 1.0);
        let z = ((pos[2] - min[2]) / (max[2] - min[2]).max(1e-6)).clamp(0.0, 1.0);

        // Convert to 10-bit integers
        let xi = (x * 1023.0) as u32;
        let yi = (y * 1023.0) as u32;
        let zi = (z * 1023.0) as u32;

        encode(xi, yi, zi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_expand() {
        let mut aabb = ClusterAABB::new();
        aabb.expand([1.0, 2.0, 3.0]);
        aabb.expand([-1.0, 0.0, 4.0]);

        assert!((aabb.min[0] - (-1.0)).abs() < 1e-6);
        assert!((aabb.max[0] - 1.0).abs() < 1e-6);
        assert!((aabb.min[1] - 0.0).abs() < 1e-6);
        assert!((aabb.max[1] - 2.0).abs() < 1e-6);
        assert!((aabb.min[2] - 3.0).abs() < 1e-6);
        assert!((aabb.max[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_hash_clustering() {
        let positions: Vec<[f32; 3]> = (0..100)
            .map(|i| [(i % 10) as f32, ((i / 10) % 10) as f32, 0.0])
            .collect();

        let assignment = ClusterAssignment::assign_spatial_hash(&positions, 10, 10.0);

        // Should have created some clusters
        assert!(assignment.num_clusters > 0);
        assert!(assignment.num_clusters <= 100);

        // All Gaussians should be assigned
        for &c in &assignment.cluster_indices {
            assert!((c as usize) < assignment.num_clusters);
        }

        // Total cluster sizes should equal number of Gaussians
        let total: usize = assignment.cluster_sizes.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_morton_encode() {
        // Test simple cases
        assert_eq!(morton::encode(0, 0, 0), 0);
        // x=1: spread(1)=1, shifted left by 2 = 4
        assert_eq!(morton::encode(1, 0, 0), 0b100); // x=1 at bit position 2
        // y=1: spread(1)=1, shifted left by 1 = 2
        assert_eq!(morton::encode(0, 1, 0), 0b010); // y=1 at bit position 1
        // z=1: spread(1)=1, not shifted = 1
        assert_eq!(morton::encode(0, 0, 1), 0b001); // z=1 at bit position 0
        // All dimensions
        assert_eq!(morton::encode(1, 1, 1), 0b111); // all bits set at positions 0, 1, 2
    }

    #[test]
    fn test_morton_from_position() {
        let min = [0.0, 0.0, 0.0];
        let max = [10.0, 10.0, 10.0];

        let m0 = morton::from_position([0.0, 0.0, 0.0], min, max);
        let m1 = morton::from_position([5.0, 5.0, 5.0], min, max);
        let m2 = morton::from_position([10.0, 10.0, 10.0], min, max);

        // Origin should have lowest Morton code
        assert!(m0 < m1);
        assert!(m1 < m2);
    }
}