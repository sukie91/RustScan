//! TSDF Volume for Mesh Extraction from 3DGS
//!
//! Implements Truncated Signed Distance Function (TSDF) volume fusion
//! for extracting mesh from Gaussian Splatting representations.
//!
//! Based on:
//! - Kinect Fusion algorithm
//! - PGSR (Planar-based Gaussian Splatting Reconstruction)

use std::collections::HashMap;
use glam::{Vec3, Mat3, Mat4};

/// A 3D voxel cell in the TSDF volume
#[derive(Debug, Clone)]
pub struct Voxel {
    /// TSDF value: -1 (inside) to 1 (outside), 0 = surface
    pub tsdf: f32,
    /// Weighted sum for averaging
    pub weight: f32,
    /// RGB color (optional, for visualization)
    pub color: [f32; 3],
    /// Color weight
    pub color_weight: f32,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            tsdf: 1.0,
            weight: 0.0,
            color: [0.0, 0.0, 0.0],
            color_weight: 0.0,
        }
    }
}

/// TSDF Volume configuration
#[derive(Debug, Clone)]
pub struct TsdfConfig {
    /// Voxel size in world units (e.g., 0.01 for 1cm)
    pub voxel_size: f32,
    /// Truncation distance (typically 3-5 * voxel_size)
    pub sdf_trunc: f32,
    /// Volume bounds
    pub min_bound: Vec3,
    pub max_bound: Vec3,
    /// Maximum weight per voxel
    pub max_weight: f32,
    /// Integration weight
    pub integration_weight: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.01,       // 1cm voxels
            sdf_trunc: 0.05,        // 5cm truncation
            min_bound: Vec3::new(-1.0, -1.0, -1.0),
            max_bound: Vec3::new(1.0, 1.0, 1.0),
            max_weight: 100.0,
            integration_weight: 1.0,
        }
    }
}

/// TSDF Volume for volumetric fusion
#[derive(Debug)]
pub struct TsdfVolume {
    /// Configuration
    config: TsdfConfig,
    /// Voxel grid: key = (x, y, z) index, value = voxel data
    voxels: HashMap<(i32, i32, i32), Voxel>,
    /// Volume dimensions
    volume_dims: (i32, i32, i32),
    /// Number of integrated frames
    frame_count: usize,
}

impl TsdfVolume {
    /// Create a new TSDF volume from bounds
    pub fn new(config: TsdfConfig) -> Self {
        let dx = ((config.max_bound.x - config.min_bound.x) / config.voxel_size).ceil() as i32;
        let dy = ((config.max_bound.y - config.min_bound.y) / config.voxel_size).ceil() as i32;
        let dz = ((config.max_bound.z - config.min_bound.z) / config.voxel_size).ceil() as i32;

        Self {
            config,
            voxels: HashMap::new(),
            volume_dims: (dx, dy, dz),
            frame_count: 0,
        }
    }

    /// Create volume centered at origin with given size
    pub fn centered(size_meters: f32, voxel_size: f32) -> Self {
        let half = size_meters / 2.0;
        let config = TsdfConfig {
            voxel_size,
            sdf_trunc: voxel_size * 4.0,
            min_bound: Vec3::new(-half, -half, -half),
            max_bound: Vec3::new(half, half, half),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Convert world position to voxel index
    pub fn world_to_voxel(&self, pos: Vec3) -> Option<(i32, i32, i32)> {
        let x = ((pos.x - self.config.min_bound.x) / self.config.voxel_size).floor() as i32;
        let y = ((pos.y - self.config.min_bound.y) / self.config.voxel_size).floor() as i32;
        let z = ((pos.z - self.config.min_bound.z) / self.config.voxel_size).floor() as i32;

        if x >= 0 && x < self.volume_dims.0 &&
           y >= 0 && y < self.volume_dims.1 &&
           z >= 0 && z < self.volume_dims.2 {
            Some((x, y, z))
        } else {
            None
        }
    }

    /// Convert voxel index to world position (center of voxel)
    pub fn voxel_to_world(&self, ix: i32, iy: i32, iz: i32) -> Vec3 {
        Vec3::new(
            self.config.min_bound.x + (ix as f32 + 0.5) * self.config.voxel_size,
            self.config.min_bound.y + (iy as f32 + 0.5) * self.config.voxel_size,
            self.config.min_bound.z + (iz as f32 + 0.5) * self.config.voxel_size,
        )
    }

    /// Get voxel at index (create if not exists)
    pub fn get_voxel_mut(&mut self, ix: i32, iy: i32, iz: i32) -> &mut Voxel {
        self.voxels
            .entry((ix, iy, iz))
            .or_insert_with(Voxel::default)
    }

    /// Get voxel at index (read only)
    pub fn get_voxel(&self, ix: i32, iy: i32, iz: i32) -> Option<&Voxel> {
        self.voxels.get(&(ix, iy, iz))
    }

    /// Integrate a depth map into the TSDF volume
    ///
    /// # Arguments
    /// * `depth` - Depth map (in meters)
    /// * `color` - RGB color map (optional, can be None)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `intrinsics` - Camera intrinsics [fx, fy, cx, cy]
    /// * `extrinsics` - Camera pose as 4x4 matrix (world to camera)
    pub fn integrate(
        &mut self,
        depth: &[f32],
        color: Option<&[[u8; 3]]>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Mat4,
    ) {
        let [fx, fy, cx, cy] = intrinsics;
        let trunc = self.config.sdf_trunc;

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let z = depth[idx];

                // Skip invalid depths
                if z <= 0.001 || z > 10.0 {
                    continue;
                }

                // Back-project to camera space
                let cam_x = (x as f32 - cx) * z / fx;
                let cam_y = (y as f32 - cy) * z / fy;
                let cam_z = z;

                // Transform to world space
                let world_pos = extrinsics.transform_point3(Vec3::new(cam_x, cam_y, cam_z));

                // Get voxel position
                if let Some((vx, vy, vz)) = self.world_to_voxel(world_pos) {
                    // Compute SDF (distance to surface)
                    // Positive = in front of surface, Negative = behind
                    let sdf = z - cam_z;

                    // Truncate SDF
                    let tsdf = sdf.clamp(-trunc, trunc) / trunc;

                    // Get voxel and update
                    let voxel = self.get_voxel_mut(vx, vy, vz);

                    // TSDF fusion
                    let w = self.config.integration_weight;
                    voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * w) / (voxel.weight + w);
                    voxel.weight = (voxel.weight + w).min(self.config.max_weight);

                    // Color fusion (if available)
                    if let Some(colors) = color {
                        let c = colors[idx];
                        let cr = c[0] as f32 / 255.0;
                        let cg = c[1] as f32 / 255.0;
                        let cb = c[2] as f32 / 255.0;

                        voxel.color = [
                            (voxel.color[0] * voxel.color_weight + cr * w) / (voxel.color_weight + w),
                            (voxel.color[1] * voxel.color_weight + cg * w) / (voxel.color_weight + w),
                            (voxel.color[2] * voxel.color_weight + cb * w) / (voxel.color_weight + w),
                        ];
                        voxel.color_weight += w;
                    }
                }
            }
        }

        self.frame_count += 1;
    }

    /// Integrate from Gaussian rendering (project Gaussians to depth)
    ///
    /// This is the key integration for 3DGS -> Mesh
    pub fn integrate_from_gaussians<F>(
        &mut self,
        get_depth_at: F,
        get_color_at: Option<&dyn Fn(usize) -> Option<[u8; 3]>>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Mat4,
    ) where
        F: Fn(usize) -> f32,
    {
        let [fx, fy, cx, cy] = intrinsics;
        let trunc = self.config.sdf_trunc;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let z = get_depth_at(idx);

                if z <= 0.001 || z > 10.0 {
                    continue;
                }

                // Back-project to camera space
                let cam_x = (x as f32 - cx) * z / fx;
                let cam_y = (y as f32 - cy) * z / fy;

                // Transform to world space
                let world_pos = extrinsics.transform_point3(Vec3::new(cam_x, cam_y, z));

                if let Some((vx, vy, vz)) = self.world_to_voxel(world_pos) {
                    let sdf = 0.0; // Surface point
                    let tsdf = sdf.clamp(-trunc, trunc) / trunc;

                    let voxel = self.get_voxel_mut(vx, vy, vz);
                    let w = self.config.integration_weight;
                    voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * w) / (voxel.weight + w);
                    voxel.weight = (voxel.weight + w).min(self.config.max_weight);

                    // Color integration
                    if let Some(color_fn) = get_color_at {
                        if let Some(c) = color_fn(idx) {
                            let cr = c[0] as f32 / 255.0;
                            let cg = c[1] as f32 / 255.0;
                            let cb = c[2] as f32 / 255.0;
                            voxel.color = [
                                (voxel.color[0] * voxel.color_weight + cr * w) / (voxel.color_weight + w),
                                (voxel.color[1] * voxel.color_weight + cg * w) / (voxel.color_weight + w),
                                (voxel.color[2] * voxel.color_weight + cb * w) / (voxel.color_weight + w),
                            ];
                            voxel.color_weight += w;
                        }
                    }
                }
            }
        }

        self.frame_count += 1;
    }

    /// Get all voxels that are near the surface (|tsdf| < threshold)
    pub fn get_surface_voxels(&self, threshold: f32) -> Vec<(i32, i32, i32, &Voxel)> {
        self.voxels
            .iter()
            .filter(|(_, v)| v.weight > 0.0 && v.tsdf.abs() < threshold)
            .map(|((&x, &y, &z), v)| (x, y, z, v))
            .collect()
    }

    /// Get voxel grid dimensions
    pub fn dimensions(&self) -> (i32, i32, i32) {
        self.volume_dims
    }

    /// Get configuration
    pub fn config(&self) -> &TsdfConfig {
        &self.config
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get number of active voxels
    pub fn num_voxels(&self) -> usize {
        self.voxels.len()
    }

    /// Clear the volume
    pub fn clear(&mut self) {
        self.voxels.clear();
        self.frame_count = 0;
    }
}

impl Default for TsdfVolume {
    fn default() -> Self {
        Self::new(TsdfConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_creation() {
        let config = TsdfConfig::default();
        let volume = TsdfVolume::new(config);

        assert_eq!(volume.dimensions(), (200, 200, 200));
    }

    #[test]
    fn test_world_voxel_conversion() {
        let volume = TsdfVolume::centered(1.0, 0.01);

        let pos = Vec3::new(0.0, 0.0, 0.0);
        let voxel = volume.world_to_voxel(pos);
        assert!(voxel.is_some());

        let (vx, vy, vz) = voxel.unwrap();
        let world = volume.voxel_to_world(vx, vy, vz);

        assert!(world.distance(pos) < 0.01);
    }

    #[test]
    fn test_integrate_simple() {
        let mut volume = TsdfVolume::centered(0.5, 0.01);

        // Simple depth map: 10x10, all at z=1.0
        let depth = vec![1.0f32; 10 * 10];
        let intrinsics = [500.0, 500.0, 5.0, 5.0];
        let extrinsics = Mat4::IDENTITY;

        volume.integrate(&depth, None, 10, 10, intrinsics, &extrinsics);

        assert!(volume.frame_count() > 0);
    }
}
