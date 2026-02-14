//! Mesh Extractor for 3DGS -> Mesh Pipeline
//!
//! High-level API for extracting mesh from Gaussian Splatting representation.
//! Combines TSDF fusion and Marching Cubes with post-processing.
//!
//! Based on PGSR strategy:
//! 1. Render depth from Gaussians
//! 2. TSDF Fusion
//! 3. Marching Cubes extraction
//! 4. Post-processing (clustering, floaters removal)

use glam::{Vec3, Mat4};
use std::collections::{HashMap, HashSet};

pub use crate::fusion::tsdf_volume::{TsdfVolume, TsdfConfig};
pub use crate::fusion::marching_cubes::{Mesh, MeshVertex, MeshTriangle, MarchingCubes};

/// Mesh extraction configuration
#[derive(Debug, Clone)]
pub struct MeshExtractionConfig {
    /// TSDF volume configuration
    pub tsdf_config: TsdfConfig,
    /// Minimum number of triangles in a cluster to keep
    pub min_cluster_size: usize,
    /// Number of largest clusters to keep
    pub num_largest_clusters: usize,
    /// Whether to compute smooth normals
    pub smooth_normals: bool,
    /// Normal smoothing iterations
    pub normal_smoothing_iterations: usize,
}

impl Default for MeshExtractionConfig {
    fn default() -> Self {
        Self {
            tsdf_config: TsdfConfig {
                voxel_size: 0.01,      // 1cm
                sdf_trunc: 0.04,       // 4cm truncation
                min_bound: Vec3::new(-1.0, -1.0, -1.0),
                max_bound: Vec3::new(1.0, 1.0, 1.0),
                max_weight: 100.0,
                integration_weight: 1.0,
            },
            min_cluster_size: 1000,
            num_largest_clusters: 1,
            smooth_normals: true,
            normal_smoothing_iterations: 3,
        }
    }
}

/// Mesh extractor combining TSDF and Marching Cubes
pub struct MeshExtractor {
    config: MeshExtractionConfig,
    volume: TsdfVolume,
}

impl MeshExtractor {
    pub fn new(config: MeshExtractionConfig) -> Self {
        Self {
            config: config.clone(),
            volume: TsdfVolume::new(config.tsdf_config),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MeshExtractionConfig::default())
    }

    /// Create volume centered at a point with given size
    pub fn centered(center: Vec3, size_meters: f32, voxel_size: f32) -> Self {
        let half = size_meters / 2.0;
        let config = TsdfConfig {
            voxel_size,
            sdf_trunc: voxel_size * 4.0,
            min_bound: center - Vec3::splat(half),
            max_bound: center + Vec3::splat(half),
            max_weight: 100.0,
            integration_weight: 1.0,
        };

        Self {
            config: MeshExtractionConfig {
                tsdf_config: config,
                ..Default::default()
            },
            volume: TsdfVolume::new(config),
        }
    }

    /// Integrate a depth frame into the TSDF volume
    pub fn integrate_frame(
        &mut self,
        depth: &[f32],
        color: Option<&[[u8; 3]]>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Mat4,
    ) {
        self.volume.integrate(
            depth,
            color,
            width,
            height,
            intrinsics,
            extrinsics,
        );
    }

    /// Integrate depth rendered from Gaussians
    pub fn integrate_from_gaussians<F>(
        &mut self,
        render_depth_fn: F,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Mat4,
    ) where
        F: Fn(usize) -> f32,
    {
        self.volume.integrate_from_gaussians(
            render_depth_fn,
            None,
            width,
            height,
            intrinsics,
            extrinsics,
        );
    }

    /// Extract mesh from the TSDF volume
    pub fn extract(&self) -> Mesh {
        extract_mesh_from_tsdf(&self.volume)
    }

    /// Extract and post-process mesh
    pub fn extract_with_postprocessing(&self) -> Mesh {
        let mut mesh = self.extract();

        // Post-processing
        if self.config.min_cluster_size > 0 || self.config.num_largest_clusters > 0 {
            mesh = self.filter_clusters(mesh);
        }

        if self.config.smooth_normals && self.config.normal_smoothing_iterations > 0 {
            mesh = self.smooth_normals(mesh);
        }

        mesh
    }

    /// Filter out small clusters (floaters)
    fn filter_clusters(&self, mut mesh: Mesh) -> Mesh {
        if mesh.triangles.is_empty() {
            return mesh;
        }

        // Build adjacency
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        for tri in &mesh.triangles {
            for &idx in &tri.indices {
                adj.entry(idx).or_insert_with(HashSet::new);
            }
        }

        for tri in &mesh.triangles {
            let [a, b, c] = tri.indices;
            adj.get_mut(&a).unwrap().insert(b);
            adj.get_mut(&a).unwrap().insert(c);
            adj.get_mut(&b).unwrap().insert(a);
            adj.get_mut(&b).unwrap().insert(c);
            adj.get_mut(&c).unwrap().insert(a);
            adj.get_mut(&c).unwrap().insert(b);
        }

        // Find connected components
        let mut visited = vec![false; mesh.vertices.len()];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for start in 0..mesh.vertices.len() {
            if visited[start] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;

            while let Some(v) = stack.pop() {
                cluster.push(v);

                if let Some(neighbors) = adj.get(&v) {
                    for &n in neighbors {
                        if !visited[n] {
                            visited[n] = true;
                            stack.push(n);
                        }
                    }
                }
            }

            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }

        // Sort clusters by size
        clusters.sort_by(|a, b| b.len().cmp(&a.len()));

        // Keep only largest clusters
        let mut keep_vertices: HashSet<usize> = HashSet::new();

        for (i, cluster) in clusters.iter().enumerate() {
            if i < self.config.num_largest_clusters && cluster.len() >= self.config.min_cluster_size {
                for &v in cluster {
                    keep_vertices.insert(v);
                }
            }
        }

        // Rebuild mesh with kept vertices
        let mut new_vertices = Vec::new();
        let mut index_map: HashMap<usize, usize> = HashMap::new();

        for (new_idx, &old_idx) in keep_vertices.iter().enumerate() {
            index_map.insert(old_idx, new_idx);
            new_vertices.push(mesh.vertices[old_idx].clone());
        }

        let new_triangles: Vec<MeshTriangle> = mesh
            .triangles
            .iter()
            .filter_map(|tri| {
                let a = index_map.get(&tri.indices[0])?;
                let b = index_map.get(&tri.indices[1])?;
                let c = index_map.get(&tri.indices[2])?;
                Some(MeshTriangle {
                    indices: [*a, *b, *c],
                })
            })
            .collect();

        mesh.vertices = new_vertices;
        mesh.triangles = new_triangles;

        mesh
    }

    /// Smooth normals using neighbor averaging
    fn smooth_normals(&self, mut mesh: Mesh) -> Mesh {
        if mesh.triangles.is_empty() {
            return mesh;
        }

        // Build vertex to triangles mapping
        let mut vert_tris: HashMap<usize, Vec<usize>> = HashMap::new();
        for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
            for &v in &tri.indices {
                vert_tris.entry(v).or_insert_with(Vec::new).push(tri_idx);
            }
        }

        for _ in 0..self.config.normal_smoothing_iterations {
            let mut new_normals = Vec::new();

            for (i, vert) in mesh.vertices.iter().enumerate() {
                if let Some(tris) = vert_tris.get(&i) {
                    let mut avg_normal = Vec3::ZERO;

                    for &tri_idx in tris {
                        let tri = &mesh.triangles[tri_idx];
                        let v0 = mesh.vertices[tri.indices[0]].position;
                        let v1 = mesh.vertices[tri.indices[1]].position;
                        let v2 = mesh.vertices[tri.indices[2]].position;

                        let e1 = v1 - v0;
                        let e2 = v2 - v0;
                        avg_normal += e1.cross(e2);
                    }

                    avg_normal = avg_normal.normalize();

                    // Blend with original
                    let smoothed = (vert.normal + avg_normal).normalize();
                    new_normals.push(smoothed);
                } else {
                    new_normals.push(vert.normal);
                }
            }

            // Apply smoothed normals
            for (i, normal) in new_normals.into_iter().enumerate() {
                mesh.vertices[i].normal = normal;
            }
        }

        mesh
    }

    /// Get current volume
    pub fn volume(&self) -> &TsdfVolume {
        &self.volume
    }

    /// Get mutable volume
    pub fn volume_mut(&mut self) -> &mut TsdfVolume {
        &mut self.volume
    }

    /// Clear the volume
    pub fn clear(&mut self) {
        self.volume.clear();
    }

    /// Get number of integrated frames
    pub fn frame_count(&self) -> usize {
        self.volume.frame_count()
    }
}

// Re-export for convenience
pub use crate::fusion::marching_cubes::extract_mesh_from_tsdf;
