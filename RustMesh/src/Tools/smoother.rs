//! # Smoother - Mesh smoothing algorithms
//!
//! Provides Laplacian and Jacobi smoothing for mesh optimization.

use crate::handles::VertexHandle;
use crate::RustMesh;
use crate::Vec3;

/// Smoothing configuration
#[derive(Debug, Clone)]
pub struct SmootherConfig {
    /// Number of smoothing iterations
    pub iterations: usize,
    /// Smoothing strength (0.0 - 1.0)
    pub strength: f32,
    /// Use uniform weight (false = use cotangent weights)
    pub uniform: bool,
    /// Boundary vertices are fixed
    pub fixed_boundary: bool,
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        }
    }
}

/// Laplace smoothing result
#[derive(Debug)]
pub struct SmoothResult {
    pub iterations: usize,
    pub max_displacement: f32,
}

/// Compute Laplacian of a vertex (uniform weights)
fn compute_laplacian_uniform(mesh: &RustMesh, vh: VertexHandle) -> Option<Vec3> {
    let current = mesh.point(vh)?;
    let (neighbor_sum, neighbor_count) = accumulate_neighbor_positions(mesh, vh, None);

    if neighbor_count == 0 {
        return None;
    }

    let avg = neighbor_sum / neighbor_count as f32;
    Some(avg - current)
}

fn accumulate_neighbor_positions(
    mesh: &RustMesh,
    vh: VertexHandle,
    cached_positions: Option<&[Vec3]>,
) -> (Vec3, usize) {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;

    if let Some(vv) = mesh.vertex_vertices(vh) {
        for neighbor in vv {
            let position = cached_positions
                .and_then(|positions| positions.get(neighbor.idx_usize()).copied())
                .or_else(|| mesh.point(neighbor));

            if let Some(point) = position {
                sum += point;
                count += 1;
            }
        }
    }

    (sum, count)
}

/// Check if vertex is on boundary
fn is_boundary_vertex(mesh: &RustMesh, vh: VertexHandle) -> bool {
    if let Some(heh) = mesh.halfedge_handle(vh) {
        return mesh.is_boundary(heh);
    }
    false
}

/// Laplace smoothing (explicit)
///
/// Moves each vertex towards the average of its neighbors.
///
/// Formula: p' = p + lambda * (p_avg - p)
///
/// where lambda is the smoothing strength (0-1)
pub fn laplace_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    // Cache mesh topology information once, mirroring OpenMesh's property-based examples.
    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };
    let mut current_positions = vec![Vec3::ZERO; n_vertices];
    let mut next_positions = vec![Vec3::ZERO; n_vertices];
    let mut max_displacement = 0.0f32;

    for _ in 0..config.iterations {
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Vec3::ZERO);
            current_positions[i] = point;
            next_positions[i] = point;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            let (neighbor_sum, neighbor_count) =
                accumulate_neighbor_positions(mesh, vh, Some(&current_positions));
            if neighbor_count == 0 {
                continue;
            }

            let avg = neighbor_sum / neighbor_count as f32;
            let displacement = (avg - current_positions[i]) * config.strength;
            max_displacement = max_displacement.max(displacement.length());
            next_positions[i] = current_positions[i] + displacement;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

/// Tangential smoothing
///
/// Similar to Laplacian but constrains movement to be tangential
/// to preserve volume better.
pub fn tangential_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };
    let mut current_positions = vec![Vec3::ZERO; n_vertices];
    let mut next_positions = vec![Vec3::ZERO; n_vertices];
    let mut max_displacement = 0.0f32;

    for _ in 0..config.iterations {
        let mut centroid_sum = Vec3::ZERO;
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Vec3::ZERO);
            current_positions[i] = point;
            next_positions[i] = point;
            centroid_sum += point;
        }
        let centroid = centroid_sum / n_vertices as f32;

        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            let (neighbor_sum, neighbor_count) =
                accumulate_neighbor_positions(mesh, vh, Some(&current_positions));
            if neighbor_count == 0 {
                continue;
            }

            let laplacian = neighbor_sum / neighbor_count as f32 - current_positions[i];
            let normal = (current_positions[i] - centroid).normalize_or_zero();
            let tangential = laplacian - normal * laplacian.dot(normal);
            let displacement = tangential * config.strength;
            max_displacement = max_displacement.max(displacement.length());
            next_positions[i] = current_positions[i] + displacement;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

/// Compute cotangent weights for Laplacian smoothing (more accurate)
///
/// Note: This is computationally expensive but more accurate for
/// non-uniform meshes.
pub fn cotangent_weight_laplacian(_mesh: &RustMesh, _vh: VertexHandle) -> Option<Vec3> {
    // TODO: Implement cotangent weights
    // This requires computing angles for each adjacent triangle
    // Formula: w_ij = (cot(alpha) + cot(beta)) / 2
    // Laplacian = sum(w_ij * (p_j - p_i)) / sum(w_ij)
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;

    #[test]
    fn test_laplace_smooth() {
        let mut mesh = generate_cube();
        let original_count = mesh.n_vertices();

        let config = SmootherConfig {
            iterations: 2,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        };

        let result = laplace_smooth(&mut mesh, config);

        println!(
            "Laplace smooth: {} iterations, {} vertices",
            result.iterations,
            mesh.n_vertices()
        );

        // Count should not change
        assert_eq!(mesh.n_vertices(), original_count);
    }

    #[test]
    fn test_tangential_smooth() {
        let mut mesh = generate_cube();
        let original_count = mesh.n_vertices();

        let config = SmootherConfig {
            iterations: 2,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        };

        let result = tangential_smooth(&mut mesh, config);

        println!(
            "Tangential smooth: {} iterations, {} vertices",
            result.iterations,
            mesh.n_vertices()
        );

        assert_eq!(mesh.n_vertices(), original_count);
    }
}
