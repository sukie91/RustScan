//! # Smoother - Mesh smoothing algorithms
//!
//! Provides Laplacian and Jacobi smoothing for mesh optimization.

use crate::RustMesh;
use crate::handles::VertexHandle;
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
    let mut neighbors = Vec::new();
    let mut count = 0;
    let max_iter = 64; // Prevent infinite loops
    
    // Get all neighboring vertices
    if let Some(vv) = mesh.vertex_vertices(vh) {
        for neighbor in vv {
            count += 1;
            if count > max_iter {
                break;
            }
            if let Some(p) = mesh.point(neighbor) {
                neighbors.push(p);
            }
        }
    }
    
    if neighbors.is_empty() {
        return None;
    }
    
    // Compute average position of neighbors
    let sum: Vec3 = neighbors.iter().fold(Vec3::ZERO, |acc, p| acc + *p);
    let avg = sum / neighbors.len() as f32;
    
    // Laplacian = average - current
    let current = mesh.point(vh)?;
    Some(avg - current)
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
    
    // Pre-allocate once outside the loop
    let mut vhs: Vec<VertexHandle> = Vec::with_capacity(n_vertices);
    let mut displacements: Vec<Vec3> = Vec::with_capacity(n_vertices);
    
    for _ in 0..config.iterations {
        vhs.clear();
        
        // Collect vertex handles
        for vh in mesh.vertices() {
            vhs.push(vh);
        }
        
        displacements.clear();
        displacements.resize(n_vertices, Vec3::ZERO);
        
        // Compute displacements for all vertices
        for (i, &vh) in vhs.iter().enumerate() {
            // Skip boundary vertices if fixed_boundary is true
            if config.fixed_boundary && is_boundary_vertex(mesh, vh) {
                continue;
            }
            
            if let Some(laplacian) = compute_laplacian_uniform(mesh, vh) {
                displacements[i] = laplacian * config.strength;
            }
        }
        
        // Apply displacements
        for (i, &vh) in vhs.iter().enumerate() {
            let disp = displacements[i];
            if disp != Vec3::ZERO {
                if let Some(p) = mesh.point(vh) {
                    mesh.set_point(vh, p + disp);
                }
            }
        }
    }
    
    SmoothResult {
        iterations: config.iterations,
        max_displacement: 0.0,
    }
}

/// Tangential smoothing
/// 
/// Similar to Laplacian but constrains movement to be tangential
/// to preserve volume better.
pub fn tangential_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    
    // First compute average position (centroid)
    let mut sum = Vec3::ZERO;
    for vh in mesh.vertices() {
        if let Some(p) = mesh.point(vh) {
            sum = sum + p;
        }
    }
    let centroid = sum / n_vertices as f32;
    
    // Pre-allocate vectors
    let mut vhs: Vec<VertexHandle> = Vec::with_capacity(n_vertices);
    let mut points: Vec<Option<Vec3>> = Vec::with_capacity(n_vertices);
    let mut normals: Vec<Vec3> = Vec::with_capacity(n_vertices);
    
    for _ in 0..config.iterations {
        vhs.clear();
        points.clear();
        
        // Collect vertex handles and points
        for vh in mesh.vertices() {
            vhs.push(vh);
            points.push(mesh.point(vh));
        }
        
        // Compute normals
        normals.clear();
        normals.resize(n_vertices, Vec3::ZERO);
        
        for (i, &vh) in vhs.iter().enumerate() {
            if config.fixed_boundary && is_boundary_vertex(mesh, vh) {
                continue;
            }
            
            if let Some(p) = points[i] {
                if let Some(lap) = compute_laplacian_uniform(mesh, vh) {
                    let normal = (p - centroid).normalize_or_zero();
                    let dot = lap.dot(normal);
                    let tangent = lap - normal * dot;
                    normals[i] = tangent * config.strength;
                }
            }
        }
        
        // Apply displacements
        for (i, &vh) in vhs.iter().enumerate() {
            let normal = normals[i];
            if normal != Vec3::ZERO {
                if let Some(p) = mesh.point(vh) {
                    mesh.set_point(vh, p + normal);
                }
            }
        }
    }
    
    SmoothResult {
        iterations: config.iterations,
        max_displacement: 0.0,
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
        
        println!("Laplace smooth: {} iterations, {} vertices", 
            result.iterations, mesh.n_vertices());
        
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
        
        println!("Tangential smooth: {} iterations, {} vertices", 
            result.iterations, mesh.n_vertices());
        
        assert_eq!(mesh.n_vertices(), original_count);
    }
}
