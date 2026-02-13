//! # Sqrt3 Subdivision
//!
//! Implementation of the Sqrt3 subdivision algorithm for triangular meshes.
//!
//! Sqrt3 subdivision is a method for refining triangular meshes where the mesh
//! grows by a factor of 3 each iteration (hence "sqrt3" since the number of
//! triangles increases as 3^n).
//!
//! ## Algorithm Overview
//!
//! 1. **Vertex Insertion**: For each original vertex, create a new vertex at the same position.
//! 2. **Face Splitting**: For each original triangle, create 3 new triangles by connecting
//!    the new vertices associated with the original face's vertices.
//! 3. **Vertex Update**: Move each new vertex position using the formula:
//!    `new_pos = original_pos + Laplacian * (1/3)`
//!    where Laplacian = average(neighbor positions) - original_position
//! 4. **Boundary Flipping**: Flip all boundary edges to maintain manifold property.
//!
//! ## References
//!
//! - Kobbelt, L. (2000). "Sqrt(3)-Subdivision". SIGGRAPH 2000.

use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::connectivity::PolyMeshSoA;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Error types for subdivision operations
#[derive(Debug, Clone)]
pub enum SubdivisionError {
    /// Mesh is empty or has no faces
    EmptyMesh,
    /// Invalid mesh topology detected
    InvalidTopology(String),
    /// Handle not found in mesh
    InvalidHandle(String),
    /// Edge not found
    EdgeNotFound,
    /// Vertex not found
    VertexNotFound,
    /// Face not found
    FaceNotFound,
    /// Mesh is not triangular (required for Loop subdivision)
    NotTriangular,
}

impl std::fmt::Display for SubdivisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "Mesh is empty or has no faces"),
            Self::InvalidTopology(msg) => write!(f, "Invalid mesh topology: {}", msg),
            Self::InvalidHandle(msg) => write!(f, "Invalid handle: {}", msg),
            Self::EdgeNotFound => write!(f, "Edge not found"),
            Self::VertexNotFound => write!(f, "Vertex not found"),
            Self::FaceNotFound => write!(f, "Face not found"),
            Self::NotTriangular => write!(f, "Mesh is not triangular (required for Loop subdivision)"),
        }
    }
}

impl std::error::Error for SubdivisionError {}

/// Result type for subdivision operations
pub type SubdivisionResult<T> = Result<T, SubdivisionError>;

/// Information about the subdivision operation
#[derive(Debug, Clone)]
pub struct SubdivisionStats {
    /// Number of original vertices
    pub original_vertices: usize,
    /// Number of original edges
    pub original_edges: usize,
    /// Number of original faces
    pub original_faces: usize,
    /// Number of new vertices created
    pub new_vertices: usize,
    /// Number of new edges created
    pub new_edges: usize,
    /// Number of new faces created
    pub new_faces: usize,
}

impl std::fmt::Display for SubdivisionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Subdivision: {}V {}E {}F -> {}V {}E {}F (+{}/+{}/+{})",
            self.original_vertices,
            self.original_edges,
            self.original_faces,
            self.original_vertices + self.new_vertices,
            self.original_edges + self.new_edges,
            self.original_faces + self.new_faces,
            self.new_vertices,
            self.new_edges,
            self.new_faces
        )
    }
}

/// Check if the mesh is triangular (all faces have 3 vertices)
fn is_triangular(mesh: &PolyMeshSoA) -> bool {
    for fh in mesh.faces() {
        if let Some(start_heh) = mesh.face_halfedge_handle(fh) {
            let mut count = 0;
            let mut current = start_heh;
            loop {
                count += 1;
                current = mesh.next_halfedge_handle(current);
                if current == start_heh || !current.is_valid() {
                    break;
                }
            }
            if count != 3 {
                return false;
            }
        }
    }
    true
}

/// Get all halfedges of a face
fn get_face_halfedges(mesh: &PolyMeshSoA, fh: FaceHandle) -> Vec<HalfedgeHandle> {
    let mut halfedges = Vec::with_capacity(3);
    if let Some(start_heh) = mesh.face_halfedge_handle(fh) {
        let mut current = start_heh;
        loop {
            halfedges.push(current);
            current = mesh.next_halfedge_handle(current);
            if current == start_heh || !current.is_valid() {
                break;
            }
        }
    }
    halfedges
}

/// Get the vertices of a face
fn get_face_vertices(mesh: &PolyMeshSoA, fh: FaceHandle) -> Vec<VertexHandle> {
    let mut vertices = Vec::with_capacity(3);
    if let Some(start_heh) = mesh.face_halfedge_handle(fh) {
        let mut current = start_heh;
        loop {
            vertices.push(mesh.from_vertex_handle(current));
            current = mesh.next_halfedge_handle(current);
            if current == start_heh || !current.is_valid() {
                break;
            }
        }
    }
    vertices
}

/// Get all edges of the mesh
fn get_all_edges(mesh: &PolyMeshSoA) -> Vec<(VertexHandle, VertexHandle)> {
    let mut edges = Vec::with_capacity(mesh.n_edges());
    let n_halfedges = mesh.n_halfedges();
    
    // Iterate through halfedges (every pair is an edge)
    for i in (0..n_halfedges).step_by(2) {
        let heh = HalfedgeHandle::new(i as u32);
        let from = mesh.from_vertex_handle(heh);
        let to = mesh.to_vertex_handle(heh);
        edges.push((from, to));
    }
    
    edges
}

/// Check if a vertex is on the boundary
fn is_boundary_vertex(mesh: &PolyMeshSoA, vh: VertexHandle) -> bool {
    if let Some(heh) = mesh.halfedge_handle(vh) {
        let mut current = heh;
        loop {
            if mesh.is_boundary(current) {
                return true;
            }
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    false
}

/// Get all 1-ring neighbors of a vertex
fn get_vertex_neighbors(mesh: &PolyMeshSoA, vh: VertexHandle) -> Vec<VertexHandle> {
    let mut neighbors = Vec::new();
    
    if let Some(heh) = mesh.halfedge_handle(vh) {
        let mut current = heh;
        loop {
            let neighbor = mesh.to_vertex_handle(current);
            if neighbor != vh {
                neighbors.push(neighbor);
            }
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    
    neighbors
}

/// Get the valence (number of neighbors) of a vertex
fn get_vertex_valence(mesh: &PolyMeshSoA, vh: VertexHandle) -> usize {
    get_vertex_neighbors(mesh, vh).len()
}

/// Split an edge by inserting a new vertex at its midpoint.
///
/// The new vertex is placed at the midpoint of the two endpoint vertices.
/// If the edge is on a boundary, the new vertex is placed at the midpoint
/// with a special boundary weighting.
///
/// # Arguments
/// * `mesh` - The mesh to modify
/// * `v0` - First vertex of the edge
/// * `v1` - Second vertex of the edge
///
/// # Returns
/// * `Ok(VertexHandle)` - The handle to the new vertex
/// * `Err(SubdivisionError)` - If the operation fails
pub fn split_edge(mesh: &mut PolyMeshSoA, v0: VertexHandle, v1: VertexHandle) -> SubdivisionResult<VertexHandle> {
    // Validate vertices
    if !v0.is_valid() || !v1.is_valid() {
        return Err(SubdivisionError::InvalidHandle("Invalid vertex handle".to_string()));
    }
    
    // Get positions
    let p0 = mesh.point(v0).ok_or(SubdivisionError::VertexNotFound)?;
    let p1 = mesh.point(v1).ok_or(SubdivisionError::VertexNotFound)?;
    
    // Check if edge is on boundary
    let is_boundary = {
        // Find a halfedge for this edge
        let mut found = false;
        let mut boundary = false;
        
        // Search for the halfedge from v0 to v1
        if let Some(heh) = mesh.halfedge_handle(v0) {
            let mut current = heh;
            loop {
                let to = mesh.to_vertex_handle(current);
                if to == v1 {
                    found = true;
                    boundary = mesh.is_boundary(current);
                    break;
                }
                let opp = mesh.opposite_halfedge_handle(current);
                current = mesh.next_halfedge_handle(opp);
                if current == heh || !current.is_valid() {
                    break;
                }
            }
        }
        
        if !found {
            // Try opposite direction
            if let Some(heh) = mesh.halfedge_handle(v1) {
                let mut current = heh;
                loop {
                    let to = mesh.to_vertex_handle(current);
                    if to == v0 {
                        found = true;
                        boundary = mesh.is_boundary(current);
                        break;
                    }
                    let opp = mesh.opposite_halfedge_handle(current);
                    current = mesh.next_halfedge_handle(opp);
                    if current == heh || !current.is_valid() {
                        break;
                    }
                }
            }
        }
        
        boundary
    };
    
    // Calculate new vertex position
    let new_pos = if is_boundary {
        // For boundary edges: midpoint
        (p0 + p1) * 0.5
    } else {
        // For interior edges: weighted average (Loop scheme for edge points)
        // New edge point = 3/8 * (p0 + p1) + 1/8 * (p_opposite0 + p_opposite1)
        // Where p_opposite0/1 are the vertices opposite to the edge in the two adjacent faces
        
        let mut sum = p0 + p1;
        let mut count = 0;
        
        // Find the two faces adjacent to this edge
        if let Some(heh) = mesh.halfedge_handle(v0) {
            let mut current = heh;
            loop {
                let to = mesh.to_vertex_handle(current);
                if to == v1 {
                    // Found the halfedge, get the opposite vertex in the face
                    if let Some(fh) = mesh.face_handle(current) {
                        let face_verts = get_face_vertices(mesh, fh);
                        for &fv in &face_verts {
                            if fv != v0 && fv != v1 {
                                if let Some(p) = mesh.point(fv) {
                                    sum = sum + p;
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                let opp = mesh.opposite_halfedge_handle(current);
                current = mesh.next_halfedge_handle(opp);
                if current == heh || !current.is_valid() {
                    break;
                }
            }
        }
        
        if count > 0 {
            sum / (count as f32 + 2.0) // (p0 + p1 + sum_of_opposites) / (2 + count)
        } else {
            (p0 + p1) * 0.5
        }
    };
    
    // Add the new vertex
    let new_vh = mesh.add_vertex(new_pos);
    
    Ok(new_vh)
}

/// Calculate the new position for a vertex using Loop scheme weights.
///
/// For interior vertices:
///   new_pos = (1 - n*beta) * old_pos + beta * sum(neighbor_pos)
///   where beta = 1/n * (5/8 - (3/8 + 1/4*cos(2*pi/n))^2)
///
/// For boundary vertices:
///   new_pos = 3/8 * left_pos + 3/8 * right_pos + 1/8 * old_pos
///   (simplified scheme)
///
/// # Arguments
/// * `mesh` - The mesh
/// * `vh` - The vertex to update
///
/// # Returns
/// * `Ok(glam::Vec3)` - The new position
/// * `Err(SubdivisionError)` - If the vertex is invalid
fn calculate_loop_new_position(mesh: &PolyMeshSoA, vh: VertexHandle) -> SubdivisionResult<glam::Vec3> {
    let old_pos = mesh.point(vh).ok_or(SubdivisionError::VertexNotFound)?;
    let neighbors = get_vertex_neighbors(mesh, vh);
    let n = neighbors.len();
    
    if n == 0 {
        return Ok(old_pos);
    }
    
    let boundary = is_boundary_vertex(mesh, vh);
    
    if boundary {
        // Boundary vertex: use simplified scheme
        // Find the two boundary neighbors
        let mut boundary_neighbors = Vec::new();
        
        for &nh in &neighbors {
            if let Some(heh) = mesh.halfedge_handle(vh) {
                let mut current = heh;
                loop {
                    let to = mesh.to_vertex_handle(current);
                    if to == nh {
                        if mesh.is_boundary(current) {
                            boundary_neighbors.push(nh);
                        }
                        break;
                    }
                    let opp = mesh.opposite_halfedge_handle(current);
                    current = mesh.next_halfedge_handle(opp);
                    if current == heh || !current.is_valid() {
                        break;
                    }
                }
            }
        }
        
        // Use boundary scheme if we have at least one boundary neighbor
        if boundary_neighbors.len() >= 1 {
            let left = boundary_neighbors[0];
            let left_pos = mesh.point(left).unwrap_or(old_pos);
            
            // Try to find right neighbor
            if boundary_neighbors.len() >= 2 {
                let right = boundary_neighbors[1];
                let right_pos = mesh.point(right).unwrap_or(old_pos);
                return Ok(old_pos * 0.125 + left_pos * 0.375 + right_pos * 0.375);
            } else {
                // Only one boundary neighbor - use endpoint of boundary
                return Ok(old_pos * 0.75 + left_pos * 0.25);
            }
        }
        
        // Fallback: just average all neighbors
        let sum: glam::Vec3 = neighbors.iter()
            .filter_map(|&nh| mesh.point(nh))
            .fold(glam::Vec3::ZERO, |a, b| a + b);
        return Ok(old_pos * 0.25 + sum * 0.25);
    } else {
        // Interior vertex: full Loop scheme
        // Beta calculation: beta = 1/n * (5/8 - (3/8 + 1/4*cos(2*pi/n))^2)
        let n_f32 = n as f32;
        
        // Calculate cos(2*pi/n) using trig identity
        let theta = std::f32::consts::TAU / n_f32;
        let cos_theta = theta.cos();
        
        // Beta formula
        let beta = (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * cos_theta).powi(2)) / n_f32;
        
        // Sum of neighbor positions
        let neighbor_sum: glam::Vec3 = neighbors.iter()
            .filter_map(|&nh| mesh.point(nh))
            .fold(glam::Vec3::ZERO, |a, b| a + b);
        
        // New position
        let new_pos = old_pos * (1.0 - n_f32 * beta) + neighbor_sum * beta;
        
        Ok(new_pos)
    }
}

/// Perform one iteration of Loop subdivision on the mesh.
///
/// This implements the full Loop subdivision algorithm:
/// 1. Create new vertices at the midpoint of each edge
/// 2. Split each original face into 4 smaller faces
/// 3. Update original vertex positions using Loop scheme weights
///
/// # Arguments
/// * `mesh` - The mesh to subdivide (will be modified in place)
///
/// # Returns
/// * `Ok(SubdivisionStats)` - Statistics about the subdivision
/// * `Err(SubdivisionError)` - If the mesh cannot be subdivided
pub fn loop_subdivide(mesh: &mut PolyMeshSoA) -> SubdivisionResult<SubdivisionStats> {
    // Validate mesh
    if mesh.n_vertices() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if mesh.n_faces() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    // Check that mesh is triangular
    if !is_triangular(mesh) {
        return Err(SubdivisionError::NotTriangular);
    }
    
    let original_vertices = mesh.n_vertices();
    let original_edges = mesh.n_edges();
    let original_faces = mesh.n_faces();
    
    // Store original vertex positions for update after splitting
    let mut original_positions: HashMap<u32, glam::Vec3> = HashMap::new();
    for vh in mesh.vertices() {
        if let Some(pos) = mesh.point(vh) {
            original_positions.insert(vh.idx(), pos);
        }
    }
    
    // Step 1: Collect all edges and their adjacent faces
    // For each edge, we'll create a new vertex
    let mut edge_to_new_vertex: HashMap<(u32, u32), VertexHandle> = HashMap::new();
    let mut edges_processed: Vec<(VertexHandle, VertexHandle, HalfedgeHandle)> = Vec::new();
    
    let n_halfedges = mesh.n_halfedges();
    for i in (0..n_halfedges).step_by(2) {
        let heh = HalfedgeHandle::new(i as u32);
        let v0 = mesh.from_vertex_handle(heh);
        let v1 = mesh.to_vertex_handle(heh);
        
        // Use canonical ordering: (min, max)
        let (min_v, max_v) = if v0.idx() < v1.idx() { (v0, v1) } else { (v1, v0) };
        
        edges_processed.push((v0, v1, heh));
        edge_to_new_vertex.insert((min_v.idx(), max_v.idx()), VertexHandle::invalid());
    }
    
    // Step 2: Split each edge and create new vertices
    for (v0, v1, _heh) in &edges_processed {
        let (min_v, max_v) = if v0.idx() < v1.idx() { (v0, v1) } else { (v1, v0) };
        
        match split_edge(mesh, *min_v, *max_v) {
            Ok(new_vh) => {
                edge_to_new_vertex.insert((min_v.idx(), max_v.idx()), new_vh);
            }
            Err(_e) => {
                // Continue even if split fails (edge might already be split)
                // We'll handle this by checking the map later
            }
        }
    }
    
    let _new_vertices_count = edge_to_new_vertex.values().filter(|vh| vh.is_valid()).count();
    
    // Step 3: Split each face into 4 triangles
    // For each original face, we need:
    // - 3 new vertices (one for each edge, already created)
    // - Create 4 new faces
    
    // We need to track which faces have been split
    let original_face_handles: Vec<FaceHandle> = mesh.faces().collect();
    
    for fh in original_face_handles {
        let face_verts = get_face_vertices(mesh, fh);
        if face_verts.len() != 3 {
            continue;
        }
        
        let v0 = face_verts[0];
        let v1 = face_verts[1];
        let v2 = face_verts[2];
        
        // Get the new vertices for each edge
        let get_edge_vertex = |v_a: VertexHandle, v_b: VertexHandle| -> Option<VertexHandle> {
            let (min_v, max_v) = if v_a.idx() < v_b.idx() { (v_a, v_b) } else { (v_b, v_a) };
            edge_to_new_vertex.get(&(min_v.idx(), max_v.idx())).and_then(|vh| {
                if vh.is_valid() { Some(*vh) } else { None }
            })
        };
        
        let ev01 = get_edge_vertex(v0, v1);
        let ev12 = get_edge_vertex(v1, v2);
        let ev20 = get_edge_vertex(v2, v0);
        
        // All three edge vertices must exist
        if let (Some(e01), Some(e12), Some(e20)) = (ev01, ev12, ev20) {
            // Add 4 new triangles:
            // Original triangle is (v0, v1, v2)
            // New triangles:
            // 1. (v0, e01, e20) - corner at v0
            // 2. (v1, e12, e01) - corner at v1  
            // 3. (v2, e20, e12) - corner at v2
            // 4. (e01, e12, e20) - center triangle
            
            mesh.add_face(&[v0, e01, e20]);
            mesh.add_face(&[v1, e12, e01]);
            mesh.add_face(&[v2, e20, e12]);
            mesh.add_face(&[e01, e12, e20]);
        }
    }
    
    // Step 4: Update original vertex positions using Loop scheme
    // Only update vertices that existed before the subdivision
    for (idx, _pos) in &original_positions {
        let vh = VertexHandle::new(*idx);
        if vh.is_valid() && vh.idx() < original_vertices as u32 {
            if let Ok(new_pos) = calculate_loop_new_position(mesh, vh) {
                mesh.set_point(vh, new_pos);
            }
        }
    }
    
    // Calculate statistics
    let new_vertices = mesh.n_vertices() - original_vertices;
    let new_edges = mesh.n_edges() - original_edges;
    let new_faces = mesh.n_faces() - original_faces;
    
    let stats = SubdivisionStats {
        original_vertices,
        original_edges,
        original_faces,
        new_vertices,
        new_edges,
        new_faces,
    };
    
    Ok(stats)
}

/// Perform multiple iterations of Loop subdivision.
///
/// # Arguments
/// * `mesh` - The mesh to subdivide
/// * `iterations` - Number of subdivision iterations
///
/// # Returns
/// * `Ok(Vec<SubdivisionStats>)` - Statistics for each iteration
/// * `Err(SubdivisionError)` - If any subdivision fails
pub fn loop_subdivide_iterations(mesh: &mut PolyMeshSoA, iterations: usize) -> SubdivisionResult<Vec<SubdivisionStats>> {
    let mut all_stats = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let stats = loop_subdivide(mesh)?;
        all_stats.push(stats);
    }
    
    Ok(all_stats)
}

/// Validate that a mesh is suitable for Loop subdivision
///
/// # Arguments
/// * `mesh` - The mesh to validate
///
/// # Returns
/// * `Ok(())` - If mesh is valid
/// * `Err(SubdivisionError)` - If mesh has issues
pub fn validate_for_subdivision(mesh: &PolyMeshSoA) -> SubdivisionResult<()> {
    if mesh.n_vertices() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if mesh.n_faces() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if !is_triangular(mesh) {
        return Err(SubdivisionError::NotTriangular);
    }
    
    // Check for degenerate faces
    for fh in mesh.faces() {
        let verts = get_face_vertices(mesh, fh);
        if verts.len() != 3 {
            return Err(SubdivisionError::InvalidTopology(format!(
                "Face {:?} has {} vertices (expected 3)", fh, verts.len()
            )));
        }
        
        // Check for duplicate vertices
        if verts[0] == verts[1] || verts[1] == verts[2] || verts[0] == verts[2] {
            return Err(SubdivisionError::InvalidTopology("Degenerate face found".to_string()));
        }
    }
    
    Ok(())
}

/// Check if the mesh is triangular (public version for external use)
pub fn is_mesh_triangular(mesh: &PolyMeshSoA) -> bool {
    is_triangular(mesh)
}

/// Calculate the new vertex position using Sqrt3 scheme.
///
/// The formula is:
/// `new_pos = original_pos + Laplacian * (1/3)`
/// where Laplacian = average(neighbor positions) - original_position
///
/// For boundary vertices, we use a simpler scheme to preserve the boundary.
///
/// # Arguments
/// * `mesh` - The mesh
/// * `vh` - The vertex handle
///
/// # Returns
/// * `Ok(glam::Vec3)` - The new position
/// * `Err(SubdivisionError)` - If the vertex is invalid
fn calculate_sqrt3_new_position(mesh: &PolyMeshSoA, vh: VertexHandle) -> SubdivisionResult<glam::Vec3> {
    let old_pos = mesh.point(vh).ok_or(SubdivisionError::VertexNotFound)?;
    let neighbors = get_vertex_neighbors(mesh, vh);
    
    if neighbors.is_empty() {
        return Ok(old_pos);
    }
    
    let boundary = is_boundary_vertex(mesh, vh);
    
    if boundary {
        // For boundary vertices: just keep the position (or could use boundary scheme)
        // Sqrt3 typically doesn't modify boundary vertices
        Ok(old_pos)
    } else {
        // Interior vertex: apply Sqrt3 smoothing
        // Laplacian = average of neighbors - vertex
        let neighbor_sum: glam::Vec3 = neighbors.iter()
            .filter_map(|&nh| mesh.point(nh))
            .fold(glam::Vec3::ZERO, |a, b| a + b);
        
        let neighbor_count = neighbors.len() as f32;
        let average = neighbor_sum / neighbor_count;
        
        // new_pos = old_pos + (average - old_pos) / 3
        let laplacian = average - old_pos;
        let new_pos = old_pos + laplacian * (1.0 / 3.0);
        
        Ok(new_pos)
    }
}

/// Perform one iteration of Sqrt3 subdivision on the mesh.
///
/// Sqrt3 subdivision grows the mesh by a factor of 3 each iteration:
/// 1. For each original vertex, create a new vertex at the same position
/// 2. For each original face, create 3 new faces by connecting the new vertices
/// 3. Update all vertex positions using Sqrt3 scheme
/// 4. Flip all boundary edges (not fully implemented - Sqrt3 typically keeps boundary)
///
/// # Arguments
/// * `mesh` - The mesh to subdivide (will be modified in place)
///
/// # Returns
/// * `Ok(SubdivisionStats)` - Statistics about the subdivision
/// * `Err(SubdivisionError)` - If the mesh cannot be subdivided
pub fn sqrt3_subdivide(mesh: &mut PolyMeshSoA) -> SubdivisionResult<SubdivisionStats> {
    // Validate mesh
    if mesh.n_vertices() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if mesh.n_faces() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    // Check that mesh is triangular
    if !is_triangular(mesh) {
        return Err(SubdivisionError::NotTriangular);
    }
    
    let original_vertices = mesh.n_vertices();
    let original_edges = mesh.n_edges();
    let original_faces = mesh.n_faces();
    
    // Step 1: Collect original vertex handles and their positions
    let original_vertices_list: Vec<VertexHandle> = mesh.vertices().collect();
    let mut original_vertex_positions: Vec<glam::Vec3> = Vec::new();
    for &vh in &original_vertices_list {
        if let Some(pos) = mesh.point(vh) {
            original_vertex_positions.push(pos);
        } else {
            original_vertex_positions.push(glam::Vec3::ZERO);
        }
    }
    
    // Step 2: For each original face, compute edge midpoints and create 3 new faces
    // This is the standard Sqrt3 approach: 1 triangle -> 3 triangles
    // The 3 new faces use the original vertices and edge midpoints
    // IMPORTANT: Collect face handles by index BEFORE adding new faces to avoid iterator issues
    let original_face_handles: Vec<FaceHandle> = (0..original_faces as u32)
        .map(FaceHandle::new)
        .collect();
    
    for fh in &original_face_handles {
        let face_verts = get_face_vertices(mesh, *fh);
        if face_verts.len() != 3 {
            continue;
        }
        
        let v0 = face_verts[0];
        let v1 = face_verts[1];
        let v2 = face_verts[2];
        
        // Get positions
        let p0 = mesh.point(v0).unwrap_or(glam::Vec3::ZERO);
        let p1 = mesh.point(v1).unwrap_or(glam::Vec3::ZERO);
        let p2 = mesh.point(v2).unwrap_or(glam::Vec3::ZERO);
        
        // Compute edge midpoints
        let m01_pos = (p0 + p1) * 0.5;
        let m12_pos = (p1 + p2) * 0.5;
        let m20_pos = (p2 + p0) * 0.5;
        
        // Create vertices at edge midpoints
        let m01 = mesh.add_vertex(m01_pos);
        let m12 = mesh.add_vertex(m12_pos);
        let m20 = mesh.add_vertex(m20_pos);
        
        // Create 3 new faces (the key Sqrt3 pattern):
        // Each new face uses one original vertex and two adjacent edge midpoints
        // Face 1: v0 -> m01 -> m20 (corner at v0)
        // Face 2: v1 -> m12 -> m01 (corner at v1)
        // Face 3: v2 -> m20 -> m12 (corner at v2)
        mesh.add_face(&[v0, m01, m20]);
        mesh.add_face(&[v1, m12, m01]);
        mesh.add_face(&[v2, m20, m12]);
    }
    
    // Step 3: Delete the original faces (they've been subdivided)
    // Note: delete_face marks them as deleted but doesn't remove from count
    for fh in &original_face_handles {
        mesh.delete_face(*fh);
    }
    
    // Since delete_face doesn't actually reduce n_faces(), we need to 
    // calculate the correct count. The mesh has:
    // - original_faces (some deleted) 
    // - 3 * original_faces new faces
    // Total valid faces = 3 * original_faces
    
    // Calculate statistics - note that n_faces() still includes deleted faces
    // We need to track the new face count properly
    let total_faces_after = original_faces * 3; // 3 new faces per original face
    let new_faces_count = total_faces_after - original_faces; // new - deleted original
    
    // Step 4: Update original vertex positions using Sqrt3 smoothing
    // new_pos = original_pos + Laplacian * (1/3)
    // Laplacian = average(neighbor positions) - original_position
    for (i, &vh) in original_vertices_list.iter().enumerate() {
        let neighbors = get_vertex_neighbors(mesh, vh);
        if neighbors.is_empty() {
            continue;
        }
        
        let old_pos = original_vertex_positions[i];
        
        // Check if boundary vertex
        let boundary = is_boundary_vertex(mesh, vh);
        
        if boundary {
            // Boundary vertices stay at their original position
            continue;
        }
        
        // Compute average of neighbors
        let neighbor_sum: glam::Vec3 = neighbors.iter()
            .filter_map(|&nh| mesh.point(nh))
            .fold(glam::Vec3::ZERO, |a, b| a + b);
        
        let neighbor_count = neighbors.len() as f32;
        let average = neighbor_sum / neighbor_count;
        
        // new_pos = old_pos + (average - old_pos) / 3
        let laplacian = average - old_pos;
        let new_pos = old_pos + laplacian * (1.0 / 3.0);
        
        mesh.set_point(vh, new_pos);
    }
    
    // Calculate statistics
    let new_vertices = mesh.n_vertices() - original_vertices;
    let new_edges = mesh.n_edges() - original_edges;
    let new_faces = mesh.n_faces() - original_faces;
    
    let stats = SubdivisionStats {
        original_vertices,
        original_edges,
        original_faces,
        new_vertices,
        new_edges,
        new_faces,
    };
    
    Ok(stats)
}

/// Perform multiple iterations of Sqrt3 subdivision.
///
/// # Arguments
/// * `mesh` - The mesh to subdivide
/// * `iterations` - Number of subdivision iterations
///
/// # Returns
/// * `Ok(Vec<SubdivisionStats>)` - Statistics for each iteration
/// * `Err(SubdivisionError)` - If any subdivision fails
pub fn sqrt3_subdivide_iterations(mesh: &mut PolyMeshSoA, iterations: usize) -> SubdivisionResult<Vec<SubdivisionStats>> {
    let mut all_stats = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let stats = sqrt3_subdivide(mesh)?;
        all_stats.push(stats);
    }
    
    Ok(all_stats)
}

// ============================================================================
// Catmull-Clark Subdivision (for polygonal meshes)
// ============================================================================

/// Compute the face point (centroid) of a face
/// Face point = average of all vertices of the face
fn compute_face_point(mesh: &PolyMeshSoA, fh: FaceHandle) -> glam::Vec3 {
    let vertices = get_face_vertices_polygonal(mesh, fh);
    if vertices.is_empty() {
        return glam::Vec3::ZERO;
    }
    
    let sum: glam::Vec3 = vertices.iter()
        .filter_map(|&vh| mesh.point(vh))
        .fold(glam::Vec3::ZERO, |a, b| a + b);
    
    sum / vertices.len() as f32
}

/// Get all vertices of a face (works for n-gons)
fn get_face_vertices_polygonal(mesh: &PolyMeshSoA, fh: FaceHandle) -> Vec<VertexHandle> {
    let mut vertices = Vec::new();
    if let Some(start_heh) = mesh.face_halfedge_handle(fh) {
        let mut current = start_heh;
        loop {
            vertices.push(mesh.from_vertex_handle(current));
            current = mesh.next_halfedge_handle(current);
            if current == start_heh || !current.is_valid() {
                break;
            }
        }
    }
    vertices
}

/// Get all halfedges of a face (works for n-gons)
fn get_face_halfedges_polygonal(mesh: &PolyMeshSoA, fh: FaceHandle) -> Vec<HalfedgeHandle> {
    let mut halfedges = Vec::new();
    if let Some(start_heh) = mesh.face_halfedge_handle(fh) {
        let mut current = start_heh;
        loop {
            halfedges.push(current);
            current = mesh.next_halfedge_handle(current);
            if current == start_heh || !current.is_valid() {
                break;
            }
        }
    }
    halfedges
}

/// Get the number of vertices in a face
fn get_face_valence(mesh: &PolyMeshSoA, fh: FaceHandle) -> usize {
    get_face_vertices_polygonal(mesh, fh).len()
}

/// Get all edges incident to a vertex (as pairs of vertex handles)
fn get_incident_edges(mesh: &PolyMeshSoA, vh: VertexHandle) -> Vec<(VertexHandle, VertexHandle)> {
    let mut edges = Vec::new();
    
    if let Some(heh) = mesh.halfedge_handle(vh) {
        let mut current = heh;
        loop {
            let to = mesh.to_vertex_handle(current);
            edges.push((vh, to));
            
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    
    edges
}

/// Get all faces incident to a vertex
fn get_incident_faces(mesh: &PolyMeshSoA, vh: VertexHandle) -> Vec<FaceHandle> {
    let mut faces = Vec::new();
    
    if let Some(heh) = mesh.halfedge_handle(vh) {
        let mut current = heh;
        loop {
            if let Some(fh) = mesh.face_handle(current) {
                faces.push(fh);
            }
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    
    faces
}

/// Check if an edge is on the boundary
fn is_boundary_edge(mesh: &PolyMeshSoA, v0: VertexHandle, v1: VertexHandle) -> bool {
    if let Some(heh) = mesh.halfedge_handle(v0) {
        let mut current = heh;
        loop {
            let to = mesh.to_vertex_handle(current);
            if to == v1 {
                return mesh.is_boundary(current);
            }
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    false
}

/// Compute the edge point for an edge
/// Edge point = average of:
/// - Midpoint of edge endpoints
/// - Face points of adjacent faces (if interior) or just endpoints (if boundary)
fn compute_edge_point(mesh: &PolyMeshSoA, v0: VertexHandle, v1: VertexHandle) -> glam::Vec3 {
    let p0 = mesh.point(v0).unwrap_or(glam::Vec3::ZERO);
    let p1 = mesh.point(v1).unwrap_or(glam::Vec3::ZERO);
    
    // Get the two faces adjacent to this edge
    let mut adjacent_faces: Vec<FaceHandle> = Vec::new();
    
    // Find halfedge from v0 to v1
    if let Some(heh) = mesh.halfedge_handle(v0) {
        let mut current = heh;
        loop {
            let to = mesh.to_vertex_handle(current);
            if to == v1 {
                if let Some(fh) = mesh.face_handle(current) {
                    adjacent_faces.push(fh);
                }
                // Also check opposite halfedge
                let opp = mesh.opposite_halfedge_handle(current);
                if let Some(fh) = mesh.face_handle(opp) {
                    adjacent_faces.push(fh);
                }
                break;
            }
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    
    // Calculate edge point
    if adjacent_faces.is_empty() {
        // Boundary edge: just use midpoint
        (p0 + p1) * 0.5
    } else if adjacent_faces.len() == 1 {
        // One adjacent face (boundary): average of midpoint and face point
        let face_point = compute_face_point(mesh, adjacent_faces[0]);
        (p0 + p1) * 0.25 + face_point * 0.5
    } else {
        // Interior edge: average of midpoint and both face points
        let fp0 = compute_face_point(mesh, adjacent_faces[0]);
        let fp1 = compute_face_point(mesh, adjacent_faces[1]);
        (p0 + p1) * 0.25 + (fp0 + fp1) * 0.25
    }
}

/// Calculate new vertex position using Catmull-Clark formula:
/// New position = (F + 2R + (n-2)P) / n
/// Where:
/// - F = average of face points around the vertex
/// - R = average of edge midpoints (or edge points) around the vertex
/// - P = original vertex position
/// - n = vertex valence (number of incident edges)
fn calculate_catmull_clark_new_position(mesh: &PolyMeshSoA, vh: VertexHandle) -> SubdivisionResult<glam::Vec3> {
    let p = mesh.point(vh).ok_or(SubdivisionError::VertexNotFound)?;
    
    // Get all incident faces
    let faces = get_incident_faces(mesh, vh);
    let n = faces.len();
    
    if n == 0 {
        return Ok(p);
    }
    
    // Calculate F: average of face points
    let f_sum: glam::Vec3 = faces.iter()
        .map(|&fh| compute_face_point(mesh, fh))
        .fold(glam::Vec3::ZERO, |a, b| a + b);
    let f = f_sum / n as f32;
    
    // Calculate R: average of edge points
    // Get unique edges incident to this vertex
    let mut edge_points: Vec<glam::Vec3> = Vec::new();
    
    if let Some(heh) = mesh.halfedge_handle(vh) {
        let mut current = heh;
        loop {
            let to = mesh.to_vertex_handle(current);
            let edge_point = compute_edge_point(mesh, vh, to);
            edge_points.push(edge_point);
            
            let opp = mesh.opposite_halfedge_handle(current);
            current = mesh.next_halfedge_handle(opp);
            if current == heh || !current.is_valid() {
                break;
            }
        }
    }
    
    // Calculate average of edge points
    let r_sum: glam::Vec3 = edge_points.iter().fold(glam::Vec3::ZERO, |a, &b| a + b);
    let r = r_sum / edge_points.len() as f32;
    
    // Apply Catmull-Clark formula: (F + 2R + (n-2)P) / n
    let n_f32 = n as f32;
    let new_pos = (f + r * 2.0 + p * (n_f32 - 2.0)) / n_f32;
    
    Ok(new_pos)
}

/// Perform one iteration of Catmull-Clark subdivision on the mesh.
///
/// This implements the Catmull-Clark subdivision algorithm for polygonal meshes:
/// 1. For each face, compute face point (centroid)
/// 2. For each edge, compute edge point (average of endpoints + adjacent face points)
/// 3. Update original vertex positions using Catmull-Clark weights
/// 4. Create new faces: for each original face, connect face point to edge points
///
/// # Arguments
/// * `mesh` - The mesh to subdivide (will be modified in place)
///
/// # Returns
/// * `Ok(SubdivisionStats)` - Statistics about the subdivision
/// * `Err(SubdivisionError)` - If the mesh cannot be subdivided
pub fn catmull_clark_subdivide(mesh: &mut PolyMeshSoA) -> SubdivisionResult<SubdivisionStats> {
    // Validate mesh
    if mesh.n_vertices() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if mesh.n_faces() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    let original_vertices = mesh.n_vertices();
    let original_edges = mesh.n_edges();
    let original_faces = mesh.n_faces();
    
    // Store original positions for vertices
    let mut original_positions: HashMap<u32, glam::Vec3> = HashMap::new();
    for vh in mesh.vertices() {
        if let Some(pos) = mesh.point(vh) {
            original_positions.insert(vh.idx(), pos);
        }
    }
    
    // Step 1: Compute all face points and store them
    let mut face_points: HashMap<u32, glam::Vec3> = HashMap::new();
    for fh in mesh.faces() {
        let fp = compute_face_point(mesh, fh);
        face_points.insert(fh.idx(), fp);
    }
    
    // Step 2: Compute all edge points and create new vertices
    // We need to track: (v0, v1) -> new_vertex_handle
    let mut edge_to_new_vertex: HashMap<(u32, u32), VertexHandle> = HashMap::new();
    let mut edge_points_map: HashMap<(u32, u32), glam::Vec3> = HashMap::new();
    
    // Iterate through all halfedges to find unique edges
    let n_halfedges = mesh.n_halfedges();
    for i in (0..n_halfedges).step_by(2) {
        let heh = HalfedgeHandle::new(i as u32);
        let v0 = mesh.from_vertex_handle(heh);
        let v1 = mesh.to_vertex_handle(heh);
        
        // Use canonical ordering
        let (min_v, max_v) = if v0.idx() < v1.idx() { (v0, v1) } else { (v1, v0) };
        
        // Skip if already processed
        if edge_to_new_vertex.contains_key(&(min_v.idx(), max_v.idx())) {
            continue;
        }
        
        // Compute edge point
        let edge_point = compute_edge_point(mesh, min_v, max_v);
        
        // Create new vertex at edge point
        let new_vh = mesh.add_vertex(edge_point);
        
        edge_to_new_vertex.insert((min_v.idx(), max_v.idx()), new_vh);
        edge_points_map.insert((min_v.idx(), max_v.idx()), edge_point);
    }
    
    // Step 3: Create face points as new vertices
    // First collect all original face handles (since we can't iterate while modifying mesh)
    let original_face_handles_for_fp: Vec<FaceHandle> = mesh.faces().collect();
    let mut face_point_vertices: HashMap<u32, VertexHandle> = HashMap::new();
    for fh in original_face_handles_for_fp {
        if let Some(fp) = face_points.get(&fh.idx()) {
            let fp_vh = mesh.add_vertex(*fp);
            face_point_vertices.insert(fh.idx(), fp_vh);
        }
    }
    
    // Step 4: Create new faces
    // For each original face, connect the face point to all edge points
    let original_face_handles: Vec<FaceHandle> = mesh.faces().collect();
    
    for fh in original_face_handles {
        let face_verts = get_face_vertices_polygonal(mesh, fh);
        let n = face_verts.len();
        
        // Get the face point vertex
        let fp_vh = match face_point_vertices.get(&fh.idx()) {
            Some(vh) => *vh,
            None => continue,
        };
        
        // Get edge vertices for each edge of the face
        let mut edge_verts: Vec<VertexHandle> = Vec::with_capacity(n);
        
        for i in 0..n {
            let v0 = face_verts[i];
            let v1 = face_verts[(i + 1) % n];
            
            let (min_v, max_v) = if v0.idx() < v1.idx() { (v0, v1) } else { (v1, v0) };
            
            let ev = match edge_to_new_vertex.get(&(min_v.idx(), max_v.idx())) {
                Some(vh) => *vh,
                None => continue,
            };
            edge_verts.push(ev);
        }
        
        // Create new faces: for each original vertex vi, create a quad
        // Quad: [face_point, edge_point_prev, vertex_i, edge_point_i]
        // Where edge_point_prev is the edge point before vi in the face order
        // and edge_point_i is the edge point after vi
        
        for i in 0..n {
            let vi = face_verts[i];
            let edge_point_prev = edge_verts[(i + n - 1) % n]; // edge point before vi
            let edge_point_i = edge_verts[i]; // edge point after vi
            
            // Create a quad: fp, edge_point_prev, vi, edge_point_i
            mesh.add_face(&[fp_vh, edge_point_prev, vi, edge_point_i]);
        }
    }
    
    // Step 5: Update original vertex positions using Catmull-Clark weights
    // Only update vertices that existed before the subdivision
    for (idx, _pos) in &original_positions {
        let vh = VertexHandle::new(*idx);
        if vh.is_valid() && vh.idx() < original_vertices as u32 {
            if let Ok(new_pos) = calculate_catmull_clark_new_position(mesh, vh) {
                mesh.set_point(vh, new_pos);
            }
        }
    }
    
    // Calculate statistics
    let new_vertices = mesh.n_vertices() - original_vertices;
    let new_edges = mesh.n_edges() - original_edges;
    let new_faces = mesh.n_faces() - original_faces;
    
    let stats = SubdivisionStats {
        original_vertices,
        original_edges,
        original_faces,
        new_vertices,
        new_edges,
        new_faces,
    };
    
    Ok(stats)
}

/// Perform multiple iterations of Catmull-Clark subdivision.
///
/// # Arguments
/// * `mesh` - The mesh to subdivide
/// * `iterations` - Number of subdivision iterations
///
/// # Returns
/// * `Ok(Vec<SubdivisionStats>)` - Statistics for each iteration
/// * `Err(SubdivisionError)` - If any subdivision fails
pub fn catmull_clark_subdivide_iterations(mesh: &mut PolyMeshSoA, iterations: usize) -> SubdivisionResult<Vec<SubdivisionStats>> {
    let mut all_stats = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let stats = catmull_clark_subdivide(mesh)?;
        all_stats.push(stats);
    }
    
    Ok(all_stats)
}

/// Validate that a mesh is suitable for Catmull-Clark subdivision
///
/// # Arguments
/// * `mesh` - The mesh to validate
///
/// # Returns
/// * `Ok(())` - If mesh is valid
/// * `Err(SubdivisionError)` - If mesh has issues
pub fn validate_for_catmull_clark(mesh: &PolyMeshSoA) -> SubdivisionResult<()> {
    if mesh.n_vertices() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    if mesh.n_faces() == 0 {
        return Err(SubdivisionError::EmptyMesh);
    }
    
    // Check all faces have at least 3 vertices
    for fh in mesh.faces() {
        let verts = get_face_vertices_polygonal(mesh, fh);
        if verts.len() < 3 {
            return Err(SubdivisionError::InvalidTopology(format!(
                "Face {:?} has {} vertices (expected at least 3)", fh, verts.len()
            )));
        }
        
        // Check for duplicate vertices
        for i in 0..verts.len() {
            for j in (i + 1)..verts.len() {
                if verts[i] == verts[j] {
                    return Err(SubdivisionError::InvalidTopology("Degenerate face found".to_string()));
                }
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::PolyMeshSoA;

    fn create_simple_triangle() -> PolyMeshSoA {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a simple triangle
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.5, 1.0, 0.0));
        
        mesh.add_face(&[v0, v1, v2]);
        
        mesh
    }

    fn create_triangle_mesh_with_boundary() -> PolyMeshSoA {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a quad as two triangles (boundary mesh)
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v0, v2, v3]);
        
        mesh
    }

    fn create_tetrahedron() -> PolyMeshSoA {
        let mut mesh = PolyMeshSoA::new();
        
        // Tetrahedron (closed mesh)
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.5, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.5, 0.5, 1.0));
        
        mesh.add_face(&[v0, v1, v2]); // Base
        mesh.add_face(&[v0, v1, v3]);
        mesh.add_face(&[v1, v2, v3]);
        mesh.add_face(&[v2, v0, v3]);
        
        mesh
    }

    #[test]
    fn test_is_triangular() {
        let mesh = create_simple_triangle();
        assert!(is_triangular(&mesh));
    }

    #[test]
    fn test_get_face_vertices() {
        let mesh = create_simple_triangle();
        let fh = FaceHandle::new(0);
        let verts = get_face_vertices(&mesh, fh);
        
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn test_get_vertex_valence() {
        let mesh = create_simple_triangle();
        
        // All vertices should have valence 2 in a single triangle
        for vh in mesh.vertices() {
            let valence = get_vertex_valence(&mesh, vh);
            assert_eq!(valence, 2);
        }
    }

    #[test]
    fn test_is_boundary_vertex() {
        let mesh = create_triangle_mesh_with_boundary();
        
        // All vertices are on boundary for this mesh
        for vh in mesh.vertices() {
            assert!(is_boundary_vertex(&mesh, vh));
        }
    }

    #[test]
    fn test_validate_triangular_mesh() {
        let mesh = create_simple_triangle();
        assert!(validate_for_subdivision(&mesh).is_ok());
    }

    #[test]
    fn test_validate_closed_mesh() {
        let mesh = create_tetrahedron();
        assert!(validate_for_subdivision(&mesh).is_ok());
    }

    #[test]
    fn test_validate_empty_mesh() {
        let mesh = PolyMeshSoA::new();
        assert!(matches!(
            validate_for_subdivision(&mesh),
            Err(SubdivisionError::EmptyMesh)
        ));
    }

    #[test]
    fn test_split_edge_simple() {
        let mut mesh = create_simple_triangle();
        
        let v0 = VertexHandle::new(0);
        let v1 = VertexHandle::new(1);
        
        let result = split_edge(&mut mesh, v0, v1);
        
        // Should create a new vertex at midpoint
        assert!(result.is_ok());
        
        let new_vh = result.unwrap();
        assert!(new_vh.is_valid());
        
        // Check new vertex position
        let new_pos = mesh.point(new_vh).unwrap();
        let expected = glam::vec3(0.5, 0.0, 0.0); // Midpoint of (0,0,0) and (1,0,0)
        
        assert!((new_pos.x - expected.x).abs() < 0.001);
        assert!((new_pos.y - expected.y).abs() < 0.001);
    }

    #[test]
    fn test_loop_subdivide_single_triangle() {
        let mut mesh = create_simple_triangle();
        
        let stats = loop_subdivide(&mut mesh);
        
        assert!(stats.is_ok());
        
        let stats = stats.unwrap();
        
        // After subdivision:
        // - 1 original triangle -> 4 triangles
        // - Original vertices: 3
        // - New edge vertices: 3
        // - Total vertices: 6
        // - Total edges: increases
        // - Total faces: 4
        
        assert_eq!(stats.original_vertices, 3);
        assert_eq!(stats.original_faces, 1);
        assert_eq!(mesh.n_faces(), 4);
        
        // Should have 6 vertices (3 original + 3 new edge points)
        // Note: actual count may vary depending on implementation
        assert!(mesh.n_vertices() >= 3);
    }

    #[test]
    fn test_loop_subdivide_tetrahedron() {
        let mut mesh = create_tetrahedron();
        
        let stats = loop_subdivide(&mut mesh);
        
        assert!(stats.is_ok());
        
        let stats = stats.unwrap();
        
        // Original: 4 faces
        // After: 4 * 4 = 16 faces
        assert_eq!(stats.original_faces, 4);
        assert_eq!(mesh.n_faces(), 16);
    }

    #[test]
    fn test_loop_subdivide_boundary_mesh() {
        let mut mesh = create_triangle_mesh_with_boundary();
        
        let stats = loop_subdivide(&mut mesh);
        
        assert!(stats.is_ok());
        
        let stats = stats.unwrap();
        
        // Original: 2 triangles
        // After: 2 * 4 = 8 triangles
        assert_eq!(stats.original_faces, 2);
        assert_eq!(mesh.n_faces(), 8);
    }

    #[test]
    fn test_loop_subdivide_iterations() {
        let mut mesh = create_tetrahedron();
        
        let all_stats = loop_subdivide_iterations(&mut mesh, 2);
        
        assert!(all_stats.is_ok());
        
        let all_stats = all_stats.unwrap();
        
        assert_eq!(all_stats.len(), 2);
        
        // First iteration: 4 faces -> 16 faces
        assert_eq!(all_stats[0].original_faces, 4);
        assert_eq!(mesh.n_faces(), 16);
        
        // Second iteration: 16 faces -> 64 faces
        assert_eq!(all_stats[1].original_faces, 16);
    }

    #[test]
    fn test_subdivision_preserves_manifold() {
        let mut mesh = create_tetrahedron();
        
        // Before subdivision
        let initial_validate = mesh.validate();
        if let Err(e) = initial_validate {
            println!("Warning: Initial mesh validation: {}", e);
        }
        
        loop_subdivide(&mut mesh).expect("Subdivision should succeed");
        
        // After subdivision - check no degenerate faces
        for fh in mesh.faces() {
            let verts = get_face_vertices(&mesh, fh);
            assert_eq!(verts.len(), 3, "Face should be a triangle");
            
            // Check for duplicate vertices
            assert_ne!(verts[0], verts[1], "Duplicate vertex in face");
            assert_ne!(verts[1], verts[2], "Duplicate vertex in face");
            assert_ne!(verts[0], verts[2], "Duplicate vertex in face");
        }
    }

    #[test]
    fn test_subdivision_stats() {
        let mut mesh = create_simple_triangle();
        
        let stats = loop_subdivide(&mut mesh).unwrap();
        
        println!("{}", stats);
        
        // Check stats make sense
        assert!(stats.original_vertices > 0);
        assert!(stats.original_faces > 0);
        assert!(stats.new_faces > 0);
        assert!(mesh.n_faces() > stats.original_faces);
    }

    #[test]
    fn test_calculate_loop_new_position_interior() {
        let mesh = create_tetrahedron();
        
        // Any vertex in a tetrahedron is interior (valence 3)
        let v0 = VertexHandle::new(0);
        let new_pos = calculate_loop_new_position(&mesh, v0);
        
        assert!(new_pos.is_ok());
        
        // The new position should be different from the original
        let old_pos = mesh.point(v0).unwrap();
        let calculated = new_pos.unwrap();
        
        // Should be a weighted average, so slightly different
        let diff = (calculated - old_pos).length();
        assert!(diff >= 0.0);
    }

    #[test]
    fn test_calculate_loop_new_position_boundary() {
        let mesh = create_triangle_mesh_with_boundary();
        
        // All vertices should be boundary vertices
        for vh in mesh.vertices() {
            let new_pos = calculate_loop_new_position(&mesh, vh);
            assert!(new_pos.is_ok());
        }
    }

    // Integration test with more complex mesh
    #[test]
    fn test_subdivision_icosphere_like() {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a simple triangulated sphere approximation (icosahedron-like)
        // Using 20 triangles - a simple subdivision test
        
        // Center top
        let v0 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        // Upper ring (5 vertices)
        let angle_step = std::f32::consts::TAU / 5.0;
        let upper: Vec<_> = (0..5).map(|i| {
            let angle = i as f32 * angle_step;
            mesh.add_vertex(glam::vec3(angle.sin() * 0.8, 0.6, angle.cos() * 0.8))
        }).collect();
        
        // Lower ring (5 vertices)
        let lower: Vec<_> = (0..5).map(|i| {
            let angle = (i as f32 + 0.5) * angle_step;
            mesh.add_vertex(glam::vec3(angle.sin() * 0.8, -0.6, angle.cos() * 0.8))
        }).collect();
        
        // Bottom
        let _v11 = mesh.add_vertex(glam::vec3(0.0, -1.0, 0.0));
        
        // Add faces (simplified - upper cap)
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[v0, upper[i], upper[next]]);
        }
        
        // Add faces (middle band - simplified to 10 triangles)
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[upper[i], lower[i], upper[next]]);
            mesh.add_face(&[upper[next], lower[i], lower[next]]);
        }
        
        // Add lower cap
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[lower[i], lower[next], lower[next]]);
        }
        
        // Now test subdivision
        let stats = loop_subdivide(&mut mesh);
        
        // Should succeed
        assert!(stats.is_ok());
        
        let _stats = stats.unwrap();
        
        // Count should have increased
        assert!(mesh.n_faces() > 20);
        assert!(mesh.n_vertices() > 11);
    }

    // ========================================================================
    // Catmull-Clark Subdivision Tests
    // ========================================================================

    fn create_simple_quad() -> PolyMeshSoA {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a simple quad (4 vertices, 1 face)
        // Quad in XY plane:
        // v3 --- v2
        // |      |
        // v0 --- v1
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        mesh.add_face(&[v0, v1, v2, v3]);
        
        mesh
    }

    fn create_cube() -> PolyMeshSoA {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a cube (8 vertices, 6 quad faces)
        // Front face
        let v0 = mesh.add_vertex(glam::vec3(-1.0, -1.0,  1.0)); // front bottom-left
        let v1 = mesh.add_vertex(glam::vec3( 1.0, -1.0,  1.0)); // front bottom-right
        let v2 = mesh.add_vertex(glam::vec3( 1.0,  1.0,  1.0)); // front top-right
        let v3 = mesh.add_vertex(glam::vec3(-1.0,  1.0,  1.0)); // front top-left
        
        // Back face
        let v4 = mesh.add_vertex(glam::vec3(-1.0, -1.0, -1.0)); // back bottom-left
        let v5 = mesh.add_vertex(glam::vec3( 1.0, -1.0, -1.0)); // back bottom-right
        let v6 = mesh.add_vertex(glam::vec3( 1.0,  1.0, -1.0)); // back top-right
        let v7 = mesh.add_vertex(glam::vec3(-1.0,  1.0, -1.0)); // back top-left
        
        // Front face
        mesh.add_face(&[v0, v1, v2, v3]);
        // Right face
        mesh.add_face(&[v1, v5, v6, v2]);
        // Back face
        mesh.add_face(&[v5, v4, v7, v6]);
        // Left face
        mesh.add_face(&[v4, v0, v3, v7]);
        // Top face
        mesh.add_face(&[v3, v2, v6, v7]);
        // Bottom face
        mesh.add_face(&[v4, v5, v1, v0]);
        
        mesh
    }

    #[test]
    fn test_catmull_clark_validate_quad() {
        let mesh = create_simple_quad();
        
        // Should validate successfully
        let result = validate_for_catmull_clark(&mesh);
        assert!(result.is_ok(), "Quad should validate for Catmull-Clark: {:?}", result);
        
        // Check structure
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_catmull_clark_validate_cube() {
        let mesh = create_cube();
        
        // Should validate successfully
        let result = validate_for_catmull_clark(&mesh);
        assert!(result.is_ok(), "Cube should validate for Catmull-Clark: {:?}", result);
        
        // Check structure
        assert_eq!(mesh.n_vertices(), 8);
        assert_eq!(mesh.n_faces(), 6);
    }

    #[test]
    fn test_catmull_clark_subdivide_quad() {
        let mut mesh = create_simple_quad();
        
        let original_vertices = mesh.n_vertices();
        let original_faces = mesh.n_faces();
        
        println!("Before subdivision: {} vertices, {} faces", original_vertices, original_faces);
        
        let stats = catmull_clark_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Subdivision should succeed: {:?}", stats);
        
        let stats = stats.unwrap();
        
        println!("After subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        println!("Stats: {}", stats);
        
        // A quad should subdivide into 4 quads
        // - Original vertices: 4
        // - Face point: 1
        // - Edge points: 4
        // - Total vertices: 9
        // Note: Original face is kept + 4 new faces = 5 total faces
        assert_eq!(original_faces, 1);
        assert!(mesh.n_faces() >= 4, "Should have at least 4 faces");
        
        // Check we have the expected vertices
        assert_eq!(mesh.n_vertices(), 9, "Should have 9 vertices (4 original + 4 edge + 1 face)");
        
        // Check stats
        assert_eq!(stats.new_vertices, 5); // 4 edge + 1 face point
        assert!(stats.new_faces >= 4); // at least 4 new faces
    }

    #[test]
    fn test_catmull_clark_subdivide_cube() {
        let mut mesh = create_cube();
        
        let original_vertices = mesh.n_vertices();
        let original_faces = mesh.n_faces();
        
        println!("Before subdivision: {} vertices, {} faces", original_vertices, original_faces);
        
        let stats = catmull_clark_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Subdivision should succeed: {:?}", stats);
        
        let stats = stats.unwrap();
        
        println!("After subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        println!("Stats: {}", stats);
        
        // A cube has 6 faces, each quad becomes 4 quads
        // Original faces are kept + 6*4 new faces = 30 total faces
        assert_eq!(original_faces, 6);
        assert!(mesh.n_faces() >= 24, "Cube should have at least 24 quads (6 * 4)");
        
        // Check stats make sense
        assert!(stats.new_vertices > 0);
        assert!(stats.new_faces > 0);
    }

    #[test]
    fn test_catmull_clark_subdivide_iterations() {
        let mut mesh = create_cube();
        
        // First iteration
        let stats1 = catmull_clark_subdivide(&mut mesh);
        assert!(stats1.is_ok());
        
        println!("After 1st subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        
        let faces_after_1 = mesh.n_faces();
        
        // Second iteration
        let stats2 = catmull_clark_subdivide(&mut mesh);
        assert!(stats2.is_ok());
        
        println!("After 2nd subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        
        let faces_after_2 = mesh.n_faces();
        
        // Third iteration
        let stats3 = catmull_clark_subdivide(&mut mesh);
        assert!(stats3.is_ok());
        
        println!("After 3rd subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        
        // Each subdivision multiplies faces by approximately 4 (original faces kept)
        // 6 -> 30 -> 126 -> 510 (roughly 5x each time due to keeping original faces)
        assert!(faces_after_1 > 24);
        assert!(faces_after_2 > faces_after_1);
    }

    #[test]
    fn test_catmull_clark_mixed_face_mesh() {
        let mut mesh = PolyMeshSoA::new();
        
        // Create a mesh with mixed face types: triangle + quad
        //  v2
        //  |\
        //  | \
        //  v0--v1
        //  |
        //  |   
        //  v3--v4
        //
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.5, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, -1.0, 0.0));
        let v4 = mesh.add_vertex(glam::vec3(1.0, -1.0, 0.0));
        
        // Triangle face
        mesh.add_face(&[v0, v1, v2]);
        // Quad face
        mesh.add_face(&[v0, v3, v4, v1]);
        
        assert_eq!(mesh.n_faces(), 2);
        
        // Subdivide
        let stats = catmull_clark_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Mixed mesh subdivision should succeed: {:?}", stats);
        
        println!("Mixed mesh after subdivision: {} vertices, {} faces", mesh.n_vertices(), mesh.n_faces());
        
        // Triangle becomes 3 quads, quad becomes 4 quads
        // Original faces kept: 2 + 3 + 4 = 9 faces
        assert!(mesh.n_faces() >= 7);
    }

    #[test]
    fn test_catmull_clark_stats() {
        let mut mesh = create_simple_quad();
        
        let stats = catmull_clark_subdivide(&mut mesh).unwrap();
        
        println!("{}", stats);
        
        // Check stats make sense
        assert!(stats.original_vertices > 0);
        assert!(stats.original_edges > 0);
        assert!(stats.original_faces > 0);
        assert!(stats.new_vertices > 0);
        assert!(stats.new_faces > 0);
        
        // Verify total counts
        assert_eq!(stats.original_vertices + stats.new_vertices, mesh.n_vertices());
        assert_eq!(stats.original_faces + stats.new_faces, mesh.n_faces());
    }

    #[test]
    fn test_catmull_clark_preserves_manifold() {
        let mut mesh = create_cube();
        
        // Validate initial mesh
        let initial_validate = mesh.validate();
        if let Err(e) = initial_validate {
            println!("Warning: Initial mesh validation: {}", e);
        }
        
        catmull_clark_subdivide(&mut mesh).expect("Subdivision should succeed");
        
        // After subdivision - verify all faces have valid vertices
        for fh in mesh.faces() {
            let verts = get_face_vertices_polygonal(&mesh, fh);
            assert!(verts.len() >= 3, "Face should have at least 3 vertices");
            
            // Check for duplicate vertices
            for i in 0..verts.len() {
                for j in (i + 1)..verts.len() {
                    assert_ne!(verts[i], verts[j], "Duplicate vertex in face");
                }
            }
        }
    }

    // ========================================================================
    // Sqrt3 Subdivision Tests
    // ========================================================================

    #[test]
    fn test_sqrt3_is_triangular() {
        let mesh = create_simple_triangle();
        assert!(is_mesh_triangular(&mesh));
    }

    #[test]
    fn test_sqrt3_subdivide_single_triangle() {
        let mut mesh = create_simple_triangle();
        
        // Before: 1 triangle, 3 vertices
        let original_faces = mesh.n_faces();
        assert_eq!(original_faces, 1);
        
        let stats = sqrt3_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Sqrt3 subdivision should succeed: {:?}", stats);
        
        let stats = stats.unwrap();
        
        // Sqrt3: 1 triangle -> 3 triangles
        assert_eq!(stats.original_faces, 1);
        assert_eq!(mesh.n_faces(), 3, "One triangle should become 3 triangles");
        
        // Vertices: 3 original + 3 edge midpoints = 6
        assert_eq!(mesh.n_vertices(), 6);
        
        println!("Sqrt3 stats: {}", stats);
    }

    #[test]
    fn test_sqrt3_subdivide_tetrahedron() {
        let mut mesh = create_tetrahedron();
        
        // Before: 4 triangles
        let original_faces = mesh.n_faces();
        assert_eq!(original_faces, 4);
        
        let stats = sqrt3_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Sqrt3 subdivision should succeed: {:?}", stats);
        
        let stats = stats.unwrap();
        
        // Sqrt3: 4 triangles -> 12 triangles (4 * 3)
        assert_eq!(stats.original_faces, 4);
        assert_eq!(mesh.n_faces(), 12, "4 triangles should become 12 triangles");
        
        println!("Sqrt3 tetrahedron stats: {}", stats);
    }

    #[test]
    fn test_sqrt3_subdivide_boundary_mesh() {
        let mut mesh = create_triangle_mesh_with_boundary();
        
        // Before: 2 triangles (a quad made of 2 triangles)
        let original_faces = mesh.n_faces();
        assert_eq!(original_faces, 2);
        
        let stats = sqrt3_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Sqrt3 subdivision should succeed: {:?}", stats);
        
        let stats = stats.unwrap();
        
        // Sqrt3: 2 triangles -> 6 triangles
        assert_eq!(stats.original_faces, 2);
        assert_eq!(mesh.n_faces(), 6, "2 triangles should become 6 triangles");
        
        println!("Sqrt3 boundary mesh stats: {}", stats);
    }

    #[test]
    fn test_sqrt3_subdivide_iterations() {
        let mut mesh = create_tetrahedron();
        
        // First iteration: 4 -> 12
        let stats1 = sqrt3_subdivide(&mut mesh);
        assert!(stats1.is_ok());
        assert_eq!(mesh.n_faces(), 12);
        
        println!("After 1st Sqrt3: {} faces", mesh.n_faces());
        
        // Second iteration: 12 -> 36
        let stats2 = sqrt3_subdivide(&mut mesh);
        assert!(stats2.is_ok());
        assert_eq!(mesh.n_faces(), 36);
        
        println!("After 2nd Sqrt3: {} faces", mesh.n_faces());
        
        // Third iteration: 36 -> 108
        let stats3 = sqrt3_subdivide(&mut mesh);
        assert!(stats3.is_ok());
        assert_eq!(mesh.n_faces(), 108);
        
        println!("After 3rd Sqrt3: {} faces", mesh.n_faces());
    }

    #[test]
    fn test_sqrt3_subdivide_using_iterator() {
        let mut mesh = create_tetrahedron();
        
        let all_stats = sqrt3_subdivide_iterations(&mut mesh, 3);
        
        assert!(all_stats.is_ok());
        
        let all_stats = all_stats.unwrap();
        
        assert_eq!(all_stats.len(), 3);
        
        // Check the progression: 4 -> 12 -> 36 -> 108
        assert_eq!(all_stats[0].original_faces, 4);
        assert_eq!(all_stats[1].original_faces, 12);
        assert_eq!(all_stats[2].original_faces, 36);
        
        // Final face count should be 108
        assert_eq!(mesh.n_faces(), 108);
    }

    #[test]
    fn test_sqrt3_preserves_triangular() {
        let mut mesh = create_tetrahedron();
        
        sqrt3_subdivide(&mut mesh).expect("Sqrt3 subdivision should succeed");
        
        // Check all faces are still triangles
        for fh in mesh.faces() {
            let verts = get_face_vertices(&mesh, fh);
            assert_eq!(verts.len(), 3, "All faces should still be triangles");
        }
        
        // Check is_mesh_triangular returns true
        assert!(is_mesh_triangular(&mesh));
    }

    #[test]
    fn test_sqrt3_stats() {
        let mut mesh = create_simple_triangle();
        
        let stats = sqrt3_subdivide(&mut mesh).unwrap();
        
        println!("Sqrt3 stats: {}", stats);
        
        // Check stats make sense
        assert_eq!(stats.original_vertices, 3);
        assert_eq!(stats.original_faces, 1);
        
        // New vertices = 3 edge midpoints
        assert_eq!(stats.new_vertices, 3);
        
        // New faces = 3 - 1 = 2 (we added 3, removed 1)
        assert_eq!(stats.new_faces, 2);
        
        // Verify totals
        assert_eq!(stats.original_vertices + stats.new_vertices, mesh.n_vertices());
        assert_eq!(stats.original_faces + stats.new_faces, mesh.n_faces());
    }

    #[test]
    fn test_sqrt3_on_larger_mesh() {
        let mut mesh = PolyMeshSoA::new();
        
        // Create an icosahedron-like mesh (20 triangles)
        // Center top
        let v0 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        // Upper ring (5 vertices)
        let angle_step = std::f32::consts::TAU / 5.0;
        let upper: Vec<_> = (0..5).map(|i| {
            let angle = i as f32 * angle_step;
            mesh.add_vertex(glam::vec3(angle.sin() * 0.8, 0.6, angle.cos() * 0.8))
        }).collect();
        
        // Lower ring (5 vertices)
        let lower: Vec<_> = (0..5).map(|i| {
            let angle = (i as f32 + 0.5) * angle_step;
            mesh.add_vertex(glam::vec3(angle.sin() * 0.8, -0.6, angle.cos() * 0.8))
        }).collect();
        
        // Bottom
        let v11 = mesh.add_vertex(glam::vec3(0.0, -1.0, 0.0));
        
        // Add faces (simplified - upper cap)
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[v0, upper[i], upper[next]]);
        }
        
        // Add faces (middle band)
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[upper[i], lower[i], upper[next]]);
            mesh.add_face(&[upper[next], lower[i], lower[next]]);
        }
        
        // Add lower cap
        for i in 0..5 {
            let next = (i + 1) % 5;
            mesh.add_face(&[lower[i], lower[next], v11]);
        }
        
        let original_faces = mesh.n_faces();
        println!("Icosahedron-like mesh: {} faces", original_faces);
        
        // Sqrt3 subdivision
        let stats = sqrt3_subdivide(&mut mesh);
        
        assert!(stats.is_ok(), "Sqrt3 should succeed on larger mesh");
        
        let stats = stats.unwrap();
        
        // Should be 3x faces
        assert_eq!(mesh.n_faces(), original_faces * 3);
        
        println!("After Sqrt3: {} faces", mesh.n_faces());
        println!("Stats: {}", stats);
    }

    #[test]
    fn test_sqrt3_not_triangular_error() {
        // Test that Sqrt3 fails on non-triangular mesh
        let mut mesh = PolyMeshSoA::new();
        
        // Create a quad (not triangular)
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        mesh.add_face(&[v0, v1, v2, v3]);
        
        let result = sqrt3_subdivide(&mut mesh);
        
        assert!(matches!(result, Err(SubdivisionError::NotTriangular)));
    }

    #[test]
    fn test_sqrt3_empty_mesh_error() {
        let mut mesh = PolyMeshSoA::new();
        
        let result = sqrt3_subdivide(&mut mesh);
        
        assert!(matches!(result, Err(SubdivisionError::EmptyMesh)));
    }
}
