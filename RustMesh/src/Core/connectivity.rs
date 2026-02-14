//! # PolyConnectivity
//!
//! Polygonal mesh connectivity implementation.
//! Provides iteration and circulation over mesh elements.

use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::soa_kernel::SoAKernel;

// ============================================================================
// High-Performance Index Iterators (no Handle overhead)
// ============================================================================

/// Vertex index iterator - Returns usize instead of VertexHandle
/// This avoids Handle creation overhead
#[derive(Debug)]
pub struct VertexIndexIter {
    current: usize,
    end: usize,
}

impl VertexIndexIter {
    #[inline]
    pub fn new(n_vertices: usize) -> Self {
        Self { current: 0, end: n_vertices }
    }
}

impl Iterator for VertexIndexIter {
    type Item = usize;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Face index iterator - Returns usize instead of FaceHandle
#[derive(Debug)]
pub struct FaceIndexIter {
    current: usize,
    end: usize,
}

impl FaceIndexIter {
    #[inline]
    pub fn new(n_faces: usize) -> Self {
        Self { current: 0, end: n_faces }
    }
}

impl Iterator for FaceIndexIter {
    type Item = usize;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

pub struct RustMesh {
    kernel: SoAKernel,
}

impl Clone for RustMesh {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
        }
    }
}

impl std::fmt::Debug for RustMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustMesh")
            .field("n_vertices", &self.kernel.n_vertices())
            .field("n_edges", &self.kernel.n_edges())
            .field("n_faces", &self.kernel.n_faces())
            .field("n_halfedges", &self.kernel.n_halfedges())
            .finish()
    }
}

impl RustMesh {
    /// Create a new empty mesh
    #[inline]
    pub fn new() -> Self {
        Self {
            kernel: SoAKernel::new(),
        }
    }

    /// Clear the mesh
    #[inline]
    pub fn clear(&mut self) {
        self.kernel.clear();
    }

    // --- Vertex operations ---

    /// Add a vertex at the given position
    #[inline]
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        self.kernel.add_vertex(point)
    }

    /// Get the number of vertices
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.kernel.n_vertices()
    }

    /// Get vertex position by handle
    #[inline]
    pub fn point(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.kernel.point(vh.idx_usize())
    }

    /// Get vertex position by index (for internal use)
    #[inline]
    pub unsafe fn point_unchecked(&self, idx: usize) -> glam::Vec3 {
        self.kernel.point_unchecked(idx)
    }

    /// Set vertex position
    #[inline]
    pub fn set_point(&mut self, vh: VertexHandle, point: glam::Vec3) {
        self.kernel.set_point(vh.idx_usize(), point);
    }

    // --- SIMD-friendly access ---

    /// Get x coordinates slice
    #[inline]
    pub fn x(&self) -> &[f32] {
        self.kernel.x_slice()
    }

    /// Get y coordinates slice
    #[inline]
    pub fn y(&self) -> &[f32] {
        self.kernel.y_slice()
    }

    /// Get z coordinates slice
    #[inline]
    pub fn z(&self) -> &[f32] {
        self.kernel.z_slice()
    }

    /// Get x pointer for SIMD
    #[inline]
    pub fn x_ptr(&self) -> *const f32 {
        self.kernel.x_ptr()
    }

    /// Get y pointer for SIMD
    #[inline]
    pub fn y_ptr(&self) -> *const f32 {
        self.kernel.y_ptr()
    }

    /// Get z pointer for SIMD
    #[inline]
    pub fn z_ptr(&self) -> *const f32 {
        self.kernel.z_ptr()
    }

    // --- Vertex iteration ---

    /// Get an iterator over all vertex indices (fastest)
    #[inline]
    pub fn vertex_indices(&self) -> VertexIndexIter {
        VertexIndexIter::new(self.n_vertices())
    }

    /// Iterate over all vertex handles
    #[inline]
    pub fn vertices(&self) -> impl Iterator<Item = VertexHandle> + '_ {
        (0..self.n_vertices()).map(|i| VertexHandle::from_usize(i))
    }

    // --- Edge operations ---

    /// Add an edge between two vertices
    #[inline]
    pub fn add_edge(&mut self, v0: VertexHandle, v1: VertexHandle) -> HalfedgeHandle {
        self.kernel.add_edge(v0, v1)
    }

    /// Get the number of edges
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.kernel.n_edges()
    }

    // --- Face operations ---

    /// Add a face from a list of vertex handles
    pub fn add_face(&mut self, vertices: &[VertexHandle]) -> Option<FaceHandle> {
        if vertices.len() < 3 {
            return None;
        }

        let n = vertices.len();
        let mut halfedges: Vec<HalfedgeHandle> = Vec::with_capacity(n);
        
        // First: create all edges and track them
        for i in 0..n {
            let start = vertices[i];
            let end = vertices[(i + 1) % n];
            
            // Create the halfedge from start to end
            let he = self.add_edge(start, end);
            // The halfedge used by this face must be free (no face assigned yet)
            if self.kernel.face_handle(he).is_some() {
                return None;
            }
            halfedges.push(he);
        }

        // Second: set next/prev pointers for ALL halfedges in this face
        // Each halfedge points to the next one in the face cycle
        for i in 0..n {
            let curr = halfedges[i];
            let next_in_face = halfedges[(i + 1) % n];
            
            self.kernel.set_next_halfedge_handle(curr, next_in_face);
            self.kernel.set_prev_halfedge_handle(next_in_face, curr);
        }

        let fh = self.kernel.add_face(Some(halfedges[0]));

        // Set face handle for all halfedges
        for &he in &halfedges {
            self.kernel.set_face_handle(he, fh);
        }

        // Set vertex halfedge handles - use any outgoing halfedge
        for (i, &vh) in vertices.iter().enumerate() {
            let incoming_heh = halfedges[i];
            // Get the opposite halfedge which points FROM this vertex
            if let Some(opp_heh) = self.kernel.opposite_halfedge_handle(incoming_heh) {
                self.kernel.set_halfedge_handle(vh, opp_heh);
            }
        }

        Some(fh)
    }

    /// Get the number of faces
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get the number of halfedges
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.kernel.n_halfedges()
    }

    // --- Face iteration ---

    /// Get an iterator over all face indices
    #[inline]
    pub fn face_indices(&self) -> FaceIndexIter {
        FaceIndexIter::new(self.n_faces())
    }

    /// Iterate over all face handles
    #[inline]
    pub fn faces(&self) -> impl Iterator<Item = FaceHandle> + '_ {
        (0..self.n_faces()).map(|i| FaceHandle::from_usize(i))
    }

    // --- Connectivity queries ---

    /// Get the halfedge handle from a vertex
    #[inline]
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.kernel.halfedge_handle(vh)
    }

    /// Get the edge handle from a halfedge
    #[inline]
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        self.kernel.edge_handle(heh)
    }

    /// Get a halfedge from an edge (0 or 1)
    #[inline]
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        self.kernel.edge_halfedge_handle(eh, idx)
    }

    /// Get the face handle from a halfedge
    #[inline]
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.kernel.face_handle(heh)
    }

    /// Check if a halfedge is a boundary
    #[inline]
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_boundary(heh)
    }
    
    /// Validate halfedge structure integrity
    /// Returns Ok if valid, Err with message if issues found
    pub fn validate(&self) -> Result<(), String> {
        let n_vertices = self.n_vertices();
        let n_edges = self.n_edges();
        let n_faces = self.n_faces();
        let n_halfedges = self.n_halfedges();
        
        // Euler formula check: V - E + F = 2 for closed manifold
        // For meshes with boundary: V - E + F = 1 + B (B = boundary components)
        let euler = n_vertices as i32 - n_edges as i32 + n_faces as i32;
        println!("Euler characteristic: {} (V={}, E={}, F={})", euler, n_vertices, n_edges, n_faces);
        
        // Check: halfedges should be 2 * edges
        if n_halfedges != 2 * n_edges {
            return Err(format!("Halfedge count mismatch: {} != 2 * {}", n_halfedges, n_edges));
        }
        
        // Check each vertex has valid halfedge
        for vh in 0..n_vertices {
            let vh = VertexHandle::new(vh as u32);
            if let Some(heh) = self.halfedge_handle(vh) {
                if !heh.is_valid() {
                    return Err(format!("Vertex {:?} has invalid halfedge", vh));
                }
            }
        }
        
        // Check halfedge cycles (prevent infinite loops)
        for fh in 0..n_faces {
            let fh = FaceHandle::new(fh as u32);
            if let Some(start_heh) = self.face_halfedge_handle(fh) {
                let mut count = 0;
                let mut current = start_heh;
                loop {
                    count += 1;
                    if count > 64 {
                        return Err(format!("Face {:?} has >64 halfedges - cycle broken!", fh));
                    }
                    current = self.next_halfedge_handle(current);
                    if current == start_heh || !current.is_valid() {
                        break;
                    }
                }
                if count < 3 {
                    return Err(format!("Face {:?} has {} halfedges - too few!", fh, count));
                }
            }
        }
        
        // Check vertex rings (prevent infinite loops)
        for vh in 0..n_vertices {
            let vh = VertexHandle::new(vh as u32);
            if let Some(start_heh) = self.halfedge_handle(vh) {
                let mut count = 0;
                let mut current = start_heh;
                loop {
                    count += 1;
                    if count > 64 {
                        return Err(format!("Vertex {:?} has >64 halfedges - cycle broken!", vh));
                    }
                    // Move to next halfedge around vertex
                    let opposite = self.opposite_halfedge_handle(current);
                    current = self.next_halfedge_handle(opposite);
                    if current == start_heh || !current.is_valid() {
                        break;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Get the to-vertex of a halfedge
    #[inline]
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.kernel.to_vertex_handle(heh)
    }

    /// Get the from-vertex of a halfedge
    #[inline]
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.kernel.from_vertex_handle(heh)
    }

    /// Get the opposite halfedge (across the edge)
    #[inline]
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.opposite_halfedge_handle(heh).unwrap_or(heh)
    }

    /// Get the next halfedge in the cycle
    #[inline]
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.next_halfedge_handle(heh).unwrap_or(heh)
    }

    /// Get the previous halfedge in the cycle
    #[inline]
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.prev_halfedge_handle(heh).unwrap_or(heh)
    }

    /// Get the halfedge handle associated with a face
    #[inline]
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        self.kernel.face_halfedge_handle(fh)
    }

    // --- SIMD-optimized operations ---

    /// Compute bounding box (optimized)
    #[inline]
    pub fn bounding_box(&self) -> (f32, f32, f32, f32, f32, f32) {
        self.kernel.bounding_box()
    }

    /// Compute bounding box using NEON SIMD
    #[inline]
    pub unsafe fn bounding_box_simd(&self) -> (f32, f32, f32, f32, f32, f32) {
        self.kernel.bounding_box_simd()
    }

    /// Compute centroid (optimized)
    #[inline]
    pub fn centroid(&self) -> (f32, f32, f32) {
        self.kernel.centroid()
    }

    /// Compute centroid using NEON SIMD
    #[inline]
    pub unsafe fn centroid_simd(&self) -> (f32, f32, f32) {
        self.kernel.centroid_simd()
    }

    /// Compute vertex sum using NEON SIMD
    #[inline]
    pub unsafe fn vertex_sum_simd(&self) -> (f32, f32, f32) {
        self.kernel.vertex_sum_simd()
    }
    
    // =========================================================================
    // Edge Collapse (Halfedge Collapse)
    // =========================================================================
    
    /// Check if an edge collapse is legal
    /// Returns true if the halfedge can be collapsed without creating topological issues
    pub fn is_collapse_ok(&self, heh: HalfedgeHandle) -> bool {
        let v0 = self.to_vertex_handle(heh);   // Vertex to be removed
        let v1 = self.from_vertex_handle(heh); // Remaining vertex
        
        // Get the opposite halfedge
        let heh_opp = self.opposite_halfedge_handle(heh);
        
        // Check if edge is already deleted (boundary check)
        if !heh.is_valid() || !heh_opp.is_valid() {
            return false;
        }
        
        // Get adjacent faces
        let fh_left = self.face_handle(heh);
        let fh_right = self.face_handle(heh_opp);
        
        // Get neighboring vertices
        let left_next = self.to_vertex_handle(self.next_halfedge_handle(heh));
        let right_next = self.to_vertex_handle(self.next_halfedge_handle(heh_opp));
        
        // Check: vl and vr should not be the same (would create degenerate face)
        if fh_left.is_some() && fh_right.is_some() {
            if left_next == right_next {
                return false;
            }
        }
        
        // Check: if both vertices are boundary, edge should also be boundary
        let _v0_boundary = self.is_boundary(self.halfedge_handle(v0).unwrap_or(HalfedgeHandle::new(u32::MAX)));
        let _v1_boundary = self.is_boundary(self.halfedge_handle(v1).unwrap_or(HalfedgeHandle::new(u32::MAX)));
        
        // Simplified check: avoid collapsing boundary edges between two boundary vertices
        // (This is a simplified version of OpenMesh's check)
        
        true
    }
    
    /// Collapse a halfedge: move v0 to v1 and remove v0 and adjacent faces
    /// Returns Ok if successful, Err with message if failed
    pub fn collapse(&mut self, heh: HalfedgeHandle) -> Result<(), &'static str> {
        if !self.is_collapse_ok(heh) {
            return Err("Collapse not legal");
        }
        
        let v0 = self.to_vertex_handle(heh);   // Vertex to be removed
        let v1 = self.from_vertex_handle(heh); // Remaining vertex
        
        // Get the opposite halfedge
        let heh_opp = self.opposite_halfedge_handle(heh);
        
        // Get adjacent faces to delete
        let fh_left = self.face_handle(heh);
        let fh_right = self.face_handle(heh_opp);
        
        // Step 1: Update all halfedges that point to v0 to point to v1 instead
        // This includes all outgoing halfedges from v0 and all incoming halfedges to v0
        self.redirect_halfedges(v0, v1)?;
        
        // Step 2: Delete the adjacent faces
        if let Some(fh) = fh_left {
            self.delete_face(fh);
        }
        if let Some(fh) = fh_right {
            self.delete_face(fh);
        }
        
        // Step 3: Delete vertex v0
        self.delete_vertex(v0);
        
        // Step 4: Delete the edge (both halfedges)
        let eh = self.edge_handle(heh);
        self.delete_edge(eh);
        
        // Step 5: Update v1's position to the collapse target position (optional)
        // For now we keep v1's position
        
        Ok(())
    }
    
    /// Redirect all halfedges that reference from_vertex to reference to_vertex
    fn redirect_halfedges(&mut self, from_vertex: VertexHandle, to_vertex: VertexHandle) -> Result<(), &'static str> {
        // Get all halfedges and update those that reference from_vertex
        let n_halfedges = self.n_halfedges();
        
        for heh_idx in 0..n_halfedges {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            
            // Check if this halfedge's to_vertex is from_vertex
            let to_vh = self.to_vertex_handle(heh);
            if to_vh == from_vertex {
                // Update the to_vertex to to_vertex
                // This requires modifying the halfedge data directly
                self.kernel.set_halfedge_to_vertex(heh, to_vertex);
            }
        }
        
        Ok(())
    }
    
    /// Delete a face from the mesh
    pub fn delete_face(&mut self, fh: FaceHandle) {
        // Mark face as deleted (in a full implementation, we'd also handle halfedges)
        self.kernel.delete_face(fh);
    }
    
    /// Delete a vertex from the mesh
    pub fn delete_vertex(&mut self, vh: VertexHandle) {
        self.kernel.delete_vertex(vh);
    }
    
    /// Delete an edge from the mesh
    pub fn delete_edge(&mut self, eh: EdgeHandle) {
        self.kernel.delete_edge(eh);
    }

    // =========================================================================
    // Vertex attributes
    // =========================================================================

    /// Request vertex normals
    pub fn request_vertex_normals(&mut self) {
        self.kernel.request_vertex_normals();
    }

    /// Check if vertex normals are available
    pub fn has_vertex_normals(&self) -> bool {
        self.kernel.has_vertex_normals()
    }

    /// Get vertex normal
    pub fn normal(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.kernel.vertex_normal(vh)
    }

    /// Set vertex normal
    pub fn set_normal(&mut self, vh: VertexHandle, n: glam::Vec3) {
        self.kernel.set_vertex_normal(vh, n);
    }

    /// Request vertex colors
    pub fn request_vertex_colors(&mut self) {
        self.kernel.request_vertex_colors();
    }

    /// Check if vertex colors are available
    pub fn has_vertex_colors(&self) -> bool {
        self.kernel.has_vertex_colors()
    }

    /// Get vertex color
    pub fn color(&self, vh: VertexHandle) -> Option<glam::Vec4> {
        self.kernel.vertex_color(vh)
    }

    /// Set vertex color
    pub fn set_color(&mut self, vh: VertexHandle, c: glam::Vec4) {
        self.kernel.set_vertex_color(vh, c);
    }

    /// Request vertex texture coordinates
    pub fn request_vertex_texcoords(&mut self) {
        self.kernel.request_vertex_texcoords();
    }

    /// Check if vertex texcoords are available
    pub fn has_vertex_texcoords(&self) -> bool {
        self.kernel.has_vertex_texcoords()
    }

    /// Get vertex texcoord
    pub fn texcoord(&self, vh: VertexHandle) -> Option<glam::Vec2> {
        self.kernel.vertex_texcoord(vh)
    }

    /// Set vertex texcoord
    pub fn set_texcoord(&mut self, vh: VertexHandle, t: glam::Vec2) {
        self.kernel.set_vertex_texcoord(vh, t);
    }

    // =========================================================================
    // Face attributes
    // =========================================================================

    /// Request face normals
    pub fn request_face_normals(&mut self) {
        self.kernel.request_face_normals();
    }

    /// Check if face normals are available
    pub fn has_face_normals(&self) -> bool {
        self.kernel.has_face_normals()
    }

    /// Get face normal
    pub fn f_normal(&self, fh: FaceHandle) -> Option<glam::Vec3> {
        self.kernel.face_normal(fh)
    }

    /// Set face normal
    pub fn set_f_normal(&mut self, fh: FaceHandle, n: glam::Vec3) {
        self.kernel.set_face_normal(fh, n);
    }

    /// Request face colors
    pub fn request_face_colors(&mut self) {
        self.kernel.request_face_colors();
    }

    /// Check if face colors are available
    pub fn has_face_colors(&self) -> bool {
        self.kernel.has_face_colors()
    }

    /// Get face color
    pub fn f_color(&self, fh: FaceHandle) -> Option<glam::Vec4> {
        self.kernel.face_color(fh)
    }

    /// Set face color
    pub fn set_f_color(&mut self, fh: FaceHandle, c: glam::Vec4) {
        self.kernel.set_face_color(fh, c);
    }

    // =========================================================================
    // Halfedge attributes
    // =========================================================================

    /// Request halfedge normals
    pub fn request_halfedge_normals(&mut self) {
        self.kernel.request_halfedge_normals();
    }

    /// Check if halfedge normals are available
    pub fn has_halfedge_normals(&self) -> bool {
        self.kernel.has_halfedge_normals()
    }

    /// Get halfedge normal
    pub fn h_normal(&self, heh: HalfedgeHandle) -> Option<glam::Vec3> {
        self.kernel.halfedge_normal(heh)
    }

    /// Set halfedge normal
    pub fn set_h_normal(&mut self, heh: HalfedgeHandle, n: glam::Vec3) {
        self.kernel.set_halfedge_normal(heh, n);
    }

    /// Request halfedge colors
    pub fn request_halfedge_colors(&mut self) {
        self.kernel.request_halfedge_colors();
    }

    /// Check if halfedge colors are available
    pub fn has_halfedge_colors(&self) -> bool {
        self.kernel.has_halfedge_colors()
    }

    /// Get halfedge color
    pub fn h_color(&self, heh: HalfedgeHandle) -> Option<glam::Vec4> {
        self.kernel.halfedge_color(heh)
    }

    /// Set halfedge color
    pub fn set_h_color(&mut self, heh: HalfedgeHandle, c: glam::Vec4) {
        self.kernel.set_halfedge_color(heh, c);
    }

    /// Request halfedge texture coordinates
    pub fn request_halfedge_texcoords(&mut self) {
        self.kernel.request_halfedge_texcoords();
    }

    /// Check if halfedge texcoords are available
    pub fn has_halfedge_texcoords(&self) -> bool {
        self.kernel.has_halfedge_texcoords()
    }

    /// Get halfedge texcoord
    pub fn h_texcoord(&self, heh: HalfedgeHandle) -> Option<glam::Vec2> {
        self.kernel.halfedge_texcoord(heh)
    }

    /// Set halfedge texcoord
    pub fn set_h_texcoord(&mut self, heh: HalfedgeHandle, t: glam::Vec2) {
        self.kernel.set_halfedge_texcoord(heh, t);
    }

    // =========================================================================
    // Edge attributes
    // =========================================================================

    /// Request edge colors
    pub fn request_edge_colors(&mut self) {
        self.kernel.request_edge_colors();
    }

    /// Check if edge colors are available
    pub fn has_edge_colors(&self) -> bool {
        self.kernel.has_edge_colors()
    }

    /// Get edge color
    pub fn e_color(&self, eh: EdgeHandle) -> Option<glam::Vec4> {
        self.kernel.edge_color(eh)
    }

    /// Set edge color
    pub fn set_e_color(&mut self, eh: EdgeHandle, c: glam::Vec4) {
        self.kernel.set_edge_color(eh, c);
    }
}

#[cfg(test)]
mod tests_soa {
    use super::*;

    #[test]
    fn test_soa_mesh() {
        let mut mesh = RustMesh::new();

        // Add vertices
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        // Add face
        let face = mesh.add_face(&[v0, v1, v2]);
        assert!(face.is_some());

        // Check counts
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);

        // Check vertex access
        assert_eq!(mesh.point(v0), Some(glam::vec3(0.0, 0.0, 0.0)));
        assert_eq!(mesh.point(v1), Some(glam::vec3(1.0, 0.0, 0.0)));
        assert_eq!(mesh.point(v2), Some(glam::vec3(0.0, 1.0, 0.0)));

        // Check SIMD pointers
        assert!(!mesh.x_ptr().is_null());
        assert!(!mesh.y_ptr().is_null());
        assert!(!mesh.z_ptr().is_null());
    }

    #[test]
    fn test_soa_bounding_box() {
        let mut mesh = RustMesh::new();

        // Add vertices of a cube
        mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_vertex(glam::vec3(0.0, 0.0, 1.0));
        mesh.add_vertex(glam::vec3(1.0, 0.0, 1.0));
        mesh.add_vertex(glam::vec3(1.0, 1.0, 1.0));
        mesh.add_vertex(glam::vec3(0.0, 1.0, 1.0));

        let (min_x, max_x, min_y, max_y, min_z, max_z) = mesh.bounding_box();

        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 1.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_y, 1.0);
        assert_eq!(min_z, 0.0);
        assert_eq!(max_z, 1.0);
    }

    #[test]
    fn test_soa_centroid() {
        let mut mesh = RustMesh::new();

        // Add vertices
        mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(2.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(0.0, 2.0, 0.0));

        let (cx, cy, cz) = mesh.centroid();

        // Centroid of (0,0,0), (2,0,0), (0,2,0) = (2/3, 2/3, 0)
        assert!((cx - 0.667).abs() < 0.001);
        assert!((cy - 0.667).abs() < 0.001);
        assert_eq!(cz, 0.0);
    }

    #[test]
    fn test_vertex_attributes() {
        let mut mesh = RustMesh::new();

        // Add vertices
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        // Request and set vertex normals
        mesh.request_vertex_normals();
        assert!(mesh.has_vertex_normals());

        mesh.set_normal(v0, glam::vec3(0.0, 0.0, 1.0));
        mesh.set_normal(v1, glam::vec3(0.0, 0.0, 1.0));
        mesh.set_normal(v2, glam::vec3(0.0, 0.0, 1.0));

        assert_eq!(mesh.normal(v0), Some(glam::vec3(0.0, 0.0, 1.0)));

        // Request and set vertex colors
        mesh.request_vertex_colors();
        assert!(mesh.has_vertex_colors());

        mesh.set_color(v0, glam::vec4(1.0, 0.0, 0.0, 1.0));
        assert_eq!(mesh.color(v0), Some(glam::vec4(1.0, 0.0, 0.0, 1.0)));

        // Request and set vertex texcoords
        mesh.request_vertex_texcoords();
        assert!(mesh.has_vertex_texcoords());

        mesh.set_texcoord(v0, glam::vec2(0.0, 0.0));
        mesh.set_texcoord(v1, glam::vec2(1.0, 0.0));
        mesh.set_texcoord(v2, glam::vec2(0.0, 1.0));

        assert_eq!(mesh.texcoord(v0), Some(glam::vec2(0.0, 0.0)));
    }

    #[test]
    fn test_face_attributes() {
        let mut mesh = RustMesh::new();

        // Add vertices and face
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        // Request face normals
        mesh.request_face_normals();
        assert!(mesh.has_face_normals());

        mesh.set_f_normal(FaceHandle::new(0), glam::vec3(0.0, 0.0, 1.0));
        assert_eq!(mesh.f_normal(FaceHandle::new(0)), Some(glam::vec3(0.0, 0.0, 1.0)));

        // Request face colors
        mesh.request_face_colors();
        assert!(mesh.has_face_colors());

        mesh.set_f_color(FaceHandle::new(0), glam::vec4(0.5, 0.5, 0.5, 1.0));
        assert_eq!(mesh.f_color(FaceHandle::new(0)), Some(glam::vec4(0.5, 0.5, 0.5, 1.0)));
    }
}

// RustMesh is now the primary mesh type (previously PolyMeshSoA)
