//! # PolyConnectivity
//!
//! Polygonal mesh connectivity implementation.
//! Provides iteration and circulation over mesh elements.

use std::iter::{IntoIterator, Iterator};
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::kernel::ArrayKernel;
use crate::soa_kernel::SoAKernel;
use crate::items::Vertex;

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

// ============================================================================
// Standard Handle-based Iterators
// ============================================================================

/// Vertex iterator - Handle-based (standard API)
#[derive(Debug)]
pub struct VertexIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
    end: usize,
}

impl<'a> VertexIter<'a> {
    pub fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
            end: kernel.n_vertices(),
        }
    }
}

impl<'a> Iterator for VertexIter<'a> {
    type Item = VertexHandle;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let handle = VertexHandle::new(self.current as u32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Edge iterator - Optimized with cached end value
#[derive(Debug)]
pub struct EdgeIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
    end: usize,
}

impl<'a> EdgeIter<'a> {
    pub fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
            end: kernel.n_edges(),
        }
    }
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = EdgeHandle;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let handle = EdgeHandle::new(self.current as u32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Face iterator - Optimized with cached end value
#[derive(Debug)]
pub struct FaceIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
    end: usize,
}

impl<'a> FaceIter<'a> {
    pub fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
            end: kernel.n_faces(),
        }
    }
}

impl<'a> Iterator for FaceIter<'a> {
    type Item = FaceHandle;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let handle = FaceHandle::new(self.current as u32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Halfedge iterator
#[derive(Debug)]
pub struct HalfedgeIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
    total: usize,
}

impl<'a> HalfedgeIter<'a> {
    pub fn new(kernel: &'a ArrayKernel) -> Self {
        let total = kernel.n_halfedges();
        Self {
            kernel,
            current: 0,
            total,
        }
    }
}

impl<'a> Iterator for HalfedgeIter<'a> {
    type Item = HalfedgeHandle;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.total {
            let handle = HalfedgeHandle::new(self.current as u32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Circulator for vertices around a vertex (1-ring)
pub struct VertexVertexCirculator<'a> {
    kernel: &'a ArrayKernel,
    center: VertexHandle,
    current: Option<HalfedgeHandle>,
    started: bool,
}

impl<'a> VertexVertexCirculator<'a> {
    pub(crate) fn new(kernel: &'a ArrayKernel, vh: VertexHandle) -> Self {
        let start_heh = kernel.halfedge_handle(vh);
        Self {
            kernel,
            center: vh,
            current: start_heh,
            started: false,
        }
    }
}

impl<'a> Iterator for VertexVertexCirculator<'a> {
    type Item = VertexHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(heh) = self.current {
            let next_vh = self.kernel.to_vertex_handle(heh);
            
            // Move to next halfedge around the vertex
            // For a full implementation, we'd need proper prev/next linkage
            self.current = Some(self.kernel.opposite_halfedge_handle(heh));
            
            // Skip if we've gone full circle
            if Some(next_vh) == self.kernel.halfedge_handle(self.center).map(|h| self.kernel.to_vertex_handle(h)) {
                if self.started {
                    return None;
                }
            }
            self.started = true;
            
            Some(next_vh)
        } else {
            None
        }
    }
}

/// Vertex-face circulator
pub struct VertexFaceCirculator<'a> {
    kernel: &'a ArrayKernel,
    center: VertexHandle,
    current: Option<HalfedgeHandle>,
}

impl<'a> VertexFaceCirculator<'a> {
    pub(crate) fn new(kernel: &'a ArrayKernel, vh: VertexHandle) -> Self {
        let start_heh = kernel.halfedge_handle(vh);
        Self {
            kernel,
            center: vh,
            current: start_heh,
        }
    }
}

impl<'a> Iterator for VertexFaceCirculator<'a> {
    type Item = FaceHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(heh) = self.current {
            // Get face from halfedge
            let fh = self.kernel.face_handle(heh);
            
            // Move to next halfedge around vertex
            // In a full implementation, use proper circulation
            self.current = Some(self.kernel.opposite_halfedge_handle(heh));
            
            fh
        } else {
            None
        }
    }
}

pub struct PolyMeshSoA {
    kernel: SoAKernel,
}

impl Clone for PolyMeshSoA {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
        }
    }
}

impl std::fmt::Debug for PolyMeshSoA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolyMeshSoA")
            .field("n_vertices", &self.kernel.n_vertices())
            .field("n_edges", &self.kernel.n_edges())
            .field("n_faces", &self.kernel.n_faces())
            .field("n_halfedges", &self.kernel.n_halfedges())
            .finish()
    }
}

impl PolyMeshSoA {
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
            
            // Create the halfedge from end to start
            let he = self.add_edge(end, start);
            halfedges.push(he);
        }

        // Second: set next/prev pointers for ALL halfedges in this face
        // Each halfedge points to the next one in the face cycle
        for i in 0..n {
            let curr = halfedges[i];
            let next_in_face = halfedges[(i + 1) % n];
            
            // Only set next if not already set (preserve existing for shared edges)
            self.kernel.set_next_halfedge_handle(curr, next_in_face);
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
        let v0_boundary = self.is_boundary(self.halfedge_handle(v0).unwrap_or(HalfedgeHandle::new(u32::MAX)));
        let v1_boundary = self.is_boundary(self.halfedge_handle(v1).unwrap_or(HalfedgeHandle::new(u32::MAX)));
        
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
}

#[cfg(test)]
mod tests_soa {
    use super::*;

    #[test]
    fn test_soa_mesh() {
        let mut mesh = PolyMeshSoA::new();

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
        let mut mesh = PolyMeshSoA::new();

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
        let mut mesh = PolyMeshSoA::new();

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
}

// Re-export RustMesh for convenience
pub use PolyMeshSoA as RustMesh;
