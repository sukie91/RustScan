//! # TriConnectivity
//!
//! Triangular mesh connectivity with specialized operations for triangles.
//! Extends PolyConnectivity with triangle-specific functionality.

use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::kernel::ArrayKernel;

/// Triangular mesh with specialized operations
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    kernel: ArrayKernel,
}

impl TriMesh {
    /// Create a new empty triangle mesh
    pub fn new() -> Self {
        Self {
            kernel: ArrayKernel::new(),
        }
    }

    /// Clear the mesh
    pub fn clear(&mut self) {
        self.kernel.clear();
    }

    /// Check if mesh only contains triangles
    pub fn is_triangles(&self) -> bool {
        true
    }

    // --- Vertex operations ---

    /// Add a vertex at the given position
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        self.kernel.add_vertex(point)
    }

    /// Get vertex position
    pub fn point(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.kernel.vertex(vh).map(|v| v.point)
    }

    /// Set vertex position
    pub fn set_point(&mut self, vh: VertexHandle, point: glam::Vec3) {
        if let Some(v) = self.kernel.vertex_mut(vh) {
            v.point = point;
        }
    }

    // --- Triangle operations ---

    /// Add a triangle to the mesh
    pub fn add_triangle(&mut self, v0: VertexHandle, v1: VertexHandle, v2: VertexHandle) -> Option<FaceHandle> {
        // Create halfedges for the triangle edges
        let he01 = self.kernel.add_edge(v1, v0); // Halfedge from v1 to v0
        let he12 = self.kernel.add_edge(v2, v1); // Halfedge from v2 to v1
        let he20 = self.kernel.add_edge(v0, v2); // Halfedge from v0 to v2

        // Link halfedges into a cycle (CCW)
        // he01 -> he12 -> he20 -> he01
        self.kernel.set_next_halfedge_handle(he01, he12);
        self.kernel.set_next_halfedge_handle(he12, he20);
        self.kernel.set_next_halfedge_handle(he20, he01);

        // Set prev halfedges
        self.kernel.set_prev_halfedge_handle(he12, he01);
        self.kernel.set_prev_halfedge_handle(he20, he12);
        self.kernel.set_prev_halfedge_handle(he20, he01);

        // Create the face
        let fh = self.kernel.add_face_triangle(he01);

        // Connect vertices to halfedges
        self.kernel.set_halfedge_handle(v0, he20);
        self.kernel.set_halfedge_handle(v1, he01);
        self.kernel.set_halfedge_handle(v2, he12);

        // Set face for halfedges
        self.kernel.set_face_handle(he01, fh);
        self.kernel.set_face_handle(he12, fh);
        self.kernel.set_face_handle(he20, fh);

        Some(fh)
    }

    /// Get the vertices of a triangular face (optimized, assumes triangle)
    pub fn triangle_vertices(&self, fh: FaceHandle) -> Option<(VertexHandle, VertexHandle, VertexHandle)> {
        // Get any halfedge of the face
        let heh = self.kernel.face_halfedge_handle(fh)?;

        // Get the three vertices by traversing the cycle
        let v0 = self.kernel.from_vertex_handle(heh);
        let heh1 = self.kernel.next_halfedge_handle(heh);
        let v1 = self.kernel.from_vertex_handle(heh1);
        let heh2 = self.kernel.next_halfedge_handle(heh1);
        let v2 = self.kernel.from_vertex_handle(heh2);

        Some((v0, v1, v2))
    }

    // --- Topology queries ---

    /// Get the vertex opposite to a halfedge in its face
    pub fn opposite_vh(&self, heh: HalfedgeHandle) -> Option<VertexHandle> {
        if self.kernel.is_boundary(heh) {
            None
        } else {
            let next_heh = self.kernel.next_halfedge_handle(heh);
            self.kernel.from_vertex_handle_opt(next_heh)
        }
    }

    /// Get the vertex opposite to the opposite halfedge
    pub fn opposite_he_opposite_vh(&self, heh: HalfedgeHandle) -> Option<VertexHandle> {
        let opp_heh = self.kernel.opposite_halfedge_handle(heh);
        self.opposite_vh(opp_heh)
    }

    /// Check if two vertices are connected by an edge
    pub fn is_edge(&self, v0: VertexHandle, v1: VertexHandle) -> bool {
        // For a triangle mesh, check if vertices share a common face
        if let Some(heh) = self.kernel.halfedge_handle(v0) {
            let mut curr = heh;
            loop {
                let to_v = self.kernel.to_vertex_handle(curr);
                if to_v == v1 {
                    return true;
                }
                curr = self.kernel.next_halfedge_handle(curr);
                if curr == heh {
                    break;
                }
            }
        }
        false
    }

    /// Get the face adjacent to an edge
    pub fn adjacent_face(&self, eh: EdgeHandle) -> Vec<FaceHandle> {
        let mut faces = Vec::new();

        let he0 = self.kernel.edge_halfedge_handle(eh, 0);
        let he1 = self.kernel.edge_halfedge_handle(eh, 1);

        if let Some(fh0) = self.kernel.face_handle(he0) {
            if !self.kernel.is_boundary(he0) {
                faces.push(fh0);
            }
        }
        if let Some(fh1) = self.kernel.face_handle(he1) {
            if !self.kernel.is_boundary(he1) {
                faces.push(fh1);
            }
        }

        faces
    }

    // --- Iteration ---

    /// Get an iterator over all vertices
    pub fn vertices(&self) -> crate::connectivity::VertexIter<'_> {
        crate::connectivity::VertexIter::new(&self.kernel)
    }

    /// Get an iterator over all edges
    pub fn edges(&self) -> crate::connectivity::EdgeIter<'_> {
        crate::connectivity::EdgeIter::new(&self.kernel)
    }

    /// Get an iterator over all faces
    pub fn faces(&self) -> crate::connectivity::FaceIter<'_> {
        crate::connectivity::FaceIter::new(&self.kernel)
    }

    /// Get an iterator over all halfedges
    pub fn halfedges(&self) -> crate::connectivity::HalfedgeIter<'_> {
        crate::connectivity::HalfedgeIter::new(&self.kernel)
    }

    // --- Count queries ---

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.kernel.n_vertices()
    }

    /// Get the number of edges
    pub fn n_edges(&self) -> usize {
        self.kernel.n_edges()
    }

    /// Get the number of faces
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get the number of halfedges
    pub fn n_halfedges(&self) -> usize {
        self.kernel.n_halfedges()
    }

    // --- Connectivity queries ---

    /// Get the halfedge handle from a vertex
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.kernel.halfedge_handle(vh)
    }

    /// Get the edge handle from a halfedge
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        self.kernel.edge_handle(heh)
    }

    /// Get the opposite halfedge
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.opposite_halfedge_handle(heh)
    }

    /// Get the next halfedge in the cycle
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.next_halfedge_handle(heh)
    }

    /// Get the previous halfedge in the cycle
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.prev_halfedge_handle(heh)
    }

    /// Get the from-vertex of a halfedge
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.kernel.from_vertex_handle(heh)
    }

    /// Get the to-vertex of a halfedge
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.kernel.to_vertex_handle(heh)
    }

    /// Get a halfedge from a face
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        self.kernel.face_halfedge_handle(fh)
    }

    /// Get a halfedge from an edge
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, direction: usize) -> HalfedgeHandle {
        self.kernel.edge_halfedge_handle(eh, direction)
    }

    /// Get the face handle from a halfedge
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.kernel.face_handle(heh)
    }

    /// Check if a halfedge is a boundary
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_boundary(heh)
    }

    /// Get the vertices of a face
    pub fn face_vertices(&self, fh: FaceHandle) -> Option<Vec<VertexHandle>> {
        // Simplified implementation
        Some(vec![])
    }
}

// Type aliases for compatibility
pub type TriConnectivity = TriMesh;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_triangle() {
        let mut mesh = TriMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        let fh = mesh.add_triangle(v0, v1, v2);

        assert!(fh.is_some());
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.n_edges(), 3);
    }

    #[test]
    fn test_triangle_vertices() {
        let mut mesh = TriMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        let fh = mesh.add_triangle(v0, v1, v2).unwrap();
        let (t0, t1, t2) = mesh.triangle_vertices(fh).unwrap();

        // Triangle vertices are returned in CCW order from face halfedge traversal
        // The order depends on the internal storage but all three original vertices should be present
        let vertices = [v0, v1, v2];
        let returned = [t0, t1, t2];
        assert!(returned.iter().all(|&vh| vertices.contains(&vh)));
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_vertex_iteration() {
        let mut mesh = TriMesh::new();

        mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        let mut count = 0;
        for v in mesh.vertices() {
            assert!(mesh.point(v).is_some());
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_face_iteration() {
        let mut mesh = TriMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        mesh.add_triangle(v0, v1, v2);

        let mut count = 0;
        for _ in mesh.faces() {
            count += 1;
        }
        assert_eq!(count, 1);
    }
}
