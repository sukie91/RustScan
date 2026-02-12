//! # Circulators
//!
//! Circulators provide iterator-based traversal of adjacent mesh elements.

use crate::connectivity::PolyMeshSoA;
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};

/// Vertex-Vertex Circulator: Visit all unique vertices adjacent to a vertex
pub struct VertexVertexCirculator<'a> {
    mesh: &'a PolyMeshSoA,
    start_vh: VertexHandle,
    current_heh: HalfedgeHandle,
}

impl<'a> Iterator for VertexVertexCirculator<'a> {
    type Item = VertexHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Get the neighbor from current halfedge (it points to the neighbor)
        let next_target = self.mesh.to_vertex_handle(self.current_heh);
        
        // Check if we've come back to start vertex (completed the cycle)
        if next_target == self.start_vh {
            return None;
        }
        
        // Move to next halfedge around the vertex:
        // current_heh points to neighbor N1
        // We want to find the next neighbor N2
        // Go to opposite (back to start), then next halfedge in the face
        let opposite = self.mesh.opposite_halfedge_handle(self.current_heh);
        let next_heh = self.mesh.next_halfedge_handle(opposite);
        
        // Check termination
        if next_heh == self.current_heh || !next_heh.is_valid() {
            return None;
        }
        
        self.current_heh = next_heh;
        Some(next_target)
    }
}

impl<'a> PolyMeshSoA {
    pub fn vertex_vertices(&'a self, vh: VertexHandle) -> Option<VertexVertexCirculator<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        Some(VertexVertexCirculator {
            mesh: self,
            start_vh: vh,
            current_heh: start_heh,
        })
    }
}

/// Vertex-Face Circulator: Visit all faces adjacent to a vertex
pub struct VertexFaceCirculator<'a> {
    mesh: &'a PolyMeshSoA,
    start_vh: VertexHandle,
    current_heh: HalfedgeHandle,
}

impl<'a> Iterator for VertexFaceCirculator<'a> {
    type Item = FaceHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Get the face from current halfedge
        let fh = self.mesh.face_handle(self.current_heh);
        
        // Check if we've come back to start vertex (completed the cycle)
        let next_target = self.mesh.to_vertex_handle(self.current_heh);
        if next_target == self.start_vh {
            return None;
        }
        
        // Move to next halfedge around the vertex
        let opposite = self.mesh.opposite_halfedge_handle(self.current_heh);
        let next_heh = self.mesh.next_halfedge_handle(opposite);
        
        // Check termination
        if next_heh == self.current_heh || !next_heh.is_valid() {
            return None;
        }
        
        self.current_heh = next_heh;
        fh
    }
}

impl<'a> PolyMeshSoA {
    pub fn vertex_faces(&'a self, vh: VertexHandle) -> Option<VertexFaceCirculator<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        Some(VertexFaceCirculator {
            mesh: self,
            start_vh: vh,
            current_heh: start_heh,
        })
    }
}

/// Vertex-Halfedge Iterator: Visit all outgoing halfedges around a vertex
pub struct VertexHalfedgeIter<'a> {
    mesh: &'a PolyMeshSoA,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    first: bool,
}

impl<'a> Iterator for VertexHalfedgeIter<'a> {
    type Item = HalfedgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.first && self.current_heh == self.start_heh {
            return None;
        }
        
        let heh = self.current_heh;
        
        // Get the opposite halfedge (incoming to our vertex)
        let incoming = self.mesh.opposite_halfedge_handle(self.current_heh);
        
        // Get previous halfedge in the face cycle (going counter-clockwise)
        let prev_incoming = self.mesh.prev_halfedge_handle(incoming);
        
        // The previous of incoming is another outgoing halfedge from our vertex
        // But we need to check if prev_incoming is valid and different
        if prev_incoming == incoming || !prev_incoming.is_valid() {
            return None;
        }
        
        self.first = false;
        self.current_heh = prev_incoming;

        Some(heh)
    }
}

impl<'a> PolyMeshSoA {
    pub fn vertex_halfedges(&'a self, vh: VertexHandle) -> Option<VertexHalfedgeIter<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        Some(VertexHalfedgeIter {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            first: true,
        })
    }
}

/// Vertex-Edge Iterator: Visit all edges adjacent to a vertex
pub struct VertexEdgeIter<'a> {
    mesh: &'a PolyMeshSoA,
    halfedges: VertexHalfedgeIter<'a>,
}

impl<'a> Iterator for VertexEdgeIter<'a> {
    type Item = EdgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        self.halfedges.next().map(|heh| self.mesh.edge_handle(heh))
    }
}

impl<'a> PolyMeshSoA {
    pub fn vertex_edges(&'a self, vh: VertexHandle) -> Option<VertexEdgeIter<'a>> {
        let halfedges = self.vertex_halfedges(vh)?;
        Some(VertexEdgeIter {
            mesh: self,
            halfedges,
        })
    }
}

/// Face-Vertex Circulator: Visit all vertices of a face
pub struct FaceVertexCirculator<'a> {
    mesh: &'a PolyMeshSoA,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
}

impl<'a> Iterator for FaceVertexCirculator<'a> {
    type Item = VertexHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        
        let vh = self.mesh.from_vertex_handle(self.current_heh);
        
        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);
        
        if self.current_heh == self.start_heh {
            self.done = true;
        }
        
        Some(vh)
    }
}

impl<'a> PolyMeshSoA {
    pub fn face_vertices(&'a self, fh: FaceHandle) -> Option<FaceVertexCirculator<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        Some(FaceVertexCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
        })
    }
}

/// Face-Halfedge Iterator: Visit all halfedges in a face cycle
pub struct FaceHalfedgeIter<'a> {
    mesh: &'a PolyMeshSoA,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
}

impl<'a> Iterator for FaceHalfedgeIter<'a> {
    type Item = HalfedgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let heh = self.current_heh;
        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);
        
        // Check if we've completed the cycle AFTER advancing
        if self.current_heh == self.start_heh {
            self.done = true;
        }
        
        Some(heh)
    }
}

impl<'a> PolyMeshSoA {
    pub fn face_halfedges(&'a self, fh: FaceHandle) -> Option<FaceHalfedgeIter<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        Some(FaceHalfedgeIter {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
        })
    }
}

/// Face-Edge Iterator: Visit all edges of a face
pub struct FaceEdgeIter<'a> {
    mesh: &'a PolyMeshSoA,
    halfedges: FaceHalfedgeIter<'a>,
}

impl<'a> Iterator for FaceEdgeIter<'a> {
    type Item = EdgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        self.halfedges.next().map(|heh| self.mesh.edge_handle(heh))
    }
}

impl<'a> PolyMeshSoA {
    pub fn face_edges(&'a self, fh: FaceHandle) -> Option<FaceEdgeIter<'a>> {
        let halfedges = self.face_halfedges(fh)?;
        Some(FaceEdgeIter {
            mesh: self,
            halfedges,
        })
    }
}

/// Face-Face Circulator: Visit all adjacent faces
pub struct FaceFaceCirculator<'a> {
    mesh: &'a PolyMeshSoA,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
}

impl<'a> Iterator for FaceFaceCirculator<'a> {
    type Item = FaceHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        
        let opposite = self.mesh.opposite_halfedge_handle(self.current_heh);
        let fh = self.mesh.face_handle(opposite);
        
        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);
        
        if self.current_heh == self.start_heh {
            self.done = true;
        }
        
        fh
    }
}

impl<'a> PolyMeshSoA {
    pub fn face_faces(&'a self, fh: FaceHandle) -> Option<FaceFaceCirculator<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        Some(FaceFaceCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_tetrahedron;

    #[test]
    fn test_edge_count() {
        let mesh = generate_tetrahedron();
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_edges(), 6);
        assert_eq!(mesh.n_faces(), 4);
    }

    #[test]
    fn test_basic_iteration() {
        let mesh = generate_tetrahedron();
        let v_count = mesh.vertices().count();
        assert_eq!(v_count, 4);
        
        let f_count = mesh.faces().count();
        assert_eq!(f_count, 4);
    }

    #[test]
    fn test_halfedge_structure() {
        // Verify that halfedge next pointers are correctly set
        let mesh = generate_tetrahedron();
        
        // Each face should have 3 halfedges in a cycle
        for fh in mesh.faces() {
            let start_heh = mesh.face_halfedge_handle(fh).expect("Face should have halfedge");
            
            eprintln!("Face {:?}: start_heh = {:?}", fh, start_heh);
            
            // Follow the cycle
            let mut count = 0;
            let mut heh = start_heh;
            loop {
                count += 1;
                if count > 10 {
                    panic!("More than 10 halfedges in face cycle - next pointers broken!");
                }
                
                let next = mesh.next_halfedge_handle(heh);
                eprintln!("  heh={:?} -> next={:?}", heh, next);
                if next == start_heh {
                    break;
                }
                heh = next;
            }
            
            assert_eq!(count, 3, "Face should have exactly 3 halfedges in cycle");
        }
    }

    #[test]
    fn test_vertex_vertex_circulator() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);
        
        let neighbors: Vec<_> = match mesh.vertex_vertices(v0) {
            Some(c) => c.collect(),
            None => panic!("No circulator for vertex 0"),
        };
        
        // Tetrahedron: each vertex has 3 neighbors
        assert_eq!(neighbors.len(), 3, "Expected 3 neighbors, got {:?}", neighbors);
        
        // Verify neighbors are unique and not self
        let mut unique: Vec<_> = neighbors.iter().filter(|&&vh| vh != v0).collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 3, "All 3 neighbors should be unique and not self");
    }

    #[test]
    fn test_vertex_face_circulator() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);
        
        let faces: Vec<_> = match mesh.vertex_faces(v0) {
            Some(c) => c.collect(),
            None => panic!("No circulator for vertex 0"),
        };
        
        // Tetrahedron: each vertex belongs to 3 faces
        assert_eq!(faces.len(), 3, "Expected 3 faces, got {:?}", faces);
    }

    #[test]
    fn test_face_vertex_circulator() {
        let mesh = generate_tetrahedron();
        let fh = FaceHandle::from_usize(0);
        
        let vertices: Vec<_> = match mesh.face_vertices(fh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for face 0"),
        };
        
        // Each face is a triangle
        assert_eq!(vertices.len(), 3, "Expected 3 vertices, got {:?}", vertices);
    }

    #[test]
    fn test_face_face_circulator() {
        let mesh = generate_tetrahedron();
        let fh = FaceHandle::from_usize(0);
        
        let neighbors: Vec<_> = match mesh.face_faces(fh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for face 0"),
        };
        
        // Tetrahedron: each face has 3 adjacent faces
        assert_eq!(neighbors.len(), 3, "Expected 3 adjacent faces, got {:?}", neighbors);
    }

    #[test]
    fn test_vertex_halfedge_iter() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        // Get via circulator
        let halfedges: Vec<_> = match mesh.vertex_halfedges(v0) {
            Some(c) => c.collect(),
            None => panic!("No iterator for vertex 0"),
        };

        // Tetrahedron: each vertex has 3 outgoing halfedges
        assert_eq!(halfedges.len(), 3, "Expected 3 halfedges, got {:?}", halfedges);

        // All returned halfedges should be outgoing from v0
        for heh in &halfedges {
            assert_eq!(mesh.from_vertex_handle(*heh), v0,
                "Halfedge {:?} should be outgoing from vertex {:?}", heh, v0);
        }
    }

    #[test]
    fn test_vertex_edge_iter() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        let edges: Vec<_> = match mesh.vertex_edges(v0) {
            Some(c) => c.collect(),
            None => panic!("No iterator for vertex 0"),
        };
        let halfedge_edges: Vec<_> = match mesh.vertex_halfedges(v0) {
            Some(c) => c.map(|heh| mesh.edge_handle(heh)).collect(),
            None => panic!("No iterator for vertex 0"),
        };

        assert_eq!(edges.len(), 3, "Expected 3 edges, got {:?}", edges);

        let mut sorted_edges = edges.clone();
        sorted_edges.sort();
        let mut sorted_halfedge_edges = halfedge_edges.clone();
        sorted_halfedge_edges.sort();
        assert_eq!(
            sorted_edges, sorted_halfedge_edges,
            "vertex_edges should match vertex_halfedges + edge_handle"
        );

        sorted_edges.dedup();
        assert_eq!(sorted_edges.len(), 3, "Edges around vertex should be unique");
    }

    #[test]
    fn test_face_halfedge_iter() {
        let mesh = generate_tetrahedron();

        for fh in mesh.faces() {
            let halfedges: Vec<_> = match mesh.face_halfedges(fh) {
                Some(c) => c.collect(),
                None => panic!("No iterator for face {:?}", fh),
            };

            assert_eq!(halfedges.len(), 3, "Expected 3 halfedges, got {:?}", halfedges);

            for heh in &halfedges {
                assert_eq!(
                    mesh.face_handle(*heh),
                    Some(fh),
                    "Face halfedge must belong to queried face"
                );
            }
        }
    }

    #[test]
    fn test_face_edge_iter() {
        let mesh = generate_tetrahedron();

        for fh in mesh.faces() {
            let edges: Vec<_> = match mesh.face_edges(fh) {
                Some(c) => c.collect(),
                None => panic!("No iterator for face {:?}", fh),
            };
            let halfedge_edges: Vec<_> = match mesh.face_halfedges(fh) {
                Some(c) => c.map(|heh| mesh.edge_handle(heh)).collect(),
                None => panic!("No iterator for face {:?}", fh),
            };

            assert_eq!(edges.len(), 3, "Expected 3 edges, got {:?}", edges);

            let mut sorted_edges = edges.clone();
            sorted_edges.sort();
            let mut sorted_halfedge_edges = halfedge_edges.clone();
            sorted_halfedge_edges.sort();
            assert_eq!(
                sorted_edges, sorted_halfedge_edges,
                "face_edges should match face_halfedges + edge_handle"
            );

            sorted_edges.dedup();
            assert_eq!(sorted_edges.len(), 3, "Edges in a face should be unique");
        }
    }

    #[test]
    fn test_new_iterators_invalid_handles() {
        let mesh = generate_tetrahedron();

        assert!(mesh.vertex_halfedges(VertexHandle::invalid()).is_none());
        assert!(mesh.vertex_edges(VertexHandle::invalid()).is_none());
        assert!(mesh.face_halfedges(FaceHandle::invalid()).is_none());
        assert!(mesh.face_edges(FaceHandle::invalid()).is_none());
    }
}
