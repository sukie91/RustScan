//! # Mesh Items
//! 
//! Core mesh data structures: Vertex, Halfedge, Edge, Face.
//! These represent the fundamental elements stored in the mesh.

use crate::handles::{VertexHandle, HalfedgeHandle, FaceHandle};
use glam::Vec3;

/// A vertex in the mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// Position of the vertex in 3D space
    pub point: Vec3,
    /// Handle to one of the outgoing halfedges
    pub halfedge_handle: Option<HalfedgeHandle>,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            point: Vec3::ZERO,
            halfedge_handle: None,
        }
    }
}

impl Vertex {
    /// Create a new vertex at the given position
    pub fn new(point: Vec3) -> Self {
        Self {
            point,
            halfedge_handle: None,
        }
    }

    /// Check if the vertex is isolated (no connected edges)
    pub fn is_isolated(&self) -> bool {
        self.halfedge_handle.is_none()
    }
}

/// A halfedge in the mesh (directional edge)
/// Each edge consists of two halfedges in opposite directions
#[derive(Debug, Clone, PartialEq)]
pub struct Halfedge {
    /// The vertex this halfedge points to
    pub vertex_handle: VertexHandle,
    /// The face this halfedge borders (None if boundary)
    pub face_handle: Option<FaceHandle>,
    /// The next halfedge in the face boundary
    pub next_halfedge_handle: Option<HalfedgeHandle>,
    /// The previous halfedge in the face boundary
    pub prev_halfedge_handle: Option<HalfedgeHandle>,
    /// The opposite halfedge (same edge, opposite direction)
    pub opposite_halfedge_handle: Option<HalfedgeHandle>,
    /// The edge index this halfedge belongs to
    pub edge_idx: u32,
}

impl Default for Halfedge {
    #[inline]
    fn default() -> Self {
        Self {
            vertex_handle: VertexHandle::default(),
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: None,
            edge_idx: u32::MAX,
        }
    }
}

impl Halfedge {
    /// Check if this is a boundary halfedge
    pub fn is_boundary(&self) -> bool {
        self.face_handle.is_none()
    }
}

/// An edge in the mesh (undirected connection between two vertices)
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// The two halfedges belonging to this edge
    pub halfedges: [HalfedgeHandle; 2],
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            halfedges: [HalfedgeHandle::default(), HalfedgeHandle::default()],
        }
    }
}

impl Edge {
    /// Create a new edge with the given halfedge handles
    pub fn new(halfedge0: HalfedgeHandle, halfedge1: HalfedgeHandle) -> Self {
        Self {
            halfedges: [halfedge0, halfedge1],
        }
    }

    /// Get the halfedge at the given index (0 or 1)
    pub fn halfedge(&self, idx: usize) -> HalfedgeHandle {
        assert!(idx < 2, "Halfedge index must be 0 or 1");
        self.halfedges[idx]
    }
}

/// A face in the mesh (polygon)
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// Handle to one of the halfedges bordering this face
    pub halfedge_handle: Option<HalfedgeHandle>,
}

impl Default for Face {
    fn default() -> Self {
        Self {
            halfedge_handle: None,
        }
    }
}

impl Face {
    /// Create a new face with the given halfedge handle
    pub fn new(halfedge_handle: Option<HalfedgeHandle>) -> Self {
        Self {
            halfedge_handle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handles::{VertexHandle, HalfedgeHandle, FaceHandle};

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::new(glam::vec3(1.0, 2.0, 3.0));
        assert_eq!(v.point, glam::vec3(1.0, 2.0, 3.0));
        assert!(v.is_isolated());
    }

    #[test]
    fn test_edge_creation() {
        let he0 = HalfedgeHandle::new(0);
        let he1 = HalfedgeHandle::new(1);
        let e = Edge::new(he0, he1);
        assert_eq!(e.halfedge(0), he0);
        assert_eq!(e.halfedge(1), he1);
    }

    #[test]
    fn test_face_creation() {
        let he = HalfedgeHandle::new(5);
        let f = Face::new(Some(he));
        assert_eq!(f.halfedge_handle, Some(he));
    }
}
