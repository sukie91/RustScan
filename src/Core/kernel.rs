//! # ArrayKernel
//! 
//! Core mesh storage using arrays (Vec) for mesh items.
//! This is the underlying storage layer for the mesh data structure.

use std::collections::HashMap;
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::items::{Vertex, Halfedge, Edge, Face};

/// Property container for mesh attributes
/// Allows attaching arbitrary data to mesh entities
#[derive(Debug)]
pub struct PropertyContainer {
    data: HashMap<String, Box<dyn std::any::Any>>,
}

impl PropertyContainer {
    /// Create a new empty property container
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Add a property with the given name
    pub fn add_property<T: 'static>(&mut self, name: &str, value: T) {
        self.data.insert(name.to_string(), Box::new(value));
    }

    /// Get a property by name
    pub fn get_property<T: 'static>(&self, name: &str) -> Option<&T> {
        self.data.get(name).and_then(|b| b.downcast_ref::<T>())
    }

    /// Get a mutable property by name
    pub fn get_property_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.data.get_mut(name).and_then(|b| b.downcast_mut::<T>())
    }

    /// Remove a property
    pub fn remove_property(&mut self, name: &str) -> bool {
        self.data.remove(name).is_some()
    }

    /// Check if property exists
    pub fn has_property(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }
}

/// Status flags for mesh entities
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusInfo {
    bits: u32,
}

impl StatusInfo {
    /// Create a new status with all flags unset
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    /// Set a bit flag
    pub fn set_bit(&mut self, bit: u32) {
        self.bits |= 1 << bit;
    }

    /// Unset a bit flag
    pub fn unset_bit(&mut self, bit: u32) {
        self.bits &= !(1 << bit);
    }

    /// Check if a bit is set
    pub fn is_bit_set(&self, bit: u32) -> bool {
        (self.bits & (1 << bit)) != 0
    }

    /// Get the raw bits
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Set bits from another status
    pub fn set_bits(&mut self, other: &Self) {
        self.bits |= other.bits;
    }
}

/// Storage for vertex properties
#[derive(Debug, Clone)]
pub struct VertexPropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for VertexPropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for edge properties
#[derive(Debug, Clone)]
pub struct EdgePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for EdgePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for face properties
#[derive(Debug, Clone)]
pub struct FacePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for FacePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for halfedge properties
#[derive(Debug, Clone)]
pub struct HalfedgePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for HalfedgePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// The ArrayKernel - core mesh storage using Vec containers
#[derive(Debug, Clone, Default)]
pub struct ArrayKernel {
    vertices: Vec<Vertex>,
    halfedges: Vec<Halfedge>,
    edges: Vec<Edge>,
    faces: Vec<Face>,

    // Property containers
    vertex_props: VertexPropertyContainer,
    edge_props: EdgePropertyContainer,
    face_props: FacePropertyContainer,
    halfedge_props: HalfedgePropertyContainer,
}

impl ArrayKernel {
    /// Create a new empty kernel
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            halfedges: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            vertex_props: VertexPropertyContainer::default(),
            edge_props: EdgePropertyContainer::default(),
            face_props: FacePropertyContainer::default(),
            halfedge_props: HalfedgePropertyContainer::default(),
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.halfedges.clear();
        self.edges.clear();
        self.faces.clear();
    }

    // --- Handle to item conversion ---

    /// Get a vertex by handle (const)
    #[inline]
    pub fn vertex(&self, vh: VertexHandle) -> Option<&Vertex> {
        self.vertices.get(vh.idx_usize())
    }

    /// Get a vertex by handle (mutable)
    #[inline]
    pub fn vertex_mut(&mut self, vh: VertexHandle) -> Option<&mut Vertex> {
        self.vertices.get_mut(vh.idx_usize())
    }

    /// Get a halfedge by handle (const)
    #[inline]
    pub fn halfedge(&self, heh: HalfedgeHandle) -> Option<&Halfedge> {
        self.halfedges.get(heh.idx_usize())
    }

    /// Get a halfedge by handle (mutable)
    #[inline]
    pub fn halfedge_mut(&mut self, heh: HalfedgeHandle) -> Option<&mut Halfedge> {
        self.halfedges.get_mut(heh.idx_usize())
    }

    /// Get an edge by handle (const)
    #[inline]
    pub fn edge(&self, eh: EdgeHandle) -> Option<&Edge> {
        self.edges.get(eh.idx_usize())
    }

    /// Get an edge by handle (mutable)
    #[inline]
    pub fn edge_mut(&mut self, eh: EdgeHandle) -> Option<&mut Edge> {
        self.edges.get_mut(eh.idx_usize())
    }

    /// Get a face by handle (const)
    #[inline]
    pub fn face(&self, fh: FaceHandle) -> Option<&Face> {
        self.faces.get(fh.idx_usize())
    }

    /// Get a face by handle (mutable)
    #[inline]
    pub fn face_mut(&mut self, fh: FaceHandle) -> Option<&mut Face> {
        self.faces.get_mut(fh.idx_usize())
    }

    // --- Unsafe direct access (for performance) ---

    /// Get vertex by index without bounds check
    /// # Safety
    /// Caller must ensure `idx` is within bounds.
    #[inline]
    pub unsafe fn vertex_unchecked(&self, idx: usize) -> glam::Vec3 {
        let ptr = self.vertices.as_ptr();
        let v = &*ptr.add(idx);
        v.point
    }

    /// Get raw vertex coordinates (x, y, z)
    /// # Safety
    /// Caller must ensure `idx` is within bounds.
    #[inline]
    pub unsafe fn vertex_raw(&self, idx: usize) -> (f32, f32, f32) {
        let ptr = self.vertices.as_ptr() as *const f32;
        let x = *ptr.add(idx * 3);
        let y = *ptr.add(idx * 3 + 1);
        let z = *ptr.add(idx * 3 + 2);
        (x, y, z)
    }

    /// Get vertex pointer for bulk processing
    /// # Safety
    /// Caller must ensure the returned pointer is not used after modification.
    #[inline]
    pub unsafe fn vertices_ptr(&self) -> *const Vertex {
        self.vertices.as_ptr()
    }

    // --- Item creation ---

    /// Add a new vertex and return its handle
    #[inline]
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        let idx = self.vertices.len() as u32;
        self.vertices.push(Vertex::new(point));
        VertexHandle::new(idx)
    }

    /// Add a new edge and return the handle to the first halfedge
    #[inline]
    pub fn add_edge(&mut self, start_vh: VertexHandle, end_vh: VertexHandle) -> HalfedgeHandle {
        let edge_idx = self.edges.len() as u32;
        let he0_idx = self.halfedges.len() as u32;
        let he1_idx = he0_idx + 1;

        // Create halfedges
        let he0 = Halfedge {
            vertex_handle: end_vh,
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: Some(HalfedgeHandle::new(he1_idx)),
            edge_idx,
        };

        let he1 = Halfedge {
            vertex_handle: start_vh,
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: Some(HalfedgeHandle::new(he0_idx)),
            edge_idx,
        };

        // Create edge with halfedge references
        let he0_handle = HalfedgeHandle::new(he0_idx);
        let he1_handle = HalfedgeHandle::new(he1_idx);
        self.edges.push(Edge::new(he0_handle, he1_handle));

        // Store halfedges
        self.halfedges.push(he0);
        self.halfedges.push(he1);

        HalfedgeHandle::new(he0_idx)
    }

    /// Add a new face and return its handle
    #[inline]
    pub fn add_face(&mut self, halfedge_handle: Option<HalfedgeHandle>) -> FaceHandle {
        let idx = self.faces.len() as u32;
        self.faces.push(Face::new(halfedge_handle));
        FaceHandle::new(idx)
    }

    // --- Count queries ---

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get vertices as slice (for efficient iteration)
    #[inline]
    pub fn vertices_slice(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Get the number of edges
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of faces
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get the number of halfedges
    pub fn n_halfedges(&self) -> usize {
        self.halfedges.len()
    }

    /// Check if vertices are empty
    pub fn vertices_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Check if edges are empty
    pub fn edges_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Check if faces are empty
    pub fn faces_empty(&self) -> bool {
        self.faces.is_empty()
    }

    // --- Status management ---

    /// Request vertex status tracking
    pub fn request_vertex_status(&mut self) {
        self.vertex_props.status = Some(StatusInfo::new());
    }

    /// Request edge status tracking
    pub fn request_edge_status(&mut self) {
        self.edge_props.status = Some(StatusInfo::new());
    }

    /// Request face status tracking
    pub fn request_face_status(&mut self) {
        self.face_props.status = Some(StatusInfo::new());
    }

    /// Get vertex status
    pub fn vertex_status(&self, vh: VertexHandle) -> Option<&StatusInfo> {
        self.vertex_props.status.as_ref()
    }

    /// Get mutable vertex status
    pub fn vertex_status_mut(&mut self, vh: VertexHandle) -> Option<&mut StatusInfo> {
        self.vertex_props.status.as_mut()
    }

    // --- Connectivity ---

    /// Get the halfedge handle from a vertex
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.vertex(vh).and_then(|v| v.halfedge_handle)
    }

    /// Set the halfedge handle for a vertex
    pub fn set_halfedge_handle(&mut self, vh: VertexHandle, heh: HalfedgeHandle) {
        if let Some(v) = self.vertex_mut(vh) {
            v.halfedge_handle = Some(heh);
        }
    }

    /// Check if a vertex is isolated
    pub fn is_isolated(&self, vh: VertexHandle) -> bool {
        self.halfedge_handle(vh).is_none()
    }

    /// Get the to-vertex of a halfedge
    #[inline]
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.halfedge(heh)
            .map(|he| he.vertex_handle)
            .unwrap_or(VertexHandle::invalid())
    }

    /// Get the from-vertex of a halfedge
    #[inline]
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        // Get opposite halfedge to find from vertex
        let opp_heh = self.opposite_halfedge_handle(heh);
        self.halfedge(opp_heh)
            .map(|he| he.vertex_handle)
            .unwrap_or(VertexHandle::invalid())
    }

    /// Get the opposite halfedge
    #[inline]
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.halfedge(heh)
            .and_then(|he| he.opposite_halfedge_handle)
            .unwrap_or(HalfedgeHandle::invalid())
    }

    /// Get the edge handle from a halfedge
    #[inline]
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        EdgeHandle::new(heh.idx() >> 1)
    }

    /// Get the halfedge handle from an edge (0 or 1)
    #[inline]
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        HalfedgeHandle::new((eh.idx() << 1) + idx as u32)
    }

    /// Get the face handle from a halfedge
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.halfedge(heh).and_then(|he| he.face_handle)
    }

    /// Set the face handle for a halfedge
    pub fn set_face_handle(&mut self, heh: HalfedgeHandle, fh: FaceHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.face_handle = Some(fh);
        }
    }

    /// Set a halfedge as boundary (no face)
    pub fn set_boundary(&mut self, heh: HalfedgeHandle) {
        // In a full implementation, store in halfedge data
    }

    /// Check if a halfedge is a boundary
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        // Simplified implementation - check if face handle is invalid
        let fh = self.face_handle(heh);
        fh.map(|f| !f.is_valid()).unwrap_or(true)
    }

    // --- Halfedge cycle navigation ---

    /// Get the next halfedge in the cycle
    #[inline]
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.halfedge(heh)
            .and_then(|he| he.next_halfedge_handle)
            .unwrap_or(HalfedgeHandle::invalid())
    }

    /// Set the next halfedge in the cycle
    pub fn set_next_halfedge_handle(&mut self, heh: HalfedgeHandle, next_heh: HalfedgeHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.next_halfedge_handle = Some(next_heh);
        }
        // Also set prev of next
        if let Some(he) = self.halfedge_mut(next_heh) {
            he.prev_halfedge_handle = Some(heh);
        }
    }

    /// Get the previous halfedge in the cycle
    #[inline]
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.halfedge(heh)
            .and_then(|he| he.prev_halfedge_handle)
            .unwrap_or(HalfedgeHandle::invalid())
    }

    /// Set the prev halfedge in the cycle
    pub fn set_prev_halfedge_handle(&mut self, heh: HalfedgeHandle, prev_heh: HalfedgeHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.prev_halfedge_handle = Some(prev_heh);
        }
    }

    /// Get the from-vertex of a halfedge as Option
    pub fn from_vertex_handle_opt(&self, heh: HalfedgeHandle) -> Option<VertexHandle> {
        self.halfedge(self.opposite_halfedge_handle(heh))
            .map(|he| he.vertex_handle)
    }

    /// Get the face halfedge handle
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        self.face(fh).and_then(|f| f.halfedge_handle)
    }

    /// Add a triangle face (optimized)
    #[inline]
    pub fn add_face_triangle(&mut self, halfedge_handle: HalfedgeHandle) -> FaceHandle {
        let idx = self.faces.len() as u32;
        self.faces.push(Face::new(Some(halfedge_handle)));
        FaceHandle::new(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut kernel = ArrayKernel::new();
        
        // Add vertices
        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = kernel.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        assert_eq!(kernel.n_vertices(), 3);
        assert!(kernel.vertex(v0).is_some());
        
        // Add edge
        let he = kernel.add_edge(v0, v1);
        assert_eq!(kernel.n_edges(), 1);
        assert_eq!(kernel.n_halfedges(), 2);
        
        // Check connectivity
        assert_eq!(kernel.edge_handle(he), EdgeHandle::new(0));
        assert_eq!(kernel.opposite_halfedge_handle(he).idx(), he.idx() ^ 1);
    }

    #[test]
    fn test_status_management() {
        let mut kernel = ArrayKernel::new();
        kernel.request_vertex_status();
        kernel.request_edge_status();
        
        // Status should be available after request
        assert!(kernel.vertex_status(VertexHandle::new(0)).is_some());
    }
}
