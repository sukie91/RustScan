//! # PolyConnectivity
//!
//! Polygonal mesh connectivity implementation.
//! Provides iteration and circulation over mesh elements.

use crate::attrib_soa_kernel::{
    EPropHandle, FPropHandle, HPropHandle, PropHandle, PropValue, VPropHandle,
};
use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
use crate::soa_kernel::SoAKernel;
use std::collections::{HashMap, VecDeque};

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
        Self {
            current: 0,
            end: n_vertices,
        }
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
        Self {
            current: 0,
            end: n_faces,
        }
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

/// Vertex-normal accumulation mode for recomputation APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexNormalWeighting {
    /// Sum unnormalized face area normals before the final normalization.
    AreaWeighted,
    /// Sum unit face normals equally, matching OpenMesh's default vertex-normal behavior.
    FaceAverage,
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

    /// Get the number of active vertices referenced by at least one face.
    pub fn active_vertex_count(&self) -> usize {
        let mut used = vec![false; self.n_vertices()];

        for fh in self.faces() {
            if self.face_halfedge_handle(fh).is_none() {
                continue;
            }

            for vh in self.face_vertices_vec(fh) {
                let idx = vh.idx_usize();
                if idx < used.len() {
                    used[idx] = true;
                }
            }
        }

        let active = used.into_iter().filter(|used| *used).count();
        if active == 0 {
            self.vertices()
                .filter(|&vh| self.halfedge_handle(vh).is_some())
                .count()
        } else {
            active
        }
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

    // --- Dynamic properties ---

    pub fn add_vertex_property<T: PropValue>(&mut self, name: &str) -> VPropHandle<T> {
        self.kernel.add_vertex_property(name)
    }

    pub fn add_halfedge_property<T: PropValue>(&mut self, name: &str) -> HPropHandle<T> {
        self.kernel.add_halfedge_property(name)
    }

    pub fn add_edge_property<T: PropValue>(&mut self, name: &str) -> EPropHandle<T> {
        self.kernel.add_edge_property(name)
    }

    pub fn add_face_property<T: PropValue>(&mut self, name: &str) -> FPropHandle<T> {
        self.kernel.add_face_property(name)
    }

    pub fn vertex_property<T: PropValue>(
        &self,
        handle: VPropHandle<T>,
        vh: VertexHandle,
    ) -> Option<T> {
        self.kernel.vertex_property(handle, vh)
    }

    pub fn halfedge_property<T: PropValue>(
        &self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
    ) -> Option<T> {
        self.kernel.halfedge_property(handle, heh)
    }

    pub fn edge_property<T: PropValue>(&self, handle: EPropHandle<T>, eh: EdgeHandle) -> Option<T> {
        self.kernel.edge_property(handle, eh)
    }

    pub fn face_property<T: PropValue>(&self, handle: FPropHandle<T>, fh: FaceHandle) -> Option<T> {
        self.kernel.face_property(handle, fh)
    }

    pub fn set_vertex_property<T: PropValue>(
        &mut self,
        handle: VPropHandle<T>,
        vh: VertexHandle,
        value: T,
    ) -> bool {
        self.kernel.set_vertex_property(handle, vh, value)
    }

    pub fn set_halfedge_property<T: PropValue>(
        &mut self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
        value: T,
    ) -> bool {
        self.kernel.set_halfedge_property(handle, heh, value)
    }

    pub fn set_edge_property<T: PropValue>(
        &mut self,
        handle: EPropHandle<T>,
        eh: EdgeHandle,
        value: T,
    ) -> bool {
        self.kernel.set_edge_property(handle, eh, value)
    }

    pub fn set_face_property<T: PropValue>(
        &mut self,
        handle: FPropHandle<T>,
        fh: FaceHandle,
        value: T,
    ) -> bool {
        self.kernel.set_face_property(handle, fh, value)
    }

    pub fn add_property<T: PropValue>(&mut self, name: &str) -> PropHandle<T> {
        self.add_vertex_property(name)
    }

    pub fn get_property<T: PropValue>(&self, handle: PropHandle<T>, idx: usize) -> Option<T> {
        self.vertex_property(handle, VertexHandle::from_usize(idx))
    }

    pub fn set_property<T: PropValue>(
        &mut self,
        handle: PropHandle<T>,
        idx: usize,
        value: T,
    ) -> bool {
        self.set_vertex_property(handle, VertexHandle::from_usize(idx), value)
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

        // Set vertex halfedge handles - use outgoing halfedge from each vertex
        for (i, &vh) in vertices.iter().enumerate() {
            // halfedges[i] goes FROM vertices[i] TO vertices[(i+1)%n]
            // This is the outgoing halfedge from vh, which is what circulators expect
            self.kernel.set_halfedge_handle(vh, halfedges[i]);
        }

        Some(fh)
    }

    /// OpenMesh-style face insertion path used only for parity debugging.
    ///
    /// This mirrors OpenMesh's patch relinking and conditional vertex-anchor
    /// adjustment more closely than the library's default `add_face()`.
    pub fn add_face_openmesh_parity(&mut self, vertices: &[VertexHandle]) -> Option<FaceHandle> {
        #[derive(Clone, Copy)]
        struct FaceEdgeData {
            halfedge_handle: HalfedgeHandle,
            is_new: bool,
            needs_adjust: bool,
        }

        if vertices.len() < 3 {
            return None;
        }

        let n = vertices.len();
        let mut edge_data: Vec<FaceEdgeData> = Vec::with_capacity(n);
        let mut next_cache: Vec<(HalfedgeHandle, HalfedgeHandle)> = Vec::with_capacity(6 * n);

        for i in 0..n {
            let start = vertices[i];
            let end = vertices[(i + 1) % n];

            if !self.is_boundary_vertex_for_parity_face(start) {
                return None;
            }

            let existing_heh = self.find_halfedge_between(start, end);
            let is_new = existing_heh.is_none();
            let halfedge_handle = existing_heh.unwrap_or_else(HalfedgeHandle::invalid);

            if let Some(heh) = existing_heh {
                if !self.is_boundary(heh) {
                    return None;
                }
            }

            edge_data.push(FaceEdgeData {
                halfedge_handle,
                is_new,
                needs_adjust: false,
            });
        }

        for i in 0..n {
            let ii = (i + 1) % n;
            if edge_data[i].is_new || edge_data[ii].is_new {
                continue;
            }

            let inner_prev = edge_data[i].halfedge_handle;
            let inner_next = edge_data[ii].halfedge_handle;
            if self.kernel.next_halfedge_handle(inner_prev) == Some(inner_next) {
                continue;
            }

            let outer_prev = self.opposite_halfedge_handle(inner_next);
            if !outer_prev.is_valid() {
                return None;
            }

            let mut boundary_prev = outer_prev;
            let mut found_boundary = false;
            for _ in 0..self.n_halfedges().max(1) {
                let next = self
                    .kernel
                    .next_halfedge_handle(boundary_prev)
                    .unwrap_or_else(HalfedgeHandle::invalid);
                if !next.is_valid() {
                    return None;
                }
                boundary_prev = self.opposite_halfedge_handle(next);
                if !boundary_prev.is_valid() {
                    return None;
                }
                if self.is_boundary(boundary_prev) {
                    found_boundary = true;
                    break;
                }
            }

            if !found_boundary || boundary_prev == inner_prev {
                return None;
            }

            let boundary_next = self
                .kernel
                .next_halfedge_handle(boundary_prev)
                .unwrap_or_else(HalfedgeHandle::invalid);
            let patch_start = self
                .kernel
                .next_halfedge_handle(inner_prev)
                .unwrap_or_else(HalfedgeHandle::invalid);
            let patch_end = self
                .kernel
                .prev_halfedge_handle(inner_next)
                .unwrap_or_else(HalfedgeHandle::invalid);
            if !boundary_next.is_valid() || !patch_start.is_valid() || !patch_end.is_valid() {
                return None;
            }

            next_cache.push((boundary_prev, patch_start));
            next_cache.push((patch_end, boundary_next));
            next_cache.push((inner_prev, inner_next));
        }

        for i in 0..n {
            if edge_data[i].is_new {
                edge_data[i].halfedge_handle = self.add_edge(vertices[i], vertices[(i + 1) % n]);
            }
        }

        let fh = self.kernel.add_face(Some(edge_data[n - 1].halfedge_handle));

        for i in 0..n {
            let ii = (i + 1) % n;
            let vh = vertices[ii];
            let inner_prev = edge_data[i].halfedge_handle;
            let inner_next = edge_data[ii].halfedge_handle;

            let mut id = 0u8;
            if edge_data[i].is_new {
                id |= 1;
            }
            if edge_data[ii].is_new {
                id |= 2;
            }

            if id != 0 {
                let outer_prev = self.opposite_halfedge_handle(inner_next);
                let outer_next = self.opposite_halfedge_handle(inner_prev);
                if !outer_prev.is_valid() || !outer_next.is_valid() {
                    return None;
                }

                match id {
                    1 => {
                        let boundary_prev = self
                            .kernel
                            .prev_halfedge_handle(inner_next)
                            .unwrap_or_else(HalfedgeHandle::invalid);
                        if !boundary_prev.is_valid() {
                            return None;
                        }
                        next_cache.push((boundary_prev, outer_next));
                        self.kernel.set_halfedge_handle(vh, outer_next);
                    }
                    2 => {
                        let boundary_next = self
                            .kernel
                            .next_halfedge_handle(inner_prev)
                            .unwrap_or_else(HalfedgeHandle::invalid);
                        if !boundary_next.is_valid() {
                            return None;
                        }
                        next_cache.push((outer_prev, boundary_next));
                        self.kernel.set_halfedge_handle(vh, boundary_next);
                    }
                    3 => {
                        if self.halfedge_handle(vh).is_none() {
                            self.kernel.set_halfedge_handle(vh, outer_next);
                            next_cache.push((outer_prev, outer_next));
                        } else {
                            let boundary_next = self.halfedge_handle(vh)?;
                            let boundary_prev = self
                                .kernel
                                .prev_halfedge_handle(boundary_next)
                                .unwrap_or_else(HalfedgeHandle::invalid);
                            if !boundary_prev.is_valid() {
                                return None;
                            }
                            next_cache.push((boundary_prev, outer_next));
                            next_cache.push((outer_prev, boundary_next));
                        }
                    }
                    _ => unreachable!(),
                }

                next_cache.push((inner_prev, inner_next));
            } else {
                edge_data[ii].needs_adjust = self.halfedge_handle(vh) == Some(inner_next);
            }

            self.kernel.set_face_handle(inner_prev, fh);
        }

        for (from, to) in next_cache {
            self.kernel.set_next_halfedge_handle(from, to);
        }

        for i in 0..n {
            if edge_data[i].needs_adjust {
                self.adjust_outgoing_halfedge(vertices[i]);
            }
        }

        Some(fh)
    }

    /// Get the number of faces (includes deleted faces)
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get the number of active (non-deleted) faces
    #[inline]
    pub fn n_active_faces(&self) -> usize {
        self.kernel.n_active_faces()
    }

    /// Get the number of halfedges
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.kernel.n_halfedges()
    }

    fn find_halfedge_between(
        &self,
        start_vh: VertexHandle,
        end_vh: VertexHandle,
    ) -> Option<HalfedgeHandle> {
        self.kernel.find_halfedge(start_vh, end_vh)
    }

    /// Find an edge between two vertices.
    pub fn find_edge_between(&self, v0: VertexHandle, v1: VertexHandle) -> Option<EdgeHandle> {
        self.find_halfedge_between(v0, v1)
            .map(|heh| self.edge_handle(heh))
    }

    fn boundary_outgoing_halfedge(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return None;
        }

        if let Some(heh) = self.halfedge_handle(vh) {
            if self.from_vertex_handle(heh) == vh && self.is_boundary(heh) {
                return Some(heh);
            }
        }

        for heh_idx in 0..self.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(heh) || self.is_edge_deleted(self.edge_handle(heh)) {
                continue;
            }
            if self.from_vertex_handle(heh) == vh && self.is_boundary(heh) {
                return Some(heh);
            }
        }

        None
    }

    fn is_boundary_vertex_for_parity_face(&self, vh: VertexHandle) -> bool {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return false;
        }

        self.halfedge_handle(vh).is_none() || self.boundary_outgoing_halfedge(vh).is_some()
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

    #[inline]
    pub fn is_vertex_deleted(&self, vh: VertexHandle) -> bool {
        self.kernel.is_vertex_deleted(vh)
    }

    #[inline]
    pub fn is_halfedge_deleted(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_halfedge_deleted(heh)
    }

    #[inline]
    pub fn is_edge_deleted(&self, eh: EdgeHandle) -> bool {
        self.kernel.is_edge_deleted(eh)
    }

    #[inline]
    pub fn is_face_deleted(&self, fh: FaceHandle) -> bool {
        self.kernel.is_face_deleted(fh)
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
        println!(
            "Euler characteristic: {} (V={}, E={}, F={})",
            euler, n_vertices, n_edges, n_faces
        );

        // Check: halfedges should be 2 * edges
        if n_halfedges != 2 * n_edges {
            return Err(format!(
                "Halfedge count mismatch: {} != 2 * {}",
                n_halfedges, n_edges
            ));
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

    pub(crate) fn next_outgoing_halfedge(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        if !heh.is_valid() {
            return HalfedgeHandle::invalid();
        }

        let center = self.from_vertex_handle(heh);
        if !center.is_valid() {
            return HalfedgeHandle::invalid();
        }

        let rotated = self.next_halfedge_handle(self.opposite_halfedge_handle(heh));
        if rotated.is_valid() && rotated != heh && self.from_vertex_handle(rotated) == center {
            return rotated;
        }

        if let Some(boundary_heh) = self.halfedge_handle(center) {
            if boundary_heh != heh
                && self.is_boundary(boundary_heh)
                && self.from_vertex_handle(boundary_heh) == center
            {
                return boundary_heh;
            }
        }

        let mut first_outgoing = HalfedgeHandle::invalid();
        for idx in 0..self.n_halfedges() {
            let candidate = HalfedgeHandle::new(idx as u32);
            if self.is_halfedge_deleted(candidate)
                || self.is_edge_deleted(self.edge_handle(candidate))
                || candidate == heh
                || self.from_vertex_handle(candidate) != center
            {
                continue;
            }

            if self.is_boundary(candidate) {
                return candidate;
            }
            if !first_outgoing.is_valid() {
                first_outgoing = candidate;
            }
        }

        first_outgoing
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

    #[inline]
    fn set_face_halfedge_handle(&mut self, fh: FaceHandle, heh: Option<HalfedgeHandle>) {
        self.kernel.set_face_halfedge_handle(fh, heh);
    }

    #[inline]
    fn clear_face_handle(&mut self, heh: HalfedgeHandle) {
        self.kernel.clear_face_handle(heh);
    }

    #[inline]
    fn clear_halfedge_handle(&mut self, vh: VertexHandle) {
        self.kernel.clear_halfedge_handle(vh);
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
    /// Returns true if the halfedge can be collapsed without creating topological issues.
    ///
    /// Checks:
    /// 1. For triangle faces: link condition (1-ring intersection)
    /// 2. For polygon faces: no duplicate edges after collapse
    /// 3. Boundary consistency
    pub fn is_collapse_ok(&self, heh: HalfedgeHandle) -> bool {
        if !heh.is_valid() {
            return false;
        }
        if self.is_halfedge_deleted(heh) || self.is_edge_deleted(self.edge_handle(heh)) {
            return false;
        }

        let v0 = self.from_vertex_handle(heh); // Vertex to be removed
        let v1 = self.to_vertex_handle(heh); // Remaining vertex
        if !v0.is_valid()
            || !v1.is_valid()
            || self.is_vertex_deleted(v0)
            || self.is_vertex_deleted(v1)
        {
            return false;
        }

        // Both vertices must have valid halfedges
        if self.halfedge_handle(v0).is_none() || self.halfedge_handle(v1).is_none() {
            return false;
        }

        let heh_opp = self.opposite_halfedge_handle(heh);

        // Get adjacent faces
        let fh_left = self.face_handle(heh);
        let fh_right = self.face_handle(heh_opp);

        // At least one face must exist
        if fh_left.is_none() && fh_right.is_none() {
            return false;
        }

        // Match OpenMesh TriConnectivity: collapsing through a boundary wedge
        // would create invalid local boundary topology.
        if fh_left.is_some() {
            let h1 = self.next_halfedge_handle(heh);
            let h2 = self.next_halfedge_handle(h1);
            if self.is_boundary(self.opposite_halfedge_handle(h1))
                && self.is_boundary(self.opposite_halfedge_handle(h2))
            {
                return false;
            }
        }

        if fh_right.is_some() {
            let h1 = self.next_halfedge_handle(heh_opp);
            let h2 = self.next_halfedge_handle(h1);
            if self.is_boundary(self.opposite_halfedge_handle(h1))
                && self.is_boundary(self.opposite_halfedge_handle(h2))
            {
                return false;
            }
        }

        // Collect neighbors of v0 (excluding v1)
        let neighbors_v0: Vec<VertexHandle> = self
            .collect_vertex_neighbors(v0)
            .into_iter()
            .filter(|&v| v != v1)
            .collect();

        // Collect neighbors of v1 (excluding v0)
        let neighbors_v1: Vec<VertexHandle> = self
            .collect_vertex_neighbors(v1)
            .into_iter()
            .filter(|&v| v != v0)
            .collect();

        // Check for duplicate edges: after collapse, v0's neighbors become v1's neighbors.
        // If any neighbor of v0 is already a neighbor of v1 (and not in an adjacent face),
        // collapsing would create a duplicate edge.
        //
        // Vertices that are allowed to be shared are those in the faces adjacent to the edge.
        let mut allowed_shared: Vec<VertexHandle> = Vec::new();
        if let Some(fh) = fh_left {
            for vh in self.face_vertices_vec(fh) {
                if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                    allowed_shared.push(vh);
                }
            }
        }
        if let Some(fh) = fh_right {
            for vh in self.face_vertices_vec(fh) {
                if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                    allowed_shared.push(vh);
                }
            }
        }

        // Any shared neighbor NOT in the adjacent faces would create a duplicate edge
        for &nv in &neighbors_v0 {
            if neighbors_v1.contains(&nv) && !allowed_shared.contains(&nv) {
                return false;
            }
        }

        // For triangle faces: check that the opposite vertices are distinct
        if fh_left.is_some() && fh_right.is_some() {
            let vl = self.to_vertex_handle(self.next_halfedge_handle(heh));
            let vr = self.to_vertex_handle(self.next_halfedge_handle(heh_opp));
            if vl == vr {
                return false;
            }
        }

        // Boundary check: both boundary vertices can only collapse along a boundary edge
        let v0_boundary = self.is_boundary_vertex(v0);
        let v1_boundary = self.is_boundary_vertex(v1);
        let edge_boundary = fh_left.is_none() || fh_right.is_none();

        if v0_boundary && v1_boundary && !edge_boundary {
            return false;
        }

        // Match OpenMesh PolyConnectivity: prevent collapses that would
        // degenerate a backside polygon sharing both adjacent opposite edges.
        if let Some(fh) = fh_left {
            if self.face_vertices_vec(fh).len() == 3 {
                let one = self.opposite_halfedge_handle(self.next_halfedge_handle(heh));
                let two = self.opposite_halfedge_handle(
                    self.next_halfedge_handle(self.next_halfedge_handle(heh)),
                );
                if let (Some(face_one), Some(face_two)) =
                    (self.face_handle(one), self.face_handle(two))
                {
                    if face_one == face_two && self.face_vertices_vec(face_one).len() != 3 {
                        return false;
                    }
                }
            }
        }

        if let Some(fh) = fh_right {
            if self.face_vertices_vec(fh).len() == 3 {
                let one = self.opposite_halfedge_handle(self.next_halfedge_handle(heh_opp));
                let two = self.opposite_halfedge_handle(
                    self.next_halfedge_handle(self.next_halfedge_handle(heh_opp)),
                );
                if let (Some(face_one), Some(face_two)) =
                    (self.face_handle(one), self.face_handle(two))
                {
                    if face_one == face_two && self.face_vertices_vec(face_one).len() != 3 {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a vertex is on the boundary (has at least one boundary halfedge)
    pub fn is_boundary_vertex(&self, vh: VertexHandle) -> bool {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return false;
        }

        // A vertex is on boundary if ANY incident halfedge has no face.
        // Keep the local one-ring walk here because this path is hot in decimation.
        if let Some(heh) = self.halfedge_handle(vh) {
            let mut current = heh;
            let max_iterations = 100; // Valence is typically small
            for _ in 0..max_iterations {
                if self.is_boundary(current) {
                    return true;
                }
                let opp = self.opposite_halfedge_handle(current);
                if self.is_boundary(opp) {
                    return true;
                }
                current = self.next_outgoing_halfedge(current);
                if current == heh || !current.is_valid() {
                    break;
                }
            }
        }
        false
    }

    fn collect_vertex_neighbors(&self, vh: VertexHandle) -> Vec<VertexHandle> {
        let mut neighbors = Vec::new();

        // Use circulator for O(valence) instead of O(n_faces)
        if let Some(heh) = self.halfedge_handle(vh) {
            let start_heh = heh;
            let mut current_heh = heh;
            let max_iterations = 100; // Valence is typically small

            for _ in 0..max_iterations {
                // Get the neighbor vertex from current outgoing halfedge
                let neighbor = self.to_vertex_handle(current_heh);
                if !neighbors.contains(&neighbor) {
                    neighbors.push(neighbor);
                }

                let next_heh = self.next_outgoing_halfedge(current_heh);

                // Check if we've completed the cycle or hit invalid halfedge
                if !next_heh.is_valid() || next_heh == start_heh {
                    break;
                }

                // Critical: verify the next halfedge is still from our vertex
                if self.from_vertex_handle(next_heh) != vh {
                    break; // Prevent jumping to wrong vertex's halfedge at boundary
                }

                current_heh = next_heh;
            }
        }

        neighbors
    }

    /// Collapse a halfedge: move v0 to v1 and remove v0 and adjacent faces.
    ///
    /// Requested normal arrays remain allocated but are not recomputed
    /// automatically. Callers must refresh normals explicitly after collapse if
    /// they need up-to-date values.
    pub fn collapse(&mut self, heh: HalfedgeHandle) -> Result<(), &'static str> {
        if !self.is_collapse_ok(heh) {
            return Err("Collapse not legal");
        }

        let h0 = heh;
        let h1 = self.next_halfedge_handle(h0);
        let o0 = self.opposite_halfedge_handle(h0);
        let o1 = self.next_halfedge_handle(o0);
        let removed_vh = self.from_vertex_handle(h0);
        let kept_vh = self.to_vertex_handle(h0);

        self.kernel.merge_vertex_props(removed_vh, kept_vh);

        self.collapse_edge_local(h0);

        if !self.is_halfedge_deleted(h1)
            && !self.is_edge_deleted(self.edge_handle(h1))
            && self.next_halfedge_handle(self.next_halfedge_handle(h1)) == h1
        {
            self.collapse_loop_local(self.next_halfedge_handle(h1));
        }
        if !self.is_halfedge_deleted(o1)
            && !self.is_edge_deleted(self.edge_handle(o1))
            && self.next_halfedge_handle(self.next_halfedge_handle(o1)) == o1
        {
            self.collapse_loop_local(o1);
        }

        Ok(())
    }

    fn collapse_edge_local(&mut self, heh: HalfedgeHandle) {
        let h = heh;
        let hn = self.next_halfedge_handle(h);
        let hp = self.prev_halfedge_handle(h);

        let o = self.opposite_halfedge_handle(h);
        let on = self.next_halfedge_handle(o);
        let op = self.prev_halfedge_handle(o);

        let fh = self.face_handle(h);
        let fo = self.face_handle(o);

        let vh = self.to_vertex_handle(h);
        let vo = self.to_vertex_handle(o);

        for heh_idx in 0..self.n_halfedges() {
            let incoming = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(incoming)
                || self.is_edge_deleted(self.edge_handle(incoming))
            {
                continue;
            }
            if self.to_vertex_handle(incoming) == vo {
                self.kernel.set_halfedge_to_vertex(incoming, vh);
            }
        }

        self.kernel.set_next_halfedge_handle(hp, hn);
        self.kernel.set_next_halfedge_handle(op, on);

        if let Some(fh) = fh {
            self.set_face_halfedge_handle(fh, Some(hn));
        }
        if let Some(fo) = fo {
            self.set_face_halfedge_handle(fo, Some(on));
        }

        if self.halfedge_handle(vh) == Some(o) {
            self.kernel.set_halfedge_handle(vh, hn);
        }
        self.adjust_outgoing_halfedge(vh);
        self.clear_halfedge_handle(vo);

        self.kernel.mark_edge_deleted(self.edge_handle(h), true);
        self.kernel.mark_vertex_deleted(vo, true);
        self.kernel.mark_halfedge_deleted(h, true);
        self.kernel.mark_halfedge_deleted(o, true);
    }

    fn collapse_loop_local(&mut self, heh: HalfedgeHandle) {
        if !heh.is_valid()
            || self.is_halfedge_deleted(heh)
            || self.is_edge_deleted(self.edge_handle(heh))
        {
            return;
        }

        let h0 = heh;
        let h1 = self.next_halfedge_handle(h0);

        let o0 = self.opposite_halfedge_handle(h0);
        let o1 = self.opposite_halfedge_handle(h1);

        let v0 = self.to_vertex_handle(h0);
        let v1 = self.to_vertex_handle(h1);

        let fh = self.face_handle(h0);
        let fo = self.face_handle(o0);

        if self.next_halfedge_handle(h1) != h0 || h1 == o0 {
            return;
        }

        self.kernel
            .set_next_halfedge_handle(h1, self.next_halfedge_handle(o0));
        self.kernel
            .set_next_halfedge_handle(self.prev_halfedge_handle(o0), h1);

        match fo {
            Some(fo) => self.kernel.set_face_handle(h1, fo),
            None => self.clear_face_handle(h1),
        }

        self.kernel.set_halfedge_handle(v0, h1);
        self.adjust_outgoing_halfedge(v0);
        self.kernel.set_halfedge_handle(v1, o1);
        self.adjust_outgoing_halfedge(v1);

        if let Some(fo) = fo {
            if self.face_halfedge_handle(fo) == Some(o0) {
                self.set_face_halfedge_handle(fo, Some(h1));
            }
        }

        if let Some(fh) = fh {
            self.set_face_halfedge_handle(fh, None);
            self.kernel.mark_face_deleted(fh, true);
        }
        self.kernel.mark_edge_deleted(self.edge_handle(h0), true);
        self.kernel.mark_halfedge_deleted(h0, true);
        self.kernel.mark_halfedge_deleted(o0, true);
    }

    fn adjust_outgoing_halfedge(&mut self, vh: VertexHandle) {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            self.clear_halfedge_handle(vh);
            return;
        }

        let current = self.halfedge_handle(vh).filter(|&heh| {
            heh.is_valid()
                && !self.is_halfedge_deleted(heh)
                && !self.is_edge_deleted(self.edge_handle(heh))
                && self.from_vertex_handle(heh) == vh
        });

        if let Some(start) = current {
            let mut heh = start;
            let max_iterations = self.n_halfedges().max(1000);
            for _ in 0..max_iterations {
                if self.is_boundary(heh) {
                    self.kernel.set_halfedge_handle(vh, heh);
                    return;
                }

                let next = self.next_outgoing_halfedge(heh);
                if !next.is_valid() || next == start || self.from_vertex_handle(next) != vh {
                    break;
                }
                heh = next;
            }

            // OpenMesh keeps the current outgoing halfedge when no boundary
            // halfedge is found around the one-ring.
            self.kernel.set_halfedge_handle(vh, start);
            return;
        }

        for heh_idx in 0..self.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(heh) || self.is_edge_deleted(self.edge_handle(heh)) {
                continue;
            }
            if self.from_vertex_handle(heh) == vh {
                self.kernel.set_halfedge_handle(vh, heh);
                return;
            }
        }

        self.clear_halfedge_handle(vh);
    }

    pub fn normalize_boundary_halfedge_handles(&mut self) {
        let vertices: Vec<_> = self.vertices().collect();
        for vh in vertices {
            self.adjust_outgoing_halfedge(vh);
        }
    }

    pub fn rebuild_boundary_halfedge_links(&mut self) {
        let n_vertices = self.n_vertices();
        let mut incoming = vec![None; n_vertices];
        let mut outgoing = vec![None; n_vertices];
        let mut incoming_ambiguous = vec![false; n_vertices];
        let mut outgoing_ambiguous = vec![false; n_vertices];

        for heh_idx in 0..self.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(heh)
                || self.is_edge_deleted(self.edge_handle(heh))
                || !self.is_boundary(heh)
            {
                continue;
            }

            let from = self.from_vertex_handle(heh);
            if from.is_valid() {
                let idx = from.idx_usize();
                if outgoing[idx].is_some() && outgoing[idx] != Some(heh) {
                    outgoing_ambiguous[idx] = true;
                } else {
                    outgoing[idx] = Some(heh);
                }
            }

            let to = self.to_vertex_handle(heh);
            if to.is_valid() {
                let idx = to.idx_usize();
                if incoming[idx].is_some() && incoming[idx] != Some(heh) {
                    incoming_ambiguous[idx] = true;
                } else {
                    incoming[idx] = Some(heh);
                }
            }
        }

        for idx in 0..n_vertices {
            if incoming_ambiguous[idx] || outgoing_ambiguous[idx] {
                continue;
            }

            if let (Some(incoming_heh), Some(outgoing_heh)) = (incoming[idx], outgoing[idx]) {
                self.kernel
                    .set_next_halfedge_handle(incoming_heh, outgoing_heh);
            }
        }
    }

    /// Flip an edge in a triangular mesh.
    ///
    /// This operation rotates an edge by flipping it to connect the opposite
    /// vertices of the two adjacent triangles. The edge must have exactly two
    /// adjacent triangular faces.
    ///
    /// Normal arrays are not recomputed automatically. Call `update_normals()`
    /// or `update_vertex_normals_with_mode()` explicitly after topology edits.
    ///
    /// Before flip:
    /// ```
    ///     v2                v2
    ///     /|\              / | \
    ///    / | \            /  |  \
    ///   /  |  \          /   |   \
    ///  v0--e--v1   =>   v0---+---v1
    ///   \  |  /          \   |   /
    ///    \ | /            \  |  /
    ///     \|/              \ | /
    ///     v3                v3
    /// ```
    ///
    /// After flip:
    /// ```
    ///     v2                v2
    ///     / \              /|\
    ///    /   \            / | \
    ///   /     \          /  |  \
    ///  v0--e---v1   =>   v0--+--v1
    ///   \     /          \  |  /
    ///    \   /            \ | /
    ///     \ /              \|/
    ///     v3                v3
    /// ```
    ///
    /// The edge `e` now connects v2 and v3 instead of v0 and v1.
    ///
    /// # Arguments
    /// * `eh` - The edge to flip
    ///
    /// # Returns
    /// * `Ok(())` if the flip was successful
    /// * `Err(&'static str)` if the flip is not possible
    ///
    /// # Conditions for flipping
    /// 1. The edge must have exactly two adjacent faces (not a boundary edge)
    /// 2. Both adjacent faces must be triangles
    /// 3. The resulting edge must not already exist (would create a non-manifold)
    pub fn flip_edge(&mut self, eh: EdgeHandle) -> Result<(), &'static str> {
        // Get the two halfedges of the edge
        let h0 = self.edge_halfedge_handle(eh, 0);
        let h1 = self.edge_halfedge_handle(eh, 1);

        // Get the two faces adjacent to this edge
        let fh0 = self.face_handle(h0);
        let fh1 = self.face_handle(h1);

        // Both faces must exist (no boundary edge)
        let fh0 = fh0.ok_or("Cannot flip boundary edge")?;
        let fh1 = fh1.ok_or("Cannot flip boundary edge")?;

        // Check that both faces are triangles
        let verts0 = self.face_vertices_vec(fh0);
        let verts1 = self.face_vertices_vec(fh1);

        if verts0.len() != 3 || verts1.len() != 3 {
            return Err("Can only flip edge between two triangles");
        }

        // Get the vertices
        // h0: v0 -> v1 (edge from v0 to v1)
        // h1: v1 -> v0 (opposite direction)
        let v0 = self.from_vertex_handle(h0);
        let v1 = self.to_vertex_handle(h0);

        // Get the opposite vertices in the two triangles
        // For triangle (v0, v1, v2), v2 is opposite to edge h0
        // For triangle (v1, v0, v3), v3 is opposite to edge h1
        let h0_next = self.next_halfedge_handle(h0);
        let h1_next = self.next_halfedge_handle(h1);

        let v2 = self.to_vertex_handle(h0_next);
        let v3 = self.to_vertex_handle(h1_next);

        // Check that the new edge (v2, v3) doesn't already exist
        // This would create a non-manifold configuration
        if self.vertices_connected(v2, v3) {
            return Err("Cannot flip: resulting edge already exists");
        }

        let h0_prev = self.prev_halfedge_handle(h0);
        let h1_prev = self.prev_halfedge_handle(h1);

        // Now perform the flip:
        // h0/h1 become the new diagonal, and the adjacent halfedge cycles are:
        // fh0: h0 (v2->v3) -> h1_prev (v3->v1) -> h0_next (v1->v2)
        // fh1: h1 (v3->v2) -> h0_prev (v2->v0) -> h1_next (v0->v3)

        // Set the new vertex endpoints
        self.kernel.set_halfedge_to_vertex(h0, v3);
        self.kernel.set_halfedge_to_vertex(h1, v2);

        // Rewire the next pointers for both triangle loops.
        self.kernel.set_next_halfedge_handle(h0, h1_prev);
        self.kernel.set_next_halfedge_handle(h1_prev, h0_next);
        self.kernel.set_next_halfedge_handle(h0_next, h0);

        self.kernel.set_next_halfedge_handle(h1, h0_prev);
        self.kernel.set_next_halfedge_handle(h0_prev, h1_next);
        self.kernel.set_next_halfedge_handle(h1_next, h1);

        // Keep prev pointers consistent with the new face cycles.
        self.kernel.set_prev_halfedge_handle(h1_prev, h0);
        self.kernel.set_prev_halfedge_handle(h0_next, h1_prev);
        self.kernel.set_prev_halfedge_handle(h0, h0_next);

        self.kernel.set_prev_halfedge_handle(h0_prev, h1);
        self.kernel.set_prev_halfedge_handle(h1_next, h0_prev);
        self.kernel.set_prev_halfedge_handle(h1, h1_next);

        // Update face handles
        self.kernel.set_face_handle(h0, fh0);
        self.kernel.set_face_handle(h1_prev, fh0);
        self.kernel.set_face_handle(h0_next, fh0);

        self.kernel.set_face_handle(h1, fh1);
        self.kernel.set_face_handle(h0_prev, fh1);
        self.kernel.set_face_handle(h1_next, fh1);

        // Update face halfedge handles
        self.kernel.set_face_halfedge_handle(fh0, Some(h0));
        self.kernel.set_face_halfedge_handle(fh1, Some(h1));

        // Recompute outgoing halfedge anchors for the affected one-ring.
        self.adjust_outgoing_halfedge(v0);
        self.adjust_outgoing_halfedge(v1);
        self.adjust_outgoing_halfedge(v2);
        self.adjust_outgoing_halfedge(v3);

        Ok(())
    }

    /// Check if two vertices are connected by an edge
    fn vertices_connected(&self, v0: VertexHandle, v1: VertexHandle) -> bool {
        if !v0.is_valid() || !v1.is_valid() || v0 == v1 {
            return false;
        }

        for eh_idx in 0..self.n_edges() {
            let eh = EdgeHandle::new(eh_idx as u32);
            if self.is_edge_deleted(eh) {
                continue;
            }

            let h0 = self.edge_halfedge_handle(eh, 0);
            let e0 = self.from_vertex_handle(h0);
            let e1 = self.to_vertex_handle(h0);

            if !e0.is_valid()
                || !e1.is_valid()
                || self.is_vertex_deleted(e0)
                || self.is_vertex_deleted(e1)
            {
                continue;
            }

            if (e0 == v0 && e1 == v1) || (e0 == v1 && e1 == v0) {
                return true;
            }
        }
        false
    }

    /// Redirect all halfedges that reference from_vertex to reference to_vertex
    fn redirect_halfedges(
        &mut self,
        from_vertex: VertexHandle,
        to_vertex: VertexHandle,
    ) -> Result<(), &'static str> {
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
        // Delete the face in the kernel
        // This clears the face_handle references in all halfedges of this face
        // but preserves the halfedge connectivity (next/prev pointers)
        // so that vertex circulators can still traverse around vertices
        self.kernel.delete_face(fh);
    }

    /// Get all halfedges of a face
    fn get_face_halfedges(&self, fh: FaceHandle) -> Vec<HalfedgeHandle> {
        let mut result = Vec::new();
        if let Some(start_heh) = self.kernel.face(fh).and_then(|f| f.halfedge_handle) {
            let mut current = start_heh;
            loop {
                result.push(current);
                let next = self.next_halfedge_handle(current);
                if !next.is_valid() || next == start_heh {
                    break;
                }
                current = next;
            }
        }
        result
    }

    /// Delete a vertex from the mesh
    pub fn delete_vertex(&mut self, vh: VertexHandle) {
        self.kernel.delete_vertex(vh);
    }

    /// Delete an edge from the mesh
    pub fn delete_edge(&mut self, eh: EdgeHandle) {
        self.kernel.delete_edge(eh);
    }

    fn split_edge_side_after_old(
        &mut self,
        old_heh: HalfedgeHandle,
        inserted_heh: HalfedgeHandle,
        fh: FaceHandle,
        next_heh: HalfedgeHandle,
        prev_heh: HalfedgeHandle,
        new_vh: VertexHandle,
        affected_vertices: &mut Vec<VertexHandle>,
    ) -> Result<(), &'static str> {
        let face_degree = self.get_face_halfedges(fh).len();
        if face_degree < 3 {
            return Err("Face must have at least three edges");
        }

        if face_degree == 3 {
            let opposite_vh = self.to_vertex_handle(next_heh);
            if !opposite_vh.is_valid() || self.is_vertex_deleted(opposite_vh) {
                return Err("Invalid triangle topology");
            }

            let diagonal = self.add_edge(opposite_vh, new_vh);
            let diagonal_opp = self.opposite_halfedge_handle(diagonal);
            let new_fh = self.kernel.add_face(Some(inserted_heh));
            self.kernel.copy_face_props(fh, new_fh);

            self.kernel.set_next_halfedge_handle(old_heh, diagonal_opp);
            self.kernel.set_next_halfedge_handle(diagonal_opp, prev_heh);
            self.kernel.set_next_halfedge_handle(prev_heh, old_heh);

            self.kernel.set_face_handle(old_heh, fh);
            self.kernel.set_face_handle(diagonal_opp, fh);
            self.kernel.set_face_handle(prev_heh, fh);
            self.set_face_halfedge_handle(fh, Some(old_heh));

            self.kernel.set_next_halfedge_handle(inserted_heh, next_heh);
            self.kernel.set_next_halfedge_handle(next_heh, diagonal);
            self.kernel.set_next_halfedge_handle(diagonal, inserted_heh);

            self.kernel.set_face_handle(inserted_heh, new_fh);
            self.kernel.set_face_handle(next_heh, new_fh);
            self.kernel.set_face_handle(diagonal, new_fh);
            self.set_face_halfedge_handle(new_fh, Some(inserted_heh));

            affected_vertices.push(opposite_vh);
        } else {
            self.kernel.set_next_halfedge_handle(old_heh, inserted_heh);
            self.kernel.set_next_halfedge_handle(inserted_heh, next_heh);
            self.kernel.set_face_handle(old_heh, fh);
            self.kernel.set_face_handle(inserted_heh, fh);
            self.set_face_halfedge_handle(fh, Some(old_heh));
        }

        Ok(())
    }

    fn split_edge_side_before_old(
        &mut self,
        old_heh: HalfedgeHandle,
        inserted_heh: HalfedgeHandle,
        fh: FaceHandle,
        next_heh: HalfedgeHandle,
        prev_heh: HalfedgeHandle,
        new_vh: VertexHandle,
        affected_vertices: &mut Vec<VertexHandle>,
    ) -> Result<(), &'static str> {
        let face_degree = self.get_face_halfedges(fh).len();
        if face_degree < 3 {
            return Err("Face must have at least three edges");
        }

        if face_degree == 3 {
            let opposite_vh = self.to_vertex_handle(next_heh);
            if !opposite_vh.is_valid() || self.is_vertex_deleted(opposite_vh) {
                return Err("Invalid triangle topology");
            }

            let diagonal = self.add_edge(opposite_vh, new_vh);
            let diagonal_opp = self.opposite_halfedge_handle(diagonal);
            let new_fh = self.kernel.add_face(Some(old_heh));
            self.kernel.copy_face_props(fh, new_fh);

            self.kernel
                .set_next_halfedge_handle(inserted_heh, diagonal_opp);
            self.kernel.set_next_halfedge_handle(diagonal_opp, prev_heh);
            self.kernel.set_next_halfedge_handle(prev_heh, inserted_heh);

            self.kernel.set_face_handle(inserted_heh, fh);
            self.kernel.set_face_handle(diagonal_opp, fh);
            self.kernel.set_face_handle(prev_heh, fh);
            self.set_face_halfedge_handle(fh, Some(inserted_heh));

            self.kernel.set_next_halfedge_handle(old_heh, next_heh);
            self.kernel.set_next_halfedge_handle(next_heh, diagonal);
            self.kernel.set_next_halfedge_handle(diagonal, old_heh);

            self.kernel.set_face_handle(old_heh, new_fh);
            self.kernel.set_face_handle(next_heh, new_fh);
            self.kernel.set_face_handle(diagonal, new_fh);
            self.set_face_halfedge_handle(new_fh, Some(old_heh));

            affected_vertices.push(opposite_vh);
        } else {
            self.kernel.set_next_halfedge_handle(prev_heh, inserted_heh);
            self.kernel.set_next_halfedge_handle(inserted_heh, old_heh);
            self.kernel.set_next_halfedge_handle(old_heh, next_heh);
            self.kernel.set_face_handle(inserted_heh, fh);
            self.kernel.set_face_handle(old_heh, fh);
            self.set_face_halfedge_handle(fh, Some(inserted_heh));
        }

        Ok(())
    }

    fn split_face_triangle_local(
        &mut self,
        fh: FaceHandle,
        point: glam::Vec3,
    ) -> Result<VertexHandle, &'static str> {
        let face_halfedges = self.get_face_halfedges(fh);
        if face_halfedges.len() != 3 {
            return Err("Local split_face() currently supports triangles only");
        }

        let h0 = face_halfedges[0];
        let h1 = face_halfedges[1];
        let h2 = face_halfedges[2];

        if self.next_halfedge_handle(h0) != h1
            || self.next_halfedge_handle(h1) != h2
            || self.next_halfedge_handle(h2) != h0
        {
            return Err("Invalid triangle topology");
        }

        let v0 = self.from_vertex_handle(h0);
        let v1 = self.to_vertex_handle(h0);
        let v2 = self.to_vertex_handle(h1);
        if !v0.is_valid()
            || !v1.is_valid()
            || !v2.is_valid()
            || v0 == v1
            || v1 == v2
            || v2 == v0
            || self.is_vertex_deleted(v0)
            || self.is_vertex_deleted(v1)
            || self.is_vertex_deleted(v2)
        {
            return Err("Invalid triangle topology");
        }

        let new_vh = self.add_vertex(point);
        self.kernel
            .interpolate_vertex_props_triangle(v0, v1, v2, new_vh);

        let v0_to_new = self.add_edge(v0, new_vh);
        let new_to_v0 = self.opposite_halfedge_handle(v0_to_new);
        let v1_to_new = self.add_edge(v1, new_vh);
        let new_to_v1 = self.opposite_halfedge_handle(v1_to_new);
        let v2_to_new = self.add_edge(v2, new_vh);
        let new_to_v2 = self.opposite_halfedge_handle(v2_to_new);

        let fh1 = self.kernel.add_face(Some(h1));
        let fh2 = self.kernel.add_face(Some(h2));
        self.kernel.copy_face_props(fh, fh1);
        self.kernel.copy_face_props(fh, fh2);

        self.kernel.set_next_halfedge_handle(h0, v1_to_new);
        self.kernel.set_next_halfedge_handle(v1_to_new, new_to_v0);
        self.kernel.set_next_halfedge_handle(new_to_v0, h0);

        self.kernel.set_face_handle(h0, fh);
        self.kernel.set_face_handle(v1_to_new, fh);
        self.kernel.set_face_handle(new_to_v0, fh);
        self.set_face_halfedge_handle(fh, Some(h0));

        self.kernel.set_next_halfedge_handle(h1, v2_to_new);
        self.kernel.set_next_halfedge_handle(v2_to_new, new_to_v1);
        self.kernel.set_next_halfedge_handle(new_to_v1, h1);

        self.kernel.set_face_handle(h1, fh1);
        self.kernel.set_face_handle(v2_to_new, fh1);
        self.kernel.set_face_handle(new_to_v1, fh1);
        self.set_face_halfedge_handle(fh1, Some(h1));

        self.kernel.set_next_halfedge_handle(h2, v0_to_new);
        self.kernel.set_next_halfedge_handle(v0_to_new, new_to_v2);
        self.kernel.set_next_halfedge_handle(new_to_v2, h2);

        self.kernel.set_face_handle(h2, fh2);
        self.kernel.set_face_handle(v0_to_new, fh2);
        self.kernel.set_face_handle(new_to_v2, fh2);
        self.set_face_halfedge_handle(fh2, Some(h2));

        for vh in [v0, v1, v2, new_vh] {
            self.adjust_outgoing_halfedge(vh);
        }

        Ok(new_vh)
    }

    /// Split an edge by inserting a new vertex at the provided point.
    ///
    /// Adjacent triangle faces are split into two triangles each. For non-triangle
    /// faces, the new vertex is inserted into the face loop between the edge endpoints.
    /// Requested normal arrays are resized as needed, but no automatic normal
    /// refresh is performed.
    pub fn split_edge(
        &mut self,
        eh: EdgeHandle,
        point: glam::Vec3,
    ) -> Result<VertexHandle, &'static str> {
        if !eh.is_valid() || self.is_edge_deleted(eh) {
            return Err("Invalid edge");
        }

        let he0 = self.edge_halfedge_handle(eh, 0);
        if !he0.is_valid() || self.is_halfedge_deleted(he0) {
            return Err("Invalid halfedge");
        }

        let v0 = self.from_vertex_handle(he0);
        let v1 = self.to_vertex_handle(he0);
        if !v0.is_valid()
            || !v1.is_valid()
            || v0 == v1
            || self.is_vertex_deleted(v0)
            || self.is_vertex_deleted(v1)
        {
            return Err("Invalid edge endpoints");
        }

        let he1 = self.edge_halfedge_handle(eh, 1);
        let fh0 = self.face_handle(he0);
        let fh1 = self.face_handle(he1);
        if fh0.is_none() && fh1.is_none() {
            return Err("Edge has no incident faces");
        }

        let he0_next = fh0.map(|_| self.next_halfedge_handle(he0));
        let he0_prev = fh0.map(|_| self.prev_halfedge_handle(he0));
        let he1_next = fh1.map(|_| self.next_halfedge_handle(he1));
        let he1_prev = fh1.map(|_| self.prev_halfedge_handle(he1));

        let new_vh = self.add_vertex(point);
        self.kernel.interpolate_vertex_props_pair(v0, v1, new_vh);
        let split_heh = self.add_edge(new_vh, v1);
        let split_heh_opp = self.opposite_halfedge_handle(split_heh);
        self.kernel.copy_edge_props(eh, self.edge_handle(split_heh));

        self.kernel.remap_edge_lookup(v0, v1, v0, new_vh, he0);
        self.kernel.set_halfedge_to_vertex(he0, new_vh);

        let mut affected_vertices = vec![v0, v1, new_vh];

        if let Some(fh) = fh0 {
            self.split_edge_side_after_old(
                he0,
                split_heh,
                fh,
                he0_next.unwrap(),
                he0_prev.unwrap(),
                new_vh,
                &mut affected_vertices,
            )?;
        }

        if let Some(fh) = fh1 {
            self.split_edge_side_before_old(
                he1,
                split_heh_opp,
                fh,
                he1_next.unwrap(),
                he1_prev.unwrap(),
                new_vh,
                &mut affected_vertices,
            )?;
        }

        let boundary_changed = fh0.is_none() || fh1.is_none();
        if boundary_changed {
            self.rebuild_boundary_halfedge_links();
            self.normalize_boundary_halfedge_handles();
        } else {
            affected_vertices.sort_unstable_by_key(|vh| vh.idx());
            affected_vertices.dedup();
            for vh in affected_vertices {
                self.adjust_outgoing_halfedge(vh);
            }
        }

        Ok(new_vh)
    }

    /// Split a face by inserting a new vertex at the provided point and
    /// fan-triangulating the face around it.
    ///
    /// Triangle faces use the maintained local-edit path, while n-gons still
    /// use the rebuild-backed fallback. Neither path recomputes normals
    /// automatically; the rebuild-backed path also drops face-normal storage
    /// until the caller explicitly requests or refreshes it again.
    pub fn split_face(
        &mut self,
        fh: FaceHandle,
        point: glam::Vec3,
    ) -> Result<VertexHandle, &'static str> {
        if !fh.is_valid() || self.is_face_deleted(fh) {
            return Err("Invalid face");
        }

        let face_degree = self.get_face_halfedges(fh).len();
        if face_degree < 3 {
            return Err("Face must have at least three vertices");
        }
        if face_degree == 3 {
            return self.split_face_triangle_local(fh, point);
        }

        let face = self.sanitized_face_vertices(fh);
        let new_vh = VertexHandle::from_usize(self.n_vertices());
        let mut replacements: HashMap<usize, Vec<Vec<VertexHandle>>> = HashMap::new();
        replacements.insert(
            fh.idx_usize(),
            triangulate_face_loop_with_vertex(&face, new_vh),
        );

        let inserted = self.add_vertex(point);
        debug_assert_eq!(inserted, new_vh);

        let rebuilt_faces = self.rebuild_faces_with_replacements(&replacements);
        self.rebuild_preserving_vertex_indices(&rebuilt_faces);

        Ok(new_vh)
    }

    /// Triangulate a face using a fan rooted at the face's first vertex.
    ///
    /// This is currently implemented via the same rebuild-backed baseline used by
    /// the public split primitives, so it provides the API and semantics first
    /// while local half-edge surgery remains future work.
    /// Requested vertex normals are preserved across the rebuild, but face
    /// normals are not preserved and must be recomputed explicitly.
    pub fn triangulate_face(&mut self, fh: FaceHandle) -> Result<(), &'static str> {
        if !fh.is_valid() || self.is_face_deleted(fh) {
            return Err("Invalid face");
        }

        let face = self.sanitized_face_vertices(fh);
        if face.len() < 3 {
            return Err("Face must have at least three vertices");
        }
        if face.len() == 3 {
            return Ok(());
        }

        let mut replacements: HashMap<usize, Vec<Vec<VertexHandle>>> = HashMap::new();
        replacements.insert(fh.idx_usize(), triangulate_face_loop(&face));

        let rebuilt_faces = self.rebuild_faces_with_replacements(&replacements);
        self.rebuild_preserving_vertex_indices(&rebuilt_faces);

        Ok(())
    }

    /// Rebuild the mesh from currently active faces and vertices.
    ///
    /// This mirrors OpenMesh's `garbage_collection()` at a higher level:
    /// deleted elements are discarded and a clean connectivity structure is rebuilt.
    pub fn garbage_collection(&mut self) {
        let old_vertex_count = self.n_vertices();
        if old_vertex_count == 0 {
            return;
        }

        let preserve_normals = self.has_vertex_normals();
        let preserve_colors = self.has_vertex_colors();
        let preserve_texcoords = self.has_vertex_texcoords();

        let mut used_vertices = vec![false; old_vertex_count];
        let mut face_loops: Vec<Vec<VertexHandle>> = Vec::new();

        for fh in self.faces() {
            let vertices = self.sanitized_face_vertices(fh);
            if vertices.len() < 3 {
                continue;
            }

            for vh in &vertices {
                let idx = vh.idx_usize();
                if idx < used_vertices.len() {
                    used_vertices[idx] = true;
                }
            }
            face_loops.push(vertices);
        }

        if face_loops.is_empty() {
            self.clear();
            return;
        }

        let mut remap: Vec<Option<usize>> = vec![None; old_vertex_count];
        let mut positions: Vec<glam::Vec3> = Vec::new();
        let mut normals: Vec<glam::Vec3> = Vec::new();
        let mut colors: Vec<glam::Vec4> = Vec::new();
        let mut texcoords: Vec<glam::Vec2> = Vec::new();

        for (idx, &used) in used_vertices.iter().enumerate() {
            if !used {
                continue;
            }

            let Some(point) = self.point_by_index(idx) else {
                continue;
            };

            remap[idx] = Some(positions.len());
            positions.push(point);

            if preserve_normals {
                normals.push(self.vertex_normal_by_index(idx).unwrap_or(glam::Vec3::ZERO));
            }
            if preserve_colors {
                colors.push(
                    self.vertex_color_by_index(idx)
                        .unwrap_or(glam::Vec4::new(1.0, 1.0, 1.0, 1.0)),
                );
            }
            if preserve_texcoords {
                texcoords.push(
                    self.vertex_texcoord_by_index(idx)
                        .unwrap_or(glam::Vec2::ZERO),
                );
            }
        }

        let mut rebuilt = RustMesh::new();
        for point in &positions {
            rebuilt.add_vertex(*point);
        }

        if preserve_normals {
            rebuilt.request_vertex_normals();
            for (idx, normal) in normals.iter().enumerate() {
                rebuilt.set_vertex_normal_by_index(idx, *normal);
            }
        }
        if preserve_colors {
            rebuilt.request_vertex_colors();
            for (idx, color) in colors.iter().enumerate() {
                rebuilt.set_vertex_color_by_index(idx, *color);
            }
        }
        if preserve_texcoords {
            rebuilt.request_vertex_texcoords();
            for (idx, texcoord) in texcoords.iter().enumerate() {
                rebuilt.set_vertex_texcoord_by_index(idx, *texcoord);
            }
        }

        let remapped_faces: Vec<Vec<VertexHandle>> = face_loops
            .into_iter()
            .map(|face| {
                dedupe_face_vertices(
                    face.into_iter()
                        .filter_map(|vh| remap.get(vh.idx_usize()).and_then(|idx| *idx))
                        .map(VertexHandle::from_usize)
                        .collect(),
                )
            })
            .filter(|face: &Vec<VertexHandle>| face.len() >= 3)
            .collect();

        if remapped_faces.iter().all(|face| face.len() == 3) {
            rebuild_oriented_triangles(&mut rebuilt, &remapped_faces);
        } else {
            rebuild_polygon_faces(&mut rebuilt, &remapped_faces);
        }

        *self = rebuilt;
    }

    fn rebuild_faces_with_replacements(
        &self,
        replacements: &HashMap<usize, Vec<Vec<VertexHandle>>>,
    ) -> Vec<Vec<VertexHandle>> {
        let mut faces = Vec::new();

        for fh in self.faces() {
            if let Some(replacement_faces) = replacements.get(&fh.idx_usize()) {
                for replacement in replacement_faces {
                    let sanitized = dedupe_face_vertices(replacement.clone());
                    if sanitized.len() >= 3 {
                        faces.push(sanitized);
                    }
                }
                continue;
            }

            let vertices = self.sanitized_face_vertices(fh);
            if vertices.len() >= 3 {
                faces.push(vertices);
            }
        }

        faces
    }

    fn rebuild_preserving_vertex_indices(&mut self, faces: &[Vec<VertexHandle>]) {
        let old_vertex_count = self.n_vertices();
        let preserve_normals = self.has_vertex_normals();
        let preserve_colors = self.has_vertex_colors();
        let preserve_texcoords = self.has_vertex_texcoords();

        let positions: Vec<glam::Vec3> = (0..old_vertex_count)
            .map(|idx| self.point_by_index(idx).unwrap_or(glam::Vec3::ZERO))
            .collect();
        let normals: Vec<glam::Vec3> = if preserve_normals {
            (0..old_vertex_count)
                .map(|idx| self.vertex_normal_by_index(idx).unwrap_or(glam::Vec3::ZERO))
                .collect()
        } else {
            Vec::new()
        };
        let colors: Vec<glam::Vec4> = if preserve_colors {
            (0..old_vertex_count)
                .map(|idx| {
                    self.vertex_color_by_index(idx)
                        .unwrap_or(glam::Vec4::new(1.0, 1.0, 1.0, 1.0))
                })
                .collect()
        } else {
            Vec::new()
        };
        let texcoords: Vec<glam::Vec2> = if preserve_texcoords {
            (0..old_vertex_count)
                .map(|idx| {
                    self.vertex_texcoord_by_index(idx)
                        .unwrap_or(glam::Vec2::ZERO)
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut rebuilt = RustMesh::new();
        for point in positions {
            rebuilt.add_vertex(point);
        }

        if preserve_normals {
            rebuilt.request_vertex_normals();
            for (idx, normal) in normals.iter().enumerate() {
                rebuilt.set_vertex_normal_by_index(idx, *normal);
            }
        }
        if preserve_colors {
            rebuilt.request_vertex_colors();
            for (idx, color) in colors.iter().enumerate() {
                rebuilt.set_vertex_color_by_index(idx, *color);
            }
        }
        if preserve_texcoords {
            rebuilt.request_vertex_texcoords();
            for (idx, texcoord) in texcoords.iter().enumerate() {
                rebuilt.set_vertex_texcoord_by_index(idx, *texcoord);
            }
        }

        rebuild_polygon_faces(&mut rebuilt, faces);

        *self = rebuilt;
    }

    // =========================================================================
    // Normal computation
    // =========================================================================

    /// Compute a normalized face normal.
    ///
    /// Returns `Vec3::ZERO` for invalid, deleted, or degenerate faces.
    pub fn calc_face_normal(&self, fh: FaceHandle) -> glam::Vec3 {
        self.calc_face_area_normal(fh).normalize_or_zero()
    }

    /// Compute a vertex normal with the requested weighting mode.
    ///
    /// Returns `Vec3::ZERO` for invalid, deleted, or isolated vertices.
    pub fn calc_vertex_normal_with_mode(
        &self,
        vh: VertexHandle,
        weighting: VertexNormalWeighting,
    ) -> glam::Vec3 {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return glam::Vec3::ZERO;
        }

        let mut accumulated = glam::Vec3::ZERO;

        if let Some(faces) = self.vertex_faces(vh) {
            for fh in faces {
                let face_area_normal = self.calc_face_area_normal(fh);
                accumulated +=
                    Self::vertex_normal_contribution_from_area_normal(weighting, face_area_normal);
            }
        }

        if accumulated.length_squared() > 0.0 {
            accumulated.normalize()
        } else {
            glam::Vec3::ZERO
        }
    }

    /// Compute the default RustMesh vertex normal.
    ///
    /// RustMesh defaults to area-weighted accumulation. Use
    /// [`RustMesh::calc_vertex_normal_with_mode`] with
    /// [`VertexNormalWeighting::FaceAverage`] for an OpenMesh-compatible path.
    pub fn calc_vertex_normal(&self, vh: VertexHandle) -> glam::Vec3 {
        self.calc_vertex_normal_with_mode(vh, VertexNormalWeighting::AreaWeighted)
    }

    /// Recompute and store all face normals.
    pub fn update_face_normals(&mut self) {
        self.request_face_normals();
        let Some(face_normals_ptr) = self.kernel.face_normals_mut_ptr() else {
            return;
        };
        let n_faces = self.n_faces();
        for idx in 0..n_faces {
            let fh = FaceHandle::from_usize(idx);
            let area_normal = self.calc_face_area_normal(fh);
            let normal = area_normal.normalize_or_zero();
            unsafe {
                *face_normals_ptr.add(idx) = normal;
            }
        }
    }

    /// Recompute and store all vertex normals using the requested weighting mode.
    pub fn update_vertex_normals_with_mode(&mut self, weighting: VertexNormalWeighting) {
        let mut accumulated = vec![glam::Vec3::ZERO; self.n_vertices()];

        let n_faces = self.n_faces();
        for idx in 0..n_faces {
            let fh = FaceHandle::from_usize(idx);
            if let Some((vertex_indices, face_area_normal)) =
                self.triangle_face_vertex_indices_and_area_normal(fh)
            {
                let contribution =
                    Self::vertex_normal_contribution_from_area_normal(weighting, face_area_normal);
                for vertex_idx in vertex_indices {
                    accumulated[vertex_idx] += contribution;
                }
                continue;
            }

            let face_area_normal = self.calc_face_area_normal(fh);
            let contribution =
                Self::vertex_normal_contribution_from_area_normal(weighting, face_area_normal);
            self.accumulate_face_normal_contribution(fh, contribution, &mut accumulated);
        }

        self.request_vertex_normals();
        let Some(vertex_normals_ptr) = self.kernel.vertex_normals_mut_ptr() else {
            return;
        };
        for (idx, normal) in accumulated.into_iter().enumerate() {
            let normalized = normal.normalize_or_zero();
            unsafe {
                *vertex_normals_ptr.add(idx) = normalized;
            }
        }
    }

    /// Recompute and store all vertex normals using RustMesh's default area-weighted mode.
    pub fn update_vertex_normals(&mut self) {
        self.update_vertex_normals_with_mode(VertexNormalWeighting::AreaWeighted);
    }

    /// Recompute both face and vertex normals using the requested vertex-normal weighting mode.
    pub fn update_normals_with_mode(&mut self, weighting: VertexNormalWeighting) {
        let mut accumulated = vec![glam::Vec3::ZERO; self.n_vertices()];

        self.request_face_normals();
        self.request_vertex_normals();
        let Some(face_normals_ptr) = self.kernel.face_normals_mut_ptr() else {
            return;
        };
        let Some(vertex_normals_ptr) = self.kernel.vertex_normals_mut_ptr() else {
            return;
        };

        let n_faces = self.n_faces();
        for idx in 0..n_faces {
            let fh = FaceHandle::from_usize(idx);
            if let Some((vertex_indices, face_area_normal)) =
                self.triangle_face_vertex_indices_and_area_normal(fh)
            {
                let normal = face_area_normal.normalize_or_zero();
                unsafe {
                    *face_normals_ptr.add(idx) = normal;
                }

                let contribution =
                    Self::vertex_normal_contribution_from_area_normal(weighting, face_area_normal);
                for vertex_idx in vertex_indices {
                    accumulated[vertex_idx] += contribution;
                }
                continue;
            }

            let face_area_normal = self.calc_face_area_normal(fh);
            let normal = face_area_normal.normalize_or_zero();
            unsafe {
                *face_normals_ptr.add(idx) = normal;
            }
            let contribution =
                Self::vertex_normal_contribution_from_area_normal(weighting, face_area_normal);
            self.accumulate_face_normal_contribution(fh, contribution, &mut accumulated);
        }

        for (idx, normal) in accumulated.into_iter().enumerate() {
            let normalized = normal.normalize_or_zero();
            unsafe {
                *vertex_normals_ptr.add(idx) = normalized;
            }
        }
    }

    /// Recompute both face and vertex normals using RustMesh's default area-weighted mode.
    pub fn update_normals(&mut self) {
        self.update_normals_with_mode(VertexNormalWeighting::AreaWeighted);
    }

    #[inline]
    fn vertex_normal_contribution_from_area_normal(
        weighting: VertexNormalWeighting,
        face_area_normal: glam::Vec3,
    ) -> glam::Vec3 {
        match weighting {
            VertexNormalWeighting::AreaWeighted => face_area_normal,
            VertexNormalWeighting::FaceAverage => face_area_normal.normalize_or_zero(),
        }
    }

    #[inline]
    fn calc_face_area_normal(&self, fh: FaceHandle) -> glam::Vec3 {
        if !fh.is_valid() || self.is_face_deleted(fh) {
            return glam::Vec3::ZERO;
        }

        if let Some((_, area_normal)) = self.triangle_face_vertex_indices_and_area_normal(fh) {
            return area_normal;
        }

        let Some(first_heh) = self.face_halfedge_handle(fh) else {
            return glam::Vec3::ZERO;
        };

        let mut current_heh = first_heh;
        let mut accumulated = glam::Vec3::ZERO;
        let mut steps = 0usize;
        let max_steps = self.n_halfedges().max(1);

        loop {
            if steps >= max_steps || !current_heh.is_valid() {
                return glam::Vec3::ZERO;
            }
            steps += 1;

            let next_heh = self.next_halfedge_handle(current_heh);
            if !next_heh.is_valid() {
                return glam::Vec3::ZERO;
            }

            let current_vh = self.to_vertex_handle(current_heh);
            let next_vh = self.to_vertex_handle(next_heh);
            let (Some(current_point), Some(next_point)) =
                (self.point(current_vh), self.point(next_vh))
            else {
                return glam::Vec3::ZERO;
            };

            accumulated += current_point.cross(next_point);

            current_heh = next_heh;
            if current_heh == first_heh {
                break;
            }
        }

        if steps >= 3 {
            accumulated
        } else {
            glam::Vec3::ZERO
        }
    }

    #[inline]
    fn triangle_face_vertex_indices_and_area_normal(
        &self,
        fh: FaceHandle,
    ) -> Option<([usize; 3], glam::Vec3)> {
        self.kernel.triangle_face_vertex_indices_and_area_normal(fh)
    }

    #[inline]
    fn accumulate_face_normal_contribution(
        &self,
        fh: FaceHandle,
        contribution: glam::Vec3,
        accumulated: &mut [glam::Vec3],
    ) {
        if contribution.length_squared() == 0.0 {
            return;
        }

        let Some(first_heh) = self.face_halfedge_handle(fh) else {
            return;
        };

        let mut current_heh = first_heh;
        let mut steps = 0usize;
        let max_steps = self.n_halfedges().max(1);

        loop {
            if steps >= max_steps || !current_heh.is_valid() {
                break;
            }
            steps += 1;

            let vh = self.to_vertex_handle(current_heh);
            if vh.is_valid() && !self.is_vertex_deleted(vh) {
                accumulated[vh.idx_usize()] += contribution;
            }

            let next_heh = self.next_halfedge_handle(current_heh);
            if !next_heh.is_valid() || next_heh == current_heh {
                break;
            }

            current_heh = next_heh;
            if current_heh == first_heh {
                break;
            }
        }
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

    // =========================================================================
    // IO helper methods (index-based access for export)
    // =========================================================================

    /// Get vertex position by index (for IO operations)
    pub fn point_by_index(&self, idx: usize) -> Option<glam::Vec3> {
        self.kernel.point(idx)
    }

    /// Get vertex normal by index (for IO operations)
    pub fn vertex_normal_by_index(&self, idx: usize) -> Option<glam::Vec3> {
        if idx < self.n_vertices() {
            self.kernel.vertex_normal(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Get vertex color by index (for IO operations)
    pub fn vertex_color_by_index(&self, idx: usize) -> Option<glam::Vec4> {
        if idx < self.n_vertices() {
            self.kernel.vertex_color(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Get vertex texcoord by index (for IO operations)
    pub fn vertex_texcoord_by_index(&self, idx: usize) -> Option<glam::Vec2> {
        if idx < self.n_vertices() {
            self.kernel.vertex_texcoord(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Set vertex normal by index (for IO operations)
    pub fn set_vertex_normal_by_index(&mut self, idx: usize, normal: glam::Vec3) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_normal(VertexHandle::from_usize(idx), normal);
        }
    }

    /// Set vertex color by index (for IO operations)
    pub fn set_vertex_color_by_index(&mut self, idx: usize, color: glam::Vec4) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_color(VertexHandle::from_usize(idx), color);
        }
    }

    /// Set vertex texcoord by index (for IO operations)
    pub fn set_vertex_texcoord_by_index(&mut self, idx: usize, texcoord: glam::Vec2) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_texcoord(VertexHandle::from_usize(idx), texcoord);
        }
    }

    /// Get all vertices of a face (for IO operations - returns Vec)
    pub fn face_vertices_vec(&self, fh: FaceHandle) -> Vec<VertexHandle> {
        let mut vertices = Vec::new();

        // Get the first halfedge of the face
        if let Some(first_heh) = self.kernel.face_halfedge_handle(fh) {
            let mut current_heh = first_heh;
            let mut steps = 0usize;
            let max_steps = self.n_halfedges().max(1);

            loop {
                if steps >= max_steps || !current_heh.is_valid() {
                    break;
                }
                steps += 1;

                // Get the target vertex of this halfedge
                let to_vh = self.kernel.to_vertex_handle(current_heh);
                if !to_vh.is_valid() {
                    break;
                }
                vertices.push(to_vh);

                // Move to next halfedge
                if let Some(next_heh) = self.kernel.next_halfedge_handle(current_heh) {
                    if !next_heh.is_valid() || next_heh == current_heh {
                        break;
                    }
                    current_heh = next_heh;
                    if current_heh == first_heh {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        vertices
    }

    fn sanitized_face_vertices(&self, fh: FaceHandle) -> Vec<VertexHandle> {
        dedupe_face_vertices(self.face_vertices_vec(fh))
    }

    // =========================================================================
    // Conversion from RustSLAM mesh
    // =========================================================================

    /// Create RustMesh from simple triangle mesh (e.g., from RustSLAM marching cubes)
    ///
    /// Accepts:
    /// - vertices: list of positions
    /// - triangles: list of (v0, v1, v2) index triplets
    /// - normals: optional per-vertex normals
    /// - colors: optional per-vertex colors (RGB as [f32; 3])
    pub fn from_triangle_mesh(
        vertices: &[glam::Vec3],
        triangles: &[[usize; 3]],
        normals: Option<&[glam::Vec3]>,
        colors: Option<&[[f32; 3]]>,
    ) -> Self {
        let mut mesh = RustMesh::new();

        // Add vertices
        for pos in vertices {
            mesh.add_vertex(*pos);
        }

        // Add normals if provided
        if let Some(norms) = normals {
            mesh.request_vertex_normals();
            for (i, normal) in norms.iter().enumerate() {
                if i < mesh.n_vertices() {
                    mesh.set_vertex_normal_by_index(i, *normal);
                }
            }
        }

        // Add colors if provided
        if let Some(cols) = colors {
            mesh.request_vertex_colors();
            for (i, color) in cols.iter().enumerate() {
                if i < mesh.n_vertices() {
                    // Convert RGB to RGBA (add alpha = 1.0)
                    let rgba = glam::Vec4::new(color[0], color[1], color[2], 1.0);
                    mesh.set_vertex_color_by_index(i, rgba);
                }
            }
        }

        // Add faces
        for tri in triangles {
            let v0 = VertexHandle::from_usize(tri[0]);
            let v1 = VertexHandle::from_usize(tri[1]);
            let v2 = VertexHandle::from_usize(tri[2]);
            mesh.add_face(&[v0, v1, v2]);
        }

        mesh
    }
}

fn dedupe_face_vertices(vertices: Vec<VertexHandle>) -> Vec<VertexHandle> {
    if vertices.is_empty() {
        return vertices;
    }

    let mut deduped: Vec<VertexHandle> = Vec::with_capacity(vertices.len());
    for vh in vertices {
        if deduped.last().copied() != Some(vh) {
            deduped.push(vh);
        }
    }

    while deduped.len() >= 2 && deduped.first() == deduped.last() {
        deduped.pop();
    }

    let mut unique = Vec::with_capacity(deduped.len());
    for vh in deduped {
        if !unique.contains(&vh) {
            unique.push(vh);
        }
    }
    unique
}

fn canonical_face_key(vertices: &[VertexHandle]) -> Vec<usize> {
    let ids: Vec<usize> = vertices.iter().map(|vh| vh.idx_usize()).collect();
    if ids.is_empty() {
        return ids;
    }

    let mut best: Option<Vec<usize>> = None;
    for base in [ids.clone(), ids.iter().copied().rev().collect()] {
        for shift in 0..base.len() {
            let rotated: Vec<usize> = base
                .iter()
                .cycle()
                .skip(shift)
                .take(base.len())
                .copied()
                .collect();

            if best.as_ref().is_none_or(|current| rotated < *current) {
                best = Some(rotated);
            }
        }
    }

    best.unwrap_or_default()
}

fn triangulate_face_loop_with_vertex(
    face: &[VertexHandle],
    new_vh: VertexHandle,
) -> Vec<Vec<VertexHandle>> {
    let mut triangles = Vec::with_capacity(face.len());
    for idx in 0..face.len() {
        triangles.push(vec![face[idx], face[(idx + 1) % face.len()], new_vh]);
    }
    triangles
}

fn triangulate_face_loop(face: &[VertexHandle]) -> Vec<Vec<VertexHandle>> {
    if face.len() <= 3 {
        return vec![face.to_vec()];
    }

    let anchor = face[0];
    let mut triangles = Vec::with_capacity(face.len() - 2);
    for idx in 1..(face.len() - 1) {
        triangles.push(vec![anchor, face[idx], face[idx + 1]]);
    }
    triangles
}

fn try_add_face_with_rotations(mesh: &mut RustMesh, vertices: &[VertexHandle]) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let len = vertices.len();
    let mut candidate = vertices.to_vec();
    for _ in 0..len {
        if mesh.add_face(&candidate).is_some() {
            return true;
        }
        candidate.rotate_left(1);
    }

    candidate = vertices.iter().copied().rev().collect();
    for _ in 0..len {
        if mesh.add_face(&candidate).is_some() {
            return true;
        }
        candidate.rotate_left(1);
    }

    false
}

fn rebuild_polygon_faces(mesh: &mut RustMesh, faces: &[Vec<VertexHandle>]) {
    for face in faces {
        let _ = try_add_face_with_rotations(mesh, face);
    }
}

fn rebuild_oriented_triangles(mesh: &mut RustMesh, faces: &[Vec<VertexHandle>]) {
    let triangles: Vec<[usize; 3]> = faces
        .iter()
        .map(|face| {
            [
                face[0].idx_usize(),
                face[1].idx_usize(),
                face[2].idx_usize(),
            ]
        })
        .collect();

    let flips = orient_triangle_faces(&triangles);
    for (triangle, flip) in triangles.iter().zip(flips.iter()) {
        let mut face = vec![
            VertexHandle::from_usize(triangle[0]),
            VertexHandle::from_usize(triangle[1]),
            VertexHandle::from_usize(triangle[2]),
        ];
        if *flip {
            face.swap(1, 2);
        }
        let _ = try_add_face_with_rotations(mesh, &face);
    }
}

fn orient_triangle_faces(triangles: &[[usize; 3]]) -> Vec<bool> {
    #[derive(Clone, Copy)]
    struct EdgeUse {
        triangle_idx: usize,
        sign: i8,
    }

    let mut edge_map: HashMap<(usize, usize), Vec<EdgeUse>> = HashMap::new();
    for (triangle_idx, triangle) in triangles.iter().enumerate() {
        for (from, to) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if from < to { (from, to) } else { (to, from) };
            let sign = if from < to { 1 } else { -1 };
            edge_map
                .entry(key)
                .or_default()
                .push(EdgeUse { triangle_idx, sign });
        }
    }

    let mut flips: Vec<Option<bool>> = vec![None; triangles.len()];
    for start in 0..triangles.len() {
        if flips[start].is_some() {
            continue;
        }

        flips[start] = Some(false);
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(current_idx) = queue.pop_front() {
            let current_flip = flips[current_idx].unwrap_or(false);
            let triangle = triangles[current_idx];

            for (from, to) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let key = if from < to { (from, to) } else { (to, from) };
                let current_sign = if from < to { 1 } else { -1 };

                let Some(uses) = edge_map.get(&key) else {
                    continue;
                };
                for other in uses {
                    if other.triangle_idx == current_idx {
                        continue;
                    }

                    let should_flip_other = current_flip ^ (current_sign == other.sign);
                    match flips[other.triangle_idx] {
                        Some(existing) if existing != should_flip_other => {}
                        Some(_) => {}
                        None => {
                            flips[other.triangle_idx] = Some(should_flip_other);
                            queue.push_back(other.triangle_idx);
                        }
                    }
                }
            }
        }
    }

    flips
        .into_iter()
        .map(|flip| flip.unwrap_or(false))
        .collect()
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
        assert_eq!(
            mesh.f_normal(FaceHandle::new(0)),
            Some(glam::vec3(0.0, 0.0, 1.0))
        );

        // Request face colors
        mesh.request_face_colors();
        assert!(mesh.has_face_colors());

        mesh.set_f_color(FaceHandle::new(0), glam::vec4(0.5, 0.5, 0.5, 1.0));
        assert_eq!(
            mesh.f_color(FaceHandle::new(0)),
            Some(glam::vec4(0.5, 0.5, 0.5, 1.0))
        );
    }

    #[test]
    fn test_update_face_normals_and_calc_face_normal() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        let expected = glam::vec3(0.0, 0.0, 1.0);
        let calc = mesh.calc_face_normal(fh);
        assert!((calc - expected).length() < 1e-6);

        mesh.update_face_normals();
        let stored = mesh.f_normal(fh).unwrap();
        assert!((stored - expected).length() < 1e-6);
    }

    #[test]
    fn test_update_vertex_normals_area_weighted() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 0.0, 1.0));

        mesh.add_face(&[v0, v1, v2]).unwrap();
        mesh.add_face(&[v0, v2, v3]).unwrap();

        let expected = glam::vec3(1.0, 0.0, 1.0).normalize();

        let calc = mesh.calc_vertex_normal(v0);
        assert!((calc - expected).length() < 1e-6);

        mesh.update_vertex_normals();
        let stored = mesh.normal(v0).unwrap();
        assert!((stored - expected).length() < 1e-6);
    }

    #[test]
    fn test_vertex_normal_face_average_mode_matches_openmesh_style_accumulation() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(2.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 0.0, 1.0));
        let f0 = mesh.add_face(&[v0, v1, v2]).unwrap();
        let f1 = mesh.add_face(&[v0, v2, v3]).unwrap();

        let expected_area_weighted = glam::vec3(1.0, 0.0, 2.0).normalize();
        let expected_face_average = glam::vec3(1.0, 0.0, 1.0).normalize();

        let area_weighted = mesh.calc_vertex_normal(v0);
        let face_average =
            mesh.calc_vertex_normal_with_mode(v0, VertexNormalWeighting::FaceAverage);

        assert!((area_weighted - expected_area_weighted).length() < 1e-6);
        assert!((face_average - expected_face_average).length() < 1e-6);

        mesh.update_vertex_normals_with_mode(VertexNormalWeighting::FaceAverage);
        assert!((mesh.normal(v0).unwrap() - expected_face_average).length() < 1e-6);

        mesh.update_normals_with_mode(VertexNormalWeighting::FaceAverage);
        assert!((mesh.normal(v0).unwrap() - expected_face_average).length() < 1e-6);
        assert!((mesh.f_normal(f0).unwrap() - glam::vec3(0.0, 0.0, 1.0)).length() < 1e-6);
        assert!((mesh.f_normal(f1).unwrap() - glam::vec3(1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_update_normals_default_mode_stays_area_weighted_and_faces_stay_mode_independent() {
        let mut default_mesh = RustMesh::new();

        let v0 = default_mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = default_mesh.add_vertex(glam::vec3(2.0, 0.0, 0.0));
        let v2 = default_mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let v3 = default_mesh.add_vertex(glam::vec3(0.0, 0.0, 1.0));
        let f0 = default_mesh.add_face(&[v0, v1, v2]).unwrap();
        let f1 = default_mesh.add_face(&[v0, v2, v3]).unwrap();

        let mut explicit_area_mesh = default_mesh.clone();
        let mut face_average_mesh = default_mesh.clone();

        let expected_area_weighted = glam::vec3(1.0, 0.0, 2.0).normalize();
        let expected_face_average = glam::vec3(1.0, 0.0, 1.0).normalize();
        let expected_face0 = glam::vec3(0.0, 0.0, 1.0);
        let expected_face1 = glam::vec3(1.0, 0.0, 0.0);

        default_mesh.update_normals();
        explicit_area_mesh.update_normals_with_mode(VertexNormalWeighting::AreaWeighted);
        face_average_mesh.update_normals_with_mode(VertexNormalWeighting::FaceAverage);

        let default_vertex = default_mesh.normal(v0).unwrap();
        let explicit_area_vertex = explicit_area_mesh.normal(v0).unwrap();
        let compatible_vertex = face_average_mesh.normal(v0).unwrap();

        assert!((default_vertex - expected_area_weighted).length() < 1e-6);
        assert!((explicit_area_vertex - expected_area_weighted).length() < 1e-6);
        assert!((compatible_vertex - expected_face_average).length() < 1e-6);
        assert!((default_vertex - compatible_vertex).length() > 1e-3);

        for mesh in [&default_mesh, &explicit_area_mesh, &face_average_mesh] {
            assert!((mesh.f_normal(f0).unwrap() - expected_face0).length() < 1e-6);
            assert!((mesh.f_normal(f1).unwrap() - expected_face1).length() < 1e-6);
        }
    }

    #[test]
    fn test_split_edge_does_not_auto_refresh_normals() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();
        mesh.add_face(&[v0, v2, v3]).unwrap();

        mesh.update_normals();
        let stored_before = mesh.normal(v0).unwrap();

        let eh = mesh.find_edge_between(v0, v2).unwrap();
        let new_vh = mesh.split_edge(eh, glam::vec3(0.5, 0.5, 1.0)).unwrap();

        assert!((mesh.normal(v0).unwrap() - stored_before).length() < 1e-6);
        assert!((mesh.calc_vertex_normal(v0) - stored_before).length() > 1e-3);
        assert_eq!(mesh.normal(new_vh), Some(glam::Vec3::ZERO));

        let zero_face_normals = mesh
            .faces()
            .filter_map(|fh| mesh.f_normal(fh))
            .filter(|normal| normal.length_squared() == 0.0)
            .count();
        assert_eq!(zero_face_normals, 2);
    }

    #[test]
    fn test_split_face_triangle_does_not_auto_refresh_normals() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        mesh.update_normals();
        let stored_face_before = mesh.f_normal(fh).unwrap();

        let new_vh = mesh.split_face(fh, glam::vec3(0.25, 0.25, 1.0)).unwrap();

        assert!((mesh.f_normal(fh).unwrap() - stored_face_before).length() < 1e-6);
        assert!((mesh.calc_face_normal(fh) - stored_face_before).length() > 1e-3);
        assert_eq!(mesh.normal(new_vh), Some(glam::Vec3::ZERO));

        let zero_face_normals = mesh
            .faces()
            .filter(|&split_fh| !mesh.is_face_deleted(split_fh))
            .filter_map(|split_fh| mesh.f_normal(split_fh))
            .filter(|normal| normal.length_squared() == 0.0)
            .count();
        assert_eq!(zero_face_normals, 2);
    }

    #[test]
    fn test_collapse_does_not_auto_refresh_normals() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 1.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();
        mesh.add_face(&[v0, v2, v3]).unwrap();

        mesh.update_normals();
        let stored_before = mesh.normal(v1).unwrap();

        let eh = mesh.find_edge_between(v0, v1).unwrap();
        let he0 = mesh.edge_halfedge_handle(eh, 0);
        let heh = if mesh.from_vertex_handle(he0) == v0 {
            he0
        } else {
            mesh.opposite_halfedge_handle(he0)
        };

        assert!(mesh.is_collapse_ok(heh));
        mesh.collapse(heh).unwrap();

        assert!((mesh.normal(v1).unwrap() - stored_before).length() < 1e-6);
        assert!((mesh.calc_vertex_normal(v1) - stored_before).length() > 1e-3);
    }

    #[test]
    fn test_triangulate_face_drops_face_normals_until_refresh() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 1.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2, v3]).unwrap();

        mesh.update_normals();
        assert!(mesh.has_face_normals());
        assert!(mesh.has_vertex_normals());
        assert!(mesh.f_normal(fh).unwrap().length() > 0.0);

        mesh.triangulate_face(fh).unwrap();

        assert!(!mesh.has_face_normals());
        assert!(mesh.has_vertex_normals());

        mesh.update_normals();
        assert!(mesh.has_face_normals());
    }

    #[test]
    fn test_update_normals_requests_and_updates_both_arrays() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        mesh.update_normals();

        assert!(mesh.has_face_normals());
        assert!(mesh.has_vertex_normals());

        let expected = glam::vec3(0.0, 0.0, 1.0);
        assert!((mesh.f_normal(fh).unwrap() - expected).length() < 1e-6);
        assert!((mesh.normal(v0).unwrap() - expected).length() < 1e-6);
        assert!((mesh.normal(v1).unwrap() - expected).length() < 1e-6);
        assert!((mesh.normal(v2).unwrap() - expected).length() < 1e-6);
    }

    #[test]
    fn test_split_edge_boundary_triangle_updates_counts() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(2.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();

        let eh = mesh.find_edge_between(v0, v1).unwrap();
        let new_vh = mesh.split_edge(eh, glam::vec3(1.0, 0.0, 0.0)).unwrap();

        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_edges(), 5);
        assert_eq!(mesh.n_active_faces(), 2);
        assert_eq!(mesh.point(new_vh), Some(glam::vec3(1.0, 0.0, 0.0)));
        assert!(mesh.is_boundary_vertex(new_vh));

        let actual: Vec<Vec<usize>> = mesh
            .faces()
            .map(|fh| canonical_face_key(&mesh.face_vertices_vec(fh)))
            .collect();
        assert_eq!(actual.len(), 2);
        assert!(actual.contains(&vec![0, 2, 3]));
        assert!(actual.contains(&vec![1, 2, 3]));
    }

    #[test]
    fn test_split_edge_interior_triangle_pair_creates_four_triangles() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();
        mesh.add_face(&[v0, v2, v3]).unwrap();

        let eh = mesh.find_edge_between(v0, v2).unwrap();
        let new_vh = mesh.split_edge(eh, glam::vec3(0.5, 0.5, 0.0)).unwrap();

        assert_eq!(mesh.n_vertices(), 5);
        assert_eq!(mesh.n_edges(), 8);
        assert_eq!(mesh.n_active_faces(), 4);
        assert_eq!(mesh.point(new_vh), Some(glam::vec3(0.5, 0.5, 0.0)));
        assert!(!mesh.is_boundary_vertex(new_vh));

        let actual: Vec<Vec<usize>> = mesh
            .faces()
            .map(|fh| canonical_face_key(&mesh.face_vertices_vec(fh)))
            .collect();
        assert_eq!(actual.len(), 4);
        assert!(actual.contains(&vec![0, 1, 4]));
        assert!(actual.contains(&vec![1, 2, 4]));
        assert!(actual.contains(&vec![0, 3, 4]));
        assert!(actual.contains(&vec![2, 3, 4]));
    }

    #[test]
    fn test_split_edge_propagates_vertex_edge_and_face_properties() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let f0 = mesh.add_face(&[v0, v1, v2]).unwrap();
        let f1 = mesh.add_face(&[v0, v2, v3]).unwrap();

        let vprop = mesh.add_vertex_property::<f32>("vprop");
        let eprop = mesh.add_edge_property::<i32>("eprop");
        let fprop = mesh.add_face_property::<i32>("fprop");

        assert!(mesh.set_vertex_property(vprop, v0, 2.0));
        assert!(mesh.set_vertex_property(vprop, v2, 8.0));

        let eh = mesh.find_edge_between(v0, v2).unwrap();
        let he0 = mesh.edge_halfedge_handle(eh, 0);
        let split_to = mesh.to_vertex_handle(he0);
        assert!(mesh.set_edge_property(eprop, eh, 9));
        assert!(mesh.set_face_property(fprop, f0, 11));
        assert!(mesh.set_face_property(fprop, f1, 17));

        let new_vh = mesh.split_edge(eh, glam::vec3(0.5, 0.5, 0.0)).unwrap();

        assert_eq!(mesh.vertex_property(vprop, new_vh), Some(5.0));

        let new_split_edge = mesh.find_edge_between(new_vh, split_to).unwrap();
        assert_eq!(mesh.edge_property(eprop, eh), Some(9));
        assert_eq!(mesh.edge_property(eprop, new_split_edge), Some(9));

        let actual: Vec<(Vec<usize>, i32)> = mesh
            .faces()
            .filter(|&fh| !mesh.is_face_deleted(fh))
            .map(|fh| {
                (
                    canonical_face_key(&mesh.face_vertices_vec(fh)),
                    mesh.face_property(fprop, fh).unwrap(),
                )
            })
            .collect();

        assert_eq!(actual.len(), 4);
        assert!(actual.contains(&(vec![0, 1, 4], 11)));
        assert!(actual.contains(&(vec![1, 2, 4], 11)));
        assert!(actual.contains(&(vec![0, 3, 4], 17)));
        assert!(actual.contains(&(vec![2, 3, 4], 17)));
    }

    #[test]
    fn test_split_edge_keeps_normals_arrays_usable_after_face_growth() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(2.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();

        mesh.request_face_normals();
        mesh.request_vertex_normals();

        let eh = mesh.find_edge_between(v0, v1).unwrap();
        let _ = mesh.split_edge(eh, glam::vec3(1.0, 0.0, 0.0)).unwrap();
        mesh.update_normals();

        assert_eq!(mesh.n_active_faces(), 2);
        for fh in mesh.faces() {
            assert!(mesh.f_normal(fh).unwrap().length() > 0.0);
        }
    }

    #[test]
    fn test_split_face_triangle_creates_three_triangles() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        let new_vh = mesh.split_face(fh, glam::vec3(0.25, 0.25, 0.0)).unwrap();

        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 3);
        assert_eq!(mesh.n_edges(), 6);
        assert_eq!(mesh.n_halfedges(), 12);
        assert_eq!(mesh.n_active_faces(), 3);
        assert_eq!(mesh.point(new_vh), Some(glam::vec3(0.25, 0.25, 0.0)));

        let actual: Vec<Vec<usize>> = mesh
            .faces()
            .filter(|&split_fh| !mesh.is_face_deleted(split_fh))
            .map(|split_fh| canonical_face_key(&mesh.face_vertices_vec(split_fh)))
            .collect();
        assert_eq!(actual.len(), 3);
        assert!(actual.contains(&vec![0, 1, 3]));
        assert!(actual.contains(&vec![1, 2, 3]));
        assert!(actual.contains(&vec![0, 2, 3]));
    }

    #[test]
    fn test_split_face_triangle_keeps_local_connectivity_valid() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        let _ = mesh.split_face(fh, glam::vec3(0.25, 0.25, 0.0)).unwrap();

        assert!(mesh.validate().is_ok());
        assert!(!mesh.is_face_deleted(fh));
        assert_eq!(mesh.face_vertices_vec(fh).len(), 3);
    }

    #[test]
    fn test_split_face_triangle_propagates_vertex_and_face_properties() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        let vprop = mesh.add_vertex_property::<glam::Vec3>("vprop");
        let fprop = mesh.add_face_property::<i32>("fprop");

        assert!(mesh.set_vertex_property(vprop, v0, glam::vec3(0.0, 0.0, 0.0)));
        assert!(mesh.set_vertex_property(vprop, v1, glam::vec3(3.0, 0.0, 0.0)));
        assert!(mesh.set_vertex_property(vprop, v2, glam::vec3(0.0, 6.0, 0.0)));
        assert!(mesh.set_face_property(fprop, fh, 23));

        let new_vh = mesh.split_face(fh, glam::vec3(0.25, 0.25, 0.0)).unwrap();

        assert_eq!(
            mesh.vertex_property(vprop, new_vh),
            Some(glam::vec3(1.0, 2.0, 0.0))
        );

        for split_fh in mesh
            .faces()
            .filter(|&split_fh| !mesh.is_face_deleted(split_fh))
        {
            assert_eq!(mesh.face_property(fprop, split_fh), Some(23));
        }
    }

    #[test]
    fn test_split_face_keeps_normals_arrays_usable_after_face_growth() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        mesh.request_face_normals();
        mesh.request_vertex_normals();

        let _ = mesh.split_face(fh, glam::vec3(0.25, 0.25, 0.0)).unwrap();
        mesh.update_normals();

        assert_eq!(mesh.n_active_faces(), 3);
        for split_fh in mesh
            .faces()
            .filter(|&split_fh| !mesh.is_face_deleted(split_fh))
        {
            assert!(mesh.f_normal(split_fh).unwrap().length() > 0.0);
        }
    }

    #[test]
    fn test_collapse_propagates_surviving_vertex_property() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]).unwrap();
        mesh.add_face(&[v0, v2, v3]).unwrap();

        let vprop = mesh.add_vertex_property::<i32>("collapse_vprop");
        assert!(mesh.set_vertex_property(vprop, v0, 10));
        assert!(mesh.set_vertex_property(vprop, v1, 4));

        let eh = mesh.find_edge_between(v0, v1).unwrap();
        let he0 = mesh.edge_halfedge_handle(eh, 0);
        let heh = if mesh.from_vertex_handle(he0) == v0 {
            he0
        } else {
            mesh.opposite_halfedge_handle(he0)
        };

        assert!(mesh.is_collapse_ok(heh));
        mesh.collapse(heh).unwrap();

        assert_eq!(mesh.vertex_property(vprop, v1), Some(7));
        assert!(mesh.is_vertex_deleted(v0));
        assert_eq!(mesh.n_active_faces(), 1);
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn test_triangulate_face_quad_creates_two_triangles() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2, v3]).unwrap();

        mesh.triangulate_face(fh).unwrap();

        assert_eq!(mesh.n_active_faces(), 2);
        let actual: Vec<Vec<usize>> = mesh
            .faces()
            .map(|tri_fh| canonical_face_key(&mesh.face_vertices_vec(tri_fh)))
            .collect();
        assert_eq!(actual.len(), 2);
        let diagonal_02 = actual.contains(&vec![0, 1, 2]) && actual.contains(&vec![0, 2, 3]);
        let diagonal_13 = actual.contains(&vec![0, 1, 3]) && actual.contains(&vec![1, 2, 3]);
        assert!(diagonal_02 || diagonal_13);
    }

    #[test]
    fn test_triangulate_face_triangle_is_noop() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        let fh = mesh.add_face(&[v0, v1, v2]).unwrap();

        mesh.triangulate_face(fh).unwrap();

        assert_eq!(mesh.n_active_faces(), 1);
        assert_eq!(
            canonical_face_key(&mesh.face_vertices_vec(fh)),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_add_face_openmesh_parity_keeps_boundary_vertex_handles_on_boundary() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        let f0 = mesh.add_face_openmesh_parity(&[v0, v1, v2]).unwrap();
        let f1 = mesh.add_face_openmesh_parity(&[v0, v2, v3]).unwrap();

        assert_eq!(mesh.face_vertices_vec(f0), vec![v0, v1, v2]);
        assert_eq!(mesh.face_vertices_vec(f1), vec![v0, v2, v3]);

        for vh in [v0, v1, v2, v3] {
            let heh = mesh
                .halfedge_handle(vh)
                .expect("square boundary vertex should keep an outgoing halfedge");
            assert_eq!(mesh.from_vertex_handle(heh), vh);
            assert!(
                mesh.is_boundary(heh),
                "vertex {:?} anchor should stay on a boundary halfedge, got {:?}",
                vh,
                heh
            );
        }
    }
}

// RustMesh is now the primary mesh type (previously PolyMeshSoA)
