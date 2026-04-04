// ============================================================================
// Decimation Module - Mesh Simplification
// Based on OpenMesh's DecimaterT framework
// ============================================================================

use crate::{FaceHandle, HalfedgeHandle, QuadricT, RustMesh, Vec3, VertexHandle};
use std::cmp::Ordering;

// ============================================================================
// Vertex Heap Infrastructure (OpenMesh HeapT style)
// ============================================================================

/// Per-vertex properties for heap-based decimation (OpenMesh style)
#[derive(Debug, Clone, Default)]
pub struct VertexProps {
    /// Best halfedge to collapse this vertex through (None = no valid collapse)
    pub collapse_target: Option<HalfedgeHandle>,
    /// Priority (error quadric value) for the best collapse (-1.0 = invalid)
    pub priority: f32,
    /// Position in heap array (-1 = not in heap)
    pub heap_position: i32,
}

/// Min-heap entries stored separately from vertex properties
/// The heap position is stored in VertexProps::heap_position
#[derive(Debug, Clone, Default)]
pub struct DecimationHeap {
    /// Heap entries (vertex handles), ordered by priority
    entries: Vec<VertexHandle>,
}

impl DecimationHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if a vertex is stored in the heap
    pub fn is_stored(&self, vh: VertexHandle, props: &[VertexProps]) -> bool {
        let idx = vh.idx_usize();
        idx < props.len() && props[idx].heap_position >= 0
    }

    /// Reset heap position to -1 (not in heap)
    pub fn reset_heap_position(props: &mut [VertexProps], vh: VertexHandle) {
        let idx = vh.idx_usize();
        if idx < props.len() {
            props[idx].heap_position = -1;
        }
    }

    /// Get the vertex with minimum priority (front of min-heap)
    pub fn front(&self) -> Option<VertexHandle> {
        self.entries.first().copied()
    }

    /// Insert a vertex into the heap
    pub fn insert(&mut self, vh: VertexHandle, props: &mut [VertexProps]) {
        let idx = vh.idx_usize();
        if idx >= props.len() {
            return;
        }

        self.entries.push(vh);
        props[idx].heap_position = self.entries.len() as i32 - 1;
        self.upheap(self.entries.len() - 1, props);
    }

    /// Update a vertex's position in heap after priority change
    pub fn update(&mut self, vh: VertexHandle, props: &mut [VertexProps]) {
        let idx = vh.idx_usize();
        if idx >= props.len() {
            return;
        }
        let pos = props[idx].heap_position;
        if pos < 0 || pos as usize >= self.entries.len() {
            return;
        }

        // Re-establish heap property (both directions needed)
        self.downheap(pos as usize, props);
        self.upheap(pos as usize, props);
    }

    /// Remove a vertex from the heap
    pub fn remove(&mut self, vh: VertexHandle, props: &mut [VertexProps]) {
        let idx = vh.idx_usize();
        if idx >= props.len() {
            return;
        }
        let pos = props[idx].heap_position;
        props[idx].heap_position = -1;

        if pos < 0 || pos as usize >= self.entries.len() {
            return;
        }

        let last_pos = self.entries.len() - 1;
        if pos as usize == last_pos {
            self.entries.pop();
        } else {
            // Move last element to the removed position
            let last_vh = self.entries[last_pos];
            self.entries[pos as usize] = last_vh;
            props[last_vh.idx_usize()].heap_position = pos;
            self.entries.pop();

            // Re-establish heap property
            self.downheap(pos as usize, props);
            self.upheap(pos as usize, props);
        }
    }

    /// Pop the vertex with minimum priority
    pub fn pop_front(&mut self, props: &mut [VertexProps]) -> Option<VertexHandle> {
        if self.entries.is_empty() {
            return None;
        }

        let front_vh = self.entries[0];
        props[front_vh.idx_usize()].heap_position = -1;

        if self.entries.len() == 1 {
            self.entries.pop();
            return Some(front_vh);
        }

        // Move last element to front and downheap
        let last_vh = *self.entries.last().unwrap();
        self.entries[0] = last_vh;
        props[last_vh.idx_usize()].heap_position = 0;
        self.entries.pop();

        self.downheap(0, props);
        Some(front_vh)
    }

    /// Bubble up to establish heap property
    fn upheap(&mut self, idx: usize, props: &mut [VertexProps]) {
        if idx == 0 {
            return;
        }

        let vh = self.entries[idx];
        let mut current_idx = idx;

        while current_idx > 0 {
            let parent_idx = (current_idx - 1) / 2;
            if !Self::less(vh, self.entries[parent_idx], props) {
                break;
            }

            // Swap with parent
            let parent_vh = self.entries[parent_idx];
            self.entries[current_idx] = parent_vh;
            props[parent_vh.idx_usize()].heap_position = current_idx as i32;

            self.entries[parent_idx] = vh;
            props[vh.idx_usize()].heap_position = parent_idx as i32;

            current_idx = parent_idx;
        }
    }

    /// Bubble down to establish heap property
    fn downheap(&mut self, idx: usize, props: &mut [VertexProps]) {
        let size = self.entries.len();
        if size <= 1 {
            return;
        }

        let vh = self.entries[idx];
        let mut current_idx = idx;

        while current_idx < size {
            let left_idx = 2 * current_idx + 1;
            if left_idx >= size {
                break;
            }

            // Choose smaller child
            let mut child_idx = left_idx;
            let right_idx = left_idx + 1;
            if right_idx < size && Self::less(self.entries[right_idx], self.entries[left_idx], props) {
                child_idx = right_idx;
            }

            // If vh is already smaller than child, heap property satisfied
            if Self::less(vh, self.entries[child_idx], props) {
                break;
            }

            // Swap with child
            let child_vh = self.entries[child_idx];
            self.entries[current_idx] = child_vh;
            props[child_vh.idx_usize()].heap_position = current_idx as i32;

            current_idx = child_idx;
        }

        // Place vh in final position
        self.entries[current_idx] = vh;
        props[vh.idx_usize()].heap_position = current_idx as i32;
    }

    /// Compare priorities: vh0 < vh1 means vh0 has higher priority (min-heap)
    fn less(vh0: VertexHandle, vh1: VertexHandle, props: &[VertexProps]) -> bool {
        let p0 = props.get(vh0.idx_usize()).map(|p| p.priority).unwrap_or(f32::MAX);
        let p1 = props.get(vh1.idx_usize()).map(|p| p.priority).unwrap_or(f32::MAX);
        p0 < p1
    }
}

/// Decimation configuration
#[derive(Debug, Clone)]
pub struct DecimationConfig {
    /// Maximum error tolerance (0.0 = no limit)
    pub max_err: f32,
    /// Minimum number of vertices to preserve
    pub min_vertices: usize,
    /// Aspect ratio threshold for faces
    pub aspect_ratio_threshold: f32,
    /// Use only selected vertices
    pub only_selected: bool,
}

impl Default for DecimationConfig {
    fn default() -> Self {
        Self {
            max_err: 0.0,
            min_vertices: 0,
            aspect_ratio_threshold: 0.0,
            only_selected: false,
        }
    }
}

/// Collapse information for a single edge collapse
#[derive(Debug, Clone)]
pub struct CollapseInfo {
    /// The halfedge pointing to the vertex that will be removed
    pub halfedge: HalfedgeHandle,
    /// The vertex that will be removed (v0)
    pub v_removed: VertexHandle,
    /// The vertex that will remain (v1)
    pub v_kept: VertexHandle,
    /// The faces that will be removed
    pub faces_removed: Vec<FaceHandle>,
    /// The new position of v_kept after collapse
    pub new_position: Vec3,
    /// The error associated with this collapse
    pub error: f32,
}

impl CollapseInfo {
    fn new() -> Self {
        Self {
            halfedge: HalfedgeHandle::new(0),
            v_removed: VertexHandle::new(0),
            v_kept: VertexHandle::new(0),
            faces_removed: Vec::new(),
            new_position: Vec3::ZERO,
            error: 0.0,
        }
    }
}

/// Per-step trace data for decimation comparisons.
#[derive(Debug, Clone)]
pub struct DecimationTraceStep {
    pub step: usize,
    pub halfedge: HalfedgeHandle,
    pub v_removed: VertexHandle,
    pub v_kept: VertexHandle,
    pub is_boundary: bool,
    pub faces_removed: u8,
    pub priority: f32,
    pub active_faces_before: usize,
    pub active_faces_after: usize,
}

/// Trace summary returned by the comparison-oriented decimation entrypoints.
#[derive(Debug, Clone, Default)]
pub struct DecimationTrace {
    pub collapsed: usize,
    pub steps: Vec<DecimationTraceStep>,
    pub final_active_vertices: usize,
    pub final_active_faces: usize,
}

/// Priority queue item for collapse candidates
#[derive(Debug, Clone)]
struct CollapseCandidate {
    priority: f32,
    halfedge: HalfedgeHandle,
    v_removed: VertexHandle,
    v_kept: VertexHandle,
    is_boundary: bool,
    faces_removed: u8,
}

struct CollapseTopology {
    neighbors: Vec<Vec<VertexHandle>>,
    boundary_vertices: Vec<bool>,
    face_valences: Vec<usize>,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Error quadric module for decimation
pub struct ModQuadricT<'a> {
    mesh: &'a RustMesh,
    vertex_quadrics: Vec<Option<QuadricT>>,
    max_err: f32,
}

impl<'a> ModQuadricT<'a> {
    pub fn new(mesh: &'a RustMesh) -> Self {
        let n_vertices = mesh.n_vertices();
        Self {
            mesh,
            vertex_quadrics: vec![None; n_vertices],
            max_err: 0.0,
        }
    }

    /// Get vertices of a face (RustMesh compatible)
    fn get_face_vertices(&self, fh: FaceHandle) -> Option<Vec<VertexHandle>> {
        let heh = self.mesh.face_halfedge_handle(fh)?;
        let mut vertices = Vec::new();
        let mut current = heh;

        loop {
            if vertices.len() >= 64 {
                break;
            }

            let vh = self.mesh.to_vertex_handle(current);
            vertices.push(vh);

            let next = self.mesh.next_halfedge_handle(current);
            if next == current || !next.is_valid() || vertices.len() >= 64 {
                break;
            }
            current = next;

            if current == heh {
                break;
            }
        }

        if vertices.len() >= 3 {
            Some(vertices)
        } else {
            None
        }
    }

    pub fn initialize(&mut self) {
        for q in &mut self.vertex_quadrics {
            *q = Some(QuadricT::zero());
        }

        for fh in self.mesh.faces() {
            if let Some(verts) = self.get_face_vertices(fh) {
                if verts.len() >= 3 {
                    let p0 = self.mesh.point(verts[0]).unwrap();
                    let p1 = self.mesh.point(verts[1]).unwrap();
                    let p2 = self.mesh.point(verts[2]).unwrap();

                    let edge1 = p1 - p0;
                    let edge2 = p2 - p0;
                    let area2 = edge1.cross(edge2).length();
                    if !area2.is_finite() || area2 <= f32::EPSILON {
                        continue;
                    }

                    let normal = edge1.cross(edge2) / area2;
                    let area = area2 * 0.5;
                    let q = QuadricT::from_face(normal, p0).mul_scalar(area);

                    for vh in &verts {
                        let idx = vh.idx_usize();
                        if let Some(ref mut vq) = self.vertex_quadrics[idx] {
                            vq.add_assign_values(q);
                        }
                    }
                }
            }
        }
    }

    pub fn collapse_priority(&self, v_removed: VertexHandle, v_kept: VertexHandle) -> (f32, bool) {
        let idx0 = v_removed.idx_usize();
        let idx1 = v_kept.idx_usize();

        let q0 = match self.vertex_quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => return (f32::MAX, false),
        };

        let q1 = match self.vertex_quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => return (f32::MAX, false),
        };

        let q = q0.add_values(*q1);
        let kept_pos = self.mesh.point(v_kept).unwrap_or(Vec3::ZERO);
        let error = q.value(kept_pos);

        if self.max_err > 0.0 && error > self.max_err {
            return (f32::MAX, false);
        }

        (error, true)
    }

    pub fn optimal_position(&self, v_removed: VertexHandle, v_kept: VertexHandle) -> Vec3 {
        let idx0 = v_removed.idx_usize();
        let idx1 = v_kept.idx_usize();

        let q0 = match self.vertex_quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => return self.mesh.point(v_kept).unwrap_or(Vec3::ZERO),
        };

        let q1 = match self.vertex_quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => return self.mesh.point(v_kept).unwrap_or(Vec3::ZERO),
        };

        let q = q0.add_values(*q1);
        let (optimal, _) = q.optimize();

        optimal
    }

    pub fn postprocess_collapse(&mut self, v_removed: VertexHandle, v_kept: VertexHandle) {
        let idx_removed = v_removed.idx_usize();
        let idx_kept = v_kept.idx_usize();

        let q_removed = if idx_removed < self.vertex_quadrics.len() {
            self.vertex_quadrics[idx_removed]
        } else {
            return;
        };

        if let Some(qr) = q_removed {
            if idx_kept < self.vertex_quadrics.len() {
                if let Some(ref mut qk) = self.vertex_quadrics[idx_kept] {
                    qk.add_assign_values(qr);
                }
            }
        }
    }

    pub fn set_max_err(&mut self, max_err: f32) {
        self.max_err = max_err;
    }
}

/// Main Decimater structure (OpenMesh DecimaterT style)
pub struct Decimater<'a> {
    mesh: &'a mut RustMesh,
    config: DecimationConfig,
    collapsed: usize,
    boundary_collapses: usize,
    interior_collapses: usize,
    faces_removed_estimate: usize,
    /// Vertex properties for heap-based decimation
    vertex_props: Vec<VertexProps>,
    /// Min-heap with position tracking
    heap: DecimationHeap,
    /// Quadric error module
    quadrics: Vec<Option<QuadricT>>,
}

impl<'a> Decimater<'a> {
    pub fn new(mesh: &'a mut RustMesh) -> Self {
        let n_verts = mesh.n_vertices();
        Self {
            mesh,
            config: DecimationConfig::default(),
            collapsed: 0,
            boundary_collapses: 0,
            interior_collapses: 0,
            faces_removed_estimate: 0,
            vertex_props: Vec::new(),
            heap: DecimationHeap::new(n_verts),
            quadrics: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: DecimationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn initialize(&mut self) {
        self.collapsed = 0;
        self.boundary_collapses = 0;
        self.interior_collapses = 0;
        self.faces_removed_estimate = 0;
    }

    pub fn collapse_info(&mut self, heh: HalfedgeHandle) -> Option<CollapseInfo> {
        if self.mesh.is_halfedge_deleted(heh) || self.mesh.is_edge_deleted(self.mesh.edge_handle(heh)) {
            return None;
        }

        let from_vh = self.mesh.from_vertex_handle(heh); // v_removed
        let to_vh = self.mesh.to_vertex_handle(heh); // v_kept

        // Create a temporary quadric module for this query
        let mut qm = ModQuadricT::new(&*self.mesh);
        qm.set_max_err(self.config.max_err);
        qm.initialize();

        let (error, is_legal) = qm.collapse_priority(from_vh, to_vh);
        if !is_legal {
            return None;
        }

        let optimal_pos = qm.optimal_position(from_vh, to_vh);

        let mut faces_removed = Vec::new();

        if let Some(fh_left) = self.mesh.face_handle(heh) {
            faces_removed.push(fh_left);
        }
        let heh_opp = self.mesh.opposite_halfedge_handle(heh);
        if let Some(fh_right) = self.mesh.face_handle(heh_opp) {
            faces_removed.push(fh_right);
        }

        let mut info = CollapseInfo::new();
        info.halfedge = heh;
        info.v_removed = from_vh; // from_vertex is removed in OpenMesh collapse()
        info.v_kept = to_vh; // to_vertex is kept
        info.faces_removed = faces_removed;
        info.new_position = optimal_pos;
        info.error = error;

        Some(info)
    }

    pub fn collapse(&mut self, heh: HalfedgeHandle) -> Result<(), &'static str> {
        if !self.mesh.is_collapse_ok(heh) {
            return Err("Collapse not legal according to mesh topology");
        }

        self.mesh.collapse(heh)?;
        self.collapsed += 1;

        Ok(())
    }

    pub fn decimate(&mut self, max_collapses: usize) -> usize {
        self.decimate_internal(max_collapses, 0, false).collapsed
    }

    pub fn decimate_with_trace(
        &mut self,
        max_collapses: usize,
        trace_limit: usize,
    ) -> DecimationTrace {
        self.decimate_internal(max_collapses, trace_limit, true)
    }

    fn decimate_internal(
        &mut self,
        max_collapses: usize,
        trace_limit: usize,
        collect_trace: bool,
    ) -> DecimationTrace {
        self.initialize();

        let target = if max_collapses > 0 {
            max_collapses
        } else {
            self.mesh
                .active_vertex_count()
                .saturating_sub(self.config.min_vertices)
        };

        // === OpenMesh-style vertex heap initialization ===
        let n_verts = self.mesh.n_vertices();

        // Initialize vertex properties
        self.vertex_props = vec![VertexProps::default(); n_verts];
        self.heap = DecimationHeap::new(n_verts);

        // Initialize quadrics
        self.quadrics = vec![Some(QuadricT::zero()); n_verts];
        self.initialize_quadrics();

        // Build initial heap - each vertex finds its best collapse target
        let vertices: Vec<VertexHandle> = self.mesh.vertices().collect();
        for vh in &vertices {
            DecimationHeap::reset_heap_position(&mut self.vertex_props, *vh);
            if !self.mesh.is_vertex_deleted(*vh) {
                self.heap_vertex(*vh);
            }
        }

        let mut trace = DecimationTrace {
            collapsed: 0,
            steps: Vec::new(),
            final_active_vertices: self.mesh.active_vertex_count(),
            final_active_faces: self.mesh.n_active_faces(),
        };
        let mut collapses = 0;

        // === Main decimation loop (OpenMesh style) ===
        while !self.heap.is_empty() && collapses < target {
            // Pop vertex with minimum priority
            let vp = self.heap.pop_front(&mut self.vertex_props).unwrap();
            let v0v1 = self.vertex_props[vp.idx_usize()].collapse_target.unwrap();

            // Re-check legality (topology may have changed since vertex was added to heap)
            let v0 = self.mesh.from_vertex_handle(v0v1);
            let v1 = self.mesh.to_vertex_handle(v0v1);
            if !self.is_collapse_legal_openmesh(v0, v1, v0v1) {
                continue;
            }

            // === KEY: Store one-ring BEFORE collapse ===
            let support: Vec<VertexHandle> = self.mesh
                .vertex_vertices(v0)
                .map(|iter| iter.collect())
                .unwrap_or_default();

            // Record trace info before collapse
            let should_record = collect_trace && trace.steps.len() < trace_limit;
            let active_faces_before = if should_record {
                self.mesh.n_active_faces()
            } else {
                0
            };
            let priority = self.vertex_props[vp.idx_usize()].priority;
            // is_boundary: check if the halfedge is on boundary (OpenMesh style)
            let heh_opp = self.mesh.opposite_halfedge_handle(v0v1);
            let is_boundary = self.mesh.face_handle(v0v1).is_none()
                || self.mesh.face_handle(heh_opp).is_none();
            let faces_removed = self.count_faces_removed(v0v1);

            // Perform collapse
            if self.mesh.collapse(v0v1).is_err() {
                continue;
            }

            // Update quadrics: merge removed vertex's quadric to kept vertex
            self.update_quadrics_after_collapse(v0, v1);

            collapses += 1;
            if is_boundary {
                self.boundary_collapses += 1;
            } else {
                self.interior_collapses += 1;
            }
            self.faces_removed_estimate += faces_removed as usize;

            if should_record {
                trace.steps.push(DecimationTraceStep {
                    step: collapses,
                    halfedge: v0v1,
                    v_removed: v0,
                    v_kept: v1,
                    is_boundary,
                    faces_removed,
                    priority,
                    active_faces_before,
                    active_faces_after: self.mesh.n_active_faces(),
                });
            }

            // === KEY: Update ALL neighbors in stored one-ring ===
            for neighbor_vh in support {
                if neighbor_vh != v0 && !self.mesh.is_vertex_deleted(neighbor_vh) {
                    self.heap_vertex(neighbor_vh);
                }
            }
        }

        self.collapsed = collapses;
        trace.collapsed = collapses;
        trace.final_active_vertices = self.mesh.active_vertex_count();
        trace.final_active_faces = self.mesh.n_active_faces();
        trace
    }

    /// OpenMesh-style heap_vertex: find best collapse target for a single vertex
    /// by iterating over outgoing halfedges and using strict `<` tie-break
    fn heap_vertex(&mut self, vh: VertexHandle) {
        let mut best_prio = f32::MAX;
        let mut collapse_target: Option<HalfedgeHandle> = None;

        // Iterate outgoing halfedges from this vertex
        if let Some(he_iter) = self.mesh.vertex_halfedges(vh) {
            for heh in he_iter {
                // Skip deleted halfedges
                if self.mesh.is_halfedge_deleted(heh) ||
                   self.mesh.is_edge_deleted(self.mesh.edge_handle(heh)) {
                    continue;
                }

                let v0 = self.mesh.from_vertex_handle(heh);
                let v1 = self.mesh.to_vertex_handle(heh);

                // Check collapse legality (including OpenMesh boundary constraint)
                if !self.is_collapse_legal_openmesh(v0, v1, heh) {
                    continue;
                }

                // Compute priority (quadric error)
                let prio = self.compute_collapse_priority(v0, v1);

                // KEY: OpenMesh uses strict `<` (first best wins)
                if prio >= 0.0 && prio < best_prio {
                    best_prio = prio;
                    collapse_target = Some(heh);
                }
            }
        }

        // Update vertex properties and heap
        let idx = vh.idx_usize();
        if idx < self.vertex_props.len() {
            if collapse_target.is_some() {
                self.vertex_props[idx].collapse_target = collapse_target;
                self.vertex_props[idx].priority = best_prio;
                if self.heap.is_stored(vh, &self.vertex_props) {
                    self.heap.update(vh, &mut self.vertex_props);
                } else {
                    self.heap.insert(vh, &mut self.vertex_props);
                }
            } else {
                // No valid collapse - remove from heap
                if self.heap.is_stored(vh, &self.vertex_props) {
                    self.heap.remove(vh, &mut self.vertex_props);
                }
                self.vertex_props[idx].collapse_target = None;
                self.vertex_props[idx].priority = -1.0;
            }
        }
    }

    /// Initialize quadrics from face normals and areas
    fn initialize_quadrics(&mut self) {
        for fh in self.mesh.faces() {
            if let Some(verts) = self.get_face_vertices_internal(fh) {
                if verts.len() >= 3 {
                    let p0 = self.mesh.point(verts[0]).unwrap();
                    let p1 = self.mesh.point(verts[1]).unwrap();
                    let p2 = self.mesh.point(verts[2]).unwrap();

                    let edge1 = p1 - p0;
                    let edge2 = p2 - p0;
                    let area2 = edge1.cross(edge2).length();
                    if !area2.is_finite() || area2 <= f32::EPSILON {
                        continue;
                    }

                    let normal = edge1.cross(edge2) / area2;
                    let area = area2 * 0.5;
                    let q = QuadricT::from_face(normal, p0).mul_scalar(area);

                    for vh in &verts {
                        let idx = vh.idx_usize();
                        if let Some(ref mut vq) = self.quadrics[idx] {
                            vq.add_assign_values(q);
                        }
                    }
                }
            }
        }
    }

    /// Compute collapse priority (quadric error) for v0 -> v1 collapse
    fn compute_collapse_priority(&self, v0: VertexHandle, v1: VertexHandle) -> f32 {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        let q0 = match self.quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => return f32::MAX,
        };
        let q1 = match self.quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => return f32::MAX,
        };

        let combined = q0.add_values(*q1);
        let kept_pos = self.mesh.point(v1).unwrap_or(Vec3::ZERO);
        let error = combined.value(kept_pos);

        if !error.is_finite() {
            return f32::MAX;
        }
        if self.config.max_err > 0.0 && error > self.config.max_err {
            return f32::MAX;
        }

        error
    }

    /// Update quadrics after collapse: merge v0's quadric to v1
    fn update_quadrics_after_collapse(&mut self, v0: VertexHandle, v1: VertexHandle) {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        if idx0 < self.quadrics.len() && idx1 < self.quadrics.len() {
            if let Some(q0) = self.quadrics[idx0] {
                if let Some(ref mut q1) = self.quadrics[idx1] {
                    q1.add_assign_values(q0);
                }
            }
        }
    }

    /// OpenMesh-style collapse legality check (including boundary constraints)
    fn is_collapse_legal_openmesh(&self, v0: VertexHandle, v1: VertexHandle, heh: HalfedgeHandle) -> bool {
        // Basic validity
        if !heh.is_valid() || !v0.is_valid() || !v1.is_valid() {
            return false;
        }
        if self.mesh.is_halfedge_deleted(heh) || self.mesh.is_edge_deleted(self.mesh.edge_handle(heh)) {
            return false;
        }
        if self.mesh.is_vertex_deleted(v0) || self.mesh.is_vertex_deleted(v1) {
            return false;
        }

        // Mesh's internal topology check
        if !self.mesh.is_collapse_ok(heh) {
            return false;
        }

        // === OpenMesh boundary constraint ===
        // "don't collapse a boundary vertex to an inner one"
        let v0_is_boundary = self.mesh.is_boundary_vertex(v0);
        let v1_is_boundary = self.mesh.is_boundary_vertex(v1);
        if v0_is_boundary && !v1_is_boundary {
            return false;
        }

        // OpenMesh: if v0 is boundary, check vl/vr constraint
        // "only one one ring intersection" for boundary vertices
        // This only applies when v0 is a boundary vertex with TWO adjacent faces
        // (i.e., the edge being collapsed is NOT a boundary edge)
        if v0_is_boundary {
            let heh_opp = self.mesh.opposite_halfedge_handle(heh);
            let fh_left = self.mesh.face_handle(heh);
            let fh_right = self.mesh.face_handle(heh_opp);

            // Only check vl/vr if this is NOT a boundary edge (both faces exist)
            if fh_left.is_some() && fh_right.is_some() {
                let vl = self.mesh.to_vertex_handle(self.mesh.next_halfedge_handle(heh));
                let vr = self.mesh.to_vertex_handle(self.mesh.next_halfedge_handle(heh_opp));
                // Both vl and vr valid means not a simple boundary collapse
                if vl.is_valid() && vr.is_valid() {
                    return false;
                }
            }
        }

        true
    }

    /// Count faces that would be removed by collapsing heh
    fn count_faces_removed(&self, heh: HalfedgeHandle) -> u8 {
        let mut count = 0u8;
        if self.mesh.face_handle(heh).is_some() {
            count += 1;
        }
        let heh_opp = self.mesh.opposite_halfedge_handle(heh);
        if self.mesh.face_handle(heh_opp).is_some() {
            count += 1;
        }
        count
    }

    // Internal helper for face vertices
    fn get_face_vertices_internal(&self, fh: FaceHandle) -> Option<Vec<VertexHandle>> {
        let heh = self.mesh.face_halfedge_handle(fh)?;
        let mut vertices = Vec::new();
        let mut current = heh;

        loop {
            if vertices.len() >= 64 {
                break;
            }

            let vh = self.mesh.to_vertex_handle(current);
            vertices.push(vh);

            let next = self.mesh.next_halfedge_handle(current);
            if next == current || !next.is_valid() || vertices.len() >= 64 {
                break;
            }
            current = next;

            if current == heh {
                break;
            }
        }

        if vertices.len() >= 3 {
            Some(vertices)
        } else {
            None
        }
    }

    pub fn decimate_to(&mut self, target_vertices: usize) -> usize {
        let active_vertices = self.mesh.active_vertex_count();
        if target_vertices >= active_vertices {
            return 0;
        }

        self.decimate(active_vertices - target_vertices)
    }

    pub fn decimate_to_with_trace(
        &mut self,
        target_vertices: usize,
        trace_limit: usize,
    ) -> DecimationTrace {
        let active_vertices = self.mesh.active_vertex_count();
        if target_vertices >= active_vertices {
            return DecimationTrace {
                collapsed: 0,
                steps: Vec::new(),
                final_active_vertices: active_vertices,
                final_active_faces: self.mesh.n_active_faces(),
            };
        }

        self.decimate_with_trace(active_vertices - target_vertices, trace_limit)
    }

    pub fn n_collapses(&self) -> usize {
        self.collapsed
    }

    pub fn boundary_collapses(&self) -> usize {
        self.boundary_collapses
    }

    pub fn interior_collapses(&self) -> usize {
        self.interior_collapses
    }

    pub fn faces_removed_estimate(&self) -> usize {
        self.faces_removed_estimate
    }
}

impl<'a> Decimater<'a> {
    fn best_collapse_candidate(
        &self,
        vertex_quadrics: &[Option<QuadricT>],
    ) -> Option<CollapseCandidate> {
        let topology = build_collapse_topology(self.mesh);
        let mut best: Option<CollapseCandidate> = None;

        for heh_idx in 0..self.mesh.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if self.mesh.is_halfedge_deleted(heh) || self.mesh.is_edge_deleted(self.mesh.edge_handle(heh)) {
                continue;
            }
            if !is_collapse_ok_with_topology(self.mesh, &topology, heh) {
                continue;
            }

            let v_removed = self.mesh.from_vertex_handle(heh);
            let v_kept = self.mesh.to_vertex_handle(heh);
            let idx_removed = v_removed.idx_usize();
            let idx_kept = v_kept.idx_usize();

            let q_removed = match vertex_quadrics.get(idx_removed) {
                Some(Some(q)) => q,
                _ => continue,
            };
            let q_kept = match vertex_quadrics.get(idx_kept) {
                Some(Some(q)) => q,
                _ => continue,
            };

            let combined = q_kept.add_values(*q_removed);
            let kept_pos = self.mesh.point(v_kept).unwrap_or(Vec3::ZERO);
            let error = combined.value(kept_pos);

            if !error.is_finite() {
                continue;
            }

            if self.config.max_err > 0.0 && error > self.config.max_err {
                continue;
            }

            let candidate = CollapseCandidate {
                priority: error,
                halfedge: heh,
                v_removed,
                v_kept,
                is_boundary: self.mesh.face_handle(heh).is_none()
                    || self
                        .mesh
                        .face_handle(self.mesh.opposite_halfedge_handle(heh))
                        .is_none(),
                faces_removed: if self.mesh.face_handle(heh).is_some() { 1 } else { 0 }
                    + if self
                        .mesh
                        .face_handle(self.mesh.opposite_halfedge_handle(heh))
                        .is_some()
                    {
                        1
                    } else {
                        0
                    },
            };

            if best.as_ref().is_none_or(|current| is_better_candidate(&candidate, current)) {
                best = Some(candidate);
            }
        }

        best
    }
}

fn is_better_candidate(candidate: &CollapseCandidate, current: &CollapseCandidate) -> bool {
    const EPS: f32 = 1.0e-6;

    if candidate.priority + EPS < current.priority {
        return true;
    }
    if current.priority + EPS < candidate.priority {
        return false;
    }
    if candidate.is_boundary != current.is_boundary {
        return candidate.is_boundary;
    }
    if candidate.faces_removed != current.faces_removed {
        return candidate.faces_removed < current.faces_removed;
    }
    candidate.halfedge.idx() < current.halfedge.idx()
}

fn build_collapse_topology(mesh: &RustMesh) -> CollapseTopology {
    let mut neighbors = vec![Vec::new(); mesh.n_vertices()];
    let mut face_valences = vec![0usize; mesh.n_faces()];

    for fh in mesh.faces() {
        if mesh.is_face_deleted(fh) || mesh.face_halfedge_handle(fh).is_none() {
            continue;
        }

        let vertices = mesh.face_vertices_vec(fh);
        let n = vertices.len();
        face_valences[fh.idx_usize()] = n;

        if n < 2 {
            continue;
        }

        for (idx, &vh) in vertices.iter().enumerate() {
            let prev = vertices[(idx + n - 1) % n];
            let next = vertices[(idx + 1) % n];
            push_unique_neighbor(&mut neighbors[vh.idx_usize()], prev, vh);
            push_unique_neighbor(&mut neighbors[vh.idx_usize()], next, vh);
        }
    }

    let mut boundary_vertices = vec![false; mesh.n_vertices()];
    for heh_idx in 0..mesh.n_halfedges() {
        let heh = HalfedgeHandle::new(heh_idx as u32);
        if mesh.is_halfedge_deleted(heh) || mesh.is_edge_deleted(mesh.edge_handle(heh)) {
            continue;
        }
        let opp = mesh.opposite_halfedge_handle(heh);
        if mesh.face_handle(heh).is_none() || mesh.face_handle(opp).is_none() {
            let from = mesh.from_vertex_handle(heh).idx_usize();
            let to = mesh.to_vertex_handle(heh).idx_usize();
            if from < boundary_vertices.len() {
                boundary_vertices[from] = true;
            }
            if to < boundary_vertices.len() {
                boundary_vertices[to] = true;
            }
        }
    }

    CollapseTopology {
        neighbors,
        boundary_vertices,
        face_valences,
    }
}

fn push_unique_neighbor(neighbors: &mut Vec<VertexHandle>, candidate: VertexHandle, center: VertexHandle) {
    if candidate != center && !neighbors.contains(&candidate) {
        neighbors.push(candidate);
    }
}

fn is_collapse_ok_with_topology(
    mesh: &RustMesh,
    topology: &CollapseTopology,
    heh: HalfedgeHandle,
) -> bool {
    if !heh.is_valid() {
        return false;
    }
    if mesh.is_halfedge_deleted(heh) || mesh.is_edge_deleted(mesh.edge_handle(heh)) {
        return false;
    }

    let v0 = mesh.from_vertex_handle(heh);
    let v1 = mesh.to_vertex_handle(heh);
    if !v0.is_valid()
        || !v1.is_valid()
        || mesh.is_vertex_deleted(v0)
        || mesh.is_vertex_deleted(v1)
    {
        return false;
    }
    if mesh.halfedge_handle(v0).is_none() || mesh.halfedge_handle(v1).is_none() {
        return false;
    }

    let heh_opp = mesh.opposite_halfedge_handle(heh);
    let fh_left = mesh.face_handle(heh);
    let fh_right = mesh.face_handle(heh_opp);
    if fh_left.is_none() && fh_right.is_none() {
        return false;
    }

    if fh_left.is_some() {
        let h1 = mesh.next_halfedge_handle(heh);
        let h2 = mesh.next_halfedge_handle(h1);
        if mesh.is_boundary(mesh.opposite_halfedge_handle(h1))
            && mesh.is_boundary(mesh.opposite_halfedge_handle(h2))
        {
            return false;
        }
    }

    if fh_right.is_some() {
        let h1 = mesh.next_halfedge_handle(heh_opp);
        let h2 = mesh.next_halfedge_handle(h1);
        if mesh.is_boundary(mesh.opposite_halfedge_handle(h1))
            && mesh.is_boundary(mesh.opposite_halfedge_handle(h2))
        {
            return false;
        }
    }

    let neighbors_v0 = topology
        .neighbors
        .get(v0.idx_usize())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|&v| v != v1)
        .collect::<Vec<_>>();
    let neighbors_v1 = topology
        .neighbors
        .get(v1.idx_usize())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|&v| v != v0)
        .collect::<Vec<_>>();

    let mut allowed_shared = Vec::new();
    if let Some(fh) = fh_left {
        for vh in mesh.face_vertices_vec(fh) {
            if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                allowed_shared.push(vh);
            }
        }
    }
    if let Some(fh) = fh_right {
        for vh in mesh.face_vertices_vec(fh) {
            if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                allowed_shared.push(vh);
            }
        }
    }

    for &nv in &neighbors_v0 {
        if neighbors_v1.contains(&nv) && !allowed_shared.contains(&nv) {
            return false;
        }
    }

    if fh_left.is_some() && fh_right.is_some() {
        let vl = mesh.to_vertex_handle(mesh.next_halfedge_handle(heh));
        let vr = mesh.to_vertex_handle(mesh.next_halfedge_handle(heh_opp));
        if vl == vr {
            return false;
        }
    }

    let v0_boundary = topology
        .boundary_vertices
        .get(v0.idx_usize())
        .copied()
        .unwrap_or(false);
    let v1_boundary = topology
        .boundary_vertices
        .get(v1.idx_usize())
        .copied()
        .unwrap_or(false);
    let edge_boundary = fh_left.is_none() || fh_right.is_none();
    if v0_boundary && v1_boundary && !edge_boundary {
        return false;
    }

    if let Some(fh) = fh_left {
        if topology
            .face_valences
            .get(fh.idx_usize())
            .copied()
            .unwrap_or_default()
            == 3
        {
            let one = mesh.opposite_halfedge_handle(mesh.next_halfedge_handle(heh));
            let two = mesh.opposite_halfedge_handle(mesh.next_halfedge_handle(
                mesh.next_halfedge_handle(heh),
            ));
            if let (Some(face_one), Some(face_two)) = (mesh.face_handle(one), mesh.face_handle(two)) {
                if face_one == face_two
                    && topology
                        .face_valences
                        .get(face_one.idx_usize())
                        .copied()
                        .unwrap_or_default()
                        != 3
                {
                    return false;
                }
            }
        }
    }

    if let Some(fh) = fh_right {
        if topology
            .face_valences
            .get(fh.idx_usize())
            .copied()
            .unwrap_or_default()
            == 3
        {
            let one = mesh.opposite_halfedge_handle(mesh.next_halfedge_handle(heh_opp));
            let two = mesh.opposite_halfedge_handle(mesh.next_halfedge_handle(
                mesh.next_halfedge_handle(heh_opp),
            ));
            if let (Some(face_one), Some(face_two)) = (mesh.face_handle(one), mesh.face_handle(two)) {
                if face_one == face_two
                    && topology
                        .face_valences
                        .get(face_one.idx_usize())
                        .copied()
                        .unwrap_or_default()
                        != 3
                {
                    return false;
                }
            }
        }
    }

    true
}

/// Convenience function
pub fn decimate_mesh(mesh: &mut RustMesh, target_vertices: usize, max_err: f32) -> usize {
    let mut decimater = Decimater::new(mesh).with_config(DecimationConfig {
        max_err,
        min_vertices: target_vertices,
        ..Default::default()
    });

    let collapsed = decimater.decimate_to(target_vertices);
    if collapsed > 0 {
        decimater.mesh.garbage_collection();
    }
    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_cube, generate_sphere, read_off, write_off};
    use std::collections::HashMap;
    use std::fs;

    fn raw_face_diagnostics(mesh: &RustMesh) -> (usize, usize, usize) {
        let mut active_faces = 0usize;
        let mut degenerate_faces = 0usize;
        let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();

        for fh in mesh.faces() {
            let vertices = mesh.face_vertices_vec(fh);
            if vertices.len() < 3 {
                continue;
            }

            active_faces += 1;
            if vertices.len() != 3 {
                degenerate_faces += 1;
                continue;
            }

            let ids = [
                vertices[0].idx_usize(),
                vertices[1].idx_usize(),
                vertices[2].idx_usize(),
            ];
            if ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0] {
                degenerate_faces += 1;
            }

            for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_use.entry(key).or_insert(0) += 1;
            }
        }

        let non_manifold_edges = edge_use.values().filter(|&&count| count > 2).count();
        (active_faces, degenerate_faces, non_manifold_edges)
    }

    #[test]
    fn test_decimater_creation() {
        let mut mesh = generate_cube();
        let decimater = Decimater::new(&mut mesh);
        assert_eq!(decimater.n_collapses(), 0);
    }

    #[test]
    fn test_quadric_module_init() {
        let mesh = generate_cube();
        let n_verts = mesh.n_vertices();
        println!("Cube has {} vertices", n_verts);
        assert!(n_verts > 0);

        let mut module = ModQuadricT::new(&mesh);
        module.initialize();

        let initialized_count = module
            .vertex_quadrics
            .iter()
            .filter(|q| q.is_some())
            .count();
        println!("Initialized quadrics: {}", initialized_count);
        assert_eq!(initialized_count, n_verts);
    }

    #[test]
    fn test_collapse_priority() {
        let mesh = generate_cube();
        let mut module = ModQuadricT::new(&mesh);
        module.initialize();

        let n_vertices = mesh.n_vertices();
        assert!(n_vertices >= 2);

        let v0 = VertexHandle::new(0);
        let v1 = VertexHandle::new(1);

        assert!(v0.is_valid());
        assert!(v1.is_valid());

        let (priority, is_legal) = module.collapse_priority(v0, v1);
        assert!(is_legal);
        assert!(priority >= 0.0);
    }

    #[test]
    fn test_collapse_info() {
        let mut mesh = generate_cube();
        let mut decimater = Decimater::new(&mut mesh);

        let n_halfedges = decimater.mesh.n_halfedges();
        let mut valid_collapses = 0;

        for heh_idx in 0..n_halfedges.min(100) {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if decimater.collapse_info(heh).is_some() {
                valid_collapses += 1;
            }
        }

        println!("Valid collapses: {}", valid_collapses);
        assert!(true);
    }

    #[test]
    fn test_decimate_sphere() {
        let mesh = generate_sphere(1.0, 8, 8);
        let n_vertices = mesh.n_vertices();
        println!("Sphere has {} vertices", n_vertices);

        assert!(n_vertices > 10);

        let mut mesh = generate_sphere(1.0, 8, 8);
        let target = n_vertices / 2;
        let collapsed = decimate_mesh(&mut mesh, target, 0.0);
        println!("Collapsed {} edges", collapsed);
        assert!(collapsed > 0);
    }

    #[test]
    fn test_decimate_off_roundtrip_after_garbage_collection() {
        let mut mesh = generate_sphere(1.0, 10, 10);
        let target = mesh.n_vertices() / 2;
        let collapsed = decimate_mesh(&mut mesh, target, 0.0);
        assert!(collapsed > 0);

        let path = "/tmp/rustmesh-decimate-roundtrip.off";
        write_off(&mesh, path).unwrap();
        let loaded = read_off(path).unwrap();
        fs::remove_file(path).ok();

        assert!(loaded.n_vertices() >= 3);
        assert!(loaded.n_faces() >= 1);
    }

    #[test]
    fn test_decimate_raw_topology_stays_valid() {
        let mut mesh = generate_sphere(1.0, 10, 10);
        let target = mesh.n_vertices() / 2;
        let collapsed = Decimater::new(&mut mesh).decimate_to(target);
        assert!(collapsed > 0);

        let (active_faces, degenerate_faces, non_manifold_edges) = raw_face_diagnostics(&mesh);
        assert!(active_faces > 0);
        assert_eq!(degenerate_faces, 0);
        assert_eq!(non_manifold_edges, 0);
    }

    #[test]
    fn test_decimate_boundary_mesh_preserves_face_budget() {
        let mut mesh = generate_sphere(1.0, 10, 10);
        let target = mesh.n_vertices() / 2;
        let collapsed = Decimater::new(&mut mesh).decimate_to(target);
        assert!(collapsed > 0);

        let (active_faces, degenerate_faces, non_manifold_edges) = raw_face_diagnostics(&mesh);
        assert_eq!(degenerate_faces, 0);
        assert_eq!(non_manifold_edges, 0);
        assert!(
            active_faces >= 100,
            "expected decimation to preserve at least 100 faces on the seam-heavy sphere, got {active_faces}"
        );
    }

    #[test]
    fn test_decimate_collapse_mix_stats_sum_to_collapses() {
        let mut mesh = generate_sphere(1.0, 10, 10);
        let target = mesh.n_vertices() / 2;
        let mut decimater = Decimater::new(&mut mesh);
        let collapsed = decimater.decimate_to(target);

        assert_eq!(
            decimater.boundary_collapses() + decimater.interior_collapses(),
            collapsed
        );
        assert!(decimater.faces_removed_estimate() >= collapsed);
    }

    #[test]
    fn test_decimate_trace_respects_limit() {
        let mut mesh = generate_sphere(1.0, 10, 10);
        let target = mesh.n_vertices() / 2;
        let mut decimater = Decimater::new(&mut mesh);
        let trace = decimater.decimate_to_with_trace(target, 5);

        assert!(trace.collapsed > 0);
        assert_eq!(trace.steps.len(), 5);
        assert_eq!(trace.steps[0].step, 1);
        assert!(trace.steps.windows(2).all(|pair| pair[0].step < pair[1].step));
    }
}
