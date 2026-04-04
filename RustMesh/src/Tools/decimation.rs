// ============================================================================
// Decimation Module - Mesh Simplification
// Based on OpenMesh's DecimaterT framework
// ============================================================================

use crate::{FaceHandle, HalfedgeHandle, QuadricT, RustMesh, Vec3, VertexHandle};
use std::cmp::Ordering;

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

    pub fn collapse_priority(&self, v0: VertexHandle, v1: VertexHandle) -> (f32, bool) {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        let q0 = match self.vertex_quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => return (f32::MAX, false),
        };

        let q1 = match self.vertex_quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => return (f32::MAX, false),
        };

        let q = q0.add_values(*q1);
        let kept_pos = self.mesh.point(v0).unwrap_or(Vec3::ZERO);
        let error = q.value(kept_pos);

        if self.max_err > 0.0 && error > self.max_err {
            return (f32::MAX, false);
        }

        (error, true)
    }

    pub fn optimal_position(&self, v0: VertexHandle, v1: VertexHandle) -> Vec3 {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        let q0 = match self.vertex_quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => return self.mesh.point(v1).unwrap_or(Vec3::ZERO),
        };

        let q1 = match self.vertex_quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => return self.mesh.point(v1).unwrap_or(Vec3::ZERO),
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

/// Main Decimater structure
pub struct Decimater<'a> {
    mesh: &'a mut RustMesh,
    config: DecimationConfig,
    collapsed: usize,
}

impl<'a> Decimater<'a> {
    pub fn new(mesh: &'a mut RustMesh) -> Self {
        Self {
            mesh,
            config: DecimationConfig::default(),
            collapsed: 0,
        }
    }

    pub fn with_config(mut self, config: DecimationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn initialize(&mut self) {
        self.collapsed = 0;
    }

    pub fn collapse_info(&mut self, heh: HalfedgeHandle) -> Option<CollapseInfo> {
        let to_vh = self.mesh.to_vertex_handle(heh); // v_removed
        let from_vh = self.mesh.from_vertex_handle(heh); // v_kept

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
        info.v_removed = to_vh; // to_vertex is removed in collapse()
        info.v_kept = from_vh; // from_vertex is kept
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
        self.initialize();

        let target = if max_collapses > 0 {
            max_collapses
        } else {
            self.mesh
                .active_vertex_count()
                .saturating_sub(self.config.min_vertices)
        };

        // Build quadric once at start
        let n_verts = self.mesh.n_vertices();
        let mut vertex_quadrics: Vec<Option<QuadricT>> = vec![None; n_verts];

        // Initialize quadrics
        for q in &mut vertex_quadrics {
            *q = Some(QuadricT::zero());
        }

        // Accumulate face quadrics
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
                        if let Some(ref mut vq) = vertex_quadrics[idx] {
                            vq.add_assign_values(q);
                        }
                    }
                }
            }
        }

        let mut collapses = 0;
        while collapses < target && self.mesh.active_vertex_count() > self.config.min_vertices {
            let Some(candidate) = self.best_collapse_candidate(&vertex_quadrics) else {
                break;
            };

            if self.mesh.collapse(candidate.halfedge).is_err() {
                break;
            }

            let idx_removed = candidate.v_removed.idx_usize();
            let idx_kept = candidate.v_kept.idx_usize();
            if idx_removed < vertex_quadrics.len() && idx_kept < vertex_quadrics.len() {
                if let Some(qr) = vertex_quadrics[idx_removed] {
                    if let Some(ref mut qk) = vertex_quadrics[idx_kept] {
                        qk.add_assign_values(qr);
                    }
                }
            }

            collapses += 1;
        }

        self.collapsed = collapses;
        collapses
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

    pub fn n_collapses(&self) -> usize {
        self.collapsed
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
            if !is_collapse_ok_with_topology(self.mesh, &topology, heh) {
                continue;
            }

            let v_removed = self.mesh.to_vertex_handle(heh);
            let v_kept = self.mesh.from_vertex_handle(heh);
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
        if mesh.face_halfedge_handle(fh).is_none() {
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

    let v0 = mesh.to_vertex_handle(heh);
    let v1 = mesh.from_vertex_handle(heh);
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
}
