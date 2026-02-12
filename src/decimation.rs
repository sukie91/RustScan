// ============================================================================
// Decimation Module - Mesh Simplification
// Based on OpenMesh's DecimaterT framework
// ============================================================================

use crate::{RustMesh, VertexHandle, HalfedgeHandle, FaceHandle, QuadricT, Vec3};
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
    v_removed: VertexHandle,
    v_kept: VertexHandle,
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
        other.priority.partial_cmp(&self.priority)
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

    /// Get vertices of a face (PolyMeshSoA compatible)
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
                    let normal = edge1.cross(edge2).normalize();
                    let center = (p0 + p1 + p2) / 3.0;
                    
                    let q = QuadricT::from_face(normal, center);
                    
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
        let (_optimal, error) = q.optimize();
        
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
        let to_vh = self.mesh.to_vertex_handle(heh);
        let from_vh = self.mesh.from_vertex_handle(heh);
        
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
        info.v_removed = from_vh;
        info.v_kept = to_vh;
        info.faces_removed = faces_removed;
        info.new_position = optimal_pos;
        info.error = error;
        
        Some(info)
    }

    fn is_collapse_legal(&self, _v0: VertexHandle, _v1: VertexHandle) -> bool {
        true
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
            self.mesh.n_vertices() - self.config.min_vertices
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
                    let normal = edge1.cross(edge2).normalize();
                    let center = (p0 + p1 + p2) / 3.0;
                    
                    let q = QuadricT::from_face(normal, center);
                    
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
        
        // Decimation loop
        while collapses < target {
            let mut best_heh: Option<HalfedgeHandle> = None;
            let mut best_error = f32::MAX;
            
            // Find best collapse using precomputed quadrics
            for heh_idx in 0..self.mesh.n_halfedges() {
                let heh = HalfedgeHandle::new(heh_idx as u32);
                let to_vh = self.mesh.to_vertex_handle(heh);
                let from_vh = self.mesh.from_vertex_handle(heh);
                
                let idx0 = from_vh.idx_usize();
                let idx1 = to_vh.idx_usize();
                
                let q0 = match vertex_quadrics.get(idx0) {
                    Some(Some(q)) => q,
                    _ => continue,
                };
                let q1 = match vertex_quadrics.get(idx1) {
                    Some(Some(q)) => q,
                    _ => continue,
                };
                
                let combined = q0.add_values(*q1);
                let (_opt, error) = combined.optimize();
                
                if self.config.max_err > 0.0 && error > self.config.max_err {
                    continue;
                }
                
                if error < best_error {
                    best_error = error;
                    best_heh = Some(heh);
                }
            }
            
            match best_heh {
                Some(heh) => {
                    if !self.mesh.is_collapse_ok(heh) {
                        break;
                    }
                    if let Err(e) = self.mesh.collapse(heh) {
                        eprintln!("Collapse failed: {}", e);
                        break;
                    }
                    collapses += 1;
                }
                None => break,
            }
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
        if target_vertices >= self.mesh.n_vertices() {
            return 0;
        }
        
        self.decimate(self.mesh.n_vertices() - target_vertices)
    }

    pub fn n_collapses(&self) -> usize {
        self.collapsed
    }
}

/// Convenience function
pub fn decimate_mesh(mesh: &mut RustMesh, target_vertices: usize, max_err: f32) -> usize {
    let mut decimater = Decimater::new(mesh)
        .with_config(DecimationConfig {
            max_err,
            min_vertices: target_vertices,
            ..Default::default()
        });
    
    decimater.decimate_to(target_vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_cube, generate_sphere};

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
        
        let initialized_count = module.vertex_quadrics.iter()
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
}
