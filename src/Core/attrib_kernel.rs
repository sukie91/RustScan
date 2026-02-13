//! # AttribKernel
//!
//! Attribute kernel for mesh elements.
//! Provides properties for normals, colors, texture coordinates, and more.

use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::kernel::ArrayKernel;

/// Property handle types for different attribute types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VNormalPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VColorPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VTexCoordPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HNormalPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HColorPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HTexCoordPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EColorPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FNormalPropHandle(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FColorPropHandle(usize);

/// Vertex attributes
#[derive(Debug, Clone)]
pub struct VertexAttributes {
    pub normal: Option<Vec<glam::Vec3>>,
    pub color: Option<Vec<glam::Vec4>>,
    pub texcoord: Option<Vec<glam::Vec2>>,
}

impl Default for VertexAttributes {
    fn default() -> Self {
        Self {
            normal: None,
            color: None,
            texcoord: None,
        }
    }
}

/// Halfedge attributes
#[derive(Debug, Clone)]
pub struct HalfedgeAttributes {
    pub prev_halfedge: Option<Vec<HalfedgeHandle>>,
    pub normal: Option<Vec<glam::Vec3>>,
    pub color: Option<Vec<glam::Vec4>>,
    pub texcoord: Option<Vec<glam::Vec2>>,
}

impl Default for HalfedgeAttributes {
    fn default() -> Self {
        Self {
            prev_halfedge: None,
            normal: None,
            color: None,
            texcoord: None,
        }
    }
}

/// Edge attributes
#[derive(Debug, Clone)]
pub struct EdgeAttributes {
    pub color: Option<Vec<glam::Vec4>>,
}

impl Default for EdgeAttributes {
    fn default() -> Self {
        Self {
            color: None,
        }
    }
}

/// Face attributes
#[derive(Debug, Clone)]
pub struct FaceAttributes {
    pub normal: Option<Vec<glam::Vec3>>,
    pub color: Option<Vec<glam::Vec4>>,
}

impl Default for FaceAttributes {
    fn default() -> Self {
        Self {
            normal: None,
            color: None,
        }
    }
}

/// AttribKernel - Attribute management for mesh elements
#[derive(Debug, Clone)]
pub struct AttribKernel {
    kernel: ArrayKernel,

    // Vertex attributes
    v_attribs: VertexAttributes,

    // Halfedge attributes
    he_attribs: HalfedgeAttributes,

    // Edge attributes
    e_attribs: EdgeAttributes,

    // Face attributes
    f_attribs: FaceAttributes,
}

impl AttribKernel {
    /// Create a new attribute kernel
    pub fn new() -> Self {
        Self {
            kernel: ArrayKernel::new(),
            v_attribs: VertexAttributes::default(),
            he_attribs: HalfedgeAttributes::default(),
            e_attribs: EdgeAttributes::default(),
            f_attribs: FaceAttributes::default(),
        }
    }

    /// Clear the kernel
    pub fn clear(&mut self) {
        self.kernel.clear();
        self.v_attribs = VertexAttributes::default();
        self.he_attribs = HalfedgeAttributes::default();
        self.e_attribs = EdgeAttributes::default();
        self.f_attribs = FaceAttributes::default();
    }

    // --- Vertex attributes ---

    /// Request vertex normals
    pub fn request_vertex_normals(&mut self) {
        if self.v_attribs.normal.is_none() {
            let size = self.kernel.n_vertices();
            self.v_attribs.normal = Some(vec![glam::Vec3::ZERO; size]);
        }
    }

    /// Check if vertex normals are available
    pub fn has_vertex_normals(&self) -> bool {
        self.v_attribs.normal.is_some()
    }

    /// Get vertex normal
    pub fn normal(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.v_attribs.normal.as_ref()
            .and_then(|normals| normals.get(vh.idx() as usize).copied())
    }

    /// Set vertex normal
    pub fn set_normal(&mut self, vh: VertexHandle, n: glam::Vec3) {
        if let Some(ref mut normals) = self.v_attribs.normal {
            if let Some(normal) = normals.get_mut(vh.idx() as usize) {
                *normal = n;
            }
        }
    }

    /// Request vertex colors
    pub fn request_vertex_colors(&mut self) {
        if self.v_attribs.color.is_none() {
            let size = self.kernel.n_vertices();
            self.v_attribs.color = Some(vec![glam::Vec4::new(1.0, 1.0, 1.0, 1.0); size]);
        }
    }

    /// Check if vertex colors are available
    pub fn has_vertex_colors(&self) -> bool {
        self.v_attribs.color.is_some()
    }

    /// Get vertex color
    pub fn color(&self, vh: VertexHandle) -> Option<glam::Vec4> {
        self.v_attribs.color.as_ref()
            .and_then(|colors| colors.get(vh.idx() as usize).copied())
    }

    /// Set vertex color
    pub fn set_color(&mut self, vh: VertexHandle, c: glam::Vec4) {
        if let Some(ref mut colors) = self.v_attribs.color {
            if let Some(color) = colors.get_mut(vh.idx() as usize) {
                *color = c;
            }
        }
    }

    /// Request vertex texture coordinates
    pub fn request_vertex_texcoords(&mut self) {
        if self.v_attribs.texcoord.is_none() {
            let size = self.kernel.n_vertices();
            self.v_attribs.texcoord = Some(vec![glam::Vec2::ZERO; size]);
        }
    }

    /// Check if vertex texcoords are available
    pub fn has_vertex_texcoords(&self) -> bool {
        self.v_attribs.texcoord.is_some()
    }

    /// Get vertex texcoord
    pub fn texcoord(&self, vh: VertexHandle) -> Option<glam::Vec2> {
        self.v_attribs.texcoord.as_ref()
            .and_then(|texcoords| texcoords.get(vh.idx() as usize).copied())
    }

    /// Set vertex texcoord
    pub fn set_texcoord(&mut self, vh: VertexHandle, t: glam::Vec2) {
        if let Some(ref mut texcoords) = self.v_attribs.texcoord {
            if let Some(texcoord) = texcoords.get_mut(vh.idx() as usize) {
                *texcoord = t;
            }
        }
    }

    // --- Halfedge attributes ---

    /// Request halfedge texture coordinates
    pub fn request_halfedge_texcoords(&mut self) {
        if self.he_attribs.texcoord.is_none() {
            let size = self.kernel.n_halfedges();
            self.he_attribs.texcoord = Some(vec![glam::Vec2::ZERO; size]);
        }
    }

    /// Check if halfedge texcoords are available
    pub fn has_halfedge_texcoords(&self) -> bool {
        self.he_attribs.texcoord.is_some()
    }

    /// Get halfedge texcoord
    pub fn h_texcoord(&self, heh: HalfedgeHandle) -> Option<glam::Vec2> {
        self.he_attribs.texcoord.as_ref()
            .and_then(|texcoords| texcoords.get(heh.idx() as usize).copied())
    }

    /// Set halfedge texcoord
    pub fn set_h_texcoord(&mut self, heh: HalfedgeHandle, t: glam::Vec2) {
        if let Some(ref mut texcoords) = self.he_attribs.texcoord {
            if let Some(texcoord) = texcoords.get_mut(heh.idx() as usize) {
                *texcoord = t;
            }
        }
    }

    // --- Halfedge Normals ---

    /// Request halfedge normals
    pub fn request_halfedge_normals(&mut self) {
        if self.he_attribs.normal.is_none() {
            let size = self.kernel.n_halfedges();
            self.he_attribs.normal = Some(vec![glam::Vec3::ZERO; size]);
        }
    }

    /// Check if halfedge normals are available
    pub fn has_halfedge_normals(&self) -> bool {
        self.he_attribs.normal.is_some()
    }

    /// Get halfedge normal
    pub fn h_normal(&self, heh: HalfedgeHandle) -> Option<glam::Vec3> {
        self.he_attribs.normal.as_ref()
            .and_then(|normals| normals.get(heh.idx() as usize).copied())
    }

    /// Set halfedge normal
    pub fn set_h_normal(&mut self, heh: HalfedgeHandle, n: glam::Vec3) {
        if let Some(ref mut normals) = self.he_attribs.normal {
            if let Some(normal) = normals.get_mut(heh.idx() as usize) {
                *normal = n;
            }
        }
    }

    // --- Halfedge Colors ---

    /// Request halfedge colors
    pub fn request_halfedge_colors(&mut self) {
        if self.he_attribs.color.is_none() {
            let size = self.kernel.n_halfedges();
            self.he_attribs.color = Some(vec![glam::Vec4::new(0.5, 0.5, 0.5, 1.0); size]);
        }
    }

    /// Check if halfedge colors are available
    pub fn has_halfedge_colors(&self) -> bool {
        self.he_attribs.color.is_some()
    }

    /// Get halfedge color
    pub fn h_color(&self, heh: HalfedgeHandle) -> Option<glam::Vec4> {
        self.he_attribs.color.as_ref()
            .and_then(|colors| colors.get(heh.idx() as usize).copied())
    }

    /// Set halfedge color
    pub fn set_h_color(&mut self, heh: HalfedgeHandle, c: glam::Vec4) {
        if let Some(ref mut colors) = self.he_attribs.color {
            if let Some(color) = colors.get_mut(heh.idx() as usize) {
                *color = c;
            }
        }
    }

    // --- Edge attributes ---

    /// Request edge colors
    pub fn request_edge_colors(&mut self) {
        if self.e_attribs.color.is_none() {
            let size = self.kernel.n_edges();
            self.e_attribs.color = Some(vec![glam::Vec4::new(0.5, 0.5, 0.5, 1.0); size]);
        }
    }

    /// Check if edge colors are available
    pub fn has_edge_colors(&self) -> bool {
        self.e_attribs.color.is_some()
    }

    /// Get edge color
    pub fn e_color(&self, eh: EdgeHandle) -> Option<glam::Vec4> {
        self.e_attribs.color.as_ref()
            .and_then(|colors| colors.get(eh.idx() as usize).copied())
    }

    /// Set edge color
    pub fn set_e_color(&mut self, eh: EdgeHandle, c: glam::Vec4) {
        if let Some(ref mut colors) = self.e_attribs.color {
            if let Some(color) = colors.get_mut(eh.idx() as usize) {
                *color = c;
            }
        }
    }

    // --- Face attributes ---

    /// Request face normals
    pub fn request_face_normals(&mut self) {
        if self.f_attribs.normal.is_none() {
            let size = self.kernel.n_faces();
            self.f_attribs.normal = Some(vec![glam::Vec3::ZERO; size]);
        }
    }

    /// Check if face normals are available
    pub fn has_face_normals(&self) -> bool {
        self.f_attribs.normal.is_some()
    }

    /// Get face normal
    pub fn f_normal(&self, fh: FaceHandle) -> Option<glam::Vec3> {
        self.f_attribs.normal.as_ref()
            .and_then(|normals| normals.get(fh.idx() as usize).copied())
    }

    /// Set face normal
    pub fn set_f_normal(&mut self, fh: FaceHandle, n: glam::Vec3) {
        if let Some(ref mut normals) = self.f_attribs.normal {
            if let Some(normal) = normals.get_mut(fh.idx() as usize) {
                *normal = n;
            }
        }
    }

    /// Request face colors
    pub fn request_face_colors(&mut self) {
        if self.f_attribs.color.is_none() {
            let size = self.kernel.n_faces();
            self.f_attribs.color = Some(vec![glam::Vec4::new(0.8, 0.8, 0.8, 1.0); size]);
        }
    }

    /// Check if face colors are available
    pub fn has_face_colors(&self) -> bool {
        self.f_attribs.color.is_some()
    }

    /// Get face color
    pub fn f_color(&self, fh: FaceHandle) -> Option<glam::Vec4> {
        self.f_attribs.color.as_ref()
            .and_then(|colors| colors.get(fh.idx() as usize).copied())
    }

    /// Set face color
    pub fn set_f_color(&mut self, fh: FaceHandle, c: glam::Vec4) {
        if let Some(ref mut colors) = self.f_attribs.color {
            if let Some(color) = colors.get_mut(fh.idx() as usize) {
                *color = c;
            }
        }
    }

    // --- Delegate to kernel ---

    /// Add a vertex
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        let vh = self.kernel.add_vertex(point);

        // Resize attribute arrays if needed
        if let Some(ref mut normals) = self.v_attribs.normal {
            normals.push(glam::Vec3::ZERO);
        }
        if let Some(ref mut colors) = self.v_attribs.color {
            colors.push(glam::Vec4::new(1.0, 1.0, 1.0, 1.0));
        }
        if let Some(ref mut texcoords) = self.v_attribs.texcoord {
            texcoords.push(glam::Vec2::ZERO);
        }

        vh
    }

    /// Add an edge
    pub fn add_edge(&mut self, start_vh: VertexHandle, end_vh: VertexHandle) -> HalfedgeHandle {
        let heh = self.kernel.add_edge(start_vh, end_vh);

        // Resize halfedge attribute arrays if needed
        if let Some(ref mut normals) = self.he_attribs.normal {
            normals.push(glam::Vec3::ZERO);
        }
        if let Some(ref mut colors) = self.he_attribs.color {
            colors.push(glam::Vec4::new(0.5, 0.5, 0.5, 1.0));
        }
        if let Some(ref mut texcoords) = self.he_attribs.texcoord {
            texcoords.push(glam::Vec2::ZERO);
        }

        heh
    }

    /// Get vertex point
    pub fn point(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.kernel.vertex(vh).map(|v| v.point)
    }

    /// Set vertex point
    pub fn set_point(&mut self, vh: VertexHandle, p: glam::Vec3) {
        if let Some(v) = self.kernel.vertex_mut(vh) {
            v.point = p;
        }
    }

    /// Get number of vertices
    pub fn n_vertices(&self) -> usize {
        self.kernel.n_vertices()
    }

    /// Get number of edges
    pub fn n_edges(&self) -> usize {
        self.kernel.n_edges()
    }

    /// Get number of faces
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get iterator over vertices
    pub fn vertices(&self) -> crate::connectivity::VertexIter<'_> {
        crate::connectivity::VertexIter::new(&self.kernel)
    }

    /// Get iterator over edges
    pub fn edges(&self) -> crate::connectivity::EdgeIter<'_> {
        crate::connectivity::EdgeIter::new(&self.kernel)
    }

    /// Get iterator over faces
    pub fn faces(&self) -> crate::connectivity::FaceIter<'_> {
        crate::connectivity::FaceIter::new(&self.kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_normals() {
        let mut kernel = AttribKernel::new();

        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = kernel.add_vertex(glam::vec3(0.0, 1.0, 0.0));

        kernel.request_vertex_normals();
        assert!(kernel.has_vertex_normals());

        kernel.set_normal(v0, glam::vec3(0.0, 0.0, 1.0));
        assert_eq!(kernel.normal(v0), Some(glam::vec3(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_vertex_colors() {
        let mut kernel = AttribKernel::new();

        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));

        kernel.request_vertex_colors();
        assert!(kernel.has_vertex_colors());

        kernel.set_color(v0, glam::vec4(1.0, 0.0, 0.0, 1.0));
        assert_eq!(kernel.color(v0), Some(glam::vec4(1.0, 0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_face_normals() {
        let mut kernel = AttribKernel::new();

        kernel.request_face_normals();
        assert!(kernel.has_face_normals());

        // Note: Can't test actual normals without a face
        assert!(kernel.has_face_normals());
    }

    #[test]
    fn test_halfedge_normals() {
        let mut kernel = AttribKernel::new();

        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let _he = kernel.add_edge(v0, v1);

        kernel.request_halfedge_normals();
        assert!(kernel.has_halfedge_normals());

        let heh = HalfedgeHandle::new(0);
        kernel.set_h_normal(heh, glam::vec3(0.0, 1.0, 0.0));
        assert_eq!(kernel.h_normal(heh), Some(glam::vec3(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_halfedge_colors() {
        let mut kernel = AttribKernel::new();

        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let _he = kernel.add_edge(v0, v1);

        kernel.request_halfedge_colors();
        assert!(kernel.has_halfedge_colors());

        let heh = HalfedgeHandle::new(0);
        kernel.set_h_color(heh, glam::vec4(0.0, 1.0, 0.0, 1.0));
        assert_eq!(kernel.h_color(heh), Some(glam::vec4(0.0, 1.0, 0.0, 1.0)));
    }

    #[test]
    fn test_attribute_availability() {
        let mut kernel = AttribKernel::new();

        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let _he = kernel.add_edge(v0, v1);

        // Initially no attributes
        assert!(!kernel.has_vertex_normals());
        assert!(!kernel.has_vertex_colors());
        assert!(!kernel.has_vertex_texcoords());
        assert!(!kernel.has_halfedge_normals());
        assert!(!kernel.has_halfedge_colors());
        assert!(!kernel.has_halfedge_texcoords());
        assert!(!kernel.has_face_normals());
        assert!(!kernel.has_face_colors());
        assert!(!kernel.has_edge_colors());

        // Request attributes
        kernel.request_vertex_normals();
        kernel.request_vertex_colors();
        kernel.request_halfedge_normals();
        kernel.request_halfedge_colors();
        kernel.request_face_normals();

        assert!(kernel.has_vertex_normals());
        assert!(kernel.has_vertex_colors());
        assert!(kernel.has_halfedge_normals());
        assert!(kernel.has_halfedge_colors());
        assert!(kernel.has_face_normals());
    }
}
