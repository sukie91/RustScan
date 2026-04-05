//! # RustMesh - SIMD-Accelerated Mesh Library
//!
//! A Rust port of OpenMesh with **SIMD acceleration**.
//!
//! ## Quick Start
//!
//! ```rust
//! use rustmesh::{RustMesh, Vec3};
//!
//! let mut mesh = RustMesh::new();
//! let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
//! let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
//! let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
//! mesh.add_face(&[v0, v1, v2]);
//! ```

// Re-export types
pub use attrib_soa_kernel::AttribSoAKernel;
pub use circulators::*;
pub use connectivity::RustMesh;
pub use analysis::{
    analyze_mesh, compute_all_curvatures, compute_edge_length_stats, compute_mesh_quality,
    compute_surface_area, compute_vertex_curvature, compute_volume, export_curvature_field,
    CurvatureType, EdgeLengthStats, MeshAnalysis, MeshQuality, VertexCurvature,
};
pub use decimation::{
    decimate_mesh, CollapseInfo, Decimater, DecimationConfig, DecimationTrace,
    DecimationTraceStep, ModQuadricT,
};
pub use decimation_modules::{
    CollapseInfo as ModuleCollapseInfo, CollapseResult, CombinedModules, DecimationModule,
    ModAspectRatio, ModBoundary, ModNormal, ModQuadric,
};
pub use dualizer::{
    dual_mesh, dual_mesh_with_boundary, dualize, dualize_with_boundary, is_dualizable,
    BoundaryDualStrategy, DualError, DualResult,
};
pub use geometry::*;
pub use handles::{BaseHandle, EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
pub use hole_filling::*;
pub use io::*; // IO module now implemented
pub use items::{Edge, Face, Halfedge, Vertex};
pub use mesh_repair::*;
pub use quadric::QuadricT;
pub use smart_ranges::{FaceRange, SmartMesh, VertexRange};
pub use smoother::{laplace_smooth, tangential_smooth, SmoothResult, SmootherConfig};
pub use soa_kernel::SoAKernel;
pub use status::{StatusFlags, StatusSet};
pub use subdivision::{
    catmull_clark_subdivide, catmull_clark_subdivide_iterations, loop_subdivide,
    loop_subdivide_iterations, sqrt3_subdivide, sqrt3_subdivide_iterations, split_edge,
    validate_for_catmull_clark, validate_for_subdivision,
    SubdivisionError, SubdivisionStats,
};
pub use test_data::*;
pub use vdpm::*;

pub use glam::Vec3;

// ============================================================================
// Module Structure (following OpenMesh: Core + Tools + Utils)
// ============================================================================

// Core module (对应 OpenMesh/Core)
pub mod core {
    pub mod attrib_soa_kernel;
    pub mod connectivity;
    pub mod geometry;
    pub mod handles;
    pub mod io;
    pub mod items;
    pub mod soa_kernel;
}

// Re-export for convenience
mod handles {
    pub use crate::core::handles::*;
}
mod items {
    pub use crate::core::items::*;
}
mod soa_kernel {
    pub use crate::core::soa_kernel::*;
}
mod attrib_soa_kernel {
    pub use crate::core::attrib_soa_kernel::*;
}
mod connectivity {
    pub use crate::core::connectivity::*;
}
mod geometry {
    pub use crate::core::geometry::*;
}
pub mod io {
    pub use crate::core::io::*;
}

// Utils modules (对应 OpenMesh/Core/Utils)
pub mod utils {
    pub mod circulators;
    pub mod performance;
    pub mod quadric;
    pub mod smart_ranges;
    pub mod status;
    pub mod test_data;
}

// Re-export Utils
mod status {
    pub use crate::utils::status::*;
}
mod circulators {
    pub use crate::utils::circulators::*;
}
mod quadric {
    pub use crate::utils::quadric::*;
}
mod smart_ranges {
    pub use crate::utils::smart_ranges::*;
}
mod test_data {
    pub use crate::utils::test_data::*;
}

// Tools modules (对应 OpenMesh/Tools)
pub mod tools {
    pub mod analysis;
    pub mod decimation;
    pub mod decimation_modules;
    pub mod dualizer;
    pub mod hole_filling;
    pub mod mesh_repair;
    pub mod remeshing;
    pub mod smoother;
    pub mod subdivision;
    pub mod vdpm;
}

// Re-export Tools
mod analysis {
    pub use crate::tools::analysis::*;
}
mod decimation {
    pub use crate::tools::decimation::*;
}
mod decimation_modules {
    pub use crate::tools::decimation_modules::*;
}
mod remeshing {
    pub use crate::tools::remeshing::*;
}
mod smoother {
    pub use crate::tools::smoother::*;
}
mod subdivision {
    pub use crate::tools::subdivision::*;
}
mod hole_filling {
    pub use crate::tools::hole_filling::*;
}
mod mesh_repair {
    pub use crate::tools::mesh_repair::*;
}
mod dualizer {
    pub use crate::tools::dualizer::*;
}
mod vdpm {
    pub use crate::tools::vdpm::*;
}
