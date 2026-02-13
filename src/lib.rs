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
pub use handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle, BaseHandle};
pub use items::{Vertex, Halfedge, Edge, Face};
pub use soa_kernel::SoAKernel;
pub use connectivity::PolyMeshSoA as RustMesh;
pub use tri_connectivity::TriMesh;
pub use test_data::*;
pub use geometry::*;
pub use io::*;
pub use status::{StatusFlags, StatusSet};
pub use circulators::*;
pub use quadric::QuadricT;
pub use decimation::{Decimater, decimate_mesh, DecimationConfig, CollapseInfo, ModQuadricT};
pub use smart_ranges::{SmartMesh, VertexRange, FaceRange};
pub use smoother::{SmootherConfig, SmoothResult, laplace_smooth, tangential_smooth};
pub use subdivision::{
    loop_subdivide, loop_subdivide_iterations, split_edge, 
    validate_for_subdivision, catmull_clark_subdivide, 
    catmull_clark_subdivide_iterations, validate_for_catmull_clark,
    SubdivisionStats, SubdivisionError
};
pub use hole_filling::*;
pub use mesh_repair::*;
pub use dualizer::*;
pub use vdpm::*;

pub use glam::Vec3;

// ============================================================================
// Module Structure (following OpenMesh: Core + Tools + Utils)
// ============================================================================

// Core module (对应 OpenMesh/Core)
pub mod core {
    pub mod handles;
    pub mod items;
    pub mod kernel;
    pub mod soa_kernel;
    pub mod connectivity;
    pub mod tri_connectivity;
    pub mod attrib_kernel;
    pub mod geometry;
    pub mod io;
}

// Re-export for convenience
mod handles { pub use crate::core::handles::*; }
mod items { pub use crate::core::items::*; }
mod kernel { pub use crate::core::kernel::*; }
mod soa_kernel { pub use crate::core::soa_kernel::*; }
mod connectivity { pub use crate::core::connectivity::*; }
mod tri_connectivity { pub use crate::core::tri_connectivity::*; }
mod attrib_kernel { pub use crate::core::attrib_kernel::*; }
mod geometry { pub use crate::core::geometry::*; }
mod io { pub use crate::core::io::*; }

// Utils modules (对应 OpenMesh/Core/Utils)
pub mod utils {
    pub mod status;
    pub mod circulators;
    pub mod quadric;
    pub mod smart_ranges;
    pub mod test_data;
    pub mod performance;
}

// Re-export Utils
mod status { pub use crate::utils::status::*; }
mod circulators { pub use crate::utils::circulators::*; }
mod quadric { pub use crate::utils::quadric::*; }
mod smart_ranges { pub use crate::utils::smart_ranges::*; }
mod test_data { pub use crate::utils::test_data::*; }

// Tools modules (对应 OpenMesh/Tools)
pub mod tools {
    pub mod decimation;
    pub mod smoother;
    pub mod subdivision;
    pub mod hole_filling;
    pub mod mesh_repair;
    pub mod dualizer;
    pub mod vdpm;
}

// Re-export Tools
mod decimation { pub use crate::tools::decimation::*; }
mod smoother { pub use crate::tools::smoother::*; }
mod subdivision { pub use crate::tools::subdivision::*; }
mod hole_filling { pub use crate::tools::hole_filling::*; }
mod mesh_repair { pub use crate::tools::mesh_repair::*; }
mod dualizer { pub use crate::tools::dualizer::*; }
mod vdpm { pub use crate::tools::vdpm::*; }
