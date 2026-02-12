//! # RustMesh - SIMD-Accelerated Mesh Library
//!
//! A Rust port of OpenMesh with **SIMD acceleration**.
//!
//! ## Quick Start
//!
//! ```rust
//! use rustmesh::{FastMesh, Vec3};
//!
//! let mut mesh = FastMesh::new();
//! let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
//! let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
//! let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
//! mesh.add_face(&[v0, v1, v2]);
//!
//! let sum = unsafe { mesh.vertex_sum_simd() };
//! ```

// Re-export types
pub use handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle, BaseHandle};
pub use items::{Vertex, Halfedge, Edge, Face};
pub use soa_kernel::SoAKernel;
pub use connectivity::PolyMeshSoA as FastMesh;
pub use tri_connectivity::TriMesh;
pub use test_data::*;
pub use geometry::*;
pub use io::*;
pub use status::{StatusFlags, StatusSet};
pub use circulators::*;
pub use glam::Vec3;

// Core modules only
mod handles;
mod items;
mod kernel;
mod soa_kernel;
mod connectivity;
mod tri_connectivity;
mod test_data;
mod geometry;
mod io;
mod status;
mod circulators;
