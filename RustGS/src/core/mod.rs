//! Core Gaussian data structures.

mod gaussian;
mod camera;

pub use gaussian::{Gaussian3D, GaussianMap, GaussianState};
pub use camera::GaussianCamera;