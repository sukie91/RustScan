//! Gaussian initialization module.
//!
//! Provides utilities to create initial Gaussians from point clouds.

pub mod initialization;

pub use initialization::{
    initialize_gaussian3d_from_points, initialize_gaussians_from_points, GaussianInitConfig,
};

#[cfg(feature = "gpu")]
pub use initialization::initialize_trainable_gaussians_from_points;
