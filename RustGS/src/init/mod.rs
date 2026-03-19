//! Gaussian initialization module.
//!
//! Provides utilities to create initial Gaussians from point clouds.

pub mod initialization;

pub use initialization::{
    GaussianInitConfig,
    initialize_gaussians_from_points,
    initialize_gaussian3d_from_points,
};

#[cfg(feature = "gpu")]
pub use initialization::initialize_trainable_gaussians_from_points;
