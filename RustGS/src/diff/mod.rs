//! Differentiable rendering module.
//!
//! This module provides GPU-accelerated differentiable rendering
//! for 3D Gaussian Splatting training.

pub mod analytical_backward;

#[cfg(feature = "gpu")]
pub mod diff_renderer;

#[cfg(feature = "gpu")]
pub mod diff_splat;

// Always export analytical backward (no GPU dependency)
pub use analytical_backward::{
    GaussianRenderRecord, ForwardIntermediate, AnalyticalGradients, backward,
};

#[cfg(feature = "gpu")]
pub use diff_renderer::{
    GaussianTensors, CameraTensors, RenderLoss, DiffGaussianRenderer,
};

#[cfg(feature = "gpu")]
pub use diff_splat::{
    DiffSplatRenderer, TrainableGaussians, DiffCamera, DiffRenderOutput, DiffLoss,
};