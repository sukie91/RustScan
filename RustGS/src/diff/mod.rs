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
    backward, AnalyticalGradients, ForwardIntermediate, GaussianRenderRecord,
};

#[cfg(feature = "gpu")]
pub use diff_renderer::{CameraTensors, DiffGaussianRenderer, GaussianTensors, RenderLoss};

#[cfg(feature = "gpu")]
pub use diff_splat::{
    DiffCamera, DiffLoss, DiffRenderOutput, DiffSplatRenderer, TrainableGaussians,
};
