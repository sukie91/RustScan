//! Rendering module.
//!
//! Contains Gaussian rendering implementations:
//! - `renderer`: Basic forward renderer with depth rendering
//! - `tiled_renderer`: Tiled rasterization for efficient rendering

pub mod tiled_renderer;
mod renderer;

pub use renderer::{GaussianRenderer, RenderOutput};
pub use tiled_renderer::{
    Gaussian, ProjectedGaussian, TiledRenderer, RenderBuffer,
    densify, prune,
};