//! Rendering module.
//!
//! Contains Gaussian rendering implementations:
//! - `renderer`: Basic forward renderer with depth rendering
//! - `tiled_renderer`: Tiled rasterization for efficient rendering

mod renderer;
pub mod tiled_renderer;

pub use renderer::{GaussianRenderer, RenderOutput};
pub use tiled_renderer::{
    densify, prune, Gaussian, ProjectedGaussian, RenderBuffer, TiledRenderer,
};
