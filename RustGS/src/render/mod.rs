//! Rendering module.
//!
//! Contains Gaussian rendering implementations:
//! - `renderer`: Basic forward renderer with depth rendering
//! - `tiled_renderer`: Tiled rasterization for efficient rendering

mod renderer;
mod tiled_renderer;

pub use renderer::{GaussianRenderer, RenderOutput};
pub use tiled_renderer::{ProjectedGaussian, RenderBuffer, TiledRenderer};
