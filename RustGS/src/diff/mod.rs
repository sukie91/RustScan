//! Differentiable rendering module.
//!
//! This module will be populated from RustSLAM/src/fusion/diff_*.rs in Story 9-5.
//!
//! Currently empty - will be implemented when migrating from RustSLAM.

#[cfg(feature = "gpu")]
mod diff_splat;

#[cfg(feature = "gpu")]
pub use diff_splat::DiffSplatRenderer;