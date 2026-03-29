//! Depth estimation module
//!
//! This module provides depth estimation capabilities:
//! - Stereo matching (for stereo cameras like KITTI)
//! - Depth fusion (combining multiple depth sources)

pub mod fusion;
pub mod stereo;

#[cfg(test)]
mod additional_tests;

pub use fusion::{DepthFusion, DepthFusionConfig, DepthObservation, TemporalDepthFusion};
pub use stereo::{BlockMatcher, StereoConfig, StereoMatcher};
