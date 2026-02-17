//! RustSLAM - A pure Rust Visual SLAM library
//!
//! ## Quick Start
//!
//! ```rust
//! use rustslam::{SE3, Frame};
//! ```

// Re-export core types
pub use core::{Frame, FrameFeatures, KeyFrame, MapPoint, Map, Camera, SE3};
pub use features::{FeatureExtractor, FeatureMatcher, KeyPoint, Descriptors, Match, OrbExtractor, KnnMatcher};
pub use tracker::{VisualOdometry, VOState, VOResult};

// Modules
pub mod core;
pub mod features;
pub mod tracker;
pub mod mapping;
pub mod optimizer;
pub mod loop_closing;
pub mod fusion;
pub mod io;
pub mod viewer;
pub mod pipeline;
pub mod depth;
pub mod config;
pub mod test_utils;
pub mod cli;
