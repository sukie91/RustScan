//! Pipeline module for multi-threaded SLAM processing

#[cfg(feature = "slam-pipeline")]
pub mod realtime;
pub mod checkpoint;

#[cfg(all(test, feature = "slam-pipeline"))]
mod additional_tests;
