//! Pipeline module for multi-threaded SLAM processing

pub mod checkpoint;
#[cfg(feature = "slam-pipeline")]
pub mod realtime;

#[cfg(all(test, feature = "slam-pipeline"))]
mod additional_tests;
