//! Pipeline module for multi-threaded SLAM processing

#[cfg(feature = "slam-pipeline")]
pub mod realtime;
pub mod checkpoint;

#[cfg(test)]
mod additional_tests;
