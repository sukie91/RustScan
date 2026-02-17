//! IO module for data loading and serialization
//!
//! This module provides dataset loaders for standard SLAM benchmarks
//! and utilities for reading/writing SLAM data.

mod dataset;
pub mod video_decoder;
#[cfg(feature = "opencv")]
mod video_loader;

pub use dataset::{
    Dataset, DatasetConfig, DatasetError, DatasetIterator, DatasetMetadata, Frame, Result,
    TumRgbdDataset, KittiDataset, EurocDataset,
};

#[cfg(feature = "opencv")]
pub use video_loader::{VideoLoader, VideoError};
