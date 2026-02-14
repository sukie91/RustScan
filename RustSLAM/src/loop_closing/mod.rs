//! Loop closing module
//!
//! Provides loop detection using Bag of Words vocabulary.

pub mod vocabulary;
pub mod detector;
pub mod database;
pub mod relocalization;

pub use vocabulary::{Vocabulary, Word, hamming_distance, kmeans};
pub use detector::{LoopDetector, LoopCandidate, LoopDetectionResult};
pub use database::{KeyFrameDatabase, KeyFrameEntry};
pub use relocalization::{Relocalizer, RelocalizationResult};
