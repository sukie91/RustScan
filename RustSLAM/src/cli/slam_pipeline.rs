//! SLAM processing pipeline helpers for the RustScan CLI.

use crate::config::SlamConfig;
use crate::core::{Frame, FrameFeatures};
use crate::features::{
    Descriptors, FeatureExtractor, HarrisDetector, HarrisParams, Match, OrbExtractor, FastDetector,
    FastParams, KnnMatcher,
};
use crate::features::base::FeatureError;

const PATCH_DESCRIPTOR_SIZE: usize = 32;
const PATCH_OFFSETS: [(i32, i32); PATCH_DESCRIPTOR_SIZE] = [
    (-6, -6), (-4, -6), (-2, -6), (0, -6), (2, -6), (4, -6), (6, -6), (8, -6),
    (-6, -2), (-4, -2), (-2, -2), (0, -2), (2, -2), (4, -2), (6, -2), (8, -2),
    (-6, 2), (-4, 2), (-2, 2), (0, 2), (2, 2), (4, 2), (6, 2), (8, 2),
    (-6, 6), (-4, 6), (-2, 6), (0, 6), (2, 6), (4, 6), (6, 6), (8, 6),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    Orb,
    Harris,
    Fast,
}

impl FeatureType {
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("RUSTSCAN_FEATURE_TYPE").ok()?;
        match raw.to_ascii_lowercase().as_str() {
            "orb" => Some(Self::Orb),
            "harris" => Some(Self::Harris),
            "fast" => Some(Self::Fast),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlamPipelineConfig {
    pub feature_type: FeatureType,
    pub max_features: usize,
}

impl SlamPipelineConfig {
    pub fn from_slam_config(config: &SlamConfig) -> Self {
        let feature_type = FeatureType::from_env().unwrap_or(FeatureType::Orb);
        Self {
            feature_type,
            max_features: config.tracker.max_features,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub frame: Frame,
    pub keypoints: Vec<[f32; 2]>,
    pub descriptors: Descriptors,
}

impl FeatureFrame {
    pub fn to_frame_features(&self) -> FrameFeatures {
        FrameFeatures {
            keypoints: self.keypoints.clone(),
            descriptors: self.descriptors.data.clone(),
            map_points: vec![None; self.keypoints.len()],
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SlamPipelineError {
    #[error("feature extraction failed: {0}")]
    FeatureExtraction(#[from] FeatureError),
    #[error("feature matching failed: {0}")]
    FeatureMatching(FeatureError),
}

pub struct FeatureExtractorEngine {
    feature_type: FeatureType,
    orb: OrbExtractor,
    harris: HarrisDetector,
    fast: FastDetector,
    max_features: usize,
}

impl FeatureExtractorEngine {
    pub fn new(config: &SlamPipelineConfig) -> Self {
        Self {
            feature_type: config.feature_type,
            orb: OrbExtractor::new(config.max_features),
            harris: HarrisDetector::new(HarrisParams::default()),
            fast: FastDetector::new(FastParams::default()),
            max_features: config.max_features,
        }
    }

    pub fn extract(
        &mut self,
        frame: Frame,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<FeatureFrame, SlamPipelineError> {
        let gray = rgb_to_grayscale(rgb, width, height);
        match self.feature_type {
            FeatureType::Orb => {
                let (keypoints, descriptors) = self.orb.detect_and_compute(&gray, width, height)?;
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x(), kp.y()]).collect(),
                    descriptors,
                })
            }
            FeatureType::Harris => {
                let mut keypoints = self.harris.detect(&gray, width, height);
                keypoints.truncate(self.max_features);
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x, kp.y]).collect(),
                    descriptors: build_patch_descriptors(&gray, width, height, &keypoints),
                })
            }
            FeatureType::Fast => {
                let mut keypoints = self.fast.detect(&gray, width, height);
                keypoints.truncate(self.max_features);
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x, kp.y]).collect(),
                    descriptors: build_patch_descriptors(&gray, width, height, &keypoints),
                })
            }
        }
    }
}

pub struct FeatureMatcherEngine {
    matcher: KnnMatcher,
    ratio_threshold: f64,
}

impl FeatureMatcherEngine {
    pub fn new(ratio_threshold: f64) -> Self {
        Self {
            matcher: KnnMatcher::new(2),
            ratio_threshold,
        }
    }

    pub fn match_frames(
        &mut self,
        current: &FeatureFrame,
        previous: &FeatureFrame,
    ) -> Result<Vec<Match>, SlamPipelineError> {
        if current.descriptors.is_empty() || previous.descriptors.is_empty() {
            return Ok(Vec::new());
        }

        let train = descriptors_to_arrays(&previous.descriptors);
        let query = descriptors_to_arrays(&current.descriptors);
        self.matcher.build_tree(&train);

        let raw_matches = self.matcher.match_batch_with_ratio(&query, self.ratio_threshold);
        let mut matches = Vec::with_capacity(raw_matches.len());
        for (query_idx, candidates) in raw_matches {
            for (distance, train_idx) in candidates {
                matches.push(Match {
                    query_idx: query_idx as u32,
                    train_idx: train_idx as u32,
                    distance: distance as f32,
                });
            }
        }

        if matches.is_empty() {
            return Ok(Vec::new());
        }

        matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        Ok(matches)
    }
}

fn rgb_to_grayscale(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb.len() < expected {
        return Vec::new();
    }
    let mut gray = Vec::with_capacity((width as usize) * (height as usize));
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }
    gray
}

fn build_patch_descriptors(
    gray: &[u8],
    width: u32,
    height: u32,
    keypoints: &[crate::features::pure_rust::KeyPoint],
) -> Descriptors {
    let mut descriptors = Descriptors::with_capacity(keypoints.len(), PATCH_DESCRIPTOR_SIZE);
    if keypoints.is_empty() || gray.is_empty() {
        return descriptors;
    }
    let w = width as i32;
    let h = height as i32;
    let mut data = Vec::with_capacity(keypoints.len() * PATCH_DESCRIPTOR_SIZE);
    for kp in keypoints {
        let x = kp.x.round() as i32;
        let y = kp.y.round() as i32;
        for (dx, dy) in PATCH_OFFSETS {
            let px = x + dx;
            let py = y + dy;
            let intensity = if px >= 0 && px < w && py >= 0 && py < h {
                let idx = (py as usize) * (w as usize) + (px as usize);
                gray.get(idx).copied().unwrap_or(0)
            } else {
                0
            };
            data.push(intensity);
        }
    }
    descriptors.data = data;
    descriptors.size = PATCH_DESCRIPTOR_SIZE;
    descriptors.count = keypoints.len();
    descriptors
}

fn descriptors_to_arrays(descriptors: &Descriptors) -> Vec<[f64; PATCH_DESCRIPTOR_SIZE]> {
    descriptors
        .data
        .chunks(PATCH_DESCRIPTOR_SIZE)
        .map(|chunk| {
            let mut arr = [0.0; PATCH_DESCRIPTOR_SIZE];
            for (i, &byte) in chunk.iter().enumerate() {
                arr[i] = byte as f64;
            }
            arr
        })
        .collect()
}
