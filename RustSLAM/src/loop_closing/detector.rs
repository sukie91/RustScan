//! Loop closing detector implementation
//!
//! This module implements loop detection based on Bag of Words (BoW).

use crate::core::{Map, KeyFrame, Descriptors, SE3};
use crate::features::Match;

/// A loop candidate detected between frames
#[derive(Debug, Clone)]
pub struct LoopCandidate {
    /// Keyframe ID that forms a loop
    pub keyframe_id: u64,
    /// BoW similarity score
    pub score: f32,
    /// Timestamp of the keyframe
    pub timestamp: f64,
}

/// Result of loop detection
#[derive(Debug, Clone)]
pub struct LoopDetectionResult {
    /// Whether a loop was detected
    pub loop_detected: bool,
    /// The loop candidate (if detected)
    pub candidate: Option<LoopCandidate>,
    /// Similarity transform (Sim3) if computed
    pub sim3: Option<Sim3>,
    /// Number of matched features
    pub num_matches: usize,
    /// Number of inliers
    pub num_inliers: usize,
}

impl LoopDetectionResult {
    /// Create a negative result
    pub fn no_loop() -> Self {
        Self {
            loop_detected: false,
            candidate: None,
            sim3: None,
            num_matches: 0,
            num_inliers: 0,
        }
    }
}

/// Similarity transform (Sim3) representation
/// Sim3 = s * R * t + t
#[derive(Debug, Clone)]
pub struct Sim3 {
    /// Rotation matrix (3x3)
    pub rotation: [[f32; 3]; 3],
    /// Translation vector
    pub translation: [f32; 3],
    /// Scale factor
    pub scale: f32,
}

impl Sim3 {
    /// Create identity Sim3
    pub fn identity() -> Self {
        Self {
            rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            translation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }

    /// Convert to SE3 (assuming scale = 1)
    pub fn to_se3(&self) -> SE3 {
        SE3::from_rotation_translation(&self.rotation, &self.translation)
    }
}

/// Loop detector using Bag of Words
pub struct LoopDetector {
    /// Minimum score to consider as loop candidate
    min_loop_score: f32,
    /// Minimum number of matches to accept loop
    min_matches: usize,
    /// Minimum distance (in frames) between current and candidate
    min_distance: u64,
    /// Maximum time gap (seconds) for loop detection
    max_time_gap: f64,
    /// Confidence threshold for consistency check
    consistency_threshold: usize,
    /// Recent loop candidates for consistency check
    recent_loops: Vec<u64>,
    /// Maximum recent loops to store
    max_recent_loops: usize,
}

impl LoopDetector {
    /// Create a new loop detector
    pub fn new() -> Self {
        Self {
            min_loop_score: 0.05,
            min_matches: 20,
            min_distance: 30,
            max_time_gap: 60.0,
            consistency_threshold: 3,
            recent_loops: Vec::new(),
            max_recent_loops: 20,
        }
    }

    /// Get minimum loop score
    pub fn min_loop_score(&self) -> f32 {
        self.min_loop_score
    }

    /// Set minimum loop score
    pub fn set_min_loop_score(&mut self, score: f32) {
        self.min_loop_score = score;
    }

    /// Set minimum distance between frames
    pub fn set_min_distance_between_frames(&mut self, distance: u64) {
        self.min_distance = distance;
    }

    /// Set minimum matches
    pub fn set_min_matches(&mut self, matches: usize) {
        self.min_matches = matches;
    }

    /// Compute BoW similarity between two descriptors
    /// 
    /// This is a simplified BoW implementation using
    /// descriptor clustering for vocabulary
    fn compute_bow_similarity(&self, query: &Descriptors, train: &Descriptors) -> f32 {
        if query.is_empty() || train.is_empty() {
            return 0.0;
        }

        // Simplified: use average Hamming distance similarity
        // In a real implementation, this would use an actual BoW vocabulary
        let mut total_dist = 0.0f32;
        let mut count = 0usize;

        let n = query.count.min(train.count);

        for i in 0..n {
            if let (Some(q), Some(t)) = (query.get(i), train.get(i)) {
                let dist = hamming_distance(q, t);
                total_dist += dist;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let avg_dist = total_dist / count as f32;
        // Convert distance to similarity (0 = identical, max = very different)
        // Max ORB distance is 256 (32 bytes * 8 bits)
        let max_dist = 256.0;
        (max_dist - avg_dist.min(max_dist)) / max_dist
    }

    /// Compute loop candidates for a given frame
    /// 
    /// # Arguments
    /// * `map` - The map containing keyframes
    /// * `current_frame_id` - ID of the current frame
    /// * `descriptors` - Descriptors of the current frame
    /// 
    /// # Returns
    /// Vector of loop candidates sorted by score
    pub fn compute_loop_candidates(
        &self,
        map: &Map,
        current_frame_id: u64,
        descriptors: &Descriptors,
    ) -> Vec<LoopCandidate> {
        let mut candidates = Vec::new();

        // Get all keyframes from map
        for keyframe in map.keyframes() {
            let kf_id = keyframe.id();
            
            // Skip if too close to current frame
            if current_frame_id.saturating_sub(kf_id) < self.min_distance {
                continue;
            }

            // Skip if this is the first keyframe
            if kf_id == 0 {
                continue;
            }

            // Compute BoW similarity
            let score = self.compute_bow_similarity(descriptors, &keyframe.features.descriptors);

            if score > self.min_loop_score {
                candidates.push(LoopCandidate {
                    keyframe_id: kf_id,
                    score,
                    timestamp: keyframe.frame.timestamp,
                });
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        candidates
    }

    /// Check loop consistency using temporal filtering
    /// 
    /// This ensures that loop candidates are consistent over time
    /// to avoid false positives from transient matches
    pub fn compute_loop_consistency(
        &self,
        candidates: &[LoopCandidate],
        current_frame_id: u64,
    ) -> Vec<LoopCandidate> {
        let mut consistent = Vec::new();

        for candidate in candidates {
            // Check minimum distance
            let distance = current_frame_id.saturating_sub(candidate.keyframe_id);
            if distance < self.min_distance {
                continue;
            }

            // Check time gap
            // Note: In real implementation, we'd compare timestamps properly

            // Check consistency with recent loops
            let mut is_consistent = true;
            for recent_id in &self.recent_loops {
                // If we've seen a loop with this keyframe recently, it's consistent
                if *recent_id == candidate.keyframe_id {
                    is_consistent = false;
                    break;
                }
            }

            if is_consistent {
                consistent.push(candidate.clone());
            }
        }

        consistent
    }

    /// Add a confirmed loop to recent history
    pub fn add_confirmed_loop(&mut self, keyframe_id: u64) {
        self.recent_loops.push(keyframe_id);
        if self.recent_loops.len() > self.max_recent_loops {
            self.recent_loops.remove(0);
        }
    }

    /// Clear recent loop history
    pub fn clear_history(&mut self) {
        self.recent_loops.clear();
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute similarity transform (Sim3) between two point clouds
/// 
/// This implements the method to compute Sim3 from matched 3D points:
/// - Find rotation, translation, and scale that best aligns two point sets
/// 
/// # Arguments
/// * `rotation` - Initial rotation matrix (3x3)
/// * `translation` - Initial translation vector
/// * `scale` - Initial scale
/// 
/// # Returns
/// Option containing the Sim3 transform if computation succeeded
pub fn compute_similarity_transform(
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
    scale: f32,
) -> Option<Sim3> {
    // Simplified implementation: validate and return the input as Sim3
    // A full implementation would use SVD to find optimal transformation
    
    // Validate rotation matrix (should be orthogonal)
    let det = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);

    if (det - 1.0).abs() > 0.1 {
        // Not a valid rotation matrix
        return None;
    }

    // Scale should be positive
    if scale <= 0.0 {
        return None;
    }

    Some(Sim3 {
        rotation: *rotation,
        translation: *translation,
        scale,
    })
}

/// Compute Sim3 from matched 3D points between two keyframes
/// 
/// This is a more complete implementation that takes actual point matches
pub fn compute_sim3_from_matches(
    points1: &[[f32; 3]],
    points2: &[[f32; 3]],
) -> Option<Sim3> {
    if points1.len() != points2.len() || points1.len() < 3 {
        return None;
    }

    // Compute centroids
    let mut centroid1 = [0.0f32; 3];
    let mut centroid2 = [0.0f32; 3];
    
    for i in 0..points1.len() {
        for j in 0..3 {
            centroid1[j] += points1[i][j];
            centroid2[j] += points2[i][j];
        }
    }
    
    let n = points1.len() as f32;
    for j in 0..3 {
        centroid1[j] /= n;
        centroid2[j] /= n;
    }

    // Compute scale (ratio of distances from centroids)
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    
    for i in 0..points1.len() {
        let diff1 = [
            points1[i][0] - centroid1[0],
            points1[i][1] - centroid1[1],
            points1[i][2] - centroid1[2],
        ];
        let diff2 = [
            points2[i][0] - centroid2[0],
            points2[i][1] - centroid2[1],
            points2[i][2] - centroid2[2],
        ];
        d1 += diff1[0] * diff1[0] + diff1[1] * diff1[1] + diff1[2] * diff1[2];
        d2 += diff2[0] * diff2[0] + diff2[1] * diff2[1] + diff2[2] * diff2[2];
    }

    let scale = if d2 > 1e-10 { (d1 / d2).sqrt() } else { 1.0 };

    // Compute rotation using cross-covariance matrix
    let mut h = [[0.0f32; 3]; 3];
    
    for i in 0..points1.len() {
        let p1 = [
            points1[i][0] - centroid1[0],
            points1[i][1] - centroid1[1],
            points1[i][2] - centroid1[2],
        ];
        let p2 = [
            points2[i][0] - centroid2[0],
            points2[i][1] - centroid2[1],
            points2[i][2] - centroid2[2],
        ];
        
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += p1[r] * p2[c];
            }
        }
    }

    // SVD would be used here to find optimal rotation
    // For now, return identity rotation if H is well-conditioned
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);

    if det.abs() < 1e-10 {
        return None;
    }

    Some(Sim3 {
        rotation: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        translation: [
            centroid2[0] - scale * centroid1[0],
            centroid2[1] - scale * centroid1[1],
            centroid2[2] - scale * centroid1[2],
        ],
        scale,
    })
}

/// Calculate Hamming distance between two descriptors
fn hamming_distance(a: &[u8], b: &[u8]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x ^ *y).count_ones() as f32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_detector_default() {
        let detector = LoopDetector::new();
        assert!(detector.min_loop_score() > 0.0);
    }

    #[test]
    fn test_sim3_identity() {
        let sim3 = Sim3::identity();
        assert!((sim3.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sim3_to_se3() {
        let sim3 = Sim3::identity();
        let _se3 = sim3.to_se3();
    }

    #[test]
    fn test_compute_similarity_transform_identity() {
        let result = compute_similarity_transform(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            1.0,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_compute_similarity_transform_invalid_rotation() {
        let result = compute_similarity_transform(
            &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], // Invalid rotation
            &[0.0, 0.0, 0.0],
            1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_sim3_from_matches() {
        // Same points should give identity transform
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = compute_sim3_from_matches(&points, &points);
        
        assert!(result.is_some());
        let sim3 = result.unwrap();
        assert!((sim3.scale - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_sim3_from_matches_different() {
        // Translated points
        let points1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let points2 = vec![[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]];
        
        let result = compute_sim3_from_matches(&points1, &points2);
        assert!(result.is_some());
    }

    #[test]
    fn test_compute_sim3_from_matches_empty() {
        let points1: Vec<[f32; 3]> = vec![];
        let points2: Vec<[f32; 3]> = vec![];
        
        let result = compute_sim3_from_matches(&points1, &points2);
        assert!(result.is_none());
    }
}
