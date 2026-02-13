//! KNN Matcher using kiddo KD-Tree for efficient feature matching
//!
//! This module provides a KD-Tree based K-Nearest Neighbors matcher
//! optimized for 32-dimensional feature descriptors (e.g., ORB).

use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use std::collections::HashMap;

/// KNN Matcher using KD-Tree for efficient nearest neighbor search
/// 
/// Optimized for 32-dimensional descriptors (256-bit ORB descriptors)
pub struct KnnMatcher {
    /// KD-Tree storing the training descriptors
    tree: KdTree<f64, 32>,
    /// Number of neighbors to search for
    k: usize,
    /// Whether the tree has been built
    built: bool,
}

impl KnnMatcher {
    /// Create a new KnnMatcher with specified number of neighbors
    /// 
    /// # Arguments
    /// * `k` - Number of nearest neighbors to find
    /// 
    /// # Example
    /// ```
    /// let matcher = KnnMatcher::new(2);
    /// ```
    pub fn new(k: usize) -> Self {
        Self {
            tree: KdTree::new(),
            k,
            built: false,
        }
    }

    /// Build the KD-Tree from training descriptors
    /// 
    /// # Arguments
    /// * `descriptors` - Vector of descriptors, each descriptor is a [f64; 32]
    /// 
    /// # Example
    /// ```
    /// let mut matcher = KnnMatcher::new(2);
    /// let descriptors: Vec<[f64; 32]> = vec![...];
    /// matcher.build_tree(&descriptors);
    /// ```
    pub fn build_tree(&mut self, descriptors: &[[f64; 32]]) {
        self.tree = KdTree::new();
        for (idx, descriptor) in descriptors.iter().enumerate() {
            self.tree.add(descriptor, idx as u64);
        }
        self.built = true;
    }

    /// Perform KNN matching for a single query descriptor
    /// 
    /// # Arguments
    /// * `query` - Query descriptor [f64; 32]
    /// 
    /// # Returns
    /// Vector of (distance, index) pairs for the k nearest neighbors
    pub fn knn_match(&self, query: &[f64; 32]) -> Vec<(f64, u64)> {
        if !self.built {
            return Vec::new();
        }
        
        let results = self.tree.nearest_n::<SquaredEuclidean>(query, self.k);
        results
            .into_iter()
            .map(|r| (r.distance, r.item))
            .collect()
    }

    /// Perform KNN matching for multiple query descriptors
    /// 
    /// # Arguments
    /// * `queries` - Vector of query descriptors
    /// 
    /// # Returns
    /// Vector of vectors, each containing (distance, index) pairs
    pub fn knn_match_batch(&self, queries: &[[f64; 32]]) -> Vec<Vec<(f64, u64)>> {
        queries.iter().map(|q| self.knn_match(q)).collect()
    }

    /// Match with Lowe's ratio test for filtering ambiguous matches
    /// 
    /// The ratio test filters matches where the best match is significantly
    /// better than the second best match. This helps eliminate ambiguous matches.
    /// 
    /// # Arguments
    /// * `query` - Query descriptor [f64; 32]
    /// * `ratio_threshold` - Lowe's ratio test threshold (typically 0.7-0.8)
    /// 
    /// # Returns
    /// Vector of (distance, index) pairs that pass the ratio test
    pub fn match_with_ratio(&self, query: &[f64; 32], ratio_threshold: f64) -> Vec<(f64, u64)> {
        let matches = self.knn_match(query);
        
        if matches.len() < 2 {
            return matches;
        }

        let best_distance = matches[0].0;
        let second_best_distance = matches[1].0;

        // Apply Lowe's ratio test
        if best_distance < ratio_threshold * second_best_distance {
            vec![matches[0].clone()]
        } else {
            Vec::new()
        }
    }

    /// Match multiple queries with ratio test
    /// 
    /// # Arguments
    /// * `queries` - Vector of query descriptors
    /// * `ratio_threshold` - Lowe's ratio test threshold
    /// 
    /// # Returns
    /// HashMap mapping query index to matched (distance, index) pairs
    pub fn match_batch_with_ratio(&self, queries: &[[f64; 32]], ratio_threshold: f64) -> HashMap<usize, Vec<(f64, u64)>> {
        let mut results = HashMap::new();
        for (idx, query) in queries.iter().enumerate() {
            let matches = self.match_with_ratio(query, ratio_threshold);
            if !matches.is_empty() {
                results.insert(idx, matches);
            }
        }
        results
    }

    /// Check if the KD-Tree has been built
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Get the number of neighbors (k)
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the number of descriptors in the tree
    pub fn size(&self) -> u64 {
        self.tree.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to generate test descriptors
    fn generate_test_descriptors(count: usize, seed: usize) -> Vec<[f64; 32]> {
        (0..count)
            .map(|i| {
                let base = (i + seed) as f64;
                let mut desc = [0.0; 32];
                for j in 0..32 {
                    desc[j] = (base + j as f64 * 0.1).sin().abs();
                }
                desc
            })
            .collect()
    }

    /// Helper function to compute Euclidean distance between descriptors
    fn descriptor_distance(a: &[f64; 32], b: &[f64; 32]) -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_new_matcher() {
        let matcher = KnnMatcher::new(2);
        assert_eq!(matcher.k(), 2);
        assert!(!matcher.is_built());
    }

    #[test]
    fn test_build_tree() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(10, 42);
        
        matcher.build_tree(&descriptors);
        
        assert!(matcher.is_built());
        assert_eq!(matcher.size(), 10);
    }

    #[test]
    fn test_knn_match_single() {
        let mut matcher = KnnMatcher::new(3);
        let descriptors = generate_test_descriptors(10, 0);
        
        matcher.build_tree(&descriptors);
        
        // Query with the first descriptor - should find itself as nearest
        let query = descriptors[0];
        let matches = matcher.knn_match(&query);
        
        assert_eq!(matches.len(), 3);
        // First match should be the query itself (distance = 0)
        assert_eq!(matches[0].1, 0); // index 0
        assert!((matches[0].0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_match_batch() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(20, 100);
        
        matcher.build_tree(&descriptors);
        
        let queries: Vec<[f64; 32]> = vec![descriptors[0], descriptors[5], descriptors[10]];
        let results = matcher.knn_match_batch(&queries);
        
        assert_eq!(results.len(), 3);
        for (i, matches) in results.iter().enumerate() {
            assert_eq!(matches.len(), 2);
            // Each query should find itself as the best match
            assert!((matches[0].0).abs() < 1e-10);
            assert_eq!(matches[0].1, (i * 5) as u64); // indices 0, 5, 10
        }
    }

    #[test]
    fn test_match_with_ratio() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(10, 0);
        
        matcher.build_tree(&descriptors);
        
        // Query with first descriptor
        let query = descriptors[0];
        let matches = matcher.match_with_ratio(&query, 0.75);
        
        // Should return match since best distance << second best
        assert!(!matches.is_empty());
        assert_eq!(matches[0].1, 0);
    }

    #[test]
    fn test_match_with_ratio_threshold() {
        let mut matcher = KnnMatcher::new(2);
        
        // Create descriptors where second match is too close to best
        let mut desc1 = [0.0; 32];
        let mut desc2 = [0.1; 32]; 
        let mut desc3 = [1.0; 32]; // Far
        
        for i in 0..32 {
            desc1[i] = (i as f64).sin();
            desc2[i] = desc1[i] + 0.01; // Very small difference
            desc3[i] = (i as f64).cos();
        }
        
        let descriptors = vec![desc1.clone(), desc2.clone(), desc3.clone()];
        matcher.build_tree(&descriptors);
        
        // Query with first descriptor - desc2 is too close to desc1
        // Ratio test should fail since best_distance / second_best_distance > threshold
        let matches = matcher.match_with_ratio(&desc1, 0.75);
        
        // With 0.75 threshold, if the second best is very close to best, test should fail
        let distance_1_1 = descriptor_distance(&desc1, &desc1);
        let distance_1_2 = descriptor_distance(&desc1, &desc2);
        let distance_1_3 = descriptor_distance(&desc1, &desc3);
        
        // If ratio is greater than threshold, match should be rejected
        let ratio = distance_1_1 / distance_1_2;
        if ratio > 0.75 {
            // Ratio test should fail - no match returned
            assert!(matches.is_empty(), "Expected no match due to ratio test, but got matches");
        } else {
            // Ratio test passes - match should be returned
            assert!(!matches.is_empty(), "Expected match but got none");
        }
    }

    #[test]
    fn test_match_batch_with_ratio() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(15, 0);
        
        matcher.build_tree(&descriptors);
        
        let queries: Vec<[f64; 32]> = vec![descriptors[0], descriptors[3], descriptors[7]];
        let results = matcher.match_batch_with_ratio(&queries, 0.75);
        
        // Should have matches for all 3 queries
        assert_eq!(results.len(), 3);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(results.contains_key(&2));
    }

    #[test]
    fn test_empty_tree_match() {
        let matcher = KnnMatcher::new(2);
        let query = [0.0; 32];
        
        let matches = matcher.knn_match(&query);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_distance_ordering() {
        let mut matcher = KnnMatcher::new(3);
        
        // Create descriptors at known distances
        let mut center = [0.0; 32];
        for i in 0..32 {
            center[i] = (i as f64 * 0.5).sin();
        }
        
        let mut near = center;
        near[0] += 0.1; // Small perturbation
        
        let mut far = center;
        far[0] += 10.0; // Large perturbation
        
        let descriptors = vec![far, near, center];
        matcher.build_tree(&descriptors);
        
        let matches = matcher.knn_match(&center);
        
        // Should be ordered by distance
        assert!(matches[0].0 <= matches[1].0);
        assert!(matches[1].0 <= matches[2].0);
    }
}
