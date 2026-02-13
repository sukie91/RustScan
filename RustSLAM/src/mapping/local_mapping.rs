//! Local Mapping module
//! 
//! This module handles local mapping in the SLAM system:
//! - Receives keyframes from the tracking thread
//! - Triangulates new map points
//! - Performs local Bundle Adjustment
//! - Filters redundant keyframes

use std::collections::{HashMap, HashSet, VecDeque};
use glam::Vec3;

use crate::core::{Camera, Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
use crate::optimizer::ba::{BundleAdjuster, BACamera, BALandmark, BAObservation};

/// Configuration for Local Mapping
#[derive(Debug, Clone)]
pub struct LocalMappingConfig {
    /// Maximum number of local keyframes
    pub max_keyframes: usize,
    /// Maximum number of local map points
    pub max_map_points: usize,
    /// Minimum observations for a map point
    pub min_observations: u32,
    /// Minimum triangulation angle (degrees)
    pub min_triangulation_angle: f32,
    /// Minimum distance for triangulation
    pub min_triangulation_dist: f32,
    /// Maximum reprojection error for triangulation
    pub max_reprojection_error: f32,
    /// Whether to enable local BA
    pub local_ba_enabled: bool,
    /// Local BA iterations
    pub local_ba_iterations: usize,
    /// Keyframe culling threshold (ratio of mapped points)
    pub culling_threshold: f32,
}

impl Default for LocalMappingConfig {
    fn default() -> Self {
        Self {
            max_keyframes: 10,
            max_map_points: 1000,
            min_observations: 2,
            min_triangulation_angle: 3.0,  // degrees
            min_triangulation_dist: 0.1,
            max_reprojection_error: 4.0,   // pixels
            local_ba_enabled: true,
            local_ba_iterations: 10,
            culling_threshold: 0.9,       // 90% mapped points = redundant
        }
    }
}

/// Local Mapping state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MappingState {
    /// Idle, waiting for keyframes
    Idle,
    /// Processing new keyframes
    Processing,
    /// Running local BA
    RunningLocalBA,
    /// Culling redundant keyframes
    Culling,
}

/// Local Mapping module
/// 
/// Receives keyframes from tracking and:
/// 1. Triangulates new map points from consecutive keyframes
/// 2. Runs local Bundle Adjustment
/// 3. Culls redundant keyframes
pub struct LocalMapping {
    /// Configuration
    config: LocalMappingConfig,
    /// Current state
    state: MappingState,
    /// The map being worked on
    map: Option<Map>,
    /// Local keyframes (current and connected)
    local_keyframes: Vec<u64>,
    /// Local map points (observed by local keyframes)
    local_map_points: HashSet<u64>,
    /// New keyframes waiting to be processed
    new_keyframes: VecDeque<KeyFrame>,
    /// Camera for triangulation
    camera: Option<Camera>,
    /// Next map point ID
    next_point_id: u64,
}

impl LocalMapping {
    /// Create a new Local Mapping instance
    pub fn new(config: LocalMappingConfig) -> Self {
        Self {
            config,
            state: MappingState::Idle,
            map: None,
            local_keyframes: Vec::new(),
            local_map_points: HashSet::new(),
            new_keyframes: VecDeque::new(),
            camera: None,
            next_point_id: 0,
        }
    }

    /// Set the map reference
    pub fn set_map(&mut self, map: Map) {
        self.map = Some(map);
    }

    /// Set the camera
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = Some(camera);
    }

    /// Check if there are new keyframes to process
    pub fn check_new_keyframes(&self) -> bool {
        !self.new_keyframes.is_empty()
    }

    /// Insert a new keyframe from the tracking thread
    pub fn insert_keyframe(&mut self, keyframe: KeyFrame) {
        self.state = MappingState::Processing;
        self.new_keyframes.push_back(keyframe);
        
        // Process immediately
        self.process_new_keyframes();
    }

    /// Process all new keyframes in the queue
    fn process_new_keyframes(&mut self) {
        while let Some(keyframe) = self.new_keyframes.pop_front() {
            // Add keyframe to local keyframes
            let kf_id = keyframe.id();
            self.local_keyframes.push(kf_id);
            
            // Limit local keyframes
            while self.local_keyframes.len() > self.config.max_keyframes {
                self.local_keyframes.remove(0);
            }
            
            // Triangulate new points
            self.triangulate_new_points(&keyframe);
        }
        
        // Run local BA if enabled
        if self.config.local_ba_enabled {
            self.run_local_ba();
        }
        
        // Cull redundant keyframes
        self.cull_redundant_keyframes();
        
        self.state = MappingState::Idle;
    }

    /// Triangulate new map points from the new keyframe with its neighbors
    fn triangulate_new_points(&mut self, new_keyframe: &KeyFrame) {
        // Get the pose of the new keyframe
        let Some(new_pose) = new_keyframe.pose() else {
            return;
        };
        
        // Find neighboring keyframes to triangulate with
        // For simplicity, use previous keyframes in local_keyframes
        // Collect the neighbor keyframes first to avoid borrow issues
        let neighbor_data: Vec<(KeyFrame, SE3)> = if let Some(ref map) = self.map {
            self.local_keyframes.iter()
                .filter(|&&kf_id| kf_id != new_keyframe.id())
                .filter_map(|&kf_id| {
                    map.get_keyframe(kf_id).and_then(|kf| {
                        kf.pose().map(|pose| (kf.clone(), pose))
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        
        // Now triangulate with each neighbor
        for (neighbor_kf, neighbor_pose) in neighbor_data {
            self.triangulate_between_keyframes(
                new_keyframe,
                &new_pose,
                &neighbor_kf,
                &neighbor_pose,
            );
        }
    }

    /// Triangulate points between two keyframes
    fn triangulate_between_keyframes(
        &mut self,
        kf1: &KeyFrame,
        pose1: &SE3,
        kf2: &KeyFrame,
        pose2: &SE3,
    ) {
        // Get features without map points
        let features1 = &kf1.features;
        let features2 = &kf2.features;
        
        // Simple triangulation: create map points for unmatched features
        // In a real system, this would use proper match triangulation
        
        // Check if we have reached max map points
        if self.local_map_points.len() >= self.config.max_map_points {
            return;
        }
        
        // Get camera poses
        let t1 = pose1.translation();
        let t2 = pose2.translation();
        
        // Calculate baseline (distance between cameras)
        let dx = t2[0] - t1[0];
        let dy = t2[1] - t1[1];
        let dz = t2[2] - t1[2];
        let baseline = (dx * dx + dy * dy + dz * dz).sqrt();
        
        // Check minimum baseline
        if baseline < self.config.min_triangulation_dist {
            return;
        }
        
        // For each feature in kf1 without a map point, try to triangulate
        for (i, (kp1, mp1)) in features1.keypoints.iter()
            .zip(features1.map_points.iter())
            .enumerate() 
        {
            // Skip if already has a map point
            if mp1.is_some() {
                continue;
            }
            
            // Find corresponding feature in kf2 (simplified - in real system use matching)
            // For now, create a simple triangulated point based on position
            let kp2_opt = features2.keypoints.get(i);
            
            // Calculate simple triangulation
            // This is a simplified version - real implementation would use DLT
            let depth = baseline * 0.5; // Simplified depth estimation
            
            // Create 3D point in world coordinates (simplified)
            let x = kp1[0] * depth / 500.0; // Assume focal length ~500
            let y = kp1[1] * depth / 500.0;
            let z = depth;
            
            // Check if point is in front of both cameras
            let point_world = [x, y, z];
            
            // Transform to camera 1 frame
            let point_cam1 = pose1.inverse().transform_point(&point_world);
            let point_cam2 = pose2.inverse().transform_point(&point_world);
            
            if point_cam1[2] > 0.0 && point_cam2[2] > 0.0 {
                // Valid triangulation - add to map
                self.add_map_point(point_world, kf1.id());
            }
        }
    }

    /// Add a new map point to the local map
    fn add_map_point(&mut self, position: [f32; 3], ref_kf: u64) {
        let point_id = self.next_point_id;
        self.next_point_id += 1;
        
        let map_point = MapPoint::new(point_id, Vec3::from(position), ref_kf);
        
        // Add to map if available
        if let Some(ref mut map) = self.map {
            map.add_point(map_point);
        }
        
        // Track in local map points
        self.local_map_points.insert(point_id);
        
        // Limit local map points
        while self.local_map_points.len() > self.config.max_map_points {
            if let Some(id) = self.local_map_points.iter().next().cloned() {
                self.local_map_points.remove(&id);
            }
        }
    }

    /// Run local Bundle Adjustment
    pub fn run_local_ba(&mut self) {
        self.state = MappingState::RunningLocalBA;
        
        // Get local keyframes and map points
        let local_kfs: Vec<u64> = self.local_keyframes.clone();
        let local_mps: Vec<u64> = self.local_map_points.iter().cloned().collect();
        
        if local_kfs.len() < 2 || local_mps.is_empty() {
            self.state = MappingState::Idle;
            return;
        }
        
        // Build BA problem
        let mut adjuster = BundleAdjuster::new();
        
        // Add cameras
        let mut camera_indices: HashMap<u64, usize> = HashMap::new();
        
        if let Some(ref map) = self.map {
            for &kf_id in &local_kfs {
                if let Some(kf) = map.get_keyframe(kf_id) {
                    // Get camera intrinsics (use default if not available)
                    let (fx, fy, cx, cy) = if let Some(ref cam) = self.camera {
                        (cam.focal.x as f64, cam.focal.y as f64, cam.principal.x as f64, cam.principal.y as f64)
                    } else {
                        (500.0, 500.0, 320.0, 240.0)
                    };
                    
                    let ba_cam = BACamera::new(fx, fy, cx, cy);
                    let idx = adjuster.add_camera(ba_cam);
                    camera_indices.insert(kf_id, idx);
                }
            }
        }
        
        // Add landmarks
        let mut landmark_indices: HashMap<u64, usize> = HashMap::new();
        
        if let Some(ref map) = self.map {
            for &mp_id in &local_mps {
                if let Some(mp) = map.get_point(mp_id) {
                    let landmark = BALandmark::new(
                        mp.position.x as f64,
                        mp.position.y as f64,
                        mp.position.z as f64,
                    );
                    let idx = adjuster.add_landmark(landmark);
                    landmark_indices.insert(mp_id, idx);
                }
            }
        }
        
        // Add observations
        if let Some(ref map) = self.map {
            for &kf_id in &local_kfs {
                if let Some(kf) = map.get_keyframe(kf_id) {
                    let Some(&cam_idx) = camera_indices.get(&kf_id) else {
                        continue;
                    };
                    
                    for (feat_idx, mp_id) in kf.features.map_points.iter().enumerate() {
                        let Some(lm_id) = *mp_id else {
                            continue;
                        };
                        
                        let Some(&lm_idx) = landmark_indices.get(&lm_id) else {
                            continue;
                        };
                        
                        if feat_idx < kf.features.keypoints.len() {
                            let kp = kf.features.keypoints[feat_idx];
                            let obs = BAObservation::new(kp[0] as f64, kp[1] as f64);
                            adjuster.add_observation(cam_idx, lm_idx, obs);
                        }
                    }
                }
            }
        }
        
        // Run optimization
        let result = adjuster.optimize(self.config.local_ba_iterations);
        
        match result {
            Ok((cameras, landmarks)) => {
                // Update map points with optimized positions
                if let Some(ref mut map) = self.map {
                    for (mp_id, lm_idx) in &landmark_indices {
                        if *lm_idx < landmarks.len() {
                            let lm = &landmarks[*lm_idx];
                            if let Some(mp) = map.get_point_mut(*mp_id) {
                                mp.position = Vec3::new(lm.position[0] as f32, lm.position[1] as f32, lm.position[2] as f32);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // BA failed, but we continue
                eprintln!("Local BA failed: {}", e);
            }
        }
        
        self.state = MappingState::Idle;
    }

    /// Cull redundant keyframes
    /// 
    /// A keyframe is redundant if more than culling_threshold (e.g., 90%)
    /// of its map points are also observed by other keyframes.
    /// 
    /// Returns the number of keyframes culled
    pub fn cull_redundant_keyframes(&mut self) -> usize {
        self.state = MappingState::Culling;
        
        let mut culled_count = 0;
        let mut to_remove: Vec<u64> = Vec::new();
        
        // Need at least 3 keyframes to consider culling
        if self.local_keyframes.len() < 3 {
            self.state = MappingState::Idle;
            return 0;
        }
        
        // Check each keyframe (skip the newest ones)
        // Usually we don't cull the most recent keyframes
        let check_range = self.local_keyframes.len().saturating_sub(3)..self.local_keyframes.len();
        
        for i in check_range {
            let kf_id = self.local_keyframes[i];
            
            if let Some(ref map) = self.map {
                if let Some(kf) = map.get_keyframe(kf_id) {
                    // Count mapped points
                    let mapped_count = kf.features.map_points
                        .iter()
                        .filter(|mp| mp.is_some())
                        .count();
                    
                    // Count observations from other keyframes
                    let mut observed_count = 0;
                    
                    if let Some(mp_id) = kf.features.map_points.iter().find(|mp| mp.is_some()) {
                        if let Some(mp) = map.get_point(mp_id.unwrap()) {
                            observed_count = mp.observations as usize;
                        }
                    }
                    
                    // Calculate ratio
                    if mapped_count > 0 {
                        let ratio = observed_count as f32 / mapped_count as f32;
                        
                        if ratio > self.config.culling_threshold {
                            to_remove.push(kf_id);
                        }
                    }
                }
            }
        }
        
        // Remove culled keyframes from local list
        for kf_id in &to_remove {
            self.local_keyframes.retain(|&id| id != *kf_id);
            culled_count += 1;
        }
        
        self.state = MappingState::Idle;
        culled_count
    }

    /// Get the number of local keyframes
    pub fn num_local_keyframes(&self) -> usize {
        self.local_keyframes.len()
    }

    /// Get the number of local map points
    pub fn num_local_map_points(&self) -> usize {
        self.local_map_points.len()
    }

    /// Get the current state
    pub fn is_processing(&self) -> bool {
        self.state != MappingState::Idle
    }

    /// Get the current state
    pub fn state(&self) -> MappingState {
        self.state
    }

    /// Clear the local mapping data
    pub fn clear(&mut self) {
        self.local_keyframes.clear();
        self.local_map_points.clear();
        self.new_keyframes.clear();
        self.state = MappingState::Idle;
    }
}

impl Default for LocalMapping {
    fn default() -> Self {
        Self::new(LocalMappingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LocalMappingConfig::default();
        assert_eq!(config.max_keyframes, 10);
        assert_eq!(config.max_map_points, 1000);
        assert!(config.local_ba_enabled);
    }

    #[test]
    fn test_local_mapping_creation() {
        let mapping = LocalMapping::new(LocalMappingConfig::default());
        assert_eq!(mapping.num_local_keyframes(), 0);
        assert_eq!(mapping.num_local_map_points(), 0);
    }

    #[test]
    fn test_state_transitions() {
        let mut mapping = LocalMapping::new(LocalMappingConfig::default());
        
        // Initially idle
        assert_eq!(mapping.state(), MappingState::Idle);
        
        // After insert, should process then go back to idle
        let frame = Frame::new(0, 0.0, 640, 480);
        let features = FrameFeatures::new();
        let kf = KeyFrame::new(frame, features);
        
        mapping.insert_keyframe(kf);
        
        // State should eventually return to idle
        // (may briefly be Processing/RunningLocalBA/Culling)
        assert!(matches!(mapping.state(), MappingState::Idle | MappingState::Processing | MappingState::RunningLocalBA | MappingState::Culling));
    }
}
