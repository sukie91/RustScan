//! Visual Odometry

use crate::core::{Camera, SE3};
use crate::features::{FeatureExtractor, FeatureMatcher, KeyPoint, Descriptors};
use crate::tracker::solver::{PnPSolver, EssentialSolver, Triangulator};
use glam::Mat3;

/// Visual Odometry state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VOState {
    /// Not initialized
    NotInitialized,
    /// Initializing (need more frames)
    Initializing,
    /// Tracking OK
    TrackingOk,
    /// Lost tracking
    TrackingLost,
}

/// Visual Odometry result
#[derive(Debug, Clone)]
pub struct VOResult {
    /// Estimated pose
    pub pose: SE3,
    /// Number of matches
    pub num_matches: usize,
    /// Number of inliers
    pub num_inliers: usize,
    /// Whether tracking is successful
    pub success: bool,
}

impl VOResult {
    fn failure() -> Self {
        Self {
            pose: SE3::identity(),
            num_matches: 0,
            num_inliers: 0,
            success: false,
        }
    }
}

/// Visual Odometry
pub struct VisualOdometry {
    /// Feature extractor
    extractor: Box<dyn FeatureExtractor>,
    /// Feature matcher
    matcher: Box<dyn FeatureMatcher>,
    /// Camera model
    camera: Camera,
    /// Current state
    state: VOState,
    /// PnP solver for 3D-2D
    pnp_solver: PnPSolver,
    /// Essential matrix solver for 2D-2D
    essential_solver: EssentialSolver,
    /// Triangulator
    triangulator: Triangulator,
    /// Previous frame ID
    prev_frame_id: Option<u64>,
    /// Previous frame pose
    prev_frame_pose: Option<SE3>,
    /// Previous keypoints
    prev_keypoints: Vec<KeyPoint>,
    /// Previous descriptors
    prev_descriptors: Descriptors,
    /// Previous 3D points (triangulated)
    prev_3d_points: Vec<Option<[f32; 3]>>,
    /// Frame count
    frame_count: u64,
    /// Min matches to proceed
    min_matches: usize,
    /// Min inliers to maintain tracking
    min_inliers: usize,
}

impl VisualOdometry {
    /// Create a new VO
    pub fn new(
        extractor: Box<dyn FeatureExtractor>,
        matcher: Box<dyn FeatureMatcher>,
        camera: Camera,
    ) -> Self {
        // Get intrinsics from camera
        let fx = camera.focal.x;
        let fy = camera.focal.y;
        let cx = camera.principal.x;
        let cy = camera.principal.y;
        
        let pnp_solver = PnPSolver::new(fx, fy, cx, cy);
        
        Self {
            extractor,
            matcher,
            camera,
            state: VOState::NotInitialized,
            pnp_solver,
            essential_solver: EssentialSolver::new(),
            triangulator: Triangulator::new(),
            prev_frame_id: None,
            prev_frame_pose: None,
            prev_keypoints: Vec::new(),
            prev_descriptors: Descriptors::new(),
            prev_3d_points: Vec::new(),
            frame_count: 0,
            min_matches: 20,
            min_inliers: 10,
        }
    }

    /// Process a new frame and estimate pose
    pub fn process_frame(&mut self, image: &[u8], width: u32, height: u32) -> VOResult {
        self.frame_count += 1;

        // Extract features
        let (keypoints, descriptors) = match self.extractor.detect_and_compute(image, width, height) {
            Ok(result) => result,
            Err(_) => return VOResult::failure(),
        };

        if keypoints.is_empty() {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        let result = match self.state {
            VOState::NotInitialized | VOState::Initializing => {
                self.initialize(keypoints.clone(), descriptors.clone())
            }
            VOState::TrackingOk => {
                self.track(keypoints.clone(), descriptors.clone())
            }
            VOState::TrackingLost => {
                // Try to relocalize
                self.relocalize(keypoints.clone(), descriptors.clone())
            }
        };

        // Store for next iteration
        if result.success {
            self.prev_frame_id = Some(self.frame_count);
            self.prev_keypoints = keypoints;
            self.prev_descriptors = descriptors;
            self.prev_frame_pose = Some(result.pose);
        }

        result
    }

    /// Initialize with first few frames (2D-2D motion)
    fn initialize(&mut self, keypoints: Vec<KeyPoint>, descriptors: Descriptors) -> VOResult {
        if self.prev_keypoints.is_empty() {
            // First frame - just store
            self.state = VOState::Initializing;
            self.prev_keypoints = keypoints;
            self.prev_descriptors = descriptors;
            return VOResult {
                pose: SE3::identity(),
                num_matches: 0,
                num_inliers: 0,
                success: true,
            };
        }

        // Second frame - estimate motion using essential matrix
        let matches = match self.matcher.match_descriptors(&descriptors, &self.prev_descriptors) {
            Ok(m) => m,
            Err(_) => {
                self.state = VOState::TrackingLost;
                return VOResult::failure();
            }
        };

        if matches.len() < self.min_matches {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        // Extract points
        let pts1: Vec<[f32; 2]> = self.prev_keypoints.iter()
            .map(|kp| [kp.x(), kp.y()])
            .collect();
        let pts2: Vec<[f32; 2]> = keypoints.iter()
            .map(|kp| [kp.x(), kp.y()])
            .collect();

        // Compute essential matrix
        if let Some((_E, inliers)) = self.essential_solver.compute(&matches, &pts1, &pts2) {
            let inlier_count = inliers.iter().filter(|&&x| x).count();
            
            if inlier_count >= self.min_inliers || matches.len() >= self.min_matches {
                // Recover pose from essential matrix
                let poses = self.essential_solver.recover_pose(Mat3::IDENTITY);
                
                // Use the first pose (simplified)
                let pose = poses[0].clone();
                
                // Triangulate 3D points
                let prev_pose = SE3::identity();
                self.prev_3d_points = self.triangulator.triangulate(
                    &prev_pose,
                    &pose,
                    &pts1,
                    &pts2,
                );
                
                self.state = VOState::TrackingOk;
                
                return VOResult {
                    pose,
                    num_matches: matches.len(),
                    num_inliers: inlier_count,
                    success: true,
                };
            }
        }

        self.state = VOState::TrackingLost;
        VOResult::failure()
    }

    /// Track using 3D-2D PnP
    fn track(&mut self, keypoints: Vec<KeyPoint>, descriptors: Descriptors) -> VOResult {
        // Match with previous frame
        let matches = match self.matcher.match_descriptors(&descriptors, &self.prev_descriptors) {
            Ok(m) => m,
            Err(_) => return VOResult::failure(),
        };

        if matches.len() < self.min_matches {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        // Build PnP problem from 3D-2D correspondences
        let mut pnp_problem = crate::tracker::solver::PnPProblem::new();
        
        for m in &matches {
            let query_idx = m.query_idx as usize;
            let train_idx = m.train_idx as usize;
            
            if query_idx < keypoints.len() && train_idx < self.prev_3d_points.len() {
                if let Some(pt3d) = self.prev_3d_points[train_idx] {
                    let pt2d = [keypoints[query_idx].x(), keypoints[query_idx].y()];
                    pnp_problem.add_correspondence(pt2d, pt3d);
                }
            }
        }

        // Solve PnP with RANSAC
        if let Some((pose, inliers)) = self.pnp_solver.solve(&pnp_problem) {
            let inlier_count = inliers.iter().filter(|&&x| x).count();
            
            if inlier_count >= self.min_inliers {
                return VOResult {
                    pose,
                    num_matches: matches.len(),
                    num_inliers: inlier_count,
                    success: true,
                };
            }
        }

        // Fallback to 2D-2D if PnP fails
        self.state = VOState::TrackingLost;
        VOResult::failure()
    }

    /// Try to relocalize after tracking lost
    fn relocalize(&mut self, keypoints: Vec<KeyPoint>, descriptors: Descriptors) -> VOResult {
        // Similar to initialize, but start fresh
        self.prev_keypoints = keypoints;
        self.prev_descriptors = descriptors;
        self.state = VOState::Initializing;
        
        VOResult {
            pose: SE3::identity(),
            num_matches: 0,
            num_inliers: 0,
            success: true,
        }
    }

    /// Get current state
    pub fn state(&self) -> VOState {
        self.state
    }

    /// Reset the VO
    pub fn reset(&mut self) {
        self.state = VOState::NotInitialized;
        self.prev_frame_id = None;
        self.prev_frame_pose = None;
        self.prev_keypoints.clear();
        self.prev_descriptors = Descriptors::new();
        self.prev_3d_points.clear();
        self.frame_count = 0;
    }

    /// Set minimum matches threshold
    pub fn set_min_matches(&mut self, min: usize) {
        self.min_matches = min;
    }

    /// Set minimum inliers threshold
    pub fn set_min_inliers(&mut self, min: usize) {
        self.min_inliers = min;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vo_state() {
        assert_eq!(VOState::NotInitialized, VOState::NotInitialized);
        assert_ne!(VOState::TrackingOk, VOState::TrackingLost);
    }

    #[test]
    fn test_vo_result() {
        let result = VOResult {
            pose: SE3::identity(),
            num_matches: 100,
            num_inliers: 50,
            success: true,
        };
        
        assert!(result.success);
        assert_eq!(result.num_inliers, 50);
    }
}
