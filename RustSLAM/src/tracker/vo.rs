//! Visual Odometry

use crate::config::{FeatureType, TrackerParams};
use crate::core::{Camera, FrameFeatures, SE3};
use crate::features::{
    FeatureExtractor,
    FeatureMatcher,
    KeyPoint,
    Descriptors,
    OrbExtractor,
    HammingMatcher,
    HarrisExtractor,
    HarrisParams,
    FastExtractor,
    FastParams,
};
use crate::tracker::solver::{PnPSolver, EssentialSolver, Triangulator};

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
    /// Create a new VO with default feature extraction
    pub fn new(camera: Camera) -> Self {
        Self::with_params(camera, TrackerParams::default())
    }

    /// Create a new VO with tracker parameters
    pub fn with_params(camera: Camera, params: TrackerParams) -> Self {
        let extractor: Box<dyn FeatureExtractor> = match params.feature_type {
            FeatureType::Orb => Box::new(OrbExtractor::new(params.max_features)),
            FeatureType::Harris => Box::new(HarrisExtractor::new(params.max_features, HarrisParams::default())),
            FeatureType::Fast => Box::new(FastExtractor::new(params.max_features, FastParams::default())),
        };
        // All feature types now produce binary BRIEF descriptors → use Hamming distance
        let matcher: Box<dyn FeatureMatcher> = Box::new(
            HammingMatcher::new(2).with_ratio_threshold(params.match_ratio as f64)
        );

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
            min_matches: params.min_matches,
            min_inliers: params.min_inliers,
        }
    }

    /// Create a new VO with custom extractor and matcher
    pub fn with_extractor(
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
            min_matches: 50,
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

        // Extract matched points
        let mut pts1 = Vec::with_capacity(matches.len());
        let mut pts2 = Vec::with_capacity(matches.len());
        for m in &matches {
            let train_idx = m.train_idx as usize;
            let query_idx = m.query_idx as usize;
            if train_idx < self.prev_keypoints.len() && query_idx < keypoints.len() {
                pts1.push([self.prev_keypoints[train_idx].x(), self.prev_keypoints[train_idx].y()]);
                pts2.push([keypoints[query_idx].x(), keypoints[query_idx].y()]);
            }
        }

        // Compute essential matrix
        if let Some((E, inliers)) = self.essential_solver.compute(&matches, &pts1, &pts2) {
            let inlier_count = inliers.iter().filter(|&&x| x).count();
            
            if inlier_count >= self.min_inliers && matches.len() >= self.min_matches {
                // Recover pose from essential matrix
                let poses = self.essential_solver.recover_pose(E);
                let prev_pose = SE3::identity();
                let mut best_pose = poses[0];
                let mut best_points = Vec::new();
                let mut best_count = 0usize;

                for pose in poses {
                    let triangulated = self.triangulator.triangulate(
                        &prev_pose,
                        &pose,
                        &pts1,
                        &pts2,
                    );
                    let count = triangulated.iter().filter(|pt| pt.is_some()).count();
                    if count > best_count {
                        best_count = count;
                        best_pose = pose;
                        best_points = triangulated;
                    }
                }

                // Map triangulated points to current keypoint indices
                // (process_frame stores current keypoints as prev_keypoints)
                let mut kp_3d_points = vec![None; keypoints.len()];
                for (i, m) in matches.iter().enumerate() {
                    let query_idx = m.query_idx as usize;
                    if query_idx < keypoints.len() && i < best_points.len() {
                        kp_3d_points[query_idx] = best_points[i];
                    }
                }
                self.prev_3d_points = kp_3d_points;

                self.state = VOState::TrackingOk;
                
                return VOResult {
                    pose: best_pose,
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
                // Update prev_3d_points: propagate existing + triangulate new
                let prev_pose = self.prev_frame_pose.unwrap_or(SE3::identity());
                let mut new_3d_points = vec![None; keypoints.len()];

                // Collect matches needing triangulation
                let mut tri_prev_pts = Vec::new();
                let mut tri_curr_pts = Vec::new();
                let mut tri_query_indices = Vec::new();

                for m in &matches {
                    let query_idx = m.query_idx as usize;
                    let train_idx = m.train_idx as usize;
                    if query_idx >= keypoints.len() { continue; }

                    // Propagate existing 3D point through match
                    if train_idx < self.prev_3d_points.len() {
                        if let Some(pt3d) = self.prev_3d_points[train_idx] {
                            new_3d_points[query_idx] = Some(pt3d);
                            continue;
                        }
                    }
                    // No existing 3D point — queue for triangulation
                    if train_idx < self.prev_keypoints.len() {
                        tri_prev_pts.push([self.prev_keypoints[train_idx].x(),
                                           self.prev_keypoints[train_idx].y()]);
                        tri_curr_pts.push([keypoints[query_idx].x(),
                                           keypoints[query_idx].y()]);
                        tri_query_indices.push(query_idx);
                    }
                }

                // Triangulate new points from two-view geometry
                if !tri_prev_pts.is_empty() {
                    let triangulated = self.triangulator.triangulate(
                        &prev_pose, &pose, &tri_prev_pts, &tri_curr_pts,
                    );
                    for (i, pt) in triangulated.into_iter().enumerate() {
                        if let Some(p) = pt {
                            new_3d_points[tri_query_indices[i]] = Some(p);
                        }
                    }
                }

                self.prev_3d_points = new_3d_points;

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
        // First try a two-view re-initialization against the last retained frame.
        let attempt = if !self.prev_keypoints.is_empty() && !self.prev_descriptors.is_empty() {
            self.initialize(keypoints.clone(), descriptors.clone())
        } else {
            VOResult::failure()
        };
        if attempt.success {
            return attempt;
        }

        // If re-initialization failed, stage the current frame as the new seed.
        self.prev_keypoints = keypoints;
        self.prev_descriptors = descriptors;
        self.prev_3d_points.clear();
        self.prev_frame_pose = None;
        self.state = VOState::Initializing;

        VOResult::failure()
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

    /// Get features from the last successfully processed frame.
    pub fn last_features(&self) -> Option<FrameFeatures> {
        if self.prev_keypoints.is_empty() || self.prev_descriptors.is_empty() {
            return None;
        }

        Some(FrameFeatures {
            keypoints: self
                .prev_keypoints
                .iter()
                .map(|kp| [kp.x(), kp.y()])
                .collect(),
            descriptors: self.prev_descriptors.data.clone(),
            map_points: vec![None; self.prev_keypoints.len()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::base::{FeatureError, Match};

    struct TestExtractor {
        keypoints: Vec<KeyPoint>,
        descriptors: Descriptors,
    }

    impl FeatureExtractor for TestExtractor {
        fn detect_and_compute(
            &mut self,
            _image: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
            Ok((self.keypoints.clone(), self.descriptors.clone()))
        }

        fn detect(
            &mut self,
            _image: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<Vec<KeyPoint>, FeatureError> {
            Ok(self.keypoints.clone())
        }

        fn num_features(&self) -> usize {
            self.keypoints.len()
        }

        fn set_num_features(&mut self, _num: usize) {}
    }

    struct TestMatcher;

    impl FeatureMatcher for TestMatcher {
        fn match_descriptors(
            &self,
            _query: &Descriptors,
            _train: &Descriptors,
        ) -> Result<Vec<Match>, FeatureError> {
            Ok(Vec::new())
        }
    }

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

    #[test]
    fn test_relocalize_failure_does_not_report_success() {
        let camera = Camera::new(525.0, 525.0, 319.5, 239.5, 640, 480);
        let current_kp = KeyPoint::new(10.0, 10.0);
        let mut current_desc = Descriptors::with_capacity(1, 32);
        current_desc.data[0] = 1;

        let extractor = Box::new(TestExtractor {
            keypoints: vec![current_kp.clone()],
            descriptors: current_desc.clone(),
        });
        let matcher = Box::new(TestMatcher);
        let mut vo = VisualOdometry::with_extractor(extractor, matcher, camera);

        vo.state = VOState::TrackingLost;
        vo.prev_keypoints = vec![KeyPoint::new(5.0, 5.0)];
        vo.prev_descriptors = Descriptors::with_capacity(1, 32);
        vo.prev_descriptors.data[0] = 2;
        vo.prev_frame_pose = Some(SE3::identity());
        vo.prev_3d_points = vec![Some([0.0, 0.0, 1.0])];

        let result = vo.process_frame(&vec![0; 640 * 480], 640, 480);

        assert!(!result.success);
        assert_eq!(vo.state(), VOState::Initializing);
        assert_eq!(vo.prev_keypoints.len(), 1);
        assert_eq!(vo.prev_keypoints[0].x(), current_kp.x());
        assert!(vo.prev_frame_pose.is_none());
        assert!(vo.prev_3d_points.is_empty());
    }
}
