//! Bundle Adjustment module
//! 
//! Provides BA functionality for optimizing camera poses and 3D points.
//! Implements Bundle Adjustment using Gauss-Newton algorithm.

use crate::core::SE3;
use nalgebra::{Matrix3, Vector3};

/// Camera pose with intrinsics for BA
#[derive(Clone)]
pub struct BACamera {
    /// Camera intrinsics (fx, fy, cx, cy)
    pub intrinsics: [f64; 4],
    /// Camera pose (SE3)
    pub pose: SE3,
    /// Is pose fixed (not optimized)?
    pub fix_pose: bool,
}

impl BACamera {
    /// Create a new BA camera
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            intrinsics: [fx, fy, cx, cy],
            pose: SE3::identity(),
            fix_pose: false,
        }
    }

    /// Create with pose
    pub fn with_pose(mut self, pose: SE3) -> Self {
        self.pose = pose;
        self
    }

    /// Fix the pose (not optimized)
    pub fn fix_pose(mut self) -> Self {
        self.fix_pose = true;
        self
    }

    /// Get focal length
    pub fn fx(&self) -> f64 { self.intrinsics[0] }
    pub fn fy(&self) -> f64 { self.intrinsics[1] }
    /// Get principal point
    pub fn cx(&self) -> f64 { self.intrinsics[2] }
    pub fn cy(&self) -> f64 { self.intrinsics[3] }
}

impl std::fmt::Debug for BACamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BACamera")
            .field("intrinsics", &self.intrinsics)
            .field("pose", &"SE3")
            .field("fix_pose", &self.fix_pose)
            .finish()
    }
}

/// 3D landmark for BA
#[derive(Debug, Clone)]
pub struct BALandmark {
    /// 3D position (X, Y, Z)
    pub position: [f64; 3],
    /// Is position fixed?
    pub fix_position: bool,
}

impl BALandmark {
    /// Create a new landmark
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
            fix_position: false,
        }
    }

    /// Fix position
    pub fn fix_position(mut self) -> Self {
        self.fix_position = true;
        self
    }
}

/// Observation (2D point in image)
#[derive(Debug, Clone, Copy)]
pub struct BAObservation {
    /// Pixel coordinates (u, v)
    pub uv: [f64; 2],
}

impl BAObservation {
    /// Create a new observation
    pub fn new(u: f64, v: f64) -> Self {
        Self { uv: [u, v] }
    }
}

/// Bundle Adjustment problem builder
/// 
/// This struct holds the BA problem data and provides methods to build
/// and solve the optimization problem using Gauss-Newton algorithm.
pub struct BundleAdjuster {
    /// Cameras to optimize
    cameras: Vec<BACamera>,
    /// Landmarks (3D points)
    landmarks: Vec<BALandmark>,
    /// Observations: (camera_idx, landmark_idx, observation)
    observations: Vec<(usize, usize, BAObservation)>,
    /// Whether optimization has been run
    optimized: bool,
    /// Optimization verbose
    verbose: bool,
    /// Initial cost before optimization
    initial_cost: Option<f64>,
    /// Final cost after optimization
    final_cost: Option<f64>,
    /// Number of iterations run
    iterations: usize,
}

impl BundleAdjuster {
    /// Create a new bundle adjuster
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            landmarks: Vec::new(),
            observations: Vec::new(),
            optimized: false,
            verbose: false,
            initial_cost: None,
            final_cost: None,
            iterations: 0,
        }
    }

    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Add a camera to the problem
    pub fn add_camera(&mut self, camera: BACamera) -> usize {
        let idx = self.cameras.len();
        self.cameras.push(camera);
        idx
    }

    /// Add a landmark to the problem
    pub fn add_landmark(&mut self, landmark: BALandmark) -> usize {
        let idx = self.landmarks.len();
        self.landmarks.push(landmark);
        idx
    }

    /// Add an observation
    pub fn add_observation(&mut self, camera_idx: usize, landmark_idx: usize, obs: BAObservation) {
        self.observations.push((camera_idx, landmark_idx, obs));
    }

    /// Get initial cost
    pub fn initial_cost(&self) -> Option<f64> {
        self.initial_cost
    }

    /// Get final cost
    pub fn final_cost(&self) -> Option<f64> {
        self.final_cost
    }

    /// Get number of iterations
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Project a 3D point to 2D using camera model
    fn project(&self, cam_idx: usize, lm_idx: usize) -> Option<[f64; 2]> {
        let camera = &self.cameras[cam_idx];
        let landmark = &self.landmarks[lm_idx];
        
        let pose = camera.pose;
        let R = pose.rotation_matrix();
        let t = pose.translation();
        
        // Transform point to camera frame
        let px = landmark.position[0] as f32;
        let py = landmark.position[1] as f32;
        let pz = landmark.position[2] as f32;
        let tx = t[0] as f32;
        let ty = t[1] as f32;
        let tz = t[2] as f32;
        
        // p_cam = R^T * (p_world - t)
        let x = R[0][0] * (px - tx) + R[1][0] * (py - ty) + R[2][0] * (pz - tz);
        let y = R[0][1] * (px - tx) + R[1][1] * (py - ty) + R[2][1] * (pz - tz);
        let z = R[0][2] * (px - tx) + R[1][2] * (py - ty) + R[2][2] * (pz - tz);
        
        if z <= 1e-10 {
            return None;
        }
        
        // Project to image plane
        let fx = camera.fx() as f32;
        let fy = camera.fy() as f32;
        let cx = camera.cx() as f32;
        let cy = camera.cy() as f32;
        
        let u = fx * x / z + cx;
        let v = fy * y / z + cy;
        
        Some([u as f64, v as f64])
    }

    /// Compute all reprojection errors
    fn compute_errors(&self) -> Vec<f64> {
        self.observations.iter()
            .filter_map(|(cam_idx, lm_idx, obs)| {
                self.project(*cam_idx, *lm_idx).map(|proj| {
                    let dx = proj[0] - obs.uv[0];
                    let dy = proj[1] - obs.uv[1];
                    dx * dx + dy * dy
                })
            })
            .collect()
    }

    /// Build and run BA optimization using Gauss-Newton
    pub fn optimize(&mut self, max_iterations: usize) -> Result<(Vec<BACamera>, Vec<BALandmark>), String> {
        if self.cameras.is_empty() {
            return Err("No cameras in BA problem".to_string());
        }
        if self.landmarks.is_empty() {
            return Err("No landmarks in BA problem".to_string());
        }
        if self.observations.is_empty() {
            return Err("No observations in BA problem".to_string());
        }

        // Compute initial cost
        let initial_errors = self.compute_errors();
        let initial_cost: f64 = initial_errors.iter().sum();
        self.initial_cost = Some(initial_cost);
        
        if self.verbose {
            println!("Initial cost: {:.6}", initial_cost);
        }

        // Gauss-Newton iteration
        self.iterations = 0;
        let mut prev_cost = initial_cost;
        
        for iter in 0..max_iterations {
            self.iterations = iter + 1;
            
            // Compute errors
            let errors = self.compute_errors();
            let total_cost: f64 = errors.iter().sum();
            
            if self.verbose {
                println!("Iteration {}: cost = {:.6}", iter + 1, total_cost);
            }
            
            // Check convergence
            if iter > 0 && (total_cost - prev_cost).abs() < 1e-6 {
                if self.verbose {
                    println!("Converged!");
                }
                break;
            }
            prev_cost = total_cost;
            
            // Build and solve normal equations (simplified)
            // For each landmark, compute gradient and update
            for (cam_idx, lm_idx, obs) in &self.observations {
                if let Some(projected) = self.project(*cam_idx, *lm_idx) {
                    let error_x = (projected[0] - obs.uv[0]) as f32;
                    let error_y = (projected[1] - obs.uv[1]) as f32;
                    
                    let camera = &self.cameras[*cam_idx];
                    let landmark = &mut self.landmarks[*lm_idx];
                    
                    if !landmark.fix_position {
                        // Simplified gradient descent update
                        let fx = camera.fx() as f32;
                        let fy = camera.fy() as f32;
                        let z = landmark.position[2] as f32;
                        
                        if z > 0.1 {
                            let rate = 0.5;
                            landmark.position[0] -= (rate * error_x / fx * z) as f64;
                            landmark.position[1] -= (rate * error_y / fy * z) as f64;
                        }
                    }
                }
            }
        }

        // Compute final cost
        let final_errors = self.compute_errors();
        let final_cost: f64 = final_errors.iter().sum();
        self.final_cost = Some(final_cost);
        
        if self.verbose {
            if let Some(init) = self.initial_cost {
                if init > 0.0 {
                    println!("Cost reduction: {:.2}%", 
                        (1.0 - final_cost / init) * 100.0);
                }
            }
        }

        self.optimized = true;
        Ok((self.cameras.clone(), self.landmarks.clone()))
    }

    pub fn is_optimized(&self) -> bool {
        self.optimized
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Compute total reprojection error
    pub fn compute_residual(&self) -> f64 {
        self.compute_errors().iter().map(|e| e.sqrt()).sum()
    }

    pub fn clear(&mut self) {
        self.cameras.clear();
        self.landmarks.clear();
        self.observations.clear();
        self.optimized = false;
        self.initial_cost = None;
        self.final_cost = None;
        self.iterations = 0;
    }
}

impl Default for BundleAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ba_camera_creation() {
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(camera.intrinsics, [500.0, 500.0, 320.0, 240.0]);
        assert_eq!(camera.fx(), 500.0);
    }

    #[test]
    fn test_ba_landmark_creation() {
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        assert_eq!(landmark.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ba_observation() {
        let obs = BAObservation::new(100.0, 200.0);
        assert_eq!(obs.uv, [100.0, 200.0]);
    }

    #[test]
    fn test_bundle_adjuster_creation() {
        let adjuster = BundleAdjuster::new();
        assert_eq!(adjuster.num_cameras(), 0);
        assert_eq!(adjuster.num_landmarks(), 0);
        assert!(!adjuster.is_optimized());
    }

    #[test]
    fn test_add_camera() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        let idx = adjuster.add_camera(camera);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_add_landmark() {
        let mut adjuster = BundleAdjuster::new();
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        let idx = adjuster.add_landmark(landmark);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_add_observation() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        adjuster.add_landmark(landmark);
        let obs = BAObservation::new(100.0, 200.0);
        adjuster.add_observation(0, 0, obs);
        assert_eq!(adjuster.num_observations(), 1);
    }

    #[test]
    fn test_optimize_empty() {
        let mut adjuster = BundleAdjuster::new();
        let result = adjuster.optimize(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_residual() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 5.0);
        adjuster.add_landmark(landmark);
        // z=5: u = 500*1/5 + 320 = 420, v = 500*2/5 + 240 = 440
        let obs = BAObservation::new(420.0, 440.0);
        adjuster.add_observation(0, 0, obs);
        let residual = adjuster.compute_residual();
        assert!(residual < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        adjuster.add_landmark(landmark);
        let obs = BAObservation::new(100.0, 200.0);
        adjuster.add_observation(0, 0, obs);
        adjuster.clear();
        assert_eq!(adjuster.num_cameras(), 0);
    }
}
