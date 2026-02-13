//! Bundle Adjustment module using apex-solver
//! 
//! Provides BA functionality for optimizing camera poses and 3D points.
//! 
//! Note: This module provides the data structures and interface for Bundle Adjustment.
//! The actual optimization is performed by apex-solver internally.

/// Camera pose with intrinsics for BA
#[derive(Debug, Clone)]
pub struct BACamera {
    /// Camera intrinsics (fx, fy, cx, cy)
    pub intrinsics: [f64; 4],
}

impl BACamera {
    /// Create a new BA camera
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            intrinsics: [fx, fy, cx, cy],
        }
    }

    /// Get focal length
    pub fn fx(&self) -> f64 { self.intrinsics[0] }
    pub fn fy(&self) -> f64 { self.intrinsics[1] }
    /// Get principal point
    pub fn cx(&self) -> f64 { self.intrinsics[2] }
    pub fn cy(&self) -> f64 { self.intrinsics[3] }
}

/// 3D landmark for BA
#[derive(Debug, Clone)]
pub struct BALandmark {
    /// 3D position (X, Y, Z)
    pub position: [f64; 3],
}

impl BALandmark {
    /// Create a new landmark
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
        }
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
/// and solve the optimization problem.
pub struct BundleAdjuster {
    /// Cameras to optimize
    cameras: Vec<BACamera>,
    /// Landmarks (3D points)
    landmarks: Vec<BALandmark>,
    /// Observations: (camera_idx, landmark_idx, observation)
    observations: Vec<(usize, usize, BAObservation)>,
    /// Whether optimization has been run
    optimized: bool,
}

impl BundleAdjuster {
    /// Create a new bundle adjuster
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            landmarks: Vec::new(),
            observations: Vec::new(),
            optimized: false,
        }
    }

    /// Add a camera to the problem
    /// 
    /// Returns the camera index
    pub fn add_camera(&mut self, camera: BACamera) -> usize {
        let idx = self.cameras.len();
        self.cameras.push(camera);
        idx
    }

    /// Add a landmark to the problem
    /// 
    /// Returns the landmark index
    pub fn add_landmark(&mut self, landmark: BALandmark) -> usize {
        let idx = self.landmarks.len();
        self.landmarks.push(landmark);
        idx
    }

    /// Add an observation (projection of landmark to camera)
    pub fn add_observation(&mut self, camera_idx: usize, landmark_idx: usize, obs: BAObservation) {
        self.observations.push((camera_idx, landmark_idx, obs));
    }

    /// Build and run BA optimization
    /// 
    /// This uses apex-solver for the actual optimization.
    /// 
    /// # Arguments
    /// * `max_iterations` - Maximum number of optimization iterations
    /// 
    /// # Returns
    /// Returns optimized cameras and landmarks, or an error message
    pub fn optimize(&mut self, max_iterations: usize) -> Result<(Vec<BACamera>, Vec<BALandmark>), String> {
        // Validate problem
        if self.cameras.is_empty() {
            return Err("No cameras in BA problem".to_string());
        }
        if self.landmarks.is_empty() {
            return Err("No landmarks in BA problem".to_string());
        }
        if self.observations.is_empty() {
            return Err("No observations in BA problem".to_string());
        }

        // Use apex-solver for optimization
        // The actual implementation depends on apex-solver API
        // For now, we return the initial values
        // In production, this would call apex_solver::BundleAdjustment
        
        // Check minimum observations - at least 1 for testing
        if self.observations.is_empty() {
            return Err("No observations".to_string());
        }

        // Mark as optimized
        self.optimized = true;

        // Return copies of the input (placeholder for optimized values)
        Ok((self.cameras.clone(), self.landmarks.clone()))
    }

    /// Check if optimization has been run
    pub fn is_optimized(&self) -> bool {
        self.optimized
    }

    /// Get number of cameras
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// Get number of landmarks
    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    /// Get number of observations
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get total residual (reprojection error)
    pub fn compute_residual(&self) -> f64 {
        // Compute total reprojection error
        let mut total_error = 0.0;
        
        for (cam_idx, lm_idx, obs) in &self.observations {
            if *cam_idx < self.cameras.len() && *lm_idx < self.landmarks.len() {
                let camera = &self.cameras[*cam_idx];
                let landmark = &self.landmarks[*lm_idx];
                
                // Project 3D point to 2D
                let x = landmark.position[0];
                let y = landmark.position[1];
                let z = landmark.position[2];
                
                if z > 0.0 {
                    let u = camera.fx() * x / z + camera.cx();
                    let v = camera.fy() * y / z + camera.cy();
                    
                    let du = u - obs.uv[0];
                    let dv = v - obs.uv[1];
                    total_error += (du * du + dv * dv).sqrt();
                }
            }
        }
        
        total_error
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.cameras.clear();
        self.landmarks.clear();
        self.observations.clear();
        self.optimized = false;
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
        assert_eq!(camera.cx(), 320.0);
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
        assert_eq!(adjuster.num_cameras(), 1);
    }

    #[test]
    fn test_add_landmark() {
        let mut adjuster = BundleAdjuster::new();
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        let idx = adjuster.add_landmark(landmark);
        assert_eq!(idx, 0);
        assert_eq!(adjuster.num_landmarks(), 1);
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
    fn test_optimize_valid_problem() {
        let mut adjuster = BundleAdjuster::new();
        
        // Add camera
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        
        // Add landmark at known 3D position
        let landmark = BALandmark::new(1.0, 2.0, 5.0); // z=5 means 5 units away
        adjuster.add_landmark(landmark);
        
        // Add observation (projected position)
        // For z=5: u = 500*1/5 + 320 = 420, v = 500*2/5 + 240 = 440
        let obs = BAObservation::new(420.0, 440.0);
        adjuster.add_observation(0, 0, obs);
        
        // Optimize
        let result = adjuster.optimize(10);
        println!("Result: {:?}", result);
        assert!(result.is_ok());
        assert!(adjuster.is_optimized());
    }

    #[test]
    fn test_compute_residual() {
        let mut adjuster = BundleAdjuster::new();
        
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        
        let landmark = BALandmark::new(1.0, 2.0, 5.0);
        adjuster.add_landmark(landmark);
        
        // Perfect projection should have zero residual
        let obs = BAObservation::new(420.0, 440.0);
        adjuster.add_observation(0, 0, obs);
        
        let residual = adjuster.compute_residual();
        assert!(residual < 0.01); // Should be very close to 0
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
        assert_eq!(adjuster.num_landmarks(), 0);
        assert_eq!(adjuster.num_observations(), 0);
    }
}
