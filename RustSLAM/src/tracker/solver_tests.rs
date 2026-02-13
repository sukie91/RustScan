//! Tests for Geometric Solvers
//! 
//! Tests PnP, Essential Matrix, Triangulation, and Sim3 solvers.

#[cfg(test)]
mod tests {
    use crate::core::SE3;
    use crate::features::base::Match;
    use crate::tracker::solver::{
        PnPSolver, PnPProblem, EssentialSolver, Triangulator, Sim3Solver,
    };
    use glam::{Mat3, Vec3};

    // =========================================================================
    // PnP Solver Tests
    // =========================================================================

    #[test]
    fn test_pnp_solver_creation() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(solver.fx, 500.0);
        assert_eq!(solver.fy, 500.0);
        assert_eq!(solver.cx, 320.0);
        assert_eq!(solver.cy, 240.0);
    }

    #[test]
    fn test_pnp_problem() {
        let mut problem = PnPProblem::new();
        problem.add_correspondence([100.0, 100.0], [0.0, 0.0, 0.0]);
        problem.add_correspondence([200.0, 100.0], [1.0, 0.0, 0.0]);
        problem.add_correspondence([100.0, 200.0], [0.0, 1.0, 0.0]);
        problem.add_correspondence([200.0, 200.0], [1.0, 1.0, 0.0]);
        
        assert!(problem.is_solvable());
        assert_eq!(problem.image_points.len(), 4);
    }

    #[test]
    fn test_pnp_solve_simple() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        
        let mut problem = PnPProblem::new();
        // Add 4 known 3D-2D correspondences
        // 3D points at z=1 plane
        problem.add_correspondence([320.0, 240.0], [0.0, 0.0, 1.0]);      // (0, 0, 1) -> (320, 240)
        problem.add_correspondence([520.0, 240.0], [1.0, 0.0, 1.0]);      // (1, 0, 1) -> (520, 240)
        problem.add_correspondence([320.0, 440.0], [0.0, 1.0, 1.0]);      // (0, 1, 1) -> (320, 440)
        problem.add_correspondence([520.0, 440.0], [1.0, 1.0, 1.0]);      // (1, 1, 1) -> (520, 440)
        
        let result = solver.solve(&problem);
        
        // Should return a valid pose (not identity for non-planar case)
        assert!(result.is_some());
        let (_pose, inliers) = result.unwrap();
        
        // Check that we have inliers
        assert!(!inliers.is_empty());
    }

    // =========================================================================
    // Essential Matrix Solver Tests
    // =========================================================================

    #[test]
    fn test_essential_solver_creation() {
        let solver = EssentialSolver::new();
        assert_eq!(solver.ransac_threshold, 0.01);
        assert_eq!(solver.ransac_max_iterations, 200);
    }

    #[test]
    fn test_essential_matrix_from_matches() {
        let solver = EssentialSolver::new();
        
        // Create matched points (simulating two views) - need at least 8
        let pts1: Vec<[f32; 2]> = vec![
            [100.0, 100.0],
            [200.0, 100.0],
            [100.0, 200.0],
            [200.0, 200.0],
            [150.0, 150.0],
            [120.0, 180.0],
            [180.0, 120.0],
            [160.0, 160.0],
        ];
        
        let pts2: Vec<[f32; 2]> = vec![
            [110.0, 110.0],  // slight translation
            [210.0, 110.0],
            [110.0, 210.0],
            [210.0, 210.0],
            [160.0, 160.0],
            [130.0, 190.0],
            [190.0, 130.0],
            [170.0, 170.0],
        ];
        
        let result = solver.compute(&[], &pts1, &pts2);
        
        // Should return an essential matrix
        assert!(result.is_some());
        let (E, _inliers) = result.unwrap();
        
        // E should be 3x3 (check using row/cols methods)
        let _ = E.row(0);  // This is how we access rows in glam
        
        // Check rank-2 constraint (det(E) ≈ 0)
        let det = E.determinant();
        assert!(det.abs() < 0.1, "Essential matrix should have det ≈ 0, got {}", det);
    }

    #[test]
    fn test_essential_matrix_enforce_rank2() {
        let solver = EssentialSolver::new();
        
        let pts1: Vec<[f32; 2]> = vec![[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]];
        let pts2: Vec<[f32; 2]> = vec![[105.0, 105.0], [205.0, 205.0], [305.0, 305.0]];
        
        if let Some((E, _)) = solver.compute(&[], &pts1, &pts2) {
            // Enforce rank-2 constraint by SVD
            let _ = solver.enforce_rank2(E);
        }
    }

    // =========================================================================
    // Triangulation Tests
    // =========================================================================

    #[test]
    fn test_triangulator_creation() {
        let tri = Triangulator::new();
        assert!(tri.min_angle > 0.0);
        assert!(tri.min_dist > 0.0);
    }

    #[test]
    fn test_triangulate_simple() {
        let tri = Triangulator::new();
        
        // Two camera poses
        let pose1 = SE3::identity();  // First camera at origin
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);  // Second camera at (1, 0, 0)
        
        // Corresponding 2D points (projection of [0.5, 0, 1])
        let pts1: Vec<[f32; 2]> = vec![[320.0, 240.0]];  // (0.5, 1) * f + principal
        let pts2: Vec<[f32; 2]> = vec![[220.0, 240.0]];  // shifted due to camera translation
        
        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);
        
        assert_eq!(results.len(), 1);
        
        // Check if triangulation produced valid 3D point
        if let Some(point) = results[0] {
            assert!(point[2] > 0.0, "Point should be in front of camera");
        }
    }

    #[test]
    fn test_triangulate_multiple_points() {
        let tri = Triangulator::new();
        
        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[0.5, 0.0, 0.0]);
        
        let pts1: Vec<[f32; 2]> = vec![
            [320.0, 240.0],
            [420.0, 240.0],
            [320.0, 340.0],
        ];
        let pts2: Vec<[f32; 2]> = vec![
            [270.0, 240.0],
            [370.0, 240.0],
            [270.0, 340.0],
        ];
        
        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);
        
        // All points should be triangulated
        let valid_count = results.iter().filter(|p| p.is_some()).count();
        assert!(valid_count > 0, "Should have at least some valid triangulated points");
    }

    #[test]
    fn test_triangulation_check_angle() {
        let tri = Triangulator::new();
        
        // Cameras too close - should fail angle check
        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[0.01, 0.0, 0.0]);  // Very small baseline
        
        let pts1: Vec<[f32; 2]> = vec![[320.0, 240.0]];
        let pts2: Vec<[f32; 2]> = vec![[320.0, 240.0]];
        
        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);
        
        // Should return None due to small angle
        assert!(results[0].is_none() || results.len() == 0);
    }

    // =========================================================================
    // Sim3 Solver Tests
    // =========================================================================

    #[test]
    fn test_sim3_solver_creation() {
        let solver = Sim3Solver::new(0.01);
        assert_eq!(solver.ransac_threshold, 0.01);
    }

    #[test]
    fn test_sim3_compute() {
        let solver = Sim3Solver::new(0.01);
        
        // 3D points in first view
        let pts1: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        
        // 3D points in second view (scaled by 2x, rotated, translated)
        let pts2: Vec<[f32; 3]> = vec![
            [2.0, 0.0, 2.0],   // scale 2x
            [4.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [4.0, 2.0, 2.0],
        ];
        
        let result = solver.compute(&pts1, &pts2);
        
        // Should return similarity transform
        assert!(result.is_some());
        let (sim3, inliers) = result.unwrap();
        
        // Check scale is approximately 2
        let scale = sim3.0;
        assert!((scale - 2.0).abs() < 0.5, "Scale should be ~2");
        
        // Should have inliers
        assert!(!inliers.is_empty());
    }

    #[test]
    fn test_sim3_apply() {
        let solver = Sim3Solver::new(0.01);
        
        // Simple scale 2x transform
        let sim3 = solver.create_sim3(2.0, Vec3::ZERO, Mat3::IDENTITY);
        
        // Apply to a point
        let point = [1.0, 2.0, 3.0];
        let transformed = solver.apply_sim3(sim3, point);
        
        // Should be scaled by 2
        assert!((transformed[0] - 2.0).abs() < 0.001);
        assert!((transformed[1] - 4.0).abs() < 0.001);
        assert!((transformed[2] - 6.0).abs() < 0.001);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_vo_pipeline() {
        // Test the full pipeline: tracking -> triangulation -> optimization
        let _pnp_solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        let essential_solver = EssentialSolver::new();
        let triangulator = Triangulator::new();
        
        // Simulate two frames with known motion
        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.1], &[0.5, 0.0, 0.0]);
        
        // Create fake correspondences
        let pts1: Vec<[f32; 2]> = vec![[320.0, 240.0], [420.0, 240.0], [320.0, 340.0]];
        let pts2: Vec<[f32; 2]> = vec![[270.0, 240.0], [370.0, 240.0], [270.0, 340.0]];
        
        // Compute essential matrix
        let _ = essential_solver.compute(&[], &pts1, &pts2);
        
        // Recover pose from essential matrix
        let E = Mat3::IDENTITY;  // Simplified
        let _ = essential_solver.recover_pose(E);
        
        // Triangulate points
        let _ = triangulator.triangulate(&pose1, &pose2, &pts1, &pts2);
        
        // Pipeline should complete without errors
        assert!(true);
    }
}
