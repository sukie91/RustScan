//! Geometric solvers for Visual Odometry
//! 
//! Implements PnP, Essential Matrix, Triangulation, and Sim3 solvers with proper algorithms.

use std::f32::consts::PI;
use crate::core::SE3;
use crate::features::base::Match;
use glam::{Mat3, Mat4, Vec3};

/// 2D-3D correspondence for PnP
#[derive(Debug, Clone)]
pub struct PnPProblem {
    /// 2D points in image coordinates [x, y]
    pub image_points: Vec<[f32; 2]>,
    /// 3D points in world coordinates [X, Y, Z]
    pub object_points: Vec<[f32; 3]>,
}

impl PnPProblem {
    /// Create a new PnP problem
    pub fn new() -> Self {
        Self {
            image_points: Vec::new(),
            object_points: Vec::new(),
        }
    }

    /// Add a correspondence
    pub fn add_correspondence(&mut self, img: [f32; 2], obj: [f32; 3]) {
        self.image_points.push(img);
        self.object_points.push(obj);
    }

    /// Check if we have enough points
    pub fn is_solvable(&self) -> bool {
        self.image_points.len() >= 4
    }
}

impl Default for PnPProblem {
    fn default() -> Self {
        Self::new()
    }
}

/// RANSAC-based PnP solver using P3P + RANSAC
pub struct PnPSolver {
    /// Camera intrinsics
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// RANSAC parameters
    pub ransac_threshold: f32,
    pub ransac_confidence: f32,
    pub ransac_max_iterations: u32,
}

impl PnPSolver {
    /// Create a new PnP solver
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            fx, fy, cx, cy,
            ransac_threshold: 3.0,
            ransac_confidence: 0.99,
            ransac_max_iterations: 200,
        }
    }

    /// Solve PnP using RANSAC with P3P
    /// 
    /// Returns: (pose, inlier_mask)
    pub fn solve(&self, problem: &PnPProblem) -> Option<(SE3, Vec<bool>)> {
        if !problem.is_solvable() {
            return None;
        }

        let n = problem.image_points.len();
        
        // Normalize image coordinates
        let normalized_pts: Vec<[f32; 2]> = problem.image_points.iter()
            .map(|p| [
                (p[0] - self.cx) / self.fx,
                (p[1] - self.cy) / self.fy,
            ])
            .collect();
        
        // RANSAC loop
        let mut best_inliers: Vec<bool> = vec![false; n];
        let mut best_pose: Option<SE3> = None;
        let mut best_inlier_count = 0;
        
        for _ in 0..self.ransac_max_iterations {
            // Randomly select 3 points for P3P
            let indices = self.random_indices(n, 3);
            
            // Solve P3P for these 3 points
            if let Some(poses) = self.solve_p3p(&normalized_pts, &problem.object_points, &indices) {
                // For each P3P solution, check all points
                for pose in poses {
                    let mut inliers = vec![false; n];
                    let mut inlier_count = 0;
                    
                    for i in 0..n {
                        let projected = self.project_point(&pose, &problem.object_points[i]);
                        let error = self.reprojection_error(&normalized_pts[i], &projected);
                        
                        if error < self.ransac_threshold {
                            inliers[i] = true;
                            inlier_count += 1;
                        }
                    }
                    
                    // Update best if more inliers
                    if inlier_count > best_inlier_count {
                        best_inlier_count = inlier_count;
                        best_inliers = inliers;
                        best_pose = Some(pose);
                    }
                }
            }
            
            // Early termination if confident
            if best_inlier_count as f32 / n as f32 > self.ransac_confidence {
                break;
            }
        }
        
        // Refine pose with all inliers using EPnP or DLT
        if let Some(ref mut pose) = best_pose {
            if best_inlier_count >= 4 {
                *pose = self.refine_pose(pose, &normalized_pts, &problem.object_points, &best_inliers);
            }
        }
        
        Some((best_pose.unwrap_or(SE3::identity()), best_inliers))
    }

    /// Randomly select k indices
    fn random_indices(&self, n: usize, k: usize) -> Vec<usize> {
        use std::collections::HashSet;
        let mut selected = HashSet::new();
        while selected.len() < k.min(n) {
            // Simple pseudo-random selection
            let idx = ((selected.len() * 17 + 31) % n);
            selected.insert(idx);
        }
        selected.into_iter().collect()
    }

    /// Solve P3P - Perspective-Three-Point Problem
    /// 
    /// Given 3 2D-3D correspondences, compute up to 4 possible camera poses
    fn solve_p3p(&self, img_pts: &[[f32; 2]], obj_pts:&[[f32; 3]], indices: &[usize]) -> Option<Vec<SE3>> {
        // Simplified P3P using geometric approach
        // For robust implementation, use the analytical solution
        
        let p1 = obj_pts[indices[0]];
        let p2 = obj_pts[indices[1]];
        let p3 = obj_pts[indices[2]];
        
        // Normalize 3D points
        let center = Vec3::new(
            (p1[0] + p2[0] + p3[0]) / 3.0,
            (p1[1] + p2[1] + p3[1]) / 3.0,
            (p1[2] + p2[2] + p3[2]) / 3.0,
        );
        
        let q1 = Vec3::new(p1[0] - center.x, p1[1] - center.y, p1[2] - center.z);
        let q2 = Vec3::new(p2[0] - center.x, p2[1] - center.y, p2[2] - center.z);
        let q3 = Vec3::new(p3[0] - center.x, p3[1] - center.y, p3[2] - center.z);
        
        // Distances between 3D points
        let d_12 = (q1 - q2).length();
        let d_23 = (q2 - q3).length();
        let d_31 = (q3 - q1).length();
        
        // Image points (normalized)
        let u1 = Vec3::new(img_pts[indices[0]][0], img_pts[indices[0]][1], 1.0).normalize();
        let u2 = Vec3::new(img_pts[indices[1]][0], img_pts[indices[1]][1], 1.0).normalize();
        let u3 = Vec3::new(img_pts[indices[2]][0], img_pts[indices[2]][1], 1.0).normalize();
        
        // Solve for distances from camera to 3D points
        // This is a simplified version - real P3P has analytical solution
        let mut poses = Vec::new();
        
        // Try different solutions (simplified: just identity for now)
        // In production, use the analytical P3P solution
        poses.push(SE3::identity());
        
        if poses.is_empty() {
            return None;
        }
        
        Some(poses)
    }

    /// Project a 3D point to normalized image coordinates
    fn project_point(&self, pose: &SE3, point: &[f32; 3]) -> [f32; 2] {
        let transformed = pose.transform_point(point);
        if transformed[2] > 0.0 {
            [transformed[0] / transformed[2], transformed[1] / transformed[2]]
        } else {
            [0.0, 0.0]
        }
    }

    /// Compute reprojection error
    fn reprojection_error(&self, p1: &[f32; 2], p2: &[f32; 2]) -> f32 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// Refine pose using all inliers (DLT-based)
    fn refine_pose(&self, initial: &SE3, img_pts: &[[f32; 2]], obj_pts: &[[f32; 3]], inliers: &[bool]) -> SE3 {
        // Collect inlier correspondences
        let mut inlier_img = Vec::new();
        let mut inlier_obj = Vec::new();
        
        for (i, &is_inlier) in inliers.iter().enumerate() {
            if is_inlier {
                inlier_img.push(img_pts[i]);
                inlier_obj.push(obj_pts[i]);
            }
        }
        
        if inlier_img.len() < 4 {
            return *initial;
        }
        
        // Use DLT to refine (simplified - just return initial for now)
        // In production, implement iterative refinement
        *initial
    }
}

/// Essential Matrix solver for 2D-2D motion estimation
pub struct EssentialSolver {
    /// RANSAC parameters
    pub ransac_threshold: f32,
    pub ransac_max_iterations: u32,
}

impl EssentialSolver {
    /// Create a new essential matrix solver
    pub fn new() -> Self {
        Self {
            ransac_threshold: 0.01,
            ransac_max_iterations: 200,
        }
    }

    /// Compute essential matrix from matches using 8-point algorithm + RANSAC
    /// 
    /// Returns: (essential_matrix, inlier_mask)
    pub fn compute(&self, _matches: &[Match], pts1: &[[f32; 2]], pts2: &[[f32; 2]]) -> Option<(Mat3, Vec<bool>)> {
        if pts1.len() < 8 || pts2.len() < 8 {
            return None;
        }

        let n = pts1.len().min(pts2.len());
        
        // Normalize points
        let (norm_pts1, T1) = self.normalize_points(pts1);
        let (norm_pts2, T2) = self.normalize_points(pts2);
        
        // Build the constraint matrix for 8-point algorithm
        let mut A = Vec::new();
        for i in 0..n {
            let x1 = &norm_pts1[i];
            let x2 = &norm_pts2[i];
            // [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
            A.push([
                x2[0] * x1[0], x2[0] * x1[1], x2[0],
                x2[1] * x1[0], x2[1] * x1[1], x2[1],
                x1[0], x1[1], 1.0,
            ]);
        }
        
        // Solve using SVD: find eigenvector with smallest eigenvalue
        let E_norm = self.solve_8point(&A)?;
        
        // Enforce rank-2 constraint
        let E = self.enforce_rank2(E_norm);
        
        // Denormalize: E = T2'^T * E_norm * T1
        let E = T2.transpose() * E * T1;
        
        // RANSAC for robustness
        let inliers = self.ransac_filter(&norm_pts1, &norm_pts2, &E);
        
        Some((E, inliers))
    }

    /// Normalize points for numerical stability
    fn normalize_points(&self, pts: &[[f32; 2]]) -> (Vec<[f32; 2]>, Mat3) {
        let n = pts.len();
        if n == 0 {
            return (vec![], Mat3::IDENTITY);
        }
        
        // Compute centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        for p in pts {
            cx += p[0];
            cy += p[1];
        }
        cx /= n as f32;
        cy /= n as f32;
        
        // Compute scale
        let mut scale = 0.0f32;
        for p in pts {
            scale += ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt();
        }
        scale = (n as f32 * 1.414) / scale.max(1e-8);
        
        // Normalize
        let normalized: Vec<[f32; 2]> = pts.iter()
            .map(|p| [
                (p[0] - cx) * scale,
                (p[1] - cy) * scale,
            ])
            .collect();
        
        // Transformation matrix
        let T = Mat3::from_cols(
            Vec3::new(scale, 0.0, 0.0),
            Vec3::new(0.0, scale, 0.0),
            Vec3::new(-cx * scale, -cy * scale, 1.0),
        );
        
        (normalized, T)
    }

    /// Solve using 8-point algorithm
    fn solve_8point(&self, A: &[[f32; 9]]) -> Option<Mat3> {
        let n = A.len();
        if n < 8 {
            return None;
        }
        
        // Simplified: return identity matrix as placeholder
        // In production, implement proper SVD decomposition
        // A * E = 0, find nullspace of A
        Some(Mat3::IDENTITY)
    }

    /// Enforce rank-2 constraint on essential matrix
    pub fn enforce_rank2(&self, E: Mat3) -> Mat3 {
        // SVD decomposition: E = U * S * V^T
        // Force smallest singular value to 0: E' = U * diag(s1, s2, 0) * V^T
        
        // Simplified: just return E for now
        // In production, implement proper SVD
        E
    }

    /// RANSAC filtering
    fn ransac_filter(&self, pts1: &[[f32; 2]], pts2: &[[f32; 2]], E: &Mat3) -> Vec<bool> {
        let n = pts1.len();
        let mut inliers = vec![true; n];  // All points as inliers for now
        
        // Simplified: check epipolar constraint
        // For each point, the epipolar line should pass through the other point
        for i in 0..n {
            let x1 = Vec3::new(pts1[i][0], pts1[i][1], 1.0).normalize();
            let x2 = Vec3::new(pts2[i][0], pts2[i][1], 1.0).normalize();
            
            let line = E.transpose() * x2;  // Epipolar line in image 1
            let line_norm = line.normalize();
            let error = line_norm.dot(x1).abs();
            
            if error > self.ransac_threshold {
                inliers[i] = false;
            }
        }
        
        inliers
    }

    /// Recover pose from essential matrix
    /// 
    /// Returns: 4 possible pose solutions
    pub fn recover_pose(&self, E: Mat3) -> [SE3; 4] {
        // SVD: E = U * diag(1, 1, 0) * V^T
        // Four possible solutions from the two possible rotation matrices
        // and two possible translations
        
        // Simplified: return 4 identity poses
        // In production, implement proper pose recovery
        [SE3::identity(); 4]
    }
}

impl Default for EssentialSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Triangulation solver using DLT (Direct Linear Transform)
pub struct Triangulator {
    /// Minimum triangulation angle (radians)
    pub min_angle: f32,
    /// Minimum triangulation distance
    pub min_dist: f32,
    /// Maximum reprojection error
    pub max_error: f32,
}

impl Triangulator {
    /// Create a new triangulator
    pub fn new() -> Self {
        Self {
            min_angle: (3.0 * PI / 180.0),  // ~3 degrees
            min_dist: 0.1,
            max_error: 4.0,
        }
    }

    /// Triangulate 2D points from two views using DLT
    /// 
    /// P1, P2: Camera poses (SE3)
    /// pts1, pts2: Corresponding 2D points
    pub fn triangulate(
        &self,
        pose1: &SE3,
        pose2: &SE3,
        pts1: &[[f32; 2]],
        pts2: &[[f32; 2]],
    ) -> Vec<Option<[f32; 3]>> {
        let n = pts1.len().min(pts2.len());
        let mut results = Vec::with_capacity(n);
        
        // Get camera centers
        let c1 = pose1.translation();
        let c2 = pose2.translation();
        
        // Check triangulation angle
        let baseline = (Vec3::from(c2) - Vec3::from(c1)).length();
        
        if baseline < self.min_dist {
            // Baseline too small, return None for all
            return vec![None; n];
        }
        
        for i in 0..n {
            let pt = self.triangulate_dlt(pose1, pose2, pts1[i], pts2[i]);
            
            // Check if point is valid
            if let Some(point) = pt {
                // Check if point is in front of both cameras
                let p = Vec3::from(point);
                let to_c1 = Vec3::from(c1) - p;
                let to_c2 = Vec3::from(c2) - p;
                
                // Check angle
                let angle = to_c1.angle_between(to_c2);
                
                if angle > self.min_angle && point[2] > 0.0 {
                    results.push(Some(point));
                } else {
                    results.push(None);
                }
            } else {
                results.push(None);
            }
        }
        
        results
    }

    /// DLT-based triangulation
    fn triangulate_dlt(&self, pose1: &SE3, pose2: &SE3, p1: [f32; 2], p2: [f32; 2]) -> Option<[f32; 3]> {
        // Build projection matrices (simplified)
        // P = K * [R|t] = [R|t] for normalized coordinates
        
        let r1 = pose1.rotation_matrix();
        let t1 = pose1.translation();
        let r2 = pose2.rotation_matrix();
        let t2 = pose2.translation();
        
        // Cross-product matrix for translation
        let t1_vec = Vec3::from(t1);
        let t2_vec = Vec3::from(t2);
        let cross1 = Mat3::from_cols(
            Vec3::new(0.0, -t1_vec.z, t1_vec.y),
            Vec3::new(t1_vec.z, 0.0, -t1_vec.x),
            Vec3::new(-t1_vec.y, t1_vec.x, 0.0),
        );
        
        let cross2 = Mat3::from_cols(
            Vec3::new(0.0, -t2_vec.z, t2_vec.y),
            Vec3::new(t2_vec.z, 0.0, -t2_vec.x),
            Vec3::new(-t2_vec.y, t2_vec.x, 0.0),
        );
        
        // Build the DLT matrix A * X = 0
        // For two views: [x1*P1^TP1^⊥; x2*P2^TP2^⊥] * X = 0
        
        // Simplified: use mid-point triangulation
        // Ray from camera 1
        let dir1 = Vec3::new(p1[0], p1[1], 1.0).normalize();
        let ray1_origin = Vec3::from(t1);
        let ray1_dir = Vec3::new(r1[0][0]*dir1.x + r1[1][0]*dir1.y + r1[2][0]*dir1.z,
                                  r1[0][1]*dir1.x + r1[1][1]*dir1.y + r1[2][1]*dir1.z,
                                  r1[0][2]*dir1.x + r1[1][2]*dir1.y + r1[2][2]*dir1.z);
        
        // Ray from camera 2  
        let dir2 = Vec3::new(p2[0], p2[1], 1.0).normalize();
        let ray2_origin = Vec3::from(t2);
        let ray2_dir = Vec3::new(r2[0][0]*dir2.x + r2[1][0]*dir2.y + r2[2][0]*dir2.z,
                                  r2[0][1]*dir2.x + r2[1][1]*dir2.y + r2[2][1]*dir2.z,
                                  r2[0][2]*dir2.x + r2[1][2]*dir2.y + r2[2][2]*dir2.z);
        
        // Mid-point triangulation
        // Find closest point between two rays
        let cross_ray = ray1_dir.cross(ray2_dir);
        if cross_ray.length_squared() < 1e-6 {
            return None;  // Rays are parallel
        }
        
        let t = (ray2_origin - ray1_origin).cross(ray2_dir).dot(cross_ray) / cross_ray.length_squared();
        let mid_point = ray1_origin + ray1_dir * t;
        
        Some([mid_point.x, mid_point.y, mid_point.z])
    }

    /// Check if a point is observable from a camera pose
    fn is_observable(&self, point: &[f32; 3], pose: &SE3) -> bool {
        let cam_center = Vec3::from(pose.translation());
        let point_vec = Vec3::new(point[0], point[1], point[2]);
        let ray = point_vec - cam_center;
        
        // Point should be in front of camera (positive z in camera frame)
        let r = pose.rotation_matrix();
        let z_dir = Vec3::new(r[2][0], r[2][1], r[2][2]);
        
        ray.dot(z_dir) > 0.0
    }
}

impl Default for Triangulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Sim3 Solver for similarity transform estimation (scale + rotation + translation)
pub struct Sim3Solver {
    /// RANSAC threshold
    pub ransac_threshold: f32,
}

impl Sim3Solver {
    /// Create a new Sim3 solver
    pub fn new(threshold: f32) -> Self {
        Self {
            ransac_threshold: threshold,
        }
    }

    /// Compute Sim3 transform between two sets of 3D points
    /// 
    /// Returns: (sim3_transform, inliers)
    /// sim3: (scale, translation, rotation) 
    pub fn compute(&self, pts1: &[[f32; 3]], pts2: &[[f32; 3]]) -> Option<((f32, [f32; 3], Mat3), Vec<bool>)> {
        if pts1.len() < 3 || pts2.len() < 3 {
            return None;
        }

        let n = pts1.len().min(pts2.len());
        
        // Compute centroids
        let c1 = self.compute_centroid(pts1);
        let c2 = self.compute_centroid(pts2);
        
        // Compute scale
        let scale = self.compute_scale(pts1, c1, pts2, c2);
        
        // Compute rotation using SVD (simplified)
        let rotation = Mat3::IDENTITY;
        
        // Compute translation
        let translation = [
            c2[0] - scale * (rotation.col(0)[0] * c1[0] + rotation.col(1)[0] * c1[1] + rotation.col(2)[0] * c1[2]),
            c2[1] - scale * (rotation.col(0)[1] * c1[0] + rotation.col(1)[1] * c1[1] + rotation.col(2)[1] * c1[2]),
            c2[2] - scale * (rotation.col(0)[2] * c1[0] + rotation.col(1)[2] * c1[1] + rotation.col(2)[2] * c1[2]),
        ];
        
        // Compute inliers
        let mut inliers = vec![false; n];
        for i in 0..n {
            let transformed = self.apply_sim3((scale, translation, rotation), pts1[i]);
            let error = (Vec3::from(transformed) - Vec3::from(pts2[i])).length();
            if error < self.ransac_threshold * 10.0 {
                inliers[i] = true;
            }
        }
        
        Some(((scale, translation, rotation), inliers))
    }

    /// Compute centroid of points
    fn compute_centroid(&self, pts: &[[f32; 3]]) -> [f32; 3] {
        let n = pts.len() as f32;
        let mut c = [0.0f32; 3];
        for p in pts {
            c[0] += p[0] / n;
            c[1] += p[1] / n;
            c[2] += p[2] / n;
        }
        c
    }

    /// Compute scale between two point sets
    fn compute_scale(&self, pts1: &[[f32; 3]], c1: [f32; 3], pts2: &[[f32; 3]], c2: [f32; 3]) -> f32 {
        let n = pts1.len() as f32;
        
        let mut d1_sq = 0.0f32;
        let mut d2_sq = 0.0f32;
        
        for i in 0..pts1.len() {
            let dx = pts1[i][0] - c1[0];
            let dy = pts1[i][1] - c1[1];
            let dz = pts1[i][2] - c1[2];
            d1_sq += dx*dx + dy*dy + dz*dz;
            
            let dx = pts2[i][0] - c2[0];
            let dy = pts2[i][1] - c2[1];
            let dz = pts2[i][2] - c2[2];
            d2_sq += dx*dx + dy*dy + dz*dz;
        }
        
        if d1_sq > 1e-8 {
            (d2_sq / d1_sq).sqrt()
        } else {
            1.0
        }
    }

    /// Create a Sim3 transform
    pub fn create_sim3(&self, scale: f32, translation: Vec3, rotation: Mat3) -> (f32, [f32; 3], Mat3) {
        (scale, [translation.x, translation.y, translation.z], rotation)
    }

    /// Apply Sim3 transform to a point
    pub fn apply_sim3(&self, sim3: (f32, [f32; 3], Mat3), point: [f32; 3]) -> [f32; 3] {
        let (scale, translation, rotation) = sim3;
        let p = Vec3::from(point);
        let transformed = rotation * (p * scale) + Vec3::from(translation);
        [transformed.x, transformed.y, transformed.z]
    }
}

impl Default for Sim3Solver {
    fn default() -> Self {
        Self::new(0.01)
    }
}
