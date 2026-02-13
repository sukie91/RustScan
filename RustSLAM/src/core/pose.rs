//! SE3 Pose representation using glam
//! 
//! This module provides SE(3) pose representation using glam's Quat and Vec3.
//! All internal operations use f32 for performance with glam.

use glam::{Mat3, Mat4, Quat, Vec3};

/// SE3 pose: rotation + translation
/// 
/// This is a wrapper around glam's types that provides
/// a convenient API for SLAM applications.
#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    /// Rotation as quaternion
    rotation: Quat,
    /// Translation vector
    translation: Vec3,
}

impl SE3 {
    /// Create a new SE3 from quaternion and translation
    /// 
    /// # Arguments
    /// * `quaternion` - [x, y, z, w] format
    /// * `translation` - [x, y, z] format
    pub fn new(quaternion: &[f32; 4], translation: &[f32; 3]) -> Self {
        Self {
            rotation: Quat::from_xyzw(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
            translation: Vec3::new(translation[0], translation[1], translation[2]),
        }
    }

    /// Create from axis-angle and translation
    pub fn from_axis_angle(axis_angle: &[f32; 3], translation: &[f32; 3]) -> Self {
        let axis = Vec3::new(axis_angle[0], axis_angle[1], axis_angle[2]);
        let angle = axis.length();
        let axis = if angle > 1e-10 { axis / angle } else { axis };
        let rotation = Quat::from_axis_angle(axis, angle);
        let translation = Vec3::new(translation[0], translation[1], translation[2]);
        
        Self { rotation, translation }
    }

    /// Create identity pose
    pub fn identity() -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        }
    }

    /// Create from rotation matrix (3x3) and translation (3D)
    #[allow(dead_code)]
    pub fn from_rotation_translation(rotation: &[[f32; 3]; 3], translation: &[f32; 3]) -> Self {
        let r = Mat3::from_cols_array(&[
            rotation[0][0], rotation[0][1], rotation[0][2],
            rotation[1][0], rotation[1][1], rotation[1][2],
            rotation[2][0], rotation[2][1], rotation[2][2],
        ]);
        let rot = Quat::from_mat3(&r);
        let t = Vec3::new(translation[0], translation[1], translation[2]);
        
        Self { rotation: rot, translation: t }
    }

    /// Convert to 4x4 transformation matrix
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        let m = Mat4::from_rotation_translation(self.rotation, self.translation);
        let cols = m.to_cols_array();
        [
            [cols[0], cols[1], cols[2], cols[3]],
            [cols[4], cols[5], cols[6], cols[7]],
            [cols[8], cols[9], cols[10], cols[11]],
            [cols[12], cols[13], cols[14], cols[15]],
        ]
    }

    /// Compose two poses: self * other
    pub fn compose(&self, other: &SE3) -> SE3 {
        let rotation = self.rotation * other.rotation;
        let translation = self.translation + self.rotation * other.translation;
        SE3 { rotation, translation }
    }

    /// Inverse of the pose
    pub fn inverse(&self) -> SE3 {
        let rotation = self.rotation.inverse();
        let translation = -(rotation * self.translation);
        SE3 { rotation, translation }
    }

    /// Transform a 3D point
    pub fn transform_point(&self, point: &[f32; 3]) -> [f32; 3] {
        let p = Vec3::new(point[0], point[1], point[2]);
        let transformed = self.rotation * p + self.translation;
        [transformed.x, transformed.y, transformed.z]
    }

    /// Transform a 3D vector (direction)
    pub fn transform_vector(&self, vec: &[f32; 3]) -> [f32; 3] {
        let v = Vec3::new(vec[0], vec[1], vec[2]);
        let transformed = self.rotation * v;
        [transformed.x, transformed.y, transformed.z]
    }

    /// Get rotation matrix as 3x3
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let r = Mat3::from_quat(self.rotation);
        let cols = r.to_cols_array();
        [
            [cols[0], cols[1], cols[2]],
            [cols[3], cols[4], cols[5]],
            [cols[6], cols[7], cols[8]],
        ]
    }

    /// Get translation as 3D vector
    pub fn translation(&self) -> [f32; 3] {
        [self.translation.x, self.translation.y, self.translation.z]
    }

    /// Get quaternion as [x, y, z, w]
    pub fn quaternion(&self) -> [f32; 4] {
        [self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w]
    }

    /// Get rotation as axis-angle [rx, ry, rz]
    pub fn axis_angle(&self) -> [f32; 3] {
        let (axis, angle) = self.rotation.to_axis_angle();
        [axis.x * angle, axis.y * angle, axis.z * angle]
    }

    /// Create from Lie algebra (tangent space) - exp map
    /// tangent: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    pub fn exp(tangent: &[f32; 6]) -> Self {
        // omega = rotation vector
        let omega = Vec3::new(tangent[0], tangent[1], tangent[2]);
        let v = Vec3::new(tangent[3], tangent[4], tangent[5]);
        
        // Compute rotation from so3 exponential
        let angle = omega.length();
        let axis = if angle > 1e-10 { omega / angle } else { omega };
        let rotation = Quat::from_axis_angle(axis, angle);
        
        // Compute translation: V * v
        // V = I + (1 - cos(θ))/θ² * [ω]ₓ + (θ - sin(θ))/θ³ * [ω]ₓ²
        let t_vec: Vec3 = if angle < 1e-10 {
            v
        } else {
            let c1 = (1.0 - angle.cos()) / (angle * angle);
            let c2 = (angle - angle.sin()) / (angle * angle * angle);
            
            // Skew-symmetric matrix of omega
            let omega_hat = Mat3::from_cols(
                Vec3::new(0.0, -omega.z, omega.y),
                Vec3::new(omega.z, 0.0, -omega.x),
                Vec3::new(-omega.y, omega.x, 0.0),
            );
            
            v + (c1 * omega_hat * v) + (c2 * omega_hat * omega_hat * v)
        };
        
        SE3 { rotation, translation: t_vec }
    }

    /// Log map: group element to Lie algebra
    pub fn log(&self) -> [f32; 6] {
        // Get rotation vector (so3)
        let (axis, angle) = self.rotation.to_axis_angle();
        let omega = Vec3::new(axis.x * angle, axis.y * angle, axis.z * angle);
        
        // Get translation
        let t = self.translation;
        
        // Compute V inverse
        let v: Vec3 = if angle < 1e-10 {
            t
        } else {
            let c1 = (1.0 - angle.cos()) / (angle * angle);
            let c2 = (angle - angle.sin()) / (angle * angle * angle);
            
            let omega_hat = Mat3::from_cols(
                Vec3::new(0.0, -omega.z, omega.y),
                Vec3::new(omega.z, 0.0, -omega.x),
                Vec3::new(-omega.y, omega.x, 0.0),
            );
            
            t - (c1 * omega_hat * t) + (c2 * omega_hat * omega_hat * t)
        };
        
        [omega.x, omega.y, omega.z, v.x, v.y, v.z]
    }

    /// Get rotation matrix (for compatibility)
    pub fn rotation(&self) -> [[f32; 3]; 3] {
        self.rotation_matrix()
    }
}

impl Default for SE3 {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let pose = SE3::identity();
        let matrix = pose.to_matrix();
        
        // Check it's identity
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((matrix[2][2] - 1.0).abs() < 1e-10);
        assert!((matrix[3][3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let a = SE3::identity();
        let b = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        
        let c = a.compose(&b);
        let t = c.translation();
        assert!((t[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse() {
        let pose = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        let inv = pose.inverse();
        
        // pose * inverse should be identity
        let composed = pose.compose(&inv);
        let matrix = composed.to_matrix();
        
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);
        assert!((matrix[2][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_point() {
        let pose = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        let point = [1.0, 2.0, 3.0];
        
        let transformed = pose.transform_point(&point);
        assert!((transformed[0] - 2.0).abs() < 1e-6);
        assert!((transformed[1] - 2.0).abs() < 1e-6);
        assert!((transformed[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_exp_log() {
        // Test exp/log roundtrip
        let tangent = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let pose = SE3::exp(&tangent);
        let log_tangent = pose.log();
        
        for i in 0..6 {
            assert!((tangent[i] - log_tangent[i]).abs() < 1e-4);
        }
    }
}
