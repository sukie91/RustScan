//! Gaussian camera for rendering.
//!
//! This module will be populated from RustSLAM/src/fusion/gaussian.rs in Story 9-3.

use crate::Intrinsics;
use crate::SE3;
use glam::Mat4;

/// Camera for Gaussian rendering.
///
/// Combines intrinsics and extrinsics for rendering.
#[derive(Debug, Clone)]
pub struct GaussianCamera {
    /// Camera intrinsics
    pub intrinsics: Intrinsics,
    /// Camera extrinsics (world-to-camera transform)
    pub extrinsics: SE3,
}

impl GaussianCamera {
    /// Create a new Gaussian camera.
    pub fn new(intrinsics: Intrinsics, extrinsics: SE3) -> Self {
        Self {
            intrinsics,
            extrinsics,
        }
    }

    /// Get the view matrix (world-to-camera).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.extrinsics.to_matrix())
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        let Intrinsics {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        } = self.intrinsics;
        let near = 0.01;
        let far = 100.0;

        // OpenGL-style projection matrix
        let w = width as f32;
        let h = height as f32;

        Mat4::from_cols_array(&[
            2.0 * fx / w,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * fy / h,
            0.0,
            0.0,
            1.0 - 2.0 * cx / w,
            1.0 - 2.0 * cy / h,
            -(far + near) / (far - near),
            -1.0,
            0.0,
            0.0,
            -2.0 * far * near / (far - near),
            0.0,
        ])
    }

    /// Project a 3D point to screen coordinates.
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        let p = self.extrinsics.transform_point(&point);
        let z = p[2];
        if z <= 0.0 {
            return None;
        }

        let Intrinsics { fx, fy, cx, cy, .. } = self.intrinsics;
        let u = fx * p[0] / z + cx;
        let v = fy * p[1] / z + cy;

        Some([u, v])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_camera_creation() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let extrinsics = SE3::identity();
        let camera = GaussianCamera::new(intrinsics, extrinsics);

        assert_eq!(camera.intrinsics.width, 1920);
    }

    #[test]
    fn test_gaussian_camera_project() {
        let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
        let extrinsics = SE3::identity();
        let camera = GaussianCamera::new(intrinsics, extrinsics);

        // Point at z=1 should project near center
        let result = camera.project([0.0, 0.0, 1.0]);
        assert!(result.is_some());

        let [u, v] = result.unwrap();
        assert!((u - 960.0).abs() < 1.0);
        assert!((v - 540.0).abs() < 1.0);
    }
}
