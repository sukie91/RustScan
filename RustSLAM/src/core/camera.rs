//! Camera model representation

use glam::{Mat3, Vec3};

/// Camera intrinsic parameters
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Focal length in pixels (fx, fy)
    pub focal: Vec3,
    /// Principal point (cx, cy)
    pub principal: Vec3,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Distortion parameters (k1, k2, p1, p2, k3)
    pub distortion: Option<[f32; 5]>,
}

impl Camera {
    /// Create a new camera with given parameters
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            focal: Vec3::new(fx, fy, 1.0),
            principal: Vec3::new(cx, cy, 0.0),
            width,
            height,
            distortion: None,
        }
    }

    /// Create a camera from K matrix
    pub fn from_k(k: &Mat3, width: u32, height: u32) -> Self {
        Self {
            focal: Vec3::new(k.col(0).x, k.col(1).y, 1.0),
            principal: Vec3::new(k.col(0).z, k.col(1).z, 0.0),
            width,
            height,
            distortion: None,
        }
    }

    /// Get the intrinsic matrix
    pub fn k(&self) -> Mat3 {
        Mat3::from_cols(
            self.focal.x * Vec3::X,
            self.focal.y * Vec3::Y,
            self.principal.x * Vec3::X + self.principal.y * Vec3::Y,
        )
    }

    /// Project a 3D point to pixel coordinates
    pub fn project(&self, point: &Vec3) -> Option<Vec3> {
        if point.z <= 0.0 {
            return None;
        }
        
        let x = point.x / point.z;
        let y = point.y / point.z;
        
        let px = self.focal.x * x + self.principal.x;
        let py = self.focal.y * y + self.principal.y;
        
        Some(Vec3::new(px, py, point.z))
    }

    /// Unproject a pixel to 3D ray
    pub fn unproject(&self, pixel: &Vec3, depth: f32) -> Vec3 {
        let x = (pixel.x - self.principal.x) / self.focal.x;
        let y = (pixel.y - self.principal.y) / self.focal.y;
        Vec3::new(x, y, 1.0) * depth
    }

    /// Check if a pixel is inside the image
    pub fn is_in_image(&self, pixel: &Vec3, margin: i32) -> bool {
        pixel.x >= margin as f32 
            && pixel.x < (self.width as i32 - margin) as f32
            && pixel.y >= margin as f32
            && pixel.y < (self.height as i32 - margin) as f32
    }
}

/// Default camera (640x480, common parameters)
impl Default for Camera {
    fn default() -> Self {
        Self::new(525.0, 525.0, 319.5, 239.5, 640, 480)
    }
}
