//! Differentiable Gaussian Renderer
//!
//! Implements the core rendering pipeline for 3D Gaussian Splatting.
//! Based on:
//! - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! - RTG-SLAM: Real-time 3D Reconstruction

use crate::fusion::gaussian::{Gaussian3D, GaussianCamera, GaussianMap};
use glam::Mat3;

/// Output of rendering
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Rendered color image (RGB)
    pub color: Vec<u8>,
    /// Rendered depth image
    pub depth: Vec<f32>,
    /// Rendered normal (optional)
    pub normal: Option<Vec<[f32; 3]>>,
    /// Image dimensions
    pub width: usize,
    pub height: usize,
}

/// Differentiable Gaussian Renderer
pub struct GaussianRenderer {
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// Background color (RGB)
    background: [f32; 3],
}

impl GaussianRenderer {
    /// Create a new renderer
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            background: [0.0, 0.0, 0.0],
        }
    }

    /// Set background color
    pub fn with_background(mut self, r: f32, g: f32, b: f32) -> Self {
        self.background = [r, g, b];
        self
    }

    /// Render the Gaussian map from a camera view
    pub fn render(&self, map: &GaussianMap, camera: &GaussianCamera) -> RenderOutput {
        let mut color = vec![0u8; self.width * self.height * 3];
        let mut depth = vec![0.0f32; self.width * self.height];
        
        // Sort Gaussians by depth (back to front)
        let mut gaussians_with_depth: Vec<(&Gaussian3D, f32)> = map.gaussians()
            .iter()
            .filter_map(|g| {
                // Get depth in camera frame
                let pos = g.position;
                let depth_val = pos.z;
                if depth_val > 0.0 && depth_val < 100.0 {
                    Some((g, depth_val))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by depth (far to near for alpha blending)
        gaussians_with_depth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Render each Gaussian
        for (gaussian, _) in gaussians_with_depth {
            // Project Gaussian to 2D
            if let Some([ux, uy, radius]) = gaussian.project(
                camera.fx, camera.fy, camera.cx, camera.cy,
                &camera.rotation,
                &camera.translation,
            ) {
                // Render as circle (simplified)
                self.render_gaussian(
                    &mut color,
                    &mut depth,
                    gaussian,
                    ux as i32,
                    uy as i32,
                    radius,
                );
            }
        }
        
        // Convert to u8
        for i in 0..self.width * self.height {
            color[i * 3] = (color[i * 3] as f32).clamp(0.0, 255.0) as u8;
            color[i * 3 + 1] = (color[i * 3 + 1] as f32).clamp(0.0, 255.0) as u8;
            color[i * 3 + 2] = (color[i * 3 + 2] as f32).clamp(0.0, 255.0) as u8;
        }
        
        RenderOutput {
            color,
            depth,
            normal: None,
            width: self.width,
            height: self.height,
        }
    }

    /// Render a single Gaussian as a filled circle
    fn render_gaussian(
        &self,
        color: &mut [u8],
        depth: &mut [f32],
        gaussian: &Gaussian3D,
        cx: i32,
        cy: i32,
        radius: f32,
    ) {
        let radius = radius.max(1.0);
        let r_sq = radius * radius;
        
        let min_x = ((cx as f32 - radius) as i32).max(0);
        let max_x = ((cx as f32 + radius) as i32).min(self.width as i32 - 1);
        let min_y = ((cy as f32 - radius) as i32).max(0);
        let max_y = ((cy as f32 + radius) as i32).min(self.height as i32 - 1);
        
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist_sq = dx * dx + dy * dy;
                
                if dist_sq <= r_sq {
                    let idx = y as usize * self.width + x as usize;
                    
                    // Gaussian falloff (simple)
                    let alpha = gaussian.opacity * (-dist_sq / r_sq).exp();
                    
                    // Alpha blend
                    let bg_r = self.background[0] * 255.0;
                    let bg_g = self.background[1] * 255.0;
                    let bg_b = self.background[2] * 255.0;
                    
                    color[idx * 3] = ((1.0 - alpha) * bg_r + alpha * gaussian.color[0] * 255.0) as u8;
                    color[idx * 3 + 1] = ((1.0 - alpha) * bg_g + alpha * gaussian.color[1] * 255.0) as u8;
                    color[idx * 3 + 2] = ((1.0 - alpha) * bg_b + alpha * gaussian.color[2] * 255.0) as u8;
                    
                    // Depth (simple, use Gaussian center depth)
                    if depth[idx] == 0.0 || gaussian.position.z < depth[idx] {
                        depth[idx] = gaussian.position.z;
                    }
                }
            }
        }
    }

    /// Render depth only (for tracking)
    pub fn render_depth(&self, map: &GaussianMap, camera: &GaussianCamera) -> Vec<f32> {
        let mut depth = vec![0.0f32; self.width * self.height];
        
        // Sort by depth
        let mut gaussians_with_depth: Vec<(&Gaussian3D, f32)> = map.gaussians()
            .iter()
            .filter_map(|g| {
                let d = g.position.z;
                if d > 0.0 && d < 100.0 {
                    Some((g, d))
                } else {
                    None
                }
            })
            .collect();
        
        gaussians_with_depth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        for (gaussian, _) in gaussians_with_depth {
            if let Some([ux, uy, radius]) = gaussian.project(
                camera.fx, camera.fy, camera.cx, camera.cy,
                &camera.rotation,
                &camera.translation,
            ) {
                self.render_depth_circle(
                    &mut depth,
                    gaussian.position.z,
                    ux as i32,
                    uy as i32,
                    radius,
                );
            }
        }
        
        depth
    }

    /// Render depth as circle
    fn render_depth_circle(
        &self,
        depth: &mut [f32],
        z: f32,
        cx: i32,
        cy: i32,
        radius: f32,
    ) {
        let radius = radius.max(1.0);
        let r_sq = radius * radius;
        
        let min_x = ((cx as f32 - radius) as i32).max(0);
        let max_x = ((cx as f32 + radius) as i32).min(self.width as i32 - 1);
        let min_y = ((cy as f32 - radius) as i32).max(0);
        let max_y = ((cy as f32 + radius) as i32).min(self.height as i32 - 1);
        
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist_sq = dx * dx + dy * dy;
                
                if dist_sq <= r_sq {
                    let idx = y as usize * self.width + x as usize;
                    // Keep minimum depth
                    if depth[idx] == 0.0 || z < depth[idx] {
                        depth[idx] = z;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = GaussianRenderer::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[test]
    fn test_render_empty_map() {
        let renderer = GaussianRenderer::new(64, 64);
        let map = GaussianMap::new(100);
        let camera = GaussianCamera::new(500.0, 500.0, 32.0, 32.0);
        
        let output = renderer.render(&map, &camera);
        assert_eq!(output.color.len(), 64 * 64 * 3);
        assert_eq!(output.depth.len(), 64 * 64);
    }

    #[test]
    fn test_render_depth() {
        let renderer = GaussianRenderer::new(64, 64);
        let mut map = GaussianMap::new(100);
        
        // Add a Gaussian at z=1.0
        let g = Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]);
        map.add(g);
        
        let camera = GaussianCamera::new(500.0, 500.0, 32.0, 32.0);
        let depth = renderer.render_depth(&map, &camera);
        
        // Should have some depth values
        assert!(depth.iter().any(|&d| d > 0.0));
    }
}
