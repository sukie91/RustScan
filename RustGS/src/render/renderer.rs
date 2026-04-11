//! Gaussian Renderer
//!
//! Implements the core rendering pipeline for 3D Gaussian Splatting.
//! Based on:
//! - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! - RTG-SLAM: Real-time 3D Reconstruction

#[cfg(feature = "gpu")]
use crate::core::GaussianCamera;
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::sh0_to_rgb_value;
#[cfg(feature = "gpu")]
use crate::training::{HostSplats, SplatView};
#[cfg(feature = "gpu")]
use glam::{Mat3, Vec3};

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

/// Gaussian Renderer
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

    /// Render host-side splats.
    #[cfg(feature = "gpu")]
    pub fn render_splats(
        &self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> candle_core::Result<RenderOutput> {
        let mut color = self.background_rgb_buffer();
        let mut depth = vec![0.0f32; self.width * self.height];
        let mut projected_splats = self.project_visible_splats(splats.as_view(), camera);

        projected_splats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (gc, cam_depth, ux, uy, radius) in projected_splats {
            self.render_gaussian(
                &mut color, &mut depth, cam_depth, gc, ux as i32, uy as i32, radius,
            );
        }

        Ok(RenderOutput {
            color,
            depth,
            normal: None,
            width: self.width,
            height: self.height,
        })
    }

    /// Render a single Gaussian as a filled circle using camera-space depth
    fn render_gaussian(
        &self,
        color: &mut [u8],
        depth: &mut [f32],
        cam_depth: f32,
        gc: [u8; 3],
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

                    // Keep minimum camera-space depth (nearest surface wins)
                    if depth[idx] == 0.0 || cam_depth < depth[idx] {
                        color[idx * 3] = gc[0];
                        color[idx * 3 + 1] = gc[1];
                        color[idx * 3 + 2] = gc[2];
                        depth[idx] = cam_depth;
                    }
                }
            }
        }
    }

    /// Render depth directly from host-side splats.
    #[cfg(feature = "gpu")]
    pub fn render_depth_splats(
        &self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> candle_core::Result<Vec<f32>> {
        let mut depth = vec![0.0f32; self.width * self.height];
        let mut projected_splats = self.project_visible_splats(splats.as_view(), camera);

        projected_splats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (_, cam_depth, ux, uy, radius) in projected_splats {
            self.render_depth_circle(&mut depth, cam_depth, ux as i32, uy as i32, radius);
        }

        Ok(depth)
    }

    /// Render depth and color directly from host-side splats.
    #[cfg(feature = "gpu")]
    pub fn render_depth_and_color_splats(
        &self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> candle_core::Result<(Vec<f32>, Vec<[u8; 3]>)> {
        let mut depth = vec![0.0f32; self.width * self.height];
        let mut color = self.background_color_buffer();
        let mut projected_splats = self.project_visible_splats(splats.as_view(), camera);

        projected_splats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (gc, cam_depth, ux, uy, radius) in projected_splats {
            self.render_depth_color_circle(
                &mut depth, &mut color, cam_depth, gc, ux as i32, uy as i32, radius,
            );
        }

        Ok((depth, color))
    }

    /// Render depth and color as circle
    fn render_depth_color_circle(
        &self,
        depth: &mut [f32],
        color: &mut [[u8; 3]],
        z: f32,
        gc: [u8; 3],
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
                    if depth[idx] == 0.0 || z < depth[idx] {
                        depth[idx] = z;
                        color[idx] = gc;
                    }
                }
            }
        }
    }

    /// Render depth as circle
    fn render_depth_circle(&self, depth: &mut [f32], z: f32, cx: i32, cy: i32, radius: f32) {
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

    fn background_rgb_buffer(&self) -> Vec<u8> {
        let pixel = self.background_u8();
        let mut color = vec![0u8; self.width * self.height * 3];
        for rgb in color.chunks_exact_mut(3) {
            rgb.copy_from_slice(&pixel);
        }
        color
    }

    fn background_color_buffer(&self) -> Vec<[u8; 3]> {
        vec![self.background_u8(); self.width * self.height]
    }

    fn background_u8(&self) -> [u8; 3] {
        [
            (self.background[0].clamp(0.0, 1.0) * 255.0) as u8,
            (self.background[1].clamp(0.0, 1.0) * 255.0) as u8,
            (self.background[2].clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }

    #[cfg(feature = "gpu")]
    fn project_visible_splats(
        &self,
        splats: SplatView<'_>,
        camera: &GaussianCamera,
    ) -> Vec<([u8; 3], f32, f32, f32, f32)> {
        let (fx, fy, cx, cy) = (
            camera.intrinsics.fx,
            camera.intrinsics.fy,
            camera.intrinsics.cx,
            camera.intrinsics.cy,
        );
        let rotation = camera.extrinsics.rotation_matrix();
        let rotation_mat = Mat3::from_cols(
            Vec3::new(rotation[0][0], rotation[0][1], rotation[0][2]),
            Vec3::new(rotation[1][0], rotation[1][1], rotation[1][2]),
            Vec3::new(rotation[2][0], rotation[2][1], rotation[2][2]),
        );
        let translation = camera.extrinsics.translation();
        let camera_origin = Vec3::new(translation[0], translation[1], translation[2]);
        let sh_row_width = ((splats.sh_degree + 1) * (splats.sh_degree + 1)) * 3;

        (0..splats.opacity_logits.len())
            .filter_map(|idx| {
                let pos_base = idx * 3;
                let position = Vec3::new(
                    splats.positions[pos_base],
                    splats.positions[pos_base + 1],
                    splats.positions[pos_base + 2],
                );
                let cam_pos = rotation_mat.transpose() * (position - camera_origin);
                if cam_pos.z <= 0.001 || cam_pos.z >= 100.0 {
                    return None;
                }

                let u = fx * cam_pos.x / cam_pos.z + cx;
                let v = fy * cam_pos.y / cam_pos.z + cy;
                let radius = (fx + fy) * splats.log_scales[pos_base].exp() / cam_pos.z;
                let sh_base = idx * sh_row_width;
                let gc = [
                    (sh0_to_rgb_value(splats.sh_coeffs[sh_base]).clamp(0.0, 1.0) * 255.0) as u8,
                    (sh0_to_rgb_value(splats.sh_coeffs[sh_base + 1]).clamp(0.0, 1.0) * 255.0) as u8,
                    (sh0_to_rgb_value(splats.sh_coeffs[sh_base + 2]).clamp(0.0, 1.0) * 255.0) as u8,
                ];

                Some((gc, cam_pos.z, u, v, radius))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::{Intrinsics, SE3};

    #[cfg(feature = "gpu")]
    fn single_rgb_splats(position: [f32; 3], scale: [f32; 3], color: [f32; 3]) -> HostSplats {
        HostSplats::from_raw_parts(
            position.into(),
            scale.map(f32::ln).into(),
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            color
                .map(crate::diff::diff_splat::rgb_to_sh0_value)
                .into(),
            0,
        )
        .unwrap()
    }

    #[test]
    fn test_renderer_creation() {
        let renderer = GaussianRenderer::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_render_empty_map() {
        let renderer = GaussianRenderer::new(64, 64);
        let splats = HostSplats::default();
        let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
        let camera = GaussianCamera::new(intrinsics, SE3::identity());

        let output = renderer.render_splats(&splats, &camera).unwrap();
        assert_eq!(output.color.len(), 64 * 64 * 3);
        assert_eq!(output.depth.len(), 64 * 64);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_render_depth() {
        let renderer = GaussianRenderer::new(64, 64);
        let splats = single_rgb_splats(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 128.0 / 255.0, 64.0 / 255.0],
        );

        let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
        let camera = GaussianCamera::new(intrinsics, SE3::identity());

        let depth = renderer.render_depth_splats(&splats, &camera).unwrap();

        // Should have some depth values
        assert!(depth.iter().any(|&d| d > 0.0));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_render_depth_and_color() {
        let renderer = GaussianRenderer::new(64, 64);
        let splats = single_rgb_splats(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 128.0 / 255.0, 64.0 / 255.0],
        );

        let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
        let camera = GaussianCamera::new(intrinsics, SE3::identity());

        let (depth, color) = renderer.render_depth_and_color_splats(&splats, &camera).unwrap();

        // Should have some depth values
        assert!(depth.iter().any(|&d| d > 0.0));
        // Should have some color values
        assert!(color.iter().any(|&c| c != [0, 0, 0]));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_render_splats() {
        let renderer = GaussianRenderer::new(64, 64).with_background(0.1, 0.2, 0.3);
        let splats = single_rgb_splats(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 128.0 / 255.0, 64.0 / 255.0],
        );
        let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
        let camera = GaussianCamera::new(intrinsics, SE3::identity());

        let output = renderer.render_splats(&splats, &camera).unwrap();
        assert!(output.depth.iter().any(|&d| d > 0.0));
        assert!(output.color.iter().any(|&channel| channel != 25 && channel != 51 && channel != 76));
    }
}
