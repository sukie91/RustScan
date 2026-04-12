//! Complete 3DGS Renderer - Tiled Rasterization
//!
//! Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! Kerbl et al. SIGGRAPH 2023
//!
//! This implements:
//! 1. Gaussian projection to 2D
//! 2. Tiled rasterization
//! 3. Depth sorting
//! 4. Alpha blending

#[cfg(feature = "gpu")]
use crate::sh::sh0_to_rgb_value;
#[cfg(all(test, feature = "gpu"))]
use crate::sh::rgb_to_sh0_value;
#[cfg(feature = "gpu")]
use crate::training::SplatView;

/// 2D projected splat
#[derive(Debug, Clone)]
pub struct ProjectedGaussian {
    /// Center x in image coordinates
    pub x: f32,
    /// Center y in image coordinates
    pub y: f32,
    /// Depth (for sorting)
    pub depth: f32,
    /// 2D covariance xx
    pub cov_xx: f32,
    /// 2D covariance xy
    pub cov_xy: f32,
    /// 2D covariance yy
    pub cov_yy: f32,
    /// Opacity
    pub opacity: f32,
    /// Color
    pub color: [f32; 3],
    /// Original index (for debugging)
    pub orig_idx: usize,
}

/// Tiled rasterization renderer
pub struct TiledRenderer {
    pub width: usize,
    pub height: usize,
    tile_width: usize,
    tile_height: usize,
    num_tiles_x: usize,
    num_tiles_y: usize,
}

impl TiledRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        let tile_width = 16;
        let tile_height = 16;
        let num_tiles_x = (width + tile_width - 1) / tile_width;
        let num_tiles_y = (height + tile_height - 1) / tile_height;

        Self {
            width,
            height,
            tile_width,
            tile_height,
            num_tiles_x,
            num_tiles_y,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn project_splats(
        &self,
        splats: SplatView<'_>,
        fx: f32,
        fy: f32,
        _cx: f32,
        _cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> Vec<ProjectedGaussian> {
        let mut projected = Vec::with_capacity(splats.opacity_logits.len());
        let sh_row_width = ((splats.sh_degree + 1) * (splats.sh_degree + 1)) * 3;

        for idx in 0..splats.opacity_logits.len() {
            let pos_base = idx * 3;
            let rot_base = idx * 4;
            let sh_base = idx * sh_row_width;
            if let Some(projected_gaussian) = project_projected_gaussian(
                idx,
                [
                    splats.positions[pos_base],
                    splats.positions[pos_base + 1],
                    splats.positions[pos_base + 2],
                ],
                [
                    splats.log_scales[pos_base].exp().abs(),
                    splats.log_scales[pos_base + 1].exp().abs(),
                    splats.log_scales[pos_base + 2].exp().abs(),
                ],
                [
                    splats.rotations[rot_base],
                    splats.rotations[rot_base + 1],
                    splats.rotations[rot_base + 2],
                    splats.rotations[rot_base + 3],
                ],
                1.0 / (1.0 + (-splats.opacity_logits[idx]).exp()),
                [
                    sh0_to_rgb_value(splats.sh_coeffs[sh_base]),
                    sh0_to_rgb_value(splats.sh_coeffs[sh_base + 1]),
                    sh0_to_rgb_value(splats.sh_coeffs[sh_base + 2]),
                ],
                fx,
                fy,
                rotation,
                translation,
            ) {
                projected.push(projected_gaussian);
            }
        }

        projected
    }

    /// Compute bounding box in tiles
    pub fn compute_tile_bounds(
        &self,
        g: &ProjectedGaussian,
        tile_alpha: f32,
    ) -> (usize, usize, usize, usize) {
        // Compute standard deviation
        let sigma_x = (g.cov_xx * tile_alpha).sqrt().max(1.0);
        let sigma_y = (g.cov_yy * tile_alpha).sqrt().max(1.0);

        // Bounding box in pixels
        let x_min = (g.x - 3.0 * sigma_x).max(0.0) as usize;
        let x_max = (g.x + 3.0 * sigma_x).min(self.width as f32 - 1.0) as usize;
        let y_min = (g.y - 3.0 * sigma_y).max(0.0) as usize;
        let y_max = (g.y + 3.0 * sigma_y).min(self.height as f32 - 1.0) as usize;

        // Convert to tile coordinates
        let tile_x_min = x_min / self.tile_width;
        let tile_x_max = (x_max + self.tile_width - 1) / self.tile_width;
        let tile_y_min = y_min / self.tile_height;
        let tile_y_max = (y_max + self.tile_height - 1) / self.tile_height;

        (tile_x_min, tile_x_max, tile_y_min, tile_y_max)
    }

    /// Sort Gaussians by depth (front to back)
    pub fn sort_by_depth(&self, gaussians: &mut [ProjectedGaussian]) {
        gaussians.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Render with tiled rasterization
    ///
    /// Algorithm:
    /// 1. Project Gaussians to 2D
    /// 2. Compute tile bounds
    /// 3. For each tile, sort Gaussians that overlap it
    /// 4. Render with alpha blending (front to back)
    #[cfg(feature = "gpu")]
    pub fn render_splats(
        &self,
        splats: SplatView<'_>,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> RenderBuffer {
        let projected = self.project_splats(splats, fx, fy, cx, cy, rotation, translation);
        self.render_projected(projected)
    }

    fn render_projected(&self, projected: Vec<ProjectedGaussian>) -> RenderBuffer {
        // Initialize output buffers
        let mut color_buf = vec![0.0f32; self.width * self.height * 3];
        let mut depth_buf = vec![f32::MAX; self.width * self.height];
        let mut alpha_buf = vec![0.0f32; self.width * self.height];

        let tile_alpha = 4.0; // Alpha multiplier for tile assignment

        let tile_count = self.num_tiles_x * self.num_tiles_y;
        let mut tile_lists: Vec<Vec<usize>> = vec![Vec::new(); tile_count];

        // Assign Gaussians to tiles based on bounds
        for (idx, g) in projected.iter().enumerate() {
            let (tile_x_min, tile_x_max, tile_y_min, tile_y_max) =
                self.compute_tile_bounds(g, tile_alpha);

            for ty in tile_y_min..tile_y_max {
                for tx in tile_x_min..tile_x_max {
                    let tile_idx = ty * self.num_tiles_x + tx;
                    tile_lists[tile_idx].push(idx);
                }
            }
        }

        // Process each tile independently
        for ty in 0..self.num_tiles_y {
            for tx in 0..self.num_tiles_x {
                let tile_idx = ty * self.num_tiles_x + tx;
                let gaussians_in_tile = &mut tile_lists[tile_idx];
                if gaussians_in_tile.is_empty() {
                    continue;
                }

                // Front-to-back sort for correct alpha blending
                gaussians_in_tile.sort_by(|a, b| {
                    projected[*a]
                        .depth
                        .partial_cmp(&projected[*b].depth)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let px_start = tx * self.tile_width;
                let py_start = ty * self.tile_height;
                let px_end = (px_start + self.tile_width).min(self.width);
                let py_end = (py_start + self.tile_height).min(self.height);

                for &g_idx in gaussians_in_tile.iter() {
                    let g = &projected[g_idx];

                    // Precompute inverse covariance for Mahalanobis distance
                    // Σ⁻¹ = (1/det) * [[cov_yy, -cov_xy], [-cov_xy, cov_xx]]
                    let det = g.cov_xx * g.cov_yy - g.cov_xy * g.cov_xy;
                    if det < 1e-10 {
                        continue;
                    }
                    let inv_det = 1.0 / det;

                    for py in py_start..py_end {
                        for px in px_start..px_end {
                            let idx = py * self.width + px;

                            if alpha_buf[idx] >= 0.999 {
                                continue;
                            }

                            let dx = px as f32 - g.x;
                            let dy = py as f32 - g.y;
                            let d_sq = (g.cov_yy * dx * dx - 2.0 * g.cov_xy * dx * dy
                                + g.cov_xx * dy * dy)
                                * inv_det;

                            if d_sq < 9.0 {
                                let weight = (-0.5 * d_sq).exp() * g.opacity;
                                if weight > 0.001 {
                                    let transmittance = 1.0 - alpha_buf[idx];
                                    let alpha = weight * transmittance;
                                    if alpha > 0.0 {
                                        color_buf[idx * 3] += g.color[0] * alpha;
                                        color_buf[idx * 3 + 1] += g.color[1] * alpha;
                                        color_buf[idx * 3 + 2] += g.color[2] * alpha;
                                        alpha_buf[idx] += alpha;

                                        if depth_buf[idx] == f32::MAX && alpha > 0.01 {
                                            depth_buf[idx] = g.depth;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clamp colors to [0, 1]
        for v in &mut color_buf {
            *v = v.clamp(0.0, 1.0);
        }

        RenderBuffer {
            color: color_buf,
            depth: depth_buf,
            width: self.width,
            height: self.height,
        }
    }
}

fn project_projected_gaussian(
    idx: usize,
    position: [f32; 3],
    scale: [f32; 3],
    rotation_q: [f32; 4],
    opacity: f32,
    color: [f32; 3],
    fx: f32,
    fy: f32,
    camera_rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
) -> Option<ProjectedGaussian> {
    let r = *camera_rotation;
    let wx = position[0];
    let wy = position[1];
    let wz = position[2];

    let cx = r[0][0] * wx + r[0][1] * wy + r[0][2] * wz + translation[0];
    let cy = r[1][0] * wx + r[1][1] * wy + r[1][2] * wz + translation[1];
    let cz = r[2][0] * wx + r[2][1] * wy + r[2][2] * wz + translation[2];
    if cz <= 0.0 {
        return None;
    }

    let px = fx * cx / cz;
    let py = fy * cy / cz;

    let [qw, qx, qy, qz] = rotation_q;
    let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    let r01 = 2.0 * (qx * qy - qw * qz);
    let r02 = 2.0 * (qx * qz + qw * qy);
    let r10 = 2.0 * (qx * qy + qw * qz);
    let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    let r12 = 2.0 * (qy * qz - qw * qx);
    let r20 = 2.0 * (qx * qz - qw * qy);
    let r21 = 2.0 * (qy * qz + qw * qx);
    let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

    let sx = scale[0].abs();
    let sy = scale[1].abs();
    let sz = scale[2].abs();
    let sxx = sx * sx;
    let syy = sy * sy;
    let szz = sz * sz;
    let cw00 = r00 * r00 * sxx + r01 * r01 * syy + r02 * r02 * szz;
    let cw01 = r00 * r10 * sxx + r01 * r11 * syy + r02 * r12 * szz;
    let cw02 = r00 * r20 * sxx + r01 * r21 * syy + r02 * r22 * szz;
    let cw11 = r10 * r10 * sxx + r11 * r11 * syy + r12 * r12 * szz;
    let cw12 = r10 * r20 * sxx + r11 * r21 * syy + r12 * r22 * szz;
    let cw22 = r20 * r20 * sxx + r21 * r21 * syy + r22 * r22 * szz;

    let c = r;
    let m00 = c[0][0] * cw00 + c[0][1] * cw01 + c[0][2] * cw02;
    let m01 = c[0][0] * cw01 + c[0][1] * cw11 + c[0][2] * cw12;
    let m02 = c[0][0] * cw02 + c[0][1] * cw12 + c[0][2] * cw22;
    let m10 = c[1][0] * cw00 + c[1][1] * cw01 + c[1][2] * cw02;
    let m11 = c[1][0] * cw01 + c[1][1] * cw11 + c[1][2] * cw12;
    let m12 = c[1][0] * cw02 + c[1][1] * cw12 + c[1][2] * cw22;
    let m20 = c[2][0] * cw00 + c[2][1] * cw01 + c[2][2] * cw02;
    let m21 = c[2][0] * cw01 + c[2][1] * cw11 + c[2][2] * cw12;
    let m22 = c[2][0] * cw02 + c[2][1] * cw12 + c[2][2] * cw22;
    let cc00 = m00 * c[0][0] + m01 * c[0][1] + m02 * c[0][2];
    let cc01 = m00 * c[1][0] + m01 * c[1][1] + m02 * c[1][2];
    let cc02 = m00 * c[2][0] + m01 * c[2][1] + m02 * c[2][2];
    let cc11 = m10 * c[1][0] + m11 * c[1][1] + m12 * c[1][2];
    let cc12 = m10 * c[2][0] + m11 * c[2][1] + m12 * c[2][2];
    let cc22 = m20 * c[2][0] + m21 * c[2][1] + m22 * c[2][2];

    let inv_z = 1.0 / cz;
    let inv_z2 = inv_z * inv_z;
    let jx0 = fx * inv_z;
    let jx2 = -fx * cx * inv_z2;
    let jy1 = fy * inv_z;
    let jy2 = -fy * cy * inv_z2;
    let cov_xx = jx0 * (cc00 * jx0 + cc02 * jx2) + jx2 * (cc02 * jx0 + cc22 * jx2);
    let cov_xy = jx0 * (cc01 * jy1 + cc02 * jy2) + jx2 * (cc12 * jy1 + cc22 * jy2);
    let cov_yy = jy1 * (cc11 * jy1 + cc12 * jy2) + jy2 * (cc12 * jy1 + cc22 * jy2);

    Some(ProjectedGaussian {
        x: px,
        y: py,
        depth: cz,
        cov_xx,
        cov_xy,
        cov_yy,
        opacity,
        color,
        orig_idx: idx,
    })
}

/// Render output buffer
pub struct RenderBuffer {
    pub color: Vec<f32>, // [H, W, 3] RGB
    pub depth: Vec<f32>, // [H, W]
    pub width: usize,
    pub height: usize,
}

impl RenderBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            color: vec![0.0f32; width * height * 3],
            depth: vec![f32::MAX; width * height],
            width,
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    fn opacity_to_logit(opacity: f32) -> f32 {
        let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
        (clamped / (1.0 - clamped)).ln()
    }

    #[cfg(feature = "gpu")]
    fn single_rgb_splats(
        position: [f32; 3],
        scale: [f32; 3],
        rotation: [f32; 4],
        opacity: f32,
        color: [f32; 3],
    ) -> crate::training::HostSplats {
        crate::training::HostSplats::from_raw_parts(
            position.into(),
            scale.map(f32::ln).into(),
            rotation.into(),
            vec![opacity_to_logit(opacity)],
            color.map(rgb_to_sh0_value).into(),
            0,
        )
        .unwrap()
    }

    #[test]
    fn test_tiled_renderer() {
        let renderer = TiledRenderer::new(640, 480);
        assert_eq!(renderer.num_tiles_x, 40);
        assert_eq!(renderer.num_tiles_y, 30);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_splat_projection() {
        let renderer = TiledRenderer::new(64, 64);
        let splats = single_rgb_splats(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [1.0, 0.5, 0.25],
        );

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let projected = renderer.project_splats(
            splats.as_view(),
            500.0,
            500.0,
            32.0,
            32.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );

        assert!(!projected.is_empty());
        assert!(projected[0].depth > 0.0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_render_splats_matches_project_then_render() {
        let renderer = TiledRenderer::new(64, 64);
        let splats = single_rgb_splats(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [1.0, 0.5, 0.25],
        );
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let projected = renderer.project_splats(
            splats.as_view(),
            500.0,
            500.0,
            32.0,
            32.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );
        let projected_render = renderer.render_projected(projected);
        let splat_render = renderer.render_splats(
            splats.as_view(),
            500.0,
            500.0,
            32.0,
            32.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );

        assert_eq!(projected_render.color, splat_render.color);
        assert_eq!(projected_render.depth, splat_render.depth);
    }

    /// Verify that the full 2D covariance projection is correct.
    ///
    /// For an axis-aligned Gaussian (identity rotation) at [0,0,z] the cross-term
    /// cov_xy should be zero.  For a tilted Gaussian it should be non-zero.
    #[cfg(feature = "gpu")]
    #[test]
    fn test_full_2d_covariance_identity_rotation() {
        let renderer = TiledRenderer::new(64, 64);
        let splats = single_rgb_splats(
            [0.0, 0.0, 2.0],
            [0.1, 0.05, 0.02],
            [1.0, 0.0, 0.0, 0.0], // identity
            0.8,
            [1.0, 0.0, 0.0],
        );

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let projected = renderer.project_splats(
            splats.as_view(),
            500.0,
            500.0,
            0.0,
            0.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );

        assert!(!projected.is_empty());
        let p = &projected[0];
        // With identity rotation and centered projection, cov_xy should be ~0.
        assert!(
            p.cov_xy.abs() < 1e-4,
            "cov_xy should be ~0 for identity rotation at origin, got {}",
            p.cov_xy
        );
        // cov_xx and cov_yy must be positive.
        assert!(p.cov_xx > 0.0, "cov_xx must be positive, got {}", p.cov_xx);
        assert!(p.cov_yy > 0.0, "cov_yy must be positive, got {}", p.cov_yy);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_full_2d_covariance_rotated_gaussian() {
        let renderer = TiledRenderer::new(64, 64);

        // 45-degree rotation around Z axis: quaternion = [cos(π/8), 0, 0, sin(π/8)]
        let angle = std::f32::consts::PI / 4.0;
        let (sin_half, cos_half) = ((angle / 2.0).sin(), (angle / 2.0).cos());

        let splats = single_rgb_splats(
            [0.5, 0.5, 2.0],
            [0.3, 0.05, 0.01],
            [cos_half, 0.0, 0.0, sin_half], // 45° around Z
            0.8,
            [0.0, 1.0, 0.0],
        );

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let projected = renderer.project_splats(
            splats.as_view(),
            500.0,
            500.0,
            0.0,
            0.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );

        assert!(!projected.is_empty());
        let p = &projected[0];
        // For a rotated elongated Gaussian with asymmetric scales, cov_xy != 0.
        assert!(
            p.cov_xy.abs() > 1e-6,
            "cov_xy should be non-zero for a rotated Gaussian, got {}",
            p.cov_xy
        );
    }

    #[test]
    fn test_depth_sorting() {
        let mut gaussians = vec![
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 2.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [1.0, 0.0, 0.0],
                orig_idx: 0,
            },
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 1.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [0.0, 1.0, 0.0],
                orig_idx: 1,
            },
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 3.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [0.0, 0.0, 1.0],
                orig_idx: 2,
            },
        ];

        let renderer = TiledRenderer::new(64, 64);
        renderer.sort_by_depth(&mut gaussians);

        // Should be sorted front to back (near to far)
        assert_eq!(gaussians[0].depth, 1.0);
        assert_eq!(gaussians[1].depth, 2.0);
        assert_eq!(gaussians[2].depth, 3.0);
    }
}
