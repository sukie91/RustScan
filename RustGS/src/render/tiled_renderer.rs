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
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "gpu")]
use crate::core::{GaussianCamera, SplatView};
#[cfg(feature = "gpu")]
use crate::sh::{evaluate_sh_rgb, sh0_to_rgb_value};

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
    raster_cov_blur: f32,
}

impl TiledRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        let tile_width = 16;
        let tile_height = 16;
        let num_tiles_x = width.div_ceil(tile_width);
        let num_tiles_y = height.div_ceil(tile_height);

        Self {
            width,
            height,
            tile_width,
            tile_height,
            num_tiles_x,
            num_tiles_y,
            raster_cov_blur: crate::training::DEFAULT_RASTER_COV_BLUR,
        }
    }

    pub fn with_raster_cov_blur(mut self, raster_cov_blur: f32) -> Self {
        self.raster_cov_blur = raster_cov_blur.max(0.0);
        self
    }

    #[cfg(feature = "gpu")]
    pub fn project_splats(
        &self,
        splats: SplatView<'_>,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
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
                cx,
                cy,
                rotation,
                translation,
                self.raster_cov_blur,
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
        let tile_x_max = x_max.div_ceil(self.tile_width);
        let tile_y_min = y_min / self.tile_height;
        let tile_y_max = y_max.div_ceil(self.tile_height);

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
    pub fn render_camera_splats(
        &self,
        splats: SplatView<'_>,
        camera: &GaussianCamera,
        background: [f32; 3],
    ) -> RenderBuffer {
        let projected = self.project_camera_splats(splats, camera);
        self.render_projected_with_background(projected, background)
    }

    #[cfg(feature = "gpu")]
    fn project_camera_splats(
        &self,
        splats: SplatView<'_>,
        camera: &GaussianCamera,
    ) -> Vec<ProjectedGaussian> {
        let rotation = camera_rotation_rows(camera);
        let translation = camera.extrinsics.translation();
        let camera_position = camera.position();
        let camera_position = [camera_position.x, camera_position.y, camera_position.z];
        let sh_row_width = ((splats.sh_degree + 1) * (splats.sh_degree + 1)) * 3;
        let mut projected = Vec::with_capacity(splats.opacity_logits.len());

        for idx in 0..splats.opacity_logits.len() {
            let pos_base = idx * 3;
            let rot_base = idx * 4;
            let sh_base = idx * sh_row_width;
            let position = [
                splats.positions[pos_base],
                splats.positions[pos_base + 1],
                splats.positions[pos_base + 2],
            ];
            let viewdir = normalize3([
                position[0] - camera_position[0],
                position[1] - camera_position[1],
                position[2] - camera_position[2],
            ]);
            let color = evaluate_sh_rgb(
                splats
                    .sh_coeffs
                    .get(sh_base..sh_base + sh_row_width)
                    .unwrap_or(&[]),
                splats.sh_degree,
                viewdir,
            );

            if let Some(projected_gaussian) = project_projected_gaussian(
                idx,
                position,
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
                color,
                camera.intrinsics.fx,
                camera.intrinsics.fy,
                camera.intrinsics.cx,
                camera.intrinsics.cy,
                &rotation,
                &translation,
                self.raster_cov_blur,
            ) {
                projected.push(projected_gaussian);
            }
        }

        projected
    }

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
        self.render_projected_with_background(projected, [0.0, 0.0, 0.0])
    }

    #[cfg(test)]
    fn render_projected(&self, projected: Vec<ProjectedGaussian>) -> RenderBuffer {
        self.render_projected_with_background(projected, [0.0, 0.0, 0.0])
    }

    fn render_projected_with_background(
        &self,
        projected: Vec<ProjectedGaussian>,
        background: [f32; 3],
    ) -> RenderBuffer {
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

                            let dx = g.x - (px as f32 + 0.5);
                            let dy = g.y - (py as f32 + 0.5);
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

        for idx in 0..alpha_buf.len() {
            let transmittance = 1.0 - alpha_buf[idx];
            color_buf[idx * 3] += background[0] * transmittance;
            color_buf[idx * 3 + 1] += background[1] * transmittance;
            color_buf[idx * 3 + 2] += background[2] * transmittance;
        }

        for value in &mut color_buf {
            *value = value.clamp(0.0, 1.0);
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
    px_center_x: f32,
    px_center_y: f32,
    camera_rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
    raster_cov_blur: f32,
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

    let px = fx * cx / cz + px_center_x;
    let py = fy * cy / cz + px_center_y;

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
    let mut cov_xx = jx0 * (cc00 * jx0 + cc02 * jx2) + jx2 * (cc02 * jx0 + cc22 * jx2);
    let cov_xy = jx0 * (cc01 * jy1 + cc02 * jy2) + jx2 * (cc12 * jy1 + cc22 * jy2);
    let mut cov_yy = jy1 * (cc11 * jy1 + cc12 * jy2) + jy2 * (cc12 * jy1 + cc22 * jy2);
    let det_raw = cov_xx * cov_yy - cov_xy * cov_xy;
    cov_xx += raster_cov_blur.max(0.0);
    cov_yy += raster_cov_blur.max(0.0);
    let det_blurred = (cov_xx * cov_yy - cov_xy * cov_xy).max(1e-12);
    if det_raw <= 0.0 {
        return None;
    }
    let opacity = opacity * (det_raw / det_blurred).sqrt();
    if opacity < 1.0 / 255.0 {
        return None;
    }

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

#[cfg(feature = "gpu")]
fn camera_rotation_rows(camera: &GaussianCamera) -> [[f32; 3]; 3] {
    let col0 = camera.extrinsics.transform_vector(&[1.0, 0.0, 0.0]);
    let col1 = camera.extrinsics.transform_vector(&[0.0, 1.0, 0.0]);
    let col2 = camera.extrinsics.transform_vector(&[0.0, 0.0, 1.0]);
    [
        [col0[0], col1[0], col2[0]],
        [col0[1], col1[1], col2[1]],
        [col0[2], col1[2], col2[2]],
    ]
}

#[cfg(feature = "gpu")]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
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
mod tests;
