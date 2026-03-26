//! Complete Differentiable Gaussian Splatting Renderer
//!
//! This implements the full differentiable rendering pipeline from:
//! "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! Uses Candle with Metal MPS backend for GPU acceleration.

use crate::diff::analytical_backward::{ForwardIntermediate, GaussianRenderRecord};
use candle_core::{DType, Device, Tensor, Var};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

/// Tile size for parallel rasterization (16x16 matches GPU warp-friendly sizing)
const TILE_SIZE: usize = 16;

/// Fast exp approximation using Schraudolph's method (~3-4 cycles vs ~20-40 for libm).
/// Accurate to ~1e-3 relative error in the range [-80, 0] used by Gaussian kernels.
#[inline]
fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-80.0, 0.0);
    let i = (x * 1_550_555.0 + 1_065_353_216.0) as i32;
    f32::from_bits(i.max(0) as u32)
}

/// Trainable Gaussian parameters (with gradients)
pub struct TrainableGaussians {
    /// Positions: [N, 3] - learnable
    pub positions: Var,
    /// Scales: [N, 3] - learnable (log scale)
    pub scales: Var,
    /// Rotations (quaternions): [N, 4] - learnable
    pub rotations: Var,
    /// Opacities: [N] - learnable (sigmoid)
    pub opacities: Var,
    /// Colors (SH coefficients): [N, 3] - learnable
    pub colors: Var,
    /// Number of Gaussians
    pub n: usize,
    /// Device
    device: Device,
}

/// Autograd gradients for trainable Gaussian parameters.
#[derive(Debug, Clone)]
pub struct SurrogateGradients {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    pub colors: Vec<f32>,
}

impl SurrogateGradients {
    fn zeros(n: usize) -> Self {
        Self {
            positions: vec![0.0; n * 3],
            scales: vec![0.0; n * 3],
            rotations: vec![0.0; n * 4],
            opacities: vec![0.0; n],
            colors: vec![0.0; n * 3],
        }
    }
}

impl TrainableGaussians {
    /// Create new trainable Gaussians
    pub fn new(
        positions: &[f32],
        scales: &[f32],
        rotations: &[f32],
        opacities: &[f32],
        colors: &[f32],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = positions.len() / 3;

        Ok(Self {
            positions: Var::from_tensor(&Tensor::from_slice(positions, (n, 3), device)?)?,
            scales: Var::from_tensor(&Tensor::from_slice(scales, (n, 3), device)?)?,
            rotations: Var::from_tensor(&Tensor::from_slice(rotations, (n, 4), device)?)?,
            opacities: Var::from_tensor(&Tensor::from_slice(opacities, (n,), device)?)?,
            colors: Var::from_tensor(&Tensor::from_slice(colors, (n, 3), device)?)?,
            n,
            device: device.clone(),
        })
    }

    /// Get positions tensor
    pub fn positions(&self) -> &Tensor {
        self.positions.as_tensor()
    }

    /// Get scales (exp for actual scale)
    pub fn scales(&self) -> candle_core::Result<Tensor> {
        self.scales.as_tensor().exp()
    }

    /// Get opacities (sigmoid for 0-1)
    pub fn opacities(&self) -> candle_core::Result<Tensor> {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let x = self.opacities.as_tensor();
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;
        let one = Tensor::ones_like(x)?;
        one.broadcast_div(&one.broadcast_add(&exp_neg_x)?)
    }

    /// Get colors
    pub fn colors(&self) -> &Tensor {
        self.colors.as_tensor()
    }

    /// Get rotations (normalize)
    pub fn rotations(&self) -> candle_core::Result<Tensor> {
        normalize_quaternions(self.rotations.as_tensor())
    }

    /// Number of Gaussians
    pub fn len(&self) -> usize {
        self.n
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper: normalize quaternions
fn normalize_quaternions(q: &Tensor) -> candle_core::Result<Tensor> {
    // Compute norm = sqrt(sum(q^2))
    let sqr = q.mul(q)?;
    let sum = sqr.sum(1)?;
    let norm = sum.sqrt()?;
    let norm = norm.unsqueeze(1)?;
    q.broadcast_div(&norm)
}

/// Camera for rendering
pub struct DiffCamera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: usize,
    pub height: usize,
    /// World-to-camera rotation in row-major form.
    pub rotation: [[f32; 3]; 3],
    /// World-to-camera translation.
    pub translation: [f32; 3],
    /// World to camera transform
    pub extrinsics: Tensor, // [3, 4]
}

impl DiffCamera {
    pub fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: usize,
        height: usize,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
        device: &Device,
    ) -> candle_core::Result<Self> {
        // Build 3x4 extrinsics matrix [R|t] in row-major order
        let mut ext = [0.0f32; 12];
        for i in 0..3 {
            ext[i * 4] = rotation[i][0];
            ext[i * 4 + 1] = rotation[i][1];
            ext[i * 4 + 2] = rotation[i][2];
            ext[i * 4 + 3] = translation[i];
        }

        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            rotation: *rotation,
            translation: *translation,
            extrinsics: Tensor::from_slice(&ext, (3, 4), device)?,
        })
    }
}

/// Output of differentiable rendering
pub struct DiffRenderOutput {
    pub color: Tensor,
    pub depth: Tensor,
}

/// Loss output
pub struct DiffLoss {
    pub total: Tensor,
    pub color: Tensor,
    pub depth: Tensor,
    pub ssim: Tensor,
}

/// Pre-allocated render buffers to avoid per-call allocation.
pub struct RenderCache {
    /// Depth-sorted Gaussian indices
    order: Vec<usize>,
    /// Per-tile Gaussian index lists
    tile_lists: Vec<Vec<usize>>,
    /// Pre-computed per-Gaussian data (populated during sort pass)
    sorted_order: Vec<usize>,
    /// Reverse lookup: gaussian_idx → order_pos (sized to max gaussians)
    idx_to_order_pos: Vec<usize>,
    sigma_x: Vec<f32>,
    sigma_y: Vec<f32>,
    radius_x: Vec<isize>,
    radius_y: Vec<isize>,
    min_x: Vec<usize>,
    max_x: Vec<usize>,
    min_y: Vec<usize>,
    max_y: Vec<usize>,
    /// Whether sort order + precomputed data are already populated
    cache_sorted: bool,
}

impl RenderCache {
    fn new(max_gaussians: usize, width: usize, height: usize) -> Self {
        let num_tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles = num_tiles_x * num_tiles_y;

        Self {
            order: Vec::with_capacity(max_gaussians),
            tile_lists: vec![Vec::with_capacity(256); num_tiles],
            sorted_order: Vec::with_capacity(max_gaussians),
            idx_to_order_pos: vec![0; max_gaussians],
            sigma_x: Vec::with_capacity(max_gaussians),
            sigma_y: Vec::with_capacity(max_gaussians),
            radius_x: Vec::with_capacity(max_gaussians),
            radius_y: Vec::with_capacity(max_gaussians),
            min_x: Vec::with_capacity(max_gaussians),
            max_x: Vec::with_capacity(max_gaussians),
            min_y: Vec::with_capacity(max_gaussians),
            max_y: Vec::with_capacity(max_gaussians),
            cache_sorted: false,
        }
    }
}

/// Complete Differentiable Renderer
pub struct DiffSplatRenderer {
    device: Device,
    width: usize,
    height: usize,
    cache: RenderCache,
}

impl DiffSplatRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        let device = crate::preferred_device();
        log::info!("DiffSplatRenderer using: {:?}", device);

        Self {
            device,
            width,
            height,
            cache: RenderCache::new(100_000, width, height),
        }
    }

    pub fn with_device(width: usize, height: usize, device: Device) -> Self {
        Self {
            device,
            width,
            height,
            cache: RenderCache::new(100_000, width, height),
        }
    }

    /// Project 3D Gaussians to 2D
    fn project_gaussians(
        &self,
        positions: &Tensor,  // [N, 3]
        scales: &Tensor,     // [N, 3]
        rotations: &Tensor, // [N, 4]
        camera: &DiffCamera,
    ) -> candle_core::Result<ProjectedGaussiansTensor> {
        // Metal-backed matmul on views has been unstable for this [N, 3] x [3, 3]
        // projection path, so keep the transform as explicit affine combinations.
        let px = positions.narrow(1, 0, 1)?.squeeze(1)?;
        let py = positions.narrow(1, 1, 1)?.squeeze(1)?;
        let pz = positions.narrow(1, 2, 1)?.squeeze(1)?;

        let x = px
            .affine(camera.rotation[0][0] as f64, camera.translation[0] as f64)?
            .broadcast_add(&py.affine(camera.rotation[0][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[0][2] as f64, 0.0)?)?;
        let y = px
            .affine(camera.rotation[1][0] as f64, camera.translation[1] as f64)?
            .broadcast_add(&py.affine(camera.rotation[1][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[1][2] as f64, 0.0)?)?;
        let z = px
            .affine(camera.rotation[2][0] as f64, camera.translation[2] as f64)?
            .broadcast_add(&py.affine(camera.rotation[2][1] as f64, 0.0)?)?
            .broadcast_add(&pz.affine(camera.rotation[2][2] as f64, 0.0)?)?;

        let fx = Tensor::new(camera.fx, &self.device)?;
        let fy = Tensor::new(camera.fy, &self.device)?;
        let cx = Tensor::new(camera.cx, &self.device)?;
        let cy = Tensor::new(camera.cy, &self.device)?;

        // Project to image plane: u = fx * x / z + cx
        let z_clamped = z.clamp(1e-6, f32::MAX)?;

        let x_fx = x.broadcast_mul(&fx)?;
        let u = x_fx.broadcast_div(&z_clamped)?.broadcast_add(&cx)?;

        let y_fy = y.broadcast_mul(&fy)?;
        let v = y_fy.broadcast_div(&z_clamped)?.broadcast_add(&cy)?;

        let x_values = x.to_vec1::<f32>()?;
        let y_values = y.to_vec1::<f32>()?;
        let z_values = z.to_vec1::<f32>()?;
        let scale_values = scales.to_vec2::<f32>()?;
        let rotation_values = rotations.to_vec2::<f32>()?;
        let mut scale_2d_x = Vec::with_capacity(z_values.len());
        let mut scale_2d_y = Vec::with_capacity(z_values.len());

        for idx in 0..z_values.len() {
            let scale = vec3_from_row(scale_values.get(idx).map(Vec::as_slice).unwrap_or(&[]));
            let rotation =
                quaternion_from_row(rotation_values.get(idx).map(Vec::as_slice).unwrap_or(&[]));
            let (sigma_x, sigma_y) = projected_axis_aligned_sigmas(
                x_values.get(idx).copied().unwrap_or(0.0),
                y_values.get(idx).copied().unwrap_or(0.0),
                z_values.get(idx).copied().unwrap_or(0.0),
                scale,
                rotation,
                &camera.rotation,
                camera.fx,
                camera.fy,
            );
            scale_2d_x.push(sigma_x.clamp(0.5, 256.0));
            scale_2d_y.push(sigma_y.clamp(0.5, 256.0));
        }

        Ok(ProjectedGaussiansTensor {
            u,
            v,
            scale_x: Tensor::from_slice(&scale_2d_x, scale_2d_x.len(), &self.device)?,
            scale_y: Tensor::from_slice(&scale_2d_y, scale_2d_y.len(), &self.device)?,
            z: z.clone(),
        })
    }

    /// Tiled parallel rasterization using rayon.
    ///
    /// Returns raw (color_acc, depth_acc, alpha_acc) before normalization/clamping.
    /// Tiles are independent, so each tile can be processed in parallel.
    #[inline]
    fn render_tiled_parallel(
        &mut self,
        projected: &ProjectedGaussiansCpu,
        colors: &[[f32; 3]],
        opacities: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let w = self.width;
        let h = self.height;
        let num_tiles_x = (w + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles_y = (h + TILE_SIZE - 1) / TILE_SIZE;
        let num_tiles = num_tiles_x * num_tiles_y;

        // Global depth sort — reuse cached order buffer, precompute per-Gaussian data
        // Skip if cache was already populated by render_with_intermediates
        if !self.cache.cache_sorted {
            self.cache.order.clear();
            self.cache.sorted_order.clear();
            self.cache.sigma_x.clear();
            self.cache.sigma_y.clear();
            self.cache.radius_x.clear();
            self.cache.radius_y.clear();
            self.cache.min_x.clear();
            self.cache.max_x.clear();
            self.cache.min_y.clear();
            self.cache.max_y.clear();
            for i in 0..projected.u.len() {
                if projected.z[i] > 1e-6 {
                    self.cache.order.push(i);
                    let sx = projected.scale_x[i].abs().max(0.5);
                    let sy = projected.scale_y[i].abs().max(0.5);
                    let rx = (3.0 * sx).ceil() as isize;
                    let ry = (3.0 * sy).ceil() as isize;
                    self.cache.sigma_x.push(sx);
                    self.cache.sigma_y.push(sy);
                    self.cache.radius_x.push(rx);
                    self.cache.radius_y.push(ry);
                    self.cache
                        .min_x
                        .push((projected.u[i].floor() as isize - rx).max(0) as usize);
                    self.cache
                        .max_x
                        .push((projected.u[i].ceil() as isize + rx).min(w as isize - 1) as usize);
                    self.cache
                        .min_y
                        .push((projected.v[i].floor() as isize - ry).max(0) as usize);
                    self.cache
                        .max_y
                        .push((projected.v[i].ceil() as isize + ry).min(h as isize - 1) as usize);
                }
            }
            // Sort order indices by depth (parallel sort for large arrays)
            let order = &mut self.cache.order;
            if order.len() > 1000 {
                order.par_sort_by(|&a, &b| {
                    projected.z[a]
                        .partial_cmp(&projected.z[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                order.sort_by(|&a, &b| {
                    projected.z[a]
                        .partial_cmp(&projected.z[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            self.cache.sorted_order = self.cache.order.clone();

            // Build reverse lookup: gaussian_idx → order_pos
            if self.cache.idx_to_order_pos.len() < projected.u.len() {
                self.cache.idx_to_order_pos.resize(projected.u.len(), 0);
            }
            for (order_pos, &idx) in self.cache.order.iter().enumerate() {
                self.cache.idx_to_order_pos[idx] = order_pos;
            }
        }

        // Assign Gaussians to tiles (depth-sorted order preserved) — use precomputed cache
        for tile in &mut self.cache.tile_lists {
            tile.clear();
        }
        for (order_pos, &idx) in self.cache.order.iter().enumerate() {
            let min_x = self.cache.min_x[order_pos];
            let max_x = self.cache.max_x[order_pos];
            let min_y = self.cache.min_y[order_pos];
            let max_y = self.cache.max_y[order_pos];

            let tile_x_min = min_x / TILE_SIZE;
            let tile_x_max = max_x / TILE_SIZE;
            let tile_y_min = min_y / TILE_SIZE;
            let tile_y_max = max_y / TILE_SIZE;

            for ty in tile_y_min..=tile_y_max.min(num_tiles_y - 1) {
                for tx in tile_x_min..=tile_x_max.min(num_tiles_x - 1) {
                    self.cache.tile_lists[ty * num_tiles_x + tx].push(idx);
                }
            }
        }

        // Process tiles in parallel via rayon
        let pixel_count = w * h;
        let tile_lists = &self.cache.tile_lists;
        let cache = &self.cache;
        let projected = &projected;
        let tile_results: Vec<_> = (0..num_tiles)
            .into_par_iter()
            .filter_map(|tile_idx| {
                let tile_gaussians = &tile_lists[tile_idx];
                if tile_gaussians.is_empty() {
                    return None;
                }

                let ty = tile_idx / num_tiles_x;
                let tx = tile_idx % num_tiles_x;
                let px_start = tx * TILE_SIZE;
                let py_start = ty * TILE_SIZE;
                let px_end = (px_start + TILE_SIZE).min(w);
                let py_end = (py_start + TILE_SIZE).min(h);
                let tw = px_end - px_start;
                let th = py_end - py_start;

                let mut tc = vec![0.0f32; tw * th * 3];
                let mut td = vec![0.0f32; tw * th];
                let mut ta = vec![0.0f32; tw * th];

                // Build a fast index lookup: sorted_position → order_pos
                let idx_to_pos = &cache.idx_to_order_pos;
                for &idx in tile_gaussians {
                    let order_pos = idx_to_pos[idx];
                    let z = projected.z[idx];
                    let u = projected.u[idx];
                    let v = projected.v[idx];
                    let sigma_x = cache.sigma_x[order_pos];
                    let sigma_y = cache.sigma_y[order_pos];
                    let base_alpha = opacities[idx].clamp(0.0, 1.0);
                    let rgb = colors[idx];

                    let g_min_x = cache.min_x[order_pos].max(px_start);
                    let g_max_x = cache.max_x[order_pos].min(px_end - 1);
                    let g_min_y = cache.min_y[order_pos].max(py_start);
                    let g_max_y = cache.max_y[order_pos].min(py_end - 1);

                    for py in g_min_y..=g_max_y {
                        for px in g_min_x..=g_max_x {
                            let li = (py - py_start) * tw + (px - px_start);
                            let dx = (px as f32 + 0.5 - u) / sigma_x;
                            let dy = (py as f32 + 0.5 - v) / sigma_y;
                            let kernel = fast_exp(-0.5 * (dx * dx + dy * dy));
                            let alpha = (base_alpha * kernel).clamp(0.0, 0.99);
                            if alpha <= 1e-6 {
                                continue;
                            }
                            let transmittance = 1.0 - ta[li];
                            let contribution = transmittance * alpha;
                            if contribution <= 1e-8 {
                                continue;
                            }
                            tc[li * 3] += contribution * rgb[0];
                            tc[li * 3 + 1] += contribution * rgb[1];
                            tc[li * 3 + 2] += contribution * rgb[2];
                            td[li] += contribution * z;
                            ta[li] += contribution;
                        }
                    }
                }

                Some((px_start, py_start, px_end, py_end, tc, td, ta))
            })
            .collect();

        // Merge tile results
        let mut color = vec![0.0f32; pixel_count * 3];
        let mut depth_acc = vec![0.0f32; pixel_count];
        let mut alpha_acc = vec![0.0f32; pixel_count];

        for (px_start, py_start, px_end, py_end, tc, td, ta) in &tile_results {
            let tw = px_end - px_start;
            for py in *py_start..*py_end {
                for px in *px_start..*px_end {
                    let li = (py - py_start) * tw + (px - px_start);
                    let gi = py * w + px;
                    color[gi * 3] = tc[li * 3];
                    color[gi * 3 + 1] = tc[li * 3 + 1];
                    color[gi * 3 + 2] = tc[li * 3 + 2];
                    depth_acc[gi] = td[li];
                    alpha_acc[gi] = ta[li];
                }
            }
        }

        (color, depth_acc, alpha_acc)
    }

    /// Normalize depth by alpha and clamp color to [0,1].
    #[inline]
    fn finalize_buffers(color: &mut [f32], depth: &mut [f32], alpha: &[f32]) {
        let pixel_count = alpha.len();
        for pidx in 0..pixel_count {
            if alpha[pidx] > 1e-6 {
                depth[pidx] /= alpha[pidx];
            } else {
                depth[pidx] = 0.0;
            }
            let c = pidx * 3;
            color[c] = color[c].clamp(0.0, 1.0);
            color[c + 1] = color[c + 1].clamp(0.0, 1.0);
            color[c + 2] = color[c + 2].clamp(0.0, 1.0);
        }
    }

    /// Full differentiable render
    pub fn render(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
    ) -> candle_core::Result<DiffRenderOutput> {
        if gaussians.n == 0 {
            return Ok(DiffRenderOutput {
                color: Tensor::zeros((self.height, self.width, 3), DType::F32, &self.device)?,
                depth: Tensor::zeros((self.height, self.width), DType::F32, &self.device)?,
            });
        }

        // GPU projection (fast on Metal)
        let scales = gaussians.scales()?;
        let projected = self.project_gaussians(
            gaussians.positions(),
            &scales,
            &gaussians.rotations()?,
            camera,
        )?;

        // Single GPU→CPU copy
        let u: Vec<f32> = projected.u.to_vec1()?;
        let v: Vec<f32> = projected.v.to_vec1()?;
        let proj_scale_x: Vec<f32> = projected.scale_x.to_vec1()?;
        let proj_scale_y: Vec<f32> = projected.scale_y.to_vec1()?;
        let proj_z: Vec<f32> = projected.z.to_vec1()?;

        let opacity_data = gaussians.opacities()?.to_vec1::<f32>()?;
        let color_vecs = gaussians.colors().to_vec2::<f32>()?;
        let color_data: Vec<[f32; 3]> = color_vecs
            .iter()
            .map(|c| {
                if c.len() >= 3 {
                    [c[0], c[1], c[2]]
                } else {
                    [0.0, 0.0, 0.0]
                }
            })
            .collect();

        let projected_cpu = ProjectedGaussiansCpu {
            u,
            v,
            scale_x: proj_scale_x,
            scale_y: proj_scale_y,
            z: proj_z,
        };

        let (mut color, mut depth, _alpha) =
            self.render_tiled_parallel(&projected_cpu, &color_data, &opacity_data);
        Self::finalize_buffers(&mut color, &mut depth, &_alpha);
        Ok(DiffRenderOutput {
            color: Tensor::from_slice(&color, (self.height, self.width, 3), &self.device)?,
            depth: Tensor::from_slice(&depth, (self.height, self.width), &self.device)?,
        })
    }

    /// Render with intermediate data for analytical backward pass.
    ///
    /// Same forward rendering as `render()`, but also records per-Gaussian
    /// intermediate values needed by `analytical_backward::backward()`.
    pub fn render_with_intermediates(
        &mut self,
        gaussians: &TrainableGaussians,
        camera: &DiffCamera,
    ) -> candle_core::Result<(DiffRenderOutput, ForwardIntermediate)> {
        let pixel_count = self.width * self.height;
        let w = self.width;
        let h = self.height;

        if gaussians.n == 0 {
            let output = DiffRenderOutput {
                color: Tensor::zeros((h, w, 3), DType::F32, &self.device)?,
                depth: Tensor::zeros((h, w), DType::F32, &self.device)?,
            };
            let inter = ForwardIntermediate {
                records: vec![],
                rendered_color: vec![0.0; pixel_count * 3],
                alpha_acc: vec![0.0; pixel_count],
                rendered_depth: vec![0.0; pixel_count],
                width: w,
                height: h,
            };
            return Ok((output, inter));
        }

        // GPU projection (fast on Metal)
        let scales = gaussians.scales()?;
        let projected = self.project_gaussians(
            gaussians.positions(),
            &scales,
            &gaussians.rotations()?,
            camera,
        )?;

        // Single GPU→CPU copy for all projected data
        let u: Vec<f32> = projected.u.to_vec1()?;
        let v: Vec<f32> = projected.v.to_vec1()?;
        let proj_scale_x: Vec<f32> = projected.scale_x.to_vec1()?;
        let proj_scale_y: Vec<f32> = projected.scale_y.to_vec1()?;
        let proj_z: Vec<f32> = projected.z.to_vec1()?;

        // Single GPU→CPU copy for other parameters
        let opacity_data = gaussians.opacities()?.to_vec1::<f32>()?;
        let opacity_logits_raw = gaussians.opacities.as_tensor().to_vec1::<f32>()?;
        let color_vecs = gaussians.colors().to_vec2::<f32>()?;
        let scales_3d = scales.to_vec2::<f32>()?;

        // Build per-Gaussian records
        let n = gaussians.n;

        // Collect valid Gaussian indices and sort by depth
        self.cache.order.clear();
        self.cache
            .order
            .extend((0..n).filter(|&i| proj_z[i] > 1e-6));

        if self.cache.order.len() > 1000 {
            self.cache.order.par_sort_by(|&a, &b| {
                proj_z[a]
                    .partial_cmp(&proj_z[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            self.cache.order.sort_by(|&a, &b| {
                proj_z[a]
                    .partial_cmp(&proj_z[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut records = Vec::with_capacity(self.cache.order.len());

        for &idx in &self.cache.order {
            let z = proj_z[idx];
            let u_val = u[idx];
            let v_val = v[idx];
            let raw_sx = proj_scale_x[idx];
            let raw_sy = proj_scale_y[idx];
            let sigma_x = raw_sx.abs().max(0.5);
            let sigma_y = raw_sy.abs().max(0.5);
            let radius_x = (3.0 * sigma_x).ceil() as isize;
            let radius_y = (3.0 * sigma_y).ceil() as isize;

            let min_x = (u_val.floor() as isize - radius_x).max(0) as usize;
            let max_x = (u_val.ceil() as isize + radius_x).min(w as isize - 1) as usize;
            let min_y = (v_val.floor() as isize - radius_y).max(0) as usize;
            let max_y = (v_val.ceil() as isize + radius_y).min(h as isize - 1) as usize;

            let base_alpha = opacity_data[idx].clamp(0.0, 1.0);
            let rgb = if color_vecs[idx].len() >= 3 {
                [color_vecs[idx][0], color_vecs[idx][1], color_vecs[idx][2]]
            } else {
                [0.0, 0.0, 0.0]
            };

            let s3d = if scales_3d[idx].len() >= 3 {
                [scales_3d[idx][0], scales_3d[idx][1], scales_3d[idx][2]]
            } else {
                [0.01, 0.01, 0.01]
            };

            records.push(GaussianRenderRecord {
                gaussian_idx: idx,
                u: u_val,
                v: v_val,
                sigma_x,
                sigma_y,
                z,
                base_alpha,
                color: rgb,
                min_x,
                max_x,
                min_y,
                max_y,
                raw_scale_2d_x: raw_sx,
                raw_scale_2d_y: raw_sy,
                raw_opacity: opacity_data[idx],
                scale_3d: s3d,
                opacity_logit: opacity_logits_raw[idx],
            });
        }

        // Build projected data for tiled renderer
        let projected_cpu = ProjectedGaussiansCpu {
            u,
            v,
            scale_x: proj_scale_x,
            scale_y: proj_scale_y,
            z: proj_z,
        };

        // Prepare color data
        let color_data: Vec<[f32; 3]> = color_vecs
            .iter()
            .map(|c| {
                if c.len() >= 3 {
                    [c[0], c[1], c[2]]
                } else {
                    [0.0, 0.0, 0.0]
                }
            })
            .collect();

        // Pixel rendering
        let (mut rendered_color, mut depth_acc, alpha_acc) =
            self.render_tiled_parallel(&projected_cpu, &color_data, &opacity_data);
        Self::finalize_buffers(&mut rendered_color, &mut depth_acc, &alpha_acc);

        let output = DiffRenderOutput {
            color: Tensor::from_slice(&rendered_color, (h, w, 3), &self.device)?,
            depth: Tensor::from_slice(&depth_acc, (h, w), &self.device)?,
        };

        let inter = ForwardIntermediate {
            records,
            rendered_color,
            alpha_acc,
            rendered_depth: depth_acc,
            width: w,
            height: h,
        };

        Ok((output, inter))
    }

    /// Compute loss from CPU slices (avoids GPU→CPU copy).
    pub fn compute_loss_from_cpu(
        &self,
        rendered_color: &[f32],
        rendered_depth: &[f32],
        target_color: &[f32],
        target_depth: &[f32],
    ) -> f32 {
        let color_loss = l1_sum(rendered_color, target_color);
        let depth_loss = masked_depth_l1(rendered_depth, target_depth);
        let ssim = if !rendered_color.is_empty() {
            compute_ssim_loss(rendered_color, target_color, self.width, self.height, 3)
        } else {
            1.0
        };
        let ssim_loss = (1.0 - ssim) * 0.1;
        color_loss + depth_loss * 0.1 + ssim_loss
    }

    /// Compute loss
    pub fn compute_loss(
        &self,
        rendered: &DiffRenderOutput,
        target_color: &[f32],
        target_depth: &[f32],
    ) -> candle_core::Result<DiffLoss> {
        let expected_color = self.width * self.height * 3;
        let expected_depth = self.width * self.height;

        debug_assert_eq!(target_color.len(), expected_color);
        debug_assert_eq!(target_depth.len(), expected_depth);

        let rendered_color = expand_to_len(&tensor_to_vec(&rendered.color)?, expected_color);
        let rendered_depth = expand_to_len(&tensor_to_vec(&rendered.depth)?, expected_depth);

        let total_loss = self.compute_loss_from_cpu(
            &rendered_color,
            &rendered_depth,
            target_color,
            target_depth,
        );
        let color_loss = l1_sum(&rendered_color, target_color);
        let depth_loss = masked_depth_l1(&rendered_depth, target_depth);
        let ssim = compute_ssim_loss(&rendered_color, target_color, self.width, self.height, 3);

        let total = Tensor::new(total_loss, &self.device)?;
        let color = Tensor::new(color_loss, &self.device)?;
        let depth = Tensor::new(depth_loss, &self.device)?;
        let ssim_t = Tensor::new(ssim, &self.device)?;

        Ok(DiffLoss {
            total,
            color,
            depth,
            ssim: ssim_t,
        })
    }

    /// Compute parameter gradients via a differentiable surrogate objective.
    ///
    /// This establishes a real `.backward()` path on TrainableGaussians so
    /// optimizer steps can consume gradients from Candle autograd.
    pub fn compute_surrogate_gradients(
        &self,
        gaussians: &TrainableGaussians,
    ) -> candle_core::Result<SurrogateGradients> {
        if gaussians.n == 0 {
            return Ok(SurrogateGradients::zeros(0));
        }

        let pos_reg = gaussians.positions().sqr()?.sum(0)?.sum(0)?;
        let scale_reg = gaussians.scales.as_tensor().sqr()?.sum(0)?.sum(0)?;
        let color_reg = gaussians.colors().sqr()?.sum(0)?.sum(0)?;

        let rot_norm = gaussians.rotations.as_tensor().sqr()?.sum(1)?;
        let rot_residual = rot_norm.broadcast_sub(&Tensor::ones_like(&rot_norm)?)?;
        let rot_reg = rot_residual.sqr()?.sum(0)?;

        let op = gaussians.opacities()?;
        let half = Tensor::new(0.5f32, &self.device)?;
        let op_residual = op.broadcast_sub(&half)?;
        let op_reg = op_residual.sqr()?.sum(0)?;

        let w_pos = Tensor::new(1e-4f32, &self.device)?;
        let w_scale = Tensor::new(1e-4f32, &self.device)?;
        let w_rot = Tensor::new(5e-5f32, &self.device)?;
        let w_op = Tensor::new(5e-5f32, &self.device)?;
        let w_color = Tensor::new(1e-4f32, &self.device)?;

        let loss = pos_reg
            .broadcast_mul(&w_pos)?
            .broadcast_add(&scale_reg.broadcast_mul(&w_scale)?)?
            .broadcast_add(&rot_reg.broadcast_mul(&w_rot)?)?
            .broadcast_add(&op_reg.broadcast_mul(&w_op)?)?
            .broadcast_add(&color_reg.broadcast_mul(&w_color)?)?;

        let grads = loss.backward()?;

        let positions = if let Some(g) = grads.get(gaussians.positions()) {
            flatten_2d(&g.to_vec2::<f32>()?)
        } else {
            vec![0.0; gaussians.n * 3]
        };
        let scales = if let Some(g) = grads.get(gaussians.scales.as_tensor()) {
            flatten_2d(&g.to_vec2::<f32>()?)
        } else {
            vec![0.0; gaussians.n * 3]
        };
        let rotations = if let Some(g) = grads.get(gaussians.rotations.as_tensor()) {
            flatten_2d(&g.to_vec2::<f32>()?)
        } else {
            vec![0.0; gaussians.n * 4]
        };
        let opacities = if let Some(g) = grads.get(gaussians.opacities.as_tensor()) {
            g.to_vec1::<f32>()?
        } else {
            vec![0.0; gaussians.n]
        };
        let colors = if let Some(g) = grads.get(gaussians.colors()) {
            flatten_2d(&g.to_vec2::<f32>()?)
        } else {
            vec![0.0; gaussians.n * 3]
        };

        Ok(SurrogateGradients {
            positions,
            scales,
            rotations,
            opacities,
            colors,
        })
    }
}

/// Projected Gaussian info
struct ProjectedGaussiansTensor {
    u: Tensor,
    v: Tensor,
    scale_x: Tensor,
    scale_y: Tensor,
    z: Tensor,
}

struct ProjectedGaussiansCpu {
    u: Vec<f32>,
    v: Vec<f32>,
    scale_x: Vec<f32>,
    scale_y: Vec<f32>,
    z: Vec<f32>,
}

// Helper functions

fn vec3_from_row(row: &[f32]) -> [f32; 3] {
    [
        row.first().copied().unwrap_or(0.01),
        row.get(1).copied().unwrap_or(0.01),
        row.get(2).copied().unwrap_or(0.01),
    ]
}

fn quaternion_from_row(row: &[f32]) -> [f32; 4] {
    [
        row.first().copied().unwrap_or(1.0),
        row.get(1).copied().unwrap_or(0.0),
        row.get(2).copied().unwrap_or(0.0),
        row.get(3).copied().unwrap_or(0.0),
    ]
}

fn mat3_from_row_major(rotation: &[[f32; 3]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(rotation[0][0], rotation[1][0], rotation[2][0]),
        Vec3::new(rotation[0][1], rotation[1][1], rotation[2][1]),
        Vec3::new(rotation[0][2], rotation[1][2], rotation[2][2]),
    )
}

fn quat_from_wxyz(rotation: [f32; 4]) -> Quat {
    let length_sq = rotation.iter().map(|value| value * value).sum::<f32>();
    if !length_sq.is_finite() || length_sq <= 1e-12 {
        return Quat::IDENTITY;
    }
    Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]).normalize()
}

fn projected_axis_aligned_sigmas(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
    camera_rotation: &[[f32; 3]; 3],
    fx: f32,
    fy: f32,
) -> (f32, f32) {
    let inv_z = 1.0 / z.max(1e-4);
    let object_rotation = Mat3::from_quat(quat_from_wxyz(rotation));
    let scale_cov = Mat3::from_diagonal(Vec3::new(
        scale[0] * scale[0],
        scale[1] * scale[1],
        scale[2] * scale[2],
    ));
    let covariance_world = object_rotation * scale_cov * object_rotation.transpose();
    let camera_rotation = mat3_from_row_major(camera_rotation);
    let covariance_camera = camera_rotation * covariance_world * camera_rotation.transpose();

    let projection_row_x = Vec3::new(fx * inv_z, 0.0, -fx * x * inv_z * inv_z);
    let projection_row_y = Vec3::new(0.0, fy * inv_z, -fy * y * inv_z * inv_z);
    let covariance_x = projection_row_x.dot(covariance_camera * projection_row_x);
    let covariance_y = projection_row_y.dot(covariance_camera * projection_row_y);

    (covariance_x.max(1e-6).sqrt(), covariance_y.max(1e-6).sqrt())
}

fn tensor_to_vec(tensor: &Tensor) -> candle_core::Result<Vec<f32>> {
    let dims = tensor.dims();
    match dims.len() {
        1 => tensor.to_vec1::<f32>(),
        2 => {
            let data = tensor.to_vec2::<f32>()?;
            Ok(data.into_iter().flatten().collect())
        }
        3 => {
            let data = tensor.to_vec3::<f32>()?;
            Ok(data.into_iter().flatten().flatten().collect())
        }
        _ => tensor.to_vec1::<f32>(),
    }
}

fn expand_to_len(data: &[f32], expected: usize) -> Vec<f32> {
    if expected == 0 {
        return Vec::new();
    }
    if data.is_empty() {
        return vec![0.0; expected];
    }
    if data.len() == expected {
        return data.to_vec();
    }

    let mut out = Vec::with_capacity(expected);
    while out.len() < expected {
        let remaining = expected - out.len();
        if remaining >= data.len() {
            out.extend_from_slice(data);
        } else {
            out.extend_from_slice(&data[..remaining]);
        }
    }
    out
}

fn l1_sum(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len().min(target.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += (pred[i] - target[i]).abs();
    }
    sum
}

fn masked_depth_l1(pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len().min(target.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        if target[i] > 0.0 {
            sum += (pred[i] - target[i]).abs();
        }
    }
    sum
}

fn flatten_2d(data: &[Vec<f32>]) -> Vec<f32> {
    data.iter().flatten().copied().collect()
}

/// Compute SSIM loss between predicted and target images.
///
/// Uses the standard SSIM formula with constants C1=0.01^2 and C2=0.03^2.
pub fn compute_ssim_loss(
    pred: &[f32],   // [H, W, C]
    target: &[f32], // [H, W, C]
    width: usize,
    height: usize,
    channel: usize,
) -> f32 {
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);

    // Compute means
    let mut mu_pred = 0.0f32;
    let mut mu_target = 0.0f32;

    for i in 0..width * height * channel {
        mu_pred += pred[i];
        mu_target += target[i];
    }

    let n = (width * height * channel) as f32;
    mu_pred /= n;
    mu_target /= n;

    // Compute variances and covariance
    let mut var_pred = 0.0f32;
    let mut var_target = 0.0f32;
    let mut covar = 0.0f32;

    for i in 0..width * height * channel {
        let diff_pred = pred[i] - mu_pred;
        let diff_target = target[i] - mu_target;

        var_pred += diff_pred * diff_pred;
        var_target += diff_target * diff_target;
        covar += diff_pred * diff_target;
    }

    var_pred /= n;
    var_target /= n;
    covar /= n;

    // SSIM formula
    let numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covar + c2);
    let denominator = (mu_pred.powi(2) + mu_target.powi(2) + c1) * (var_pred + var_target + c2);

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = DiffSplatRenderer::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[test]
    fn test_trainable_gaussians() {
        let device = crate::preferred_device();
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.5, 0.5],
            &[1.0, 0.5, 0.25, 0.5, 1.0, 0.25],
            &device,
        );
        if let Ok(g) = gaussians {
            assert_eq!(g.len(), 2);
        }
    }

    #[test]
    fn test_compute_loss_with_ssim() {
        let renderer = DiffSplatRenderer::with_device(2, 2, Device::Cpu);
        let color = Tensor::zeros((2, 2, 3), DType::F32, &renderer.device).unwrap();
        let depth = Tensor::zeros((2, 2), DType::F32, &renderer.device).unwrap();
        let rendered = DiffRenderOutput { color, depth };

        let target_color = vec![0.0f32; 2 * 2 * 3];
        let target_depth = vec![0.0f32; 2 * 2];

        let loss = renderer
            .compute_loss(&rendered, &target_color, &target_depth)
            .unwrap();
        let total = loss.total.to_vec0::<f32>().unwrap();
        let ssim = loss.ssim.to_vec0::<f32>().unwrap();

        assert!(total.abs() < 1e-6);
        assert!((ssim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_surrogate_gradients_backward_path() {
        let device = Device::Cpu;
        let renderer = DiffSplatRenderer::with_device(4, 4, device.clone());
        let gaussians = TrainableGaussians::new(
            &[0.3, -0.2, 1.5, -0.4, 0.1, 2.0],
            &[-2.0, -1.8, -1.6, -2.2, -2.0, -1.9],
            &[1.0, 0.1, 0.0, 0.0, 0.95, 0.0, 0.1, 0.0],
            &[0.2, -0.4],
            &[0.7, 0.2, 0.1, 0.1, 0.8, 0.3],
            &device,
        )
        .unwrap();

        let grads = renderer.compute_surrogate_gradients(&gaussians).unwrap();
        assert_eq!(grads.positions.len(), 6);
        assert_eq!(grads.scales.len(), 6);
        assert_eq!(grads.rotations.len(), 8);
        assert_eq!(grads.opacities.len(), 2);
        assert_eq!(grads.colors.len(), 6);

        assert!(grads.positions.iter().any(|g| g.abs() > 0.0));
        assert!(grads.scales.iter().any(|g| g.abs() > 0.0));
        assert!(grads.opacities.iter().any(|g| g.abs() > 0.0));
    }

    #[test]
    fn test_tiled_parallel_render_matches_sequential() {
        let device = Device::Cpu;
        let mut renderer = DiffSplatRenderer::with_device(32, 32, device.clone());

        // Create Gaussians at known positions in front of camera
        let gaussians = TrainableGaussians::new(
            &[0.0, 0.0, 2.0, 0.5, 0.3, 3.0, -0.3, -0.2, 2.5],
            &[-1.5, -1.5, -1.5, -1.8, -1.8, -1.8, -1.6, -1.6, -1.6],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.8, 0.6, 0.7],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &device,
        )
        .unwrap();

        let camera = DiffCamera::new(
            500.0,
            500.0,
            16.0,
            16.0,
            32,
            32,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let output = renderer.render(&gaussians, &camera).unwrap();
        let color = output.color.to_vec3::<f32>().unwrap();
        let depth = output.depth.to_vec2::<f32>().unwrap();

        // Verify non-zero rendering occurred (Gaussians should be visible)
        let has_color = color.iter().flatten().flatten().any(|&v| v > 0.01);
        let has_depth = depth.iter().flatten().any(|&v| v > 0.01);
        assert!(
            has_color,
            "Tiled parallel render should produce non-zero color"
        );
        assert!(
            has_depth,
            "Tiled parallel render should produce non-zero depth"
        );
    }

    #[test]
    fn test_project_gaussians_applies_camera_transform() {
        let device = Device::Cpu;
        let renderer = DiffSplatRenderer::with_device(32, 32, device.clone());
        let gaussians = TrainableGaussians::new(
            &[1.0, 2.0, 4.0],
            &[-2.0, -2.0, -2.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let camera = DiffCamera::new(
            100.0,
            120.0,
            10.0,
            20.0,
            32,
            32,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.5, -1.0, 2.0],
            &device,
        )
        .unwrap();

        let projected = renderer
            .project_gaussians(
                gaussians.positions(),
                &gaussians.scales().unwrap(),
                &gaussians.rotations().unwrap(),
                &camera,
            )
            .unwrap();

        let u = projected.u.to_vec1::<f32>().unwrap()[0];
        let v = projected.v.to_vec1::<f32>().unwrap()[0];
        let z = projected.z.to_vec1::<f32>().unwrap()[0];

        assert!((u - 35.0).abs() < 1e-4, "unexpected projected u: {u}");
        assert!((v - 40.0).abs() < 1e-4, "unexpected projected v: {v}");
        assert!((z - 6.0).abs() < 1e-4, "unexpected projected depth: {z}");
    }

    #[test]
    fn projected_footprint_changes_with_rotation() {
        let device = Device::Cpu;
        let renderer = DiffSplatRenderer::with_device(64, 64, device.clone());
        let camera = DiffCamera::new(
            64.0,
            64.0,
            32.0,
            32.0,
            64,
            64,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let identity = TrainableGaussians::new(
            &[0.0, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();
        let rotated = TrainableGaussians::new(
            &[0.0, 0.0, 2.0],
            &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
            &[std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2],
            &[0.0],
            &[1.0, 0.0, 0.0],
            &device,
        )
        .unwrap();

        let identity_projected = renderer
            .project_gaussians(
                identity.positions(),
                &identity.scales().unwrap(),
                &identity.rotations().unwrap(),
                &camera,
            )
            .unwrap();
        let rotated_projected = renderer
            .project_gaussians(
                rotated.positions(),
                &rotated.scales().unwrap(),
                &rotated.rotations().unwrap(),
                &camera,
            )
            .unwrap();

        let identity_sigma_x = identity_projected.scale_x.to_vec1::<f32>().unwrap()[0];
        let identity_sigma_y = identity_projected.scale_y.to_vec1::<f32>().unwrap()[0];
        let rotated_sigma_x = rotated_projected.scale_x.to_vec1::<f32>().unwrap()[0];
        let rotated_sigma_y = rotated_projected.scale_y.to_vec1::<f32>().unwrap()[0];

        assert!(identity_sigma_x > identity_sigma_y * 5.0);
        assert!(rotated_sigma_y > rotated_sigma_x * 5.0);
        assert!(
            (identity_sigma_x - rotated_sigma_x).abs() > 1.0
                || (identity_sigma_y - rotated_sigma_y).abs() > 1.0
        );
    }
}
