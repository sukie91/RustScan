//! Analytical Backward Pass for 3D Gaussian Splatting
//!
//! Replaces finite-difference gradient estimation with exact analytical gradients
//! derived from the alpha blending rendering equation:
//!   C_p = Σ_i T_i · α_i · c_i,  where T_i = Π_{j<i}(1 - α_j)
//!
//! This yields ~100x speedup: 1 forward + 1 backward vs ~160 forward passes.
//!
//! The backward pass is parallelized across pixel row-chunks using rayon.
//! Each chunk processes all records with its own per-pixel running state,
//! and gradient contributions are accumulated across chunks.

/// Per-Gaussian data recorded during forward pass for backward computation.
#[derive(Debug, Clone)]
pub struct GaussianRenderRecord {
    /// Index into the original Gaussian array
    pub gaussian_idx: usize,
    /// 2D projected center x
    pub u: f32,
    /// 2D projected center y
    pub v: f32,
    /// 2D scale x (after abs().max(0.5) clamp)
    pub sigma_x: f32,
    /// 2D scale y (after abs().max(0.5) clamp)
    pub sigma_y: f32,
    /// Depth
    pub z: f32,
    /// Opacity after sigmoid + clamp (base_alpha)
    pub base_alpha: f32,
    /// RGB color
    pub color: [f32; 3],
    /// Bounding box
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
    /// Raw 2D scale before abs().max(0.5) — needed for clamp gradient
    pub raw_scale_2d_x: f32,
    pub raw_scale_2d_y: f32,
    /// Raw opacity (sigmoid output) before clamp to [0,1]
    pub raw_opacity: f32,
    /// 3D scale (exp(log_scale))
    pub scale_3d: [f32; 3],
    /// Raw opacity logit (before sigmoid)
    pub opacity_logit: f32,
}

/// Intermediate data from forward pass needed by backward.
pub struct ForwardIntermediate {
    /// Gaussian records sorted by depth (front-to-back)
    pub records: Vec<GaussianRenderRecord>,
    /// Rendered color [H*W*3]
    pub rendered_color: Vec<f32>,
    /// Per-pixel accumulated alpha [H*W]
    pub alpha_acc: Vec<f32>,
    /// Rendered depth [H*W] (normalized by alpha)
    pub rendered_depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Analytical gradients for all Gaussian parameters.
#[derive(Debug, Clone)]
pub struct AnalyticalGradients {
    /// Gradient w.r.t. raw 3D positions [N*3]
    pub positions: Vec<f32>,
    /// Gradient w.r.t. log-scales [N*3]
    pub log_scales: Vec<f32>,
    /// Gradient w.r.t. opacity logits [N]
    pub opacity_logits: Vec<f32>,
    /// Gradient w.r.t. colors [N*3]
    pub colors: Vec<f32>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute analytical gradients via backward pass through the rendering equation.
///
/// Processes Gaussians front-to-back, maintaining running transmittance and
/// accumulated color per pixel. For each Gaussian, computes exact gradients
/// through the alpha blending, Gaussian kernel, projection, and parameterization.
///
/// Parallelized across pixel row-chunks: each chunk processes all records
/// independently with its own per-pixel running state. Gradient contributions
/// are accumulated across chunks.
pub fn backward(
    intermediate: &ForwardIntermediate,
    target_color: &[f32],
    n_gaussians: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> AnalyticalGradients {
    use rayon::prelude::*;

    let w = intermediate.width;
    let h = intermediate.height;

    // Precompute dL/dC_p from L1 loss: sign(rendered - target)
    let dl_dc: Vec<f32> = intermediate
        .rendered_color
        .iter()
        .zip(target_color.iter())
        .map(|(&r, &t)| {
            let diff = r - t;
            if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    // Split rows into chunks for parallel processing.
    // Each chunk processes all records with its own per-pixel running state.
    // Using 32-row chunks reduces temporary gradient allocations by ~8x vs 4-row chunks
    // while still providing good parallelism (e.g. 15 chunks for 480 rows).
    let row_chunks: Vec<std::ops::Range<usize>> = (0..h)
        .step_by(32)
        .map(|start| start..(start + 32).min(h))
        .collect();

    // Process row chunks in parallel
    let chunk_results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = row_chunks
        .par_iter()
        .map(|row_range| {
            let mut grad_pos = vec![0.0f32; n_gaussians * 3];
            let mut grad_log_scale = vec![0.0f32; n_gaussians * 3];
            let mut grad_logit = vec![0.0f32; n_gaussians];
            let mut grad_color = vec![0.0f32; n_gaussians * 3];

            let chunk_pixel_count = w * row_range.len();
            let mut running_s = vec![0.0f32; chunk_pixel_count * 3];
            let mut running_alpha = vec![0.0f32; chunk_pixel_count];

            let row_offset = row_range.start;

            for rec in &intermediate.records {
                let idx = rec.gaussian_idx;
                let z = rec.z;
                if z <= 1e-6 {
                    continue;
                }

                let sig = sigmoid(rec.opacity_logit);
                let sig_deriv = sig * (1.0 - sig);
                let opacity_clamp_pass = rec.raw_opacity >= 0.0 && rec.raw_opacity <= 1.0;
                let sx_clamp_pass = rec.raw_scale_2d_x.abs() >= 0.5;
                let sy_clamp_pass = rec.raw_scale_2d_y.abs() >= 0.5;

                let mut dl_du = 0.0f32;
                let mut dl_dv = 0.0f32;
                let mut dl_dsigma_x = 0.0f32;
                let mut dl_dsigma_y = 0.0f32;
                let mut dl_dbase_alpha = 0.0f32;

                // Clamp y range to this chunk
                let chunk_min_y = rec.min_y.max(row_offset);
                let chunk_max_y = rec.max_y.min(row_range.end - 1);
                if chunk_min_y > chunk_max_y {
                    continue;
                }

                for py in chunk_min_y..=chunk_max_y {
                    let local_py = py - row_offset;
                    for px in rec.min_x..=rec.max_x {
                        let local_pidx = local_py * w + px;
                        let global_pidx = py * w + px;
                        let dx = (px as f32 + 0.5 - rec.u) / rec.sigma_x;
                        let dy = (py as f32 + 0.5 - rec.v) / rec.sigma_y;
                        let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                        let alpha_raw = rec.base_alpha * kernel;
                        let alpha = alpha_raw.clamp(0.0, 0.99);
                        if alpha <= 1e-6 {
                            continue;
                        }

                        let transmittance = 1.0 - running_alpha[local_pidx];
                        let contribution = transmittance * alpha;
                        if contribution <= 1e-8 {
                            continue;
                        }

                        let c3_global = global_pidx * 3;
                        let c3_local = local_pidx * 3;

                        // r_i = remaining color after this Gaussian
                        let r_i = [
                            intermediate.rendered_color[c3_global]
                                - running_s[c3_local]
                                - contribution * rec.color[0],
                            intermediate.rendered_color[c3_global + 1]
                                - running_s[c3_local + 1]
                                - contribution * rec.color[1],
                            intermediate.rendered_color[c3_global + 2]
                                - running_s[c3_local + 2]
                                - contribution * rec.color[2],
                        ];

                        // dC/dα_i = T_i · c_i - R_i / (1 - α_i)
                        let inv_one_minus_alpha = 1.0 / (1.0 - alpha).max(1e-6);
                        let dl_dalpha = (transmittance * rec.color[0]
                            - r_i[0] * inv_one_minus_alpha)
                            * dl_dc[c3_global]
                            + (transmittance * rec.color[1] - r_i[1] * inv_one_minus_alpha)
                                * dl_dc[c3_global + 1]
                            + (transmittance * rec.color[2] - r_i[2] * inv_one_minus_alpha)
                                * dl_dc[c3_global + 2];

                        // dL/dc_i = T_i · α_i · dL/dC_p
                        let gi = idx * 3;
                        grad_color[gi] += contribution * dl_dc[c3_global];
                        grad_color[gi + 1] += contribution * dl_dc[c3_global + 1];
                        grad_color[gi + 2] += contribution * dl_dc[c3_global + 2];

                        // Update running state
                        running_s[c3_local] += contribution * rec.color[0];
                        running_s[c3_local + 1] += contribution * rec.color[1];
                        running_s[c3_local + 2] += contribution * rec.color[2];
                        running_alpha[local_pidx] += contribution;

                        // Chain through alpha clamp
                        if alpha_raw <= 0.0 || alpha_raw >= 0.99 {
                            continue;
                        }

                        dl_dbase_alpha += dl_dalpha * kernel;

                        let dl_dkernel = dl_dalpha * rec.base_alpha;
                        let dk_ddx = kernel * (-dx);
                        let dk_ddy = kernel * (-dy);

                        dl_du += dl_dkernel * dk_ddx * (-1.0 / rec.sigma_x);
                        dl_dv += dl_dkernel * dk_ddy * (-1.0 / rec.sigma_y);

                        if sx_clamp_pass {
                            dl_dsigma_x += dl_dkernel * dk_ddx * (-dx / rec.sigma_x);
                        }
                        if sy_clamp_pass {
                            dl_dsigma_y += dl_dkernel * dk_ddy * (-dy / rec.sigma_y);
                        }
                    }
                }

                // 2D → 3D chain rule
                let inv_z = 1.0 / z;
                let gi = idx * 3;

                grad_pos[gi] += dl_du * fx * inv_z;
                grad_pos[gi + 1] += dl_dv * fy * inv_z;
                let dl_dz = dl_du * (-(rec.u - cx) * inv_z)
                    + dl_dv * (-(rec.v - cy) * inv_z)
                    + dl_dsigma_x * (-rec.raw_scale_2d_x * inv_z)
                    + dl_dsigma_y * (-rec.raw_scale_2d_y * inv_z);
                grad_pos[gi + 2] += dl_dz;

                let dl_dscale3d_x = if sx_clamp_pass {
                    dl_dsigma_x * fx * inv_z
                } else {
                    0.0
                };
                let dl_dscale3d_y = if sy_clamp_pass {
                    dl_dsigma_y * fy * inv_z
                } else {
                    0.0
                };
                grad_log_scale[gi] += dl_dscale3d_x * rec.scale_3d[0];
                grad_log_scale[gi + 1] += dl_dscale3d_y * rec.scale_3d[1];

                if opacity_clamp_pass {
                    grad_logit[idx] += dl_dbase_alpha * sig_deriv;
                }
            }

            (grad_pos, grad_log_scale, grad_logit, grad_color)
        })
        .collect();

    // Accumulate gradients from all row chunks in parallel
    let mut grad_pos = vec![0.0f32; n_gaussians * 3];
    let mut grad_log_scale = vec![0.0f32; n_gaussians * 3];
    let mut grad_logit = vec![0.0f32; n_gaussians];
    let mut grad_color = vec![0.0f32; n_gaussians * 3];

    for (cp, cs, co, cc) in &chunk_results {
        grad_pos
            .iter_mut()
            .zip(cp.iter())
            .for_each(|(a, b)| *a += b);
        grad_log_scale
            .iter_mut()
            .zip(cs.iter())
            .for_each(|(a, b)| *a += b);
        grad_logit
            .iter_mut()
            .zip(co.iter())
            .for_each(|(a, b)| *a += b);
        grad_color
            .iter_mut()
            .zip(cc.iter())
            .for_each(|(a, b)| *a += b);
    }

    AnalyticalGradients {
        positions: grad_pos,
        log_scales: grad_log_scale,
        opacity_logits: grad_logit,
        colors: grad_color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple 1-Gaussian scene and run forward + backward.
    fn single_gaussian_scene(
        color: [f32; 3],
        target: [f32; 3],
        opacity_logit: f32,
    ) -> (ForwardIntermediate, AnalyticalGradients) {
        let w = 4;
        let h = 4;
        let sig = sigmoid(opacity_logit);
        let base_alpha = sig.clamp(0.0, 1.0);
        let sigma = 1.0f32;
        let u = 2.0f32;
        let v = 2.0f32;

        // Forward render
        let mut rendered = vec![0.0f32; w * h * 3];
        let mut alpha_acc = vec![0.0f32; w * h];
        for py in 0..h {
            for px in 0..w {
                let dx = (px as f32 + 0.5 - u) / sigma;
                let dy = (py as f32 + 0.5 - v) / sigma;
                let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                let alpha = (base_alpha * kernel).clamp(0.0, 0.99);
                let pidx = py * w + px;
                let contribution = (1.0 - alpha_acc[pidx]) * alpha;
                rendered[pidx * 3] += contribution * color[0];
                rendered[pidx * 3 + 1] += contribution * color[1];
                rendered[pidx * 3 + 2] += contribution * color[2];
                alpha_acc[pidx] += contribution;
            }
        }

        let rec = GaussianRenderRecord {
            gaussian_idx: 0,
            u,
            v,
            sigma_x: sigma,
            sigma_y: sigma,
            z: 2.0,
            base_alpha,
            color,
            min_x: 0,
            max_x: w - 1,
            min_y: 0,
            max_y: h - 1,
            raw_scale_2d_x: sigma,
            raw_scale_2d_y: sigma,
            raw_opacity: sig,
            scale_3d: [0.1, 0.1, 0.1],
            opacity_logit,
        };

        let target_color: Vec<f32> = (0..w * h).flat_map(|_| target.iter().copied()).collect();

        let inter = ForwardIntermediate {
            records: vec![rec],
            rendered_color: rendered,
            alpha_acc,
            rendered_depth: vec![0.0; w * h],
            width: w,
            height: h,
        };

        let grads = backward(&inter, &target_color, 1, 100.0, 100.0, 2.0, 2.0);
        (inter, grads)
    }

    #[test]
    fn test_zero_gradient_when_rendered_equals_target() {
        let color = [0.5, 0.3, 0.8];
        // Build scene and use the actual rendered output as target
        let (inter, _) = single_gaussian_scene(color, color, 0.0);
        // Re-run backward with rendered == target
        let grads = backward(&inter, &inter.rendered_color, 1, 100.0, 100.0, 2.0, 2.0);
        assert!(
            grads.colors.iter().all(|g| g.abs() < 1e-6),
            "color grads should be zero: {:?}",
            grads.colors
        );
        assert!(
            grads.opacity_logits.iter().all(|g| g.abs() < 1e-6),
            "opacity grads should be zero: {:?}",
            grads.opacity_logits
        );
        assert!(
            grads.positions.iter().all(|g| g.abs() < 1e-6),
            "position grads should be zero: {:?}",
            grads.positions
        );
    }

    #[test]
    fn test_color_gradient_direction() {
        // Rendered color > target => loss gradient is positive => color grad should be positive
        // (moving color down would reduce loss)
        let (_, grads) = single_gaussian_scene([0.8, 0.8, 0.8], [0.2, 0.2, 0.2], 0.5);
        assert!(
            grads.colors[0] > 0.0,
            "color grad[0] should be positive: {}",
            grads.colors[0]
        );
        assert!(
            grads.colors[1] > 0.0,
            "color grad[1] should be positive: {}",
            grads.colors[1]
        );
        assert!(
            grads.colors[2] > 0.0,
            "color grad[2] should be positive: {}",
            grads.colors[2]
        );
    }

    #[test]
    fn test_nonzero_gradients_with_mismatch() {
        let (_, grads) = single_gaussian_scene([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        // With a color mismatch, we should get nonzero gradients
        assert!(grads.colors.iter().any(|g| g.abs() > 1e-6));
        assert!(grads.opacity_logits.iter().any(|g| g.abs() > 1e-6));
    }

    #[test]
    fn test_gradient_sizes() {
        let n = 3;
        let inter = ForwardIntermediate {
            records: vec![],
            rendered_color: vec![0.0; 4 * 4 * 3],
            alpha_acc: vec![0.0; 4 * 4],
            rendered_depth: vec![0.0; 4 * 4],
            width: 4,
            height: 4,
        };
        let target = vec![0.0; 4 * 4 * 3];
        let grads = backward(&inter, &target, n, 100.0, 100.0, 2.0, 2.0);
        assert_eq!(grads.positions.len(), n * 3);
        assert_eq!(grads.log_scales.len(), n * 3);
        assert_eq!(grads.opacity_logits.len(), n);
        assert_eq!(grads.colors.len(), n * 3);
    }

    #[test]
    fn test_analytical_vs_finite_diff_color_gradient() {
        // Verify analytical color gradient matches finite-difference approximation
        let w = 4usize;
        let h = 4usize;
        let opacity_logit = 0.5f32;
        let sig = sigmoid(opacity_logit);
        let base_alpha = sig.clamp(0.0, 1.0);
        let sigma = 1.0f32;
        let u = 2.0f32;
        let v = 2.0f32;
        let color = [0.7f32, 0.3, 0.5];
        let target: Vec<f32> = vec![0.2, 0.6, 0.4].repeat(w * h);

        let render_loss = |c: [f32; 3]| -> f32 {
            let mut rendered = vec![0.0f32; w * h * 3];
            let mut aa = vec![0.0f32; w * h];
            for py in 0..h {
                for px in 0..w {
                    let dx = (px as f32 + 0.5 - u) / sigma;
                    let dy = (py as f32 + 0.5 - v) / sigma;
                    let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                    let alpha = (base_alpha * kernel).clamp(0.0, 0.99);
                    let pidx = py * w + px;
                    let contribution = (1.0 - aa[pidx]) * alpha;
                    rendered[pidx * 3] += contribution * c[0];
                    rendered[pidx * 3 + 1] += contribution * c[1];
                    rendered[pidx * 3 + 2] += contribution * c[2];
                    aa[pidx] += contribution;
                }
            }
            rendered
                .iter()
                .zip(target.iter())
                .map(|(r, t)| (r - t).abs())
                .sum()
        };

        // Analytical gradient
        let (_, grads) = single_gaussian_scene(color, [0.2, 0.6, 0.4], opacity_logit);

        // Finite-difference gradient for color[0]
        let eps = 1e-3f32;
        let mut c_plus = color;
        c_plus[0] += eps;
        let mut c_minus = color;
        c_minus[0] -= eps;
        let fd_grad = (render_loss(c_plus) - render_loss(c_minus)) / (2.0 * eps);

        let rel_err = (grads.colors[0] - fd_grad).abs() / (fd_grad.abs() + 1e-8);
        assert!(
            rel_err < 0.05,
            "analytical={} vs fd={}, rel_err={}",
            grads.colors[0],
            fd_grad,
            rel_err
        );
    }
}
