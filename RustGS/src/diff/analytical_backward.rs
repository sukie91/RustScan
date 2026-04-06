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
    /// 2D scale x (sqrt of cov_xx, after abs().max(0.5) clamp)
    pub sigma_x: f32,
    /// 2D scale y (sqrt of cov_yy, after abs().max(0.5) clamp)
    pub sigma_y: f32,
    /// Cross-term of the 2D covariance Σ[0,1] = Σ[1,0]
    pub cov_xy: f32,
    /// Per-scale-axis x-projection: proj_axis_x[k] = J[0,:] · R_total[:,k]
    /// satisfying cov_xx = Σ_k sk² * proj_axis_x[k]²
    pub proj_axis_x: [f32; 3],
    /// Per-scale-axis y-projection: proj_axis_y[k] = J[1,:] · R_total[:,k]
    /// satisfying cov_yy = Σ_k sk² * proj_axis_y[k]²
    pub proj_axis_y: [f32; 3],
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

const DEPTH_NORMALIZATION_EPS: f32 = 1e-6;

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
    backward_weighted_l1(
        intermediate,
        target_color,
        &[],
        n_gaussians,
        fx,
        fy,
        cx,
        cy,
        1.0,
        0.0,
    )
}

/// Compute analytical gradients for weighted mean-L1 color and depth losses.
pub fn backward_weighted_l1(
    intermediate: &ForwardIntermediate,
    target_color: &[f32],
    target_depth: &[f32],
    n_gaussians: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    color_grad_scale: f32,
    depth_grad_scale: f32,
) -> AnalyticalGradients {
    backward_weighted_l1_from_buffers(
        &intermediate.records,
        &intermediate.rendered_color,
        &intermediate.alpha_acc,
        &intermediate.rendered_depth,
        intermediate.width,
        intermediate.height,
        target_color,
        target_depth,
        n_gaussians,
        fx,
        fy,
        cx,
        cy,
        color_grad_scale,
        depth_grad_scale,
    )
}

/// Compute analytical gradients from raw forward buffers without building a
/// `ForwardIntermediate` wrapper. This is used by the Metal trainer's hot path.
pub fn backward_weighted_l1_from_buffers(
    records: &[GaussianRenderRecord],
    rendered_color: &[f32],
    alpha_acc: &[f32],
    rendered_depth: &[f32],
    width: usize,
    height: usize,
    target_color: &[f32],
    target_depth: &[f32],
    n_gaussians: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    color_grad_scale: f32,
    depth_grad_scale: f32,
) -> AnalyticalGradients {
    use rayon::prelude::*;

    let w = width;
    let h = height;

    assert_eq!(
        rendered_color.len(),
        target_color.len(),
        "color target length mismatch"
    );
    if depth_grad_scale != 0.0 {
        assert_eq!(
            rendered_depth.len(),
            target_depth.len(),
            "depth target length mismatch"
        );
    }

    // Precompute dL/dC_p from weighted mean-L1 loss.
    let dl_dc: Vec<f32> = rendered_color
        .iter()
        .zip(target_color.iter())
        .map(|(&r, &t)| {
            let diff = r - t;
            if diff > 0.0 {
                color_grad_scale
            } else if diff < 0.0 {
                -color_grad_scale
            } else {
                0.0
            }
        })
        .collect();

    let dl_dd: Vec<f32> = if depth_grad_scale == 0.0 {
        vec![0.0; rendered_depth.len()]
    } else {
        rendered_depth
            .iter()
            .zip(target_depth.iter())
            .map(|(&r, &t)| {
                let diff = r - t;
                if diff > 0.0 {
                    depth_grad_scale
                } else if diff < 0.0 {
                    -depth_grad_scale
                } else {
                    0.0
                }
            })
            .collect()
    };

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
            let mut running_depth_num = vec![0.0f32; chunk_pixel_count];

            let row_offset = row_range.start;

            for rec in records {
                let idx = rec.gaussian_idx;
                let z = rec.z;
                if z <= 1e-6 {
                    continue;
                }

                let sig = sigmoid(rec.opacity_logit);
                let sig_deriv = sig * (1.0 - sig);
                let opacity_clamp_pass = rec.raw_opacity >= 0.0 && rec.raw_opacity <= 1.0;

                // Precompute full 2D covariance for Mahalanobis kernel.
                // cov_xx = sigma_x², cov_yy = sigma_y², cov_xy from record.
                let cov_xx = rec.sigma_x * rec.sigma_x;
                let cov_yy = rec.sigma_y * rec.sigma_y;
                let det = cov_xx * cov_yy - rec.cov_xy * rec.cov_xy;
                if det < 1e-10 {
                    continue;
                }
                let inv_det = 1.0 / det;

                let mut dl_du = 0.0f32;
                let mut dl_dv = 0.0f32;
                let mut dl_dcov_xx = 0.0f32;
                let mut dl_dcov_xy = 0.0f32;
                let mut dl_dcov_yy = 0.0f32;
                let mut dl_dbase_alpha = 0.0f32;
                let mut dl_dz = 0.0f32;

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
                        // Full Mahalanobis kernel: kernel = exp(-0.5 * dᵀ Σ⁻¹ d)
                        let dx = px as f32 + 0.5 - rec.u;
                        let dy = py as f32 + 0.5 - rec.v;
                        let d_sq = (cov_yy * dx * dx - 2.0 * rec.cov_xy * dx * dy
                            + cov_xx * dy * dy)
                            * inv_det;
                        let kernel = (-0.5 * d_sq).exp();
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

                        let final_alpha = alpha_acc[global_pidx];
                        let depth_denom = final_alpha + DEPTH_NORMALIZATION_EPS;
                        let final_depth = rendered_depth[global_pidx];
                        let final_depth_num = final_depth * depth_denom;

                        let c3_global = global_pidx * 3;
                        let c3_local = local_pidx * 3;

                        // r_i = remaining color after this Gaussian
                        let r_i = [
                            rendered_color[c3_global]
                                - running_s[c3_local]
                                - contribution * rec.color[0],
                            rendered_color[c3_global + 1]
                                - running_s[c3_local + 1]
                                - contribution * rec.color[1],
                            rendered_color[c3_global + 2]
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
                        let tail_alpha = final_alpha - running_alpha[local_pidx] - contribution;
                        let tail_depth_num =
                            final_depth_num - running_depth_num[local_pidx] - contribution * rec.z;
                        let ddepth_dalpha = if dl_dd[global_pidx] == 0.0 {
                            0.0
                        } else {
                            let dnum_dalpha =
                                transmittance * rec.z - tail_depth_num * inv_one_minus_alpha;
                            let dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
                            (dnum_dalpha * depth_denom - final_depth_num * dalpha_dalpha)
                                / (depth_denom * depth_denom)
                        };
                        let dl_dalpha_total = dl_dalpha + dl_dd[global_pidx] * ddepth_dalpha;
                        let dl_dz_direct = dl_dd[global_pidx] * contribution / depth_denom;

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
                        running_depth_num[local_pidx] += contribution * rec.z;

                        // Chain through alpha clamp
                        if alpha_raw <= 0.0 || alpha_raw >= 0.99 {
                            dl_dz += dl_dz_direct;
                            continue;
                        }

                        dl_dbase_alpha += dl_dalpha_total * kernel;

                        let dl_dkernel = dl_dalpha_total * rec.base_alpha;
                        // Whitened displacement: u_vec = Σ⁻¹ · [dx, dy]
                        let u0 = (cov_yy * dx - rec.cov_xy * dy) * inv_det;
                        let u1 = (-rec.cov_xy * dx + cov_xx * dy) * inv_det;

                        // d(kernel)/d(u) =  kernel * u0  (d(dx)/d(u) = -1, sign flip)
                        // d(kernel)/d(v) =  kernel * u1
                        dl_du += dl_dkernel * kernel * u0;
                        dl_dv += dl_dkernel * kernel * u1;

                        // Covariance gradients via matrix identity:
                        // d(kernel)/d(cov_xx) = 0.5 * kernel * u0²
                        // d(kernel)/d(cov_yy) = 0.5 * kernel * u1²
                        // d(kernel)/d(cov_xy) = kernel * u0 * u1
                        let dk = dl_dkernel * kernel;
                        dl_dcov_xx += dk * 0.5 * u0 * u0;
                        dl_dcov_yy += dk * 0.5 * u1 * u1;
                        dl_dcov_xy += dk * u0 * u1;
                        dl_dz += dl_dz_direct;
                    }
                }

                // 2D → 3D chain rule
                let inv_z = 1.0 / z;
                let gi = idx * 3;

                grad_pos[gi] += dl_du * fx * inv_z;
                grad_pos[gi + 1] += dl_dv * fy * inv_z;
                // z-dependence of covariance: Σ_2D ∝ 1/z², so d(cov)/dz = -2*cov/z
                let dl_dz_via_cov = dl_dcov_xx * (-2.0 * cov_xx * inv_z)
                    + dl_dcov_xy * (-2.0 * rec.cov_xy * inv_z)
                    + dl_dcov_yy * (-2.0 * cov_yy * inv_z);
                let dl_dz_projected = dl_du * (-(rec.u - cx) * inv_z)
                    + dl_dv * (-(rec.v - cy) * inv_z)
                    + dl_dz_via_cov;
                grad_pos[gi + 2] += dl_dz + dl_dz_projected;

                // ── Full chain rule: dL/d(log_scale_k) via full 2D covariance
                //
                // cov_xx = Σ_k sk² * proj_axis_x[k]²
                // cov_xy = Σ_k sk² * proj_axis_x[k] * proj_axis_y[k]
                // cov_yy = Σ_k sk² * proj_axis_y[k]²
                //
                // d(cov_ab)/d(sk²) = proj_axis_a[k] * proj_axis_b[k]
                // d(L)/d(sk²) = dl_dcov_xx*pax[k]² + dl_dcov_xy*pax[k]*pay[k] + dl_dcov_yy*pay[k]²
                // d(L)/d(log_sk) = d(L)/d(sk²) * 2*sk²   (since sk = exp(log_sk))
                let s3 = rec.scale_3d;
                for k in 0..3 {
                    let pax = rec.proj_axis_x[k];
                    let pay = rec.proj_axis_y[k];
                    let dl_dsk2 =
                        dl_dcov_xx * pax * pax + dl_dcov_xy * pax * pay + dl_dcov_yy * pay * pay;
                    grad_log_scale[gi + k] += dl_dsk2 * 2.0 * s3[k] * s3[k];
                }

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

        // For a simple axis-aligned Gaussian: proj_axis_x[0]=fx/z, proj_axis_y[1]=fy/z (others~0)
        let proj_ax = [100.0 / 2.0, 0.0, 0.0];
        let proj_ay = [0.0, 100.0 / 2.0, 0.0];
        let rec = GaussianRenderRecord {
            gaussian_idx: 0,
            u,
            v,
            sigma_x: sigma,
            sigma_y: sigma,
            cov_xy: 0.0,
            proj_axis_x: proj_ax,
            proj_axis_y: proj_ay,
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

    #[test]
    fn test_weighted_depth_gradient_matches_finite_diff() {
        let w = 4usize;
        let h = 4usize;
        let pixel_count = w * h;
        let opacity_logit = 0.5f32;
        let sig = sigmoid(opacity_logit);
        let base_alpha = sig.clamp(0.0, 1.0);
        let scale_3d = 1.2f32;
        let cx = 2.0f32;
        let cy = 2.0f32;
        let target_depth = vec![1.5f32; pixel_count];
        let target_color = vec![0.0f32; pixel_count * 3];

        let build_intermediate = |z: f32| -> ForwardIntermediate {
            let sigma = scale_3d / z;
            let rendered_color = vec![0.0f32; pixel_count * 3];
            let mut alpha_acc = vec![0.0f32; pixel_count];
            let mut rendered_depth = vec![0.0f32; pixel_count];
            for py in 0..h {
                for px in 0..w {
                    let dx = (px as f32 + 0.5 - cx) / sigma;
                    let dy = (py as f32 + 0.5 - cy) / sigma;
                    let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                    let alpha = (base_alpha * kernel).clamp(0.0, 0.99);
                    let pidx = py * w + px;
                    let contribution = (1.0 - alpha_acc[pidx]) * alpha;
                    alpha_acc[pidx] += contribution;
                    rendered_depth[pidx] += contribution * z;
                }
            }
            for (depth, alpha) in rendered_depth.iter_mut().zip(alpha_acc.iter()) {
                *depth /= alpha + DEPTH_NORMALIZATION_EPS;
            }
            // Simple axis-aligned projection axes: proj_ax[0]=fx/z, proj_ay[1]=fy/z
            // (fx=1.0, fy=1.0 as used in backward_weighted_l1 call below)
            let inv_z_local = 1.0 / z;
            let proj_ax_local = [inv_z_local, 0.0, 0.0]; // fx=1.0
            let proj_ay_local = [0.0, inv_z_local, 0.0]; // fy=1.0
            ForwardIntermediate {
                records: vec![GaussianRenderRecord {
                    gaussian_idx: 0,
                    u: cx,
                    v: cy,
                    sigma_x: sigma,
                    sigma_y: sigma,
                    cov_xy: 0.0,
                    proj_axis_x: proj_ax_local,
                    proj_axis_y: proj_ay_local,
                    z,
                    base_alpha,
                    color: [0.0, 0.0, 0.0],
                    min_x: 0,
                    max_x: w - 1,
                    min_y: 0,
                    max_y: h - 1,
                    raw_scale_2d_x: sigma,
                    raw_scale_2d_y: sigma,
                    raw_opacity: sig,
                    scale_3d: [scale_3d, scale_3d, scale_3d],
                    opacity_logit,
                }],
                rendered_color,
                alpha_acc,
                rendered_depth,
                width: w,
                height: h,
            }
        };

        let render_loss = |z: f32| -> f32 {
            let inter = build_intermediate(z);
            inter
                .rendered_depth
                .iter()
                .zip(target_depth.iter())
                .map(|(rendered, target)| (rendered - target).abs())
                .sum::<f32>()
                / pixel_count as f32
                * 0.1
        };

        let inter = build_intermediate(2.0);
        let grads = backward_weighted_l1(
            &inter,
            &target_color,
            &target_depth,
            1,
            1.0,
            1.0,
            cx,
            cy,
            0.0,
            0.1 / pixel_count as f32,
        );

        let eps = 1e-3f32;
        let fd_grad = (render_loss(2.0 + eps) - render_loss(2.0 - eps)) / (2.0 * eps);
        let rel_err = (grads.positions[2] - fd_grad).abs() / (fd_grad.abs() + 1e-8);
        assert!(
            rel_err < 0.1,
            "analytical={} vs fd={}, rel_err={}",
            grads.positions[2],
            fd_grad,
            rel_err
        );
    }

    /// Verify that a tilted Gaussian with non-zero cov_xy produces non-zero scale
    /// gradients for all three axes (not just x/y as in the diagonal approximation).
    ///
    /// When cov_xy ≠ 0 the off-diagonal gradient couples scale_x and scale_y.
    /// A rotated Gaussian (45°) with sx ≠ sy should produce non-zero scale gradients.
    #[test]
    fn test_cov_xy_gradient_nonzero_for_rotated_gaussian() {
        use crate::diff::diff_splat::{DiffCamera, DiffSplatRenderer, TrainableGaussians};
        use candle_core::Device;

        let device = Device::Cpu;
        let w = 8usize;
        let h = 8usize;

        // One elongated Gaussian at (0,0,3) rotated 45° around z so cov_xy ≠ 0
        let angle = std::f32::consts::FRAC_PI_4;
        let gaussians = TrainableGaussians::new(
            &[0.0f32, 0.0, 3.0],
            &[
                (-2.0f32).exp().ln(),
                (-4.0f32).exp().ln(),
                (-4.0f32).exp().ln(),
            ], // log-scales
            &[(angle / 2.0).cos(), 0.0, 0.0, (angle / 2.0).sin()], // quaternion [w,x,y,z]
            &[0.5f32],                                             // opacity logit
            &[0.8f32, 0.4, 0.2],
            &device,
        )
        .unwrap();

        let camera = DiffCamera::new(
            50.0,
            50.0,
            (w / 2) as f32,
            (h / 2) as f32,
            w,
            h,
            &[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            &[0., 0., 0.],
            &device,
        )
        .unwrap();

        let mut renderer = DiffSplatRenderer::with_device(w, h, device.clone());
        let (_, inter) = renderer
            .render_with_intermediates(&gaussians, &camera)
            .unwrap();

        assert_eq!(inter.records.len(), 1);
        let rec = &inter.records[0];

        // Rotated elongated Gaussian → cov_xy should be non-zero
        assert!(
            rec.cov_xy.abs() > 1e-4,
            "cov_xy should be non-zero for rotated Gaussian, got {}",
            rec.cov_xy
        );

        // Backward pass should produce non-zero scale gradients
        let target = vec![0.0f32; w * h * 3];
        let grads = backward(
            &inter, &target, 1, camera.fx, camera.fy, camera.cx, camera.cy,
        );
        let scale_grad_magnitude: f32 = grads.log_scales.iter().map(|g: &f32| g.abs()).sum();
        assert!(
            scale_grad_magnitude > 1e-6,
            "scale gradients should be non-zero for rotated Gaussian, got {:?}",
            grads.log_scales
        );
    }
}
