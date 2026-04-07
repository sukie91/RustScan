use candle_core::{DType, Device, Tensor};

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians, SH_C0};

use super::metal_forward::{
    finite_difference_sigma_wrt_rotation_component, projected_rows_to_cpu, row_to_quaternion,
    row_to_vec3, ProjectedGaussians, ProjectedTileBins, RenderedFrame,
};
use super::metal_runtime::{MetalRuntime, METAL_TILE_SIZE};

const SH_C1: f32 = 0.488_602_52;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];
const SH_C4: [f32; 9] = [
    2.503_343,
    -1.770_130_8,
    0.946_174_7,
    -0.669_046_5,
    0.105_785_55,
    -0.669_046_5,
    0.473_087_34,
    -1.770_130_8,
    0.625_835_7,
];

pub(crate) struct MetalBackwardGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
    pub opacity_logits: Tensor,
    pub colors: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalBackwardLossScales {
    pub color: f32,
    pub depth: f32,
    pub ssim: f32,
    pub alpha: f32,
}

pub(crate) struct MetalParameterGrads {
    pub positions: Tensor,
    pub log_scales: Tensor,
    pub rotations: Option<Tensor>,
    pub opacity_logits: Tensor,
    pub colors: Tensor,
    pub sh_rest: Tensor,
}

pub(crate) struct MetalBackwardPass {
    pub grads: MetalBackwardGrads,
    pub grad_magnitudes: Vec<f32>,
    pub projected_grad_magnitudes: Vec<f32>,
}

pub(crate) struct MetalBackwardRequest<'a> {
    pub tile_bins: &'a ProjectedTileBins,
    pub n_gaussians: usize,
    pub camera: &'a DiffCamera,
    pub target_color_cpu: &'a [f32],
    pub target_depth_cpu: &'a [f32],
    pub ssim_grads: &'a [f32],
    pub loss_scales: MetalBackwardLossScales,
    pub refresh_target_buffers: bool,
}

pub(crate) fn backward_loss_scales(
    color_weight: f32,
    ssim_weight: f32,
    target_color_len: usize,
    depth_scale: f32,
    enable_alpha_loss: bool,
    pixel_count: usize,
) -> MetalBackwardLossScales {
    MetalBackwardLossScales {
        color: color_weight / target_color_len.max(1) as f32,
        depth: depth_scale,
        ssim: ssim_weight / target_color_len.max(1) as f32,
        alpha: if enable_alpha_loss {
            1.0 / pixel_count.max(1) as f32
        } else {
            0.0
        },
    }
}

pub(crate) fn execute_backward_pass(
    runtime: &mut MetalRuntime,
    request: MetalBackwardRequest<'_>,
) -> candle_core::Result<MetalBackwardPass> {
    runtime.write_ssim_grad(request.ssim_grads)?;
    if request.refresh_target_buffers {
        runtime.write_target_data(
            request.target_color_cpu,
            request.target_depth_cpu,
            request.loss_scales.color,
            request.loss_scales.depth,
            request.loss_scales.ssim,
            request.loss_scales.alpha,
        )?;
    }
    backward_weighted_l1(
        runtime,
        request.tile_bins,
        request.n_gaussians,
        request.camera,
    )
}

pub(crate) fn backward_weighted_l1(
    runtime: &mut MetalRuntime,
    tile_bins: &ProjectedTileBins,
    n_gaussians: usize,
    camera: &DiffCamera,
) -> candle_core::Result<MetalBackwardPass> {
    let (frame, _profile) = runtime.rasterize_backward(
        n_gaussians,
        tile_bins.as_runtime(),
        camera.width,
        camera.height,
    )?;

    let grad_magnitude_tensor = runtime.compute_grad_magnitudes(n_gaussians)?;
    let grad_magnitudes = runtime.read_tensor_flat::<f32>(&grad_magnitude_tensor)?;
    let projected_grad_magnitude_tensor = runtime.compute_projected_grad_magnitudes(n_gaussians)?;
    let projected_grad_magnitudes =
        runtime.read_tensor_flat::<f32>(&projected_grad_magnitude_tensor)?;

    let grads = MetalBackwardGrads {
        positions: frame.grad_positions,
        log_scales: frame.grad_log_scales,
        opacity_logits: frame.grad_opacity_logits,
        colors: frame.grad_colors,
    };

    Ok(MetalBackwardPass {
        grads,
        grad_magnitudes,
        projected_grad_magnitudes,
    })
}

pub(crate) struct MetalParameterGradInputs<'a> {
    pub gaussians: &'a TrainableGaussians,
    pub raw_grads: &'a MetalBackwardGrads,
    pub projected: &'a ProjectedGaussians,
    pub rendered: &'a RenderedFrame,
    pub rendered_color_cpu: &'a [f32],
    pub target_color_cpu: &'a [f32],
    pub target_depth_cpu: &'a [f32],
    pub ssim_grads: &'a [f32],
    pub loss_scales: MetalBackwardLossScales,
    pub camera: &'a DiffCamera,
    pub active_sh_degree: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub include_rotation_grads: bool,
}

pub(crate) fn assemble_parameter_grads(
    device: &Device,
    inputs: MetalParameterGradInputs<'_>,
) -> candle_core::Result<MetalParameterGrads> {
    let (color_parameter_grads, sh_rest_parameter_grads) = parameter_grads_from_render_color_grads(
        device,
        inputs.active_sh_degree,
        inputs.gaussians,
        inputs.projected,
        &inputs.raw_grads.colors,
        inputs.camera,
    )?;
    let rotation_grads = if inputs.include_rotation_grads {
        Some(rotation_parameter_grads(
            device,
            inputs.gaussians,
            inputs.projected,
            inputs.rendered,
            inputs.rendered_color_cpu,
            inputs.target_color_cpu,
            inputs.target_depth_cpu,
            inputs.ssim_grads,
            inputs.loss_scales,
            inputs.camera,
            inputs.render_width,
            inputs.render_height,
        )?)
    } else {
        None
    };

    Ok(MetalParameterGrads {
        positions: inputs.raw_grads.positions.clone(),
        log_scales: inputs.raw_grads.log_scales.clone(),
        rotations: rotation_grads,
        opacity_logits: inputs.raw_grads.opacity_logits.clone(),
        colors: color_parameter_grads,
        sh_rest: sh_rest_parameter_grads,
    })
}

pub(crate) fn rotation_parameter_grads(
    device: &Device,
    gaussians: &TrainableGaussians,
    projected: &ProjectedGaussians,
    rendered: &RenderedFrame,
    rendered_color_cpu: &[f32],
    target_color_cpu: &[f32],
    target_depth_cpu: &[f32],
    ssim_grads: &[f32],
    loss_scales: MetalBackwardLossScales,
    camera: &DiffCamera,
    render_width: usize,
    render_height: usize,
) -> candle_core::Result<Tensor> {
    let row_count = gaussians.len();
    if row_count == 0 || projected.visible_count == 0 {
        return Tensor::zeros((row_count, 4), DType::F32, device);
    }

    let projected_cpu = projected_rows_to_cpu(projected)?;
    if projected_cpu.is_empty() || projected.tile_bins.total_assignments() == 0 {
        return Tensor::zeros((row_count, 4), DType::F32, device);
    }

    let rendered_depth_cpu = rendered.depth.flatten_all()?.to_vec1::<f32>()?;
    let rendered_alpha_cpu = rendered.alpha.flatten_all()?.to_vec1::<f32>()?;
    let raw_rotation_rows = gaussians.rotations.as_tensor().to_vec2::<f32>()?;
    let color_grads = pixel_color_grads(
        rendered_color_cpu,
        target_color_cpu,
        ssim_grads,
        loss_scales,
    );
    let depth_grads = pixel_depth_grads(&rendered_depth_cpu, target_depth_cpu, loss_scales);
    let mut dl_dsigma_x = vec![0.0f32; row_count];
    let mut dl_dsigma_y = vec![0.0f32; row_count];
    let tile_bins = &projected.tile_bins;
    let packed_indices = tile_bins.packed_indices();
    let num_tiles_x = render_width.div_ceil(METAL_TILE_SIZE);

    for &tile_idx in tile_bins.active_tiles() {
        let Some(record) = tile_bins.record(tile_idx) else {
            continue;
        };
        if record.count() == 0 {
            continue;
        }

        let tile_x = tile_idx % num_tiles_x;
        let tile_y = tile_idx / num_tiles_x;
        let min_x = tile_x * METAL_TILE_SIZE;
        let min_y = tile_y * METAL_TILE_SIZE;
        let max_x = (min_x + METAL_TILE_SIZE)
            .min(render_width)
            .saturating_sub(1);
        let max_y = (min_y + METAL_TILE_SIZE)
            .min(render_height)
            .saturating_sub(1);
        let tile_width = max_x.saturating_sub(min_x) + 1;
        let tile_height = max_y.saturating_sub(min_y) + 1;
        let tile_pixel_count = tile_width * tile_height;
        let mut running_s = vec![0.0f32; tile_pixel_count * 3];
        let mut running_alpha = vec![0.0f32; tile_pixel_count];
        let mut running_depth_num = vec![0.0f32; tile_pixel_count];

        for offset in 0..record.count() {
            let Some(&packed_idx) = packed_indices.get(record.start() + offset) else {
                continue;
            };
            let Some(g) = projected_cpu.get(packed_idx as usize) else {
                continue;
            };
            let source_idx = g.source_idx as usize;
            if source_idx >= row_count
                || !g.sigma_x.is_finite()
                || !g.sigma_y.is_finite()
                || g.sigma_x <= 0.0
                || g.sigma_y <= 0.0
            {
                continue;
            }

            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let local_idx = (py - min_y) * tile_width + (px - min_x);
                    if (1.0 - running_alpha[local_idx]) <= 1e-4 {
                        continue;
                    }

                    let pixel_idx = py * render_width + px;
                    let color_idx = pixel_idx * 3;
                    let px_center = px as f32 + 0.5;
                    let py_center = py as f32 + 0.5;
                    let dx = (px_center - g.u) / g.sigma_x;
                    let dy = (py_center - g.v) / g.sigma_y;
                    let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
                    let alpha_raw = kernel * g.opacity;
                    let alpha = alpha_raw.clamp(0.0, 0.99);
                    let contrib = alpha * (1.0 - running_alpha[local_idx]);
                    if contrib <= 1e-8 {
                        continue;
                    }

                    let final_depth = rendered_depth_cpu.get(pixel_idx).copied().unwrap_or(0.0);
                    let final_alpha = rendered_alpha_cpu.get(pixel_idx).copied().unwrap_or(0.0);
                    let depth_denom = final_alpha + 1e-6;
                    let final_color_r = rendered_color_cpu.get(color_idx).copied().unwrap_or(0.0);
                    let final_color_g = rendered_color_cpu
                        .get(color_idx + 1)
                        .copied()
                        .unwrap_or(0.0);
                    let final_color_b = rendered_color_cpu
                        .get(color_idx + 2)
                        .copied()
                        .unwrap_or(0.0);
                    let r_r = final_color_r - running_s[local_idx * 3] - contrib * g.color[0];
                    let r_g = final_color_g - running_s[local_idx * 3 + 1] - contrib * g.color[1];
                    let r_b = final_color_b - running_s[local_idx * 3 + 2] - contrib * g.color[2];

                    let transmittance = 1.0 - running_alpha[local_idx];
                    let inv_one_minus_alpha = 1.0 / (1.0 - alpha).max(1e-6);
                    let dc_r = color_grads.get(color_idx).copied().unwrap_or(0.0);
                    let dc_g = color_grads.get(color_idx + 1).copied().unwrap_or(0.0);
                    let dc_b = color_grads.get(color_idx + 2).copied().unwrap_or(0.0);
                    let dl_dalpha_color = (transmittance * g.color[0] - r_r * inv_one_minus_alpha)
                        * dc_r
                        + (transmittance * g.color[1] - r_g * inv_one_minus_alpha) * dc_g
                        + (transmittance * g.color[2] - r_b * inv_one_minus_alpha) * dc_b;
                    let dd_depth = depth_grads.get(pixel_idx).copied().unwrap_or(0.0);
                    let tail_alpha = final_alpha - running_alpha[local_idx] - contrib;
                    let tail_depth_num = final_depth * depth_denom
                        - running_depth_num[local_idx]
                        - contrib * g.depth;
                    let mut dl_dalpha_depth = 0.0f32;
                    let mut dl_dalpha_alpha = 0.0f32;
                    if dd_depth != 0.0 {
                        let dnum_dalpha =
                            transmittance * g.depth - tail_depth_num * inv_one_minus_alpha;
                        let dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
                        let ddepth_dalpha = (dnum_dalpha * depth_denom
                            - final_depth * depth_denom * dalpha_dalpha)
                            / (depth_denom * depth_denom);
                        dl_dalpha_depth = dd_depth * ddepth_dalpha;
                        if loss_scales.alpha > 0.0 {
                            dl_dalpha_alpha = loss_scales.alpha * dalpha_dalpha;
                        }
                    } else if loss_scales.alpha > 0.0 {
                        let dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
                        dl_dalpha_alpha = loss_scales.alpha * dalpha_dalpha;
                    }

                    running_s[local_idx * 3] += contrib * g.color[0];
                    running_s[local_idx * 3 + 1] += contrib * g.color[1];
                    running_s[local_idx * 3 + 2] += contrib * g.color[2];
                    running_alpha[local_idx] += contrib;
                    running_depth_num[local_idx] += contrib * g.depth;

                    if alpha_raw <= 0.0 || alpha_raw >= 0.99 {
                        continue;
                    }

                    let dl_dalpha_total = dl_dalpha_color + dl_dalpha_depth + dl_dalpha_alpha;
                    let dl_dkernel = dl_dalpha_total * g.opacity;
                    let dk_ddx = kernel * (-dx);
                    let dk_ddy = kernel * (-dy);
                    if g.sigma_x.abs() >= 0.5 {
                        dl_dsigma_x[source_idx] += dl_dkernel * dk_ddx * (-dx / g.sigma_x);
                    }
                    if g.sigma_y.abs() >= 0.5 {
                        dl_dsigma_y[source_idx] += dl_dkernel * dk_ddy * (-dy / g.sigma_y);
                    }
                }
            }
        }
    }

    let mut rotation_grads = vec![0.0f32; row_count * 4];
    for g in &projected_cpu {
        let source_idx = g.source_idx as usize;
        if source_idx >= row_count
            || (dl_dsigma_x[source_idx].abs() + dl_dsigma_y[source_idx].abs()) <= 1e-12
        {
            continue;
        }

        let raw_rotation = row_to_quaternion(
            raw_rotation_rows
                .get(source_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        let x = (g.u - camera.cx) * g.depth / camera.fx.max(1e-6);
        let y = (g.v - camera.cy) * g.depth / camera.fy.max(1e-6);

        for component in 0..4 {
            let (d_sigma_x, d_sigma_y) = finite_difference_sigma_wrt_rotation_component(
                x,
                y,
                g.depth,
                g.scale3d,
                raw_rotation,
                component,
                camera,
            );
            rotation_grads[source_idx * 4 + component] +=
                dl_dsigma_x[source_idx] * d_sigma_x + dl_dsigma_y[source_idx] * d_sigma_y;
        }
    }

    Tensor::from_slice(&rotation_grads, (row_count, 4), device)
}

pub(crate) fn parameter_grads_from_render_color_grads(
    device: &Device,
    active_sh_degree: usize,
    gaussians: &TrainableGaussians,
    projected: &ProjectedGaussians,
    render_color_grads: &Tensor,
    camera: &DiffCamera,
) -> candle_core::Result<(Tensor, Tensor)> {
    if !gaussians.uses_spherical_harmonics() {
        return Ok((
            gaussians.render_color_grads_to_parameter_grads(render_color_grads)?,
            Tensor::zeros_like(gaussians.sh_rest())?,
        ));
    }

    let row_count = gaussians.len();
    let sh_rest_coeff_count = gaussians.sh_rest().dims().get(1).copied().unwrap_or(0);
    let mut sh_0_grads = vec![0.0f32; row_count * 3];
    let mut sh_rest_grads = vec![0.0f32; row_count * sh_rest_coeff_count * 3];
    let source_indices = projected.source_indices.to_vec1::<u32>()?;
    if source_indices.is_empty() {
        return Ok((
            Tensor::from_slice(&sh_0_grads, (row_count, 3), device)?,
            Tensor::from_slice(
                &sh_rest_grads,
                (row_count, sh_rest_coeff_count, 3usize),
                device,
            )?,
        ));
    }

    let visible_grads = render_color_grads
        .index_select(&projected.source_indices, 0)?
        .to_vec2::<f32>()?;
    let visible_positions = gaussians
        .positions()
        .index_select(&projected.source_indices, 0)?
        .to_vec2::<f32>()?;
    let visible_sh_0 = gaussians
        .sh_0()
        .index_select(&projected.source_indices, 0)?
        .to_vec2::<f32>()?;
    let visible_sh_rest = if sh_rest_coeff_count > 0 {
        gaussians
            .sh_rest()
            .index_select(&projected.source_indices, 0)?
            .to_vec3::<f32>()?
    } else {
        Vec::new()
    };
    let active_degree = active_sh_degree.min(gaussians.sh_degree());
    let camera_center = camera_center_world(camera);

    for (visible_idx, &source_idx) in source_indices.iter().enumerate() {
        let source_idx = source_idx as usize;
        if source_idx >= row_count {
            continue;
        }
        let position = row_to_vec3(
            visible_positions
                .get(visible_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        let direction = normalized_view_direction(position, camera_center);
        let basis = sh_basis_values(direction, active_degree);
        let sh_0 = row_to_vec3(
            visible_sh_0
                .get(visible_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        let unclamped_rgb = sh_rgb_from_basis(
            sh_0,
            visible_sh_rest
                .get(visible_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            &basis,
        );
        let render_grad = visible_grads
            .get(visible_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        for channel in 0..3 {
            if unclamped_rgb[channel] <= 0.0 {
                continue;
            }
            let grad_value = render_grad.get(channel).copied().unwrap_or(0.0);
            sh_0_grads[source_idx * 3 + channel] += basis[0] * grad_value;
            for coeff_idx in 0..sh_rest_coeff_count.min(basis.len().saturating_sub(1)) {
                let flat_idx = (source_idx * sh_rest_coeff_count + coeff_idx) * 3 + channel;
                sh_rest_grads[flat_idx] += basis[coeff_idx + 1] * grad_value;
            }
        }
    }

    Ok((
        Tensor::from_slice(&sh_0_grads, (row_count, 3), device)?,
        Tensor::from_slice(
            &sh_rest_grads,
            (row_count, sh_rest_coeff_count, 3usize),
            device,
        )?,
    ))
}

fn camera_center_world(camera: &DiffCamera) -> [f32; 3] {
    [
        -(camera.rotation[0][0] * camera.translation[0]
            + camera.rotation[1][0] * camera.translation[1]
            + camera.rotation[2][0] * camera.translation[2]),
        -(camera.rotation[0][1] * camera.translation[0]
            + camera.rotation[1][1] * camera.translation[1]
            + camera.rotation[2][1] * camera.translation[2]),
        -(camera.rotation[0][2] * camera.translation[0]
            + camera.rotation[1][2] * camera.translation[1]
            + camera.rotation[2][2] * camera.translation[2]),
    ]
}

fn normalized_view_direction(position: [f32; 3], camera_center: [f32; 3]) -> [f32; 3] {
    let dx = position[0] - camera_center[0];
    let dy = position[1] - camera_center[1];
    let dz = position[2] - camera_center[2];
    let norm = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
    [dx / norm, dy / norm, dz / norm]
}

fn sh_basis_values(direction: [f32; 3], degree: usize) -> Vec<f32> {
    let x = direction[0];
    let y = direction[1];
    let z = direction[2];
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    let mut basis = vec![SH_C0];
    if degree > 0 {
        basis.push(-SH_C1 * y);
        basis.push(SH_C1 * z);
        basis.push(-SH_C1 * x);
    }
    if degree > 1 {
        basis.push(SH_C2[0] * xy);
        basis.push(SH_C2[1] * yz);
        basis.push(SH_C2[2] * (2.0 * zz - xx - yy));
        basis.push(SH_C2[3] * xz);
        basis.push(SH_C2[4] * (xx - yy));
    }
    if degree > 2 {
        basis.push(SH_C3[0] * y * (3.0 * xx - yy));
        basis.push(SH_C3[1] * xy * z);
        basis.push(SH_C3[2] * y * (4.0 * zz - xx - yy));
        basis.push(SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
        basis.push(SH_C3[4] * x * (4.0 * zz - xx - yy));
        basis.push(SH_C3[5] * z * (xx - yy));
        basis.push(SH_C3[6] * x * (xx - 3.0 * yy));
    }
    if degree > 3 {
        basis.push(SH_C4[0] * xy * (xx - yy));
        basis.push(SH_C4[1] * yz * (3.0 * xx - yy));
        basis.push(SH_C4[2] * xy * (7.0 * zz - 1.0));
        basis.push(SH_C4[3] * yz * (7.0 * zz - 3.0));
        basis.push(SH_C4[4] * (zz * (35.0 * zz - 30.0) + 3.0));
        basis.push(SH_C4[5] * xz * (7.0 * zz - 3.0));
        basis.push(SH_C4[6] * (xx - yy) * (7.0 * zz - 1.0));
        basis.push(SH_C4[7] * xz * (xx - 3.0 * yy));
        basis.push(SH_C4[8] * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)));
    }
    basis
}

fn sh_rgb_from_basis(sh_0: [f32; 3], sh_rest: &[Vec<f32>], basis: &[f32]) -> [f32; 3] {
    let mut rgb = [
        0.5 + basis[0] * sh_0[0],
        0.5 + basis[0] * sh_0[1],
        0.5 + basis[0] * sh_0[2],
    ];
    for (coeff_idx, basis_value) in basis.iter().enumerate().skip(1) {
        let coeff_row = sh_rest.get(coeff_idx - 1).map(Vec::as_slice).unwrap_or(&[]);
        rgb[0] += basis_value * coeff_row.first().copied().unwrap_or(0.0);
        rgb[1] += basis_value * coeff_row.get(1).copied().unwrap_or(0.0);
        rgb[2] += basis_value * coeff_row.get(2).copied().unwrap_or(0.0);
    }
    rgb
}

fn pixel_color_grads(
    rendered_color: &[f32],
    target_color: &[f32],
    ssim_grads: &[f32],
    loss_scales: MetalBackwardLossScales,
) -> Vec<f32> {
    rendered_color
        .iter()
        .enumerate()
        .map(|(idx, &rendered)| {
            let target = target_color.get(idx).copied().unwrap_or(0.0);
            let l1 = if rendered > target {
                loss_scales.color
            } else if rendered < target {
                -loss_scales.color
            } else {
                0.0
            };
            l1 + ssim_grads.get(idx).copied().unwrap_or(0.0) * loss_scales.ssim
        })
        .collect()
}

fn pixel_depth_grads(
    rendered_depth: &[f32],
    target_depth: &[f32],
    loss_scales: MetalBackwardLossScales,
) -> Vec<f32> {
    rendered_depth
        .iter()
        .enumerate()
        .map(|(idx, &rendered)| {
            if loss_scales.depth <= 0.0
                || !is_valid_depth_sample(target_depth.get(idx).copied().unwrap_or(0.0))
            {
                return 0.0;
            }
            let target = target_depth[idx];
            if rendered > target {
                loss_scales.depth
            } else if rendered < target {
                -loss_scales.depth
            } else {
                0.0
            }
        })
        .collect()
}

fn is_valid_depth_sample(depth: f32) -> bool {
    depth.is_finite() && depth > 0.0
}
