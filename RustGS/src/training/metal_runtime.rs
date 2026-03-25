use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, RwLockReadGuard};

use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_foundation::NSRange;
use objc2_metal::MTLSize;

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};

pub(crate) const METAL_TILE_SIZE: usize = 16;

const METAL_FILL_U32_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fill_u32(
    device uint* dst [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    dst[gid] = value;
}
"#;

const METAL_TILE_FORWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CameraUniform {
    float fx;
    float fy;
    float cx;
    float cy;
    uint width;
    uint height;
    uint tile_size;
    uint num_tiles_x;
    float rot00;
    float rot01;
    float rot02;
    float rot10;
    float rot11;
    float rot12;
    float rot20;
    float rot21;
    float rot22;
    float tx;
    float ty;
    float tz;
};

struct TileRecord {
    uint start;
    uint count;
    uint _pad0;
    uint _pad1;
};

struct ProjectedGaussian {
    // Hot fields: read every pixel×Gaussian (forward inner loop) — 9 floats = 36 bytes
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    // Warm fields: used by backward kernel — 9 floats
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    // Cold fields: index/count/tile-binning — 2 uint + 4 floats
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

kernel void tile_forward(
    constant CameraUniform& camera [[buffer(0)]],
    device const TileRecord* tile_records [[buffer(1)]],
    device const uint* tile_indices [[buffer(2)]],
    device const ProjectedGaussian* gaussians [[buffer(3)]],
    device float* out_color [[buffer(4)]],
    device float* out_depth [[buffer(5)]],
    device float* out_alpha [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= camera.width || gid.y >= camera.height) {
        return;
    }

    const uint pixel_idx = gid.y * camera.width + gid.x;
    const uint tile_x = gid.x / camera.tile_size;
    const uint tile_y = gid.y / camera.tile_size;
    const uint tile_idx = tile_y * camera.num_tiles_x + tile_x;
    const TileRecord record = tile_records[tile_idx];

    const float px = float(gid.x) + 0.5f;
    const float py = float(gid.y) + 0.5f;
    float trans = 1.0f;
    float3 color = float3(0.0f);
    float depth = 0.0f;
    float alpha_acc = 0.0f;

    for (uint i = 0; i < record.count; ++i) {
        const ProjectedGaussian gaussian = gaussians[tile_indices[record.start + i]];
        const float dx = (px - gaussian.u) / gaussian.sigma_x;
        const float dy = (py - gaussian.v) / gaussian.sigma_y;
        const float exponent = -0.5f * (dx * dx + dy * dy);
        const float alpha = clamp(exp(exponent) * gaussian.opacity, 0.0f, 0.99f);
        const float contrib = alpha * trans;
        color += contrib * float3(gaussian.color_r, gaussian.color_g, gaussian.color_b);
        depth += contrib * gaussian.depth;
        alpha_acc += contrib;
        trans *= (1.0f - alpha);
        if (trans <= 1e-4f) {
            break;
        }
    }

    out_color[pixel_idx * 3 + 0] = clamp(color.x, 0.0f, 1.0f);
    out_color[pixel_idx * 3 + 1] = clamp(color.y, 0.0f, 1.0f);
    out_color[pixel_idx * 3 + 2] = clamp(color.z, 0.0f, 1.0f);
    out_alpha[pixel_idx] = alpha_acc;
    out_depth[pixel_idx] = depth / (alpha_acc + 1e-6f);
}
"#;

const METAL_TILE_BACKWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BwdCameraUniform {
    float fx;
    float fy;
    float cx;
    float cy;
    uint width;
    uint height;
    uint tile_size;
    uint num_tiles_x;
    float rot00;
    float rot01;
    float rot02;
    float rot10;
    float rot11;
    float rot12;
    float rot20;
    float rot21;
    float rot22;
    float tx;
    float ty;
    float tz;
};

struct BwdTileRecord {
    uint start;
    uint count;
    uint _pad0;
    uint _pad1;
};

struct BwdProjectedGaussian {
    // Hot fields: read every pixel×Gaussian — 9 floats = 36 bytes
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    // Warm fields: used by backward kernel — 9 floats
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    // Cold fields: index/count/tile-binning
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

struct BwdLossScalars {
    float color_scale;
    float depth_scale;
};

kernel void tile_backward(
    constant BwdCameraUniform& camera      [[buffer(0)]],
    device const BwdTileRecord* tile_records [[buffer(1)]],
    device const uint* tile_indices         [[buffer(2)]],
    device const BwdProjectedGaussian* gaussians [[buffer(3)]],
    device const float* rendered_color      [[buffer(4)]],
    device const float* rendered_depth      [[buffer(5)]],
    device const float* rendered_alpha      [[buffer(6)]],
    device const float* target_color        [[buffer(7)]],
    device const float* target_depth        [[buffer(8)]],
    device atomic_float* grad_positions     [[buffer(9)]],
    device atomic_float* grad_log_scales    [[buffer(10)]],
    device atomic_float* grad_opacity_logits [[buffer(11)]],
    device atomic_float* grad_colors        [[buffer(12)]],
    constant BwdLossScalars& loss_scalars   [[buffer(13)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    if (gid.x >= camera.width || gid.y >= camera.height) return;

    const uint W = camera.width;
    const uint pixel_idx = gid.y * W + gid.x;
    const uint c3 = pixel_idx * 3;

    const float px = float(gid.x) + 0.5f;
    const float py = float(gid.y) + 0.5f;

    const float final_color_r = rendered_color[c3 + 0];
    const float final_color_g = rendered_color[c3 + 1];
    const float final_color_b = rendered_color[c3 + 2];
    const float final_depth = rendered_depth[pixel_idx];
    const float final_alpha = rendered_alpha[pixel_idx];
    const float depth_denom = final_alpha + 1e-6f;

    // dL/dC_p from weighted mean-L1
    const float dc_r = (final_color_r > target_color[c3 + 0]) ? loss_scalars.color_scale
                      : (final_color_r < target_color[c3 + 0]) ? -loss_scalars.color_scale : 0.0f;
    const float dc_g = (final_color_g > target_color[c3 + 1]) ? loss_scalars.color_scale
                      : (final_color_g < target_color[c3 + 1]) ? -loss_scalars.color_scale : 0.0f;
    const float dc_b = (final_color_b > target_color[c3 + 2]) ? loss_scalars.color_scale
                      : (final_color_b < target_color[c3 + 2]) ? -loss_scalars.color_scale : 0.0f;

    float dd_depth = 0.0f;
    if (loss_scalars.depth_scale > 0.0f) {
        const float depth_diff = final_depth - target_depth[pixel_idx];
        dd_depth = (depth_diff > 0.0f) ? loss_scalars.depth_scale
                 : (depth_diff < 0.0f) ? -loss_scalars.depth_scale : 0.0f;
    }

    // Tile binning
    const uint tile_x = gid.x / camera.tile_size;
    const uint tile_y = gid.y / camera.tile_size;
    const uint tile_idx = tile_y * camera.num_tiles_x + tile_x;
    const BwdTileRecord record = tile_records[tile_idx];

    // Running state for front-to-back compositing
    float running_s_r = 0.0f;
    float running_s_g = 0.0f;
    float running_s_b = 0.0f;
    float running_alpha = 0.0f;
    float running_depth_num = 0.0f;

    for (uint i = 0; i < record.count; ++i) {
        const uint gidx = tile_indices[record.start + i];
        const BwdProjectedGaussian g = gaussians[gidx];

        const float dx = (px - g.u) / g.sigma_x;
        const float dy = (py - g.v) / g.sigma_y;
        const float exponent = -0.5f * (dx * dx + dy * dy);
        const float kernel_val = exp(exponent);
        const float alpha_raw = kernel_val * g.opacity;
        const float alpha = clamp(alpha_raw, 0.0f, 0.99f);
        const float contrib = alpha * (1.0f - running_alpha);

        if (contrib <= 1e-8f) {
            continue;
        }

        // Remaining color after this gaussian: R_i = C - S_{i-1} - contrib_i * c_i
        const float r_r = final_color_r - running_s_r - contrib * g.color_r;
        const float r_g = final_color_g - running_s_g - contrib * g.color_g;
        const float r_b = final_color_b - running_s_b - contrib * g.color_b;

        const float transmittance = 1.0f - running_alpha;
        const float inv_one_minus_alpha = 1.0f / max(1.0f - alpha, 1e-6f);

        // dC/dalpha_i = T_i * c_i - R_i / (1 - alpha_i)
        const float dl_dalpha_color = (transmittance * g.color_r - r_r * inv_one_minus_alpha) * dc_r
                                    + (transmittance * g.color_g - r_g * inv_one_minus_alpha) * dc_g
                                    + (transmittance * g.color_b - r_b * inv_one_minus_alpha) * dc_b;

        // Depth gradient
        const float tail_alpha = final_alpha - running_alpha - contrib;
        const float tail_depth_num = final_depth * depth_denom - running_depth_num - contrib * g.depth;
        float dl_dalpha_depth = 0.0f;
        if (dd_depth != 0.0f) {
            const float dnum_dalpha = transmittance * g.depth - tail_depth_num * inv_one_minus_alpha;
            const float dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
            const float ddepth_dalpha = (dnum_dalpha * depth_denom - final_depth * depth_denom * dalpha_dalpha)
                                      / (depth_denom * depth_denom);
            dl_dalpha_depth = dd_depth * ddepth_dalpha;
        }

        const float dl_dalpha_total = dl_dalpha_color + dl_dalpha_depth;
        const float dl_dz_direct = dd_depth * contrib / depth_denom;

        // Color gradient: dL/dc_i = T_i * alpha_i * dL/dC_p
        atomic_fetch_add_explicit(&grad_colors[gidx * 3 + 0], contrib * dc_r, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_colors[gidx * 3 + 1], contrib * dc_g, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_colors[gidx * 3 + 2], contrib * dc_b, memory_order_relaxed);

        // Update running state
        running_s_r += contrib * g.color_r;
        running_s_g += contrib * g.color_g;
        running_s_b += contrib * g.color_b;
        running_alpha += contrib;
        running_depth_num += contrib * g.depth;

        // Early termination: once accumulated alpha is nearly 1 all subsequent
        // Gaussian contributions will be negligible (T_i → 0).
        if ((1.0f - running_alpha) <= 1e-4f) {
            break;
        }

        // Skip gradients if alpha is clamped
        if (alpha_raw <= 0.0f || alpha_raw >= 0.99f) {
            // Still need to update running state (done above), just skip gradient contribution
            continue;
        }

        // dL/dbase_alpha = dl_dalpha_total * kernel_val
        const float dl_dbase_alpha = dl_dalpha_total * kernel_val;

        // dL/dkernel = dl_dalpha_total * opacity
        const float dl_dkernel = dl_dalpha_total * g.opacity;
        const float dk_ddx = kernel_val * (-dx);
        const float dk_ddy = kernel_val * (-dy);

        // Gradient w.r.t. projected position
        const float dl_du = dl_dkernel * dk_ddx * (-1.0f / g.sigma_x);
        const float dl_dv = dl_dkernel * dk_ddy * (-1.0f / g.sigma_y);

        // Gradient w.r.t. 2D scale
        float dl_dsigma_x = 0.0f;
        float dl_dsigma_y = 0.0f;
        if (abs(g.sigma_x) >= 0.5f) {
            dl_dsigma_x = dl_dkernel * dk_ddx * (-dx / g.sigma_x);
        }
        if (abs(g.sigma_y) >= 0.5f) {
            dl_dsigma_y = dl_dkernel * dk_ddy * (-dy / g.sigma_y);
        }

        // Total depth gradient
        const float dl_dz = dl_dz_direct;

        // 2D -> 3D chain rule
        const float inv_z = 1.0f / max(g.depth, 1e-6f);

        atomic_fetch_add_explicit(&grad_positions[gidx * 3 + 0], dl_du * camera.fx * inv_z, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_positions[gidx * 3 + 1], dl_dv * camera.fy * inv_z, memory_order_relaxed);
        const float dl_dz_projected = dl_du * (-(g.u - camera.cx) * inv_z)
                                    + dl_dv * (-(g.v - camera.cy) * inv_z);
        atomic_fetch_add_explicit(&grad_positions[gidx * 3 + 2], dl_dz + dl_dz_projected, memory_order_relaxed);

        // Scale gradient: propagate through sigma_x and sigma_y back to 3D log-scales.
        // sigma_x² ≈ s_x²*(fx/z)² + s_z²*(fx*|x_cam|/z²)²  (dominant terms, identity rot)
        // d(sigma_x)/d(s_x)*s_x = s_x²*(fx/z)²/sigma_x  → dL/d(log_sx)
        // d(sigma_x)/d(s_z)*s_z = var_x_from_z/sigma_x   → accumulates into dL/d(log_sz)
        if (abs(g.sigma_x) >= 0.5f) {
            atomic_fetch_add_explicit(&grad_log_scales[gidx * 3 + 0], dl_dsigma_x * camera.fx * inv_z, memory_order_relaxed);
            const float raw_sx = g.raw_sigma_x;
            const float sx3 = g.scale_x;
            const float sz3 = g.scale_z;
            if (raw_sx > 1e-6f && sx3 > 1e-6f && sz3 > 1e-6f) {
                const float contrib_x = sx3 * camera.fx * inv_z;
                const float var_from_z = max(raw_sx * raw_sx - contrib_x * contrib_x, 0.0f);
                const float d_rawsx_d_sz = var_from_z / (raw_sx * sz3);
                atomic_fetch_add_explicit(&grad_log_scales[gidx * 3 + 2], dl_dsigma_x * d_rawsx_d_sz * sz3, memory_order_relaxed);
            }
        }
        if (abs(g.sigma_y) >= 0.5f) {
            atomic_fetch_add_explicit(&grad_log_scales[gidx * 3 + 1], dl_dsigma_y * camera.fy * inv_z, memory_order_relaxed);
            const float raw_sy = g.raw_sigma_y;
            const float sy3 = g.scale_y;
            const float sz3 = g.scale_z;
            if (raw_sy > 1e-6f && sy3 > 1e-6f && sz3 > 1e-6f) {
                const float contrib_y = sy3 * camera.fy * inv_z;
                const float var_from_z = max(raw_sy * raw_sy - contrib_y * contrib_y, 0.0f);
                const float d_rawsy_d_sz = var_from_z / (raw_sy * sz3);
                atomic_fetch_add_explicit(&grad_log_scales[gidx * 3 + 2], dl_dsigma_y * d_rawsy_d_sz * sz3, memory_order_relaxed);
            }
        }

        // Opacity gradient: dL/dlogit = dL/dbase_alpha * sigmoid_derivative(logit)
        // sigmoid_derivative = opacity * (1 - opacity)
        const float sig_deriv = g.opacity * (1.0f - g.opacity);
        if (g.opacity >= 0.0f && g.opacity <= 1.0f) {
            atomic_fetch_add_explicit(&grad_opacity_logits[gidx], dl_dbase_alpha * sig_deriv, memory_order_relaxed);
        }
    }
}
"#;

const METAL_GRAD_MAGNITUDE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void grad_magnitudes(
    device const float* grad_positions [[buffer(0)]],
    device const float* grad_scales [[buffer(1)]],
    device const float* grad_opacity [[buffer(2)]],
    device const float* grad_colors [[buffer(3)]],
    device float* out_magnitudes [[buffer(4)]],
    constant uint& gaussian_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussian_count) {
        return;
    }

    const uint p = gid * 3;
    float mag = 0.0f;
    mag += abs(grad_positions[p + 0]);
    mag += abs(grad_positions[p + 1]);
    mag += abs(grad_positions[p + 2]);
    mag += abs(grad_scales[p + 0]);
    mag += abs(grad_scales[p + 1]);
    mag += abs(grad_scales[p + 2]);
    mag += abs(grad_opacity[gid]);
    mag += abs(grad_colors[p + 0]);
    mag += abs(grad_colors[p + 1]);
    mag += abs(grad_colors[p + 2]);
    out_magnitudes[gid] = mag;
}
"#;

const METAL_PROJECT_GAUSSIANS_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ProjectCameraUniform {
    float fx;
    float fy;
    float cx;
    float cy;
    uint width;
    uint height;
    uint tile_size;
    uint num_tiles_x;
    float rot00;
    float rot01;
    float rot02;
    float rot10;
    float rot11;
    float rot12;
    float rot20;
    float rot21;
    float rot22;
    float tx;
    float ty;
    float tz;
};

struct ProjectionRecord {
    // Hot fields (forward inner loop)
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    // Warm fields (backward)
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    // Cold fields (index / tile binning)
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

static inline float4 quat_normalize(float4 q) {
    float n = sqrt(max(dot(q, q), 1e-12f));
    return q / n;
}

kernel void project_gaussians(
    constant ProjectCameraUniform& camera [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device const float* log_scales [[buffer(2)]],
    device const float* rotations [[buffer(3)]],
    device const float* opacity_logits [[buffer(4)]],
    device const float* colors [[buffer(5)]],
    device ProjectionRecord* out_records [[buffer(6)]],
    constant uint& gaussian_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussian_count) return;

    const uint p = gid * 3;
    const uint r = gid * 4;

    const float px = positions[p + 0];
    const float py = positions[p + 1];
    const float pz = positions[p + 2];

    const float x = camera.rot00 * px + camera.rot01 * py + camera.rot02 * pz + camera.tx;
    const float y = camera.rot10 * px + camera.rot11 * py + camera.rot12 * pz + camera.ty;
    const float z = camera.rot20 * px + camera.rot21 * py + camera.rot22 * pz + camera.tz;

    ProjectionRecord rec;
    rec.source_idx = gid;
    rec.visible = 0;
    rec.u = 0.0f; rec.v = 0.0f;
    rec.sigma_x = 0.0f; rec.sigma_y = 0.0f;
    rec.raw_sigma_x = 0.0f; rec.raw_sigma_y = 0.0f;
    rec.depth = z;
    rec.opacity_logit = opacity_logits[gid];
    rec.opacity = 1.0f / (1.0f + exp(-rec.opacity_logit));
    rec.scale_x = exp(log_scales[p + 0]);
    rec.scale_y = exp(log_scales[p + 1]);
    rec.scale_z = exp(log_scales[p + 2]);
    rec.color_r = colors[p + 0];
    rec.color_g = colors[p + 1];
    rec.color_b = colors[p + 2];
    rec.min_x = 0.0f; rec.max_x = 0.0f; rec.min_y = 0.0f; rec.max_y = 0.0f;

    if (!isfinite(z) || z < 1e-4f) {
        out_records[gid] = rec;
        return;
    }

    const float inv_z = 1.0f / max(z, 1e-4f);
    rec.u = camera.fx * x * inv_z + camera.cx;
    rec.v = camera.fy * y * inv_z + camera.cy;

    float4 q = quat_normalize(float4(rotations[r + 1], rotations[r + 2], rotations[r + 3], rotations[r + 0]));
    const float qx = q.x;
    const float qy = q.y;
    const float qz = q.z;
    const float qw = q.w;

    const float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    const float r01 = 2.0f * (qx * qy - qz * qw);
    const float r02 = 2.0f * (qx * qz + qy * qw);
    const float r10 = 2.0f * (qx * qy + qz * qw);
    const float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    const float r12 = 2.0f * (qy * qz - qx * qw);
    const float r20 = 2.0f * (qx * qz - qy * qw);
    const float r21 = 2.0f * (qy * qz + qx * qw);
    const float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    const float sxx = rec.scale_x * rec.scale_x;
    const float syy = rec.scale_y * rec.scale_y;
    const float szz = rec.scale_z * rec.scale_z;

    // covariance_world = R * diag(s^2) * R^T
    const float cw00 = r00*r00*sxx + r01*r01*syy + r02*r02*szz;
    const float cw01 = r00*r10*sxx + r01*r11*syy + r02*r12*szz;
    const float cw02 = r00*r20*sxx + r01*r21*syy + r02*r22*szz;
    const float cw11 = r10*r10*sxx + r11*r11*syy + r12*r12*szz;
    const float cw12 = r10*r20*sxx + r11*r21*syy + r12*r22*szz;
    const float cw22 = r20*r20*sxx + r21*r21*syy + r22*r22*szz;

    // covariance_camera = C * covariance_world * C^T
    const float c00 = camera.rot00 * cw00 + camera.rot01 * cw01 + camera.rot02 * cw02;
    const float c01 = camera.rot00 * cw01 + camera.rot01 * cw11 + camera.rot02 * cw12;
    const float c02 = camera.rot00 * cw02 + camera.rot01 * cw12 + camera.rot02 * cw22;
    const float c10 = camera.rot10 * cw00 + camera.rot11 * cw01 + camera.rot12 * cw02;
    const float c11 = camera.rot10 * cw01 + camera.rot11 * cw11 + camera.rot12 * cw12;
    const float c12 = camera.rot10 * cw02 + camera.rot11 * cw12 + camera.rot12 * cw22;
    const float c20 = camera.rot20 * cw00 + camera.rot21 * cw01 + camera.rot22 * cw02;
    const float c21 = camera.rot20 * cw01 + camera.rot21 * cw11 + camera.rot22 * cw12;
    const float c22 = camera.rot20 * cw02 + camera.rot21 * cw12 + camera.rot22 * cw22;

    const float cc00 = c00 * camera.rot00 + c01 * camera.rot01 + c02 * camera.rot02;
    const float cc02 = c00 * camera.rot20 + c01 * camera.rot21 + c02 * camera.rot22;
    const float cc11 = c10 * camera.rot10 + c11 * camera.rot11 + c12 * camera.rot12;
    const float cc12 = c10 * camera.rot20 + c11 * camera.rot21 + c12 * camera.rot22;
    const float cc22 = c20 * camera.rot20 + c21 * camera.rot21 + c22 * camera.rot22;

    const float3 proj_x = float3(camera.fx * inv_z, 0.0f, -camera.fx * x * inv_z * inv_z);
    const float3 proj_y = float3(0.0f, camera.fy * inv_z, -camera.fy * y * inv_z * inv_z);

    const float cov_x = proj_x.x * (cc00 * proj_x.x + cc02 * proj_x.z)
                      + proj_x.z * (cc02 * proj_x.x + cc22 * proj_x.z);
    const float cov_y = proj_y.y * (cc11 * proj_y.y + cc12 * proj_y.z)
                      + proj_y.z * (cc12 * proj_y.y + cc22 * proj_y.z);
    rec.raw_sigma_x = sqrt(max(cov_x, 1e-6f));
    rec.raw_sigma_y = sqrt(max(cov_y, 1e-6f));

    if (!isfinite(rec.raw_sigma_x) || !isfinite(rec.raw_sigma_y)) {
        out_records[gid] = rec;
        return;
    }

    rec.sigma_x = clamp(rec.raw_sigma_x, 0.5f, 256.0f);
    rec.sigma_y = clamp(rec.raw_sigma_y, 0.5f, 256.0f);
    const float support_x = rec.sigma_x * 3.0f;
    const float support_y = rec.sigma_y * 3.0f;

    if (rec.u + support_x < 0.0f || rec.u - support_x > float(camera.width)
        || rec.v + support_y < 0.0f || rec.v - support_y > float(camera.height)) {
        out_records[gid] = rec;
        return;
    }

    rec.min_x = clamp(rec.u - support_x, 0.0f, float(camera.width - 1));
    rec.max_x = clamp(rec.u + support_x, 0.0f, float(camera.width - 1));
    rec.min_y = clamp(rec.v - support_y, 0.0f, float(camera.height - 1));
    rec.max_y = clamp(rec.v + support_y, 0.0f, float(camera.height - 1));
    rec.visible = 1;
    out_records[gid] = rec;
}
"#;

const METAL_TILE_COUNT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CountCameraUniform {
    float fx; float fy; float cx; float cy;
    uint width; uint height; uint tile_size; uint num_tiles_x;
    float rot00; float rot01; float rot02;
    float rot10; float rot11; float rot12;
    float rot20; float rot21; float rot22;
    float tx; float ty; float tz;
};

struct CountProjectedGaussian {
    float u; float v; float sigma_x; float sigma_y;
    float depth; float opacity; float color_r; float color_g; float color_b;
    float raw_sigma_x; float raw_sigma_y; float opacity_logit;
    float scale_x; float scale_y; float scale_z;
    uint source_idx; uint visible;
    float min_x; float max_x; float min_y; float max_y;
};

kernel void tile_count(
    constant CountCameraUniform& camera [[buffer(0)]],
    device const CountProjectedGaussian* gaussians [[buffer(1)]],
    device atomic_uint* tile_counts [[buffer(2)]],
    constant uint& gaussian_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussian_count) return;
    const CountProjectedGaussian g = gaussians[gid];
    const uint max_tx = (camera.width - 1u) / camera.tile_size;
    const uint max_ty = (camera.height - 1u) / camera.tile_size;
    const uint tile_x_min = min((uint)floor(max(g.min_x, 0.0f)) / camera.tile_size, max_tx);
    const uint tile_x_max = min((uint)ceil(max(g.max_x, 0.0f)) / camera.tile_size, max_tx);
    const uint tile_y_min = min((uint)floor(max(g.min_y, 0.0f)) / camera.tile_size, max_ty);
    const uint tile_y_max = min((uint)ceil(max(g.max_y, 0.0f)) / camera.tile_size, max_ty);
    for (uint ty = tile_y_min; ty <= tile_y_max; ++ty) {
        for (uint tx = tile_x_min; tx <= tile_x_max; ++tx) {
            atomic_fetch_add_explicit(&tile_counts[ty * camera.num_tiles_x + tx], 1u, memory_order_relaxed);
        }
    }
}
"#;

const METAL_TILE_ASSIGN_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AssignCameraUniform {
    float fx; float fy; float cx; float cy;
    uint width; uint height; uint tile_size; uint num_tiles_x;
    float rot00; float rot01; float rot02;
    float rot10; float rot11; float rot12;
    float rot20; float rot21; float rot22;
    float tx; float ty; float tz;
};

struct AssignProjectedGaussian {
    float u; float v; float sigma_x; float sigma_y;
    float depth; float opacity; float color_r; float color_g; float color_b;
    float raw_sigma_x; float raw_sigma_y; float opacity_logit;
    float scale_x; float scale_y; float scale_z;
    uint source_idx; uint visible;
    float min_x; float max_x; float min_y; float max_y;
};

kernel void tile_assign(
    constant AssignCameraUniform& camera [[buffer(0)]],
    device const AssignProjectedGaussian* gaussians [[buffer(1)]],
    device atomic_uint* tile_offsets [[buffer(2)]],
    device uint* tile_indices [[buffer(3)]],
    constant uint& gaussian_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussian_count) return;
    const AssignProjectedGaussian g = gaussians[gid];
    const uint max_tx = (camera.width - 1u) / camera.tile_size;
    const uint max_ty = (camera.height - 1u) / camera.tile_size;
    const uint tile_x_min = min((uint)floor(max(g.min_x, 0.0f)) / camera.tile_size, max_tx);
    const uint tile_x_max = min((uint)ceil(max(g.max_x, 0.0f)) / camera.tile_size, max_tx);
    const uint tile_y_min = min((uint)floor(max(g.min_y, 0.0f)) / camera.tile_size, max_ty);
    const uint tile_y_max = min((uint)ceil(max(g.max_y, 0.0f)) / camera.tile_size, max_ty);
    for (uint ty = tile_y_min; ty <= tile_y_max; ++ty) {
        for (uint tx = tile_x_min; tx <= tile_x_max; ++tx) {
            const uint tile_idx = ty * camera.num_tiles_x + tx;
            const uint write_idx = atomic_fetch_add_explicit(&tile_offsets[tile_idx], 1u, memory_order_relaxed);
            tile_indices[write_idx] = gid;
        }
    }
}
"#;

const METAL_FILL_PROJECTION_PADDING_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PaddingProjectionRecord {
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

kernel void fill_projection_padding(
    device PaddingProjectionRecord* records [[buffer(0)]],
    constant uint& start_idx [[buffer(1)]],
    constant uint& total_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint idx = start_idx + gid;
    if (idx >= total_count) return;
    PaddingProjectionRecord rec;
    rec.source_idx = 0;
    rec.visible = 0;
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.sigma_x = 0.0f;
    rec.sigma_y = 0.0f;
    rec.raw_sigma_x = 0.0f;
    rec.raw_sigma_y = 0.0f;
    rec.depth = INFINITY;
    rec.opacity = 0.0f;
    rec.opacity_logit = 0.0f;
    rec.scale_x = 0.0f;
    rec.scale_y = 0.0f;
    rec.scale_z = 0.0f;
    rec.color_r = 0.0f;
    rec.color_g = 0.0f;
    rec.color_b = 0.0f;
    rec.min_x = 0.0f;
    rec.max_x = 0.0f;
    rec.min_y = 0.0f;
    rec.max_y = 0.0f;
    records[idx] = rec;
}
"#;

const METAL_COUNT_VISIBLE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CountProjectionRecord {
    uint source_idx;
    uint visible;
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float raw_sigma_x;
    float raw_sigma_y;
    float depth;
    float opacity;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    float color_r;
    float color_g;
    float color_b;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

kernel void count_visible(
    device const CountProjectionRecord* records [[buffer(0)]],
    device atomic_uint* visible_count [[buffer(1)]],
    constant uint& record_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= record_count) return;
    if (records[gid].visible != 0u) {
        atomic_fetch_add_explicit(&visible_count[0], 1u, memory_order_relaxed);
    }
}
"#;

const METAL_BITONIC_SORT_PROJECTIONS_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SortProjectionRecord {
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

static inline bool projection_less(SortProjectionRecord a, SortProjectionRecord b) {
    if (a.visible != b.visible) return a.visible > b.visible;
    if (a.depth != b.depth) return a.depth < b.depth;
    return a.source_idx < b.source_idx;
}

kernel void bitonic_sort_projections(
    device SortProjectionRecord* records [[buffer(0)]],
    constant uint& stage_j [[buffer(1)]],
    constant uint& stage_k [[buffer(2)]],
    constant uint& total_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_count) return;
    const uint ixj = gid ^ stage_j;
    if (ixj <= gid || ixj >= total_count) return;

    const bool ascending = ((gid & stage_k) == 0u);
    const SortProjectionRecord a = records[gid];
    const SortProjectionRecord b = records[ixj];
    const bool a_less_b = projection_less(a, b);

    if ((ascending && !a_less_b) || (!ascending && a_less_b)) {
        records[gid] = b;
        records[ixj] = a;
    }
}
"#;

const METAL_EXTRACT_SOURCE_INDICES_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct IndexProjectionRecord {
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    float raw_sigma_x;
    float raw_sigma_y;
    float opacity_logit;
    float scale_x;
    float scale_y;
    float scale_z;
    uint source_idx;
    uint visible;
    float min_x;
    float max_x;
    float min_y;
    float max_y;
};

kernel void extract_source_indices(
    device const IndexProjectionRecord* records [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out_indices[gid] = records[gid].source_idx;
}
"#;

// GPU prefix-sum (exclusive scan) for tile binning offsets.
// Uses a simple single-pass approach for small tile counts (<=64K tiles).
const METAL_TILE_PREFIX_SUM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Exclusive prefix sum of `tile_counts` into `tile_offsets`.
// Also writes the total sum into `total_out[0]`.
// For tile counts up to 64K this single-pass approach is fast enough.
// Limitation: all threads must fit in one threadgroup, so max_tiles = 1024.
kernel void tile_prefix_sum(
    device const uint* tile_counts [[buffer(0)]],
    device uint*       tile_offsets [[buffer(1)]],
    device uint*       total_out    [[buffer(2)]],
    constant uint&     tile_count   [[buffer(3)]],
    uint               lid          [[thread_position_in_threadgroup]],
    uint               tpg          [[threads_per_threadgroup]]
) {
    // Shared memory exclusive scan (Blelloch / Hillis-Steele hybrid)
    // We use a simple sequential scan in a single thread for correctness
    // across arbitrary tile_count. The caller guarantees tile_count <= tpg.
    if (lid == 0) {
        uint running = 0;
        for (uint i = 0; i < tile_count; ++i) {
            tile_offsets[i] = running;
            running += tile_counts[i];
        }
        total_out[0] = running;
    }
}
"#;

// Fused Adam update kernel: m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g;
// m_hat = m / (1 - b1^t); v_hat = v / (1 - b2^t); param -= lr * m_hat / (sqrt(v_hat) + eps)
// Operates in-place on float buffers. `stride` is the number of floats per element
// (3 for positions/scales/colors, 1 for opacities).
const METAL_ADAM_STEP_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AdamHyperparams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float bc1;   // 1 / (1 - beta1^step)
    float bc2;   // 1 / (1 - beta2^step)
};

kernel void adam_step(
    device float*       params  [[buffer(0)]],
    device const float* grads   [[buffer(1)]],
    device float*       m_buf   [[buffer(2)]],
    device float*       v_buf   [[buffer(3)]],
    constant AdamHyperparams& hp [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const float g = grads[gid];
    float m = hp.beta1 * m_buf[gid] + (1.0f - hp.beta1) * g;
    float v = hp.beta2 * v_buf[gid] + (1.0f - hp.beta2) * g * g;
    m_buf[gid] = m;
    v_buf[gid] = v;
    const float m_hat = m * hp.bc1;
    const float v_hat = v * hp.bc2;
    params[gid] -= hp.lr * m_hat / (sqrt(v_hat) + hp.eps);
}
"#;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenRect {
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
}

pub(crate) struct ChunkPixelWindow {
    pub pixel_x: Tensor,
    pub pixel_y: Tensor,
    pub indices: Tensor,
    pub pixel_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalCameraUniform {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    tile_size: u32,
    num_tiles_x: u32,
    rot00: f32,
    rot01: f32,
    rot02: f32,
    rot10: f32,
    rot11: f32,
    rot12: f32,
    rot20: f32,
    rot21: f32,
    rot22: f32,
    tx: f32,
    ty: f32,
    tz: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalTileDispatchRecord {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

impl MetalTileDispatchRecord {
    pub(crate) fn new(start: u32, count: u32) -> Self {
        Self {
            start,
            count,
            ..Default::default()
        }
    }

    pub(crate) fn start(&self) -> usize {
        self.start as usize
    }

    pub(crate) fn count(&self) -> usize {
        self.count as usize
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MetalTileBins {
    records: Vec<MetalTileDispatchRecord>,
    active_tiles: Vec<usize>,
    packed_indices: Vec<u32>,
    total_assignments: usize,
    max_gaussians_per_tile: usize,
}

impl MetalTileBins {
    pub(crate) fn active_tiles(&self) -> &[usize] {
        &self.active_tiles
    }

    pub(crate) fn active_tile_count(&self) -> usize {
        self.active_tiles.len()
    }

    pub(crate) fn total_assignments(&self) -> usize {
        self.total_assignments
    }

    pub(crate) fn max_gaussians_per_tile(&self) -> usize {
        self.max_gaussians_per_tile
    }

    pub(crate) fn packed_indices(&self) -> &[u32] {
        &self.packed_indices
    }

    pub(crate) fn records(&self) -> &[MetalTileDispatchRecord] {
        &self.records
    }

    pub(crate) fn record(&self, tile_idx: usize) -> Option<MetalTileDispatchRecord> {
        self.records.get(tile_idx).copied()
    }

    #[cfg(test)]
    pub(crate) fn indices_for_tile(&self, tile_idx: usize) -> &[u32] {
        let Some(record) = self.records.get(tile_idx) else {
            return &[];
        };
        let start = record.start();
        let end = start.saturating_add(record.count());
        self.packed_indices.get(start..end).unwrap_or(&[])
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalProjectedGaussian {
    pub u: f32,
    pub v: f32,
    pub sigma_x: f32,
    pub sigma_y: f32,
    pub depth: f32,
    pub opacity: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalProjectionRecord {
    // Hot fields (forward inner loop)
    pub u: f32,
    pub v: f32,
    pub sigma_x: f32,
    pub sigma_y: f32,
    pub depth: f32,
    pub opacity: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    // Warm fields (backward)
    pub raw_sigma_x: f32,
    pub raw_sigma_y: f32,
    pub opacity_logit: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    // Cold fields (index / tile binning)
    pub source_idx: u32,
    pub visible: u32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

impl MetalProjectedGaussian {
    pub(crate) fn new(
        u: f32,
        v: f32,
        sigma_x: f32,
        sigma_y: f32,
        depth: f32,
        opacity: f32,
        color_r: f32,
        color_g: f32,
        color_b: f32,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Self {
        Self {
            u,
            v,
            sigma_x,
            sigma_y,
            depth,
            opacity,
            color_r,
            color_g,
            color_b,
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }
}

pub(crate) struct NativeForwardFrame {
    pub color: Tensor,
    pub depth: Tensor,
    pub alpha: Tensor,
}

pub(crate) struct NativeBackwardFrame {
    pub grad_positions: Tensor,
    pub grad_log_scales: Tensor,
    pub grad_opacity_logits: Tensor,
    pub grad_colors: Tensor,
}

pub(crate) struct ProjectedGpuBatch {
    pub visible_count: usize,
    pub visible_source_indices: Vec<u32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NativeForwardProfile {
    pub setup: std::time::Duration,
    pub staging: std::time::Duration,
    pub kernel: std::time::Duration,
    pub total: std::time::Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MetalBufferSlot {
    CameraUniforms,
    VisibleIndices,
    TileCounts,
    TileOffsets,
    TileMetadata,
    TileIndices,
    ProjectedGaussians,
    ProjectionRecords,
    VisibleSourceIndices,
    GradPositions,
    GradRotations,
    GradScales,
    GradOpacity,
    GradColors,
    OutputColor,
    OutputDepth,
    OutputAlpha,
    TargetColor,
    TargetDepth,
    LossScalars,
    GradMagnitudes,
    VisibleCount,
    TotalAssignments,
    // Fused Adam buffers for each parameter group
    AdamGradPos,
    AdamMPos,
    AdamVPos,
    AdamParamPos,
    AdamGradScale,
    AdamMScale,
    AdamVScale,
    AdamParamScale,
    AdamGradOpacity,
    AdamMOpacity,
    AdamVOpacity,
    AdamParamOpacity,
    AdamGradColor,
    AdamMColor,
    AdamVColor,
    AdamParamColor,
}

impl MetalBufferSlot {
    fn label(self) -> &'static str {
        match self {
            Self::CameraUniforms => "camera_uniforms",
            Self::VisibleIndices => "visible_indices",
            Self::TileCounts => "tile_counts",
            Self::TileOffsets => "tile_offsets",
            Self::TileMetadata => "tile_metadata",
            Self::TileIndices => "tile_indices",
            Self::ProjectedGaussians => "projected_gaussians",
            Self::ProjectionRecords => "projection_records",
            Self::VisibleSourceIndices => "visible_source_indices",
            Self::GradPositions => "grad_positions",
            Self::GradRotations => "grad_rotations",
            Self::GradScales => "grad_scales",
            Self::GradOpacity => "grad_opacity",
            Self::GradColors => "grad_colors",
            Self::OutputColor => "output_color",
            Self::OutputDepth => "output_depth",
            Self::OutputAlpha => "output_alpha",
            Self::TargetColor => "target_color",
            Self::TargetDepth => "target_depth",
            Self::LossScalars => "loss_scalars",
            Self::GradMagnitudes => "grad_magnitudes",
            Self::VisibleCount => "visible_count",
            Self::TotalAssignments => "total_assignments",
            Self::AdamGradPos => "adam_grad_pos",
            Self::AdamMPos => "adam_m_pos",
            Self::AdamVPos => "adam_v_pos",
            Self::AdamParamPos => "adam_param_pos",
            Self::AdamGradScale => "adam_grad_scale",
            Self::AdamMScale => "adam_m_scale",
            Self::AdamVScale => "adam_v_scale",
            Self::AdamParamScale => "adam_param_scale",
            Self::AdamGradOpacity => "adam_grad_opacity",
            Self::AdamMOpacity => "adam_m_opacity",
            Self::AdamVOpacity => "adam_v_opacity",
            Self::AdamParamOpacity => "adam_param_opacity",
            Self::AdamGradColor => "adam_grad_color",
            Self::AdamMColor => "adam_m_color",
            Self::AdamVColor => "adam_v_color",
            Self::AdamParamColor => "adam_param_color",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetalKernel {
    FillU32,
    ProjectGaussians,
    TileCount,
    TileAssign,
    FillProjectionPadding,
    CountVisible,
    BitonicSortProjections,
    ExtractSourceIndices,
    TilePrefixSum,
    AdamStep,
    TileForward,
    TileBackward,
    GradMagnitudes,
}

impl MetalKernel {
    fn function_name(self) -> &'static str {
        match self {
            Self::FillU32 => "fill_u32",
            Self::ProjectGaussians => "project_gaussians",
            Self::TileCount => "tile_count",
            Self::TileAssign => "tile_assign",
            Self::FillProjectionPadding => "fill_projection_padding",
            Self::CountVisible => "count_visible",
            Self::BitonicSortProjections => "bitonic_sort_projections",
            Self::ExtractSourceIndices => "extract_source_indices",
            Self::TilePrefixSum => "tile_prefix_sum",
            Self::AdamStep => "adam_step",
            Self::TileForward => "tile_forward",
            Self::TileBackward => "tile_backward",
            Self::GradMagnitudes => "grad_magnitudes",
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::FillU32 => METAL_FILL_U32_KERNEL,
            Self::ProjectGaussians => METAL_PROJECT_GAUSSIANS_KERNEL,
            Self::TileCount => METAL_TILE_COUNT_KERNEL,
            Self::TileAssign => METAL_TILE_ASSIGN_KERNEL,
            Self::FillProjectionPadding => METAL_FILL_PROJECTION_PADDING_KERNEL,
            Self::CountVisible => METAL_COUNT_VISIBLE_KERNEL,
            Self::BitonicSortProjections => METAL_BITONIC_SORT_PROJECTIONS_KERNEL,
            Self::ExtractSourceIndices => METAL_EXTRACT_SOURCE_INDICES_KERNEL,
            Self::TilePrefixSum => METAL_TILE_PREFIX_SUM_KERNEL,
            Self::AdamStep => METAL_ADAM_STEP_KERNEL,
            Self::TileForward => METAL_TILE_FORWARD_KERNEL,
            Self::TileBackward => METAL_TILE_BACKWARD_KERNEL,
            Self::GradMagnitudes => METAL_GRAD_MAGNITUDE_KERNEL,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MetalRuntimeStats {
    pub tile_windows: usize,
    pub buffer_allocations: usize,
    pub buffer_reuses: usize,
    pub pipeline_compilations: usize,
}

struct PersistentMetalBuffer {
    byte_capacity: usize,
    backing: Option<Arc<Buffer>>,
}

impl PersistentMetalBuffer {
    fn new(byte_capacity: usize, backing: Option<Arc<Buffer>>) -> Self {
        Self {
            byte_capacity,
            backing,
        }
    }
}

pub(crate) struct MetalTensorView<'a> {
    storage: RwLockReadGuard<'a, Storage>,
    byte_offset: usize,
    element_count: usize,
    dtype: DType,
}

impl<'a> MetalTensorView<'a> {
    pub(crate) fn buffer(&self) -> candle_core::Result<&Buffer> {
        match &*self.storage {
            Storage::Metal(storage) => Ok(storage.buffer()),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }
}

pub(crate) struct MetalGaussianBindings<'a> {
    pub positions: MetalTensorView<'a>,
    pub scales: MetalTensorView<'a>,
    pub rotations: MetalTensorView<'a>,
    pub opacities: MetalTensorView<'a>,
    pub colors: MetalTensorView<'a>,
}

pub(crate) struct MetalRuntime {
    pub(crate) device: Device,
    tile_windows: Vec<ChunkPixelWindow>,
    num_tiles_x: usize,
    num_tiles_y: usize,
    buffers: HashMap<MetalBufferSlot, PersistentMetalBuffer>,
    pipelines: HashMap<MetalKernel, ComputePipeline>,
    stats: MetalRuntimeStats,
}

impl MetalRuntime {
    pub(crate) fn new(
        render_width: usize,
        render_height: usize,
        device: Device,
    ) -> candle_core::Result<Self> {
        let num_tiles_x = render_width.div_ceil(METAL_TILE_SIZE);
        let num_tiles_y = render_height.div_ceil(METAL_TILE_SIZE);
        let mut tile_windows = Vec::with_capacity(num_tiles_x * num_tiles_y);

        for tile_y in 0..num_tiles_y {
            for tile_x in 0..num_tiles_x {
                let min_x = tile_x * METAL_TILE_SIZE;
                let min_y = tile_y * METAL_TILE_SIZE;
                let max_x = (min_x + METAL_TILE_SIZE)
                    .min(render_width)
                    .saturating_sub(1);
                let max_y = (min_y + METAL_TILE_SIZE)
                    .min(render_height)
                    .saturating_sub(1);
                tile_windows.push(build_chunk_pixel_window(
                    &device,
                    render_width,
                    ScreenRect {
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    },
                )?);
            }
        }

        Ok(Self {
            device,
            tile_windows,
            num_tiles_x,
            num_tiles_y,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            stats: MetalRuntimeStats {
                tile_windows: num_tiles_x * num_tiles_y,
                ..Default::default()
            },
        })
    }

    pub(crate) fn tile_window(&self, tile_idx: usize) -> candle_core::Result<&ChunkPixelWindow> {
        self.tile_windows
            .get(tile_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("invalid tile index {tile_idx}")))
    }

    #[cfg(test)]
    pub(crate) fn tile_grid(&self) -> (usize, usize) {
        (self.num_tiles_x, self.num_tiles_y)
    }

    pub(crate) fn stats(&self) -> MetalRuntimeStats {
        self.stats
    }

    pub(crate) fn buffer_capacity(&self, slot: MetalBufferSlot) -> usize {
        self.buffers
            .get(&slot)
            .map(|buffer| buffer.byte_capacity)
            .unwrap_or(0)
    }

    pub(crate) fn reserve_core_buffers(
        &mut self,
        gaussian_capacity: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::CameraUniforms,
            size_of::<MetalCameraUniform>(),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::VisibleIndices,
            gaussian_capacity.saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileCounts,
            self.tile_windows.len().saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileOffsets,
            self.tile_windows.len().saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileDispatchRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            gaussian_capacity
                .saturating_mul(4)
                .saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradPositions,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradRotations,
            gaussian_capacity
                .saturating_mul(4)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradScales,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradOpacity,
            gaussian_capacity.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradColors,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        Ok(())
    }

    pub(crate) fn stage_camera(&mut self, camera: &DiffCamera) -> candle_core::Result<()> {
        let camera_uniform = MetalCameraUniform {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            width: camera.width as u32,
            height: camera.height as u32,
            tile_size: METAL_TILE_SIZE as u32,
            num_tiles_x: self.num_tiles_x as u32,
            rot00: camera.rotation[0][0],
            rot01: camera.rotation[0][1],
            rot02: camera.rotation[0][2],
            rot10: camera.rotation[1][0],
            rot11: camera.rotation[1][1],
            rot12: camera.rotation[1][2],
            rot20: camera.rotation[2][0],
            rot21: camera.rotation[2][1],
            rot22: camera.rotation[2][2],
            tx: camera.translation[0],
            ty: camera.translation[1],
            tz: camera.translation[2],
        };
        self.write_struct(MetalBufferSlot::CameraUniforms, &camera_uniform)
    }

    pub(crate) fn bind_tensor<'a>(
        &self,
        tensor: &'a Tensor,
    ) -> candle_core::Result<MetalTensorView<'a>> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            candle_core::bail!("metal tensor binding requires contiguous layout");
        }
        let byte_offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        let element_count = layout.shape().elem_count();
        match &*storage {
            Storage::Metal(_) => Ok(MetalTensorView {
                storage,
                byte_offset,
                element_count,
                dtype: tensor.dtype(),
            }),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn bind_gaussians<'a>(
        &self,
        gaussians: &'a TrainableGaussians,
    ) -> candle_core::Result<MetalGaussianBindings<'a>> {
        Ok(MetalGaussianBindings {
            positions: self.bind_tensor(gaussians.positions())?,
            scales: self.bind_tensor(gaussians.scales.as_tensor())?,
            rotations: self.bind_tensor(gaussians.rotations.as_tensor())?,
            opacities: self.bind_tensor(gaussians.opacities.as_tensor())?,
            colors: self.bind_tensor(gaussians.colors())?,
        })
    }

    pub(crate) fn project_gaussians(
        &mut self,
        gaussians: &TrainableGaussians,
        extract_visible_source_indices: bool,
    ) -> candle_core::Result<ProjectedGpuBatch> {
        let gaussian_count = gaussians.len();
        if gaussian_count == 0 {
            return Ok(ProjectedGpuBatch {
                visible_count: 0,
                visible_source_indices: Vec::new(),
            });
        }
        let padded_count = gaussian_count.next_power_of_two();
        self.ensure_buffer(
            MetalBufferSlot::ProjectionRecords,
            padded_count.saturating_mul(size_of::<MetalProjectionRecord>()),
        )?;
        self.ensure_buffer(MetalBufferSlot::VisibleCount, size_of::<u32>())?;
        let pipeline = self.ensure_pipeline(MetalKernel::ProjectGaussians)?.clone();
        let bindings = self.bind_gaussians(gaussians)?;
        let metal = self.device.as_metal_device()?.clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::ProjectGaussians.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            Some(bindings.positions.buffer()?),
            bindings.positions.byte_offset(),
        );
        encoder.set_buffer(
            2,
            Some(bindings.scales.buffer()?),
            bindings.scales.byte_offset(),
        );
        encoder.set_buffer(
            3,
            Some(bindings.rotations.buffer()?),
            bindings.rotations.byte_offset(),
        );
        encoder.set_buffer(
            4,
            Some(bindings.opacities.buffer()?),
            bindings.opacities.byte_offset(),
        );
        encoder.set_buffer(
            5,
            Some(bindings.colors.buffer()?),
            bindings.colors.byte_offset(),
        );
        encoder.set_buffer(
            6,
            Some(
                self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                    .ok_or_else(|| candle_core::Error::Msg("missing projection records".into()))?
                    .as_ref(),
            ),
            0,
        );
        let count = gaussian_count as u32;
        encoder.set_bytes(7, &count);
        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(gaussian_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: gaussian_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;

        if padded_count > gaussian_count {
            let pipeline = self
                .ensure_pipeline(MetalKernel::FillProjectionPadding)?
                .clone();
            let encoder = metal.command_encoder()?;
            encoder.set_label(MetalKernel::FillProjectionPadding.function_name());
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(
                0,
                self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                    .map(|buffer| buffer.as_ref()),
                0,
            );
            let start_idx = gaussian_count as u32;
            let total_count = padded_count as u32;
            encoder.set_bytes(1, &start_idx);
            encoder.set_bytes(2, &total_count);
            let padding_count = padded_count - gaussian_count;
            let threads_per_group = pipeline
                .max_total_threads_per_threadgroup()
                .min(padding_count)
                .max(1);
            encoder.dispatch_threads(
                MTLSize {
                    width: padding_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group,
                    height: 1,
                    depth: 1,
                },
            );
            drop(encoder);
            self.device.synchronize()?;
        }

        let sort_pipeline = self
            .ensure_pipeline(MetalKernel::BitonicSortProjections)?
            .clone();
        let metal = self.device.as_metal_device()?.clone();
        let total_count = padded_count as u32;
        let threads_per_group = sort_pipeline
            .max_total_threads_per_threadgroup()
            .min(padded_count)
            .max(1);
        let mut k = 2usize;
        while k <= padded_count {
            let mut j = k >> 1;
            while j > 0 {
                // All stages share a single command encoder — no synchronize between stages.
                // Metal guarantees memory ordering within a single encoder's dispatches.
                let encoder = metal.command_encoder()?;
                encoder.set_label(MetalKernel::BitonicSortProjections.function_name());
                encoder.set_compute_pipeline_state(&sort_pipeline);
                encoder.set_buffer(
                    0,
                    self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                        .map(|buffer| buffer.as_ref()),
                    0,
                );
                let stage_j = j as u32;
                let stage_k = k as u32;
                encoder.set_bytes(1, &stage_j);
                encoder.set_bytes(2, &stage_k);
                encoder.set_bytes(3, &total_count);
                encoder.dispatch_threads(
                    MTLSize {
                        width: padded_count,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads_per_group,
                        height: 1,
                        depth: 1,
                    },
                );
                drop(encoder);
                // No synchronize here — stages within one sort can pipeline on GPU
                j >>= 1;
            }
            k <<= 1;
        }
        // Single synchronize after the entire sort completes
        self.device.synchronize()?;

        self.dispatch_fill_u32(MetalBufferSlot::VisibleCount, 0, 1)?;
        let pipeline = self.ensure_pipeline(MetalKernel::CountVisible)?.clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::CountVisible.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            self.buffer_handle(MetalBufferSlot::VisibleCount)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        let record_count = gaussian_count as u32;
        encoder.set_bytes(2, &record_count);
        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(gaussian_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: gaussian_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;

        let visible_count = self.read_buffer_structs::<u32>(MetalBufferSlot::VisibleCount, 1)?;
        let visible_count = visible_count.first().copied().unwrap_or(0) as usize;
        let visible_source_indices = if extract_visible_source_indices && visible_count > 0 {
            self.ensure_buffer(
                MetalBufferSlot::VisibleSourceIndices,
                visible_count.saturating_mul(size_of::<u32>()),
            )?;
            let pipeline = self
                .ensure_pipeline(MetalKernel::ExtractSourceIndices)?
                .clone();
            let encoder = metal.command_encoder()?;
            encoder.set_label(MetalKernel::ExtractSourceIndices.function_name());
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(
                0,
                self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                    .map(|buffer| buffer.as_ref()),
                0,
            );
            encoder.set_buffer(
                1,
                self.buffer_handle(MetalBufferSlot::VisibleSourceIndices)?
                    .map(|buffer| buffer.as_ref()),
                0,
            );
            let count = visible_count as u32;
            encoder.set_bytes(2, &count);
            let threads_per_group = pipeline
                .max_total_threads_per_threadgroup()
                .min(visible_count)
                .max(1);
            encoder.dispatch_threads(
                MTLSize {
                    width: visible_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group,
                    height: 1,
                    depth: 1,
                },
            );
            drop(encoder);
            self.device.synchronize()?;
            self.read_buffer_structs::<u32>(MetalBufferSlot::VisibleSourceIndices, visible_count)?
        } else {
            Vec::new()
        };
        Ok(ProjectedGpuBatch {
            visible_count,
            visible_source_indices,
        })
    }

    pub(crate) fn ensure_projection_record_buffer(
        &mut self,
        count: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::ProjectionRecords,
            count.saturating_mul(size_of::<MetalProjectionRecord>()),
        )
    }

    pub(crate) fn write_projection_records(
        &mut self,
        records: &[MetalProjectionRecord],
    ) -> candle_core::Result<()> {
        self.write_slice(MetalBufferSlot::ProjectionRecords, records)
    }

    pub(crate) fn build_tile_bins(
        &self,
        min_x_values: &[f32],
        max_x_values: &[f32],
        min_y_values: &[f32],
        max_y_values: &[f32],
    ) -> candle_core::Result<MetalTileBins> {
        let total = min_x_values.len();
        if max_x_values.len() != total || min_y_values.len() != total || max_y_values.len() != total
        {
            candle_core::bail!("tile binning expects matching bound lengths");
        }

        let tile_count = self.num_tiles_x.saturating_mul(self.num_tiles_y);
        let mut tile_counts = vec![0usize; tile_count];
        let mut total_assignments = 0usize;

        for idx in 0..total {
            let tile_x_min = (min_x_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_x_max = (max_x_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_y_min = (min_y_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_y_max = (max_y_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;

            for ty in tile_y_min..=tile_y_max.min(self.num_tiles_y.saturating_sub(1)) {
                for tx in tile_x_min..=tile_x_max.min(self.num_tiles_x.saturating_sub(1)) {
                    tile_counts[ty * self.num_tiles_x + tx] += 1;
                    total_assignments += 1;
                }
            }
        }

        let mut records = Vec::with_capacity(tile_count);
        let mut active_tiles = Vec::new();
        let mut max_gaussians_per_tile = 0usize;
        let mut start = 0usize;
        for (tile_idx, count) in tile_counts.iter().copied().enumerate() {
            records.push(MetalTileDispatchRecord::new(start as u32, count as u32));
            if count > 0 {
                active_tiles.push(tile_idx);
                max_gaussians_per_tile = max_gaussians_per_tile.max(count);
            }
            start += count;
        }

        let mut packed_indices = vec![0u32; total_assignments];
        let mut write_offsets: Vec<usize> = records.iter().map(|record| record.start()).collect();
        for idx in 0..total {
            let tile_x_min = (min_x_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_x_max = (max_x_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_y_min = (min_y_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
            let tile_y_max = (max_y_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;

            for ty in tile_y_min..=tile_y_max.min(self.num_tiles_y.saturating_sub(1)) {
                for tx in tile_x_min..=tile_x_max.min(self.num_tiles_x.saturating_sub(1)) {
                    let tile_idx = ty * self.num_tiles_x + tx;
                    let write_idx = write_offsets[tile_idx];
                    packed_indices[write_idx] = idx as u32;
                    write_offsets[tile_idx] += 1;
                }
            }
        }

        Ok(MetalTileBins {
            records,
            active_tiles,
            packed_indices,
            total_assignments,
            max_gaussians_per_tile,
        })
    }

    pub(crate) fn build_tile_bins_gpu(
        &mut self,
        gaussian_count: usize,
    ) -> candle_core::Result<MetalTileBins> {
        let tile_count = self.num_tiles_x.saturating_mul(self.num_tiles_y);
        if gaussian_count == 0 {
            return Ok(MetalTileBins::default());
        }
        self.ensure_buffer(
            MetalBufferSlot::TileCounts,
            tile_count.saturating_mul(size_of::<u32>()),
        )?;
        self.dispatch_fill_u32(MetalBufferSlot::TileCounts, 0, tile_count)?;

        let count_pipeline = self.ensure_pipeline(MetalKernel::TileCount)?.clone();
        let metal = self.device.as_metal_device()?.clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::TileCount.function_name());
        encoder.set_compute_pipeline_state(&count_pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            2,
            self.buffer_handle(MetalBufferSlot::TileCounts)?
                .map(|b| b.as_ref()),
            0,
        );
        let count = gaussian_count as u32;
        encoder.set_bytes(3, &count);
        let threads_per_group = count_pipeline
            .max_total_threads_per_threadgroup()
            .min(gaussian_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: gaussian_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;

        // GPU prefix-sum: compute exclusive scan of tile_counts → tile_offsets
        // and total assignment count, without reading tile_counts back to CPU.
        let tile_count_u32 = tile_count as u32;
        self.ensure_buffer(
            MetalBufferSlot::TileOffsets,
            tile_count.saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(MetalBufferSlot::TotalAssignments, size_of::<u32>())?;

        let prefix_pipeline = self.ensure_pipeline(MetalKernel::TilePrefixSum)?.clone();
        let metal2 = self.device.as_metal_device()?.clone();
        let prefix_encoder = metal2.command_encoder()?;
        prefix_encoder.set_label(MetalKernel::TilePrefixSum.function_name());
        prefix_encoder.set_compute_pipeline_state(&prefix_pipeline);
        prefix_encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::TileCounts)?
                .map(|b| b.as_ref()),
            0,
        );
        prefix_encoder.set_buffer(
            1,
            self.buffer_handle(MetalBufferSlot::TileOffsets)?
                .map(|b| b.as_ref()),
            0,
        );
        prefix_encoder.set_buffer(
            2,
            self.buffer_handle(MetalBufferSlot::TotalAssignments)?
                .map(|b| b.as_ref()),
            0,
        );
        prefix_encoder.set_bytes(3, &tile_count_u32);
        // Single-thread sequential scan — correct for any tile_count
        prefix_encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        drop(prefix_encoder);
        self.device.synchronize()?;

        // Read back the total assignment count (1 u32) and tile offsets
        let total_vec = self.read_buffer_structs::<u32>(MetalBufferSlot::TotalAssignments, 1)?;
        let total_assignments = total_vec.first().copied().unwrap_or(0) as usize;

        // Read tile_counts and tile_offsets to build MetalTileDispatchRecords on CPU
        // (needed for rasterizer dispatch).
        let tile_counts_u32 =
            self.read_buffer_structs::<u32>(MetalBufferSlot::TileCounts, tile_count)?;
        let tile_offsets_u32 =
            self.read_buffer_structs::<u32>(MetalBufferSlot::TileOffsets, tile_count)?;
        let mut records = Vec::with_capacity(tile_count);
        let mut active_tiles = Vec::new();
        let mut max_gaussians_per_tile = 0usize;
        for tile_idx in 0..tile_count {
            let cnt = tile_counts_u32[tile_idx];
            let off = tile_offsets_u32[tile_idx];
            records.push(MetalTileDispatchRecord::new(off, cnt));
            if cnt > 0 {
                active_tiles.push(tile_idx);
                max_gaussians_per_tile = max_gaussians_per_tile.max(cnt as usize);
            }
        }

        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            total_assignments.saturating_mul(size_of::<u32>()),
        )?;

        let assign_pipeline = self.ensure_pipeline(MetalKernel::TileAssign)?.clone();
        let metal = self.device.as_metal_device()?.clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::TileAssign.function_name());
        encoder.set_compute_pipeline_state(&assign_pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            2,
            self.buffer_handle(MetalBufferSlot::TileOffsets)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            3,
            self.buffer_handle(MetalBufferSlot::TileIndices)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_bytes(4, &count);
        let threads_per_group = assign_pipeline
            .max_total_threads_per_threadgroup()
            .min(gaussian_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: gaussian_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;
        let packed_indices =
            self.read_buffer_structs::<u32>(MetalBufferSlot::TileIndices, total_assignments)?;

        Ok(MetalTileBins {
            records,
            active_tiles,
            packed_indices,
            total_assignments,
            max_gaussians_per_tile,
        })
    }

    pub(crate) fn reserve_forward_buffers(
        &mut self,
        gaussian_count: usize,
        tile_ref_count: usize,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::ProjectedGaussians,
            gaussian_count.saturating_mul(size_of::<MetalProjectedGaussian>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileDispatchRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            tile_ref_count.saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputColor,
            pixel_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputDepth,
            pixel_count.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputAlpha,
            pixel_count.saturating_mul(size_of::<f32>()),
        )?;
        Ok(())
    }

    pub(crate) fn rasterize_forward(
        &mut self,
        gaussian_count: usize,
        tile_bins: &MetalTileBins,
        render_width: usize,
        render_height: usize,
    ) -> candle_core::Result<(NativeForwardFrame, NativeForwardProfile)> {
        let total_start = std::time::Instant::now();
        let pixel_count = render_width.saturating_mul(render_height);
        let setup_start = std::time::Instant::now();
        self.reserve_forward_buffers(gaussian_count, tile_bins.total_assignments(), pixel_count)?;
        let pipeline = self.ensure_pipeline(MetalKernel::TileForward)?.clone();
        let color_buffer = self
            .buffer_handle(MetalBufferSlot::OutputColor)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output color buffer".into()))?;
        let depth_buffer = self
            .buffer_handle(MetalBufferSlot::OutputDepth)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output depth buffer".into()))?;
        let alpha_buffer = self
            .buffer_handle(MetalBufferSlot::OutputAlpha)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output alpha buffer".into()))?;
        let tile_buffer = self
            .buffer_handle(MetalBufferSlot::TileMetadata)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing tile metadata buffer".into()))?;
        let tile_index_buffer = self
            .buffer_handle(MetalBufferSlot::TileIndices)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing tile index buffer".into()))?;
        let gaussian_buffer = self
            .buffer_handle(MetalBufferSlot::ProjectionRecords)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing gaussian buffer".into()))?;
        let setup = setup_start.elapsed();

        let staging_start = std::time::Instant::now();
        self.write_slice(MetalBufferSlot::TileMetadata, tile_bins.records())?;
        self.write_slice(MetalBufferSlot::TileIndices, tile_bins.packed_indices())?;
        let staging = staging_start.elapsed();

        let kernel_start = std::time::Instant::now();
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::TileForward.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        encoder.set_buffer(1, Some(tile_buffer.as_ref()), 0);
        encoder.set_buffer(2, Some(tile_index_buffer.as_ref()), 0);
        encoder.set_buffer(3, Some(gaussian_buffer.as_ref()), 0);
        encoder.set_buffer(4, Some(color_buffer.as_ref()), 0);
        encoder.set_buffer(5, Some(depth_buffer.as_ref()), 0);
        encoder.set_buffer(6, Some(alpha_buffer.as_ref()), 0);
        let threads_per_group = tile_group_dims(&pipeline);
        encoder.dispatch_threads(
            MTLSize {
                width: render_width,
                height: render_height,
                depth: 1,
            },
            threads_per_group,
        );
        drop(encoder);
        self.device.synchronize()?;
        let kernel = kernel_start.elapsed();

        let frame = NativeForwardFrame {
            color: self.tensor_from_buffer(
                MetalBufferSlot::OutputColor,
                pixel_count.saturating_mul(3),
                DType::F32,
                (pixel_count, 3),
            )?,
            depth: self.tensor_from_buffer(
                MetalBufferSlot::OutputDepth,
                pixel_count,
                DType::F32,
                (pixel_count,),
            )?,
            alpha: self.tensor_from_buffer(
                MetalBufferSlot::OutputAlpha,
                pixel_count,
                DType::F32,
                (pixel_count,),
            )?,
        };

        Ok((
            frame,
            NativeForwardProfile {
                setup,
                staging,
                kernel,
                total: total_start.elapsed(),
            },
        ))
    }

    pub(crate) fn reserve_backward_buffers(
        &mut self,
        gaussian_count: usize,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::TargetColor,
            pixel_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TargetDepth,
            pixel_count.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(MetalBufferSlot::LossScalars, 2 * size_of::<f32>())?;
        self.ensure_buffer(
            MetalBufferSlot::GradPositions,
            gaussian_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradScales,
            gaussian_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradOpacity,
            gaussian_count.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradColors,
            gaussian_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count.saturating_mul(size_of::<f32>()),
        )?;
        Ok(())
    }

    pub(crate) fn write_target_data(
        &mut self,
        target_color: &[f32],
        target_depth: &[f32],
        color_scale: f32,
        depth_scale: f32,
    ) -> candle_core::Result<()> {
        self.write_slice(MetalBufferSlot::TargetColor, target_color)?;
        self.write_slice(MetalBufferSlot::TargetDepth, target_depth)?;
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct LossScalars {
            color: f32,
            depth: f32,
        }
        self.write_struct(
            MetalBufferSlot::LossScalars,
            &LossScalars {
                color: color_scale,
                depth: depth_scale,
            },
        )
    }

    pub(crate) fn rasterize_backward(
        &mut self,
        gaussian_count: usize,
        tile_bins: &MetalTileBins,
        render_width: usize,
        render_height: usize,
    ) -> candle_core::Result<(NativeBackwardFrame, NativeForwardProfile)> {
        let total_start = std::time::Instant::now();
        let pixel_count = render_width.saturating_mul(render_height);

        // Ensure all buffers exist
        self.reserve_backward_buffers(gaussian_count, pixel_count)?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileDispatchRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            tile_bins
                .total_assignments()
                .saturating_mul(size_of::<u32>()),
        )?;

        let pipeline = self.ensure_pipeline(MetalKernel::TileBackward)?.clone();

        // Zero out gradient buffers
        self.dispatch_fill_u32(MetalBufferSlot::GradPositions, 0, gaussian_count * 3)?;
        self.dispatch_fill_u32(MetalBufferSlot::GradScales, 0, gaussian_count * 3)?;
        self.dispatch_fill_u32(MetalBufferSlot::GradOpacity, 0, gaussian_count)?;
        self.dispatch_fill_u32(MetalBufferSlot::GradColors, 0, gaussian_count * 3)?;

        let setup = total_start.elapsed();

        let staging_start = std::time::Instant::now();
        self.write_slice(MetalBufferSlot::TileMetadata, tile_bins.records())?;
        self.write_slice(MetalBufferSlot::TileIndices, tile_bins.packed_indices())?;
        let staging = staging_start.elapsed();

        let kernel_start = std::time::Instant::now();
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::TileBackward.function_name());
        encoder.set_compute_pipeline_state(&pipeline);

        // Bind buffers: camera(0), tile_records(1), tile_indices(2), gaussians(3)
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
                .map(|b| b.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            Some(
                self.buffer_handle(MetalBufferSlot::TileMetadata)?
                    .ok_or_else(|| candle_core::Error::Msg("missing tile metadata".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            2,
            Some(
                self.buffer_handle(MetalBufferSlot::TileIndices)?
                    .ok_or_else(|| candle_core::Error::Msg("missing tile indices".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            3,
            Some(
                self.buffer_handle(MetalBufferSlot::ProjectionRecords)?
                    .ok_or_else(|| candle_core::Error::Msg("missing projected gaussians".into()))?
                    .as_ref(),
            ),
            0,
        );

        // rendered_color(4), rendered_depth(5), rendered_alpha(6)
        encoder.set_buffer(
            4,
            Some(
                self.buffer_handle(MetalBufferSlot::OutputColor)?
                    .ok_or_else(|| candle_core::Error::Msg("missing output color".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            5,
            Some(
                self.buffer_handle(MetalBufferSlot::OutputDepth)?
                    .ok_or_else(|| candle_core::Error::Msg("missing output depth".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            6,
            Some(
                self.buffer_handle(MetalBufferSlot::OutputAlpha)?
                    .ok_or_else(|| candle_core::Error::Msg("missing output alpha".into()))?
                    .as_ref(),
            ),
            0,
        );

        // target_color(7), target_depth(8)
        encoder.set_buffer(
            7,
            Some(
                self.buffer_handle(MetalBufferSlot::TargetColor)?
                    .ok_or_else(|| candle_core::Error::Msg("missing target color".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            8,
            Some(
                self.buffer_handle(MetalBufferSlot::TargetDepth)?
                    .ok_or_else(|| candle_core::Error::Msg("missing target depth".into()))?
                    .as_ref(),
            ),
            0,
        );

        // grad_positions(9), grad_scales(10), grad_opacity(11), grad_colors(12)
        encoder.set_buffer(
            9,
            Some(
                self.buffer_handle(MetalBufferSlot::GradPositions)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad positions".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            10,
            Some(
                self.buffer_handle(MetalBufferSlot::GradScales)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad scales".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            11,
            Some(
                self.buffer_handle(MetalBufferSlot::GradOpacity)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad opacity".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            12,
            Some(
                self.buffer_handle(MetalBufferSlot::GradColors)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad colors".into()))?
                    .as_ref(),
            ),
            0,
        );

        // loss_scalars(13)
        encoder.set_buffer(
            13,
            Some(
                self.buffer_handle(MetalBufferSlot::LossScalars)?
                    .ok_or_else(|| candle_core::Error::Msg("missing loss scalars".into()))?
                    .as_ref(),
            ),
            0,
        );

        let threads_per_group = tile_group_dims(&pipeline);
        encoder.dispatch_threads(
            MTLSize {
                width: render_width,
                height: render_height,
                depth: 1,
            },
            threads_per_group,
        );
        drop(encoder);
        self.device.synchronize()?;
        let kernel = kernel_start.elapsed();

        // Read back gradients from GPU
        let frame = NativeBackwardFrame {
            grad_positions: self.tensor_from_buffer(
                MetalBufferSlot::GradPositions,
                gaussian_count * 3,
                DType::F32,
                (gaussian_count, 3),
            )?,
            grad_log_scales: self.tensor_from_buffer(
                MetalBufferSlot::GradScales,
                gaussian_count * 3,
                DType::F32,
                (gaussian_count, 3),
            )?,
            grad_opacity_logits: self.tensor_from_buffer(
                MetalBufferSlot::GradOpacity,
                gaussian_count,
                DType::F32,
                (gaussian_count,),
            )?,
            grad_colors: self.tensor_from_buffer(
                MetalBufferSlot::GradColors,
                gaussian_count * 3,
                DType::F32,
                (gaussian_count, 3),
            )?,
        };

        Ok((
            frame,
            NativeForwardProfile {
                setup,
                staging,
                kernel,
                total: total_start.elapsed(),
            },
        ))
    }

    /// Fused Adam update: runs entirely on GPU, no intermediate Tensor allocations.
    /// `param_buf` is the raw Metal buffer of a `Var`'s storage.
    pub(crate) fn adam_step_fused(
        &mut self,
        param_slot: MetalBufferSlot,
        grad_slot: MetalBufferSlot,
        m_slot: MetalBufferSlot,
        v_slot: MetalBufferSlot,
        element_count: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
    ) -> candle_core::Result<()> {
        if element_count == 0 {
            return Ok(());
        }
        let bc1 = 1.0f32 / (1.0 - beta1.powi(step as i32));
        let bc2 = 1.0f32 / (1.0 - beta2.powi(step as i32));

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AdamHyperparams {
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            bc1: f32,
            bc2: f32,
        }
        let hp = AdamHyperparams {
            lr,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
        };

        let pipeline = self.ensure_pipeline(MetalKernel::AdamStep)?.clone();
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::AdamStep.function_name());
        encoder.set_compute_pipeline_state(&pipeline);

        let get_buf = |slot: MetalBufferSlot, name: &'static str, rt: &MetalRuntime| {
            rt.buffers
                .get(&slot)
                .and_then(|b| b.backing.as_ref().cloned())
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("adam_step_fused: missing {name}").into())
                })
        };

        encoder.set_buffer(0, Some(get_buf(param_slot, "params", self)?.as_ref()), 0);
        encoder.set_buffer(1, Some(get_buf(grad_slot, "grads", self)?.as_ref()), 0);
        encoder.set_buffer(2, Some(get_buf(m_slot, "m", self)?.as_ref()), 0);
        encoder.set_buffer(3, Some(get_buf(v_slot, "v", self)?.as_ref()), 0);
        self.write_struct(MetalBufferSlot::LossScalars, &hp)?; // reuse small temp slot
        encoder.set_buffer(
            4,
            self.buffer_handle(MetalBufferSlot::LossScalars)?
                .map(|b| b.as_ref()),
            0,
        );

        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(element_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: element_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok(())
    }

    pub(crate) fn compute_grad_magnitudes(
        &mut self,
        gaussian_count: usize,
    ) -> candle_core::Result<Tensor> {
        if gaussian_count == 0 {
            return Tensor::zeros((0,), DType::F32, &self.device);
        }
        self.ensure_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count.saturating_mul(size_of::<f32>()),
        )?;
        let pipeline = self.ensure_pipeline(MetalKernel::GradMagnitudes)?.clone();
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::GradMagnitudes.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(
                self.buffer_handle(MetalBufferSlot::GradPositions)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad positions".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            1,
            Some(
                self.buffer_handle(MetalBufferSlot::GradScales)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad scales".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            2,
            Some(
                self.buffer_handle(MetalBufferSlot::GradOpacity)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad opacity".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            3,
            Some(
                self.buffer_handle(MetalBufferSlot::GradColors)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad colors".into()))?
                    .as_ref(),
            ),
            0,
        );
        encoder.set_buffer(
            4,
            Some(
                self.buffer_handle(MetalBufferSlot::GradMagnitudes)?
                    .ok_or_else(|| candle_core::Error::Msg("missing grad magnitudes".into()))?
                    .as_ref(),
            ),
            0,
        );
        let count = gaussian_count as u32;
        encoder.set_bytes(5, &count);
        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(gaussian_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: gaussian_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;
        self.tensor_from_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count,
            DType::F32,
            (gaussian_count,),
        )
    }

    pub(crate) fn dispatch_fill_u32(
        &mut self,
        slot: MetalBufferSlot,
        value: u32,
        len: usize,
    ) -> candle_core::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_buffer(slot, len.saturating_mul(size_of::<u32>()))?;
        let pipeline = self.ensure_pipeline(MetalKernel::FillU32)?.clone();
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("fill_u32 requires a Metal device");
        };
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::FillU32.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(buffer.as_ref()), 0);
        encoder.set_bytes(1, &value);
        let count = len as u32;
        encoder.set_bytes(2, &count);
        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(len).max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: len,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        self.device.synchronize()?;
        Ok(())
    }

    pub(crate) fn stage_tensor_from_slice<T: candle_core::WithDType + Copy, S: Into<Shape>>(
        &mut self,
        slot: MetalBufferSlot,
        values: &[T],
        shape: S,
    ) -> candle_core::Result<Tensor> {
        let shape = shape.into();
        if !self.device.is_metal() {
            return Tensor::from_slice(values, shape, &self.device);
        }
        self.write_slice(slot, values)?;
        self.tensor_from_buffer(slot, values.len(), T::DTYPE, shape)
    }

    pub(crate) fn read_tensor_flat<T: candle_core::WithDType + Copy>(
        &self,
        tensor: &Tensor,
    ) -> candle_core::Result<Vec<T>> {
        let element_count: usize = tensor.elem_count();
        if !self.device.is_metal() {
            return tensor.flatten_all()?.to_vec1::<T>();
        }

        let view = self.bind_tensor(tensor)?;
        if view.dtype() != T::DTYPE {
            candle_core::bail!(
                "tensor dtype mismatch for runtime read: expected {:?}, got {:?}",
                T::DTYPE,
                view.dtype()
            );
        }
        if view.element_count() != element_count {
            candle_core::bail!(
                "tensor element count mismatch for runtime read: expected {element_count}, got {}",
                view.element_count()
            );
        }

        self.device.synchronize()?;
        let buffer = view.buffer()?;
        let byte_offset = view.byte_offset();
        let ptr = unsafe { (buffer.contents() as *const u8).add(byte_offset) }.cast::<T>();
        let values = unsafe { std::slice::from_raw_parts(ptr, element_count) };
        Ok(values.to_vec())
    }

    pub(crate) fn read_buffer_structs<T: Copy>(
        &self,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<T>> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        self.device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) };
        Ok(values.to_vec())
    }

    #[cfg(test)]
    pub(crate) fn read_u32_buffer(
        &self,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<u32>> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        self.device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, len) };
        Ok(values.to_vec())
    }

    fn buffer_handle(&self, slot: MetalBufferSlot) -> candle_core::Result<Option<&Arc<Buffer>>> {
        Ok(self
            .buffers
            .get(&slot)
            .map(|buffer| buffer.backing.as_ref())
            .flatten())
    }

    fn ensure_pipeline(&mut self, kernel: MetalKernel) -> candle_core::Result<&ComputePipeline> {
        if !self.pipelines.contains_key(&kernel) {
            let metal = self.device.as_metal_device()?;
            let library = metal
                .new_library_with_source(kernel.source(), None)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            let function = library
                .get_function(kernel.function_name(), None)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            let pipeline = metal
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            self.pipelines.insert(kernel, pipeline);
            self.stats.pipeline_compilations += 1;
        }
        Ok(self
            .pipelines
            .get(&kernel)
            .expect("pipeline inserted before lookup"))
    }

    pub(crate) fn ensure_buffer(
        &mut self,
        slot: MetalBufferSlot,
        byte_capacity: usize,
    ) -> candle_core::Result<()> {
        let required = byte_capacity.max(1);
        if let Some(existing) = self.buffers.get(&slot) {
            if existing.byte_capacity >= required {
                self.stats.buffer_reuses += 1;
                return Ok(());
            }
        }

        let backing = if self.device.is_metal() {
            let metal = self.device.as_metal_device()?;
            Some(metal.new_buffer(required, DType::U8, slot.label())?)
        } else {
            None
        };
        self.buffers
            .insert(slot, PersistentMetalBuffer::new(required, backing));
        self.stats.buffer_allocations += 1;
        Ok(())
    }

    fn tensor_from_buffer<S: Into<Shape>>(
        &self,
        slot: MetalBufferSlot,
        element_count: usize,
        dtype: DType,
        shape: S,
    ) -> candle_core::Result<Tensor> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        let metal = self.device.as_metal_device()?.clone();
        let storage = Storage::Metal(MetalStorage::new(buffer, metal, element_count, dtype));
        Ok(Tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    pub(crate) fn write_struct<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        value: &T,
    ) -> candle_core::Result<()> {
        let data =
            unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.write_bytes(slot, data)
    }

    pub(crate) fn write_slice<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        values: &[T],
    ) -> candle_core::Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        self.write_bytes(slot, data)
    }

    fn write_bytes(&mut self, slot: MetalBufferSlot, data: &[u8]) -> candle_core::Result<()> {
        self.ensure_buffer(slot, data.len())?;
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            return Ok(());
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.contents().cast::<u8>(),
                data.len(),
            );
        }
        buffer.did_modify_range(NSRange::new(0, data.len()));
        Ok(())
    }
}

fn tile_group_dims(pipeline: &ComputePipeline) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let side = (max_threads as f64).sqrt().floor() as usize;
    let width = side.clamp(1, METAL_TILE_SIZE);
    let height = (max_threads / width).max(1).min(METAL_TILE_SIZE);
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn build_chunk_pixel_window(
    device: &Device,
    render_width: usize,
    rect: ScreenRect,
) -> candle_core::Result<ChunkPixelWindow> {
    let width = rect.max_x - rect.min_x + 1;
    let height = rect.max_y - rect.min_y + 1;
    let mut xs = Vec::with_capacity(width * height);
    let mut ys = Vec::with_capacity(width * height);
    let mut indices = Vec::with_capacity(width * height);

    for y in rect.min_y..=rect.max_y {
        for x in rect.min_x..=rect.max_x {
            xs.push(x as f32 + 0.5);
            ys.push(y as f32 + 0.5);
            indices.push((y * render_width + x) as u32);
        }
    }

    let pixel_count = indices.len();
    Ok(ChunkPixelWindow {
        pixel_x: Tensor::from_slice(&xs, (1, pixel_count), device)?,
        pixel_y: Tensor::from_slice(&ys, (1, pixel_count), device)?,
        indices: Tensor::from_slice(&indices, pixel_count, device)?,
        pixel_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_windows_cover_entire_surface() {
        let runtime = MetalRuntime::new(32, 17, Device::Cpu).unwrap();
        assert_eq!(runtime.tile_grid(), (2, 2));
        assert_eq!(runtime.stats().tile_windows, 4);
        assert_eq!(runtime.tile_window(0).unwrap().pixel_count, 16 * 16);
        assert_eq!(runtime.tile_window(3).unwrap().pixel_count, 16);
    }

    #[test]
    fn core_buffer_reservations_reuse_capacity() {
        let mut runtime = MetalRuntime::new(32, 16, Device::Cpu).unwrap();
        runtime.reserve_core_buffers(64).unwrap();
        let initial = runtime.stats();
        assert_eq!(initial.buffer_allocations, 11);

        runtime.reserve_core_buffers(32).unwrap();
        let reused = runtime.stats();
        assert_eq!(reused.buffer_allocations, initial.buffer_allocations);
        assert!(reused.buffer_reuses >= 11);
    }

    #[test]
    fn fill_kernel_writes_shared_buffer() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };

        let mut runtime = MetalRuntime::new(16, 16, device).unwrap();
        runtime
            .dispatch_fill_u32(MetalBufferSlot::TileIndices, 7, 8)
            .unwrap();
        let values = runtime
            .read_u32_buffer(MetalBufferSlot::TileIndices, 8)
            .unwrap();
        assert_eq!(values, vec![7; 8]);
    }

    #[test]
    fn tile_bins_pack_indices_by_tile() {
        let runtime = MetalRuntime::new(32, 16, Device::Cpu).unwrap();
        let bins = runtime
            .build_tile_bins(
                &[2.0f32, 14.0],
                &[15.0f32, 18.0],
                &[1.0f32, 1.0],
                &[14.0f32, 14.0],
            )
            .unwrap();

        assert_eq!(bins.active_tile_count(), 2);
        assert_eq!(bins.total_assignments(), 3);
        assert_eq!(bins.max_gaussians_per_tile(), 2);
        assert_eq!(bins.packed_indices(), &[0, 1, 1]);
        assert_eq!(bins.indices_for_tile(0), &[0, 1]);
        assert_eq!(bins.indices_for_tile(1), &[1]);
    }

    #[test]
    fn stage_tensor_from_slice_falls_back_to_cpu_tensor() {
        let mut runtime = MetalRuntime::new(16, 16, Device::Cpu).unwrap();
        let tensor = runtime
            .stage_tensor_from_slice(MetalBufferSlot::TileIndices, &[1u32, 2, 3], 3)
            .unwrap();
        assert_eq!(tensor.to_vec1::<u32>().unwrap(), vec![1, 2, 3]);
    }
}
