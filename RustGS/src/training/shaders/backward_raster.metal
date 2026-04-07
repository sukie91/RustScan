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

struct BwdLossScalars {
    float color_scale;
    float depth_scale;
    float ssim_scale;
    float alpha_scale;
};

kernel void tile_backward(
    constant BwdCameraUniform& camera [[buffer(0)]],
    device const BwdTileRecord* tile_records [[buffer(1)]],
    device const uint* tile_indices [[buffer(2)]],
    device const BwdProjectedGaussian* gaussians [[buffer(3)]],
    device const float* rendered_color [[buffer(4)]],
    device const float* rendered_depth [[buffer(5)]],
    device const float* rendered_alpha [[buffer(6)]],
    device const float* target_color [[buffer(7)]],
    device const float* target_depth [[buffer(8)]],
    device atomic_float* grad_positions [[buffer(9)]],
    device atomic_float* grad_log_scales [[buffer(10)]],
    device atomic_float* grad_opacity_logits [[buffer(11)]],
    device atomic_float* grad_colors [[buffer(12)]],
    constant BwdLossScalars& loss_scalars [[buffer(13)]],
    device const float* ssim_color_grad [[buffer(14)]],
    device atomic_float* grad_projected_positions [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]
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

    const float dc_r = ((final_color_r > target_color[c3 + 0]) ? loss_scalars.color_scale
                      : (final_color_r < target_color[c3 + 0]) ? -loss_scalars.color_scale : 0.0f)
                      + ssim_color_grad[c3 + 0] * loss_scalars.ssim_scale;
    const float dc_g = ((final_color_g > target_color[c3 + 1]) ? loss_scalars.color_scale
                      : (final_color_g < target_color[c3 + 1]) ? -loss_scalars.color_scale : 0.0f)
                      + ssim_color_grad[c3 + 1] * loss_scalars.ssim_scale;
    const float dc_b = ((final_color_b > target_color[c3 + 2]) ? loss_scalars.color_scale
                      : (final_color_b < target_color[c3 + 2]) ? -loss_scalars.color_scale : 0.0f)
                      + ssim_color_grad[c3 + 2] * loss_scalars.ssim_scale;

    float dd_depth = 0.0f;
    if (loss_scalars.depth_scale > 0.0f && target_depth[pixel_idx] > 0.0f) {
        const float depth_diff = final_depth - target_depth[pixel_idx];
        dd_depth = (depth_diff > 0.0f) ? loss_scalars.depth_scale
                 : (depth_diff < 0.0f) ? -loss_scalars.depth_scale : 0.0f;
    }

    const uint tile_x = gid.x / camera.tile_size;
    const uint tile_y = gid.y / camera.tile_size;
    const uint tile_idx = tile_y * camera.num_tiles_x + tile_x;
    const BwdTileRecord record = tile_records[tile_idx];

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

        const float r_r = final_color_r - running_s_r - contrib * g.color_r;
        const float r_g = final_color_g - running_s_g - contrib * g.color_g;
        const float r_b = final_color_b - running_s_b - contrib * g.color_b;

        const float transmittance = 1.0f - running_alpha;
        const float inv_one_minus_alpha = 1.0f / max(1.0f - alpha, 1e-6f);

        const float dl_dalpha_color = (transmittance * g.color_r - r_r * inv_one_minus_alpha) * dc_r
                                    + (transmittance * g.color_g - r_g * inv_one_minus_alpha) * dc_g
                                    + (transmittance * g.color_b - r_b * inv_one_minus_alpha) * dc_b;

        const float tail_alpha = final_alpha - running_alpha - contrib;
        const float tail_depth_num = final_depth * depth_denom - running_depth_num - contrib * g.depth;
        float dl_dalpha_depth = 0.0f;
        float dl_dalpha_alpha = 0.0f;
        if (dd_depth != 0.0f) {
            const float dnum_dalpha = transmittance * g.depth - tail_depth_num * inv_one_minus_alpha;
            const float dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
            const float ddepth_dalpha = (dnum_dalpha * depth_denom - final_depth * depth_denom * dalpha_dalpha)
                                      / (depth_denom * depth_denom);
            dl_dalpha_depth = dd_depth * ddepth_dalpha;
            if (loss_scalars.alpha_scale > 0.0f) {
                dl_dalpha_alpha = loss_scalars.alpha_scale * dalpha_dalpha;
            }
        } else if (loss_scalars.alpha_scale > 0.0f) {
            const float dalpha_dalpha = transmittance - tail_alpha * inv_one_minus_alpha;
            dl_dalpha_alpha = loss_scalars.alpha_scale * dalpha_dalpha;
        }

        const float dl_dalpha_total = dl_dalpha_color + dl_dalpha_depth + dl_dalpha_alpha;
        const float dl_dz_direct = dd_depth * contrib / depth_denom;

        const uint src_idx = g.source_idx;
        atomic_fetch_add_explicit(&grad_colors[src_idx * 3 + 0], contrib * dc_r, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_colors[src_idx * 3 + 1], contrib * dc_g, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_colors[src_idx * 3 + 2], contrib * dc_b, memory_order_relaxed);

        running_s_r += contrib * g.color_r;
        running_s_g += contrib * g.color_g;
        running_s_b += contrib * g.color_b;
        running_alpha += contrib;
        running_depth_num += contrib * g.depth;

        if ((1.0f - running_alpha) <= 1e-4f) {
            break;
        }

        if (alpha_raw <= 0.0f || alpha_raw >= 0.99f) {
            continue;
        }

        const float dl_dbase_alpha = dl_dalpha_total * kernel_val;

        const float dl_dkernel = dl_dalpha_total * g.opacity;
        const float dk_ddx = kernel_val * (-dx);
        const float dk_ddy = kernel_val * (-dy);

        const float dl_du = dl_dkernel * dk_ddx * (-1.0f / g.sigma_x);
        const float dl_dv = dl_dkernel * dk_ddy * (-1.0f / g.sigma_y);
        atomic_fetch_add_explicit(&grad_projected_positions[src_idx * 2 + 0], dl_du, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_projected_positions[src_idx * 2 + 1], dl_dv, memory_order_relaxed);

        float dl_dsigma_x = 0.0f;
        float dl_dsigma_y = 0.0f;
        if (abs(g.sigma_x) >= 0.5f) {
            dl_dsigma_x = dl_dkernel * dk_ddx * (-dx / g.sigma_x);
        }
        if (abs(g.sigma_y) >= 0.5f) {
            dl_dsigma_y = dl_dkernel * dk_ddy * (-dy / g.sigma_y);
        }

        const float inv_z = 1.0f / max(g.depth, 1e-6f);
        const float dl_dx_cam = dl_du * camera.fx * inv_z;
        const float dl_dy_cam = dl_dv * camera.fy * inv_z;
        const float dl_dz_projected = dl_du * (-(g.u - camera.cx) * inv_z)
                                    + dl_dv * (-(g.v - camera.cy) * inv_z);
        const float dl_dz_cam = dl_dz_direct + dl_dz_projected;

        const float dl_dworld_x = camera.rot00 * dl_dx_cam
                                + camera.rot10 * dl_dy_cam
                                + camera.rot20 * dl_dz_cam;
        const float dl_dworld_y = camera.rot01 * dl_dx_cam
                                + camera.rot11 * dl_dy_cam
                                + camera.rot21 * dl_dz_cam;
        const float dl_dworld_z = camera.rot02 * dl_dx_cam
                                + camera.rot12 * dl_dy_cam
                                + camera.rot22 * dl_dz_cam;

        atomic_fetch_add_explicit(&grad_positions[src_idx * 3 + 0], dl_dworld_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_positions[src_idx * 3 + 1], dl_dworld_y, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_positions[src_idx * 3 + 2], dl_dworld_z, memory_order_relaxed);

        if (abs(g.sigma_x) >= 0.5f) {
            atomic_fetch_add_explicit(&grad_log_scales[src_idx * 3 + 0], dl_dsigma_x * camera.fx * inv_z, memory_order_relaxed);
            const float raw_sx = g.raw_sigma_x;
            const float sx3 = g.scale_x;
            const float sz3 = g.scale_z;
            if (raw_sx > 1e-6f && sx3 > 1e-6f && sz3 > 1e-6f) {
                const float contrib_x = sx3 * camera.fx * inv_z;
                const float var_from_z = max(raw_sx * raw_sx - contrib_x * contrib_x, 0.0f);
                const float d_rawsx_d_sz = var_from_z / (raw_sx * sz3);
                atomic_fetch_add_explicit(&grad_log_scales[src_idx * 3 + 2], dl_dsigma_x * d_rawsx_d_sz * sz3, memory_order_relaxed);
            }
        }
        if (abs(g.sigma_y) >= 0.5f) {
            atomic_fetch_add_explicit(&grad_log_scales[src_idx * 3 + 1], dl_dsigma_y * camera.fy * inv_z, memory_order_relaxed);
            const float raw_sy = g.raw_sigma_y;
            const float sy3 = g.scale_y;
            const float sz3 = g.scale_z;
            if (raw_sy > 1e-6f && sy3 > 1e-6f && sz3 > 1e-6f) {
                const float contrib_y = sy3 * camera.fy * inv_z;
                const float var_from_z = max(raw_sy * raw_sy - contrib_y * contrib_y, 0.0f);
                const float d_rawsy_d_sz = var_from_z / (raw_sy * sz3);
                atomic_fetch_add_explicit(&grad_log_scales[src_idx * 3 + 2], dl_dsigma_y * d_rawsy_d_sz * sz3, memory_order_relaxed);
            }
        }

        const float sig_deriv = g.opacity * (1.0f - g.opacity);
        if (g.opacity >= 0.0f && g.opacity <= 1.0f) {
            atomic_fetch_add_explicit(&grad_opacity_logits[src_idx], dl_dbase_alpha * sig_deriv, memory_order_relaxed);
        }
    }
}
