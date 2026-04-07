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
