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

kernel void projected_grad_magnitudes(
    device const float* grad_projected_positions [[buffer(0)]],
    device float* out_magnitudes [[buffer(1)]],
    constant uint& gaussian_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= gaussian_count) {
        return;
    }

    const uint p = gid * 2;
    const float grad_u = grad_projected_positions[p + 0];
    const float grad_v = grad_projected_positions[p + 1];
    out_magnitudes[gid] = sqrt(grad_u * grad_u + grad_v * grad_v);
}
