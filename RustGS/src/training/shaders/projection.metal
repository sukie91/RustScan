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

    const float cw00 = r00*r00*sxx + r01*r01*syy + r02*r02*szz;
    const float cw01 = r00*r10*sxx + r01*r11*syy + r02*r12*szz;
    const float cw02 = r00*r20*sxx + r01*r21*syy + r02*r22*szz;
    const float cw11 = r10*r10*sxx + r11*r11*syy + r12*r12*szz;
    const float cw12 = r10*r20*sxx + r11*r21*syy + r12*r22*szz;
    const float cw22 = r20*r20*sxx + r21*r21*syy + r22*r22*szz;

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

    const float lowpass_filter = 0.3f;
    const float lowpass_var = lowpass_filter * lowpass_filter;

    rec.raw_sigma_x = sqrt(max(cov_x + lowpass_var, 1e-6f));
    rec.raw_sigma_y = sqrt(max(cov_y + lowpass_var, 1e-6f));

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
