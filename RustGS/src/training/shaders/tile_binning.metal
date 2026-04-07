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

kernel void tile_prefix_sum(
    device const uint* tile_counts [[buffer(0)]],
    device uint* tile_offsets [[buffer(1)]],
    device uint* total_out [[buffer(2)]],
    constant uint& tile_count [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (lid == 0) {
        uint running = 0;
        for (uint i = 0; i < tile_count; ++i) {
            tile_offsets[i] = running;
            running += tile_counts[i];
        }
        total_out[0] = running;
    }
}
