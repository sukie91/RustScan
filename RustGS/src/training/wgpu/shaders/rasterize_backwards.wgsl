#import helpers;

struct RasterizeBwdUniforms {
    tile_bounds: vec2<u32>,
    img_size: vec2<u32>,
    background: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(3) var<storage, read> out_img: array<f32>;
@group(0) @binding(4) var<storage, read> v_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_splats: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> uniforms: RasterizeBwdUniforms;

var<workgroup> range_uniform: vec2<u32>;
var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;
var<workgroup> local_gid: array<u32, helpers::TILE_SIZE>;

@compute @workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let tile_id = wg_id.x;
    let num_tiles = uniforms.tile_bounds.x * uniforms.tile_bounds.y;
    if tile_id >= num_tiles {
        return;
    }

    let tile_x = tile_id % uniforms.tile_bounds.x;
    let tile_y = tile_id / uniforms.tile_bounds.x;
    let pixel = vec2<u32>(
        tile_x * helpers::TILE_WIDTH + (local_idx % helpers::TILE_WIDTH),
        tile_y * helpers::TILE_WIDTH + (local_idx / helpers::TILE_WIDTH),
    );
    let inside = pixel.x < uniforms.img_size.x && pixel.y < uniforms.img_size.y;
    let pixel_coord = vec2<f32>(pixel) + vec2<f32>(0.5);

    if local_idx == 0u {
        range_uniform = vec2<u32>(
            tile_offsets[tile_id * 2u],
            tile_offsets[tile_id * 2u + 1u],
        );
    }
    workgroupBarrier();

    var t_accum = 1.0;
    var rgb_accum = vec3<f32>(0.0);
    var done = !inside;

    var rgb_pixel_final = vec3<f32>(0.0);
    var v_rgb = vec3<f32>(0.0);
    var v_alpha_out = 0.0;

    if inside {
        let pixel_id = pixel.x + pixel.y * uniforms.img_size.x;
        let base = pixel_id * 4u;
        let final_rgba = vec4<f32>(
            out_img[base],
            out_img[base + 1u],
            out_img[base + 2u],
            out_img[base + 3u],
        );
        let v_out = vec4<f32>(
            v_output[base],
            v_output[base + 1u],
            v_output[base + 2u],
            v_output[base + 3u],
        );
        let t_final = 1.0 - final_rgba.a;
        rgb_pixel_final = final_rgba.rgb - t_final * uniforms.background.rgb;
        v_rgb = v_out.rgb;
        v_alpha_out = (v_out.a - dot(uniforms.background.rgb, v_rgb)) * t_final;
    }

    let range = range_uniform;
    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        if local_idx < remaining {
            let isect_id = batch_start + local_idx;
            let compact_gid = compact_gid_from_isect[isect_id];
            local_batch[local_idx] = projected[compact_gid];
            local_gid[local_idx] = compact_gid;
        }
        workgroupBarrier();

        for (var t = 0u; !done && t < remaining; t++) {
            let splat = local_batch[t];
            let compact_gid = local_gid[t];
            let xy = vec2<f32>(splat.xy_x, splat.xy_y);
            let conic = vec3<f32>(splat.conic_x, splat.conic_y, splat.conic_z);
            let color = vec4<f32>(splat.color_r, splat.color_g, splat.color_b, splat.color_a);
            let clamped_rgb = max(color.rgb, vec3<f32>(0.0));

            let delta = xy - pixel_coord;
            let sigma = 0.5 * (
                conic.x * delta.x * delta.x +
                2.0 * conic.y * delta.x * delta.y +
                conic.z * delta.y * delta.y
            );
            let gaussian = exp(-sigma);
            let alpha = min(0.999, color.a * gaussian);

            if sigma < 0.0 || alpha < (1.0 / 255.0) {
                continue;
            }

            let vis = alpha * t_accum;
            let next_rgb_accum = rgb_accum + vis * clamped_rgb;
            let ra = 1.0 / (1.0 - alpha);

            let v_rgb_local = select(vec3<f32>(0.0), vis * v_rgb, color.rgb >= vec3<f32>(0.0));
            let v_alpha = dot(
                t_accum * clamped_rgb + (next_rgb_accum - rgb_pixel_final) * ra,
                v_rgb,
            ) + v_alpha_out * ra;
            let v_sigma = -alpha * v_alpha;
            let v_xy = v_sigma * vec2<f32>(
                conic.x * delta.x + conic.y * delta.y,
                conic.y * delta.x + conic.z * delta.y,
            );
            let v_conic = vec3<f32>(
                0.5 * v_sigma * delta.x * delta.x,
                v_sigma * delta.x * delta.y,
                0.5 * v_sigma * delta.y * delta.y,
            );
            let v_color_a = gaussian * v_alpha;

            let base = compact_gid * 10u;

            // Atomic add for v_xy.x
            var old_0 = atomicLoad(&v_splats[base + 0u]);
            loop {
                let new_0 = bitcast<u32>(bitcast<f32>(old_0) + v_xy.x);
                let cas_0 = atomicCompareExchangeWeak(&v_splats[base + 0u], old_0, new_0);
                if cas_0.exchanged { break; }
                old_0 = cas_0.old_value;
            }

            // Atomic add for v_xy.y
            var old_1 = atomicLoad(&v_splats[base + 1u]);
            loop {
                let new_1 = bitcast<u32>(bitcast<f32>(old_1) + v_xy.y);
                let cas_1 = atomicCompareExchangeWeak(&v_splats[base + 1u], old_1, new_1);
                if cas_1.exchanged { break; }
                old_1 = cas_1.old_value;
            }

            // Atomic add for v_conic.x
            var old_2 = atomicLoad(&v_splats[base + 2u]);
            loop {
                let new_2 = bitcast<u32>(bitcast<f32>(old_2) + v_conic.x);
                let cas_2 = atomicCompareExchangeWeak(&v_splats[base + 2u], old_2, new_2);
                if cas_2.exchanged { break; }
                old_2 = cas_2.old_value;
            }

            // Atomic add for v_conic.y
            var old_3 = atomicLoad(&v_splats[base + 3u]);
            loop {
                let new_3 = bitcast<u32>(bitcast<f32>(old_3) + v_conic.y);
                let cas_3 = atomicCompareExchangeWeak(&v_splats[base + 3u], old_3, new_3);
                if cas_3.exchanged { break; }
                old_3 = cas_3.old_value;
            }

            // Atomic add for v_conic.z
            var old_4 = atomicLoad(&v_splats[base + 4u]);
            loop {
                let new_4 = bitcast<u32>(bitcast<f32>(old_4) + v_conic.z);
                let cas_4 = atomicCompareExchangeWeak(&v_splats[base + 4u], old_4, new_4);
                if cas_4.exchanged { break; }
                old_4 = cas_4.old_value;
            }

            // Atomic add for v_rgb_local.x
            var old_5 = atomicLoad(&v_splats[base + 5u]);
            loop {
                let new_5 = bitcast<u32>(bitcast<f32>(old_5) + v_rgb_local.x);
                let cas_5 = atomicCompareExchangeWeak(&v_splats[base + 5u], old_5, new_5);
                if cas_5.exchanged { break; }
                old_5 = cas_5.old_value;
            }

            // Atomic add for v_rgb_local.y
            var old_6 = atomicLoad(&v_splats[base + 6u]);
            loop {
                let new_6 = bitcast<u32>(bitcast<f32>(old_6) + v_rgb_local.y);
                let cas_6 = atomicCompareExchangeWeak(&v_splats[base + 6u], old_6, new_6);
                if cas_6.exchanged { break; }
                old_6 = cas_6.old_value;
            }

            // Atomic add for v_rgb_local.z
            var old_7 = atomicLoad(&v_splats[base + 7u]);
            loop {
                let new_7 = bitcast<u32>(bitcast<f32>(old_7) + v_rgb_local.z);
                let cas_7 = atomicCompareExchangeWeak(&v_splats[base + 7u], old_7, new_7);
                if cas_7.exchanged { break; }
                old_7 = cas_7.old_value;
            }

            // Atomic add for v_color_a
            var old_8 = atomicLoad(&v_splats[base + 8u]);
            loop {
                let new_8 = bitcast<u32>(bitcast<f32>(old_8) + v_color_a);
                let cas_8 = atomicCompareExchangeWeak(&v_splats[base + 8u], old_8, new_8);
                if cas_8.exchanged { break; }
                old_8 = cas_8.old_value;
            }

            rgb_accum = next_rgb_accum;
            t_accum *= 1.0 - alpha;
            if t_accum < 1e-4 {
                done = true;
            }
        }

        workgroupBarrier();
    }
}
