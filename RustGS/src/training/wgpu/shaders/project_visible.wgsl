#import helpers;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(2) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(3) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(4) var<storage, read_write> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(5) var<storage, read> uniforms: helpers::ProjectUniforms;

fn read_coeff(base_id: ptr<function, u32>) -> vec3<f32> {
    let idx = *base_id;
    *base_id += 3u;
    return vec3<f32>(sh_coeffs[idx], sh_coeffs[idx + 1u], sh_coeffs[idx + 2u]);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_gid = gid.x;
    if compact_gid >= uniforms.num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];
    let base = global_gid * 10u;
    let mean = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let scale = exp(vec3<f32>(
        transforms[base + 7u],
        transforms[base + 8u],
        transforms[base + 9u],
    ));

    let quat = normalize(vec4<f32>(
        transforms[base + 3u],
        transforms[base + 4u],
        transforms[base + 5u],
        transforms[base + 6u],
    ));
    let mean_c = (uniforms.viewmat * vec4<f32>(mean, 1.0)).xyz;
    var opacity = helpers::sigmoid(raw_opacities[global_gid]);

    let cov3d = helpers::calc_cov3d(scale, quat);
    var cov2d = helpers::calc_cov2d(
        cov3d,
        mean_c,
        uniforms.focal,
        uniforms.img_size,
        uniforms.pixel_center,
        uniforms.viewmat,
    );
    opacity *= helpers::compensate_cov2d(&cov2d);

    let conic = helpers::inverse2x2(cov2d);
    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;

    let dir = mean - uniforms.camera_position.xyz;
    let viewdir = dir * inverseSqrt(max(dot(dir, dir), 1e-12));
    let num_coeffs = helpers::num_sh_coeffs(uniforms.sh_degree);
    var coeff_base = global_gid * num_coeffs * 3u;

    var sh = helpers::ShCoeffs();
    sh.b0_c0 = read_coeff(&coeff_base);
    if uniforms.sh_degree >= 1u {
        sh.b1_c0 = read_coeff(&coeff_base);
        sh.b1_c1 = read_coeff(&coeff_base);
        sh.b1_c2 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 2u {
        sh.b2_c0 = read_coeff(&coeff_base);
        sh.b2_c1 = read_coeff(&coeff_base);
        sh.b2_c2 = read_coeff(&coeff_base);
        sh.b2_c3 = read_coeff(&coeff_base);
        sh.b2_c4 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 3u {
        sh.b3_c0 = read_coeff(&coeff_base);
        sh.b3_c1 = read_coeff(&coeff_base);
        sh.b3_c2 = read_coeff(&coeff_base);
        sh.b3_c3 = read_coeff(&coeff_base);
        sh.b3_c4 = read_coeff(&coeff_base);
        sh.b3_c5 = read_coeff(&coeff_base);
        sh.b3_c6 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 4u {
        sh.b4_c0 = read_coeff(&coeff_base);
        sh.b4_c1 = read_coeff(&coeff_base);
        sh.b4_c2 = read_coeff(&coeff_base);
        sh.b4_c3 = read_coeff(&coeff_base);
        sh.b4_c4 = read_coeff(&coeff_base);
        sh.b4_c5 = read_coeff(&coeff_base);
        sh.b4_c6 = read_coeff(&coeff_base);
        sh.b4_c7 = read_coeff(&coeff_base);
        sh.b4_c8 = read_coeff(&coeff_base);
    }

    let color = helpers::sh_coeffs_to_color(uniforms.sh_degree, viewdir, sh) + vec3<f32>(0.5);
    projected[compact_gid] = helpers::create_projected_splat(
        mean2d,
        vec3<f32>(conic[0][0], conic[0][1], conic[1][1]),
        vec4<f32>(color, opacity),
    );
}
