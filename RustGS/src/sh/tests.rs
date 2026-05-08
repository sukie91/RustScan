use super::{evaluate_sh_rgb, rgb_to_sh0_value};

#[test]
fn evaluate_degree_zero_matches_sh0_conversion() {
    let rgb = [0.25, 0.5, 0.75];
    let coeffs: Vec<f32> = rgb.into_iter().map(rgb_to_sh0_value).collect();

    assert_eq!(evaluate_sh_rgb(&coeffs, 0, [0.0, 0.0, 1.0]), rgb);
}
