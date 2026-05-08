use super::HostSplats;
use crate::sh::rgb_to_sh0_value;

#[test]
fn validation_rejects_mismatched_component_lengths() {
    let invalid = HostSplats {
        positions: vec![0.0; 5],
        log_scales: vec![0.0; 6],
        rotations: vec![0.0; 8],
        opacity_logits: vec![0.0; 2],
        sh_coeffs: vec![0.0; 6],
        sh_degree: 0,
    };

    let err = invalid.validate().unwrap_err().to_string();
    assert!(err.contains("positions expected 6 values"));
}

#[test]
fn scene_extent_tracks_radius_from_positions() {
    let splats = HostSplats {
        positions: vec![-2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        log_scales: vec![0.0; 6],
        rotations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        opacity_logits: vec![0.0, 0.0],
        sh_coeffs: vec![rgb_to_sh0_value(0.2); 6],
        sh_degree: 0,
    };

    assert!((splats.scene_extent() - 2.0).abs() < 1e-6);
    assert_eq!(
        splats.positions_vec3(),
        vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    );
}
