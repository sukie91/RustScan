use super::*;

#[cfg(feature = "gpu")]
#[test]
fn test_initialize_host_splats_scale_and_color() {
    let points = vec![
        ([0.0, 0.0, 0.0], Some([0.2, 0.3, 0.4])),
        ([1.0, 0.0, 0.0], Some([0.9, 0.1, 0.2])),
        ([0.0, 1.0, 0.0], Some([0.1, 0.8, 0.2])),
        ([100.0, 100.0, 100.0], Some([0.7, 0.6, 0.5])),
    ];

    let config = GaussianInitConfig::default();

    let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();
    assert_eq!(splats.len(), 4);

    let origin_scale = splats.scale(0)[0];
    assert!((origin_scale - 0.5).abs() < 1e-6);
    let near_scale = splats.scale(1)[0];
    assert!((near_scale - 0.6035534).abs() < 1e-5);
    let outlier_scale = splats.scale(3)[0];
    assert!((outlier_scale - 10.0).abs() < 1e-6);

    for idx in 0..splats.len() {
        let scale = splats.scale(idx);
        assert_eq!(scale[0], scale[1]);
        assert_eq!(scale[1], scale[2]);
        assert!((splats.opacity(idx) - 0.5).abs() < 1e-6);
    }

    assert!((splats.rgb_color(0)[0] - 0.2).abs() < 1e-6);
    assert!((splats.rgb_color(0)[1] - 0.3).abs() < 1e-6);
    assert!((splats.rgb_color(0)[2] - 0.4).abs() < 1e-6);
    assert!((splats.rgb_color(1)[0] - 0.9).abs() < 1e-6);
    assert!((splats.rgb_color(1)[1] - 0.1).abs() < 1e-6);
    assert!((splats.rgb_color(1)[2] - 0.2).abs() < 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
fn test_initialize_host_splats_defaults() {
    let points = vec![([0.0, 0.0, 1.0], None)];

    let config = GaussianInitConfig::default();
    let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();

    assert_eq!(splats.len(), 1);
    assert_eq!(splats.rotation(0), [1.0, 0.0, 0.0, 0.0]);
    assert!((splats.opacity(0) - config.opacity).abs() < 1e-6);
    assert_eq!(splats.rgb_color(0), config.default_color);
    assert_eq!(splats.scale(0)[0], 1.0);
    assert_eq!(splats.scale(0)[1], 1.0);
    assert_eq!(splats.scale(0)[2], 1.0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_initialize_host_splats_empty_points() {
    let points: Vec<([f32; 3], Option<[f32; 3]>)> = vec![];
    let config = GaussianInitConfig::default();
    let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();
    assert!(splats.is_empty());
}
