use super::{resize_depth, resize_rgb_u8_to_f32};

#[test]
fn resize_rgb_u8_to_f32_preserves_identity_dimensions() {
    let src = vec![0u8, 64, 255, 255, 128, 0];
    let resized = resize_rgb_u8_to_f32(&src, 2, 1, 2, 1);
    assert_eq!(resized.len(), 6);
    assert!((resized[0] - 0.0).abs() < 1e-6);
    assert!((resized[1] - (64.0 / 255.0)).abs() < 1e-6);
    assert!((resized[2] - 1.0).abs() < 1e-6);
}

#[test]
fn resize_depth_averages_only_valid_samples() {
    let src = vec![0.0, 1.0, 2.0, f32::NAN];
    let resized = resize_depth(&src, 2, 2, 1, 1);
    assert_eq!(resized.len(), 1);
    assert!((resized[0] - 1.5).abs() < 1e-6);
}
