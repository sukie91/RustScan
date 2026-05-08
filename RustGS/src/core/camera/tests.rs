use super::*;

#[test]
fn test_gaussian_camera_creation() {
    let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
    let extrinsics = SE3::identity();
    let camera = GaussianCamera::new(intrinsics, extrinsics);

    assert_eq!(camera.intrinsics.width, 1920);
}

#[test]
fn test_gaussian_camera_project() {
    let intrinsics = Intrinsics::from_focal(1000.0, 1920, 1080);
    let extrinsics = SE3::identity();
    let camera = GaussianCamera::new(intrinsics, extrinsics);

    // Point at z=1 should project near center
    let result = camera.project([0.0, 0.0, 1.0]);
    assert!(result.is_some());

    let [u, v] = result.unwrap();
    assert!((u - 960.0).abs() < 1.0);
    assert!((v - 540.0).abs() < 1.0);
}
