use super::se3_rotation_row_major;
use crate::SE3;

#[test]
fn se3_rotation_row_major_matches_expected_z_rotation() {
    let pose = SE3::from_axis_angle(&[0.0, 0.0, std::f32::consts::FRAC_PI_2], &[0.0, 0.0, 0.0]);
    let rotation = se3_rotation_row_major(&pose);
    let expected = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    for row in 0..3 {
        for col in 0..3 {
            assert!(
                (rotation[row][col] - expected[row][col]).abs() < 1e-6,
                "row={row} col={col} actual={} expected={}",
                rotation[row][col],
                expected[row][col]
            );
        }
    }
}
