use crate::SE3;

/// Convert an SE3 quaternion into a row-major 3x3 rotation matrix.
///
/// `SE3::rotation()` currently returns the underlying glam matrix in
/// column-vector order. That happens to work for a few call sites that rebuild
/// a glam matrix with `Mat3::from_cols`, but camera assembly needs an explicit
/// row-major matrix because `DiffCamera` stores `[R | t]` in row-major form.
pub(crate) fn se3_rotation_row_major(pose: &SE3) -> [[f32; 3]; 3] {
    let cols = glam::Mat3::from_quat(pose.quat()).to_cols_array();
    [
        [cols[0], cols[3], cols[6]],
        [cols[1], cols[4], cols[7]],
        [cols[2], cols[5], cols[8]],
    ]
}

#[cfg(test)]
mod tests {
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
}
