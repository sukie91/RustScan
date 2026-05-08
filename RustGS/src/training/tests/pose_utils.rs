#[cfg(test)]
use crate::SE3;

/// Convert an SE3 quaternion into a row-major 3x3 rotation matrix.
///
/// `SE3::rotation()` currently returns the underlying glam matrix in
/// column-vector order. That happens to work for a few call sites that rebuild
/// a glam matrix with `Mat3::from_cols`, but camera assembly needs an explicit
/// row-major matrix because `DiffCamera` stores `[R | t]` in row-major form.
#[cfg(test)]
pub(crate) fn se3_rotation_row_major(pose: &SE3) -> [[f32; 3]; 3] {
    let cols = glam::Mat3::from_quat(pose.quat()).to_cols_array();
    [
        [cols[0], cols[3], cols[6]],
        [cols[1], cols[4], cols[7]],
        [cols[2], cols[5], cols[8]],
    ]
}

#[cfg(test)]
mod tests;
