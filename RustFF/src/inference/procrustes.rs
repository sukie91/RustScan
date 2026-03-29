//! Weighted Procrustes alignment for pose estimation from pointmaps.
//!
//! Given predicted 3D pointmaps and confidence weights, estimate the
//! optimal rigid body transformation (rotation + translation).
//!
//! Reference: Python implementation in scripts/procrustes.py

use glam::{Mat3, Mat4, Vec3};

/// Estimate rigid body transformation from source to target point cloud.
///
/// Returns (R, t) where R is rotation [3x3] and t is translation [3].
pub fn weighted_procrustes(src: &[[f32; 3]], tgt: &[[f32; 3]], weights: &[f32]) -> (Mat3, Vec3) {
    assert_eq!(src.len(), tgt.len());
    assert_eq!(src.len(), weights.len());
    assert!(!src.is_empty());

    let w_sum: f32 = weights.iter().sum();
    let w_norm: Vec<f32> = weights.iter().map(|w| w / w_sum).collect();

    // Weighted centroids
    let mut src_centroid = [0.0f32; 3];
    let mut tgt_centroid = [0.0f32; 3];
    for i in 0..src.len() {
        for j in 0..3 {
            src_centroid[j] += w_norm[i] * src[i][j];
            tgt_centroid[j] += w_norm[i] * tgt[i][j];
        }
    }

    // Centered points
    let src_centered: Vec<[f32; 3]> = src
        .iter()
        .map(|p| {
            [
                p[0] - src_centroid[0],
                p[1] - src_centroid[1],
                p[2] - src_centroid[2],
            ]
        })
        .collect();
    let tgt_centered: Vec<[f32; 3]> = tgt
        .iter()
        .map(|p| {
            [
                p[0] - tgt_centroid[0],
                p[1] - tgt_centroid[1],
                p[2] - tgt_centroid[2],
            ]
        })
        .collect();

    // Weighted cross-covariance matrix H = src_centered^T * W * tgt_centered
    let mut h = [[0.0f32; 3]; 3];
    for i in 0..src.len() {
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += w_norm[i] * src_centered[i][r] * tgt_centered[i][c];
            }
        }
    }

    // SVD using simple 3x3 Jacobi SVD
    let (u, _s, vt) = svd_3x3(&h);

    // Correct reflection
    let det_uv = det_3x3(&matmul_3x3(&vt, &u));
    let sign = if det_uv < 0.0 { -1.0 } else { 1.0 };

    // R = V * diag(1, 1, sign) * U^T
    let mut r_matrix = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                let s = if k == 2 { sign } else { 1.0 };
                sum += vt[k][i] * s * u[k][j]; // V * diag * U^T
            }
            r_matrix[i][j] = sum;
        }
    }

    // Translation: t = tgt_centroid - R * src_centroid
    let r_src = matvec_3x3(&r_matrix, &src_centroid);
    let t = Vec3::new(
        tgt_centroid[0] - r_src[0],
        tgt_centroid[1] - r_src[1],
        tgt_centroid[2] - r_src[2],
    );

    let r = Mat3::from_cols(
        Vec3::new(r_matrix[0][0], r_matrix[1][0], r_matrix[2][0]),
        Vec3::new(r_matrix[0][1], r_matrix[1][1], r_matrix[2][1]),
        Vec3::new(r_matrix[0][2], r_matrix[1][2], r_matrix[2][2]),
    );

    (r, t)
}

/// Extract camera pose from pointmap via weighted Procrustes.
///
/// Given a predicted pointmap and confidence, estimate the camera-to-world pose.
pub fn pointmap_to_pose(points: &[[f32; 3]], confidence: &[f32]) -> Mat4 {
    // Filter by confidence
    let valid: Vec<(&[f32; 3], f32)> = points
        .iter()
        .zip(confidence.iter())
        .filter(|(_, &c)| c > 0.1)
        .map(|(p, &c)| (p, c))
        .collect();

    if valid.len() < 10 {
        return Mat4::IDENTITY;
    }

    let valid_points: Vec<[f32; 3]> = valid.iter().map(|(p, _)| **p).collect();
    let valid_conf: Vec<f32> = valid.iter().map(|(_, c)| *c).collect();

    // Weighted centroid
    let w_sum: f32 = valid_conf.iter().sum();
    let mut centroid = [0.0f32; 3];
    for (p, c) in valid.iter() {
        for j in 0..3 {
            centroid[j] += *c * p[j] / w_sum;
        }
    }

    // Compute covariance for principal axes
    let mut cov = [[0.0f32; 3]; 3];
    for (p, c) in valid.iter() {
        let d = [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]];
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += *c * d[r] * d[col];
            }
        }
    }

    // Eigendecomposition (power iteration for dominant eigenvector)
    let (_evals, evecs) = eigendecompose_3x3(&cov);

    // Build rotation from eigenvectors (column vectors)
    let r = Mat3::from_cols(
        Vec3::new(evecs[0][0], evecs[1][0], evecs[2][0]),
        Vec3::new(evecs[0][1], evecs[1][1], evecs[2][1]),
        Vec3::new(evecs[0][2], evecs[1][2], evecs[2][2]),
    );

    // Build 4x4 pose
    Mat4::from_cols(
        r.x_axis.extend(0.0),
        r.y_axis.extend(0.0),
        r.z_axis.extend(0.0),
        Vec3::new(centroid[0], centroid[1], centroid[2]).extend(1.0),
    )
}

// ============================================================
// Linear algebra helpers
// ============================================================

/// Simple 3x3 SVD using Jacobi iterations
fn svd_3x3(m: &[[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3], [[f32; 3]; 3]) {
    // For simplicity, use eigendecomposition of M^T*M and M*M^T
    let mtm = matmul_3x3t(m, m);
    let mmt = matmul_3xt3(m, m);

    let (s2, v) = eigendecompose_3x3(&mtm);
    let (_s1, u) = eigendecompose_3x3(&mmt);

    // Singular values
    let mut s = [0.0f32; 3];
    for i in 0..3 {
        s[i] = s2[i].max(0.0).sqrt();
    }

    (u, s, transpose_3x3(&v))
}

/// 3x3 eigendecomposition using Jacobi rotations
fn eigendecompose_3x3(m: &[[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut a = *m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..50 {
        // Find largest off-diagonal
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-7 {
            break;
        }

        // Jacobi rotation
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[p][p] - a[q][q]);
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to a
        let mut new_a = a;
        for i in 0..3 {
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
        }
        let mut new_a2 = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                new_a2[p][j] = c * new_a[p][j] + s * new_a[q][j];
                new_a2[q][j] = -s * new_a[p][j] + c * new_a[q][j];
                if i != p && i != q {
                    new_a2[i][j] = new_a[i][j];
                }
            }
        }
        a = new_a2;

        // Update eigenvectors
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    ([a[0][0], a[1][1], a[2][2]], v)
}

fn matmul_3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

fn matmul_3x3t(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[k][i] * b[k][j]; // a^T * b
            }
        }
    }
    r
}

fn matmul_3xt3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[j][k]; // a * b^T
            }
        }
    }
    r
}

fn transpose_3x3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = m[j][i];
        }
    }
    r
}

fn matvec_3x3(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    let mut r = [0.0f32; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

fn det_3x3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procrustes_identity() {
        let points = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let weights = vec![1.0; 4];

        // Same points -> R = I, t = 0
        let (r, t) = weighted_procrustes(&points, &points, &weights);

        // Check rotation is close to identity
        let identity = Mat3::IDENTITY;
        let diff = (r.x_axis - identity.x_axis).length()
            + (r.y_axis - identity.y_axis).length()
            + (r.z_axis - identity.z_axis).length();
        assert!(diff < 0.01, "Rotation should be identity, diff={diff}");

        // Check translation is close to zero
        assert!(t.length() < 0.01, "Translation should be zero, t={t:?}");
    }

    #[test]
    fn test_procrustes_translation() {
        let src = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let offset = [2.0, 3.0, 4.0];
        let tgt: Vec<[f32; 3]> = src
            .iter()
            .map(|p| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]])
            .collect();
        let weights = vec![1.0; 4];

        let (r, t) = weighted_procrustes(&src, &tgt, &weights);

        // R should be identity
        let identity = Mat3::IDENTITY;
        let diff = (r.x_axis - identity.x_axis).length()
            + (r.y_axis - identity.y_axis).length()
            + (r.z_axis - identity.z_axis).length();
        assert!(diff < 0.01, "Rotation should be identity, diff={diff}");

        // t should equal offset
        let t_err = (t - Vec3::from(offset)).length();
        assert!(
            t_err < 0.01,
            "Translation should be {offset:?}, err={t_err}"
        );
    }
}
