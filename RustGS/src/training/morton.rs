//! Morton code (Z-order curve) utilities for spatial sorting.
//!
//! Morton codes map 3D coordinates to 1D values while preserving spatial locality.
//! Sorting Gaussians by Morton code improves memory coherence during rendering.

/// Default number of bits per dimension for Morton encoding.
pub const DEFAULT_MORTON_BITS: usize = 21;

/// Compute Morton code from normalized integer coordinates.
///
/// Interleaves bits from x, y, z coordinates to create a Z-order curve index.
/// Each coordinate is limited to `bits` bits, producing a 3*bits bit code.
///
/// # Arguments
/// * `x` - Normalized x coordinate (0 to 2^bits - 1)
/// * `y` - Normalized y coordinate (0 to 2^bits - 1)
/// * `z` - Normalized z coordinate (0 to 2^bits - 1)
/// * `bits` - Number of bits per dimension (default 21)
///
/// # Example
/// ```
/// let code = morton_encode_3d(5, 10, 15, 21);
/// ```
pub fn morton_encode_3d(x: u32, y: u32, z: u32, bits: usize) -> u64 {
    spread_bits_3d(x, bits) | (spread_bits_3d(y, bits) << 1) | (spread_bits_3d(z, bits) << 2)
}

/// Spread bits of a value for 3D Morton encoding.
///
/// Takes the lower `bits` bits of `v` and spreads them out,
/// placing each bit at positions 0, 3, 6, 9, ...
fn spread_bits_3d(mut v: u32, bits: usize) -> u64 {
    // Mask to keep only the relevant bits
    let mask = if bits >= 32 {
        !0u32
    } else {
        (1u32 << bits) - 1
    };
    v &= mask;
    let mut result = v as u64;

    // Standard bit-spreading for Morton codes:
    // Input:  bits at positions 0,1,2,...,20
    // Output: bits at positions 0,3,6,...,60

    // Step 1: spread to positions with spacing 2
    result = (result | (result << 16)) & 0x00_00_FF_FF;

    // Step 2: further spread
    result = (result | (result << 8)) & 0x00_FF_00_FF;

    // Step 3
    result = (result | (result << 4)) & 0x0F_0F_0F_0F;

    // Step 4
    result = (result | (result << 2)) & 0x33_33_33_33;

    // Step 5
    result = (result | (result << 1)) & 0x55_55_55_55;

    result
}

/// Compute Morton codes for a set of 3D points.
///
/// Points are first normalized to [0, 2^bits - 1] range based on the provided bounds.
///
/// # Arguments
/// * `positions` - Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
/// * `bounds_min` - Minimum corner of bounding box [min_x, min_y, min_z]
/// * `bounds_max` - Maximum corner of bounding box [max_x, max_y, max_z]
/// * `bits` - Number of bits per dimension for Morton encoding
///
/// # Returns
/// Vector of Morton codes, one per point.
pub fn compute_morton_codes(
    positions: &[f32],
    bounds_min: &[f32; 3],
    bounds_max: &[f32; 3],
    bits: usize,
) -> Vec<u64> {
    if positions.is_empty() {
        return Vec::new();
    }

    let n = positions.len() / 3;
    let mut codes = Vec::with_capacity(n);

    // Compute scale to normalize to [0, 2^bits - 1]
    let max_val = (1u32 << bits).saturating_sub(1);
    let scale = [
        if bounds_max[0] > bounds_min[0] {
            max_val as f32 / (bounds_max[0] - bounds_min[0])
        } else {
            0.0
        },
        if bounds_max[1] > bounds_min[1] {
            max_val as f32 / (bounds_max[1] - bounds_min[1])
        } else {
            0.0
        },
        if bounds_max[2] > bounds_min[2] {
            max_val as f32 / (bounds_max[2] - bounds_min[2])
        } else {
            0.0
        },
    ];

    for i in 0..n {
        let base = i * 3;
        let x = ((positions[base] - bounds_min[0]) * scale[0]).clamp(0.0, max_val as f32) as u32;
        let y =
            ((positions[base + 1] - bounds_min[1]) * scale[1]).clamp(0.0, max_val as f32) as u32;
        let z =
            ((positions[base + 2] - bounds_min[2]) * scale[2]).clamp(0.0, max_val as f32) as u32;
        codes.push(morton_encode_3d(x, y, z, bits));
    }

    codes
}

/// Compute bounding box from positions.
///
/// # Arguments
/// * `positions` - Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
///
/// # Returns
/// Tuple of (min, max) bounds.
pub fn compute_bounding_box(positions: &[f32]) -> ([f32; 3], [f32; 3]) {
    if positions.is_empty() {
        return ([0.0; 3], [1.0; 3]);
    }

    let n = positions.len() / 3;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for i in 0..n {
        let base = i * 3;
        for j in 0..3 {
            let v = positions[base + j];
            if v < min[j] {
                min[j] = v;
            }
            if v > max[j] {
                max[j] = v;
            }
        }
    }

    // Handle degenerate bounds
    for j in 0..3 {
        if min[j] == max[j] {
            min[j] -= 0.5;
            max[j] += 0.5;
        }
    }

    (min, max)
}

/// Sort Gaussians by Morton code for better memory coherence.
///
/// Returns permutation indices that sort the Gaussians by their Morton codes.
///
/// # Arguments
/// * `positions` - Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
/// * `bits` - Number of bits per dimension for Morton encoding
///
/// # Returns
/// Permutation vector where result[i] is the index of the i-th Gaussian after sorting.
pub fn morton_sort_permutation(positions: &[f32], bits: usize) -> Vec<usize> {
    let n = positions.len() / 3;
    if n == 0 {
        return Vec::new();
    }

    // Compute bounding box
    let (bounds_min, bounds_max) = compute_bounding_box(positions);

    // Compute Morton codes
    let codes = compute_morton_codes(positions, &bounds_min, &bounds_max, bits);

    // Create index-value pairs and sort by Morton code
    let mut indexed: Vec<(usize, u64)> = codes.into_iter().enumerate().collect();
    indexed.sort_by_key(|&(_, code)| code);

    // Extract permutation
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Apply permutation to a flat array of 3D vectors.
///
/// # Arguments
/// * `data` - Flat array [x0, y0, z0, x1, y1, z1, ...]
/// * `permutation` - Permutation indices
///
/// # Returns
/// Permuted flat array.
pub fn permute_vec3(data: &[f32], permutation: &[usize]) -> Vec<f32> {
    let n = permutation.len();
    let mut result = Vec::with_capacity(n * 3);

    for &i in permutation {
        let base = i * 3;
        result.push(data[base]);
        result.push(data[base + 1]);
        result.push(data[base + 2]);
    }

    result
}

/// Apply permutation to a flat array of 4D vectors (quaternions).
///
/// # Arguments
/// * `data` - Flat array [w0, x0, y0, z0, w1, x1, y1, z1, ...]
/// * `permutation` - Permutation indices
///
/// # Returns
/// Permuted flat array.
pub fn permute_vec4(data: &[f32], permutation: &[usize]) -> Vec<f32> {
    let n = permutation.len();
    let mut result = Vec::with_capacity(n * 4);

    for &i in permutation {
        let base = i * 4;
        result.push(data[base]);
        result.push(data[base + 1]);
        result.push(data[base + 2]);
        result.push(data[base + 3]);
    }

    result
}

/// Apply permutation to a scalar array.
///
/// # Arguments
/// * `data` - Scalar array
/// * `permutation` - Permutation indices
///
/// # Returns
/// Permuted scalar array.
pub fn permute_scalar(data: &[f32], permutation: &[usize]) -> Vec<f32> {
    permutation.iter().map(|&i| data[i]).collect()
}

/// Apply permutation to a 2D array (rows are permuted).
///
/// # Arguments
/// * `data` - Flat array representing row-major 2D array
/// * `row_width` - Number of elements per row
/// * `permutation` - Permutation indices for rows
///
/// # Returns
/// Permuted flat array.
pub fn permute_rows(data: &[f32], row_width: usize, permutation: &[usize]) -> Vec<f32> {
    let n = permutation.len();
    let mut result = Vec::with_capacity(n * row_width);

    for &i in permutation {
        let base = i * row_width;
        for j in 0..row_width {
            result.push(data[base + j]);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_encode_simple() {
        // Test that adjacent coordinates produce adjacent Morton codes
        let code_000 = morton_encode_3d(0, 0, 0, 10);
        let code_001 = morton_encode_3d(1, 0, 0, 10);
        let code_010 = morton_encode_3d(0, 1, 0, 10);
        let _code_100 = morton_encode_3d(0, 0, 1, 10);

        // Morton codes should preserve Z-order spatial locality
        assert!(code_000 < code_001);
        assert!(code_001 < code_010);
    }

    #[test]
    fn test_morton_sort_basic() {
        // Create points in a known order
        let positions = vec![
            1.0, 1.0, 1.0, // point 0
            0.0, 0.0, 0.0, // point 1 (should be first after sort)
            2.0, 2.0, 2.0, // point 2 (should be last after sort)
        ];

        let perm = morton_sort_permutation(&positions, 10);

        assert_eq!(perm.len(), 3);
        // Point 1 (origin) should come first
        assert_eq!(perm[0], 1);
        // Point 0 (center) should be second
        assert_eq!(perm[1], 0);
        // Point 2 (far corner) should be last
        assert_eq!(perm[2], 2);
    }

    #[test]
    fn test_permute_vec3() {
        let data = vec![
            1.0, 2.0, 3.0, // point 0
            4.0, 5.0, 6.0, // point 1
            7.0, 8.0, 9.0, // point 2
        ];
        let perm = vec![2, 0, 1];

        let result = permute_vec3(&data, &perm);

        assert_eq!(result, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_permute_scalar() {
        let data = vec![10.0, 20.0, 30.0];
        let perm = vec![2, 1, 0];

        let result = permute_scalar(&data, &perm);

        assert_eq!(result, vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_compute_bounding_box() {
        let positions = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0];

        let (min, max) = compute_bounding_box(&positions);

        assert_eq!(min, [-1.0, -2.0, -3.0]);
        assert_eq!(max, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_empty_positions() {
        let positions: Vec<f32> = vec![];
        let perm = morton_sort_permutation(&positions, 10);
        assert!(perm.is_empty());
    }
}
