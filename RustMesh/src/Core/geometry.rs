//! # Geometry
//!
//! Geometric utilities for mesh operations.

pub type Point = glam::Vec3;
pub type Vector = glam::Vec3;
pub type Normal = glam::Vec3;

/// Calculate the centroid of a triangle
#[inline]
pub fn triangle_centroid(p0: Point, p1: Point, p2: Point) -> Point {
    (p0 + p1 + p2) / 3.0
}

/// Calculate the area of a triangle using cross product
#[inline]
pub fn triangle_area(p0: Point, p1: Point, p2: Point) -> f32 {
    (p1 - p0).cross(p2 - p0).length() * 0.5
}

/// Calculate the normal of a triangle
#[inline]
pub fn triangle_normal(p0: Point, p1: Point, p2: Point) -> Vector {
    (p1 - p0).cross(p2 - p0).normalize()
}

/// Check if a 2D point is inside a triangle (barycentric technique)
#[inline]
pub fn point_in_triangle_2d(p: Point, a: Point, b: Point, c: Point) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}

/// Calculate the bounding box of a point set
#[inline]
pub fn bounding_box(points: &[Point]) -> (Point, Point) {
    if points.is_empty() {
        return (Point::ZERO, Point::ZERO);
    }

    let mut min = points[0];
    let mut max = points[0];

    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }

    (min, max)
}

/// Calculate the center of mass of a point set
#[inline]
pub fn center_of_mass(points: &[Point]) -> Point {
    if points.is_empty() {
        return Point::ZERO;
    }

    let sum: Point = points.iter().fold(Point::ZERO, |acc, &p| acc + p);
    sum / points.len() as f32
}

/// Calculate the squared distance between two points
#[inline]
pub fn squared_distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).length_squared()
}

/// Calculate the distance between two points
#[inline]
pub fn distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).length()
}

/// Normalize a vector
#[inline]
pub fn normalize(v: Vector) -> Vector {
    v.normalize()
}

/// Calculate the dot product of two vectors
#[inline]
pub fn dot(v0: Vector, v1: Vector) -> f32 {
    v0.dot(v1)
}

/// Calculate the cross product of two vectors
#[inline]
pub fn cross(v0: Vector, v1: Vector) -> Vector {
    v0.cross(v1)
}

/// Calculate the length of a vector
#[inline]
pub fn length(v: Vector) -> f32 {
    v.length()
}

/// Calculate the squared length of a vector
#[inline]
pub fn squared_length(v: Vector) -> f32 {
    v.length_squared()
}

// ============================================================================
// Triangle Angle Functions (E2-S1)
// ============================================================================

/// Calculate the angle at vertex p0 in a triangle
///
/// Returns angle in radians
#[inline]
pub fn triangle_angle_at(p0: Point, p1: Point, p2: Point) -> f32 {
    let v01 = p1 - p0;
    let v02 = p2 - p0;

    let len01 = v01.length();
    let len02 = v02.length();

    if len01 < 1e-10 || len02 < 1e-10 {
        return 0.0;
    }

    let cos_angle = (v01.dot(v02) / (len01 * len02)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Calculate all three angles of a triangle
///
/// Returns (angle_at_p0, angle_at_p1, angle_at_p2) in radians
#[inline]
pub fn triangle_angles(p0: Point, p1: Point, p2: Point) -> (f32, f32, f32) {
    let a0 = triangle_angle_at(p0, p1, p2);
    let a1 = triangle_angle_at(p1, p2, p0);
    let a2 = triangle_angle_at(p2, p0, p1);
    (a0, a1, a2)
}

/// Calculate the cotangent of an angle (used in Laplacian smoothing)
///
/// cot(θ) = cos(θ) / sin(θ) = adjacent / opposite
#[inline]
pub fn cotangent(angle: f32) -> f32 {
    let sin_val = angle.sin();
    if sin_val.abs() < 1e-10 {
        return 0.0; // Avoid division by zero for very small angles
    }
    angle.cos() / sin_val
}

/// Calculate the cotangent weight for an edge in a mesh
///
/// For an edge between vertices v0 and v1, the cotangent weight is:
/// w = (cot(α) + cot(β)) / 2
/// where α and β are the angles opposite to the edge in the two adjacent triangles
///
/// This requires the positions of the two opposite vertices in the adjacent triangles
#[inline]
pub fn cotangent_weight(
    v0: Point,
    v1: Point,
    opposite0: Point, // opposite vertex in first triangle (v0, v1, opposite0)
    opposite1: Point, // opposite vertex in second triangle (v1, v0, opposite1)
) -> f32 {
    // In triangle (v0, v1, opposite0), the angle at opposite0 is α
    let alpha = triangle_angle_at(opposite0, v0, v1);
    let cot_alpha = cotangent(alpha);

    // In triangle (v1, v0, opposite1), the angle at opposite1 is β
    let beta = triangle_angle_at(opposite1, v1, v0);
    let cot_beta = cotangent(beta);

    (cot_alpha + cot_beta) * 0.5
}

/// Calculate the quality of a triangle (aspect ratio)
///
/// Returns a value between 0 (degenerate) and 1 (equilateral)
#[inline]
pub fn triangle_quality(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let s = (a + b + c) * 0.5; // semi-perimeter
    let area = triangle_area(p0, p1, p2);

    if s < 1e-10 {
        return 0.0;
    }

    // Quality = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)
    // For equilateral triangle, this equals 1
    let sum_sq = a * a + b * b + c * c;
    if sum_sq < 1e-10 {
        return 0.0;
    }

    (4.0 * (3.0_f32).sqrt() * area) / sum_sq
}

/// Calculate the circumradius of a triangle
#[inline]
pub fn triangle_circumradius(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let area = triangle_area(p0, p1, p2);
    if area < 1e-10 {
        return 0.0;
    }

    // R = (a * b * c) / (4 * area)
    (a * b * c) / (4.0 * area)
}

/// Calculate the inradius of a triangle
#[inline]
pub fn triangle_inradius(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let s = (a + b + c) * 0.5; // semi-perimeter
    let area = triangle_area(p0, p1, p2);

    if s < 1e-10 {
        return 0.0;
    }

    // r = area / s
    area / s
}

/// Calculate the dihedral angle between two triangles sharing an edge
///
/// The edge is defined by v0-v1, with opposite vertices o0 and o1
/// Returns angle in radians
#[inline]
pub fn dihedral_angle(
    v0: Point,
    v1: Point,
    o0: Point, // opposite vertex in first triangle
    o1: Point, // opposite vertex in second triangle
) -> f32 {
    let n0 = triangle_normal(o0, v0, v1);
    let n1 = triangle_normal(o1, v1, v0);

    let cos_angle = n0.dot(n1).clamp(-1.0, 1.0);
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.0, 1.0, 0.0);

        // Right triangle with legs of length 1
        assert!((triangle_area(p0, p1, p2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_normal() {
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.0, 1.0, 0.0);

        let normal = triangle_normal(p0, p1, p2);
        assert!((normal.length() - 1.0).abs() < 1e-6);
        // Should point in +Z direction
        assert!(normal.z > 0.0);
    }

    #[test]
    fn test_bounding_box() {
        let points = [
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(1.0, 2.0, 3.0),
            glam::vec3(-1.0, 5.0, -2.0),
        ];

        let (min, max) = bounding_box(&points);

        assert_eq!(min, glam::vec3(-1.0, 0.0, -2.0));
        assert_eq!(max, glam::vec3(1.0, 5.0, 3.0));
    }

    #[test]
    fn test_triangle_angles() {
        // Equilateral triangle - all angles should be 60 degrees (π/3)
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.5, (3.0_f32).sqrt() / 2.0, 0.0);

        let (a0, a1, a2) = triangle_angles(p0, p1, p2);

        let expected = std::f32::consts::PI / 3.0;
        assert!((a0 - expected).abs() < 0.01);
        assert!((a1 - expected).abs() < 0.01);
        assert!((a2 - expected).abs() < 0.01);

        // Angles should sum to π
        assert!((a0 + a1 + a2 - std::f32::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_triangle_angle_right() {
        // Right triangle with 45-45-90 angles
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.0, 1.0, 0.0);

        let (a0, a1, a2) = triangle_angles(p0, p1, p2);

        // Angle at p0 should be 90 degrees (π/2)
        assert!((a0 - std::f32::consts::PI / 2.0).abs() < 0.01);

        // Angles at p1 and p2 should be 45 degrees (π/4)
        assert!((a1 - std::f32::consts::PI / 4.0).abs() < 0.01);
        assert!((a2 - std::f32::consts::PI / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_cotangent() {
        // cot(π/4) = 1
        assert!((cotangent(std::f32::consts::PI / 4.0) - 1.0).abs() < 0.01);

        // cot(π/3) = 1/sqrt(3)
        let expected = 1.0 / (3.0_f32).sqrt();
        assert!((cotangent(std::f32::consts::PI / 3.0) - expected).abs() < 0.01);
    }

    #[test]
    fn test_cotangent_weight() {
        // Equilateral triangle: cot(60°) = 1/sqrt(3)
        let v0 = glam::vec3(0.0, 0.0, 0.0);
        let v1 = glam::vec3(1.0, 0.0, 0.0);
        let o0 = glam::vec3(0.5, (3.0_f32).sqrt() / 2.0, 0.0);
        let o1 = glam::vec3(0.5, -(3.0_f32).sqrt() / 2.0, 0.0);

        let weight = cotangent_weight(v0, v1, o0, o1);

        // For equilateral triangles, cot(60°) ≈ 0.577
        let expected_cot = 1.0 / (3.0_f32).sqrt();
        let expected = expected_cot; // (cot + cot) / 2 = cot for symmetric case

        assert!((weight - expected).abs() < 0.1);
    }

    #[test]
    fn test_triangle_quality() {
        // Equilateral triangle should have quality close to 1
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.5, (3.0_f32).sqrt() / 2.0, 0.0);

        let quality = triangle_quality(p0, p1, p2);
        assert!(quality > 0.99);

        // Right triangle should have lower quality
        let r0 = glam::vec3(0.0, 0.0, 0.0);
        let r1 = glam::vec3(1.0, 0.0, 0.0);
        let r2 = glam::vec3(0.0, 1.0, 0.0);

        let right_quality = triangle_quality(r0, r1, r2);
        assert!(right_quality < quality);
        assert!(right_quality > 0.8); // Still reasonable quality
    }

    #[test]
    fn test_dihedral_angle() {
        // Two coplanar triangles
        let v0 = glam::vec3(0.0, 0.0, 0.0);
        let v1 = glam::vec3(1.0, 0.0, 0.0);
        let o0 = glam::vec3(0.5, 1.0, 0.0);
        let o1 = glam::vec3(0.5, -1.0, 0.0);

        let angle = dihedral_angle(v0, v1, o0, o1);

        // Coplanar triangles should have dihedral angle of 0
        assert!(angle.abs() < 0.01);

        // Folded triangles (90 degrees)
        let f0 = glam::vec3(0.0, 0.0, 0.0);
        let f1 = glam::vec3(1.0, 0.0, 0.0);
        let fo0 = glam::vec3(0.5, 1.0, 0.0);
        let fo1 = glam::vec3(0.5, 0.0, 1.0);

        let folded_angle = dihedral_angle(f0, f1, fo0, fo1);
        assert!((folded_angle - std::f32::consts::PI / 2.0).abs() < 0.1);
    }
}
