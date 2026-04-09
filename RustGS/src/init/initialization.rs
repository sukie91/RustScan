//! Gaussian initialization from point clouds.
//!
//! Provides utilities to create initial Gaussians for 3DGS training
//! from sparse point clouds (e.g., SLAM map points).
//! Uses KdTree for nearest-neighbor scale computation.

use glam::Vec3;
use kiddo::{KdTree, SquaredEuclidean};

use crate::core::Gaussian3D;
use crate::render::tiled_renderer::Gaussian;

#[cfg(feature = "gpu")]
use candle_core::Device;

#[cfg(feature = "gpu")]
use crate::diff::diff_splat::Splats;

/// Configuration for Gaussian initialization from point clouds.
#[derive(Debug, Clone)]
pub struct GaussianInitConfig {
    /// Minimum scale (meters).
    pub min_scale: f32,
    /// Maximum scale (meters).
    pub max_scale: f32,
    /// Scale factor applied to nearest-neighbor distance.
    pub scale_factor: f32,
    /// Default color when point color is unavailable (RGB, 0-1).
    pub default_color: [f32; 3],
    /// Default opacity for initialized Gaussians.
    pub opacity: f32,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            min_scale: 0.005,
            max_scale: 0.2,
            scale_factor: 0.5,
            default_color: [0.5, 0.5, 0.5],
            opacity: 0.5,
        }
    }
}

/// Initialize Gaussians from a point cloud.
///
/// # Arguments
/// * `points` - Points as (position [x,y,z], optional color [r,g,b]) tuples
/// * `config` - Initialization configuration
///
/// # Returns
/// A vector of initialized `Gaussian` (array-based) primitives.
pub fn initialize_gaussians_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
) -> Vec<Gaussian> {
    if points.is_empty() {
        return Vec::new();
    }

    let positions: Vec<Vec3> = points
        .iter()
        .map(|(p, _)| Vec3::new(p[0], p[1], p[2]))
        .collect();

    let scales = compute_scales(&positions, config);

    points
        .iter()
        .zip(scales.iter())
        .map(|((pos, color), scale)| {
            let rgb = color.unwrap_or(config.default_color);
            Gaussian::new(
                *pos,
                [*scale, *scale, *scale],
                [1.0, 0.0, 0.0, 0.0],
                config.opacity,
                rgb,
            )
        })
        .collect()
}

/// Initialize Gaussians as `Gaussian3D` (glam-based) from a point cloud.
pub fn initialize_gaussian3d_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
) -> Vec<Gaussian3D> {
    if points.is_empty() {
        return Vec::new();
    }

    let positions: Vec<Vec3> = points
        .iter()
        .map(|(p, _)| Vec3::new(p[0], p[1], p[2]))
        .collect();

    let scales = compute_scales(&positions, config);

    points
        .iter()
        .zip(scales.iter())
        .map(|((pos, color), scale)| {
            let rgb = color.unwrap_or(config.default_color);
            Gaussian3D::new(
                Vec3::new(pos[0], pos[1], pos[2]),
                Vec3::splat(*scale),
                glam::Quat::IDENTITY,
                config.opacity,
                rgb,
            )
        })
        .collect()
}

/// Initialize trainable Gaussians (for differentiable training) from a point cloud.
#[cfg(feature = "gpu")]
pub fn initialize_trainable_gaussians_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
    device: &Device,
) -> candle_core::Result<Splats> {
    let gaussians = initialize_gaussians_from_points(points, config);
    let n = gaussians.len();

    let mut positions = Vec::with_capacity(n * 3);
    let mut scales = Vec::with_capacity(n * 3);
    let mut rotations = Vec::with_capacity(n * 4);
    let mut opacities = Vec::with_capacity(n);
    let mut colors = Vec::with_capacity(n * 3);

    for g in gaussians {
        positions.extend_from_slice(&g.position);
        scales.extend_from_slice(&[g.scale[0].ln(), g.scale[1].ln(), g.scale[2].ln()]);
        rotations.extend_from_slice(&g.rotation);
        opacities.push(opacity_to_logit(g.opacity));
        colors.extend_from_slice(&g.color);
    }

    Splats::new(&positions, &scales, &rotations, &opacities, &colors, device)
}

fn compute_scales(points: &[Vec3], config: &GaussianInitConfig) -> Vec<f32> {
    if points.len() == 1 {
        return vec![config.min_scale];
    }

    let mut tree: KdTree<f32, 3> = KdTree::new();
    for (idx, pos) in points.iter().enumerate() {
        tree.add(&[pos.x, pos.y, pos.z], idx as u64);
    }

    let mut scales = Vec::with_capacity(points.len());
    for (idx, pos) in points.iter().enumerate() {
        let query = [pos.x, pos.y, pos.z];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, 2);

        let mut nearest = None;
        for n in neighbors {
            if n.item as usize != idx {
                nearest = Some(n.distance);
                break;
            }
        }

        let dist = nearest.map(|d| d.sqrt()).unwrap_or(config.min_scale);
        let scale = (dist * config.scale_factor).clamp(config.min_scale, config.max_scale);
        scales.push(scale);
    }

    scales
}

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_gaussians_scale_and_color() {
        let points = vec![
            ([0.0, 0.0, 0.0], Some([0.2, 0.3, 0.4])),
            ([1.0, 0.0, 0.0], Some([0.9, 0.1, 0.2])),
        ];

        let config = GaussianInitConfig {
            min_scale: 0.1,
            max_scale: 1.0,
            scale_factor: 0.5,
            default_color: [0.5, 0.5, 0.5],
            opacity: 0.5,
        };

        let gaussians = initialize_gaussians_from_points(&points, &config);
        assert_eq!(gaussians.len(), 2);

        for g in &gaussians {
            assert!((g.scale[0] - 0.5).abs() < 1e-6);
            assert_eq!(g.scale[0], g.scale[1]);
            assert_eq!(g.scale[1], g.scale[2]);
            assert_eq!(g.opacity, 0.5);
        }

        assert!(gaussians.iter().any(|g| g.color == [0.2, 0.3, 0.4]));
        assert!(gaussians.iter().any(|g| g.color == [0.9, 0.1, 0.2]));
    }

    #[test]
    fn test_initialize_gaussian3d_count() {
        let points = vec![([0.0, 0.0, 1.0], None)];

        let config = GaussianInitConfig::default();
        let gaussians = initialize_gaussian3d_from_points(&points, &config);
        assert_eq!(gaussians.len(), 1);
    }

    #[test]
    fn test_initialize_gaussians_defaults() {
        let points = vec![([0.0, 0.0, 1.0], None)];

        let config = GaussianInitConfig::default();
        let gaussians = initialize_gaussians_from_points(&points, &config);
        assert_eq!(gaussians.len(), 1);

        let g = &gaussians[0];
        assert_eq!(g.rotation, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.opacity, config.opacity);
        assert_eq!(g.color, config.default_color);
        assert_eq!(g.scale[0], config.min_scale);
        assert_eq!(g.scale[1], config.min_scale);
        assert_eq!(g.scale[2], config.min_scale);
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<([f32; 3], Option<[f32; 3]>)> = vec![];
        let config = GaussianInitConfig::default();
        let gaussians = initialize_gaussians_from_points(&points, &config);
        assert!(gaussians.is_empty());
    }
}
