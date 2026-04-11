//! Splat initialization from point clouds.
//!
//! Provides utilities to create initial splats for 3DGS training
//! from sparse point clouds (e.g., SLAM map points).
//! Uses KdTree for nearest-neighbor scale computation.

use glam::Vec3;
use kiddo::{KdTree, SquaredEuclidean};

#[cfg(feature = "gpu")]
use candle_core::Device;

#[cfg(feature = "gpu")]
use crate::diff::diff_splat::Splats;
#[cfg(feature = "gpu")]
use crate::training::HostSplats;

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

/// Initialize runtime splats directly on device from a point cloud.
#[cfg(feature = "gpu")]
pub fn initialize_runtime_splats_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
    device: &Device,
) -> candle_core::Result<Splats> {
    initialize_host_splats_from_points(points, config, 0)?.upload(device)
}

/// Initialize host-side splats from a point cloud without materializing AoS gaussians.
#[cfg(feature = "gpu")]
pub fn initialize_host_splats_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
    sh_degree: usize,
) -> candle_core::Result<HostSplats> {
    if points.is_empty() {
        return HostSplats::from_raw_parts(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            sh_degree,
        );
    }

    let positions_vec3: Vec<Vec3> = points
        .iter()
        .map(|(p, _)| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let scales = compute_scales(&positions_vec3, config);

    let mut splats = HostSplats::with_sh_degree_capacity(sh_degree, points.len());
    for ((position, color), scale) in points.iter().zip(scales.iter()) {
        let rgb = color.unwrap_or(config.default_color);
        splats.push_rgb(
            *position,
            [scale.ln(), scale.ln(), scale.ln()],
            [1.0, 0.0, 0.0, 0.0],
            opacity_to_logit(config.opacity),
            rgb,
        );
    }

    splats.validate()?;
    Ok(splats)
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

    #[cfg(feature = "gpu")]
    #[test]
    fn test_initialize_host_splats_scale_and_color() {
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

        let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();
        assert_eq!(splats.len(), 2);

        for idx in 0..splats.len() {
            let scale = splats.scale(idx);
            assert!((scale[0] - 0.5).abs() < 1e-6);
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
        assert_eq!(splats.scale(0)[0], config.min_scale);
        assert_eq!(splats.scale(0)[1], config.min_scale);
        assert_eq!(splats.scale(0)[2], config.min_scale);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_initialize_host_splats_empty_points() {
        let points: Vec<([f32; 3], Option<[f32; 3]>)> = vec![];
        let config = GaussianInitConfig::default();
        let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();
        assert!(splats.is_empty());
    }
}
