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
    /// Absolute maximum scale (meters) after scene-derived clamping.
    pub max_scale: f32,
    /// Scale factor applied to the average of the two nearest-neighbor distances.
    pub scale_factor: f32,
    /// Default color when point color is unavailable (RGB, 0-1).
    pub default_color: [f32; 3],
    /// Default opacity for initialized Gaussians.
    pub opacity: f32,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            min_scale: 1e-3,
            max_scale: f32::MAX,
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
    if points.len() < 3 {
        return vec![1.0; points.len()];
    }

    let mut tree: KdTree<f32, 3> = KdTree::new();
    for (idx, pos) in points.iter().enumerate() {
        tree.add(&[pos.x, pos.y, pos.z], idx as u64);
    }

    let scene_max_scale = brush_scene_max_scale(points).clamp(config.min_scale, config.max_scale);
    let mut scales = Vec::with_capacity(points.len());
    for (idx, pos) in points.iter().enumerate() {
        let query = [pos.x, pos.y, pos.z];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, points.len().min(3));

        let mut nearest = [None, None];
        let mut count = 0usize;
        for neighbor in neighbors {
            if neighbor.item as usize == idx {
                continue;
            }
            nearest[count] = Some(neighbor.distance.sqrt());
            count += 1;
            if count == nearest.len() {
                break;
            }
        }

        let scale = if let [Some(first), Some(second)] = nearest {
            let avg_neighbor_distance = (first + second) * 0.5;
            (avg_neighbor_distance * config.scale_factor).clamp(config.min_scale, scene_max_scale)
        } else {
            1.0
        };
        scales.push(scale);
    }

    scales
}

fn brush_scene_max_scale(points: &[Vec3]) -> f32 {
    let bounds = percentile_bounds(points, 0.75);
    let mut extents = [
        (bounds.1.x - bounds.0.x) * 0.5,
        (bounds.1.y - bounds.0.y) * 0.5,
        (bounds.1.z - bounds.0.z) * 0.5,
    ];
    extents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_size = (extents[1] * 2.0).max(0.01);
    median_size * 0.1
}

fn percentile_bounds(points: &[Vec3], percentile: f32) -> (Vec3, Vec3) {
    let mut xs = Vec::with_capacity(points.len());
    let mut ys = Vec::with_capacity(points.len());
    let mut zs = Vec::with_capacity(points.len());
    for pos in points {
        xs.push(pos.x);
        ys.push(pos.y);
        zs.push(pos.z);
    }

    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = xs.len().max(1);
    let lower_idx = (((1.0 - percentile) * 0.5) * len as f32) as usize;
    let upper_idx = (len - 1).min((((1.0 + percentile) * 0.5) * len as f32) as usize);

    (
        Vec3::new(xs[lower_idx], ys[lower_idx], zs[lower_idx]),
        Vec3::new(xs[upper_idx], ys[upper_idx], zs[upper_idx]),
    )
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
            ([0.0, 1.0, 0.0], Some([0.1, 0.8, 0.2])),
            ([100.0, 100.0, 100.0], Some([0.7, 0.6, 0.5])),
        ];

        let config = GaussianInitConfig::default();

        let splats = initialize_host_splats_from_points(&points, &config, 0).unwrap();
        assert_eq!(splats.len(), 4);

        let origin_scale = splats.scale(0)[0];
        assert!((origin_scale - 0.5).abs() < 1e-6);
        let near_scale = splats.scale(1)[0];
        assert!((near_scale - 0.6035534).abs() < 1e-5);
        let outlier_scale = splats.scale(3)[0];
        assert!((outlier_scale - 10.0).abs() < 1e-6);

        for idx in 0..splats.len() {
            let scale = splats.scale(idx);
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
        assert_eq!(splats.scale(0)[0], 1.0);
        assert_eq!(splats.scale(0)[1], 1.0);
        assert_eq!(splats.scale(0)[2], 1.0);
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
