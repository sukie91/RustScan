#[cfg(feature = "gpu")]
use std::fs;
#[cfg(feature = "gpu")]
use std::path::Path;

#[cfg(feature = "gpu")]
use candle_core::Device;
#[cfg(feature = "gpu")]
use image::{DynamicImage, GenericImageView, ImageReader};

#[cfg(feature = "gpu")]
use crate::core::{Gaussian3D, GaussianMap, GaussianState};
#[cfg(feature = "gpu")]
use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};
#[cfg(feature = "gpu")]
use crate::init::{initialize_gaussian3d_from_points, GaussianInitConfig};
#[cfg(feature = "gpu")]
use crate::{TrainingDataset, TrainingError};

#[cfg(feature = "gpu")]
struct FrameSample {
    camera: DiffCamera,
    color_u8: Vec<u8>,
    color_f32: Vec<f32>,
    depth: Vec<f32>,
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

#[cfg(feature = "gpu")]
pub(crate) struct LoadedTrainingData {
    pub cameras: Vec<DiffCamera>,
    pub colors: Vec<Vec<f32>>,
    pub depths: Vec<Vec<f32>>,
    pub initial_map: GaussianMap,
}

#[cfg(feature = "gpu")]
pub(crate) fn load_training_data(
    dataset: &TrainingDataset,
    config: &super::TrainingConfig,
    device: &Device,
) -> Result<LoadedTrainingData, TrainingError> {
    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training dataset does not contain any poses".to_string(),
        ));
    }

    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;
    let expected_color = width.saturating_mul(height).saturating_mul(3);
    let expected_depth = width.saturating_mul(height);

    let mut frames = Vec::with_capacity(dataset.poses.len());
    for pose in &dataset.poses {
        let color_u8 = load_color_image(&pose.image_path, width, height)?;
        if color_u8.len() != expected_color {
            return Err(TrainingError::InvalidInput(format!(
                "image {} produced {} bytes, expected {}",
                pose.image_path.display(),
                color_u8.len(),
                expected_color,
            )));
        }

        let depth = match &pose.depth_path {
            Some(path) => load_depth_image(path, width, height, dataset.depth_scale)?,
            None if config.use_synthetic_depth => {
                synthetic_depth(&color_u8, width, height, config.min_depth, config.max_depth)
            }
            None => vec![0.0; expected_depth],
        };
        if depth.len() != expected_depth {
            return Err(TrainingError::InvalidInput(format!(
                "depth for frame {} produced {} values, expected {}",
                pose.frame_id,
                depth.len(),
                expected_depth,
            )));
        }

        let rotation = pose.pose.rotation();
        let translation = pose.pose.translation();
        let camera = DiffCamera::new(
            dataset.intrinsics.fx,
            dataset.intrinsics.fy,
            dataset.intrinsics.cx,
            dataset.intrinsics.cy,
            width,
            height,
            &rotation,
            &translation,
            device,
        )?;

        frames.push(FrameSample {
            camera,
            color_f32: color_u8.iter().map(|v| *v as f32 / 255.0).collect(),
            color_u8,
            depth,
            rotation,
            translation,
        });
    }

    let initial_map = if dataset.initial_points.is_empty() {
        build_initial_map_from_frames(dataset, &frames, config)?
    } else {
        let init_config = GaussianInitConfig::default();
        let gaussians = initialize_gaussian3d_from_points(&dataset.initial_points, &init_config);
        let mut map = GaussianMap::from_gaussians(gaussians);
        map.update_states();
        map
    };

    let mut cameras = Vec::with_capacity(frames.len());
    let mut colors = Vec::with_capacity(frames.len());
    let mut depths = Vec::with_capacity(frames.len());
    for frame in frames {
        cameras.push(frame.camera);
        colors.push(frame.color_f32);
        depths.push(frame.depth);
    }

    Ok(LoadedTrainingData {
        cameras,
        colors,
        depths,
        initial_map,
    })
}

#[cfg(feature = "gpu")]
pub(crate) fn trainable_from_map(
    map: &GaussianMap,
    device: &Device,
) -> candle_core::Result<TrainableGaussians> {
    let mut positions = Vec::with_capacity(map.len() * 3);
    let mut scales = Vec::with_capacity(map.len() * 3);
    let mut rotations = Vec::with_capacity(map.len() * 4);
    let mut opacities = Vec::with_capacity(map.len());
    let mut colors = Vec::with_capacity(map.len() * 3);

    for gaussian in map.gaussians() {
        positions.extend_from_slice(&[
            gaussian.position.x,
            gaussian.position.y,
            gaussian.position.z,
        ]);
        scales.extend_from_slice(&[
            gaussian.scale.x.max(1e-6).ln(),
            gaussian.scale.y.max(1e-6).ln(),
            gaussian.scale.z.max(1e-6).ln(),
        ]);
        rotations.extend_from_slice(&[
            gaussian.rotation.w,
            gaussian.rotation.x,
            gaussian.rotation.y,
            gaussian.rotation.z,
        ]);
        opacities.push(opacity_to_logit(gaussian.opacity));
        colors.extend_from_slice(&gaussian.color);
    }

    TrainableGaussians::new(&positions, &scales, &rotations, &opacities, &colors, device)
}

#[cfg(feature = "gpu")]
pub(crate) fn map_from_trainable(
    gaussians: &TrainableGaussians,
) -> candle_core::Result<GaussianMap> {
    let positions = gaussians.positions().to_vec2::<f32>()?;
    let scales = gaussians.scales()?.to_vec2::<f32>()?;
    let rotations = gaussians.rotations()?.to_vec2::<f32>()?;
    let opacities = gaussians.opacities()?.to_vec1::<f32>()?;
    let colors = gaussians.colors().to_vec2::<f32>()?;

    let mut output = Vec::with_capacity(gaussians.len());
    for idx in 0..gaussians.len() {
        output.push(Gaussian3D {
            position: glam::Vec3::new(positions[idx][0], positions[idx][1], positions[idx][2]),
            scale: glam::Vec3::new(scales[idx][0], scales[idx][1], scales[idx][2]),
            rotation: glam::Quat::from_xyzw(
                rotations[idx][1],
                rotations[idx][2],
                rotations[idx][3],
                rotations[idx][0],
            ),
            opacity: opacities[idx].clamp(0.0, 1.0),
            color: [colors[idx][0], colors[idx][1], colors[idx][2]],
            features: None,
            state: GaussianState::Stable,
        });
    }

    let mut map = GaussianMap::from_gaussians(output);
    map.update_states();
    Ok(map)
}

#[cfg(feature = "gpu")]
fn build_initial_map_from_frames(
    dataset: &TrainingDataset,
    frames: &[FrameSample],
    config: &super::TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let width = dataset.intrinsics.width as usize;
    let height = dataset.intrinsics.height as usize;
    let mut map = GaussianMap::new(config.max_initial_gaussians.max(1));

    let frame_budget = config
        .max_initial_gaussians
        .checked_div(frames.len().max(1))
        .unwrap_or(1)
        .max(1);
    let sampling_step = if config.sampling_step == 0 {
        compute_sampling_step(width, height, frame_budget)
    } else {
        config.sampling_step.max(1)
    };

    for frame in frames {
        let rotation = glam::Mat3::from_cols(
            glam::Vec3::new(
                frame.rotation[0][0],
                frame.rotation[0][1],
                frame.rotation[0][2],
            ),
            glam::Vec3::new(
                frame.rotation[1][0],
                frame.rotation[1][1],
                frame.rotation[1][2],
            ),
            glam::Vec3::new(
                frame.rotation[2][0],
                frame.rotation[2][1],
                frame.rotation[2][2],
            ),
        );
        let translation = glam::Vec3::new(
            frame.translation[0],
            frame.translation[1],
            frame.translation[2],
        );

        for y in (0..height).step_by(sampling_step) {
            for x in (0..width).step_by(sampling_step) {
                let idx = y * width + x;
                let depth = frame.depth[idx];
                if depth < config.min_depth || depth > config.max_depth {
                    continue;
                }

                let x_cam = (x as f32 - dataset.intrinsics.cx) * depth / dataset.intrinsics.fx;
                let y_cam = (y as f32 - dataset.intrinsics.cy) * depth / dataset.intrinsics.fy;
                let camera_point = glam::Vec3::new(x_cam, y_cam, depth);
                let world_point = rotation * camera_point + translation;
                let color_base = idx * 3;
                let gaussian = Gaussian3D::from_depth_point(
                    world_point.x,
                    world_point.y,
                    world_point.z,
                    [
                        frame.color_u8[color_base],
                        frame.color_u8[color_base + 1],
                        frame.color_u8[color_base + 2],
                    ],
                );
                let _ = map.add(gaussian);
            }
        }
    }

    if map.is_empty() {
        return Err(TrainingError::InvalidInput(
            "failed to build any initial gaussians from the training frames".to_string(),
        ));
    }

    if sampling_step > 1 && map.len() > config.max_initial_gaussians.max(1) {
        log::warn!(
            "initial Gaussian map exceeded budget ({} > {}) with sampling step {}",
            map.len(),
            config.max_initial_gaussians,
            sampling_step,
        );
    }

    map.update_states();
    Ok(map)
}

#[cfg(feature = "gpu")]
fn load_color_image(path: &Path, width: usize, height: usize) -> Result<Vec<u8>, TrainingError> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("rgb") => {
            let bytes = fs::read(path)?;
            let expected = width.saturating_mul(height).saturating_mul(3);
            if bytes.len() != expected {
                return Err(TrainingError::InvalidInput(format!(
                    "raw RGB frame {} has {} bytes, expected {}",
                    path.display(),
                    bytes.len(),
                    expected,
                )));
            }
            Ok(bytes)
        }
        _ => {
            let image = ImageReader::open(path)
                .map_err(TrainingError::Io)?
                .with_guessed_format()
                .map_err(|err| TrainingError::InvalidInput(err.to_string()))?
                .decode()
                .map_err(|err| TrainingError::InvalidInput(err.to_string()))?;

            let (actual_width, actual_height) = image.dimensions();
            if actual_width as usize != width || actual_height as usize != height {
                return Err(TrainingError::InvalidInput(format!(
                    "image {} has size {}x{}, expected {}x{}",
                    path.display(),
                    actual_width,
                    actual_height,
                    width,
                    height,
                )));
            }

            Ok(image.to_rgb8().into_raw())
        }
    }
}

#[cfg(feature = "gpu")]
fn load_depth_image(
    path: &Path,
    width: usize,
    height: usize,
    depth_scale: f32,
) -> Result<Vec<f32>, TrainingError> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("depth") => load_raw_depth(path, width, height),
        _ => load_raster_depth(path, width, height, depth_scale),
    }
}

#[cfg(feature = "gpu")]
fn load_raw_depth(path: &Path, width: usize, height: usize) -> Result<Vec<f32>, TrainingError> {
    let bytes = fs::read(path)?;
    let expected = width
        .saturating_mul(height)
        .saturating_mul(std::mem::size_of::<f32>());
    if bytes.len() != expected {
        return Err(TrainingError::InvalidInput(format!(
            "raw depth frame {} has {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected,
        )));
    }

    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(feature = "gpu")]
fn load_raster_depth(
    path: &Path,
    width: usize,
    height: usize,
    depth_scale: f32,
) -> Result<Vec<f32>, TrainingError> {
    if depth_scale <= 0.0 {
        return Err(TrainingError::InvalidInput(format!(
            "depth scale must be > 0, got {depth_scale}",
        )));
    }

    let image = ImageReader::open(path)
        .map_err(TrainingError::Io)?
        .with_guessed_format()
        .map_err(|err| TrainingError::InvalidInput(err.to_string()))?
        .decode()
        .map_err(|err| TrainingError::InvalidInput(err.to_string()))?;

    let (actual_width, actual_height) = image.dimensions();
    if actual_width as usize != width || actual_height as usize != height {
        return Err(TrainingError::InvalidInput(format!(
            "depth image {} has size {}x{}, expected {}x{}",
            path.display(),
            actual_width,
            actual_height,
            width,
            height,
        )));
    }

    match image {
        DynamicImage::ImageLuma16(luma) => Ok(luma
            .pixels()
            .map(|pixel| pixel.0[0] as f32 / depth_scale)
            .collect()),
        _ => Err(TrainingError::InvalidInput(format!(
            "unsupported depth image format for {}",
            path.display(),
        ))),
    }
}

#[cfg(feature = "gpu")]
fn synthetic_depth(
    color: &[u8],
    width: usize,
    height: usize,
    min_depth: f32,
    max_depth: f32,
) -> Vec<f32> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    let mut depth = Vec::with_capacity(width.saturating_mul(height));
    if color.len() < expected || min_depth >= max_depth {
        depth.resize(width.saturating_mul(height), min_depth.max(0.01));
        return depth;
    }

    let range = max_depth - min_depth;
    for chunk in color.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let value = max_depth - luma * range;
        depth.push(value.clamp(min_depth, max_depth));
    }

    depth
}

#[cfg(feature = "gpu")]
fn compute_sampling_step(width: usize, height: usize, max_gaussians: usize) -> usize {
    let pixels = width.saturating_mul(height).max(1);
    let ratio = (pixels as f32 / max_gaussians.max(1) as f32).sqrt();
    ratio.ceil().max(2.0) as usize
}

#[cfg(feature = "gpu")]
fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_load_raw_rgb() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("frame.rgb");
        fs::write(&path, [1u8, 2, 3, 4, 5, 6]).unwrap();

        let loaded = load_color_image(&path, 2, 1).unwrap();
        assert_eq!(loaded, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_load_raw_depth() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("frame.depth");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.25f32.to_le_bytes());
        bytes.extend_from_slice(&2.5f32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let loaded = load_depth_image(&path, 2, 1).unwrap();
        assert_eq!(loaded, vec![1.25, 2.5]);
    }

    #[test]
    fn test_synthetic_depth_range() {
        let color = vec![0u8, 0, 0, 255, 255, 255];
        let depth = synthetic_depth(&color, 2, 1, 0.1, 2.0);
        assert_eq!(depth.len(), 2);
        assert!(depth[0] >= 0.1 && depth[0] <= 2.0);
        assert!(depth[1] >= 0.1 && depth[1] <= 2.0);
        assert!(depth[0] > depth[1]);
    }
}
