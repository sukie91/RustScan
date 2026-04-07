use crate::diff::{DiffCamera, DiffSplatRenderer, TrainableGaussians};
use crate::{Gaussian, SceneMetadata, TrainingDataset};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

#[cfg(feature = "gpu")]
use super::splats::Splats;
#[cfg(feature = "gpu")]
use candle_core::Device;

pub const MIN_RENDER_SCALE: f32 = 0.0625;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FinalTrainingMetrics {
    pub final_loss: f32,
    pub final_step_loss: f32,
}

pub fn summarize_training_metrics(
    loss_history: &[f32],
    frame_count: usize,
) -> FinalTrainingMetrics {
    FinalTrainingMetrics {
        final_loss: summarized_final_loss(loss_history, frame_count),
        final_step_loss: loss_history.last().copied().unwrap_or(0.0),
    }
}

fn summarized_final_loss(loss_history: &[f32], frame_count: usize) -> f32 {
    if loss_history.is_empty() {
        return 0.0;
    }

    let window = frame_count.max(1).min(loss_history.len());
    let start = loss_history.len() - window;
    loss_history[start..].iter().sum::<f32>() / window as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationDevice {
    Cpu,
    Metal,
}

impl std::fmt::Display for EvaluationDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Metal => write!(f, "metal"),
        }
    }
}

impl FromStr for EvaluationDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            other => Err(format!(
                "unsupported evaluation device '{other}'. Expected cpu or metal"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneEvaluationConfig {
    pub render_scale: f32,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub worst_frame_count: usize,
}

impl Default for SceneEvaluationConfig {
    fn default() -> Self {
        Self {
            render_scale: 0.5,
            frame_stride: 1,
            max_frames: 0,
            worst_frame_count: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationFrameMetric {
    pub dataset_index: usize,
    pub frame_id: u64,
    pub psnr_db: f32,
    pub image_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PsnrSummary {
    pub mean_db: f32,
    pub median_db: f32,
    pub min_db: f32,
    pub max_db: f32,
    pub stddev_db: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEvaluationSummary {
    pub device: EvaluationDevice,
    pub render_scale: f32,
    pub render_width: usize,
    pub render_height: usize,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub frame_count: usize,
    pub scene_iterations: usize,
    pub scene_gaussian_count: usize,
    pub final_loss: f32,
    pub final_step_loss: Option<f32>,
    pub elapsed_seconds: f32,
    pub psnr_mean_db: f32,
    pub psnr_median_db: f32,
    pub psnr_min_db: f32,
    pub psnr_max_db: f32,
    pub psnr_std_db: f32,
    pub worst_frames: Vec<EvaluationFrameMetric>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneEvaluationResult {
    pub summary: SceneEvaluationSummary,
    pub frame_metrics: Vec<EvaluationFrameMetric>,
}

#[derive(Debug, thiserror::Error)]
pub enum SceneEvaluationError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    #[cfg(feature = "gpu")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("invalid evaluation input: {0}")]
    InvalidInput(String),
}

pub fn select_evaluation_frames(
    dataset: &TrainingDataset,
    max_frames: usize,
    frame_stride: usize,
) -> TrainingDataset {
    let mut selected =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
    selected.initial_points = dataset.initial_points.clone();
    for pose in dataset
        .poses
        .iter()
        .take(if max_frames == 0 {
            dataset.poses.len()
        } else {
            max_frames
        })
        .step_by(frame_stride.max(1))
    {
        selected.add_pose(pose.clone());
    }
    selected
}

pub fn summarize_psnr_samples(values: &[f32]) -> PsnrSummary {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 0 {
        let hi = sorted.len() / 2;
        (sorted[hi - 1] + sorted[hi]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len().max(1) as f32;

    PsnrSummary {
        mean_db: mean,
        median_db: median,
        min_db: min,
        max_db: max,
        stddev_db: variance.sqrt(),
    }
}

pub fn worst_frame_metrics(
    frame_metrics: &[EvaluationFrameMetric],
    count: usize,
) -> Vec<EvaluationFrameMetric> {
    let mut sorted = frame_metrics.to_vec();
    sorted.sort_by(|a, b| {
        a.psnr_db
            .partial_cmp(&b.psnr_db)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
        .into_iter()
        .take(count.min(frame_metrics.len()))
        .collect()
}

pub fn compute_psnr_f32(rendered: &[f32], target: &[f32]) -> f32 {
    if rendered.len() != target.len() || rendered.is_empty() {
        return 0.0;
    }

    let mse = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| {
            let diff = r - t;
            diff * diff
        })
        .sum::<f32>()
        / rendered.len() as f32;

    if mse <= 1e-12 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

pub fn scaled_dimensions(width: usize, height: usize, render_scale: f32) -> (usize, usize) {
    let scale = render_scale.clamp(MIN_RENDER_SCALE, 1.0);
    let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
    let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
    (scaled_width, scaled_height)
}

#[cfg(feature = "gpu")]
pub fn evaluation_device(device: EvaluationDevice) -> Result<Device, SceneEvaluationError> {
    match device {
        EvaluationDevice::Cpu => Ok(Device::Cpu),
        EvaluationDevice::Metal => {
            crate::try_metal_device().map_err(SceneEvaluationError::InvalidInput)
        }
    }
}

#[cfg(feature = "gpu")]
pub fn trainable_from_scene(
    scene: &[Gaussian],
    metadata: &SceneMetadata,
    device: &Device,
) -> Result<TrainableGaussians, SceneEvaluationError> {
    let inferred_degree = infer_sh_degree(scene);
    let sh_degree = metadata.sh_degree.max(inferred_degree);
    Ok(Splats::from_scene_gaussians(scene, sh_degree)
        .map_err(|err| SceneEvaluationError::InvalidInput(err.to_string()))?
        .to_trainable(device)?)
}

#[cfg(feature = "gpu")]
pub fn render_evaluation_frame(
    dataset: &TrainingDataset,
    pose: &rustscan_types::ScenePose,
    render_width: usize,
    render_height: usize,
    device: &Device,
    trainable: &TrainableGaussians,
    renderer: &mut DiffSplatRenderer,
) -> Result<(Vec<f32>, Vec<f32>), SceneEvaluationError> {
    let target = load_resized_target(
        &pose.image_path,
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_width,
        render_height,
    )?;
    let camera = scaled_camera_for_pose(
        pose.pose,
        dataset.intrinsics.fx,
        dataset.intrinsics.fy,
        dataset.intrinsics.cx,
        dataset.intrinsics.cy,
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_width,
        render_height,
        device,
    )?;
    let output = renderer.render(trainable, &camera)?;
    let rendered = output.color.flatten_all()?.to_vec1::<f32>()?;
    Ok((target, rendered))
}

#[cfg(feature = "gpu")]
pub fn evaluate_scene(
    dataset: &TrainingDataset,
    scene: &[Gaussian],
    metadata: &SceneMetadata,
    config: &SceneEvaluationConfig,
    device: &Device,
    training_metrics: Option<FinalTrainingMetrics>,
) -> Result<SceneEvaluationResult, SceneEvaluationError> {
    let dataset = select_evaluation_frames(dataset, config.max_frames, config.frame_stride);
    if dataset.poses.is_empty() {
        return Err(SceneEvaluationError::InvalidInput(
            "evaluation dataset resolved to zero frames".to_string(),
        ));
    }

    let (render_width, render_height) = scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        config.render_scale,
    );
    let trainable = trainable_from_scene(scene, metadata, device)?;
    let mut renderer = DiffSplatRenderer::with_device(render_width, render_height, device.clone());

    let start = Instant::now();
    let mut psnrs = Vec::with_capacity(dataset.poses.len());
    let mut frame_metrics = Vec::with_capacity(dataset.poses.len());

    for (idx, pose) in dataset.poses.iter().enumerate() {
        let (target, rendered) = render_evaluation_frame(
            &dataset,
            pose,
            render_width,
            render_height,
            device,
            &trainable,
            &mut renderer,
        )?;
        let psnr_db = compute_psnr_f32(&rendered, &target);
        psnrs.push(psnr_db);
        frame_metrics.push(EvaluationFrameMetric {
            dataset_index: idx,
            frame_id: pose.frame_id,
            psnr_db,
            image_path: pose.image_path.clone(),
        });
    }

    let psnr = summarize_psnr_samples(&psnrs);
    let final_loss = training_metrics
        .map(|metrics| metrics.final_loss)
        .unwrap_or(metadata.final_loss);
    let final_step_loss = training_metrics.map(|metrics| metrics.final_step_loss);
    let summary = SceneEvaluationSummary {
        device: if matches!(device, Device::Cpu) {
            EvaluationDevice::Cpu
        } else {
            EvaluationDevice::Metal
        },
        render_scale: config.render_scale,
        render_width,
        render_height,
        frame_stride: config.frame_stride,
        max_frames: config.max_frames,
        frame_count: dataset.poses.len(),
        scene_iterations: metadata.iterations,
        scene_gaussian_count: metadata.gaussian_count,
        final_loss,
        final_step_loss,
        elapsed_seconds: start.elapsed().as_secs_f32(),
        psnr_mean_db: psnr.mean_db,
        psnr_median_db: psnr.median_db,
        psnr_min_db: psnr.min_db,
        psnr_max_db: psnr.max_db,
        psnr_std_db: psnr.stddev_db,
        worst_frames: worst_frame_metrics(&frame_metrics, config.worst_frame_count),
    };

    Ok(SceneEvaluationResult {
        summary,
        frame_metrics,
    })
}

#[cfg(feature = "gpu")]
fn scaled_camera_for_pose(
    pose_c2w: rustscan_types::SE3,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    device: &Device,
) -> candle_core::Result<DiffCamera> {
    let view_pose = pose_c2w.inverse();
    let sx = dst_width as f32 / src_width as f32;
    let sy = dst_height as f32 / src_height as f32;
    DiffCamera::new(
        fx * sx,
        fy * sy,
        cx * sx,
        cy * sy,
        dst_width,
        dst_height,
        &view_pose.rotation(),
        &view_pose.translation(),
        device,
    )
}

fn load_resized_target(
    path: &Path,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>, SceneEvaluationError> {
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let rgb = image.to_rgb8();
    let (actual_width, actual_height) = rgb.dimensions();
    if actual_width as usize != src_width || actual_height as usize != src_height {
        return Err(SceneEvaluationError::InvalidInput(format!(
            "image {} has size {}x{}, expected {}x{}",
            path.display(),
            actual_width,
            actual_height,
            src_width,
            src_height
        )));
    }
    let src: Vec<f32> = rgb
        .into_raw()
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect();
    Ok(resize_rgb_box(
        &src, src_width, src_height, dst_width, dst_height,
    ))
}

fn resize_rgb_box(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height * 3];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = [0.0f32; 3];
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src_idx = (sy * src_width + sx) * 3;
                    acc[0] += src[src_idx];
                    acc[1] += src[src_idx + 1];
                    acc[2] += src[src_idx + 2];
                    count += 1;
                }
            }
            let dst_idx = (dy * dst_width + dx) * 3;
            let inv = 1.0 / count.max(1) as f32;
            dst[dst_idx] = acc[0] * inv;
            dst[dst_idx + 1] = acc[1] * inv;
            dst[dst_idx + 2] = acc[2] * inv;
        }
    }
    dst
}

fn infer_sh_degree(scene: &[Gaussian]) -> usize {
    scene
        .iter()
        .filter_map(|gaussian| gaussian.sh_rest.as_ref())
        .find_map(|values| infer_sh_degree_from_value_count(values.len()))
        .unwrap_or(0)
}

fn infer_sh_degree_from_value_count(value_count: usize) -> Option<usize> {
    if value_count == 0 || value_count % 3 != 0 {
        return None;
    }

    let coeff_count = value_count / 3;
    for degree in 1..=8 {
        if (degree + 1) * (degree + 1) - 1 == coeff_count {
            return Some(degree);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        infer_sh_degree_from_value_count, select_evaluation_frames, summarize_psnr_samples,
        summarize_training_metrics, worst_frame_metrics, EvaluationFrameMetric,
        FinalTrainingMetrics,
    };
    use crate::{Intrinsics, ScenePose, TrainingDataset, SE3};
    use std::path::PathBuf;

    #[test]
    fn summarize_training_metrics_tracks_last_epoch_mean_and_last_step() {
        let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
        assert_eq!(
            summarize_training_metrics(&history, 2),
            FinalTrainingMetrics {
                final_loss: 0.55,
                final_step_loss: 0.5,
            }
        );
    }

    #[test]
    fn summarize_psnr_samples_tracks_distribution() {
        let summary = summarize_psnr_samples(&[1.0, 2.0, 3.0, 4.0]);
        assert!((summary.mean_db - 2.5).abs() < 1e-6);
        assert!((summary.median_db - 2.5).abs() < 1e-6);
        assert_eq!(summary.min_db, 1.0);
        assert_eq!(summary.max_db, 4.0);
        assert!((summary.stddev_db - 1.118_034).abs() < 1e-5);
    }

    #[test]
    fn worst_frame_metrics_returns_low_psnr_prefix() {
        let metrics = vec![
            EvaluationFrameMetric {
                dataset_index: 0,
                frame_id: 0,
                psnr_db: 9.0,
                image_path: PathBuf::from("a.png"),
            },
            EvaluationFrameMetric {
                dataset_index: 1,
                frame_id: 1,
                psnr_db: 3.0,
                image_path: PathBuf::from("b.png"),
            },
            EvaluationFrameMetric {
                dataset_index: 2,
                frame_id: 2,
                psnr_db: 6.0,
                image_path: PathBuf::from("c.png"),
            },
        ];
        let worst = worst_frame_metrics(&metrics, 2);
        assert_eq!(worst.len(), 2);
        assert_eq!(worst[0].frame_id, 1);
        assert_eq!(worst[1].frame_id, 2);
    }

    #[test]
    fn infer_sh_degree_from_value_count_maps_degree_three_layout() {
        assert_eq!(infer_sh_degree_from_value_count(45), Some(3));
        assert_eq!(infer_sh_degree_from_value_count(0), None);
        assert_eq!(infer_sh_degree_from_value_count(4), None);
    }

    #[test]
    fn select_evaluation_frames_honors_stride_within_prefix() {
        let intrinsics = Intrinsics::from_focal(500.0, 64, 48);
        let mut dataset = TrainingDataset::new(intrinsics);
        for idx in 0..6 {
            dataset.add_pose(ScenePose::new(
                idx as u64,
                PathBuf::from(format!("frame_{idx:04}.png")),
                SE3::identity(),
                idx as f64,
            ));
        }

        let selected = select_evaluation_frames(&dataset, 5, 2);
        let frame_ids = selected
            .poses
            .iter()
            .map(|pose| pose.frame_id)
            .collect::<Vec<_>>();
        assert_eq!(frame_ids, vec![0, 2, 4]);
    }
}
