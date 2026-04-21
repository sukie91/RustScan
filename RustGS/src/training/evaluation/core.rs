use crate::core::GaussianCamera;
use crate::{SplatMetadata, TrainingDataset};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

#[cfg(feature = "gpu")]
use crate::core::HostSplats;

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
}

impl std::fmt::Display for EvaluationDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

impl FromStr for EvaluationDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "gpu" | "wgpu" | "metal" => Ok(Self::Cpu),
            other => Err(format!(
                "unsupported evaluation device '{other}'. Expected cpu"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplatEvaluationConfig {
    pub render_scale: f32,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub worst_frame_count: usize,
}

impl Default for SplatEvaluationConfig {
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
pub struct SplatEvaluationSummary {
    pub device: EvaluationDevice,
    pub render_scale: f32,
    pub render_width: usize,
    pub render_height: usize,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub frame_count: usize,
    #[serde(alias = "scene_iterations")]
    pub splat_iterations: usize,
    #[serde(alias = "scene_gaussian_count")]
    pub splat_count: usize,
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
pub struct SplatEvaluationResult {
    pub summary: SplatEvaluationSummary,
    pub frame_metrics: Vec<EvaluationFrameMetric>,
}

#[derive(Debug, thiserror::Error)]
pub enum SplatEvaluationError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

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
            let delta = value - mean;
            delta * delta
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
pub fn evaluation_device(
    device: EvaluationDevice,
) -> Result<EvaluationDevice, SplatEvaluationError> {
    Ok(device)
}

#[cfg(feature = "gpu")]
pub fn runtime_from_splats(
    splats: &HostSplats,
    _device: &EvaluationDevice,
) -> Result<HostSplats, SplatEvaluationError> {
    Ok(splats.clone())
}

#[cfg(feature = "gpu")]
pub struct SplatEvaluationRenderer {
    renderer: crate::GaussianRenderer,
}

#[cfg(feature = "gpu")]
impl SplatEvaluationRenderer {
    pub fn new(
        render_width: usize,
        render_height: usize,
        _device: EvaluationDevice,
    ) -> Result<Self, SplatEvaluationError> {
        Ok(Self {
            renderer: crate::GaussianRenderer::new(render_width, render_height),
        })
    }

    pub fn render(
        &mut self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> Result<Vec<f32>, SplatEvaluationError> {
        let output = self
            .renderer
            .render_splats(splats, camera)
            .map_err(|err| SplatEvaluationError::InvalidInput(err.to_string()))?;
        Ok(output
            .color
            .into_iter()
            .map(|value| value as f32 / 255.0)
            .collect())
    }
}

#[cfg(feature = "gpu")]
pub fn render_evaluation_frame(
    dataset: &TrainingDataset,
    pose: &rustscan_types::ScenePose,
    render_width: usize,
    render_height: usize,
    _device: &EvaluationDevice,
    trainable: &HostSplats,
    renderer: &mut SplatEvaluationRenderer,
) -> Result<(Vec<f32>, Vec<f32>), SplatEvaluationError> {
    let target = load_resized_target(
        &pose.image_path,
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_width,
        render_height,
    )?;
    let camera = scaled_camera_for_pose(pose.pose, dataset.intrinsics, render_width, render_height);
    let rendered = renderer.render(trainable, &camera)?;
    Ok((target, rendered))
}

#[cfg(feature = "gpu")]
pub fn evaluate_splats(
    dataset: &TrainingDataset,
    splats: &HostSplats,
    metadata: &SplatMetadata,
    config: &SplatEvaluationConfig,
    device: &EvaluationDevice,
    training_metrics: Option<FinalTrainingMetrics>,
) -> Result<SplatEvaluationResult, SplatEvaluationError> {
    let dataset = select_evaluation_frames(dataset, config.max_frames, config.frame_stride);
    if dataset.poses.is_empty() {
        return Err(SplatEvaluationError::InvalidInput(
            "evaluation dataset resolved to zero frames".to_string(),
        ));
    }

    let (render_width, render_height) = scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        config.render_scale,
    );
    let runtime_splats = runtime_from_splats(splats, device)?;
    let mut renderer = SplatEvaluationRenderer::new(render_width, render_height, *device)?;

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
            &runtime_splats,
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
    let summary = SplatEvaluationSummary {
        device: *device,
        render_scale: config.render_scale,
        render_width,
        render_height,
        frame_stride: config.frame_stride,
        max_frames: config.max_frames,
        frame_count: dataset.poses.len(),
        splat_iterations: metadata.iterations,
        splat_count: metadata.gaussian_count,
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

    Ok(SplatEvaluationResult {
        summary,
        frame_metrics,
    })
}

#[cfg(feature = "gpu")]
fn scaled_camera_for_pose(
    pose_c2w: rustscan_types::SE3,
    intrinsics: rustscan_types::Intrinsics,
    dst_width: usize,
    dst_height: usize,
) -> GaussianCamera {
    let view_pose = pose_c2w.inverse();
    let sx = dst_width as f32 / intrinsics.width as f32;
    let sy = dst_height as f32 / intrinsics.height as f32;
    GaussianCamera::new(
        rustscan_types::Intrinsics::new(
            intrinsics.fx * sx,
            intrinsics.fy * sy,
            intrinsics.cx * sx,
            intrinsics.cy * sy,
            dst_width as u32,
            dst_height as u32,
        ),
        view_pose,
    )
}

fn load_resized_target(
    path: &Path,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>, SplatEvaluationError> {
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let rgb = image.to_rgb8();
    let (actual_width, actual_height) = rgb.dimensions();
    if actual_width as usize != src_width || actual_height as usize != src_height {
        return Err(SplatEvaluationError::InvalidInput(format!(
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

#[cfg(test)]
mod tests {
    use super::{
        select_evaluation_frames, summarize_psnr_samples, summarize_training_metrics,
        worst_frame_metrics, EvaluationFrameMetric, FinalTrainingMetrics,
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
    fn select_evaluation_frames_copies_initial_points_and_stride() {
        let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], Some([1.0, 0.0, 0.0]));
        for idx in 0..6 {
            dataset.add_pose(ScenePose::new(
                idx as u64,
                PathBuf::from(format!("frame-{idx}.png")),
                SE3::identity(),
                idx as f64,
            ));
        }

        let selected = select_evaluation_frames(&dataset, 5, 2);
        assert_eq!(selected.initial_points.len(), 1);
        assert_eq!(selected.poses.len(), 3);
        assert_eq!(selected.poses[0].frame_id, 0);
        assert_eq!(selected.poses[1].frame_id, 2);
        assert_eq!(selected.poses[2].frame_id, 4);
    }
}
