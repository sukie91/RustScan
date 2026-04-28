#![allow(clippy::too_many_arguments)]

use crate::TrainArgs;
use anyhow::{bail, Context};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(feature = "gpu")]
pub(super) fn run_train_command(args: TrainArgs) -> anyhow::Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();

    log::info!("Training 3DGS splats from {:?}", args.input);
    log::info!("Output: {:?}", args.output);
    log::info!("Iterations: {}", args.iterations);
    log::info!("Backend: wgpu");
    if args.sampling_step != 0 {
        log::warn!(
            "--sampling-step={} is ignored because training now initializes strictly from dataset sparse points",
            args.sampling_step
        );
    }

    let (dataset, source) =
        load_training_dataset_for_training(&args.input, args.max_frames, args.frame_stride)?;
    let included_training_ranges = parse_frame_ranges(args.include_frame_ranges.as_deref())?;
    let dataset = filter_dataset_to_frame_ranges(dataset, &included_training_ranges, "training")?;
    let excluded_training_ranges = parse_frame_ranges(args.exclude_frame_ranges.as_deref())?;
    let dataset = filter_dataset_by_frame_ranges(dataset, &excluded_training_ranges, "training")?;
    let oversample_training_ranges = parse_frame_ranges(args.oversample_frame_ranges.as_deref())?;
    let dataset = oversample_dataset_frame_ranges(
        dataset,
        &oversample_training_ranges,
        args.oversample_frame_repeat,
        "training",
    )?;
    log::info!(
        "Loaded {} poses, {} initialization points",
        dataset.poses.len(),
        dataset.initial_points.len()
    );
    ensure_sparse_initialization_points(&dataset, source, &args.input)?;

    let config = build_training_config(&args)?;
    log::info!("Frame shuffle seed: {}", config.frame_shuffle_seed);
    log_litegs_training_config(&config);

    let training_run = rustgs::train_splats_with_report(&dataset, &config)?;
    let rustgs::TrainingRun {
        splats,
        report: training_report,
    } = training_run;
    let training_telemetry = training_report.telemetry.as_ref();

    log::info!(
        "CLI training summary | route=standard | final_gaussians={} | elapsed={:.2}s",
        training_report.gaussian_count,
        training_report.elapsed.as_secs_f64(),
    );
    log::info!("Trained {} Gaussians", splats.len());

    let metadata = rustgs::SplatMetadata {
        iterations: config.iterations,
        final_loss: training_report.metadata_final_loss_or(0.0),
        gaussian_count: splats.len(),
        sh_degree: splats.sh_degree(),
    };
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    rustgs::save_splats_ply(&args.output, &splats, &metadata)?;
    log::info!("Saved scene to {:?}", args.output);

    let evaluation_summary =
        maybe_evaluate_trained_splats(&args, &splats, &metadata, training_telemetry)?;

    if let Err(err) = maybe_write_litegs_parity_report(
        &args.input,
        &args.output,
        &dataset,
        &splats,
        &config,
        training_telemetry,
        training_report.elapsed,
        evaluation_summary.as_ref(),
    ) {
        log::warn!("failed to persist LiteGS parity report: {err}");
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub(super) fn run_train_command(args: TrainArgs) -> anyhow::Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    log::error!("GPU feature is required for training. Rebuild with --features gpu");
    std::process::exit(1);
}

pub(super) fn load_training_dataset_for_training(
    input: &Path,
    max_frames: usize,
    frame_stride: usize,
) -> anyhow::Result<(rustscan_types::TrainingDataset, rustgs::TrainingInputKind)> {
    if !input.is_dir() && (max_frames > 0 || frame_stride > 1) {
        log::warn!(
            "--max-frames and --frame-stride only apply to dataset directories; ignoring them for {:?}",
            input
        );
    }

    let (dataset, source) = rustgs::load_training_dataset_with_source(
        input,
        &rustgs::TumRgbdConfig {
            max_frames,
            frame_stride,
            ..Default::default()
        },
        &rustgs::ColmapConfig {
            max_frames,
            frame_stride,
            ..Default::default()
        },
    )?;

    log::info!(
        "Resolved {:?} as {} with {} poses",
        input,
        source,
        dataset.poses.len(),
    );

    Ok((dataset, source))
}

#[cfg(feature = "gpu")]
fn load_evaluation_dataset(
    input: &Path,
    max_frames: usize,
    frame_stride: usize,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    let (dataset, source) = rustgs::load_training_dataset_with_source(
        input,
        &rustgs::TumRgbdConfig {
            max_frames,
            frame_stride,
            ..Default::default()
        },
        &rustgs::ColmapConfig {
            max_frames,
            frame_stride,
            ..Default::default()
        },
    )?;

    log::info!(
        "Resolved evaluation dataset {:?} as {} with {} poses",
        input,
        source,
        dataset.poses.len(),
    );

    Ok(dataset)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameIdRange {
    start: u64,
    end: u64,
}

impl FrameIdRange {
    fn contains(&self, frame_id: u64) -> bool {
        self.start <= frame_id && frame_id <= self.end
    }
}

fn parse_frame_ranges(value: Option<&str>) -> anyhow::Result<Vec<FrameIdRange>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for raw_token in value.split(',') {
        let token = raw_token.trim();
        if token.is_empty() {
            continue;
        }
        let (start, end) = if let Some((start, end)) = token.split_once("..") {
            (start.trim(), end.trim())
        } else if let Some((start, end)) = token.split_once('-') {
            (start.trim(), end.trim())
        } else {
            (token, token)
        };
        if start.is_empty() || end.is_empty() {
            bail!("frame range '{token}' must be <frame_id> or <start>-<end>");
        }
        let start = start
            .parse::<u64>()
            .with_context(|| format!("invalid frame range start in '{token}'"))?;
        let end = end
            .parse::<u64>()
            .with_context(|| format!("invalid frame range end in '{token}'"))?;
        if start > end {
            bail!("frame range '{token}' has start greater than end");
        }
        ranges.push(FrameIdRange { start, end });
    }
    Ok(ranges)
}

fn filter_dataset_by_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    excluded_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if excluded_ranges.is_empty() {
        return Ok(dataset);
    }

    let original_pose_count = dataset.poses.len();
    let mut filtered = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if excluded_ranges
            .iter()
            .any(|range| range.contains(pose.frame_id))
        {
            continue;
        }
        filtered.add_pose(pose);
    }

    let removed = original_pose_count.saturating_sub(filtered.poses.len());
    log::info!(
        "Applied {label} frame exclusion | removed={} | remaining={}",
        removed,
        filtered.poses.len()
    );
    if filtered.poses.is_empty() {
        bail!("{label} frame exclusion removed all frames");
    }

    Ok(filtered)
}

fn filter_dataset_to_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    included_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if included_ranges.is_empty() {
        return Ok(dataset);
    }

    let original_pose_count = dataset.poses.len();
    let mut filtered = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if included_ranges
            .iter()
            .any(|range| range.contains(pose.frame_id))
        {
            filtered.add_pose(pose);
        }
    }

    log::info!(
        "Applied {label} frame include | kept={} | removed={}",
        filtered.poses.len(),
        original_pose_count.saturating_sub(filtered.poses.len())
    );
    if filtered.poses.is_empty() {
        bail!("{label} frame include selected no frames");
    }

    Ok(filtered)
}

fn oversample_dataset_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    oversample_ranges: &[FrameIdRange],
    repeat: usize,
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if repeat == 0 {
        bail!("--oversample-frame-repeat must be >= 1");
    }
    if oversample_ranges.is_empty() || repeat == 1 {
        return Ok(dataset);
    }

    let matching_poses: Vec<_> = dataset
        .poses
        .iter()
        .filter(|pose| {
            oversample_ranges
                .iter()
                .any(|range| range.contains(pose.frame_id))
        })
        .cloned()
        .collect();
    if matching_poses.is_empty() {
        bail!("{label} frame oversampling selected no frames");
    }

    let original_pose_count = dataset.poses.len();
    let mut augmented = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    augmented.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        augmented.add_pose(pose);
    }
    for _ in 1..repeat {
        for pose in &matching_poses {
            augmented.add_pose(pose.clone());
        }
    }

    log::info!(
        "Applied {label} frame oversampling | matched={} | repeat={} | original={} | augmented={}",
        matching_poses.len(),
        repeat,
        original_pose_count,
        augmented.poses.len()
    );

    Ok(augmented)
}

#[cfg(feature = "gpu")]
pub(super) fn evaluation_dataset_load_params(args: &TrainArgs) -> (usize, usize) {
    // Keep the evaluation prefix trimming, but do not apply frame_stride here.
    // The actual evaluation subset selection should happen once inside evaluate_splats().
    (args.eval_max_frames, 1)
}

#[cfg(feature = "gpu")]
fn final_training_metrics_from_telemetry(
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    metadata: &rustgs::SplatMetadata,
) -> Option<rustgs::FinalTrainingMetrics> {
    training_telemetry.map(|telemetry| rustgs::FinalTrainingMetrics {
        final_loss: telemetry.final_loss.unwrap_or(metadata.final_loss),
        final_step_loss: telemetry.final_step_loss.unwrap_or(metadata.final_loss),
    })
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvalCropRect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[cfg(feature = "gpu")]
fn parse_eval_crop_rect(
    value: Option<&str>,
    render_width: usize,
    render_height: usize,
) -> anyhow::Result<EvalCropRect> {
    let Some(value) = value else {
        return Ok(EvalCropRect {
            x: 0,
            y: 0,
            width: render_width,
            height: render_height,
        });
    };
    let parts = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != 4 {
        bail!("--eval-crop-rect must be x,y,width,height in evaluation pixels");
    }
    let x = parts[0].parse::<usize>()?;
    let y = parts[1].parse::<usize>()?;
    let width = parts[2].parse::<usize>()?;
    let height = parts[3].parse::<usize>()?;
    if width == 0 || height == 0 {
        bail!("--eval-crop-rect width and height must be >= 1");
    }
    if x >= render_width
        || y >= render_height
        || x + width > render_width
        || y + height > render_height
    {
        bail!(
            "--eval-crop-rect {value} exceeds evaluation resolution {}x{}",
            render_width,
            render_height
        );
    }
    Ok(EvalCropRect {
        x,
        y,
        width,
        height,
    })
}

#[cfg(feature = "gpu")]
fn parse_eval_crop_frame_ids(value: Option<&str>) -> anyhow::Result<Option<BTreeSet<u64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let mut ids = BTreeSet::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        ids.insert(token.parse::<u64>()?);
    }
    if ids.is_empty() {
        bail!("--eval-crop-frames must contain at least one frame id");
    }
    Ok(Some(ids))
}

#[cfg(feature = "gpu")]
fn crop_frame_indices(
    dataset: &rustscan_types::TrainingDataset,
    summary: &rustgs::SplatEvaluationSummary,
    requested_frame_ids: Option<&BTreeSet<u64>>,
) -> anyhow::Result<Vec<usize>> {
    if let Some(requested_frame_ids) = requested_frame_ids {
        let mut found = BTreeSet::new();
        let indices = dataset
            .poses
            .iter()
            .enumerate()
            .filter_map(|(idx, pose)| {
                requested_frame_ids.contains(&pose.frame_id).then(|| {
                    found.insert(pose.frame_id);
                    idx
                })
            })
            .collect::<Vec<_>>();
        if found.len() != requested_frame_ids.len() {
            let missing = requested_frame_ids
                .difference(&found)
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            bail!("requested --eval-crop-frames were not in the evaluated frame subset: {missing}");
        }
        return Ok(indices);
    }

    let mut deduped = BTreeSet::new();
    for frame in &summary.worst_frames {
        if frame.dataset_index < dataset.poses.len() {
            deduped.insert(frame.dataset_index);
        }
    }
    Ok(deduped.into_iter().collect())
}

#[cfg(feature = "gpu")]
fn save_rgb_crop(
    path: &Path,
    data: &[f32],
    render_width: usize,
    rect: EvalCropRect,
) -> anyhow::Result<()> {
    let mut image = image::RgbImage::new(rect.width as u32, rect.height as u32);
    for crop_y in 0..rect.height {
        let src_y = rect.y + crop_y;
        for crop_x in 0..rect.width {
            let src_x = rect.x + crop_x;
            let base = (src_y * render_width + src_x) * 3;
            let pixel = [
                float_to_u8(data.get(base).copied().unwrap_or_default()),
                float_to_u8(data.get(base + 1).copied().unwrap_or_default()),
                float_to_u8(data.get(base + 2).copied().unwrap_or_default()),
            ];
            image.put_pixel(crop_x as u32, crop_y as u32, image::Rgb(pixel));
        }
    }
    image
        .save(path)
        .with_context(|| format!("failed to save evaluation crop {}", path.display()))
}

#[cfg(feature = "gpu")]
fn save_rgb_crop_strip(
    path: &Path,
    target: &[f32],
    rendered: &[f32],
    diff: &[f32],
    render_width: usize,
    rect: EvalCropRect,
) -> anyhow::Result<()> {
    let mut image = image::RgbImage::new((rect.width * 3) as u32, rect.height as u32);
    for crop_y in 0..rect.height {
        let src_y = rect.y + crop_y;
        for (panel_idx, data) in [target, rendered, diff].iter().enumerate() {
            for crop_x in 0..rect.width {
                let src_x = rect.x + crop_x;
                let src_base = (src_y * render_width + src_x) * 3;
                let dst_x = panel_idx * rect.width + crop_x;
                let pixel = [
                    float_to_u8(data.get(src_base).copied().unwrap_or_default()),
                    float_to_u8(data.get(src_base + 1).copied().unwrap_or_default()),
                    float_to_u8(data.get(src_base + 2).copied().unwrap_or_default()),
                ];
                image.put_pixel(dst_x as u32, crop_y as u32, image::Rgb(pixel));
            }
        }
    }
    image
        .save(path)
        .with_context(|| format!("failed to save evaluation crop strip {}", path.display()))
}

#[cfg(feature = "gpu")]
fn float_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(feature = "gpu")]
fn diff_image(rendered: &[f32], target: &[f32]) -> Vec<f32> {
    rendered
        .iter()
        .zip(target.iter())
        .map(|(rendered, target)| ((rendered - target).abs() * 4.0).clamp(0.0, 1.0))
        .collect()
}

#[cfg(feature = "gpu")]
fn export_evaluation_crops(
    args: &TrainArgs,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    device: &rustgs::EvaluationDevice,
    summary: &rustgs::SplatEvaluationSummary,
) -> anyhow::Result<Vec<PathBuf>> {
    let Some(output_dir) = args.eval_crop_output_dir.as_ref() else {
        return Ok(Vec::new());
    };
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create crop output dir {}", output_dir.display()))?;
    let selected =
        rustgs::select_evaluation_frames(dataset, args.eval_max_frames, args.eval_frame_stride);
    let requested_frame_ids = parse_eval_crop_frame_ids(args.eval_crop_frames.as_deref())?;
    let frame_indices = crop_frame_indices(&selected, summary, requested_frame_ids.as_ref())?;
    let rect = parse_eval_crop_rect(
        args.eval_crop_rect.as_deref(),
        summary.render_width,
        summary.render_height,
    )?;

    let runtime_splats =
        rustgs::runtime_from_splats(splats, device).map_err(anyhow::Error::from)?;
    let mut renderer = rustgs::SplatEvaluationRenderer::new(
        summary.render_width,
        summary.render_height,
        *device,
        summary.raster_cov_blur,
    )
    .map_err(anyhow::Error::from)?;
    let mut outputs = Vec::new();

    for idx in frame_indices {
        let pose = selected.poses.get(idx).with_context(|| {
            format!("evaluated frame index {idx} was not found for crop export")
        })?;
        let (target, rendered) = rustgs::render_evaluation_frame(
            &selected,
            pose,
            summary.render_width,
            summary.render_height,
            device,
            &runtime_splats,
            &mut renderer,
        )
        .map_err(anyhow::Error::from)?;
        let diff = diff_image(&rendered, &target);
        let base_name = format!("frame_{:06}_idx_{:04}", pose.frame_id, idx);
        for (kind, data) in [
            ("target", target.as_slice()),
            ("render", rendered.as_slice()),
            ("diff_x4", diff.as_slice()),
        ] {
            let path = output_dir.join(format!("{base_name}_{kind}.png"));
            save_rgb_crop(&path, data, summary.render_width, rect)?;
            outputs.push(path);
        }
        let strip_path = output_dir.join(format!("{base_name}_strip.png"));
        save_rgb_crop_strip(
            &strip_path,
            &target,
            &rendered,
            &diff,
            summary.render_width,
            rect,
        )?;
        outputs.push(strip_path);
    }

    Ok(outputs)
}

#[cfg(feature = "gpu")]
fn maybe_evaluate_trained_splats(
    args: &TrainArgs,
    splats: &rustgs::HostSplats,
    metadata: &rustgs::SplatMetadata,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
) -> anyhow::Result<Option<rustgs::SplatEvaluationSummary>> {
    if !args.eval_after_train {
        return Ok(None);
    }
    if args.eval_frame_stride == 0 {
        bail!("--eval-frame-stride must be >= 1");
    }
    if !(0.0625..=1.0).contains(&args.eval_render_scale) {
        bail!("--eval-render-scale must be in [0.0625, 1.0]");
    }
    let eval_raster_cov_blur = args.eval_raster_cov_blur.unwrap_or(args.raster_cov_blur);
    if !eval_raster_cov_blur.is_finite() || eval_raster_cov_blur < 0.0 {
        bail!("--eval-raster-cov-blur must be finite and >= 0");
    }

    let eval_device = args
        .eval_device
        .parse::<rustgs::EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = rustgs::evaluation_device(eval_device).map_err(anyhow::Error::from)?;
    let (dataset_max_frames, dataset_frame_stride) = evaluation_dataset_load_params(args);
    let dataset = load_evaluation_dataset(&args.input, dataset_max_frames, dataset_frame_stride)?;
    let included_eval_ranges = parse_frame_ranges(args.eval_include_frame_ranges.as_deref())?;
    let dataset = filter_dataset_to_frame_ranges(dataset, &included_eval_ranges, "evaluation")?;
    let excluded_eval_ranges = parse_frame_ranges(args.eval_exclude_frame_ranges.as_deref())?;
    let dataset = filter_dataset_by_frame_ranges(dataset, &excluded_eval_ranges, "evaluation")?;
    let mut evaluation = rustgs::evaluate_splats(
        &dataset,
        splats,
        metadata,
        &rustgs::SplatEvaluationConfig {
            render_scale: args.eval_render_scale,
            raster_cov_blur: eval_raster_cov_blur,
            frame_stride: args.eval_frame_stride,
            max_frames: args.eval_max_frames,
            worst_frame_count: args.eval_worst_frames,
        },
        &device,
        final_training_metrics_from_telemetry(training_telemetry, metadata),
    )
    .map_err(anyhow::Error::from)?;

    evaluation.summary.crop_outputs =
        export_evaluation_crops(args, &dataset, splats, &device, &evaluation.summary)?;
    log_splat_evaluation_summary(&evaluation.summary, args.eval_json)?;
    Ok(Some(evaluation.summary))
}

#[cfg(feature = "gpu")]
fn log_splat_evaluation_summary(
    summary: &rustgs::SplatEvaluationSummary,
    emit_json: bool,
) -> anyhow::Result<()> {
    log::info!(
        "Splat evaluation summary | device={} | render_scale={:.3} | raster_cov_blur={:.3} | resolution={}x{} | frames={} | final_loss={:.6} | final_step_loss={:?} | psnr_mean_db={:.4} | psnr_min_db={:.4} | psnr_max_db={:.4} | sharpness_grad_ratio_mean={:.4} | sharpness_lap_ratio_mean={:.4} | elapsed={:.2}s",
        summary.device,
        summary.render_scale,
        summary.raster_cov_blur,
        summary.render_width,
        summary.render_height,
        summary.frame_count,
        summary.final_loss,
        summary.final_step_loss,
        summary.psnr_mean_db,
        summary.psnr_min_db,
        summary.psnr_max_db,
        summary.sharpness_grad_ratio_mean,
        summary.sharpness_lap_ratio_mean,
        summary.elapsed_seconds,
    );
    for (rank, frame) in summary.worst_frames.iter().enumerate() {
        log::info!(
            "Worst evaluated frame | rank={} | dataset_index={} | frame_id={} | psnr_db={:.4} | sharpness_grad_ratio={:.4} | sharpness_lap_ratio={:.4} | image={}",
            rank + 1,
            frame.dataset_index,
            frame.frame_id,
            frame.psnr_db,
            frame.sharpness_grad_ratio,
            frame.sharpness_lap_ratio,
            frame.image_path.display()
        );
    }
    for path in &summary.crop_outputs {
        log::info!("Evaluation crop exported | path={}", path.display());
    }
    if emit_json {
        println!("{}", serde_json::to_string_pretty(summary)?);
    }
    Ok(())
}

pub(super) fn build_training_config(args: &TrainArgs) -> anyhow::Result<rustgs::TrainingConfig> {
    if args.litegs_target_primitives == 0 {
        bail!("--litegs-target-primitives must be >= 1");
    }

    let (split_score_mode, split_grad_threshold, depth_scale_gamma) =
        litegs_profile_overrides(args);
    let config = rustgs::TrainingConfig {
        iterations: args.iterations,
        max_initial_gaussians: args.max_initial_gaussians,
        sampling_step: args.sampling_step,
        frame_shuffle_seed: args.frame_shuffle_seed,
        render_scale: args.render_scale,
        raster_cov_blur: args.raster_cov_blur,
        raster_cov_blur_final: args.raster_cov_blur_final,
        raster_cov_blur_final_after_epoch: args.raster_cov_blur_final_after_epoch,
        lr_position: args.lr_position,
        lr_pos_final: args.lr_position_final,
        lr_decay_iterations: (args.lr_decay_iterations > 0).then_some(args.lr_decay_iterations),
        lr_scale: args.lr_scale,
        lr_scale_final: args.lr_scale_final,
        lr_rotation: args.lr_rotation,
        lr_rotation_final: args.lr_rotation_final,
        lr_opacity: args.lr_opacity,
        lr_opacity_final: args.lr_opacity_final,
        lr_color: args.lr_color,
        lr_color_final: args.lr_color_final,
        loss_l1_weight: args.loss_l1_weight,
        loss_ssim_weight: args.loss_ssim_weight,
        loss_gradient_weight: args.loss_gradient_weight,
        loss_robust_delta: args.loss_robust_delta,
        loss_outlier_threshold: args.loss_outlier_threshold,
        loss_outlier_weight: args.loss_outlier_weight,
        litegs: rustgs::LiteGsConfig {
            sh_degree: args.litegs_sh_degree,
            tile_size: args.litegs_tile_size,
            sparse_grad: args.litegs_sparse_grad,
            reg_weight: args.litegs_reg_weight,
            enable_transmittance: args.litegs_enable_transmittance,
            enable_depth: args.litegs_enable_depth,
            training_profile: args.litegs_profile,
            densify_from: args.litegs_densify_from,
            densify_until: args.litegs_densify_until,
            topology_freeze_after_epoch: args.litegs_topology_freeze_after_epoch,
            growth_freeze_after_epoch: args.litegs_growth_freeze_after_epoch,
            refine_every: args.litegs_refine_every,
            densification_interval: args.litegs_densification_interval,
            growth_grad_threshold: args.litegs_growth_grad_threshold,
            split_score_mode,
            split_grad_threshold,
            depth_scale_gamma,
            growth_select_fraction: args.litegs_growth_select_fraction,
            growth_stop_iter: args.litegs_growth_stop_iter,
            opacity_decay: args.litegs_opacity_decay,
            scale_decay: args.litegs_scale_decay,
            opacity_reset_interval: args.litegs_opacity_reset_interval,
            opacity_reset_mode: args.litegs_opacity_reset_mode,
            prune_mode: args.litegs_prune_mode,
            prune_offset_epochs: args.litegs_prune_offset_epochs,
            prune_min_age: args.litegs_prune_min_age,
            prune_invisible_epochs: args.litegs_prune_invisible_epochs,
            prune_opacity_threshold: args.litegs_prune_opacity_threshold,
            prune_until_epoch: args.litegs_prune_until_epoch,
            target_primitives: args.litegs_target_primitives,
            learnable_viewproj: args.litegs_learnable_viewproj,
            lr_pose: args.litegs_lr_pose,
            prune_scale_threshold: args.litegs_prune_scale_threshold,
        },
        ..rustgs::TrainingConfig::default()
    };
    config.validate()?;

    Ok(config)
}

fn litegs_profile_overrides(args: &TrainArgs) -> (rustgs::LiteGsSplitScoreMode, f32, f32) {
    match args.litegs_profile {
        rustgs::LiteGsTrainingProfile::Baseline => (
            args.litegs_split_score,
            args.litegs_split_grad_threshold,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsSplit => (
            rustgs::LiteGsSplitScoreMode::Abs,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsPixel => (
            rustgs::LiteGsSplitScoreMode::AbsPixel,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsPixelDepth => (
            rustgs::LiteGsSplitScoreMode::AbsPixelDepth,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
    }
}

fn log_litegs_training_config(config: &rustgs::TrainingConfig) {
    log::info!(
        "LiteGS profile config | profile={} | sh_degree={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | learnable_viewproj={} | lr_pose={:.6} | densify_from={} | densify_until={:?} | topology_freeze_after_epoch={:?} | growth_freeze_after_epoch={:?} | refine_every={} | densification_interval={} | growth_grad_threshold={:.6} | split_score={} | split_grad_threshold={:.6} | depth_scale_gamma={:.3} | growth_select_fraction={:.3} | growth_stop_iter={} | opacity_decay={:.6} | scale_decay={:.6} | opacity_reset_interval={} | opacity_reset_mode={} | prune_mode={} | prune_opacity_threshold={:.6} | prune_until_epoch={:?} | target_primitives={} | lr_decay_iterations={:?} | lr_final(scale={:.6}, rot={:.6}, opacity={:.6}, color={:.6}) | raster_cov_blur={:.3} | raster_cov_blur_final={:?} | raster_cov_blur_final_after_epoch={:?} | loss_weights(l1={:.3}, ssim={:.3}, gradient={:.3}, robust_delta={:.3}, outlier_threshold={:.3}, outlier_weight={:.3})",
        config.litegs.training_profile,
        config.litegs.sh_degree,
        config.litegs.tile_size,
        config.litegs.sparse_grad,
        config.litegs.reg_weight,
        config.litegs.enable_transmittance,
        config.litegs.enable_depth,
        config.litegs.learnable_viewproj,
        config.litegs.lr_pose,
        config.litegs.densify_from,
        config.litegs.densify_until,
        config.litegs.topology_freeze_after_epoch,
        config.litegs.growth_freeze_after_epoch,
        config.litegs.refine_every,
        config.litegs.densification_interval,
        config.litegs.growth_grad_threshold,
        config.litegs.split_score_mode,
        config.litegs.split_grad_threshold,
        config.litegs.depth_scale_gamma,
        config.litegs.growth_select_fraction,
        config.litegs.growth_stop_iter,
        config.litegs.opacity_decay,
        config.litegs.scale_decay,
        config.litegs.opacity_reset_interval,
        config.litegs.opacity_reset_mode,
        config.litegs.prune_mode,
        config.litegs.prune_opacity_threshold,
        config.litegs.prune_until_epoch,
        config.litegs.target_primitives,
        config.lr_decay_iterations,
        config.lr_scale_final,
        config.lr_rotation_final,
        config.lr_opacity_final,
        config.lr_color_final,
        config.raster_cov_blur,
        config.raster_cov_blur_final,
        config.raster_cov_blur_final_after_epoch
            .or(config.litegs.topology_freeze_after_epoch),
        config.loss_l1_weight,
        config.loss_ssim_weight,
        config.loss_gradient_weight,
        config.loss_robust_delta,
        config.loss_outlier_threshold,
        config.loss_outlier_weight,
    );
}

fn ensure_sparse_initialization_points(
    dataset: &rustscan_types::TrainingDataset,
    source: rustgs::TrainingInputKind,
    input: &Path,
) -> anyhow::Result<()> {
    if !dataset.initial_points.is_empty() {
        return Ok(());
    }

    let source_hint = match source {
        rustgs::TrainingInputKind::TumRgbd => {
            "raw TUM RGB-D input does not carry COLMAP sparse points; convert it to a COLMAP reconstruction first"
        }
        rustgs::TrainingInputKind::Colmap => {
            "the COLMAP input is missing sparse points3D output; make sure points3D.bin or points3D.txt exists"
        }
        rustgs::TrainingInputKind::TrainingDatasetJson => {
            "the TrainingDataset JSON did not contain any initial_points; export it from a COLMAP sparse reconstruction first"
        }
    };

    bail!(
        "training initialization now requires COLMAP sparse points, but {:?} ({}) provided none: {}",
        input,
        source,
        source_hint
    );
}

pub(super) fn maybe_write_litegs_parity_report(
    input: &Path,
    output: &Path,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_elapsed: Duration,
    evaluation_summary: Option<&rustgs::SplatEvaluationSummary>,
) -> anyhow::Result<()> {
    maybe_write_litegs_parity_report_with_manifest_dir(
        input,
        output,
        dataset,
        splats,
        config,
        training_telemetry,
        training_elapsed,
        evaluation_summary,
        Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

pub(super) fn maybe_write_litegs_parity_report_with_manifest_dir(
    input: &Path,
    output: &Path,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_elapsed: Duration,
    evaluation_summary: Option<&rustgs::SplatEvaluationSummary>,
    manifest_dir: &Path,
) -> anyhow::Result<()> {
    let report_path = rustgs::default_parity_report_path(output);
    let fixture_id = rustgs::parity_fixture_id_for_input_path(input);
    let mut report = rustgs::ParityHarnessReport::new(fixture_id, &config.litegs);

    report.topology.initialization_gaussians =
        inferred_initialization_gaussian_count(dataset, config);
    report.topology.final_gaussians = Some(splats.len());
    report.topology.export_outputs = 1;

    if let Some(telemetry) = training_telemetry {
        report.loss_terms = telemetry.loss_terms.clone();
        report.loss_curve_samples = telemetry.loss_curve_samples.clone();
        report.topology = telemetry.topology.clone();
        report.topology.initialization_gaussians = report
            .topology
            .initialization_gaussians
            .or_else(|| inferred_initialization_gaussian_count(dataset, config));
        report.topology.final_gaussians = report.topology.final_gaussians.or(Some(splats.len()));
        report.topology.export_outputs = 1;
        report.metrics.active_sh_degree = telemetry.active_sh_degree;
        report.metrics.depth_valid_pixels = telemetry.depth_valid_pixels;
        report.metrics.depth_grad_scale = telemetry.depth_grad_scale;
        report.metrics.rotation_frozen = Some(telemetry.rotation_frozen);
    } else {
        report.metrics.active_sh_degree = Some(config.litegs.sh_degree);
    }
    if let Some(summary) = evaluation_summary {
        report.metrics.final_psnr = Some(summary.psnr_mean_db);
        report.notes.push(format!(
            "Evaluation summary recorded with device={}, render_scale={:.3}, raster_cov_blur={:.3}, frame_stride={}, max_frames={}, frame_count={}, mean PSNR {:.4} dB, grad sharpness ratio {:.4}, and lap sharpness ratio {:.4}.",
            summary.device,
            summary.render_scale,
            summary.raster_cov_blur,
            summary.frame_stride,
            summary.max_frames,
            summary.frame_count,
            summary.psnr_mean_db,
            summary.sharpness_grad_ratio_mean,
            summary.sharpness_lap_ratio_mean,
        ));
    }
    report.metrics.had_nan = splats_have_non_finite(splats);
    report.metrics.had_oom = false;

    report.timing.training_ms = Some(training_elapsed.as_millis() as u64);
    report.timing.total_wall_clock_ms = Some(training_elapsed.as_millis() as u64);

    report.notes.push(
        "LiteGsMacV1 now evaluates the active SH degree for view-dependent color during wgpu training and can apply rotation-aware projection gradients when rotation learning is enabled."
            .to_string(),
    );
    if training_telemetry.is_none() {
        report.notes.push(
            "Wgpu training telemetry was unavailable for this run, so the parity report fell back to config-level LiteGS metadata."
                .to_string(),
        );
    }

    let (roundtrip_splats, roundtrip_metadata) = rustgs::load_splats_ply(output)?;
    report.metrics.export_roundtrip_ok = roundtrip_splats.len() == splats.len()
        && roundtrip_metadata.gaussian_count == splats.len()
        && !splats_have_non_finite(&roundtrip_splats);

    if let Some(reference_report_path) =
        resolve_parity_reference_report_path_from_manifest_dir(&report.fixture_id, manifest_dir)
    {
        match rustgs::ParityHarnessReport::load_json(&reference_report_path) {
            Ok(reference_report) => {
                report.metrics.litegs_reference_psnr = reference_report.metrics.final_psnr;
                report.metrics.gaussian_count_delta_ratio = gaussian_count_delta_ratio(
                    report.topology.final_gaussians,
                    reference_report.topology.final_gaussians,
                );
                report.reference_comparison = rustgs::compare_loss_curve_samples(
                    &report.loss_curve_samples,
                    &reference_report.loss_curve_samples,
                );
                report.notes.push(format!(
                    "Compared parity loss curve samples against reference report at {}.",
                    reference_report_path.display()
                ));
            }
            Err(err) => {
                log::warn!(
                    "failed to load LiteGS parity reference report {:?}: {}",
                    reference_report_path,
                    err
                );
            }
        }
    } else if report.fixture_id == rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID {
        report.notes.push(
            "No checked-in LiteGS parity reference report was found for the convergence fixture, so gate evaluation is reference-blocked."
                .to_string(),
        );
    }

    report.gate = Some(report.evaluate_gate());
    report.save_json(&report_path)?;
    if let Some(gate) = report.gate.as_ref() {
        log::info!(
            "Saved LiteGS parity report to {:?} | gate_status={:?}",
            report_path,
            gate.status
        );
    } else {
        log::info!("Saved LiteGS parity report to {:?}", report_path);
    }
    Ok(())
}

fn resolve_parity_reference_report_path_from_manifest_dir(
    fixture_id: &str,
    manifest_dir: &Path,
) -> Option<PathBuf> {
    manifest_dir
        .ancestors()
        .find_map(|path| rustgs::resolve_litegs_parity_reference_report_path(fixture_id, path))
}

fn inferred_initialization_gaussian_count(
    dataset: &rustscan_types::TrainingDataset,
    config: &rustgs::TrainingConfig,
) -> Option<usize> {
    let sparse_points = dataset.initial_points.len();
    if sparse_points == 0 {
        None
    } else {
        Some(sparse_points.min(config.max_initial_gaussians.max(1)))
    }
}

fn gaussian_count_delta_ratio(current: Option<usize>, reference: Option<usize>) -> Option<f32> {
    match (current, reference) {
        (Some(current), Some(reference)) if reference > 0 => {
            Some(((current as f32) - (reference as f32)).abs() / reference as f32)
        }
        _ => None,
    }
}

#[cfg(feature = "gpu")]
fn splats_have_non_finite(splats: &rustgs::HostSplats) -> bool {
    let view = splats.as_view();
    view.positions.iter().any(|value| !value.is_finite())
        || view.log_scales.iter().any(|value| !value.is_finite())
        || view.rotations.iter().any(|value| !value.is_finite())
        || view.opacity_logits.iter().any(|value| !value.is_finite())
        || view.sh_coeffs.iter().any(|value| !value.is_finite())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        ensure_sparse_initialization_points, filter_dataset_by_frame_ranges,
        filter_dataset_to_frame_ranges, oversample_dataset_frame_ranges, parse_frame_ranges,
        run_train_command,
    };
    use crate::TrainArgs;
    use std::path::PathBuf;

    #[test]
    fn run_train_command_surfaces_missing_input_cleanly() {
        let args = TrainArgs {
            input: PathBuf::from("missing-dataset"),
            output: PathBuf::from("scene.ply"),
            iterations: 1,
            max_initial_gaussians: 16,
            sampling_step: 0,
            max_frames: 0,
            frame_stride: 1,
            include_frame_ranges: None,
            exclude_frame_ranges: None,
            oversample_frame_ranges: None,
            oversample_frame_repeat: 1,
            frame_shuffle_seed: 0,
            render_scale: 0.5,
            raster_cov_blur: rustgs::DEFAULT_RASTER_COV_BLUR,
            raster_cov_blur_final: None,
            raster_cov_blur_final_after_epoch: None,
            litegs_sh_degree: 3,
            litegs_profile: rustgs::LiteGsTrainingProfile::Baseline,
            litegs_tile_size: rustgs::LiteGsTileSize::new(8, 16),
            litegs_sparse_grad: false,
            litegs_reg_weight: 0.0,
            litegs_enable_transmittance: false,
            litegs_enable_depth: false,
            litegs_densify_from: 3,
            litegs_densify_until: None,
            litegs_topology_freeze_after_epoch: None,
            litegs_growth_freeze_after_epoch: None,
            litegs_refine_every: 16,
            litegs_densification_interval: 100,
            litegs_growth_grad_threshold: 0.0002,
            litegs_split_score: rustgs::LiteGsSplitScoreMode::Baseline,
            litegs_split_grad_threshold: 0.0002,
            litegs_depth_scale_gamma: 0.37,
            litegs_growth_select_fraction: 0.2,
            litegs_growth_stop_iter: 15_000,
            litegs_opacity_decay: 0.004,
            litegs_scale_decay: 0.002,
            litegs_opacity_reset_interval: 3000,
            litegs_opacity_reset_mode: rustgs::LiteGsOpacityResetMode::Decay,
            litegs_prune_mode: rustgs::LiteGsPruneMode::Weight,
            litegs_prune_offset_epochs: 0,
            litegs_prune_min_age: 5,
            litegs_prune_invisible_epochs: 10,
            litegs_prune_opacity_threshold: 1.0 / 255.0,
            litegs_prune_until_epoch: None,
            litegs_target_primitives: 300_000,
            litegs_learnable_viewproj: false,
            litegs_lr_pose: 0.0001,
            litegs_prune_scale_threshold: 0.5,
            lr_position: 0.00016,
            lr_position_final: 0.0000016,
            lr_decay_iterations: 0,
            lr_scale: 0.005,
            lr_scale_final: 0.0,
            lr_rotation: 0.001,
            lr_rotation_final: 0.0,
            lr_opacity: 0.05,
            lr_opacity_final: 0.0,
            lr_color: 0.0025,
            lr_color_final: 0.0,
            loss_l1_weight: 0.8,
            loss_ssim_weight: 0.2,
            loss_gradient_weight: 0.0,
            loss_robust_delta: 0.0,
            loss_outlier_threshold: 0.0,
            loss_outlier_weight: 1.0,
            log_level: "error".to_string(),
            eval_after_train: false,
            eval_render_scale: 0.25,
            eval_raster_cov_blur: None,
            eval_max_frames: 180,
            eval_frame_stride: 30,
            eval_include_frame_ranges: None,
            eval_exclude_frame_ranges: None,
            eval_worst_frames: 5,
            eval_device: "metal".to_string(),
            eval_json: false,
            eval_crop_output_dir: None,
            eval_crop_frames: None,
            eval_crop_rect: None,
        };

        let err = run_train_command(args).expect_err("missing input should fail");
        assert!(err.to_string().contains("failed to load"));
    }

    #[test]
    fn training_requires_sparse_initialization_points() {
        let dataset =
            rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
        let err = ensure_sparse_initialization_points(
            &dataset,
            rustgs::TrainingInputKind::TumRgbd,
            PathBuf::from("test_data/tum").as_path(),
        )
        .expect_err("dataset without sparse points should fail");

        assert!(
            err.to_string()
                .contains("training initialization now requires COLMAP sparse points"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_frame_ranges_accepts_singletons_and_inclusive_ranges() {
        let ranges = parse_frame_ranges(Some("76-93,155,200..202")).unwrap();

        assert_eq!(ranges.len(), 3);
        assert!(ranges[0].contains(76));
        assert!(ranges[0].contains(93));
        assert!(!ranges[0].contains(94));
        assert!(ranges[1].contains(155));
        assert!(ranges[2].contains(201));
    }

    #[test]
    fn frame_range_filter_removes_matching_frame_ids() {
        let mut dataset =
            rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
        dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
        for frame_id in [75_u64, 76, 80, 93, 94] {
            dataset.add_pose(rustgs::ScenePose::new(
                frame_id,
                PathBuf::from(format!("frame_{frame_id:04}.png")),
                rustgs::SE3::identity(),
                frame_id as f64,
            ));
        }

        let ranges = parse_frame_ranges(Some("76-93")).unwrap();
        let filtered = filter_dataset_by_frame_ranges(dataset, &ranges, "test").unwrap();

        let kept: Vec<u64> = filtered.poses.iter().map(|pose| pose.frame_id).collect();
        assert_eq!(kept, vec![75, 94]);
        assert_eq!(filtered.initial_points.len(), 1);
    }

    #[test]
    fn frame_range_include_keeps_matching_frame_ids() {
        let mut dataset =
            rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
        dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
        for frame_id in [0_u64, 30, 60, 90, 120, 150] {
            dataset.add_pose(rustgs::ScenePose::new(
                frame_id,
                PathBuf::from(format!("frame_{frame_id:04}.png")),
                rustgs::SE3::identity(),
                frame_id as f64,
            ));
        }

        let ranges = parse_frame_ranges(Some("0-60,150")).unwrap();
        let filtered = filter_dataset_to_frame_ranges(dataset, &ranges, "test").unwrap();

        let kept: Vec<u64> = filtered.poses.iter().map(|pose| pose.frame_id).collect();
        assert_eq!(kept, vec![0, 30, 60, 150]);
        assert_eq!(filtered.initial_points.len(), 1);
    }

    #[test]
    fn frame_range_oversampling_repeats_matching_frame_ids() {
        let mut dataset =
            rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
        dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
        for frame_id in [0_u64, 30, 60, 90] {
            dataset.add_pose(rustgs::ScenePose::new(
                frame_id,
                PathBuf::from(format!("frame_{frame_id:04}.png")),
                rustgs::SE3::identity(),
                frame_id as f64,
            ));
        }

        let ranges = parse_frame_ranges(Some("30-60")).unwrap();
        let oversampled = oversample_dataset_frame_ranges(dataset, &ranges, 3, "test").unwrap();

        let kept: Vec<u64> = oversampled.poses.iter().map(|pose| pose.frame_id).collect();
        assert_eq!(kept, vec![0, 30, 60, 90, 30, 60, 30, 60]);
        assert_eq!(oversampled.initial_points.len(), 1);
    }
}
