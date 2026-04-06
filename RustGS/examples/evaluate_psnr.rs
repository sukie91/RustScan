use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use image::{ImageBuffer, RgbImage};
use rustgs::core::GaussianColorRepresentation;
use rustgs::diff::{DiffCamera, DiffSplatRenderer, TrainableGaussians};
use rustgs::{load_scene_ply, load_training_dataset, Gaussian, SceneMetadata, TumRgbdConfig};
use rustscan_types::TrainingDataset;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;

    let dataset = load_training_dataset(&args.dataset, &TumRgbdConfig::default())?;
    let dataset = select_frames(&dataset, args.max_frames, args.frame_stride);
    let (scene, metadata) = load_scene_ply(&args.scene)?;

    let device = if args.device == "metal" {
        Device::new_metal(0)
            .map_err(|err| anyhow::anyhow!("failed to create Metal device: {err}"))?
    } else {
        Device::Cpu
    };

    let (render_width, render_height) = scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        args.render_scale,
    );
    let trainable = trainable_from_scene(&scene, &metadata, &device)?;
    let mut renderer = DiffSplatRenderer::with_device(render_width, render_height, device.clone());

    let start = Instant::now();
    let mut psnrs = Vec::with_capacity(dataset.poses.len());
    let mut frame_metrics = Vec::with_capacity(dataset.poses.len());

    for (idx, pose) in dataset.poses.iter().enumerate() {
        let (target, rendered) = render_frame(
            &dataset,
            pose,
            render_width,
            render_height,
            &device,
            &trainable,
            &mut renderer,
        )?;
        let psnr = compute_psnr_f32(&rendered, &target);
        psnrs.push(psnr);
        frame_metrics.push(FrameMetric {
            dataset_index: idx,
            frame_id: pose.frame_id,
            psnr,
            image_path: pose.image_path.clone(),
        });

        if idx < 5 || (idx + 1) % 25 == 0 || idx + 1 == dataset.poses.len() {
            eprintln!(
                "frame {:>3}/{:>3} id {:>4} | psnr {:>6.2} dB | elapsed {:.1}s",
                idx + 1,
                dataset.poses.len(),
                pose.frame_id,
                psnr,
                start.elapsed().as_secs_f32()
            );
        }
    }

    let summary = summarize(&psnrs);
    println!("scene={}", args.scene.display());
    println!("dataset={}", args.dataset.display());
    println!("device={}", args.device);
    println!(
        "render_scale={:.3} resolution={}x{} frames={} stride={} max_frames={}",
        args.render_scale,
        render_width,
        render_height,
        dataset.poses.len(),
        args.frame_stride,
        args.max_frames
    );
    println!(
        "scene_metadata iterations={} gaussian_count={} final_loss={} color_representation={:?}",
        metadata.iterations,
        metadata.gaussian_count,
        metadata.final_loss,
        metadata.color_representation
    );
    println!(
        "psnr_mean_db={:.4}\npsnr_median_db={:.4}\npsnr_min_db={:.4}\npsnr_max_db={:.4}\npsnr_std_db={:.4}\nelapsed_seconds={:.2}",
        summary.mean,
        summary.median,
        summary.min,
        summary.max,
        summary.stddev,
        start.elapsed().as_secs_f32()
    );

    print_worst_frames(&frame_metrics, args.export_worst_k);
    if args.export_worst_k > 0 {
        let export_dir = args
            .export_dir
            .clone()
            .unwrap_or_else(|| default_export_dir(&args.scene));
        export_worst_frames(
            &dataset,
            &frame_metrics,
            args.export_worst_k,
            render_width,
            render_height,
            &device,
            &trainable,
            &mut renderer,
            &export_dir,
        )?;
        println!("worst_frames_export_dir={}", export_dir.display());
    }

    Ok(())
}

struct Args {
    scene: PathBuf,
    dataset: PathBuf,
    render_scale: f32,
    frame_stride: usize,
    max_frames: usize,
    device: String,
    export_worst_k: usize,
    export_dir: Option<PathBuf>,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut scene = None;
        let mut dataset = None;
        let mut render_scale = 0.5f32;
        let mut frame_stride = 1usize;
        let mut max_frames = 0usize;
        let mut device = String::from("cpu");
        let mut export_worst_k = 0usize;
        let mut export_dir = None;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--scene" => scene = Some(PathBuf::from(next_value(&mut it, "--scene")?)),
                "--dataset" => dataset = Some(PathBuf::from(next_value(&mut it, "--dataset")?)),
                "--render-scale" => {
                    render_scale = next_value(&mut it, "--render-scale")?.parse()?
                }
                "--frame-stride" => {
                    frame_stride = next_value(&mut it, "--frame-stride")?.parse()?
                }
                "--max-frames" => max_frames = next_value(&mut it, "--max-frames")?.parse()?,
                "--device" => device = next_value(&mut it, "--device")?,
                "--export-worst-k" => {
                    export_worst_k = next_value(&mut it, "--export-worst-k")?.parse()?
                }
                "--export-dir" => {
                    export_dir = Some(PathBuf::from(next_value(&mut it, "--export-dir")?))
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(anyhow::anyhow!("unrecognized argument: {other}"));
                }
            }
        }

        let scene = scene.ok_or_else(|| anyhow::anyhow!("missing --scene"))?;
        let dataset = dataset.ok_or_else(|| anyhow::anyhow!("missing --dataset"))?;
        if frame_stride == 0 {
            return Err(anyhow::anyhow!("--frame-stride must be >= 1"));
        }
        if !(0.0625..=1.0).contains(&render_scale) {
            return Err(anyhow::anyhow!("--render-scale must be in [0.0625, 1.0]"));
        }
        if device != "cpu" && device != "metal" {
            return Err(anyhow::anyhow!("--device must be cpu or metal"));
        }

        Ok(Self {
            scene,
            dataset,
            render_scale,
            frame_stride,
            max_frames,
            device,
            export_worst_k,
            export_dir,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

fn print_help() {
    println!(
        "Usage: cargo run --manifest-path RustGS/Cargo.toml --example evaluate_psnr -- \
  --scene <scene.ply> \
  --dataset <training_dataset.json> \
  [--render-scale 0.5] \
  [--frame-stride 1] \
  [--max-frames 0] \
  [--export-worst-k 5] \
  [--export-dir output/psnr_review] \
  [--device cpu|metal]"
    );
}

#[derive(Clone)]
struct FrameMetric {
    dataset_index: usize,
    frame_id: u64,
    psnr: f32,
    image_path: PathBuf,
}

fn select_frames(
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

fn trainable_from_scene(
    scene: &[Gaussian],
    metadata: &SceneMetadata,
    device: &Device,
) -> candle_core::Result<TrainableGaussians> {
    let mut positions = Vec::with_capacity(scene.len() * 3);
    let mut scales = Vec::with_capacity(scene.len() * 3);
    let mut rotations = Vec::with_capacity(scene.len() * 4);
    let mut opacities = Vec::with_capacity(scene.len());
    let mut colors = Vec::with_capacity(scene.len() * 3);

    for gaussian in scene {
        positions.extend_from_slice(&gaussian.position);
        scales.extend_from_slice(&[
            gaussian.scale[0].max(1e-6).ln(),
            gaussian.scale[1].max(1e-6).ln(),
            gaussian.scale[2].max(1e-6).ln(),
        ]);
        rotations.extend_from_slice(&gaussian.rotation);
        opacities.push(opacity_to_logit(gaussian.opacity));
        colors.extend_from_slice(
            &gaussian
                .sh_dc
                .unwrap_or(gaussian.color),
        );
    }

    match metadata.color_representation {
        GaussianColorRepresentation::Rgb => {
            TrainableGaussians::new(&positions, &scales, &rotations, &opacities, &colors, device)
        }
        GaussianColorRepresentation::SphericalHarmonics { degree } => {
            let sh_rest_coeff_count = (degree + 1).saturating_mul(degree + 1).saturating_sub(1);
            let mut sh_rest = Vec::with_capacity(scene.len() * sh_rest_coeff_count * 3);
            for (idx, gaussian) in scene.iter().enumerate() {
                let coeffs = gaussian.sh_rest.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "gaussian {idx} is missing sh_rest for SH degree {degree}"
                    ))
                })?;
                let expected = sh_rest_coeff_count * 3;
                if coeffs.len() != expected {
                    return Err(candle_core::Error::Msg(format!(
                        "gaussian {idx} has {} SH-rest values, expected {expected} for degree {degree}",
                        coeffs.len()
                    )));
                }
                sh_rest.extend_from_slice(coeffs);
            }
            TrainableGaussians::new_with_sh(
                &positions,
                &scales,
                &rotations,
                &opacities,
                &colors,
                &sh_rest,
                degree,
                device,
            )
        }
    }
}

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

fn scaled_dimensions(width: usize, height: usize, render_scale: f32) -> (usize, usize) {
    let scale = render_scale.clamp(0.0625, 1.0);
    let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
    let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
    (scaled_width, scaled_height)
}

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

fn render_frame(
    dataset: &TrainingDataset,
    pose: &rustscan_types::ScenePose,
    render_width: usize,
    render_height: usize,
    device: &Device,
    trainable: &TrainableGaussians,
    renderer: &mut DiffSplatRenderer,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
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

fn load_resized_target(
    path: &std::path::Path,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> anyhow::Result<Vec<f32>> {
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let rgb = image.to_rgb8();
    let (actual_width, actual_height) = rgb.dimensions();
    if actual_width as usize != src_width || actual_height as usize != src_height {
        return Err(anyhow::anyhow!(
            "image {} has size {}x{}, expected {}x{}",
            path.display(),
            actual_width,
            actual_height,
            src_width,
            src_height
        ));
    }
    let src: Vec<f32> = rgb
        .into_raw()
        .into_iter()
        .map(|v| v as f32 / 255.0)
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

fn compute_psnr_f32(rendered: &[f32], target: &[f32]) -> f32 {
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

fn default_export_dir(scene: &std::path::Path) -> PathBuf {
    let parent = scene
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = scene
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("scene");
    parent.join(format!("{stem}_psnr_review"))
}

fn print_worst_frames(frame_metrics: &[FrameMetric], count: usize) {
    let mut sorted = frame_metrics.to_vec();
    sorted.sort_by(|a, b| {
        a.psnr
            .partial_cmp(&b.psnr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let count = count.max(5).min(sorted.len());
    for (rank, metric) in sorted.into_iter().take(count).enumerate() {
        println!(
            "worst_frame rank={} dataset_index={} frame_id={} psnr_db={:.4} image={}",
            rank + 1,
            metric.dataset_index,
            metric.frame_id,
            metric.psnr,
            metric.image_path.display()
        );
    }
}

fn export_worst_frames(
    dataset: &TrainingDataset,
    frame_metrics: &[FrameMetric],
    export_worst_k: usize,
    render_width: usize,
    render_height: usize,
    device: &Device,
    trainable: &TrainableGaussians,
    renderer: &mut DiffSplatRenderer,
    export_dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(export_dir)?;

    let mut sorted = frame_metrics.to_vec();
    sorted.sort_by(|a, b| {
        a.psnr
            .partial_cmp(&b.psnr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst = sorted
        .into_iter()
        .take(export_worst_k.min(frame_metrics.len()));

    let mut summary = String::from("rank\tdataset_index\tframe_id\tpsnr_db\timage_path\n");
    for (rank, metric) in worst.enumerate() {
        let pose = &dataset.poses[metric.dataset_index];
        let (target, rendered) = render_frame(
            dataset,
            pose,
            render_width,
            render_height,
            device,
            trainable,
            renderer,
        )?;
        let target_u8 = rgb_f32_to_u8(&target);
        let rendered_u8 = rgb_f32_to_u8(&rendered);
        let diff_u8 = diff_visualization(&target, &rendered);
        let strip_u8 = make_strip(
            &target_u8,
            &rendered_u8,
            &diff_u8,
            render_width,
            render_height,
        );

        let prefix = format!(
            "rank_{:02}_frame_{:04}_psnr_{:.2}",
            rank + 1,
            metric.frame_id,
            metric.psnr
        );
        save_rgb_png(
            export_dir.join(format!("{prefix}_gt.png")),
            render_width,
            render_height,
            &target_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_render.png")),
            render_width,
            render_height,
            &rendered_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_diff.png")),
            render_width,
            render_height,
            &diff_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_strip.png")),
            render_width * 3,
            render_height,
            &strip_u8,
        )?;
        summary.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{}\n",
            rank + 1,
            metric.dataset_index,
            metric.frame_id,
            metric.psnr,
            metric.image_path.display()
        ));
    }

    std::fs::write(export_dir.join("summary.tsv"), summary)?;
    Ok(())
}

fn rgb_f32_to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

fn diff_visualization(target: &[f32], rendered: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(target.len());
    for (target_px, rendered_px) in target.chunks_exact(3).zip(rendered.chunks_exact(3)) {
        let abs_mean = ((target_px[0] - rendered_px[0]).abs()
            + (target_px[1] - rendered_px[1]).abs()
            + (target_px[2] - rendered_px[2]).abs())
            / 3.0;
        let heat = (abs_mean * 4.0).clamp(0.0, 1.0);
        out.push((heat * 255.0).round() as u8);
        out.push((heat * 64.0).round() as u8);
        out.push(0);
    }
    out
}

fn make_strip(left: &[u8], middle: &[u8], right: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_bytes = width * 3;
    let mut out = vec![0u8; width * 3 * height * 3];
    for y in 0..height {
        let src_start = y * row_bytes;
        let dst_start = y * row_bytes * 3;
        out[dst_start..dst_start + row_bytes]
            .copy_from_slice(&left[src_start..src_start + row_bytes]);
        out[dst_start + row_bytes..dst_start + row_bytes * 2]
            .copy_from_slice(&middle[src_start..src_start + row_bytes]);
        out[dst_start + row_bytes * 2..dst_start + row_bytes * 3]
            .copy_from_slice(&right[src_start..src_start + row_bytes]);
    }
    out
}

fn save_rgb_png(path: PathBuf, width: usize, height: usize, data: &[u8]) -> anyhow::Result<()> {
    let image: RgbImage = ImageBuffer::from_raw(width as u32, height as u32, data.to_vec())
        .ok_or_else(|| {
            anyhow::anyhow!("failed to construct image buffer for {}", path.display())
        })?;
    image.save(&path)?;
    Ok(())
}

struct Summary {
    mean: f32,
    median: f32,
    min: f32,
    max: f32,
    stddev: f32,
}

fn summarize(values: &[f32]) -> Summary {
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

    Summary {
        mean,
        median,
        min,
        max,
        stddev: variance.sqrt(),
    }
}
