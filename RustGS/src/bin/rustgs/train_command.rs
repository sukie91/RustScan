#![allow(clippy::too_many_arguments)]

use crate::TrainArgs;
use anyhow::bail;
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

    let eval_device = args
        .eval_device
        .parse::<rustgs::EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = rustgs::evaluation_device(eval_device).map_err(anyhow::Error::from)?;
    let (dataset_max_frames, dataset_frame_stride) = evaluation_dataset_load_params(args);
    let dataset = load_evaluation_dataset(&args.input, dataset_max_frames, dataset_frame_stride)?;
    let evaluation = rustgs::evaluate_splats(
        &dataset,
        splats,
        metadata,
        &rustgs::SplatEvaluationConfig {
            render_scale: args.eval_render_scale,
            frame_stride: args.eval_frame_stride,
            max_frames: args.eval_max_frames,
            worst_frame_count: args.eval_worst_frames,
        },
        &device,
        final_training_metrics_from_telemetry(training_telemetry, metadata),
    )
    .map_err(anyhow::Error::from)?;

    log_splat_evaluation_summary(&evaluation.summary, args.eval_json)?;
    Ok(Some(evaluation.summary))
}

#[cfg(feature = "gpu")]
fn log_splat_evaluation_summary(
    summary: &rustgs::SplatEvaluationSummary,
    emit_json: bool,
) -> anyhow::Result<()> {
    log::info!(
        "Splat evaluation summary | device={} | render_scale={:.3} | resolution={}x{} | frames={} | final_loss={:.6} | final_step_loss={:?} | psnr_mean_db={:.4} | psnr_min_db={:.4} | psnr_max_db={:.4} | elapsed={:.2}s",
        summary.device,
        summary.render_scale,
        summary.render_width,
        summary.render_height,
        summary.frame_count,
        summary.final_loss,
        summary.final_step_loss,
        summary.psnr_mean_db,
        summary.psnr_min_db,
        summary.psnr_max_db,
        summary.elapsed_seconds,
    );
    for (rank, frame) in summary.worst_frames.iter().enumerate() {
        log::info!(
            "Worst evaluated frame | rank={} | dataset_index={} | frame_id={} | psnr_db={:.4} | image={}",
            rank + 1,
            frame.dataset_index,
            frame.frame_id,
            frame.psnr_db,
            frame.image_path.display()
        );
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

    let config = rustgs::TrainingConfig {
        iterations: args.iterations,
        max_initial_gaussians: args.max_initial_gaussians,
        sampling_step: args.sampling_step,
        frame_shuffle_seed: args.frame_shuffle_seed,
        render_scale: args.render_scale,
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
        litegs: rustgs::LiteGsConfig {
            sh_degree: args.litegs_sh_degree,
            tile_size: args.litegs_tile_size,
            sparse_grad: args.litegs_sparse_grad,
            reg_weight: args.litegs_reg_weight,
            enable_transmittance: args.litegs_enable_transmittance,
            enable_depth: args.litegs_enable_depth,
            densify_from: args.litegs_densify_from,
            densify_until: args.litegs_densify_until,
            topology_freeze_after_epoch: args.litegs_topology_freeze_after_epoch,
            growth_freeze_after_epoch: args.litegs_growth_freeze_after_epoch,
            refine_every: args.litegs_refine_every,
            densification_interval: args.litegs_densification_interval,
            growth_grad_threshold: args.litegs_growth_grad_threshold,
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

fn log_litegs_training_config(config: &rustgs::TrainingConfig) {
    log::info!(
        "LiteGS profile config | sh_degree={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | learnable_viewproj={} | lr_pose={:.6} | densify_from={} | densify_until={:?} | topology_freeze_after_epoch={:?} | growth_freeze_after_epoch={:?} | refine_every={} | densification_interval={} | growth_grad_threshold={:.6} | growth_select_fraction={:.3} | growth_stop_iter={} | opacity_decay={:.6} | scale_decay={:.6} | opacity_reset_interval={} | opacity_reset_mode={} | prune_mode={} | prune_opacity_threshold={:.6} | prune_until_epoch={:?} | target_primitives={} | lr_decay_iterations={:?} | lr_final(scale={:.6}, rot={:.6}, opacity={:.6}, color={:.6})",
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
            "Evaluation summary recorded with device={}, render_scale={:.3}, frame_stride={}, max_frames={}, frame_count={} and mean PSNR {:.4} dB.",
            summary.device,
            summary.render_scale,
            summary.frame_stride,
            summary.max_frames,
            summary.frame_count,
            summary.psnr_mean_db,
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
    use super::{ensure_sparse_initialization_points, run_train_command};
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
            frame_shuffle_seed: 0,
            render_scale: 0.5,
            litegs_sh_degree: 3,
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
            log_level: "error".to_string(),
            eval_after_train: false,
            eval_render_scale: 0.25,
            eval_max_frames: 180,
            eval_frame_stride: 30,
            eval_worst_frames: 5,
            eval_device: "metal".to_string(),
            eval_json: false,
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
}
