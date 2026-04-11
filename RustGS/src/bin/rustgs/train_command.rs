use crate::TrainArgs;
use anyhow::bail;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(feature = "gpu")]
#[derive(Debug, Default)]
struct CliTrainingEventRecorder {
    plan: Option<rustgs::TrainingPlanSelected>,
    completed_chunks: usize,
    final_merged_gaussians: Option<usize>,
}

#[cfg(feature = "gpu")]
impl CliTrainingEventRecorder {
    fn record(&mut self, event: rustgs::TrainingEvent) {
        match event {
            rustgs::TrainingEvent::PlanSelected(plan) => {
                self.plan = Some(plan);
            }
            rustgs::TrainingEvent::ChunkCompleted(chunk) => {
                self.completed_chunks += 1;
                self.final_merged_gaussians = Some(chunk.merged_gaussian_count);
            }
            _ => {}
        }
    }

    fn log_summary(&self, report: &rustgs::TrainingRunReport) {
        let route = self
            .plan
            .as_ref()
            .map(|plan| training_route_label(plan.route))
            .unwrap_or("unknown");
        let estimate = self.plan.as_ref().and_then(|plan| plan.estimate.as_ref());
        let training_chunks = self
            .plan
            .as_ref()
            .and_then(|plan| plan.training_chunks)
            .unwrap_or(0);

        match estimate {
            Some(estimate) if training_chunks > 0 => log::info!(
                "CLI training summary | route={} | completed_chunks={}/{} | planner_requested_gaussians={} | planner_affordable_gaussians={} | planner_peak_gib≈{:.1} | planner_budget_gib≈{:.1} | final_gaussians={} | elapsed={:.2}s",
                route,
                self.completed_chunks,
                training_chunks,
                estimate.requested_initial_gaussians,
                estimate.affordable_initial_gaussians,
                estimate.estimated_peak_gib,
                estimate.effective_budget_gib,
                report.gaussian_count,
                report.elapsed.as_secs_f64(),
            ),
            Some(estimate) => log::info!(
                "CLI training summary | route={} | planner_requested_gaussians={} | planner_affordable_gaussians={} | planner_peak_gib≈{:.1} | planner_budget_gib≈{:.1} | final_gaussians={} | elapsed={:.2}s",
                route,
                estimate.requested_initial_gaussians,
                estimate.affordable_initial_gaussians,
                estimate.estimated_peak_gib,
                estimate.effective_budget_gib,
                report.gaussian_count,
                report.elapsed.as_secs_f64(),
            ),
            None if training_chunks > 0 => log::info!(
                "CLI training summary | route={} | completed_chunks={}/{} | final_gaussians={} | elapsed={:.2}s",
                route,
                self.completed_chunks,
                training_chunks,
                report.gaussian_count,
                report.elapsed.as_secs_f64(),
            ),
            None => log::info!(
                "CLI training summary | route={} | final_gaussians={} | elapsed={:.2}s",
                route,
                report.gaussian_count,
                report.elapsed.as_secs_f64(),
            ),
        }
    }
}

#[cfg(feature = "gpu")]
fn training_route_label(route: rustgs::TrainingEventRoute) -> &'static str {
    match route {
        rustgs::TrainingEventRoute::Standard => "standard",
        rustgs::TrainingEventRoute::ChunkedSingleChunk => "chunked-single",
        rustgs::TrainingEventRoute::ChunkedSequential => "chunked-sequential",
    }
}

#[cfg(feature = "gpu")]
pub(super) fn run_train_command(args: TrainArgs) -> anyhow::Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();

    log::info!("Training 3DGS splats from {:?}", args.input);
    log::info!("Output: {:?}", args.output);
    log::info!("Iterations: {}", args.iterations);
    log::info!("Backend: metal");
    log::info!("Training profile: {}", args.training_profile);

    let dataset =
        load_training_dataset_for_training(&args.input, args.max_frames, args.frame_stride)?;
    log::info!(
        "Loaded {} poses, {} initialization points",
        dataset.poses.len(),
        dataset.initial_points.len()
    );

    let config = build_training_config(&args)?;
    validate_chunked_training_args(&config)?;
    log_chunked_training_config(&config);
    log_litegs_training_config(&config);

    let mut event_recorder = CliTrainingEventRecorder::default();
    let training_run = rustgs::train_splats_with_events(&dataset, &config, |event| {
        event_recorder.record(event);
    })?;
    let rustgs::TrainingRun {
        splats,
        report: training_report,
    } = training_run;
    let training_telemetry = training_report.telemetry.as_ref();

    event_recorder.log_summary(&training_report);
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
) -> anyhow::Result<rustscan_types::TrainingDataset> {
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

    Ok(dataset)
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

    let mut config = rustgs::TrainingConfig::default();
    config.training_profile = args.training_profile;
    config.iterations = args.iterations;
    config.max_initial_gaussians = args.max_initial_gaussians;
    config.sampling_step = args.sampling_step;
    config.metal_render_scale = args.metal_render_scale;
    config.metal_gaussian_chunk_size = args.metal_gaussian_chunk_size;
    config.metal_profile_steps = args.metal_profile_steps;
    config.metal_profile_interval = args.metal_profile_interval;
    config.prune_interval = args.prune_interval;
    config.topology_warmup = args.topology_warmup;
    config.topology_log_interval = args.topology_log_interval;
    config.metal_use_native_forward = !args.metal_disable_native_forward;
    config.chunked_training = args.chunked_training;
    config.chunk_budget_gb = args.chunk_budget_gb;
    config.chunk_overlap_ratio = args.chunk_overlap_ratio;
    config.min_cameras_per_chunk = args.min_cameras_per_chunk;
    config.max_chunks = args.max_chunks;
    config.lr_position = args.lr_position;
    config.lr_pos_final = args.lr_position_final;
    config.lr_scale = args.lr_scale;
    config.lr_rotation = args.lr_rotation;
    config.lr_opacity = args.lr_opacity;
    config.lr_color = args.lr_color;
    config.merge_core_only = if args.no_merge_core_only {
        false
    } else if args.merge_core_only {
        true
    } else {
        true
    };
    config.chunk_artifact_dir = if config.chunked_training {
        Some(default_chunk_artifact_dir(&args.output))
    } else {
        None
    };
    config.litegs = rustgs::LiteGsConfig {
        sh_degree: args.litegs_sh_degree,
        cluster_size: args.litegs_cluster_size,
        tile_size: args.litegs_tile_size,
        sparse_grad: args.litegs_sparse_grad,
        reg_weight: args.litegs_reg_weight,
        enable_transmittance: args.litegs_enable_transmittance,
        enable_depth: args.litegs_enable_depth,
        densify_from: args.litegs_densify_from,
        densify_until: args.litegs_densify_until,
        topology_freeze_after_epoch: args.litegs_topology_freeze_after_epoch,
        refine_every: args.litegs_refine_every,
        densification_interval: args.litegs_densification_interval,
        growth_grad_threshold: args.litegs_growth_grad_threshold,
        growth_select_fraction: args.litegs_growth_select_fraction,
        growth_stop_iter: args.litegs_growth_stop_iter,
        opacity_reset_interval: args.litegs_opacity_reset_interval,
        opacity_reset_mode: args.litegs_opacity_reset_mode,
        prune_mode: args.litegs_prune_mode,
        prune_offset_epochs: args.litegs_prune_offset_epochs,
        prune_min_age: args.litegs_prune_min_age,
        prune_invisible_epochs: args.litegs_prune_invisible_epochs,
        target_primitives: args.litegs_target_primitives,
        learnable_viewproj: args.litegs_learnable_viewproj,
        lr_pose: args.litegs_lr_pose,
        morton_sort_on_densify: args.litegs_morton_sort_on_densify,
        prune_scale_threshold: args.litegs_prune_scale_threshold,
    };

    Ok(config)
}

pub(super) fn validate_chunked_training_args(
    config: &rustgs::TrainingConfig,
) -> anyhow::Result<()> {
    if !config.chunked_training {
        return Ok(());
    }

    if config.chunk_budget_gb <= 0.0 {
        bail!(
            "--chunk-budget-gb must be > 0, got {}",
            config.chunk_budget_gb
        );
    }
    if !(0.0..0.5).contains(&config.chunk_overlap_ratio) {
        bail!(
            "--chunk-overlap-ratio must be in [0.0, 0.5), got {}",
            config.chunk_overlap_ratio
        );
    }
    if config.min_cameras_per_chunk == 0 {
        bail!("--min-cameras-per-chunk must be >= 1");
    }
    if config.max_chunks > 0 && config.max_chunks < 2 {
        bail!("--max-chunks must be 0 (auto) or >= 2 when --chunked-training is enabled");
    }

    Ok(())
}

fn log_chunked_training_config(config: &rustgs::TrainingConfig) {
    if !config.chunked_training {
        return;
    }

    let max_chunks = if config.max_chunks == 0 {
        "auto".to_string()
    } else {
        config.max_chunks.to_string()
    };

    log::info!(
        "Chunked training enabled | budget_gb={:.2} | overlap={:.2} | min_cameras={} | max_chunks={} | merge_core_only={} | artifact_dir={}",
        config.chunk_budget_gb,
        config.chunk_overlap_ratio,
        config.min_cameras_per_chunk,
        max_chunks,
        config.merge_core_only,
        config
            .chunk_artifact_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<disabled>".to_string()),
    );
}

fn log_litegs_training_config(config: &rustgs::TrainingConfig) {
    if config.training_profile != rustgs::TrainingProfile::LiteGsMacV1 {
        return;
    }

    log::info!(
        "LiteGS profile config | sh_degree={} | cluster_size={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | learnable_viewproj={} | lr_pose={:.6} | densify_from={} | densify_until={:?} | topology_freeze_after_epoch={:?} | refine_every={} | densification_interval={} | growth_grad_threshold={:.6} | growth_select_fraction={:.3} | growth_stop_iter={} | opacity_reset_interval={} | opacity_reset_mode={} | prune_mode={} | target_primitives={}",
        config.litegs.sh_degree,
        config.litegs.cluster_size,
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
        config.litegs.refine_every,
        config.litegs.densification_interval,
        config.litegs.growth_grad_threshold,
        config.litegs.growth_select_fraction,
        config.litegs.growth_stop_iter,
        config.litegs.opacity_reset_interval,
        config.litegs.opacity_reset_mode,
        config.litegs.prune_mode,
        config.litegs.target_primitives,
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
    if config.training_profile != rustgs::TrainingProfile::LiteGsMacV1 {
        return Ok(());
    }

    let report_path = rustgs::default_parity_report_path(output);
    let fixture_id = rustgs::parity_fixture_id_for_input_path(input);
    let mut report =
        rustgs::ParityHarnessReport::new(fixture_id, config.training_profile, &config.litegs);

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

    if dataset.initial_points.is_empty() {
        report.notes.push(
            "Sparse COLMAP-style points were unavailable, so initialization-count parity is approximate and frame-based fallback was used.".to_string(),
        );
    }
    report.notes.push(
        "LiteGsMacV1 now evaluates the active SH degree for view-dependent color during Metal training and can apply rotation-aware projection gradients when rotation learning is enabled."
            .to_string(),
    );
    if training_telemetry.is_none() {
        report.notes.push(
            "Metal training telemetry was unavailable for this run, so the parity report fell back to config-level LiteGS metadata."
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

pub(super) fn default_chunk_artifact_dir(output: &Path) -> PathBuf {
    let parent = output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("scene");
    parent.join(format!("{stem}-chunks"))
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::CliTrainingEventRecorder;

    #[test]
    fn event_recorder_tracks_plan_and_chunk_completion() {
        let mut recorder = CliTrainingEventRecorder::default();

        recorder.record(rustgs::TrainingEvent::PlanSelected(
            rustgs::TrainingPlanSelected {
                route: rustgs::TrainingEventRoute::ChunkedSequential,
                training_chunks: Some(2),
                estimate: Some(rustgs::TrainingPlanEstimate {
                    requested_initial_gaussians: 10_000,
                    affordable_initial_gaussians: 4_096,
                    estimated_peak_gib: 8.5,
                    effective_budget_gib: 6.0,
                }),
            },
        ));
        recorder.record(rustgs::TrainingEvent::ChunkCompleted(
            rustgs::TrainingChunkCompleted {
                chunk_index: 1,
                total_chunks: 2,
                chunk_id: 0,
                chunk_gaussian_count: 512,
                merged_gaussian_count: 512,
            },
        ));
        recorder.record(rustgs::TrainingEvent::ChunkCompleted(
            rustgs::TrainingChunkCompleted {
                chunk_index: 2,
                total_chunks: 2,
                chunk_id: 1,
                chunk_gaussian_count: 480,
                merged_gaussian_count: 992,
            },
        ));

        assert_eq!(recorder.completed_chunks, 2);
        assert_eq!(recorder.final_merged_gaussians, Some(992));
        let plan = recorder.plan.expect("plan should be recorded");
        assert_eq!(plan.route, rustgs::TrainingEventRoute::ChunkedSequential);
        assert_eq!(plan.training_chunks, Some(2));
        assert_eq!(
            plan.estimate
                .expect("estimate should be recorded")
                .affordable_initial_gaussians,
            4_096
        );
    }
}
