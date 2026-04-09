//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <slam_output.json|training_dataset.json|tum_dataset_dir|colmap_dir> --output <scene.ply>
//!   rustgs render --input <scene.ply> --camera <pose.json> --output <image.png>

use anyhow::bail;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, clap::Parser)]
#[command(name = "rustgs")]
#[command(about = "3D Gaussian Splatting Training", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, clap::Args)]
struct TrainArgs {
    /// Path to SLAM output JSON, TrainingDataset JSON, TUM RGB-D directory, or COLMAP directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for trained scene (PLY)
    #[arg(short, long)]
    output: PathBuf,

    /// Number of training iterations
    #[arg(long, default_value = "30000")]
    iterations: usize,

    /// Maximum number of Gaussians created during initialization
    #[arg(long, default_value = "100000")]
    max_initial_gaussians: usize,

    /// Pixel sampling step for initialization (0 = auto)
    #[arg(long, default_value = "0")]
    sampling_step: usize,

    /// Maximum number of input RGB frames to consider before applying --frame-stride (0 = all)
    #[arg(long, default_value = "0")]
    max_frames: usize,

    /// Keep every Nth frame within the considered prefix, e.g. 25 frames with stride 5 yields 5 training frames
    #[arg(long, default_value = "1")]
    frame_stride: usize,

    /// Relative render scale used by Metal training
    #[arg(long, default_value = "0.5")]
    metal_render_scale: f32,

    /// Number of Gaussians processed per GPU chunk during Metal training
    #[arg(long, default_value = "32")]
    metal_gaussian_chunk_size: usize,

    /// Emit per-step timing breakdowns for Metal training
    #[arg(long, default_value_t = false)]
    metal_profile_steps: bool,

    /// Log the Metal timing breakdown every N steps when profiling is enabled
    #[arg(long, default_value = "25")]
    metal_profile_interval: usize,

    /// Run prune scheduling every N iterations
    #[arg(long, default_value = "100")]
    prune_interval: usize,

    /// Delay densify/prune topology work until after this many iterations
    #[arg(long, default_value = "100")]
    topology_warmup: usize,

    /// Log topology scheduling/throughput diagnostics every N checks
    #[arg(long, default_value = "500")]
    topology_log_interval: usize,

    /// Training profile to execute
    #[arg(long, default_value_t = rustgs::TrainingProfile::LegacyMetal)]
    training_profile: rustgs::TrainingProfile,

    /// LiteGS SH degree
    #[arg(long, default_value = "3")]
    litegs_sh_degree: usize,

    /// LiteGS cluster size (0 disables clustering for Mac V1 bootstrap)
    #[arg(long, default_value = "0")]
    litegs_cluster_size: usize,

    /// LiteGS tile size, e.g. 8x16 or 8,16
    #[arg(long, default_value_t = rustgs::LiteGsTileSize::new(8, 16))]
    litegs_tile_size: rustgs::LiteGsTileSize,

    /// Enable LiteGS sparse-gradient optimizer semantics
    #[arg(long, default_value_t = false)]
    litegs_sparse_grad: bool,

    /// LiteGS scale regularization weight
    #[arg(long, default_value = "0.0")]
    litegs_reg_weight: f32,

    /// Enable LiteGS transmittance penalty
    #[arg(long, default_value_t = false)]
    litegs_enable_transmittance: bool,

    /// Enable LiteGS depth supervision path
    #[arg(long, default_value_t = false)]
    litegs_enable_depth: bool,

    /// LiteGS densify start epoch
    #[arg(long, default_value = "3")]
    litegs_densify_from: usize,

    /// LiteGS densify end epoch (omit to keep auto behavior)
    #[arg(long)]
    litegs_densify_until: Option<usize>,

    /// Freeze all LiteGS topology updates at or after this epoch
    #[arg(long)]
    litegs_topology_freeze_after_epoch: Option<usize>,

    /// LiteGS densification interval
    #[arg(long, default_value = "5")]
    litegs_densification_interval: usize,

    /// LiteGS refine cadence in training iterations
    #[arg(long, default_value = "200")]
    litegs_refine_every: usize,

    /// LiteGS xy-gradient threshold for growth candidates
    #[arg(long, default_value = "0.00015")]
    litegs_growth_grad_threshold: f32,

    /// LiteGS fraction of above-threshold candidates selected for extra growth
    #[arg(long, default_value = "0.2")]
    litegs_growth_select_fraction: f32,

    /// LiteGS iteration after which extra growth stops
    #[arg(long, default_value = "15000")]
    litegs_growth_stop_iter: usize,

    /// LiteGS opacity reset interval
    #[arg(long, default_value = "10")]
    litegs_opacity_reset_interval: usize,

    /// LiteGS opacity reset mode
    #[arg(long, default_value_t = rustgs::LiteGsOpacityResetMode::Decay)]
    litegs_opacity_reset_mode: rustgs::LiteGsOpacityResetMode,

    /// LiteGS prune mode
    #[arg(long, default_value_t = rustgs::LiteGsPruneMode::Weight)]
    litegs_prune_mode: rustgs::LiteGsPruneMode,

    /// LiteGS prune offset epochs - how many epochs to wait after densify before pruning
    #[arg(long, default_value = "0")]
    litegs_prune_offset_epochs: usize,

    /// LiteGS prune minimum age - minimum iterations before Gaussian is prune-eligible
    #[arg(long, default_value = "5")]
    litegs_prune_min_age: usize,

    /// LiteGS prune invisible epochs - consecutive invisibility required before pruning
    #[arg(long, default_value = "10")]
    litegs_prune_invisible_epochs: usize,

    /// LiteGS target primitive budget
    #[arg(long, default_value = "1000000")]
    litegs_target_primitives: usize,

    /// Enable LiteGS learnable camera extrinsics
    #[arg(long, default_value_t = false)]
    litegs_learnable_viewproj: bool,

    /// LiteGS pose learning rate
    #[arg(long, default_value = "0.0001")]
    litegs_lr_pose: f32,

    /// Enable Morton code spatial sorting after densification for better memory coherence
    #[arg(long, default_value_t = true)]
    litegs_morton_sort_on_densify: bool,

    /// Prune Gaussians with max scale > threshold (0 disables scale pruning)
    #[arg(long, default_value = "0.5")]
    litegs_prune_scale_threshold: f32,

    /// Position learning rate (initial)
    #[arg(long, default_value = "0.00016")]
    lr_position: f32,

    /// Position learning rate (final) - exponential decay target
    #[arg(long, default_value = "0.0000016")]
    lr_position_final: f32,

    /// Scale learning rate
    #[arg(long, default_value = "0.005")]
    lr_scale: f32,

    /// Rotation learning rate
    #[arg(long, default_value = "0.001")]
    lr_rotation: f32,

    /// Opacity learning rate
    #[arg(long, default_value = "0.05")]
    lr_opacity: f32,

    /// Color/SH learning rate
    #[arg(long, default_value = "0.0025")]
    lr_color: f32,

    /// Enable budget-driven chunked training mode
    #[arg(long, default_value_t = false)]
    chunked_training: bool,

    /// Per-chunk memory budget in GiB
    #[arg(long, default_value = "12.0")]
    chunk_budget_gb: f32,

    /// Relative overlap expansion applied to chunk bounds
    #[arg(long, default_value = "0.15")]
    chunk_overlap_ratio: f32,

    /// Minimum number of cameras required for a chunk to remain trainable
    #[arg(long, default_value = "3")]
    min_cameras_per_chunk: usize,

    /// Maximum number of chunks to generate (0 = automatic)
    #[arg(long, default_value = "0")]
    max_chunks: usize,

    /// Explicitly keep only chunk core-region Gaussians during merge
    #[arg(long, default_value_t = false, conflicts_with = "no_merge_core_only")]
    merge_core_only: bool,

    /// Disable core-only chunk merge filtering
    #[arg(
        long = "no-merge-core-only",
        default_value_t = false,
        conflicts_with = "merge_core_only"
    )]
    no_merge_core_only: bool,

    /// Use the tensor fallback instead of the native Metal forward rasterizer
    #[arg(long, default_value_t = false)]
    metal_disable_native_forward: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Run a post-training scene evaluation pass and emit a structured summary
    #[arg(long, default_value_t = false)]
    eval_after_train: bool,

    /// Evaluation render scale used by the post-training scene evaluation
    #[arg(long, default_value = "0.25")]
    eval_render_scale: f32,

    /// Maximum number of frames considered during post-training evaluation (0 = all)
    #[arg(long, default_value = "180")]
    eval_max_frames: usize,

    /// Keep every Nth frame during post-training evaluation
    #[arg(long, default_value = "30")]
    eval_frame_stride: usize,

    /// Number of lowest-PSNR frames to keep in the evaluation summary
    #[arg(long, default_value = "5")]
    eval_worst_frames: usize,

    /// Evaluation device used by the post-training scene evaluation
    #[arg(long, default_value = "metal")]
    eval_device: String,

    /// Print the post-training evaluation summary as JSON
    #[arg(long, default_value_t = false)]
    eval_json: bool,
}

#[derive(Debug, Clone, clap::Args)]
struct RenderArgs {
    /// Path to scene PLY file
    #[arg(short, long)]
    input: PathBuf,

    /// Path to camera pose JSON file
    #[arg(short, long)]
    camera: PathBuf,

    /// Output image path
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Train a 3DGS scene from JSON input, a TUM RGB-D dataset directory, or a COLMAP directory
    Train(TrainArgs),

    /// Render a scene from a given viewpoint
    Render(RenderArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = <Cli as clap::Parser>::parse();

    match cli.command {
        Commands::Train(args) => {
            // Initialize logging
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

            // Train
            #[cfg(feature = "gpu")]
            {
                let train_start = std::time::Instant::now();
                let splats = rustgs::train_splats(&dataset, &config)?;
                let training_elapsed = train_start.elapsed();
                let training_telemetry = rustgs::last_metal_training_telemetry();
                log::info!("Trained {} Gaussians", splats.len());

                let metadata = rustgs::SceneMetadata {
                    iterations: config.iterations,
                    final_loss: training_telemetry
                        .as_ref()
                        .and_then(|telemetry| telemetry.loss_terms.total)
                        .unwrap_or(0.0),
                    gaussian_count: splats.len(),
                    sh_degree: splats.sh_degree(),
                };
                rustgs::save_splats_ply(&args.output, &splats, &metadata)?;
                log::info!("Saved scene to {:?}", args.output);

                let evaluation_summary = maybe_evaluate_trained_splats(
                    &args,
                    &splats,
                    &metadata,
                    training_telemetry.as_ref(),
                )?;

                if let Err(err) = maybe_write_litegs_parity_report(
                    &args.input,
                    &args.output,
                    &dataset,
                    &splats,
                    &config,
                    training_telemetry.as_ref(),
                    training_elapsed,
                    evaluation_summary.as_ref(),
                ) {
                    log::warn!("failed to persist LiteGS parity report: {err}");
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU feature is required for training. Rebuild with --features gpu");
                std::process::exit(1);
            }
        }
        Commands::Render(args) => {
            log::info!("Rendering scene from {:?}", args.input);
            log::info!("Camera: {:?}", args.camera);
            log::info!("Output: {:?}", args.output);

            // Load scene
            let (splats, metadata) = rustgs::load_splats_ply(&args.input)?;
            log::info!("Loaded {} Gaussians", splats.len());
            let _ = metadata;

            // TODO: Load camera and render
            let _ = args.output;
            log::warn!("Render command not yet implemented");
        }
    }

    Ok(())
}

fn load_training_dataset_for_training(
    input: &std::path::Path,
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
    input: &std::path::Path,
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
fn evaluation_dataset_load_params(args: &TrainArgs) -> (usize, usize) {
    // Keep the evaluation prefix trimming, but do not apply frame_stride here.
    // The actual evaluation subset selection should happen once inside evaluate_splats().
    (args.eval_max_frames, 1)
}

#[cfg(feature = "gpu")]
fn final_training_metrics_from_telemetry(
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    metadata: &rustgs::SceneMetadata,
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
    metadata: &rustgs::SceneMetadata,
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
        &rustgs::SceneEvaluationConfig {
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

fn build_training_config(args: &TrainArgs) -> anyhow::Result<rustgs::TrainingConfig> {
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
    // Learning rates
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

fn validate_chunked_training_args(config: &rustgs::TrainingConfig) -> anyhow::Result<()> {
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

fn maybe_write_litegs_parity_report(
    input: &std::path::Path,
    output: &std::path::Path,
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
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

fn maybe_write_litegs_parity_report_with_manifest_dir(
    input: &std::path::Path,
    output: &std::path::Path,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_elapsed: Duration,
    evaluation_summary: Option<&rustgs::SplatEvaluationSummary>,
    manifest_dir: &std::path::Path,
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
    manifest_dir: &std::path::Path,
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

fn default_chunk_artifact_dir(output: &std::path::Path) -> PathBuf {
    let parent = output
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("scene");
    parent.join(format!("{stem}-chunks"))
}

#[cfg(test)]
mod tests {
    use super::{
        build_training_config, default_chunk_artifact_dir, evaluation_dataset_load_params,
        load_training_dataset_for_training, maybe_write_litegs_parity_report,
        maybe_write_litegs_parity_report_with_manifest_dir, validate_chunked_training_args, Cli,
        Commands, TrainArgs,
    };
    use clap::Parser;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    fn parse_cli(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("cli args should parse")
    }

    fn parse_train_args(args: &[&str]) -> TrainArgs {
        let cli = parse_cli(args);
        let Commands::Train(args) = cli.command else {
            panic!("expected train command");
        };
        args
    }

    #[cfg(feature = "gpu")]
    fn rgb_to_sh0_value(rgb: f32) -> f32 {
        (rgb - 0.5) / 0.282_094_8
    }

    #[cfg(feature = "gpu")]
    fn test_splats() -> rustgs::HostSplats {
        rustgs::HostSplats::from_raw_parts(
            vec![0.0, 0.0, 0.0],
            vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.3),
                rgb_to_sh0_value(0.4),
            ],
            0,
        )
        .unwrap()
    }

    #[cfg(feature = "gpu")]
    fn test_splats_metadata(splats: &rustgs::HostSplats) -> rustgs::SceneMetadata {
        rustgs::SceneMetadata {
            iterations: 1,
            final_loss: 0.0,
            gaussian_count: splats.len(),
            sh_degree: splats.sh_degree(),
        }
    }

    #[cfg(feature = "gpu")]
    fn write_test_output_splats(path: &std::path::Path) -> rustgs::HostSplats {
        let splats = test_splats();
        rustgs::save_splats_ply(path, &splats, &test_splats_metadata(&splats)).unwrap();
        splats
    }

    #[cfg(feature = "gpu")]
    fn convergence_fixture_input_path() -> Option<PathBuf> {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find_map(|path| {
                rustgs::resolve_litegs_parity_fixture_input_path(
                    rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID,
                    path,
                )
            })
    }

    #[cfg(feature = "gpu")]
    fn metal_training_available() -> bool {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(|| candle_core::Device::new_metal(0));
        std::panic::set_hook(previous_hook);
        matches!(result, Ok(Ok(_)))
    }

    #[test]
    fn train_command_parses_chunked_defaults() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
        ]);

        assert!(!args.chunked_training);
        assert_eq!(args.chunk_budget_gb, 12.0);
        assert_eq!(args.chunk_overlap_ratio, 0.15);
        assert_eq!(args.min_cameras_per_chunk, 3);
        assert_eq!(args.max_chunks, 0);
        assert!(!args.merge_core_only);
        assert!(!args.no_merge_core_only);
        assert_eq!(args.training_profile, rustgs::TrainingProfile::LegacyMetal);
        assert_eq!(args.litegs_sh_degree, 3);
        assert_eq!(args.litegs_cluster_size, 0);
        assert_eq!(args.litegs_tile_size, rustgs::LiteGsTileSize::new(8, 16));
        assert!(!args.litegs_sparse_grad);
        assert!(!args.eval_after_train);
        assert_eq!(args.eval_render_scale, 0.25);
        assert_eq!(args.eval_max_frames, 180);
        assert_eq!(args.eval_frame_stride, 30);
        assert_eq!(args.eval_worst_frames, 5);
        assert_eq!(args.eval_device, "metal");
        assert!(!args.eval_json);
    }

    #[test]
    fn train_command_parses_all_chunked_flags() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--chunked-training",
            "--chunk-budget-gb",
            "10.5",
            "--chunk-overlap-ratio",
            "0.2",
            "--min-cameras-per-chunk",
            "5",
            "--max-chunks",
            "8",
            "--no-merge-core-only",
        ]);

        assert!(args.chunked_training);
        assert_eq!(args.chunk_budget_gb, 10.5);
        assert_eq!(args.chunk_overlap_ratio, 0.2);
        assert_eq!(args.min_cameras_per_chunk, 5);
        assert_eq!(args.max_chunks, 8);
        assert!(args.no_merge_core_only);
    }

    #[test]
    fn train_command_parses_post_training_eval_flags() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--eval-after-train",
            "--eval-render-scale",
            "0.125",
            "--eval-max-frames",
            "60",
            "--eval-frame-stride",
            "10",
            "--eval-worst-frames",
            "3",
            "--eval-device",
            "cpu",
            "--eval-json",
        ]);

        assert!(args.eval_after_train);
        assert_eq!(args.eval_render_scale, 0.125);
        assert_eq!(args.eval_max_frames, 60);
        assert_eq!(args.eval_frame_stride, 10);
        assert_eq!(args.eval_worst_frames, 3);
        assert_eq!(args.eval_device, "cpu");
        assert!(args.eval_json);
    }

    #[test]
    fn post_training_eval_loads_prefix_without_double_applying_stride() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--eval-after-train",
            "--eval-max-frames",
            "180",
            "--eval-frame-stride",
            "30",
        ]);

        let (dataset_max_frames, dataset_frame_stride) = evaluation_dataset_load_params(&args);
        assert_eq!(dataset_max_frames, 180);
        assert_eq!(dataset_frame_stride, 1);
    }

    #[test]
    fn train_command_parses_litegs_flags_and_builds_nested_config() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--training-profile",
            "litegs-mac-v1",
            "--litegs-sh-degree",
            "4",
            "--litegs-cluster-size",
            "0",
            "--litegs-tile-size",
            "16x16",
            "--litegs-sparse-grad",
            "--litegs-reg-weight",
            "0.01",
            "--litegs-enable-transmittance",
            "--litegs-enable-depth",
            "--litegs-densify-from",
            "6",
            "--litegs-densify-until",
            "24",
            "--litegs-topology-freeze-after-epoch",
            "18",
            "--litegs-refine-every",
            "120",
            "--litegs-densification-interval",
            "8",
            "--litegs-growth-grad-threshold",
            "0.0003",
            "--litegs-growth-select-fraction",
            "0.35",
            "--litegs-growth-stop-iter",
            "2400",
            "--litegs-opacity-reset-interval",
            "12",
            "--litegs-opacity-reset-mode",
            "reset",
            "--litegs-prune-mode",
            "threshold",
            "--litegs-target-primitives",
            "200000",
            "--litegs-learnable-viewproj",
            "--litegs-lr-pose",
            "0.0002",
        ]);
        let config = build_training_config(&args).unwrap();

        assert_eq!(
            config.training_profile,
            rustgs::TrainingProfile::LiteGsMacV1
        );
        assert_eq!(config.litegs.sh_degree, 4);
        assert_eq!(config.litegs.cluster_size, 0);
        assert_eq!(config.litegs.tile_size, rustgs::LiteGsTileSize::new(16, 16));
        assert!(config.litegs.sparse_grad);
        assert_eq!(config.litegs.reg_weight, 0.01);
        assert!(config.litegs.enable_transmittance);
        assert!(config.litegs.enable_depth);
        assert_eq!(config.litegs.densify_from, 6);
        assert_eq!(config.litegs.densify_until, Some(24));
        assert_eq!(config.litegs.topology_freeze_after_epoch, Some(18));
        assert_eq!(config.litegs.refine_every, 120);
        assert_eq!(config.litegs.densification_interval, 8);
        assert_eq!(config.litegs.growth_grad_threshold, 0.0003);
        assert_eq!(config.litegs.growth_select_fraction, 0.35);
        assert_eq!(config.litegs.growth_stop_iter, 2_400);
        assert_eq!(config.litegs.opacity_reset_interval, 12);
        assert_eq!(
            config.litegs.opacity_reset_mode,
            rustgs::LiteGsOpacityResetMode::Reset
        );
        assert_eq!(config.litegs.prune_mode, rustgs::LiteGsPruneMode::Threshold);
        assert_eq!(config.litegs.prune_offset_epochs, 0); // default value
        assert_eq!(config.litegs.prune_min_age, 5); // default value
        assert_eq!(config.litegs.prune_invisible_epochs, 10); // default value
        assert_eq!(config.litegs.target_primitives, 200_000);
        assert!(config.litegs.learnable_viewproj);
        assert_eq!(config.litegs.lr_pose, 0.0002);
    }

    #[test]
    fn train_command_rejects_conflicting_merge_flags() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--merge-core-only",
            "--no-merge-core-only",
        ])
        .expect_err("conflicting merge flags should fail");

        let message = err.to_string();
        assert!(message.contains("--merge-core-only"));
        assert!(message.contains("--no-merge-core-only"));
    }

    #[test]
    fn chunk_validation_accepts_non_chunked_defaults() {
        let config = rustgs::TrainingConfig::default();
        validate_chunked_training_args(&config).unwrap();
    }

    #[test]
    fn chunk_validation_rejects_zero_budget() {
        let config = rustgs::TrainingConfig {
            chunked_training: true,
            chunk_budget_gb: 0.0,
            ..rustgs::TrainingConfig::default()
        };
        let err = validate_chunked_training_args(&config).expect_err("zero budget should fail");
        assert!(err.to_string().contains("--chunk-budget-gb must be > 0"));
    }

    #[test]
    fn chunk_validation_rejects_invalid_overlap() {
        let config = rustgs::TrainingConfig {
            chunked_training: true,
            chunk_overlap_ratio: 0.5,
            ..rustgs::TrainingConfig::default()
        };
        let err = validate_chunked_training_args(&config).expect_err("invalid overlap should fail");
        assert!(err
            .to_string()
            .contains("--chunk-overlap-ratio must be in [0.0, 0.5)"));
    }

    #[test]
    fn chunk_validation_rejects_zero_min_cameras() {
        let config = rustgs::TrainingConfig {
            chunked_training: true,
            min_cameras_per_chunk: 0,
            ..rustgs::TrainingConfig::default()
        };
        let err =
            validate_chunked_training_args(&config).expect_err("zero min cameras should fail");
        assert!(err
            .to_string()
            .contains("--min-cameras-per-chunk must be >= 1"));
    }

    #[test]
    fn chunk_validation_rejects_illegal_max_chunks() {
        let config = rustgs::TrainingConfig {
            chunked_training: true,
            max_chunks: 1,
            ..rustgs::TrainingConfig::default()
        };
        let err = validate_chunked_training_args(&config).expect_err("max_chunks=1 should fail");
        assert!(err
            .to_string()
            .contains("--max-chunks must be 0 (auto) or >= 2"));
    }

    #[test]
    fn default_chunk_artifact_dir_uses_output_stem() {
        let path = default_chunk_artifact_dir(std::path::Path::new("/tmp/scene.ply"));
        assert_eq!(path, PathBuf::from("/tmp/scene-chunks"));
    }

    #[test]
    fn litegs_parity_report_is_written_next_to_output_scene() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

        let splats = write_test_output_splats(&output);

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        dataset.add_point([1.0, 0.0, 0.0], None);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            ..rustgs::TrainingConfig::default()
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &dataset,
            &splats,
            &config,
            None,
            Duration::from_millis(42),
            None,
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        assert!(report_path.exists());
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
        assert_eq!(report.fixture_id, rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID);
        assert_eq!(report.topology.initialization_gaussians, Some(2));
        assert_eq!(report.topology.final_gaussians, Some(1));
        assert_eq!(report.topology.export_outputs, 1);
        assert_eq!(report.metrics.active_sh_degree, Some(3));
        assert_eq!(report.metrics.rotation_frozen, None);
        assert!(report.metrics.export_roundtrip_ok);
        assert_eq!(report.timing.training_ms, Some(42));
        assert_eq!(
            report.gate.as_ref().map(|gate| &gate.status),
            Some(&rustgs::ParityGateStatus::MissingReference)
        );
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("No checked-in LiteGS parity reference report was found")));
    }

    #[test]
    fn litegs_parity_report_uses_runtime_telemetry_when_available() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

        let splats = write_test_output_splats(&output);

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            ..rustgs::TrainingConfig::default()
        };
        let telemetry = rustgs::LiteGsTrainingTelemetry {
            loss_terms: rustgs::ParityLossTerms {
                l1: Some(0.1),
                ssim: Some(0.2),
                scale_regularization: Some(0.3),
                transmittance: Some(0.4),
                depth: None,
                total: Some(0.5),
            },
            loss_curve_samples: vec![rustgs::ParityLossCurveSample {
                iteration: 25,
                frame_idx: 1,
                l1: Some(0.1),
                ssim: Some(0.2),
                depth: None,
                total: Some(0.5),
                depth_valid_pixels: None,
            }],
            topology: rustgs::ParityTopologyMetrics {
                final_gaussians: Some(7),
                densify_events: 2,
                densify_added: 5,
                prune_events: 1,
                prune_removed: 3,
                opacity_reset_events: 4,
                ..Default::default()
            },
            active_sh_degree: Some(2),
            final_loss: Some(0.45),
            final_step_loss: Some(0.5),
            depth_valid_pixels: None,
            depth_grad_scale: None,
            rotation_frozen: true,
            learning_rates: rustgs::LiteGsOptimizerLrs::default(),
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &dataset,
            &splats,
            &config,
            Some(&telemetry),
            Duration::from_millis(42),
            None,
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
        assert_eq!(report.loss_terms.total, Some(0.5));
        assert_eq!(report.topology.densify_events, 2);
        assert_eq!(report.topology.densify_added, 5);
        assert_eq!(report.topology.prune_removed, 3);
        assert_eq!(report.metrics.active_sh_degree, Some(2));
        assert_eq!(report.metrics.depth_valid_pixels, None);
        assert_eq!(report.metrics.depth_grad_scale, None);
        assert_eq!(report.loss_curve_samples.len(), 1);
        assert_eq!(report.loss_curve_samples[0].iteration, 25);
        assert_eq!(report.metrics.rotation_frozen, Some(true));
        assert_eq!(report.topology.final_gaussians, Some(7));
        assert_eq!(
            report.gate.as_ref().map(|gate| &gate.status),
            Some(&rustgs::ParityGateStatus::MissingReference)
        );
    }

    #[test]
    fn litegs_parity_report_records_final_psnr_from_evaluation_summary() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

        let splats = write_test_output_splats(&output);

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            ..rustgs::TrainingConfig::default()
        };
        let evaluation_summary = rustgs::SplatEvaluationSummary {
            device: rustgs::EvaluationDevice::Metal,
            render_scale: 0.25,
            render_width: 16,
            render_height: 16,
            frame_stride: 30,
            max_frames: 180,
            frame_count: 6,
            splat_iterations: 1,
            splat_count: 1,
            final_loss: 0.4,
            final_step_loss: Some(0.35),
            elapsed_seconds: 1.2,
            psnr_mean_db: 7.25,
            psnr_median_db: 7.10,
            psnr_min_db: 6.8,
            psnr_max_db: 7.6,
            psnr_std_db: 0.3,
            worst_frames: Vec::new(),
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &dataset,
            &splats,
            &config,
            None,
            Duration::from_millis(42),
            Some(&evaluation_summary),
        )
        .unwrap();

        let report =
            rustgs::ParityHarnessReport::load_json(&rustgs::default_parity_report_path(&output))
                .unwrap();
        assert_eq!(report.metrics.final_psnr, Some(7.25));
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("mean PSNR 7.2500 dB")));
    }

    #[test]
    fn litegs_parity_report_persists_depth_and_sparse_cluster_config() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

        let splats = write_test_output_splats(&output);

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            litegs: rustgs::LiteGsConfig {
                cluster_size: 64,
                sparse_grad: true,
                enable_depth: true,
                ..rustgs::LiteGsConfig::default()
            },
            ..rustgs::TrainingConfig::default()
        };
        let telemetry = rustgs::LiteGsTrainingTelemetry {
            loss_terms: rustgs::ParityLossTerms {
                l1: Some(0.1),
                ssim: Some(0.2),
                scale_regularization: Some(0.3),
                transmittance: Some(0.4),
                depth: Some(0.6),
                total: Some(0.7),
            },
            loss_curve_samples: vec![rustgs::ParityLossCurveSample {
                iteration: 50,
                frame_idx: 2,
                l1: Some(0.1),
                ssim: Some(0.2),
                depth: Some(0.6),
                total: Some(0.7),
                depth_valid_pixels: Some(256),
            }],
            topology: rustgs::ParityTopologyMetrics {
                final_gaussians: Some(3),
                ..Default::default()
            },
            active_sh_degree: Some(2),
            final_loss: Some(0.65),
            final_step_loss: Some(0.7),
            depth_valid_pixels: Some(256),
            depth_grad_scale: Some(0.1 / 256.0),
            rotation_frozen: true,
            learning_rates: rustgs::LiteGsOptimizerLrs::default(),
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &dataset,
            &splats,
            &config,
            Some(&telemetry),
            Duration::from_millis(42),
            None,
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
        assert_eq!(report.litegs.cluster_size, 64);
        assert!(report.litegs.sparse_grad);
        assert!(report.litegs.enable_depth);
        assert_eq!(report.loss_terms.depth, Some(0.6));
        assert_eq!(report.loss_terms.total, Some(0.7));
        assert_eq!(report.metrics.depth_valid_pixels, Some(256));
        assert_eq!(report.metrics.depth_grad_scale, Some(0.1 / 256.0));
        assert_eq!(report.loss_curve_samples.len(), 1);
        assert_eq!(report.loss_curve_samples[0].depth, Some(0.6));
        assert_eq!(
            report.gate.as_ref().map(|gate| &gate.status),
            Some(&rustgs::ParityGateStatus::MissingReference)
        );
    }

    #[test]
    fn litegs_parity_report_populates_reference_comparison_from_workspace_fixture() {
        let dir = tempdir().unwrap();
        let workspace_root = dir.path().join("workspace");
        let manifest_dir = workspace_root.join("RustGS");
        std::fs::create_dir_all(&manifest_dir).unwrap();

        let reference_path =
            workspace_root.join("test_data/fixtures/litegs/colmap-small/parity-reference.json");
        std::fs::create_dir_all(reference_path.parent().unwrap()).unwrap();

        let mut reference_report = rustgs::ParityHarnessReport::new(
            rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID,
            rustgs::TrainingProfile::LiteGsMacV1,
            &rustgs::LiteGsConfig::default(),
        );
        reference_report.loss_curve_samples = vec![
            rustgs::ParityLossCurveSample {
                iteration: 0,
                frame_idx: 0,
                depth: Some(0.55),
                total: Some(1.1),
                depth_valid_pixels: Some(128),
                ..Default::default()
            },
            rustgs::ParityLossCurveSample {
                iteration: 25,
                frame_idx: 1,
                depth: Some(0.35),
                total: Some(0.8),
                depth_valid_pixels: Some(96),
                ..Default::default()
            },
        ];
        reference_report.topology.final_gaussians = Some(1);
        reference_report.save_json(&reference_path).unwrap();

        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");
        let splats = write_test_output_splats(&output);

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            ..rustgs::TrainingConfig::default()
        };
        let telemetry = rustgs::LiteGsTrainingTelemetry {
            loss_terms: rustgs::ParityLossTerms {
                depth: Some(0.4),
                total: Some(0.9),
                ..Default::default()
            },
            loss_curve_samples: vec![
                rustgs::ParityLossCurveSample {
                    iteration: 0,
                    frame_idx: 0,
                    depth: Some(0.5),
                    total: Some(1.0),
                    depth_valid_pixels: Some(128),
                    ..Default::default()
                },
                rustgs::ParityLossCurveSample {
                    iteration: 25,
                    frame_idx: 1,
                    depth: Some(0.3),
                    total: Some(0.7),
                    depth_valid_pixels: Some(96),
                    ..Default::default()
                },
            ],
            topology: rustgs::ParityTopologyMetrics {
                final_gaussians: Some(1),
                ..Default::default()
            },
            active_sh_degree: Some(3),
            final_loss: Some(0.75),
            final_step_loss: Some(0.7),
            depth_valid_pixels: Some(96),
            depth_grad_scale: Some(0.1 / 96.0),
            rotation_frozen: true,
            learning_rates: rustgs::LiteGsOptimizerLrs::default(),
        };

        maybe_write_litegs_parity_report_with_manifest_dir(
            input,
            &output,
            &dataset,
            &splats,
            &config,
            Some(&telemetry),
            Duration::from_millis(42),
            None,
            &manifest_dir,
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
        let comparison = report
            .reference_comparison
            .expect("reference comparison should be populated");
        assert_eq!(comparison.compared_iterations, 2);
        assert_eq!(comparison.compared_depth_samples, 2);
        assert_eq!(comparison.compared_total_samples, 2);
        assert!((comparison.depth_mean_abs_delta.unwrap() - 0.05).abs() < 1e-6);
        assert!((comparison.depth_max_abs_delta.unwrap() - 0.05).abs() < 1e-6);
        assert!((comparison.total_mean_abs_delta.unwrap() - 0.1).abs() < 1e-6);
        assert!((comparison.total_max_abs_delta.unwrap() - 0.1).abs() < 1e-6);
        assert_eq!(report.metrics.gaussian_count_delta_ratio, Some(0.0));
        assert_eq!(
            report.gate.as_ref().map(|gate| &gate.status),
            Some(&rustgs::ParityGateStatus::Passed)
        );
        assert!(report.notes.iter().any(
            |note| note.contains("Compared parity loss curve samples against reference report")
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn litegs_fixture_parity_regression_writes_report_from_real_training_run() {
        let Some(input) = convergence_fixture_input_path() else {
            eprintln!("skipping test: no LiteGS parity fixture was available in this workspace");
            return;
        };
        if !metal_training_available() {
            eprintln!("skipping test: Metal training is unavailable in this environment");
            return;
        }

        let tum_config = rustgs::TumRgbdConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        };
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            iterations: 1,
            max_initial_gaussians: 2048,
            metal_render_scale: 0.5,
            litegs: rustgs::LiteGsConfig {
                cluster_size: 64,
                sparse_grad: true,
                enable_depth: true,
                ..rustgs::LiteGsConfig::default()
            },
            ..rustgs::TrainingConfig::default()
        };
        let output_dir = tempdir().unwrap();
        let output = output_dir.path().join("fixture-scene.ply");
        let dataset = load_training_dataset_for_training(
            &input,
            tum_config.max_frames,
            tum_config.frame_stride,
        )
        .unwrap();

        let training_start = Instant::now();
        let splats = rustgs::train_splats_from_path(&input, &tum_config, &config).unwrap();
        let training_elapsed = training_start.elapsed();
        let training_telemetry = rustgs::last_metal_training_telemetry();
        let final_loss = training_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.loss_terms.total)
            .unwrap_or(0.0);

        rustgs::save_splats_ply(
            &output,
            &splats,
            &rustgs::SceneMetadata {
                iterations: config.iterations,
                final_loss,
                gaussian_count: splats.len(),
                sh_degree: splats.sh_degree(),
            },
        )
        .unwrap();

        maybe_write_litegs_parity_report(
            &input,
            &output,
            &dataset,
            &splats,
            &config,
            training_telemetry.as_ref(),
            training_elapsed,
            None,
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();

        assert_eq!(report.fixture_id, rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID);
        assert_eq!(report.profile, rustgs::TrainingProfile::LiteGsMacV1);
        assert_eq!(report.litegs.cluster_size, 64);
        assert!(report.litegs.sparse_grad);
        assert!(report.litegs.enable_depth);
        assert_eq!(report.topology.export_outputs, 1);
        assert_eq!(report.topology.final_gaussians, Some(splats.len()));
        assert!(report.loss_terms.total.unwrap_or(0.0) > 0.0);
        assert!(report.loss_terms.depth.is_some());
        assert!(!report.loss_curve_samples.is_empty());
        assert!(report
            .loss_curve_samples
            .iter()
            .any(|sample| sample.depth.is_some()));
        assert!(report.metrics.depth_valid_pixels.unwrap_or(0) > 0);
        assert!(report.metrics.depth_grad_scale.unwrap_or(0.0) > 0.0);
        assert!(!report.metrics.had_nan);
        assert!(!report.metrics.had_oom);
        assert!(report.metrics.export_roundtrip_ok);
        assert!(report.timing.training_ms.unwrap_or(0) > 0);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("frame-based fallback")),
            "bootstrap TUM fixture should note approximate initialization parity"
        );
    }
}
