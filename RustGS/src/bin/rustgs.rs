//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <slam_output.json|tum_dataset_dir> --output <scene.ply>
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
    /// Path to SLAM output JSON file or TUM RGB-D dataset directory
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

    /// LiteGS densification interval
    #[arg(long, default_value = "5")]
    litegs_densification_interval: usize,

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
    #[arg(long, default_value = "2")]
    litegs_prune_offset_epochs: usize,

    /// LiteGS prune minimum age - minimum iterations before Gaussian is prune-eligible
    #[arg(long, default_value = "3")]
    litegs_prune_min_age: usize,

    /// LiteGS prune invisible epochs - consecutive invisibility required before pruning
    #[arg(long, default_value = "2")]
    litegs_prune_invisible_epochs: usize,

    /// LiteGS target primitive budget
    #[arg(long, default_value = "1000000")]
    litegs_target_primitives: usize,

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
    /// Train a 3DGS scene from SLAM output JSON or a TUM RGB-D dataset directory
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

            log::info!("Training 3DGS scene from {:?}", args.input);
            log::info!("Output: {:?}", args.output);
            log::info!("Iterations: {}", args.iterations);
            log::info!("Backend: metal");
            log::info!("Training profile: {}", args.training_profile);

            let slam_output = load_training_input(&args.input, args.max_frames, args.frame_stride)?;
            log::info!(
                "Loaded {} poses, {} map points",
                slam_output.num_poses(),
                slam_output.num_points()
            );

            let config = build_training_config(&args)?;
            validate_chunked_training_args(&config)?;
            log_chunked_training_config(&config);
            log_litegs_training_config(&config);

            // Train
            #[cfg(feature = "gpu")]
            {
                let train_start = std::time::Instant::now();
                let scene = rustgs::train_from_slam(&slam_output, &config)?;
                let training_elapsed = train_start.elapsed();
                let training_telemetry = rustgs::last_metal_training_telemetry();
                log::info!("Trained {} Gaussians", scene.len());

                // Save scene - convert Gaussian3D to array-based Gaussian for PLY export
                let gaussians: Vec<rustgs::Gaussian> = scene
                    .gaussians()
                    .iter()
                    .map(|g| {
                        rustgs::Gaussian::new(
                            g.position.into(),
                            g.scale.into(),
                            [g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z],
                            g.opacity,
                            g.color,
                        )
                    })
                    .collect();
                let metadata = rustgs::SceneMetadata {
                    iterations: config.iterations,
                    final_loss: training_telemetry
                        .as_ref()
                        .and_then(|telemetry| telemetry.loss_terms.total)
                        .unwrap_or(0.0),
                    gaussian_count: gaussians.len(),
                };
                rustgs::save_scene_ply(&args.output, &gaussians, &metadata)?;
                log::info!("Saved scene to {:?}", args.output);

                if let Err(err) = maybe_write_litegs_parity_report(
                    &args.input,
                    &args.output,
                    &slam_output,
                    &scene,
                    &config,
                    training_telemetry.as_ref(),
                    training_elapsed,
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
            let (gaussians, metadata) = rustgs::load_scene_ply(&args.input)?;
            log::info!("Loaded {} Gaussians", gaussians.len());
            let _ = metadata;

            // TODO: Load camera and render
            let _ = args.output;
            log::warn!("Render command not yet implemented");
        }
    }

    Ok(())
}

fn load_training_input(
    input: &std::path::Path,
    max_frames: usize,
    frame_stride: usize,
) -> anyhow::Result<rustscan_types::SlamOutput> {
    if !input.is_dir() && (max_frames > 0 || frame_stride > 1) {
        log::warn!(
            "--max-frames and --frame-stride only apply to TUM RGB-D dataset directories; ignoring them for {:?}",
            input
        );
    }

    let dataset = rustgs::load_training_dataset(
        input,
        &rustgs::TumRgbdConfig {
            max_frames,
            frame_stride,
            ..Default::default()
        },
    )?;

    if input.is_dir() {
        log::info!(
            "Resolved {:?} as a TUM RGB-D dataset with {} poses",
            input,
            dataset.poses.len(),
        );
    } else {
        log::info!("Resolved {:?} as serialized training input", input);
    }

    Ok(rustscan_types::SlamOutput::from_dataset(dataset))
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
        densification_interval: args.litegs_densification_interval,
        opacity_reset_interval: args.litegs_opacity_reset_interval,
        opacity_reset_mode: args.litegs_opacity_reset_mode,
        prune_mode: args.litegs_prune_mode,
        prune_offset_epochs: args.litegs_prune_offset_epochs,
        prune_min_age: args.litegs_prune_min_age,
        prune_invisible_epochs: args.litegs_prune_invisible_epochs,
        target_primitives: args.litegs_target_primitives,
        learnable_viewproj: false,
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
        "LiteGS profile config | sh_degree={} | cluster_size={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | densify_from={} | densify_until={:?} | densification_interval={} | opacity_reset_interval={} | opacity_reset_mode={} | prune_mode={} | target_primitives={}",
        config.litegs.sh_degree,
        config.litegs.cluster_size,
        config.litegs.tile_size,
        config.litegs.sparse_grad,
        config.litegs.reg_weight,
        config.litegs.enable_transmittance,
        config.litegs.enable_depth,
        config.litegs.densify_from,
        config.litegs.densify_until,
        config.litegs.densification_interval,
        config.litegs.opacity_reset_interval,
        config.litegs.opacity_reset_mode,
        config.litegs.prune_mode,
        config.litegs.target_primitives,
    );
}

fn maybe_write_litegs_parity_report(
    input: &std::path::Path,
    output: &std::path::Path,
    slam_output: &rustscan_types::SlamOutput,
    scene: &rustgs::GaussianMap,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_elapsed: Duration,
) -> anyhow::Result<()> {
    if config.training_profile != rustgs::TrainingProfile::LiteGsMacV1 {
        return Ok(());
    }

    let report_path = rustgs::default_parity_report_path(output);
    let fixture_id = rustgs::parity_fixture_id_for_input_path(input);
    let mut report =
        rustgs::ParityHarnessReport::new(fixture_id, config.training_profile, &config.litegs);

    report.topology.initialization_gaussians =
        inferred_initialization_gaussian_count(slam_output, config);
    report.topology.final_gaussians = Some(scene.len());
    report.topology.export_outputs = 1;

    if let Some(telemetry) = training_telemetry {
        report.loss_terms = telemetry.loss_terms.clone();
        report.topology.densify_events = telemetry.topology.densify_events;
        report.topology.densify_added = telemetry.topology.densify_added;
        report.topology.prune_events = telemetry.topology.prune_events;
        report.topology.prune_removed = telemetry.topology.prune_removed;
        report.topology.opacity_reset_events = telemetry.topology.opacity_reset_events;
        report.topology.final_gaussians = telemetry.topology.final_gaussians.or(Some(scene.len()));
        report.metrics.active_sh_degree = telemetry.active_sh_degree;
        report.metrics.rotation_frozen = Some(telemetry.rotation_frozen);
    } else {
        report.metrics.active_sh_degree = Some(config.litegs.sh_degree);
    }
    report.metrics.had_nan = scene_contains_non_finite(scene);
    report.metrics.had_oom = false;

    report.timing.training_ms = Some(training_elapsed.as_millis() as u64);
    report.timing.total_wall_clock_ms = Some(training_elapsed.as_millis() as u64);

    if slam_output.num_points() == 0 {
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

    let (roundtrip_gaussians, roundtrip_metadata) = rustgs::load_scene_ply(output)?;
    report.metrics.export_roundtrip_ok = roundtrip_gaussians.len() == scene.len()
        && roundtrip_metadata.gaussian_count == scene.len()
        && !gaussians_have_non_finite(&roundtrip_gaussians);

    report.save_json(&report_path)?;
    log::info!("Saved LiteGS parity report to {:?}", report_path);
    Ok(())
}

fn inferred_initialization_gaussian_count(
    slam_output: &rustscan_types::SlamOutput,
    config: &rustgs::TrainingConfig,
) -> Option<usize> {
    let sparse_points = slam_output.num_points();
    if sparse_points == 0 {
        None
    } else {
        Some(sparse_points.min(config.max_initial_gaussians.max(1)))
    }
}

fn scene_contains_non_finite(scene: &rustgs::GaussianMap) -> bool {
    gaussians_have_non_finite(
        &scene
            .gaussians()
            .iter()
            .map(|gaussian| {
                rustgs::Gaussian::new(
                    gaussian.position.into(),
                    gaussian.scale.into(),
                    [
                        gaussian.rotation.w,
                        gaussian.rotation.x,
                        gaussian.rotation.y,
                        gaussian.rotation.z,
                    ],
                    gaussian.opacity,
                    gaussian.color,
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn gaussians_have_non_finite(gaussians: &[rustgs::Gaussian]) -> bool {
    gaussians.iter().any(|gaussian| {
        gaussian.position.iter().any(|value| !value.is_finite())
            || gaussian.scale.iter().any(|value| !value.is_finite())
            || gaussian.rotation.iter().any(|value| !value.is_finite())
            || !gaussian.opacity.is_finite()
            || gaussian.color.iter().any(|value| !value.is_finite())
    })
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
        build_training_config, default_chunk_artifact_dir, maybe_write_litegs_parity_report,
        validate_chunked_training_args, Cli, Commands, TrainArgs,
    };
    use clap::Parser;
    use std::path::PathBuf;
    use std::time::Duration;
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
            "--litegs-densification-interval",
            "8",
            "--litegs-opacity-reset-interval",
            "12",
            "--litegs-opacity-reset-mode",
            "reset",
            "--litegs-prune-mode",
            "threshold",
            "--litegs-target-primitives",
            "200000",
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
        assert_eq!(config.litegs.densification_interval, 8);
        assert_eq!(config.litegs.opacity_reset_interval, 12);
        assert_eq!(
            config.litegs.opacity_reset_mode,
            rustgs::LiteGsOpacityResetMode::Reset
        );
        assert_eq!(config.litegs.prune_mode, rustgs::LiteGsPruneMode::Threshold);
        assert_eq!(config.litegs.prune_offset_epochs, 2); // default value
        assert_eq!(config.litegs.prune_min_age, 3); // default value
        assert_eq!(config.litegs.prune_invisible_epochs, 2); // default value
        assert_eq!(config.litegs.target_primitives, 200_000);
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

        let scene = rustgs::GaussianMap::from_gaussians(vec![rustgs::Gaussian3D::default()]);
        let gaussians = vec![rustgs::Gaussian::new(
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [0.2, 0.3, 0.4],
        )];
        rustgs::save_scene_ply(
            &output,
            &gaussians,
            &rustgs::SceneMetadata {
                iterations: 1,
                final_loss: 0.0,
                gaussian_count: gaussians.len(),
            },
        )
        .unwrap();

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        dataset.add_point([1.0, 0.0, 0.0], None);
        let slam_output = rustscan_types::SlamOutput::from_dataset(dataset);
        let config = rustgs::TrainingConfig {
            training_profile: rustgs::TrainingProfile::LiteGsMacV1,
            ..rustgs::TrainingConfig::default()
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &slam_output,
            &scene,
            &config,
            None,
            Duration::from_millis(42),
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
    }

    #[test]
    fn litegs_parity_report_uses_runtime_telemetry_when_available() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scene.ply");
        let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

        let scene = rustgs::GaussianMap::from_gaussians(vec![rustgs::Gaussian3D::default()]);
        let gaussians = vec![rustgs::Gaussian::new(
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [0.2, 0.3, 0.4],
        )];
        rustgs::save_scene_ply(
            &output,
            &gaussians,
            &rustgs::SceneMetadata {
                iterations: 1,
                final_loss: 0.0,
                gaussian_count: gaussians.len(),
            },
        )
        .unwrap();

        let mut dataset =
            rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
        dataset.add_point([0.0, 0.0, 0.0], None);
        let slam_output = rustscan_types::SlamOutput::from_dataset(dataset);
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
            rotation_frozen: true,
            learning_rates: rustgs::LiteGsOptimizerLrs::default(),
        };

        maybe_write_litegs_parity_report(
            input,
            &output,
            &slam_output,
            &scene,
            &config,
            Some(&telemetry),
            Duration::from_millis(42),
        )
        .unwrap();

        let report_path = rustgs::default_parity_report_path(&output);
        let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
        assert_eq!(report.loss_terms.total, Some(0.5));
        assert_eq!(report.topology.densify_events, 2);
        assert_eq!(report.topology.densify_added, 5);
        assert_eq!(report.topology.prune_removed, 3);
        assert_eq!(report.metrics.active_sh_degree, Some(2));
        assert_eq!(report.metrics.rotation_frozen, Some(true));
        assert_eq!(report.topology.final_gaussians, Some(7));
    }
}
