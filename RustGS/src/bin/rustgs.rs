//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <training_dataset.json|tum_dataset_dir|colmap_dir> --output <scene.ply>
//!   rustgs render --input <scene.ply> --camera <pose.json> --output <image.png>

#[cfg(not(feature = "gpu"))]
use anyhow::bail;
use std::path::PathBuf;

#[path = "rustgs/train_command.rs"]
mod train_command;

#[derive(Debug, clap::Parser)]
#[command(name = "rustgs")]
#[command(about = "3D Gaussian Splatting Training", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, clap::Args)]
struct TrainArgs {
    /// Path to TrainingDataset JSON, TUM RGB-D directory, or COLMAP directory
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
    #[arg(long, default_value = "160")]
    litegs_refine_every: usize,

    /// LiteGS xy-gradient threshold for growth candidates
    #[arg(long, default_value = "0.00014")]
    litegs_growth_grad_threshold: f32,

    /// LiteGS fraction of above-threshold candidates selected for extra growth
    #[arg(long, default_value = "0.25")]
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
        Commands::Train(args) => train_command::run_train_command(args)?,
        Commands::Render(args) => run_render_command(args)?,
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_render_command(args: RenderArgs) -> anyhow::Result<()> {
    log::info!("Rendering scene from {:?}", args.input);
    log::info!("Camera: {:?}", args.camera);
    log::info!("Output: {:?}", args.output);

    let (splats, metadata) = rustgs::load_splats_ply(&args.input)?;
    log::info!("Loaded {} Gaussians", splats.len());
    let _ = metadata;

    let _ = args.output;
    log::warn!("Render command not yet implemented");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_render_command(_args: RenderArgs) -> anyhow::Result<()> {
    bail!("render requires the gpu feature");
}

#[cfg(test)]
mod tests {
    use super::{
        train_command::{
            build_training_config, default_chunk_artifact_dir, evaluation_dataset_load_params,
            load_training_dataset_for_training, maybe_write_litegs_parity_report,
            maybe_write_litegs_parity_report_with_manifest_dir, validate_chunked_training_args,
        },
        Cli, Commands, TrainArgs,
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
    fn test_splats_metadata(splats: &rustgs::HostSplats) -> rustgs::SplatMetadata {
        rustgs::SplatMetadata {
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

        let training_run =
            rustgs::train_splats_from_path_with_report(&input, &tum_config, &config).unwrap();
        let training_elapsed = training_run.report.elapsed;
        let final_loss = training_run.report.metadata_final_loss_or(0.0);
        let training_telemetry = training_run.report.telemetry.clone();
        let splats = training_run.splats;

        rustgs::save_splats_ply(
            &output,
            &splats,
            &rustgs::SplatMetadata {
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
