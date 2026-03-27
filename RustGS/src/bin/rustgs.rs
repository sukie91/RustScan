//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <slam_output.json|tum_dataset_dir> --output <scene.ply>
//!   rustgs render --input <scene.ply> --camera <pose.json> --output <image.png>

use anyhow::bail;
use std::path::PathBuf;

#[derive(Debug, clap::Parser)]
#[command(name = "rustgs")]
#[command(about = "3D Gaussian Splatting Training", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Train a 3DGS scene from SLAM output JSON or a TUM RGB-D dataset directory
    Train {
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
    },

    /// Render a scene from a given viewpoint
    Render {
        /// Path to scene PLY file
        #[arg(short, long)]
        input: PathBuf,

        /// Path to camera pose JSON file
        #[arg(short, long)]
        camera: PathBuf,

        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = <Cli as clap::Parser>::parse();

    match cli.command {
        Commands::Train {
            input,
            output,
            iterations,
            max_initial_gaussians,
            sampling_step,
            max_frames,
            frame_stride,
            metal_render_scale,
            metal_gaussian_chunk_size,
            metal_profile_steps,
            metal_profile_interval,
            prune_interval,
            topology_warmup,
            topology_log_interval,
            chunked_training,
            chunk_budget_gb,
            chunk_overlap_ratio,
            min_cameras_per_chunk,
            max_chunks,
            merge_core_only,
            no_merge_core_only,
            metal_disable_native_forward,
            log_level,
        } => {
            // Initialize logging
            env_logger::Builder::new().parse_filters(&log_level).init();

            log::info!("Training 3DGS scene from {:?}", input);
            log::info!("Output: {:?}", output);
            log::info!("Iterations: {}", iterations);
            log::info!("Backend: metal");

            let slam_output = load_training_input(&input, max_frames, frame_stride)?;
            log::info!(
                "Loaded {} poses, {} map points",
                slam_output.num_poses(),
                slam_output.num_points()
            );

            // Configure training
            let mut config = rustgs::TrainingConfig::default();
            config.iterations = iterations;
            config.max_initial_gaussians = max_initial_gaussians;
            config.sampling_step = sampling_step;
            config.metal_render_scale = metal_render_scale;
            config.metal_gaussian_chunk_size = metal_gaussian_chunk_size;
            config.metal_profile_steps = metal_profile_steps;
            config.metal_profile_interval = metal_profile_interval;
            config.prune_interval = prune_interval;
            config.topology_warmup = topology_warmup;
            config.topology_log_interval = topology_log_interval;
            config.metal_use_native_forward = !metal_disable_native_forward;
            config.chunked_training = chunked_training;
            config.chunk_budget_gb = chunk_budget_gb;
            config.chunk_overlap_ratio = chunk_overlap_ratio;
            config.min_cameras_per_chunk = min_cameras_per_chunk;
            config.max_chunks = max_chunks;
            config.merge_core_only = if no_merge_core_only {
                false
            } else if merge_core_only {
                true
            } else {
                true
            };
            config.chunk_artifact_dir = if config.chunked_training {
                Some(default_chunk_artifact_dir(&output))
            } else {
                None
            };
            validate_chunked_training_args(&config)?;
            log_chunked_training_config(&config);

            // Train
            #[cfg(feature = "gpu")]
            {
                let scene = rustgs::train_from_slam(&slam_output, &config)?;
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
                    final_loss: 0.0,
                    gaussian_count: gaussians.len(),
                };
                rustgs::save_scene_ply(&output, &gaussians, &metadata)?;
                log::info!("Saved scene to {:?}", output);
            }

            #[cfg(not(feature = "gpu"))]
            {
                log::error!("GPU feature is required for training. Rebuild with --features gpu");
                std::process::exit(1);
            }
        }
        Commands::Render {
            input,
            camera,
            output,
        } => {
            log::info!("Rendering scene from {:?}", input);
            log::info!("Camera: {:?}", camera);
            log::info!("Output: {:?}", output);

            // Load scene
            let (gaussians, metadata) = rustgs::load_scene_ply(&input)?;
            log::info!("Loaded {} Gaussians", gaussians.len());
            let _ = metadata;

            // TODO: Load camera and render
            let _ = output;
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
    use super::{default_chunk_artifact_dir, validate_chunked_training_args, Cli, Commands};
    use clap::Parser;
    use std::path::PathBuf;

    fn parse_cli(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("cli args should parse")
    }

    #[test]
    fn train_command_parses_chunked_defaults() {
        let cli = parse_cli(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
        ]);
        let Commands::Train {
            chunked_training,
            chunk_budget_gb,
            chunk_overlap_ratio,
            min_cameras_per_chunk,
            max_chunks,
            merge_core_only,
            no_merge_core_only,
            ..
        } = cli.command
        else {
            panic!("expected train command");
        };

        assert!(!chunked_training);
        assert_eq!(chunk_budget_gb, 12.0);
        assert_eq!(chunk_overlap_ratio, 0.15);
        assert_eq!(min_cameras_per_chunk, 3);
        assert_eq!(max_chunks, 0);
        assert!(!merge_core_only);
        assert!(!no_merge_core_only);
    }

    #[test]
    fn train_command_parses_all_chunked_flags() {
        let cli = parse_cli(&[
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
        let Commands::Train {
            chunked_training,
            chunk_budget_gb,
            chunk_overlap_ratio,
            min_cameras_per_chunk,
            max_chunks,
            no_merge_core_only,
            ..
        } = cli.command
        else {
            panic!("expected train command");
        };

        assert!(chunked_training);
        assert_eq!(chunk_budget_gb, 10.5);
        assert_eq!(chunk_overlap_ratio, 0.2);
        assert_eq!(min_cameras_per_chunk, 5);
        assert_eq!(max_chunks, 8);
        assert!(no_merge_core_only);
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
}
