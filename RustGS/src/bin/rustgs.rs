//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <slam_output.json|tum_dataset_dir> --output <scene.ply>
//!   rustgs render --input <scene.ply> --camera <pose.json> --output <image.png>

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
        #[arg(long, default_value = "0.25")]
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
