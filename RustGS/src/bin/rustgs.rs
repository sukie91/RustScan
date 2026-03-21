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

        /// Maximum number of input frames to consider (0 = all)
        #[arg(long, default_value = "0")]
        max_frames: usize,

        /// Keep every Nth frame when loading a dataset directory
        #[arg(long, default_value = "1")]
        frame_stride: usize,

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
            max_frames,
            frame_stride,
            log_level,
        } => {
            // Initialize logging
            env_logger::Builder::new().parse_filters(&log_level).init();

            log::info!("Training 3DGS scene from {:?}", input);
            log::info!("Output: {:?}", output);
            log::info!("Iterations: {}", iterations);

            let slam_output = load_training_input(&input, max_frames, frame_stride)?;
            log::info!(
                "Loaded {} poses, {} map points",
                slam_output.num_poses(),
                slam_output.num_points()
            );

            // Configure training
            let mut config = rustgs::TrainingConfig::default();
            config.iterations = iterations;

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
