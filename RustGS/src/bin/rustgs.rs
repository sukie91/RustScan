//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <training_dataset_with_initial_points.json|colmap_dir|nerfstudio_dir> --output <scene.ply>
//!   rustgs render --input <scene.ply> --camera <pose.json> --output <image.png>

use anyhow::{bail, Context};
#[cfg(feature = "gpu")]
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

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
    /// Path to a COLMAP, Nerfstudio, TUM RGB-D directory, or a TrainingDataset JSON that already contains initial_points
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for trained scene (PLY)
    #[arg(short, long)]
    output: PathBuf,

    /// Reproducible RustGS training preset, e.g. tum-prefix-compact or tum-full-798-baseline
    #[arg(long)]
    train_preset: Option<TrainPreset>,

    /// Number of training iterations
    #[arg(long, default_value = "30000")]
    iterations: usize,

    /// Maximum number of Gaussians created during initialization
    #[arg(long, default_value = "100000")]
    max_initial_gaussians: usize,

    /// Deprecated: frame/depth initialization is disabled, so this flag is ignored
    #[arg(long, default_value = "0")]
    sampling_step: usize,

    /// Maximum number of input RGB frames to consider before applying --frame-stride (0 = all)
    #[arg(long, default_value = "0")]
    max_frames: usize,

    /// Keep every Nth frame within the considered prefix, e.g. 25 frames with stride 5 yields 5 training frames
    #[arg(long, default_value = "1")]
    frame_stride: usize,

    /// Comma-separated frame ids or inclusive ranges to include for training, e.g. 0-179,240-299
    #[arg(long)]
    include_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to exclude from training, e.g. 76-93,155
    #[arg(long)]
    exclude_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to repeat during training, e.g. 0-179
    #[arg(long)]
    oversample_frame_ranges: Option<String>,

    /// Total repeat count for --oversample-frame-ranges (1 = disabled)
    #[arg(long, default_value = "1")]
    oversample_frame_repeat: usize,

    /// Deterministic shuffle seed for training frame sampling (0 preserves dataset order)
    #[arg(long, default_value = "0")]
    frame_shuffle_seed: u64,

    /// Relative render scale used by training
    #[arg(long, default_value = "0.5")]
    render_scale: f32,

    /// Rasterizer 2D covariance blur floor. Lower is sharper but can alias.
    #[arg(long, default_value_t = rustgs::DEFAULT_RASTER_COV_BLUR)]
    raster_cov_blur: f32,

    /// Optional late-training covariance blur floor. Defaults to disabled.
    #[arg(long)]
    raster_cov_blur_final: Option<f32>,

    /// Epoch at which --raster-cov-blur-final becomes active. Defaults to topology freeze epoch.
    #[arg(long)]
    raster_cov_blur_final_after_epoch: Option<usize>,

    /// LiteGS SH degree
    #[arg(long, default_value = "3")]
    litegs_sh_degree: usize,

    /// LiteGS experiment profile: baseline, abs-split, abs-pixel, abs-pixel-depth
    #[arg(long, default_value_t = rustgs::LiteGsTrainingProfile::Baseline)]
    litegs_profile: rustgs::LiteGsTrainingProfile,

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

    /// Freeze LiteGS growth/densification at or after this epoch while allowing pruning to continue
    #[arg(long)]
    litegs_growth_freeze_after_epoch: Option<usize>,

    /// LiteGS densification interval
    #[arg(long, default_value = "5")]
    litegs_densification_interval: usize,

    /// LiteGS refine cadence in training iterations
    #[arg(long, default_value = "160")]
    litegs_refine_every: usize,

    /// LiteGS xy-gradient threshold for growth candidates
    #[arg(long, default_value = "0.00014")]
    litegs_growth_grad_threshold: f32,

    /// LiteGS split score mode for large Gaussians: baseline, abs, abs-pixel, abs-pixel-depth
    #[arg(long, default_value_t = rustgs::LiteGsSplitScoreMode::Baseline)]
    litegs_split_score: rustgs::LiteGsSplitScoreMode,

    /// LiteGS split score threshold used by --litegs-split-score abs
    #[arg(long, default_value = "0.00014")]
    litegs_split_grad_threshold: f32,

    /// Pixel-GS gamma for abs-pixel-depth near-camera split suppression
    #[arg(long, default_value = "0.37")]
    litegs_depth_scale_gamma: f32,

    /// LiteGS fraction of above-threshold candidates selected for extra growth
    #[arg(long, default_value = "0.25")]
    litegs_growth_select_fraction: f32,

    /// LiteGS iteration after which extra growth stops
    #[arg(long, default_value = "15000")]
    litegs_growth_stop_iter: usize,

    /// LiteGS opacity decay applied after each refine step
    #[arg(long, default_value = "0.0")]
    litegs_opacity_decay: f32,

    /// LiteGS scale decay applied after each refine step
    #[arg(long, default_value = "0.0")]
    litegs_scale_decay: f32,

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

    /// LiteGS opacity threshold for pruning
    #[arg(long, default_value = "0.0039215689")]
    litegs_prune_opacity_threshold: f32,

    /// Report visibility-aware prune candidates without pruning them
    #[arg(long, default_value_t = false)]
    litegs_prune_visibility_dry_run: bool,

    /// Actual visibility ratio below which a Gaussian is considered low-visibility
    #[arg(long, default_value = "0.05")]
    litegs_prune_visibility_threshold: f32,

    /// Opacity ceiling for conservative visibility-prune dry-run candidates
    #[arg(long, default_value = "0.80")]
    litegs_prune_high_opacity_threshold: f32,

    /// Continue LiteGS pruning before this epoch even if growth is frozen
    #[arg(long)]
    litegs_prune_until_epoch: Option<usize>,

    /// LiteGS target primitive budget
    #[arg(long, default_value = "1000000")]
    litegs_target_primitives: usize,

    /// Enable LiteGS learnable camera extrinsics
    #[arg(long, default_value_t = false)]
    litegs_learnable_viewproj: bool,

    /// LiteGS pose learning rate
    #[arg(long, default_value = "0.0001")]
    litegs_lr_pose: f32,

    /// Prune Gaussians with max scale > threshold (0 disables scale pruning)
    #[arg(long, default_value = "0.5")]
    litegs_prune_scale_threshold: f32,

    /// Position learning rate (initial)
    #[arg(long, default_value = "0.00016")]
    lr_position: f32,

    /// Position learning rate (final) - exponential decay target
    #[arg(long, default_value = "0.0000016")]
    lr_position_final: f32,

    /// Learning-rate decay horizon in iterations (0 = use total iterations)
    #[arg(long, default_value = "0")]
    lr_decay_iterations: usize,

    /// Scale learning rate
    #[arg(long, default_value = "0.005")]
    lr_scale: f32,

    /// Scale learning rate final value (0 = keep scale LR constant)
    #[arg(long, default_value = "0")]
    lr_scale_final: f32,

    /// Rotation learning rate
    #[arg(long, default_value = "0.001")]
    lr_rotation: f32,

    /// Rotation learning rate final value (0 = keep rotation LR constant)
    #[arg(long, default_value = "0")]
    lr_rotation_final: f32,

    /// Opacity learning rate
    #[arg(long, default_value = "0.05")]
    lr_opacity: f32,

    /// Opacity learning rate final value (0 = keep opacity LR constant)
    #[arg(long, default_value = "0")]
    lr_opacity_final: f32,

    /// Color/SH learning rate
    #[arg(long, default_value = "0.0025")]
    lr_color: f32,

    /// Color/SH learning rate final value (0 = keep color LR constant)
    #[arg(long, default_value = "0")]
    lr_color_final: f32,

    /// L1 image reconstruction loss weight
    #[arg(long, default_value = "0.8")]
    loss_l1_weight: f32,

    /// D-SSIM image reconstruction loss weight
    #[arg(long, default_value = "0.2")]
    loss_ssim_weight: f32,

    /// Image-gradient reconstruction loss weight for edge/detail preservation
    #[arg(long, default_value = "0.0")]
    loss_gradient_weight: f32,

    /// Robust residual delta for saturating L1 reconstruction loss (0 = exact L1)
    #[arg(long, default_value = "0.0")]
    loss_robust_delta: f32,

    /// Soft outlier residual threshold for high-residual pixel downweighting (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_outlier_threshold: f32,

    /// Gradient floor for high-residual pixels when --loss-outlier-threshold is enabled
    #[arg(long, default_value = "1.0")]
    loss_outlier_weight: f32,

    /// Low residual threshold for late-training dynamic/occlusion soft masking (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_dynamic_mask_threshold_low: f32,

    /// High residual threshold for late-training dynamic/occlusion soft masking (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_dynamic_mask_threshold_high: f32,

    /// Minimum L1 weight for high-residual dynamic/occlusion mask pixels
    #[arg(long, default_value = "1.0")]
    loss_dynamic_mask_min_weight: f32,

    /// Epoch when dynamic/occlusion masking starts; defaults to topology freeze epoch
    #[arg(long)]
    loss_dynamic_mask_start_epoch: Option<usize>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Run a post-training scene evaluation pass and emit a structured summary
    #[arg(long, default_value_t = false)]
    eval_after_train: bool,

    /// Evaluation render scale used by the post-training scene evaluation
    #[arg(long, default_value = "0.25")]
    eval_render_scale: f32,

    /// Evaluation-only rasterizer covariance blur floor. Defaults to --raster-cov-blur.
    #[arg(long)]
    eval_raster_cov_blur: Option<f32>,

    /// Maximum number of frames considered during post-training evaluation (0 = all)
    #[arg(long, default_value = "180")]
    eval_max_frames: usize,

    /// Keep every Nth frame during post-training evaluation
    #[arg(long, default_value = "30")]
    eval_frame_stride: usize,

    /// Comma-separated frame ids or inclusive ranges to include during post-training evaluation
    #[arg(long)]
    eval_include_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to exclude from post-training evaluation
    #[arg(long)]
    eval_exclude_frame_ranges: Option<String>,

    /// Number of lowest-PSNR frames to keep in the evaluation summary
    #[arg(long, default_value = "5")]
    eval_worst_frames: usize,

    /// Evaluation device used by the post-training scene evaluation
    #[arg(long, default_value = "cpu")]
    eval_device: String,

    /// Print the post-training evaluation summary as JSON
    #[arg(long, default_value_t = false)]
    eval_json: bool,

    /// Directory for post-training target/render/diff crop exports
    #[arg(long)]
    eval_crop_output_dir: Option<PathBuf>,

    /// Comma-separated frame ids to export crops for; defaults to worst evaluated frames
    #[arg(long)]
    eval_crop_frames: Option<String>,

    /// Crop rectangle in evaluation pixels as x,y,width,height; defaults to full frame
    #[arg(long)]
    eval_crop_rect: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainPreset {
    TumPrefixQuality,
    TumPrefixCompact,
    TumPrefixEfficient,
    TumFull798Baseline,
}

impl TrainPreset {
    fn as_str(self) -> &'static str {
        match self {
            Self::TumPrefixQuality => "tum-prefix-quality",
            Self::TumPrefixCompact => "tum-prefix-compact",
            Self::TumPrefixEfficient => "tum-prefix-efficient",
            Self::TumFull798Baseline => "tum-full-798-baseline",
        }
    }

    fn apply_to(self, args: &mut TrainArgs) {
        args.iterations = 8_000;
        args.frame_stride = 1;
        args.include_frame_ranges = None;
        args.exclude_frame_ranges = None;
        args.oversample_frame_ranges = None;
        args.oversample_frame_repeat = 1;
        args.frame_shuffle_seed = 0;
        args.render_scale = 0.5;
        args.lr_decay_iterations = 8_000;
        args.lr_scale_final = 0.0005;
        args.lr_rotation_final = 0.0001;
        args.lr_opacity_final = 0.005;
        args.lr_color_final = 0.00025;
        args.raster_cov_blur = 0.3;
        args.raster_cov_blur_final = None;
        args.raster_cov_blur_final_after_epoch = None;
        args.litegs_profile = rustgs::LiteGsTrainingProfile::Baseline;
        args.litegs_topology_freeze_after_epoch = Some(match self {
            Self::TumFull798Baseline => 4,
            Self::TumPrefixQuality | Self::TumPrefixCompact | Self::TumPrefixEfficient => 18,
        });
        args.litegs_growth_freeze_after_epoch = None;
        args.litegs_growth_select_fraction = match self {
            Self::TumPrefixQuality | Self::TumFull798Baseline => 0.25,
            Self::TumPrefixCompact | Self::TumPrefixEfficient => 0.14,
        };
        args.loss_l1_weight = match self {
            Self::TumPrefixEfficient => 0.9,
            Self::TumPrefixQuality | Self::TumPrefixCompact | Self::TumFull798Baseline => 0.8,
        };
        args.loss_ssim_weight = match self {
            Self::TumPrefixEfficient => 0.1,
            Self::TumPrefixQuality | Self::TumPrefixCompact | Self::TumFull798Baseline => 0.2,
        };
        args.loss_gradient_weight = 0.0;
        args.loss_robust_delta = 0.0;
        args.loss_outlier_threshold = 0.0;
        args.loss_outlier_weight = 1.0;
        args.loss_dynamic_mask_threshold_low = 0.0;
        args.loss_dynamic_mask_threshold_high = 0.0;
        args.loss_dynamic_mask_min_weight = 1.0;
        args.loss_dynamic_mask_start_epoch = None;
        args.max_frames = match self {
            Self::TumFull798Baseline => 0,
            Self::TumPrefixQuality | Self::TumPrefixCompact | Self::TumPrefixEfficient => 180,
        };
        args.eval_after_train = true;
        args.eval_render_scale = 0.25;
        args.eval_raster_cov_blur = Some(0.2);
        args.eval_max_frames = 180;
        args.eval_frame_stride = 30;
        args.eval_include_frame_ranges = None;
        args.eval_exclude_frame_ranges = None;
        args.eval_worst_frames = 5;
        args.eval_device = "cpu".to_string();
    }
}

impl std::fmt::Display for TrainPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TrainPreset {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tum-prefix-quality" => Ok(Self::TumPrefixQuality),
            "tum-prefix-compact" => Ok(Self::TumPrefixCompact),
            "tum-prefix-efficient" => Ok(Self::TumPrefixEfficient),
            "tum-full-798-baseline" | "tum-full-baseline" => Ok(Self::TumFull798Baseline),
            other => Err(format!(
                "unsupported RustGS train preset '{other}'. Expected one of: tum-prefix-quality, tum-prefix-compact, tum-prefix-efficient, tum-full-798-baseline"
            )),
        }
    }
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

    /// Rasterizer 2D covariance blur floor. Lower is sharper but can alias.
    #[arg(long, default_value_t = rustgs::DEFAULT_RASTER_COV_BLUR)]
    raster_cov_blur: f32,
}

#[derive(Debug, Clone, clap::Args)]
struct PruneSceneArgs {
    /// Path to input scene PLY file
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for pruned scene PLY
    #[arg(short, long)]
    output: PathBuf,

    /// Remove Gaussians with opacity below this value
    #[arg(long, default_value = "0.0")]
    min_opacity: f32,

    /// Remove Gaussians with any scale below this value (0 disables)
    #[arg(long, default_value = "0.0")]
    min_scale: f32,

    /// Remove Gaussians with any scale above this value (0 disables)
    #[arg(long, default_value = "0.0")]
    max_scale: f32,

    /// Remove Gaussians farther than this absolute distance from the scene center (0 disables)
    #[arg(long, default_value = "0.0")]
    max_distance_from_center: f32,

    /// Remove Gaussians farther than scene_extent * multiplier from the scene center (0 disables)
    #[arg(long, default_value = "0.0")]
    max_distance_extent_multiplier: f32,

    /// Print summary without writing output
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Print pruning summary as JSON
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, clap::Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Train a 3DGS scene from JSON input, a TUM RGB-D, Nerfstudio, or COLMAP directory
    Train(TrainArgs),

    /// Render a scene from a given viewpoint
    Render(RenderArgs),

    /// Remove low-quality splats from an existing scene PLY
    PruneScene(PruneSceneArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = <Cli as clap::Parser>::parse();

    match cli.command {
        Commands::Train(args) => train_command::run_train_command(args)?,
        Commands::Render(args) => run_render_command(args)?,
        Commands::PruneScene(args) => run_prune_scene_command(args)?,
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, Default, Serialize)]
struct PruneReasonCounts {
    invalid: usize,
    opacity: usize,
    min_scale: usize,
    max_scale: usize,
    distance: usize,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize)]
struct PruneSceneSummary {
    input_gaussians: usize,
    output_gaussians: usize,
    removed_gaussians: usize,
    scene_center: [f32; 3],
    scene_extent: f32,
    max_distance_threshold: Option<f32>,
    min_opacity: f32,
    min_scale: Option<f32>,
    max_scale: Option<f32>,
    reasons: PruneReasonCounts,
}

#[cfg(feature = "gpu")]
fn run_prune_scene_command(args: PruneSceneArgs) -> anyhow::Result<()> {
    let _ = env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .try_init();
    let (splats, mut metadata) = rustgs::load_splats_ply(&args.input)?;
    let (pruned, summary) = prune_splats(&splats, &args)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "Pruned scene: kept {} / {} Gaussians, removed {}",
            summary.output_gaussians, summary.input_gaussians, summary.removed_gaussians
        );
    }

    if args.dry_run {
        return Ok(());
    }

    if pruned.is_empty() {
        bail!("pruning removed every Gaussian; refusing to write empty scene");
    }
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    metadata.gaussian_count = pruned.len();
    metadata.sh_degree = pruned.sh_degree();
    rustgs::save_splats_ply(&args.output, &pruned, &metadata)?;
    log::info!("Saved pruned scene to {:?}", args.output);
    Ok(())
}

#[cfg(feature = "gpu")]
fn prune_splats(
    splats: &rustgs::HostSplats,
    args: &PruneSceneArgs,
) -> anyhow::Result<(rustgs::HostSplats, PruneSceneSummary)> {
    if !args.min_opacity.is_finite() || !(0.0..=1.0).contains(&args.min_opacity) {
        bail!("--min-opacity must be finite and in [0, 1]");
    }
    validate_non_negative("min-scale", args.min_scale)?;
    validate_non_negative("max-scale", args.max_scale)?;
    validate_non_negative("max-distance-from-center", args.max_distance_from_center)?;
    validate_non_negative(
        "max-distance-extent-multiplier",
        args.max_distance_extent_multiplier,
    )?;

    let view = splats.as_view();
    let row_count = splats.len();
    let sh_row_width = ((view.sh_degree + 1) * (view.sh_degree + 1)) * 3;
    let (center, extent) = scene_center_and_extent(&view);
    let max_distance_threshold = if args.max_distance_from_center > 0.0 {
        Some(args.max_distance_from_center)
    } else if args.max_distance_extent_multiplier > 0.0 {
        Some(extent * args.max_distance_extent_multiplier)
    } else {
        None
    };

    let mut kept_positions = Vec::with_capacity(view.positions.len());
    let mut kept_log_scales = Vec::with_capacity(view.log_scales.len());
    let mut kept_rotations = Vec::with_capacity(view.rotations.len());
    let mut kept_opacities = Vec::with_capacity(view.opacity_logits.len());
    let mut kept_sh_coeffs = Vec::with_capacity(view.sh_coeffs.len());
    let mut reasons = PruneReasonCounts::default();

    for idx in 0..row_count {
        let pos_base = idx * 3;
        let rot_base = idx * 4;
        let sh_base = idx * sh_row_width;
        let position = [
            view.positions[pos_base],
            view.positions[pos_base + 1],
            view.positions[pos_base + 2],
        ];
        let log_scale = [
            view.log_scales[pos_base],
            view.log_scales[pos_base + 1],
            view.log_scales[pos_base + 2],
        ];
        let rotation = [
            view.rotations[rot_base],
            view.rotations[rot_base + 1],
            view.rotations[rot_base + 2],
            view.rotations[rot_base + 3],
        ];
        let opacity_logit = view.opacity_logits[idx];
        let scale = log_scale.map(f32::exp);

        if position.iter().any(|value| !value.is_finite())
            || log_scale.iter().any(|value| !value.is_finite())
            || rotation.iter().any(|value| !value.is_finite())
            || !opacity_logit.is_finite()
            || scale.iter().any(|value| !value.is_finite())
        {
            reasons.invalid += 1;
            continue;
        }
        if sigmoid(opacity_logit) < args.min_opacity {
            reasons.opacity += 1;
            continue;
        }
        if args.min_scale > 0.0 && scale.iter().any(|value| *value < args.min_scale) {
            reasons.min_scale += 1;
            continue;
        }
        if args.max_scale > 0.0 && scale.iter().any(|value| *value > args.max_scale) {
            reasons.max_scale += 1;
            continue;
        }
        if let Some(max_distance) = max_distance_threshold {
            if distance(position, center) > max_distance {
                reasons.distance += 1;
                continue;
            }
        }

        kept_positions.extend_from_slice(&position);
        kept_log_scales.extend_from_slice(&log_scale);
        kept_rotations.extend_from_slice(&rotation);
        kept_opacities.push(opacity_logit);
        kept_sh_coeffs.extend_from_slice(&view.sh_coeffs[sh_base..sh_base + sh_row_width]);
    }

    let pruned = rustgs::HostSplats::from_raw_parts(
        kept_positions,
        kept_log_scales,
        kept_rotations,
        kept_opacities,
        kept_sh_coeffs,
        view.sh_degree,
    )?;
    let summary = PruneSceneSummary {
        input_gaussians: row_count,
        output_gaussians: pruned.len(),
        removed_gaussians: row_count.saturating_sub(pruned.len()),
        scene_center: center,
        scene_extent: extent,
        max_distance_threshold,
        min_opacity: args.min_opacity,
        min_scale: (args.min_scale > 0.0).then_some(args.min_scale),
        max_scale: (args.max_scale > 0.0).then_some(args.max_scale),
        reasons,
    };
    Ok((pruned, summary))
}

#[cfg(feature = "gpu")]
fn validate_non_negative(name: &str, value: f32) -> anyhow::Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("--{name} must be finite and >= 0");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(feature = "gpu")]
fn scene_center_and_extent(view: &rustgs::SplatView<'_>) -> ([f32; 3], f32) {
    let row_count = view.opacity_logits.len();
    if row_count == 0 {
        return ([0.0, 0.0, 0.0], 0.0);
    }
    let mut center = [0.0f32; 3];
    for idx in 0..row_count {
        let base = idx * 3;
        center[0] += view.positions[base];
        center[1] += view.positions[base + 1];
        center[2] += view.positions[base + 2];
    }
    let inv_count = 1.0 / row_count as f32;
    center[0] *= inv_count;
    center[1] *= inv_count;
    center[2] *= inv_count;

    let mut extent = 0.0f32;
    for idx in 0..row_count {
        let base = idx * 3;
        extent = extent.max(distance(
            [
                view.positions[base],
                view.positions[base + 1],
                view.positions[base + 2],
            ],
            center,
        ));
    }
    (center, extent.max(1e-6))
}

#[cfg(feature = "gpu")]
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(not(feature = "gpu"))]
fn run_prune_scene_command(_args: PruneSceneArgs) -> anyhow::Result<()> {
    bail!("prune-scene requires the gpu feature");
}

#[cfg(feature = "gpu")]
fn run_render_command(args: RenderArgs) -> anyhow::Result<()> {
    if !args.raster_cov_blur.is_finite() || args.raster_cov_blur < 0.0 {
        bail!("--raster-cov-blur must be finite and >= 0");
    }
    log::info!("Rendering scene from {:?}", args.input);
    log::info!("Camera: {:?}", args.camera);
    log::info!("Output: {:?}", args.output);
    log::info!("Raster covariance blur: {:.3}", args.raster_cov_blur);

    let (splats, metadata) = rustgs::load_splats_ply(&args.input)?;
    log::info!("Loaded {} Gaussians", splats.len());
    let _ = metadata;

    let camera = load_render_camera(&args.camera)?;
    let renderer = rustgs::GaussianRenderer::new(
        camera.intrinsics.width as usize,
        camera.intrinsics.height as usize,
    )
    .with_raster_cov_blur(args.raster_cov_blur);
    let rendered = renderer.render_splats(&splats, &camera)?;
    let image = image::RgbImage::from_raw(
        rendered.width as u32,
        rendered.height as u32,
        rendered.color,
    )
    .context("renderer produced an invalid RGB buffer")?;
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    image.save(&args.output)?;
    log::info!(
        "Rendered {}x{} image to {:?}",
        rendered.width,
        rendered.height,
        args.output
    );
    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Debug, Deserialize)]
struct RenderCameraFile {
    intrinsics: rustgs::Intrinsics,
    pose: rustgs::SE3,
    #[serde(default)]
    pose_is_world_to_camera: bool,
}

#[cfg(feature = "gpu")]
fn load_render_camera(path: &std::path::Path) -> anyhow::Result<rustgs::GaussianCamera> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read camera JSON {}", path.display()))?;
    let camera: RenderCameraFile = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse camera JSON {}", path.display()))?;
    let extrinsics = if camera.pose_is_world_to_camera {
        camera.pose
    } else {
        camera.pose.inverse()
    };
    Ok(rustgs::GaussianCamera::new(camera.intrinsics, extrinsics))
}

#[cfg(not(feature = "gpu"))]
fn run_render_command(_args: RenderArgs) -> anyhow::Result<()> {
    bail!("render requires the gpu feature");
}

#[cfg(test)]
mod tests {
    use super::{
        train_command::{
            build_training_config, effective_train_args, evaluation_dataset_load_params,
            load_training_dataset_for_training, maybe_write_litegs_parity_report,
            maybe_write_litegs_parity_report_with_manifest_dir,
        },
        Cli, Commands, PruneSceneArgs, TrainArgs, TrainPreset,
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

    fn parse_prune_scene_args(args: &[&str]) -> PruneSceneArgs {
        let cli = parse_cli(args);
        let Commands::PruneScene(args) = cli.command else {
            panic!("expected prune-scene command");
        };
        args
    }

    #[cfg(feature = "gpu")]
    fn rgb_to_sh0_value(rgb: f32) -> f32 {
        (rgb - 0.5) / 0.282_094_8
    }

    #[cfg(feature = "gpu")]
    fn opacity_to_logit(opacity: f32) -> f32 {
        let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
        (clamped / (1.0 - clamped)).ln()
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

    #[test]
    fn train_command_parses_training_defaults() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
        ]);

        assert_eq!(args.render_scale, 0.5);
        assert_eq!(args.train_preset, None);
        assert_eq!(args.include_frame_ranges, None);
        assert_eq!(args.exclude_frame_ranges, None);
        assert_eq!(args.oversample_frame_ranges, None);
        assert_eq!(args.oversample_frame_repeat, 1);
        assert_eq!(args.raster_cov_blur, rustgs::DEFAULT_RASTER_COV_BLUR);
        assert_eq!(args.raster_cov_blur_final, None);
        assert_eq!(args.raster_cov_blur_final_after_epoch, None);
        assert_eq!(args.litegs_sh_degree, 3);
        assert_eq!(args.litegs_profile, rustgs::LiteGsTrainingProfile::Baseline);
        assert_eq!(args.litegs_tile_size, rustgs::LiteGsTileSize::new(8, 16));
        assert!(!args.litegs_sparse_grad);
        assert_eq!(args.frame_shuffle_seed, 0);
        assert!(!args.eval_after_train);
        assert_eq!(args.eval_render_scale, 0.25);
        assert_eq!(args.eval_raster_cov_blur, None);
        assert_eq!(args.eval_max_frames, 180);
        assert_eq!(args.eval_frame_stride, 30);
        assert_eq!(args.eval_include_frame_ranges, None);
        assert_eq!(args.eval_exclude_frame_ranges, None);
        assert_eq!(args.eval_worst_frames, 5);
        assert_eq!(args.eval_device, "cpu");
        assert!(!args.eval_json);
        assert_eq!(args.eval_crop_output_dir, None);
        assert_eq!(args.eval_crop_frames, None);
        assert_eq!(args.eval_crop_rect, None);
        assert_eq!(args.loss_l1_weight, 0.8);
        assert_eq!(args.loss_ssim_weight, 0.2);
        assert_eq!(args.loss_gradient_weight, 0.0);
        assert_eq!(args.loss_robust_delta, 0.0);
        assert_eq!(args.loss_outlier_threshold, 0.0);
        assert_eq!(args.loss_outlier_weight, 1.0);
        assert_eq!(args.loss_dynamic_mask_threshold_low, 0.0);
        assert_eq!(args.loss_dynamic_mask_threshold_high, 0.0);
        assert_eq!(args.loss_dynamic_mask_min_weight, 1.0);
        assert_eq!(args.loss_dynamic_mask_start_epoch, None);
    }

    #[test]
    fn train_command_parses_litegs_profile() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--litegs-profile",
            "abs-pixel",
        ]);

        assert_eq!(args.litegs_profile, rustgs::LiteGsTrainingProfile::AbsPixel);
    }

    #[test]
    fn train_command_parses_visibility_weight_prune_mode() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--litegs-prune-mode",
            "visibility-weight",
        ]);
        let config = build_training_config(&args).unwrap();

        assert_eq!(
            config.litegs.prune_mode,
            rustgs::LiteGsPruneMode::VisibilityWeight
        );
    }

    #[test]
    fn train_command_applies_tum_prefix_compact_preset() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--train-preset",
            "tum-prefix-compact",
        ]);
        let args = effective_train_args(args);
        let config = build_training_config(&args).unwrap();

        assert_eq!(args.train_preset, Some(TrainPreset::TumPrefixCompact));
        assert_eq!(args.iterations, 8_000);
        assert_eq!(args.lr_decay_iterations, 8_000);
        assert_eq!(args.lr_scale_final, 0.0005);
        assert_eq!(args.lr_rotation_final, 0.0001);
        assert_eq!(args.lr_opacity_final, 0.005);
        assert_eq!(args.lr_color_final, 0.00025);
        assert_eq!(args.max_frames, 180);
        assert_eq!(args.frame_stride, 1);
        assert!(args.eval_after_train);
        assert_eq!(args.eval_raster_cov_blur, Some(0.2));
        assert_eq!(config.raster_cov_blur, 0.3);
        assert_eq!(config.litegs.topology_freeze_after_epoch, Some(18));
        assert_eq!(config.litegs.growth_select_fraction, 0.14);
        assert_eq!(config.loss_l1_weight, 0.8);
        assert_eq!(config.loss_ssim_weight, 0.2);
    }

    #[test]
    fn train_command_applies_tum_prefix_efficient_loss_preset() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--train-preset",
            "tum-prefix-efficient",
        ]);
        let args = effective_train_args(args);
        let config = build_training_config(&args).unwrap();

        assert_eq!(config.litegs.growth_select_fraction, 0.14);
        assert_eq!(config.loss_l1_weight, 0.9);
        assert_eq!(config.loss_ssim_weight, 0.1);
    }

    #[test]
    fn train_command_applies_tum_full_798_baseline_preset() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--train-preset",
            "tum-full-798-baseline",
        ]);
        let args = effective_train_args(args);
        let config = build_training_config(&args).unwrap();

        assert_eq!(args.max_frames, 0);
        assert_eq!(args.frame_stride, 1);
        assert_eq!(config.litegs.topology_freeze_after_epoch, Some(4));
        assert_eq!(config.litegs.growth_select_fraction, 0.25);
        assert_eq!(config.loss_l1_weight, 0.8);
        assert_eq!(config.loss_ssim_weight, 0.2);
    }

    #[test]
    fn litegs_profile_applies_stable_experimental_split_defaults() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--litegs-profile",
            "abs-pixel",
        ]);
        let config = build_training_config(&args).unwrap();

        assert_eq!(
            config.litegs.training_profile,
            rustgs::LiteGsTrainingProfile::AbsPixel
        );
        assert_eq!(
            config.litegs.split_score_mode,
            rustgs::LiteGsSplitScoreMode::AbsPixel
        );
        assert_eq!(config.litegs.split_grad_threshold, 0.00001);
    }

    #[test]
    fn prune_scene_command_parses_cleanup_flags() {
        let args = parse_prune_scene_args(&[
            "rustgs",
            "prune-scene",
            "--input",
            "in.ply",
            "--output",
            "out.ply",
            "--min-opacity",
            "0.01",
            "--min-scale",
            "0.0001",
            "--max-scale",
            "0.2",
            "--max-distance-extent-multiplier",
            "2.5",
            "--dry-run",
            "--json",
        ]);

        assert_eq!(args.input, PathBuf::from("in.ply"));
        assert_eq!(args.output, PathBuf::from("out.ply"));
        assert_eq!(args.min_opacity, 0.01);
        assert_eq!(args.min_scale, 0.0001);
        assert_eq!(args.max_scale, 0.2);
        assert_eq!(args.max_distance_extent_multiplier, 2.5);
        assert!(args.dry_run);
        assert!(args.json);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn prune_splats_filters_low_opacity_and_far_outliers() {
        let splats = rustgs::HostSplats::from_raw_parts(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 0.0],
            vec![0.01f32.ln(); 9],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![
                opacity_to_logit(0.5),
                opacity_to_logit(0.001),
                opacity_to_logit(0.5),
            ],
            vec![
                rgb_to_sh0_value(0.2),
                rgb_to_sh0_value(0.3),
                rgb_to_sh0_value(0.4),
                rgb_to_sh0_value(0.5),
                rgb_to_sh0_value(0.6),
                rgb_to_sh0_value(0.7),
                rgb_to_sh0_value(0.8),
                rgb_to_sh0_value(0.7),
                rgb_to_sh0_value(0.6),
            ],
            0,
        )
        .unwrap();
        let args = PruneSceneArgs {
            input: PathBuf::from("in.ply"),
            output: PathBuf::from("out.ply"),
            min_opacity: 0.01,
            min_scale: 0.0,
            max_scale: 0.0,
            max_distance_from_center: 8.0,
            max_distance_extent_multiplier: 0.0,
            dry_run: false,
            json: false,
            log_level: "error".to_string(),
        };

        let (pruned, summary) = super::prune_splats(&splats, &args).unwrap();

        assert_eq!(pruned.len(), 1);
        assert_eq!(summary.input_gaussians, 3);
        assert_eq!(summary.output_gaussians, 1);
        assert_eq!(summary.reasons.opacity, 1);
        assert_eq!(summary.reasons.distance, 1);
    }

    #[test]
    fn train_command_rejects_removed_chunked_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--chunked-training",
        ])
        .expect_err("removed chunked flag should fail");

        assert!(err.to_string().contains("--chunked-training"));
    }

    #[test]
    fn train_command_rejects_removed_chunk_size_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--metal-gaussian-chunk-size",
            "64",
        ])
        .expect_err("removed chunk-size flag should fail");

        assert!(err.to_string().contains("--metal-gaussian-chunk-size"));
    }

    #[test]
    fn train_command_rejects_removed_litegs_mode_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--litegs-mode",
        ])
        .expect_err("litegs mode flag should fail because LiteGS is now mandatory");

        assert!(err.to_string().contains("--litegs-mode"));
    }

    #[test]
    fn train_command_rejects_removed_prune_interval_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--prune-interval",
            "100",
        ])
        .expect_err("prune interval flag should fail because LiteGS refine cadence is mandatory");

        assert!(err.to_string().contains("--prune-interval"));
    }

    #[test]
    fn train_command_rejects_removed_topology_warmup_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--topology-warmup",
            "100",
        ])
        .expect_err("topology warmup flag should fail because LiteGS topology controls are nested");

        assert!(err.to_string().contains("--topology-warmup"));
    }

    #[test]
    fn train_command_rejects_removed_topology_log_interval_flag() {
        let err = Cli::try_parse_from([
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--topology-log-interval",
            "500",
        ])
        .expect_err(
            "topology log interval flag should fail because the legacy scheduler is removed",
        );

        assert!(err.to_string().contains("--topology-log-interval"));
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
            "--include-frame-ranges",
            "0-179",
            "--oversample-frame-ranges",
            "0-179",
            "--oversample-frame-repeat",
            "3",
            "--eval-after-train",
            "--eval-render-scale",
            "0.125",
            "--eval-raster-cov-blur",
            "0.2",
            "--eval-max-frames",
            "60",
            "--eval-frame-stride",
            "10",
            "--eval-include-frame-ranges",
            "0-179,240-299",
            "--eval-exclude-frame-ranges",
            "76-93,155",
            "--eval-worst-frames",
            "3",
            "--eval-device",
            "cpu",
            "--eval-json",
            "--eval-crop-output-dir",
            "crops",
            "--eval-crop-frames",
            "0,90,120",
            "--eval-crop-rect",
            "16,12,64,48",
            "--loss-dynamic-mask-threshold-low",
            "0.12",
            "--loss-dynamic-mask-threshold-high",
            "0.32",
            "--loss-dynamic-mask-min-weight",
            "0.35",
            "--loss-dynamic-mask-start-epoch",
            "18",
        ]);

        assert!(args.eval_after_train);
        assert_eq!(args.include_frame_ranges.as_deref(), Some("0-179"));
        assert_eq!(args.oversample_frame_ranges.as_deref(), Some("0-179"));
        assert_eq!(args.oversample_frame_repeat, 3);
        assert_eq!(args.eval_render_scale, 0.125);
        assert_eq!(args.eval_raster_cov_blur, Some(0.2));
        assert_eq!(args.eval_max_frames, 60);
        assert_eq!(args.eval_frame_stride, 10);
        assert_eq!(
            args.eval_include_frame_ranges.as_deref(),
            Some("0-179,240-299")
        );
        assert_eq!(args.eval_exclude_frame_ranges.as_deref(), Some("76-93,155"));
        assert_eq!(args.eval_worst_frames, 3);
        assert_eq!(args.eval_device, "cpu");
        assert!(args.eval_json);
        assert_eq!(args.eval_crop_output_dir, Some(PathBuf::from("crops")));
        assert_eq!(args.eval_crop_frames.as_deref(), Some("0,90,120"));
        assert_eq!(args.eval_crop_rect.as_deref(), Some("16,12,64,48"));
        assert_eq!(args.loss_dynamic_mask_threshold_low, 0.12);
        assert_eq!(args.loss_dynamic_mask_threshold_high, 0.32);
        assert_eq!(args.loss_dynamic_mask_min_weight, 0.35);
        assert_eq!(args.loss_dynamic_mask_start_epoch, Some(18));
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
            "--frame-shuffle-seed",
            "42",
            "--litegs-sh-degree",
            "4",
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
            "--litegs-growth-freeze-after-epoch",
            "9",
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
            "--litegs-opacity-decay",
            "0.003",
            "--litegs-scale-decay",
            "0.0015",
            "--litegs-opacity-reset-interval",
            "12",
            "--litegs-opacity-reset-mode",
            "reset",
            "--litegs-prune-mode",
            "threshold",
            "--litegs-prune-opacity-threshold",
            "0.01",
            "--litegs-prune-visibility-dry-run",
            "--litegs-prune-visibility-threshold",
            "0.07",
            "--litegs-prune-high-opacity-threshold",
            "0.2",
            "--litegs-prune-until-epoch",
            "60",
            "--litegs-target-primitives",
            "200000",
            "--litegs-learnable-viewproj",
            "--litegs-lr-pose",
            "0.0002",
            "--lr-decay-iterations",
            "10000",
            "--lr-scale-final",
            "0.0005",
            "--lr-rotation-final",
            "0.0001",
            "--lr-opacity-final",
            "0.005",
            "--lr-color-final",
            "0.00025",
        ]);
        let config = build_training_config(&args).unwrap();

        assert_eq!(config.litegs.sh_degree, 4);
        assert_eq!(config.litegs.tile_size, rustgs::LiteGsTileSize::new(16, 16));
        assert!(config.litegs.sparse_grad);
        assert_eq!(config.litegs.reg_weight, 0.01);
        assert!(config.litegs.enable_transmittance);
        assert!(config.litegs.enable_depth);
        assert_eq!(config.litegs.densify_from, 6);
        assert_eq!(config.litegs.densify_until, Some(24));
        assert_eq!(config.litegs.topology_freeze_after_epoch, Some(18));
        assert_eq!(config.litegs.growth_freeze_after_epoch, Some(9));
        assert_eq!(config.litegs.refine_every, 120);
        assert_eq!(config.litegs.densification_interval, 8);
        assert_eq!(config.litegs.growth_grad_threshold, 0.0003);
        assert_eq!(config.litegs.growth_select_fraction, 0.35);
        assert_eq!(config.litegs.growth_stop_iter, 2_400);
        assert_eq!(config.litegs.opacity_decay, 0.003);
        assert_eq!(config.litegs.scale_decay, 0.0015);
        assert_eq!(config.litegs.opacity_reset_interval, 12);
        assert_eq!(
            config.litegs.opacity_reset_mode,
            rustgs::LiteGsOpacityResetMode::Reset
        );
        assert_eq!(config.litegs.prune_mode, rustgs::LiteGsPruneMode::Threshold);
        assert_eq!(config.litegs.prune_offset_epochs, 0); // default value
        assert_eq!(config.litegs.prune_min_age, 5); // default value
        assert_eq!(config.litegs.prune_invisible_epochs, 10); // default value
        assert_eq!(config.litegs.prune_opacity_threshold, 0.01);
        assert!(config.litegs.prune_visibility_dry_run);
        assert_eq!(config.litegs.prune_visibility_threshold, 0.07);
        assert_eq!(config.litegs.prune_high_opacity_threshold, 0.2);
        assert_eq!(config.litegs.prune_until_epoch, Some(60));
        assert_eq!(config.litegs.target_primitives, 200_000);
        assert!(config.litegs.learnable_viewproj);
        assert_eq!(config.litegs.lr_pose, 0.0002);
        assert_eq!(config.frame_shuffle_seed, 42);
        assert_eq!(config.lr_decay_iterations, Some(10_000));
        assert_eq!(config.lr_scale_final, 0.0005);
        assert_eq!(config.lr_rotation_final, 0.0001);
        assert_eq!(config.lr_opacity_final, 0.005);
        assert_eq!(config.lr_color_final, 0.00025);
    }

    #[test]
    fn train_command_builds_dynamic_mask_config() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--loss-dynamic-mask-threshold-low",
            "0.12",
            "--loss-dynamic-mask-threshold-high",
            "0.32",
            "--loss-dynamic-mask-min-weight",
            "0.35",
        ]);
        let config = build_training_config(&args).unwrap();

        assert_eq!(config.loss_dynamic_mask_threshold_low, 0.12);
        assert_eq!(config.loss_dynamic_mask_threshold_high, 0.32);
        assert_eq!(config.loss_dynamic_mask_min_weight, 0.35);
        assert_eq!(config.loss_dynamic_mask_start_epoch, None);
    }

    #[test]
    fn train_preset_preserves_dynamic_mask_override() {
        let args = parse_train_args(&[
            "rustgs",
            "train",
            "--input",
            "scene.json",
            "--output",
            "scene.ply",
            "--train-preset",
            "tum-prefix-compact",
            "--loss-dynamic-mask-threshold-low",
            "0.12",
            "--loss-dynamic-mask-threshold-high",
            "0.32",
            "--loss-dynamic-mask-min-weight",
            "0.35",
        ]);
        let args = effective_train_args(args);
        let config = build_training_config(&args).unwrap();

        assert_eq!(args.max_frames, 180);
        assert_eq!(config.litegs.growth_select_fraction, 0.14);
        assert_eq!(config.loss_dynamic_mask_threshold_low, 0.12);
        assert_eq!(config.loss_dynamic_mask_threshold_high, 0.32);
        assert_eq!(config.loss_dynamic_mask_min_weight, 0.35);
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
        let config = rustgs::TrainingConfig::default();

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
        let config = rustgs::TrainingConfig::default();
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
        let config = rustgs::TrainingConfig::default();
        let evaluation_summary = rustgs::SplatEvaluationSummary {
            device: rustgs::EvaluationDevice::Cpu,
            render_scale: 0.25,
            raster_cov_blur: rustgs::DEFAULT_RASTER_COV_BLUR,
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
            sharpness_grad_ratio_mean: 0.8,
            sharpness_lap_ratio_mean: 0.6,
            worst_frames: Vec::new(),
            crop_outputs: Vec::new(),
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
            litegs: rustgs::LiteGsConfig {
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
        let config = rustgs::TrainingConfig::default();
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
        if !rustgs::gpu_available() {
            eprintln!("skipping test: GPU training is unavailable in this environment");
            return;
        }

        let tum_config = rustgs::TumRgbdConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        };
        let config = rustgs::TrainingConfig {
            iterations: 1,
            max_initial_gaussians: 2048,
            render_scale: 0.5,
            litegs: rustgs::LiteGsConfig {
                sparse_grad: true,
                enable_depth: true,
                ..rustgs::LiteGsConfig::default()
            },
            ..rustgs::TrainingConfig::default()
        };
        let output_dir = tempdir().unwrap();
        let output = output_dir.path().join("fixture-scene.ply");
        let Ok((dataset, source)) = load_training_dataset_for_training(
            &input,
            tum_config.max_frames,
            tum_config.frame_stride,
        ) else {
            eprintln!(
                "skipping test: could not load LiteGS convergence fixture at {:?}",
                input
            );
            return;
        };
        if dataset.initial_points.is_empty() {
            eprintln!(
                "skipping test: resolved {:?} as {} without sparse points; training now requires COLMAP sparse initialization",
                input, source
            );
            return;
        }

        let training_run =
            rustgs::train_splats(&dataset, &config, rustgs::TrainingOptions::default()).unwrap();
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
        assert!(report
            .notes
            .iter()
            .all(|note| !note.contains("frame-based fallback")));
    }
}
