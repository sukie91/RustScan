use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::TrainingError;

const MIN_RENDER_SCALE: f32 = 0.0625;
pub const DEFAULT_RASTER_COV_BLUR: f32 = 0.3;

/// Training backend selection.
///
/// RustGS training now standardizes on the wgpu backend. The enum is kept so
/// existing config construction code does not break abruptly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingBackend {
    /// Burn + wgpu training path.
    #[default]
    Wgpu,
}

impl std::fmt::Display for TrainingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wgpu")
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiteGsTileSize {
    pub width: usize,
    pub height: usize,
}

impl LiteGsTileSize {
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl Default for LiteGsTileSize {
    fn default() -> Self {
        Self::new(8, 16)
    }
}

impl std::fmt::Display for LiteGsTileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl FromStr for LiteGsTileSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let token = value.trim();
        let parts: Vec<&str> = token
            .split(['x', 'X', ',', ':'])
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid LiteGS tile size '{token}'. Expected formats like 8x16 or 8,16"
            ));
        }

        let width = parts[0]
            .parse::<usize>()
            .map_err(|_| format!("invalid tile width in '{token}'"))?;
        let height = parts[1]
            .parse::<usize>()
            .map_err(|_| format!("invalid tile height in '{token}'"))?;
        if width == 0 || height == 0 {
            return Err(format!(
                "invalid LiteGS tile size '{token}'. Width and height must both be > 0"
            ));
        }

        Ok(Self::new(width, height))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsOpacityResetMode {
    #[default]
    Decay,
    Reset,
}

impl std::fmt::Display for LiteGsOpacityResetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decay => write!(f, "decay"),
            Self::Reset => write!(f, "reset"),
        }
    }
}

impl FromStr for LiteGsOpacityResetMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "decay" => Ok(Self::Decay),
            "reset" => Ok(Self::Reset),
            other => Err(format!(
                "unsupported LiteGS opacity reset mode '{other}'. Expected one of: decay, reset"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsPruneMode {
    Threshold,
    #[default]
    Weight,
}

impl std::fmt::Display for LiteGsPruneMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Threshold => write!(f, "threshold"),
            Self::Weight => write!(f, "weight"),
        }
    }
}

impl FromStr for LiteGsPruneMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "threshold" => Ok(Self::Threshold),
            "weight" => Ok(Self::Weight),
            other => Err(format!(
                "unsupported LiteGS prune mode '{other}'. Expected one of: threshold, weight"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsSplitScoreMode {
    #[default]
    Baseline,
    Abs,
    AbsPixel,
    AbsPixelDepth,
}

impl std::fmt::Display for LiteGsSplitScoreMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline => write!(f, "baseline"),
            Self::Abs => write!(f, "abs"),
            Self::AbsPixel => write!(f, "abs-pixel"),
            Self::AbsPixelDepth => write!(f, "abs-pixel-depth"),
        }
    }
}

impl FromStr for LiteGsSplitScoreMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "baseline" => Ok(Self::Baseline),
            "abs" => Ok(Self::Abs),
            "abs-pixel" => Ok(Self::AbsPixel),
            "abs-pixel-depth" => Ok(Self::AbsPixelDepth),
            other => Err(format!(
                "unsupported LiteGS split score mode '{other}'. Expected one of: baseline, abs, abs-pixel, abs-pixel-depth"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsTrainingProfile {
    #[default]
    Baseline,
    AbsSplit,
    AbsPixel,
    AbsPixelDepth,
}

impl std::fmt::Display for LiteGsTrainingProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline => write!(f, "baseline"),
            Self::AbsSplit => write!(f, "abs-split"),
            Self::AbsPixel => write!(f, "abs-pixel"),
            Self::AbsPixelDepth => write!(f, "abs-pixel-depth"),
        }
    }
}

impl FromStr for LiteGsTrainingProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match normalize_config_token(value).as_str() {
            "baseline" => Ok(Self::Baseline),
            "abs-split" => Ok(Self::AbsSplit),
            "abs-pixel" => Ok(Self::AbsPixel),
            "abs-pixel-depth" => Ok(Self::AbsPixelDepth),
            other => Err(format!(
                "unsupported LiteGS profile '{other}'. Expected one of: baseline, abs-split, abs-pixel, abs-pixel-depth"
            )),
        }
    }
}

/// Nested LiteGS-compatible configuration surface.
///
/// The defaults are chosen for the phased Apple Silicon parity plan:
/// sparse-grad off and camera optimization deferred.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiteGsConfig {
    pub sh_degree: usize,
    pub tile_size: LiteGsTileSize,
    pub sparse_grad: bool,
    pub reg_weight: f32,
    pub enable_transmittance: bool,
    pub enable_depth: bool,
    #[serde(default)]
    pub training_profile: LiteGsTrainingProfile,
    pub densify_from: usize,
    pub densify_until: Option<usize>,
    /// Freeze all topology updates at or after this epoch.
    /// When set, later epochs only optimize Gaussian parameters.
    pub topology_freeze_after_epoch: Option<usize>,
    /// Freeze only growth/densification at or after this epoch.
    /// Pruning may continue according to prune_until_epoch.
    pub growth_freeze_after_epoch: Option<usize>,
    /// Brush-style refine cadence in training iterations.
    pub refine_every: usize,
    pub densification_interval: usize,
    /// Brush-style xy-gradient threshold for growth candidates.
    pub growth_grad_threshold: f32,
    /// Score used for large-Gaussian split candidates.
    pub split_score_mode: LiteGsSplitScoreMode,
    /// Threshold for split_score_mode=abs. Baseline mode ignores this field.
    pub split_grad_threshold: f32,
    /// Gamma used by abs-pixel-depth to suppress near-camera split growth.
    pub depth_scale_gamma: f32,
    /// Fraction of above-threshold candidates selected for additional growth.
    pub growth_select_fraction: f32,
    /// Stop selecting additional growth candidates after this training iteration.
    pub growth_stop_iter: usize,
    /// Brush-style opacity decay applied after each refine step.
    pub opacity_decay: f32,
    /// Brush-style scale decay applied after each refine step.
    pub scale_decay: f32,
    pub opacity_reset_interval: usize,
    /// How many epochs to offset prune from densify.
    /// Default is 0 (densify and prune in same epoch, LiteGS semantics).
    /// Set to positive value for decoupled densify/prune timing.
    pub prune_offset_epochs: usize,
    /// Minimum age (iterations) before a Gaussian is eligible for pruning.
    /// Protects newly-added Gaussians from immediate removal.
    pub prune_min_age: usize,
    /// Number of consecutive invisible epochs required before pruning.
    /// Prevents pruning based on single-frame invisibility.
    pub prune_invisible_epochs: usize,
    /// Prune Gaussians below this opacity during topology refinement.
    pub prune_opacity_threshold: f32,
    /// Continue pruning before this epoch even if growth is frozen.
    /// When unset, pruning uses the normal refine progress window.
    pub prune_until_epoch: Option<usize>,
    pub opacity_reset_mode: LiteGsOpacityResetMode,
    pub prune_mode: LiteGsPruneMode,
    pub target_primitives: usize,
    pub learnable_viewproj: bool,
    /// Learning rate for camera pose optimization (quaternion + translation).
    /// Default is 1e-4. Only used when learnable_viewproj is true.
    pub lr_pose: f32,
    /// Prune Gaussians with max(scale) > prune_scale_threshold.
    /// Default is 0.5 (Gausplat-style). Set to 0 to disable scale-based pruning.
    pub prune_scale_threshold: f32,
}

impl Default for LiteGsConfig {
    fn default() -> Self {
        Self {
            sh_degree: 3,
            tile_size: LiteGsTileSize::default(),
            sparse_grad: false,
            reg_weight: 0.0,
            enable_transmittance: false,
            enable_depth: false,
            training_profile: LiteGsTrainingProfile::Baseline,
            densify_from: 3,
            densify_until: None,
            topology_freeze_after_epoch: None,
            growth_freeze_after_epoch: None,
            refine_every: 160,
            densification_interval: 5,
            growth_grad_threshold: LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
            split_score_mode: LiteGsSplitScoreMode::Baseline,
            split_grad_threshold: LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
            depth_scale_gamma: 0.37,
            growth_select_fraction: 0.25,
            growth_stop_iter: 15_000,
            opacity_decay: 0.0,
            scale_decay: 0.0,
            opacity_reset_interval: 10,
            prune_offset_epochs: 0,
            prune_min_age: 5,
            prune_invisible_epochs: 10,
            prune_opacity_threshold: 1.0 / 255.0,
            prune_until_epoch: None,
            opacity_reset_mode: LiteGsOpacityResetMode::Decay,
            prune_mode: LiteGsPruneMode::Weight,
            target_primitives: 1_000_000,
            learnable_viewproj: false,
            lr_pose: 1e-4,
            prune_scale_threshold: 0.5,
        }
    }
}

fn normalize_config_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

const LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD: f32 = 0.00014;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Training backend implementation to use.
    ///
    /// RustGS now standardizes on the wgpu backend.
    pub backend: TrainingBackend,
    /// Nested LiteGS-compatible configuration surface.
    pub litegs: LiteGsConfig,
    /// Number of training iterations
    pub iterations: usize,
    /// Learning rate for positions (initial)
    pub lr_position: f32,
    /// Learning rate for positions (final) – exponential decay target.
    /// Default is 1/100 of lr_position. Set to 0 to disable decay.
    pub lr_pos_final: f32,
    /// Number of iterations used for exponential learning-rate decay.
    /// Defaults to total iterations when unset.
    pub lr_decay_iterations: Option<usize>,
    /// Learning rate for scales
    pub lr_scale: f32,
    /// Scale learning-rate final value. Set to 0 to keep lr_scale constant.
    pub lr_scale_final: f32,
    /// Learning rate for rotations
    pub lr_rotation: f32,
    /// Rotation learning-rate final value. Set to 0 to keep lr_rotation constant.
    pub lr_rotation_final: f32,
    /// Learning rate for opacities
    pub lr_opacity: f32,
    /// Opacity learning-rate final value. Set to 0 to keep lr_opacity constant.
    pub lr_opacity_final: f32,
    /// Learning rate for colors
    pub lr_color: f32,
    /// Color learning-rate final value. Set to 0 to keep lr_color constant.
    pub lr_color_final: f32,
    /// L1 image reconstruction loss weight.
    pub loss_l1_weight: f32,
    /// D-SSIM image reconstruction loss weight.
    pub loss_ssim_weight: f32,
    /// Image-gradient reconstruction loss weight. Disabled by default.
    pub loss_gradient_weight: f32,
    /// Saturating robust residual delta for the L1 reconstruction term.
    /// Set to 0 to use the exact L1 loss.
    pub loss_robust_delta: f32,
    /// Residual threshold for soft high-residual pixel downweighting.
    /// Set to 0 to disable.
    pub loss_outlier_threshold: f32,
    /// Gradient floor for high-residual pixels when loss_outlier_threshold is enabled.
    pub loss_outlier_weight: f32,
    /// Maximum number of Gaussians created during initialization
    pub max_initial_gaussians: usize,
    /// Sampling step for frame-to-Gaussian initialization (0 = auto)
    pub sampling_step: usize,
    /// Minimum valid depth in meters
    pub min_depth: f32,
    /// Maximum valid depth in meters
    pub max_depth: f32,
    /// Generate synthetic depth from image luminance when depth is unavailable.
    /// Disabled by default for RGB-only datasets because pseudo-depth targets
    /// can destabilize geometric optimization.
    pub use_synthetic_depth: bool,
    /// Number of decoded frames retained by the async prefetch cache.
    pub frame_cache_capacity: usize,
    /// Number of future frames queued ahead of the current cursor.
    pub frame_prefetch_ahead: usize,
    /// Deterministic shuffle seed for frame ordering. Zero preserves dataset order.
    pub frame_shuffle_seed: u64,
    /// Render scale used by the wgpu backend (relative to input resolution).
    pub render_scale: f32,
    /// 2D covariance blur floor used by the rasterizer. Lower values are sharper but can alias.
    pub raster_cov_blur: f32,
    /// Optional late-training 2D covariance blur floor.
    ///
    /// When set, training uses `raster_cov_blur` before the configured switch epoch and
    /// `raster_cov_blur_final` after it. Evaluation/rendering still use their own config.
    pub raster_cov_blur_final: Option<f32>,
    /// Epoch at which the training raster blur switches to `raster_cov_blur_final`.
    ///
    /// If unset, the switch follows `litegs.topology_freeze_after_epoch` when available.
    pub raster_cov_blur_final_after_epoch: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::default(),
            litegs: LiteGsConfig::default(),
            iterations: 30000,
            lr_position: 0.00016,
            lr_pos_final: 0.0000016,
            lr_decay_iterations: None,
            lr_scale: 0.005,
            lr_scale_final: 0.0,
            lr_rotation: 0.001,
            lr_rotation_final: 0.0,
            lr_opacity: 0.05,
            lr_opacity_final: 0.0,
            lr_color: 0.0025,
            lr_color_final: 0.0,
            loss_l1_weight: 0.8,
            loss_ssim_weight: 0.2,
            loss_gradient_weight: 0.0,
            loss_robust_delta: 0.0,
            loss_outlier_threshold: 0.0,
            loss_outlier_weight: 1.0,
            max_initial_gaussians: 100_000,
            sampling_step: 0,
            min_depth: 0.01,
            max_depth: 10.0,
            use_synthetic_depth: false,
            frame_cache_capacity: 8,
            frame_prefetch_ahead: 4,
            frame_shuffle_seed: 0,
            render_scale: 0.5,
            raster_cov_blur: DEFAULT_RASTER_COV_BLUR,
            raster_cov_blur_final: None,
            raster_cov_blur_final_after_epoch: None,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        let mut invalid = Vec::new();

        if self.iterations == 0 {
            invalid.push("iterations must be >= 1".to_string());
        }
        if self.max_initial_gaussians == 0 {
            invalid.push("max_initial_gaussians must be >= 1".to_string());
        }
        if !self.render_scale.is_finite() || !(MIN_RENDER_SCALE..=1.0).contains(&self.render_scale)
        {
            invalid.push(format!(
                "render_scale must be finite and in [{MIN_RENDER_SCALE}, 1.0]"
            ));
        }
        if !self.raster_cov_blur.is_finite() || self.raster_cov_blur < 0.0 {
            invalid.push("raster_cov_blur must be finite and >= 0".to_string());
        }
        if let Some(raster_cov_blur_final) = self.raster_cov_blur_final {
            if !raster_cov_blur_final.is_finite() || raster_cov_blur_final < 0.0 {
                invalid.push("raster_cov_blur_final must be finite and >= 0".to_string());
            }
        }
        if self.frame_cache_capacity == 0 {
            invalid.push("frame_cache_capacity must be >= 1".to_string());
        }
        if self.frame_prefetch_ahead == 0 {
            invalid.push("frame_prefetch_ahead must be >= 1".to_string());
        }
        if !self.min_depth.is_finite() || self.min_depth < 0.0 {
            invalid.push("min_depth must be finite and >= 0".to_string());
        }
        if !self.max_depth.is_finite() || self.max_depth <= self.min_depth {
            invalid.push("max_depth must be finite and greater than min_depth".to_string());
        }

        validate_lr("lr_position", self.lr_position, true, &mut invalid);
        validate_lr("lr_pos_final", self.lr_pos_final, false, &mut invalid);
        if matches!(self.lr_decay_iterations, Some(0)) {
            invalid.push("lr_decay_iterations must be >= 1 when set".to_string());
        }
        validate_lr("lr_scale", self.lr_scale, true, &mut invalid);
        validate_lr("lr_scale_final", self.lr_scale_final, false, &mut invalid);
        validate_lr("lr_rotation", self.lr_rotation, false, &mut invalid);
        validate_lr(
            "lr_rotation_final",
            self.lr_rotation_final,
            false,
            &mut invalid,
        );
        validate_lr("lr_opacity", self.lr_opacity, true, &mut invalid);
        validate_lr(
            "lr_opacity_final",
            self.lr_opacity_final,
            false,
            &mut invalid,
        );
        validate_lr("lr_color", self.lr_color, true, &mut invalid);
        validate_lr("lr_color_final", self.lr_color_final, false, &mut invalid);
        validate_lr("litegs.lr_pose", self.litegs.lr_pose, false, &mut invalid);
        validate_loss_weight("loss_l1_weight", self.loss_l1_weight, &mut invalid);
        validate_loss_weight("loss_ssim_weight", self.loss_ssim_weight, &mut invalid);
        validate_loss_weight(
            "loss_gradient_weight",
            self.loss_gradient_weight,
            &mut invalid,
        );
        validate_loss_weight("loss_robust_delta", self.loss_robust_delta, &mut invalid);
        validate_loss_weight(
            "loss_outlier_threshold",
            self.loss_outlier_threshold,
            &mut invalid,
        );
        validate_unit_interval(
            "loss_outlier_weight",
            self.loss_outlier_weight,
            &mut invalid,
        );
        if self.loss_l1_weight == 0.0
            && self.loss_ssim_weight == 0.0
            && self.loss_gradient_weight == 0.0
        {
            invalid.push("at least one image loss weight must be > 0".to_string());
        }

        if self.litegs.sh_degree == 0 {
            invalid.push("litegs.sh_degree must be >= 1".to_string());
        }
        if self.litegs.tile_size.width == 0 || self.litegs.tile_size.height == 0 {
            invalid.push("litegs.tile_size width and height must both be >= 1".to_string());
        }
        if self.litegs.refine_every == 0 {
            invalid.push("litegs.refine_every must be >= 1".to_string());
        }
        if self.litegs.densification_interval == 0 {
            invalid.push("litegs.densification_interval must be >= 1".to_string());
        }
        if !self.litegs.growth_grad_threshold.is_finite() || self.litegs.growth_grad_threshold < 0.0
        {
            invalid.push("litegs.growth_grad_threshold must be finite and >= 0".to_string());
        }
        if !self.litegs.split_grad_threshold.is_finite() || self.litegs.split_grad_threshold < 0.0 {
            invalid.push("litegs.split_grad_threshold must be finite and >= 0".to_string());
        }
        if !self.litegs.depth_scale_gamma.is_finite() || self.litegs.depth_scale_gamma <= 0.0 {
            invalid.push("litegs.depth_scale_gamma must be finite and > 0".to_string());
        }
        if !self.litegs.growth_select_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.litegs.growth_select_fraction)
        {
            invalid.push("litegs.growth_select_fraction must be in [0, 1]".to_string());
        }
        if self.litegs.growth_stop_iter == 0 {
            invalid.push("litegs.growth_stop_iter must be >= 1".to_string());
        }
        if !self.litegs.opacity_decay.is_finite()
            || !(0.0..=1.0).contains(&self.litegs.opacity_decay)
        {
            invalid.push("litegs.opacity_decay must be finite and in [0, 1]".to_string());
        }
        if !self.litegs.scale_decay.is_finite() || !(0.0..=1.0).contains(&self.litegs.scale_decay) {
            invalid.push("litegs.scale_decay must be finite and in [0, 1]".to_string());
        }
        if !self.litegs.prune_opacity_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.litegs.prune_opacity_threshold)
        {
            invalid.push("litegs.prune_opacity_threshold must be finite and in [0, 1]".to_string());
        }
        if self.litegs.opacity_reset_interval == 0 {
            invalid.push("litegs.opacity_reset_interval must be >= 1".to_string());
        }
        if self.litegs.prune_min_age == 0 {
            invalid.push("litegs.prune_min_age must be >= 1".to_string());
        }
        if self.litegs.prune_invisible_epochs == 0 {
            invalid.push("litegs.prune_invisible_epochs must be >= 1".to_string());
        }
        if self.litegs.target_primitives == 0 {
            invalid.push("litegs.target_primitives must be >= 1".to_string());
        }
        if !self.litegs.prune_scale_threshold.is_finite() || self.litegs.prune_scale_threshold < 0.0
        {
            invalid.push("litegs.prune_scale_threshold must be finite and >= 0".to_string());
        }
        if !self.litegs.reg_weight.is_finite() || self.litegs.reg_weight < 0.0 {
            invalid.push("litegs.reg_weight must be finite and >= 0".to_string());
        }

        if invalid.is_empty() {
            Ok(())
        } else {
            Err(TrainingError::InvalidInput(format!(
                "invalid training config: {}",
                invalid.join("; ")
            )))
        }
    }
}

fn validate_lr(label: &str, value: f32, require_positive: bool, invalid: &mut Vec<String>) {
    if !value.is_finite() || value < 0.0 || (require_positive && value == 0.0) {
        let qualifier = if require_positive { "> 0" } else { ">= 0" };
        invalid.push(format!("{label} must be finite and {qualifier}"));
    }
}

fn validate_loss_weight(label: &str, value: f32, invalid: &mut Vec<String>) {
    if !value.is_finite() || value < 0.0 {
        invalid.push(format!("{label} must be finite and >= 0"));
    }
}

fn validate_unit_interval(label: &str, value: f32, invalid: &mut Vec<String>) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        invalid.push(format!("{label} must be finite and in [0, 1]"));
    }
}

/// Training result.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss
    pub final_loss: f32,
    /// Number of Gaussians
    pub num_gaussians: usize,
    /// Training time in seconds
    pub training_time: f64,
}

#[cfg(test)]
mod tests {
    use super::{
        LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsSplitScoreMode,
        LiteGsTileSize, LiteGsTrainingProfile, TrainingBackend, TrainingConfig,
        DEFAULT_RASTER_COV_BLUR, LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
    };
    use std::str::FromStr;

    #[test]
    fn default_training_backend_is_wgpu() {
        assert_eq!(TrainingBackend::default(), TrainingBackend::Wgpu);
        let config = TrainingConfig::default();
        assert_eq!(config.backend, TrainingBackend::Wgpu);
        assert_eq!(config.render_scale, 0.5);
        assert_eq!(config.raster_cov_blur, DEFAULT_RASTER_COV_BLUR);
        assert_eq!(config.raster_cov_blur_final, None);
        assert_eq!(config.raster_cov_blur_final_after_epoch, None);
        assert_eq!(config.loss_l1_weight, 0.8);
        assert_eq!(config.loss_ssim_weight, 0.2);
        assert_eq!(config.loss_gradient_weight, 0.0);
        assert_eq!(config.loss_robust_delta, 0.0);
        assert_eq!(config.loss_outlier_threshold, 0.0);
        assert_eq!(config.loss_outlier_weight, 1.0);
        assert_eq!(config.litegs, LiteGsConfig::default());
    }

    #[test]
    fn litegs_config_defaults_match_mac_bootstrap_plan() {
        let litegs = LiteGsConfig::default();
        assert_eq!(litegs.sh_degree, 3);
        assert_eq!(litegs.tile_size, LiteGsTileSize::new(8, 16));
        assert!(!litegs.sparse_grad);
        assert_eq!(litegs.reg_weight, 0.0);
        assert!(!litegs.enable_transmittance);
        assert!(!litegs.enable_depth);
        assert_eq!(litegs.training_profile, LiteGsTrainingProfile::Baseline);
        assert_eq!(litegs.densify_from, 3);
        assert_eq!(litegs.densify_until, None);
        assert_eq!(litegs.topology_freeze_after_epoch, None);
        assert_eq!(litegs.growth_freeze_after_epoch, None);
        assert_eq!(litegs.refine_every, 160);
        assert_eq!(litegs.densification_interval, 5);
        assert_eq!(
            litegs.growth_grad_threshold,
            LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD
        );
        assert_eq!(litegs.split_score_mode, LiteGsSplitScoreMode::Baseline);
        assert_eq!(
            litegs.split_grad_threshold,
            LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD
        );
        assert_eq!(litegs.depth_scale_gamma, 0.37);
        assert_eq!(litegs.growth_select_fraction, 0.25);
        assert_eq!(litegs.growth_stop_iter, 15_000);
        assert_eq!(litegs.opacity_decay, 0.0);
        assert_eq!(litegs.scale_decay, 0.0);
        assert_eq!(litegs.opacity_reset_interval, 10);
        assert_eq!(litegs.prune_offset_epochs, 0);
        assert_eq!(litegs.prune_min_age, 5);
        assert_eq!(litegs.prune_invisible_epochs, 10);
        assert_eq!(litegs.prune_opacity_threshold, 1.0 / 255.0);
        assert_eq!(litegs.prune_until_epoch, None);
        assert_eq!(litegs.opacity_reset_mode, LiteGsOpacityResetMode::Decay);
        assert_eq!(litegs.prune_mode, LiteGsPruneMode::Weight);
        assert_eq!(litegs.target_primitives, 1_000_000);
        assert!(!litegs.learnable_viewproj);
    }

    #[test]
    fn litegs_enums_parse_cli_tokens() {
        assert_eq!(
            LiteGsTileSize::from_str("16x8").unwrap(),
            LiteGsTileSize::new(16, 8)
        );
        assert_eq!(
            LiteGsTileSize::from_str("16,8").unwrap(),
            LiteGsTileSize::new(16, 8)
        );
        assert_eq!(
            LiteGsOpacityResetMode::from_str("reset").unwrap(),
            LiteGsOpacityResetMode::Reset
        );
        assert_eq!(
            LiteGsPruneMode::from_str("threshold").unwrap(),
            LiteGsPruneMode::Threshold
        );
        assert_eq!(
            LiteGsSplitScoreMode::from_str("abs").unwrap(),
            LiteGsSplitScoreMode::Abs
        );
        assert_eq!(
            LiteGsSplitScoreMode::from_str("abs_pixel").unwrap(),
            LiteGsSplitScoreMode::AbsPixel
        );
        assert_eq!(
            LiteGsSplitScoreMode::from_str("abs_pixel_depth").unwrap(),
            LiteGsSplitScoreMode::AbsPixelDepth
        );
        assert_eq!(
            LiteGsTrainingProfile::from_str("abs_pixel").unwrap(),
            LiteGsTrainingProfile::AbsPixel
        );
    }

    #[test]
    fn default_training_config_disables_synthetic_depth() {
        assert!(!TrainingConfig::default().use_synthetic_depth);
    }

    #[test]
    fn training_config_validate_rejects_invalid_render_scale() {
        let config = TrainingConfig {
            render_scale: 0.0,
            ..TrainingConfig::default()
        };

        let err = config
            .validate()
            .expect_err("render scale should be rejected");
        assert!(err.to_string().contains("render_scale"));
    }

    #[test]
    fn training_config_validate_accepts_defaults() {
        TrainingConfig::default().validate().unwrap();
    }
}
