use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Training backend selection.
///
/// RustGS training now standardizes on the wgpu backend. The enum is kept so
/// existing config construction code does not break abruptly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingBackend {
    /// Burn + wgpu training path.
    Wgpu,
}

impl Default for TrainingBackend {
    fn default() -> Self {
        Self::Wgpu
    }
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
            .split(|ch| matches!(ch, 'x' | 'X' | ',' | ':'))
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsOpacityResetMode {
    Decay,
    Reset,
}

impl Default for LiteGsOpacityResetMode {
    fn default() -> Self {
        Self::Decay
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LiteGsPruneMode {
    Threshold,
    Weight,
}

impl Default for LiteGsPruneMode {
    fn default() -> Self {
        Self::Weight
    }
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
    pub densify_from: usize,
    pub densify_until: Option<usize>,
    /// Freeze all topology updates at or after this epoch.
    /// When set, later epochs only optimize Gaussian parameters.
    pub topology_freeze_after_epoch: Option<usize>,
    /// Brush-style refine cadence in training iterations.
    pub refine_every: usize,
    pub densification_interval: usize,
    /// Brush-style xy-gradient threshold for growth candidates.
    pub growth_grad_threshold: f32,
    /// Fraction of above-threshold candidates selected for additional growth.
    pub growth_select_fraction: f32,
    /// Stop selecting additional growth candidates after this training iteration.
    pub growth_stop_iter: usize,
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
            densify_from: 3,
            densify_until: None,
            topology_freeze_after_epoch: None,
            refine_every: 160,
            densification_interval: 5,
            growth_grad_threshold: LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
            growth_select_fraction: 0.25,
            growth_stop_iter: 15_000,
            opacity_reset_interval: 10,
            prune_offset_epochs: 0,
            prune_min_age: 5,
            prune_invisible_epochs: 10,
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
    /// Enable LiteGS-compatible SH and topology semantics.
    pub litegs_mode: bool,
    /// Nested LiteGS-compatible configuration surface.
    pub litegs: LiteGsConfig,
    /// Number of training iterations
    pub iterations: usize,
    /// Learning rate for positions (initial)
    pub lr_position: f32,
    /// Learning rate for positions (final) – exponential decay target.
    /// Default is 1/100 of lr_position. Set to 0 to disable decay.
    pub lr_pos_final: f32,
    /// Learning rate for scales
    pub lr_scale: f32,
    /// Learning rate for rotations
    pub lr_rotation: f32,
    /// Learning rate for opacities
    pub lr_opacity: f32,
    /// Learning rate for colors
    pub lr_color: f32,
    /// Densification interval
    pub densify_interval: usize,
    /// Pruning interval for Metal topology updates.
    pub prune_interval: usize,
    /// Delay topology updates until after this many training iterations.
    pub topology_warmup: usize,
    /// Emit topology scheduling/throughput logs every N scheduled checks.
    pub topology_log_interval: usize,
    /// Pruning threshold
    pub prune_threshold: f32,
    /// Legacy Metal densify threshold used by the production Metal trainer path.
    /// Kept explicit here so the runtime no longer depends on legacy/reference defaults.
    pub legacy_densify_grad_threshold: f32,
    /// Maximum Gaussian scale eligible for clone in the legacy topology path.
    pub legacy_clone_scale_threshold: f32,
    /// Minimum Gaussian scale eligible for split in the legacy topology path.
    pub legacy_split_scale_threshold: f32,
    /// Maximum Gaussian scale kept during legacy topology prune.
    pub legacy_prune_scale_threshold: f32,
    /// Maximum number of Gaussians the legacy topology path may add per update.
    pub legacy_max_densify_per_update: usize,
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::default(),
            litegs_mode: false,
            litegs: LiteGsConfig::default(),
            iterations: 30000,
            lr_position: 0.00016,
            lr_pos_final: 0.0000016,
            lr_scale: 0.005,
            lr_rotation: 0.001,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            densify_interval: 100,
            prune_interval: 100,
            topology_warmup: 100,
            topology_log_interval: 500,
            prune_threshold: 0.005,
            legacy_densify_grad_threshold: 0.0002,
            legacy_clone_scale_threshold: 0.1,
            legacy_split_scale_threshold: 0.3,
            legacy_prune_scale_threshold: 0.5,
            legacy_max_densify_per_update: 2_000,
            max_initial_gaussians: 100_000,
            sampling_step: 0,
            min_depth: 0.01,
            max_depth: 10.0,
            use_synthetic_depth: false,
            frame_cache_capacity: 8,
            frame_prefetch_ahead: 4,
            frame_shuffle_seed: 0,
            render_scale: 0.5,
        }
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
        LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsTileSize, TrainingBackend,
        TrainingConfig, LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
    };
    use std::str::FromStr;

    #[test]
    fn default_training_backend_is_wgpu() {
        assert_eq!(TrainingBackend::default(), TrainingBackend::Wgpu);
        let config = TrainingConfig::default();
        assert_eq!(config.backend, TrainingBackend::Wgpu);
        assert!(!config.litegs_mode);
        assert_eq!(config.render_scale, 0.5);
        assert_eq!(config.prune_interval, 100);
        assert_eq!(config.litegs, LiteGsConfig::default());
    }

    #[test]
    fn training_config_exposes_explicit_legacy_topology_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.legacy_densify_grad_threshold, 0.0002);
        assert_eq!(config.legacy_clone_scale_threshold, 0.1);
        assert_eq!(config.legacy_split_scale_threshold, 0.3);
        assert_eq!(config.legacy_prune_scale_threshold, 0.5);
        assert_eq!(config.legacy_max_densify_per_update, 2_000);
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
        assert_eq!(litegs.densify_from, 3);
        assert_eq!(litegs.densify_until, None);
        assert_eq!(litegs.topology_freeze_after_epoch, None);
        assert_eq!(litegs.refine_every, 160);
        assert_eq!(litegs.densification_interval, 5);
        assert_eq!(
            litegs.growth_grad_threshold,
            LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD
        );
        assert_eq!(litegs.growth_select_fraction, 0.25);
        assert_eq!(litegs.growth_stop_iter, 15_000);
        assert_eq!(litegs.opacity_reset_interval, 10);
        assert_eq!(litegs.prune_offset_epochs, 0);
        assert_eq!(litegs.prune_min_age, 5);
        assert_eq!(litegs.prune_invisible_epochs, 10);
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
    }

    #[test]
    fn default_training_config_disables_synthetic_depth() {
        assert!(!TrainingConfig::default().use_synthetic_depth);
    }
}
