use super::{LiteGsConfig, TrainingConfig, TrainingProfile};
use crate::TrainingError;

#[cfg(feature = "gpu")]
use super::events::{
    emit_training_event, TrainingEvent, TrainingEventRoute, TrainingEventSink,
    TrainingPlanSelected, TrainingRun,
};
#[cfg(feature = "gpu")]
use super::splats::HostSplats;
#[cfg(feature = "gpu")]
use crate::TrainingDataset;

#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    train_splats_with_report(dataset, config).map(TrainingRun::into_splats)
}

#[cfg(feature = "gpu")]
pub fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let mut sink = |_event| {};
    train_splats_with_events(dataset, config, &mut sink)
}

#[cfg(feature = "gpu")]
pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    if config.training_profile == TrainingProfile::LiteGsMacV1 {
        validate_litegs_mac_v1_config(config)?;
    }
    crate::training::wgpu::train_splats_with_events(dataset, config, &mut on_event)
}

#[cfg(feature = "gpu")]
fn emit_standard_training_plan_event(sink: &mut TrainingEventSink<'_>) {
    emit_training_event(
        sink,
        TrainingEvent::PlanSelected(TrainingPlanSelected {
            route: TrainingEventRoute::Standard,
        }),
    );
}

pub(crate) fn validate_litegs_mac_v1_config(config: &TrainingConfig) -> Result<(), TrainingError> {
    let defaults = LiteGsConfig::default();
    let mut unsupported = Vec::new();

    if config.litegs.learnable_viewproj {
        // Learnable camera extrinsics is now supported (Story 3.2)
        // Uses sparse Adam for pose optimization
    }
    if config.litegs.cluster_size != defaults.cluster_size {
        // Clustered training is now supported
        // Uses spatial hash clustering and AABB frustum culling
    }
    if config.litegs.tile_size != defaults.tile_size {
        unsupported.push(format!(
            "tile_size={} overrides are reserved for later LiteGS parity work; bootstrap profile currently expects {}",
            config.litegs.tile_size, defaults.tile_size
        ));
    }
    if config.litegs.sh_degree == 0 {
        unsupported
            .push("sh_degree=0 is not supported for LiteGsMacV1; use degree >= 1".to_string());
    }
    if config.litegs.densification_interval == 0 {
        unsupported.push("densification_interval must be >= 1".to_string());
    }
    if config.litegs.refine_every == 0 {
        unsupported.push("refine_every must be >= 1".to_string());
    }
    if !config.litegs.growth_grad_threshold.is_finite() || config.litegs.growth_grad_threshold < 0.0
    {
        unsupported.push("growth_grad_threshold must be finite and >= 0".to_string());
    }
    if !config.litegs.growth_select_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.litegs.growth_select_fraction)
    {
        unsupported.push("growth_select_fraction must be in [0, 1]".to_string());
    }
    if config.litegs.growth_stop_iter == 0 {
        unsupported.push("growth_stop_iter must be >= 1".to_string());
    }
    if config.litegs.opacity_reset_interval == 0 {
        unsupported.push("opacity_reset_interval must be >= 1".to_string());
    }
    if config.litegs.target_primitives == 0 {
        unsupported.push("target_primitives must be >= 1".to_string());
    }
    if config.litegs.prune_min_age == 0 {
        unsupported.push("prune_min_age must be >= 1 to protect newly-added Gaussians".to_string());
    }
    if config.litegs.prune_invisible_epochs == 0 {
        unsupported.push("prune_invisible_epochs must be >= 1".to_string());
    }

    if unsupported.is_empty() {
        return Ok(());
    }

    Err(TrainingError::TrainingFailed(format!(
        "LiteGsMacV1 bootstrap profile rejected unsupported overrides: {}",
        unsupported.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::validate_litegs_mac_v1_config;
    use crate::training::{
        LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, TrainingConfig, TrainingProfile,
    };

    #[test]
    fn litegs_mac_v1_accepts_bootstrap_defaults() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        };
        validate_litegs_mac_v1_config(&config).unwrap();
    }

    #[test]
    fn litegs_mac_v1_accepts_wired_non_clustered_overrides() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                reg_weight: 0.01,
                enable_transmittance: true,
                densify_from: 4,
                densify_until: Some(16),
                refine_every: 64,
                densification_interval: 3,
                growth_grad_threshold: 0.001,
                growth_select_fraction: 0.35,
                growth_stop_iter: 2_048,
                opacity_reset_interval: 6,
                opacity_reset_mode: LiteGsOpacityResetMode::Reset,
                prune_mode: LiteGsPruneMode::Threshold,
                target_primitives: 42_000,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        validate_litegs_mac_v1_config(&config).unwrap();
    }

    #[test]
    fn litegs_mac_v1_accepts_clustered_override() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                cluster_size: 128,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        assert!(validate_litegs_mac_v1_config(&config).is_ok());
    }

    #[test]
    fn litegs_mac_v1_accepts_sparse_grad_override() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                sparse_grad: true,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        assert!(validate_litegs_mac_v1_config(&config).is_ok());
    }

    #[test]
    fn litegs_mac_v1_accepts_enable_depth_override() {
        let config = TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                enable_depth: true,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        assert!(validate_litegs_mac_v1_config(&config).is_ok());
    }
}
