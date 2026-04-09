use super::{LiteGsConfig, TrainingConfig, TrainingProfile};
use crate::legacy::GaussianMap;
use crate::TrainingError;

#[cfg(feature = "gpu")]
use super::chunk_training::train_chunked_sequentially;
#[cfg(feature = "gpu")]
use super::execution_plan::{
    select_training_execution_plan, TrainingExecutionPlan, TrainingExecutionRoute,
};
#[cfg(feature = "gpu")]
use super::metal_trainer;
#[cfg(feature = "gpu")]
use super::splats::HostSplats;
#[cfg(feature = "gpu")]
use crate::TrainingDataset;

#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    match config.training_profile {
        TrainingProfile::LegacyMetal => train_legacy_metal(dataset, config),
        TrainingProfile::LiteGsMacV1 => train_litegs_mac_v1(dataset, config),
    }
}

#[cfg(feature = "gpu")]
#[deprecated(note = "Use train_splats(...) instead to avoid materializing a legacy GaussianMap.")]
pub fn train_scene(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let splats = train_splats(dataset, config)?;
    let mut map = GaussianMap::from_gaussians(splats.to_legacy_gaussians()?);
    map.update_states();
    Ok(map)
}

#[cfg(feature = "gpu")]
#[deprecated(note = "Use train_splats(...) instead.")]
pub fn train(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<GaussianMap, TrainingError> {
    let splats = train_splats(dataset, config)?;
    let mut map = GaussianMap::from_gaussians(splats.to_legacy_gaussians()?);
    map.update_states();
    Ok(map)
}

#[cfg(feature = "gpu")]
fn train_legacy_metal(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    match config.training_profile {
        TrainingProfile::LegacyMetal => {
            let plan = select_training_execution_plan(dataset, config)?;
            execute_training_plan(dataset, config, plan)
        }
        TrainingProfile::LiteGsMacV1 => unreachable!("validated by caller"),
    }
}

#[cfg(feature = "gpu")]
fn train_litegs_mac_v1(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    validate_litegs_mac_v1_config(config)?;
    log::info!(
        "Training with LiteGS Mac V1 profile | sh_degree={} | cluster_size={} | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | learnable_viewproj={} | lr_pose={:.6}",
        config.litegs.sh_degree,
        config.litegs.cluster_size,
        config.litegs.tile_size,
        config.litegs.sparse_grad,
        config.litegs.reg_weight,
        config.litegs.enable_transmittance,
        config.litegs.enable_depth,
        config.litegs.learnable_viewproj,
        config.litegs.lr_pose,
    );

    let plan = select_training_execution_plan(dataset, config)?;
    execute_training_plan(dataset, config, plan)
}

#[cfg(feature = "gpu")]
fn execute_training_plan(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    plan: TrainingExecutionPlan,
) -> Result<HostSplats, TrainingError> {
    match plan.route {
        TrainingExecutionRoute::Standard => metal_trainer::train_splats(dataset, config),
        TrainingExecutionRoute::ChunkedSingleChunk => {
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected single-chunk pass-through | requested_gaussians={} | affordable_gaussians={} | estimated_peak≈{:.1} GiB | effective_budget≈{:.1} GiB",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    estimate.estimated_peak_gib(),
                    estimate.effective_budget_gib(),
                );
            }
            metal_trainer::train_splats(dataset, config)
        }
        TrainingExecutionRoute::ChunkedSequential => {
            let chunk_plan = plan
                .chunk_plan
                .as_ref()
                .expect("sequential route requires chunk plan");
            if let Some(estimate) = plan.chunk_estimate.as_ref() {
                log::info!(
                    "Chunked planner selected sequential chunk execution | requested_gaussians={} | affordable_gaussians={} | chunks={} | training_chunks={}",
                    estimate.requested_initial_gaussians,
                    estimate.affordable_initial_gaussians,
                    chunk_plan.chunks.len(),
                    chunk_plan.training_chunks().count(),
                );
            }
            train_chunked_sequentially(dataset, config, chunk_plan)
        }
    }
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
