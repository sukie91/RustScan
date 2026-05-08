use super::{
    LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsSplitScoreMode, LiteGsTileSize,
    LiteGsTrainingProfile, TrainingBackend, TrainingConfig, TrainingRasterConfig,
    DEFAULT_RASTER_COV_BLUR, LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD,
};
use std::str::FromStr;

#[test]
fn default_training_backend_is_wgpu() {
    assert_eq!(TrainingBackend::default(), TrainingBackend::Wgpu);
    let config = TrainingConfig::default();
    assert_eq!(config.backend, TrainingBackend::Wgpu);
    assert_eq!(config.raster.render_scale, 0.5);
    assert_eq!(config.raster.raster_cov_blur, DEFAULT_RASTER_COV_BLUR);
    assert_eq!(config.raster.raster_cov_blur_final, None);
    assert_eq!(config.raster.raster_cov_blur_final_after_epoch, None);
    assert_eq!(config.loss.loss_l1_weight, 0.8);
    assert_eq!(config.loss.loss_ssim_weight, 0.2);
    assert_eq!(config.loss.loss_gradient_weight, 0.0);
    assert_eq!(config.loss.loss_robust_delta, 0.0);
    assert_eq!(config.loss.loss_outlier_threshold, 0.0);
    assert_eq!(config.loss.loss_outlier_weight, 1.0);
    assert_eq!(config.loss.loss_dynamic_mask_threshold_low, 0.0);
    assert_eq!(config.loss.loss_dynamic_mask_threshold_high, 0.0);
    assert_eq!(config.loss.loss_dynamic_mask_min_weight, 1.0);
    assert_eq!(config.loss.loss_dynamic_mask_start_epoch, None);
    assert_eq!(config.litegs, LiteGsConfig::default());
}

#[test]
fn litegs_config_defaults_match_mac_bootstrap_plan() {
    let litegs = LiteGsConfig::default();
    assert_eq!(litegs.rendering.sh_degree, 3);
    assert_eq!(litegs.rendering.tile_size, LiteGsTileSize::new(8, 16));
    assert!(!litegs.features.sparse_grad);
    assert_eq!(litegs.features.reg_weight, 0.0);
    assert!(!litegs.features.enable_transmittance);
    assert!(!litegs.features.enable_depth);
    assert_eq!(
        litegs.features.training_profile,
        LiteGsTrainingProfile::Baseline
    );
    assert_eq!(litegs.topology.densify_from, 3);
    assert_eq!(litegs.topology.densify_until, None);
    assert_eq!(litegs.topology.topology_freeze_after_epoch, None);
    assert_eq!(litegs.topology.growth_freeze_after_epoch, None);
    assert_eq!(litegs.topology.refine_every, 160);
    assert_eq!(litegs.topology.densification_interval, 5);
    assert_eq!(
        litegs.growth.growth_grad_threshold,
        LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD
    );
    assert_eq!(
        litegs.growth.split_score_mode,
        LiteGsSplitScoreMode::Baseline
    );
    assert_eq!(
        litegs.growth.split_grad_threshold,
        LITEGS_DEFAULT_GROWTH_GRAD_THRESHOLD
    );
    assert_eq!(litegs.growth.depth_scale_gamma, 0.37);
    assert_eq!(litegs.growth.growth_select_fraction, 0.25);
    assert_eq!(litegs.growth.growth_stop_iter, 15_000);
    assert_eq!(litegs.refine.opacity_decay, 0.0);
    assert_eq!(litegs.refine.scale_decay, 0.0);
    assert_eq!(litegs.topology.opacity_reset_interval, 10);
    assert_eq!(litegs.pruning.prune_offset_epochs, 0);
    assert_eq!(litegs.pruning.prune_min_age, 5);
    assert_eq!(litegs.pruning.prune_invisible_epochs, 10);
    assert_eq!(litegs.pruning.prune_opacity_threshold, 1.0 / 255.0);
    assert!(!litegs.pruning.prune_visibility_dry_run);
    assert_eq!(litegs.pruning.prune_visibility_threshold, 0.05);
    assert_eq!(litegs.pruning.prune_high_opacity_threshold, 0.80);
    assert_eq!(litegs.pruning.prune_until_epoch, None);
    assert_eq!(
        litegs.topology.opacity_reset_mode,
        LiteGsOpacityResetMode::Decay
    );
    assert_eq!(litegs.pruning.prune_mode, LiteGsPruneMode::Weight);
    assert_eq!(litegs.topology.target_primitives, 1_000_000);
    assert!(!litegs.camera.learnable_viewproj);
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
        LiteGsPruneMode::from_str("visibility-weight").unwrap(),
        LiteGsPruneMode::VisibilityWeight
    );
    assert_eq!(
        LiteGsPruneMode::from_str("visibility_weight").unwrap(),
        LiteGsPruneMode::VisibilityWeight
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
    assert!(!TrainingConfig::default().initialization.use_synthetic_depth);
}

#[test]
fn training_config_validate_rejects_invalid_render_scale() {
    let config = TrainingConfig {
        raster: TrainingRasterConfig {
            render_scale: 0.0,
            ..TrainingRasterConfig::default()
        },
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
