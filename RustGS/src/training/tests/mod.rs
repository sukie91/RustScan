mod pose_utils;

#[cfg(feature = "gpu")]
mod config_validation {
    use super::super::validate_litegs_mac_v1_config;
    use crate::training::{LiteGsConfig, LiteGsOpacityResetMode, LiteGsPruneMode, TrainingConfig};

    #[test]
    fn litegs_mac_v1_accepts_bootstrap_defaults() {
        let config = TrainingConfig::default();
        validate_litegs_mac_v1_config(&config).unwrap();
    }

    #[test]
    fn litegs_mac_v1_accepts_wired_non_clustered_overrides() {
        let config = TrainingConfig {
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
    fn litegs_mac_v1_accepts_sparse_grad_override() {
        let config = TrainingConfig {
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
            litegs: LiteGsConfig {
                enable_depth: true,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        };

        assert!(validate_litegs_mac_v1_config(&config).is_ok());
    }
}
