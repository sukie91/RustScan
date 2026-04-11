use super::super::backward as metal_backward;
use super::*;

fn max_slice_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn make_test_camera(device: &Device) -> DiffCamera {
    DiffCamera::new(
        64.0,
        64.0,
        32.0,
        32.0,
        64,
        64,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        device,
    )
    .unwrap()
}

fn projected_with_visible_sources(
    device: &Device,
    visible_source_indices: &[u32],
) -> ProjectedGaussians {
    let visible_count = visible_source_indices.len();
    ProjectedGaussians {
        source_indices: if visible_count == 0 {
            Tensor::zeros((0,), DType::U32, device).unwrap()
        } else {
            Tensor::from_slice(visible_source_indices, visible_count, device).unwrap()
        },
        u: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        v: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        sigma_x: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        sigma_y: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        raw_sigma_x: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        raw_sigma_y: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        depth: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        opacity: Tensor::ones((visible_count,), DType::F32, device).unwrap(),
        opacity_logits: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        scale3d: Tensor::ones((visible_count, 3), DType::F32, device).unwrap(),
        colors: Tensor::zeros((visible_count, 3), DType::F32, device).unwrap(),
        min_x: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        max_x: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        min_y: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        max_y: Tensor::zeros((visible_count,), DType::F32, device).unwrap(),
        visible_source_indices: visible_source_indices.to_vec(),
        visible_count,
        tile_bins: ProjectedTileBins::default(),
        staging_source: ProjectionStagingSource::TensorReadback,
    }
}

#[test]
fn update_gaussian_stats_uses_projected_grad_magnitudes_for_litegs() {
    let device = Device::Cpu;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &config, device.clone()).unwrap();
    let projected = projected_with_visible_sources(&device, &[1]);

    trainer
        .update_gaussian_stats(&[1e-9, 2e-9], &[0.0, 0.0035], &projected, 2)
        .unwrap();

    let expected = 0.0035 * trainer.pixel_count as f32;
    assert_eq!(trainer.gaussian_stats.len(), 2);
    assert!(trainer.gaussian_stats[0].mean2d_grad.mean.abs() < 1e-12);
    assert!((trainer.gaussian_stats[1].mean2d_grad.mean - expected).abs() < 1e-6);
    assert_eq!(trainer.gaussian_stats[1].visible_count, 1);
    assert_eq!(trainer.gaussian_stats[1].consecutive_invisible_epochs, 0);
    assert!((trainer.gaussian_stats[1].fragment_weight.mean - 1.0).abs() < 1e-6);
    assert!((trainer.gaussian_stats[1].fragment_err.mean - expected).abs() < 1e-6);
}

#[test]
fn update_gaussian_stats_keeps_legacy_param_gradient_path() {
    let device = Device::Cpu;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LegacyMetal,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &config, device.clone()).unwrap();
    let projected = projected_with_visible_sources(&device, &[0]);

    trainer
        .update_gaussian_stats(&[0.001], &[9.0], &projected, 1)
        .unwrap();

    let expected = (0.001 * trainer.pixel_count as f32).min(10.0);
    assert!((trainer.gaussian_stats[0].mean2d_grad.mean - expected).abs() < 1e-6);
    assert_eq!(trainer.gaussian_stats[0].visible_count, 1);
    assert_eq!(trainer.gaussian_stats[0].consecutive_invisible_epochs, 0);
}

#[test]
fn scaled_dimensions_keep_minimum_size() {
    assert_eq!(scaled_dimensions(640, 480, 0.25), (160, 120));
    assert_eq!(scaled_dimensions(1, 1, 0.0), (1, 1));
}

#[test]
fn resize_depth_ignores_invalid_values() {
    let src = vec![1.0, 0.0, 3.0, f32::NAN];
    let resized = resize_depth(&src, 2, 2, 1, 1);
    assert_eq!(resized.len(), 1);
    assert!((resized[0] - 2.0).abs() < 1e-6);
}

#[test]
fn depth_backward_scale_uses_only_valid_depth_samples() {
    let target_depth = [1.0f32, 0.0, f32::NAN, 2.0];
    let scale = depth_backward_scale(LITEGS_DEPTH_LOSS_WEIGHT, &target_depth);
    assert!((scale - (LITEGS_DEPTH_LOSS_WEIGHT / 2.0)).abs() < 1e-6);
    assert_eq!(
        depth_backward_scale(LITEGS_DEPTH_LOSS_WEIGHT, &[0.0, f32::NAN]),
        0.0
    );
}

#[test]
fn loss_curve_sampling_captures_bootstrap_interval_and_final_step() {
    assert!(should_record_loss_curve_sample(0, 100));
    assert!(should_record_loss_curve_sample(4, 100));
    assert!(!should_record_loss_curve_sample(5, 100));
    assert!(should_record_loss_curve_sample(25, 100));
    assert!(should_record_loss_curve_sample(99, 100));
}

#[test]
fn metal_config_uses_safer_default_budget() {
    let effective = effective_metal_config(&TrainingConfig::default());
    assert_eq!(
        effective.max_initial_gaussians,
        TrainingConfig::default().max_initial_gaussians
    );
    assert_eq!(effective.lr_rotation, 0.0);
}

#[test]
fn metal_config_freezes_rotation_learning_for_legacy_profile() {
    let effective = effective_metal_config(&TrainingConfig {
        lr_rotation: 0.25,
        ..TrainingConfig::default()
    });
    assert_eq!(effective.lr_rotation, 0.0);
}

#[test]
fn litegs_profile_preserves_rotation_learning_rate() {
    let effective = effective_metal_config(&TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        lr_rotation: 0.25,
        ..TrainingConfig::default()
    });
    assert_eq!(effective.lr_rotation, 0.25);
}

#[test]
fn litegs_profile_uses_litegs_opacity_lr_default() {
    let effective = effective_metal_config(&TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        ..TrainingConfig::default()
    });
    assert_eq!(effective.lr_opacity, 0.025);
}

#[test]
fn trainer_disables_native_forward_on_cpu() {
    let trainer = MetalTrainer::new(32, 16, &TrainingConfig::default(), Device::Cpu).unwrap();
    assert!(!trainer.use_native_forward);
}

#[test]
fn trainer_respects_explicit_native_forward_disable() {
    let config = TrainingConfig {
        metal_use_native_forward: false,
        ..TrainingConfig::default()
    };
    let trainer = MetalTrainer::new(32, 16, &config, Device::Cpu).unwrap();
    assert!(!trainer.use_native_forward);
}

#[test]
fn trainer_uses_explicit_legacy_topology_thresholds_from_config() {
    let config = TrainingConfig {
        legacy_densify_grad_threshold: 0.0125,
        legacy_clone_scale_threshold: 0.22,
        legacy_split_scale_threshold: 0.44,
        legacy_prune_scale_threshold: 0.66,
        legacy_max_densify_per_update: 77,
        ..TrainingConfig::default()
    };

    let trainer = MetalTrainer::new(32, 16, &config, Device::Cpu).unwrap();

    assert_eq!(trainer.legacy_densify_grad_threshold, 0.0125);
    assert_eq!(trainer.legacy_clone_scale_threshold, 0.22);
    assert_eq!(trainer.legacy_split_scale_threshold, 0.44);
    assert_eq!(trainer.legacy_prune_scale_threshold, 0.66);
    assert_eq!(trainer.legacy_max_densify_per_update, 77);
}

#[test]
fn litegs_late_stage_start_epoch_clamps_to_topology_window() {
    let trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 1_200,
            litegs: LiteGsConfig {
                densify_from: 3,
                densify_until: Some(11),
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();

    assert_eq!(trainer.litegs_total_epochs(90), 13);
    assert_eq!(trainer.litegs_densify_until_epoch(90), 11);
    assert_eq!(trainer.litegs_late_stage_start_epoch(90), 8);
}

#[test]
fn litegs_short_run_topology_window_compresses_to_single_epoch() {
    let trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            iterations: 500,
            litegs: LiteGsConfig {
                densify_from: 3,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();

    assert_eq!(trainer.litegs_total_epochs(638), 1);
    assert_eq!(trainer.litegs_effective_densify_from_epoch(638), 0);
    assert_eq!(trainer.litegs_densify_until_epoch(638), 1);
    assert_eq!(trainer.litegs_late_stage_start_epoch(638), 0);
}

#[test]
fn litegs_topology_metrics_capture_late_stage_activity() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 3,
        topology_warmup: 0,
        max_initial_gaussians: 4,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(3),
            refine_every: 1,
            densification_interval: 1,
            opacity_reset_interval: 1,
            prune_min_age: 1,
            prune_invisible_epochs: 1,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    trainer.iteration = 3;

    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        &[
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
        ],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[2.0, -10.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            visible_count: 1,
            age: 1,
            ..Default::default()
        },
        MetalGaussianStats {
            age: 1,
            ..Default::default()
        },
    ];

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert_eq!(trainer.topology_metrics.total_epochs, Some(3));
    assert_eq!(trainer.topology_metrics.densify_until_epoch, Some(3));
    assert_eq!(trainer.topology_metrics.late_stage_start_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.first_densify_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.last_densify_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.late_stage_densify_events, 1);
    assert_eq!(trainer.topology_metrics.late_stage_densify_added, 1);
    assert_eq!(trainer.topology_metrics.first_prune_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.last_prune_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.late_stage_prune_events, 1);
    assert_eq!(trainer.topology_metrics.late_stage_prune_removed, 1);
    assert_eq!(trainer.topology_metrics.first_opacity_reset_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.last_opacity_reset_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.late_stage_opacity_reset_events, 1);
}

#[test]
fn litegs_topology_freeze_after_epoch_skips_late_stage_updates() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 3,
        topology_warmup: 0,
        max_initial_gaussians: 4,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(3),
            topology_freeze_after_epoch: Some(2),
            refine_every: 1,
            densification_interval: 1,
            opacity_reset_interval: 1,
            prune_min_age: 1,
            prune_invisible_epochs: 1,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    trainer.iteration = 3;

    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0],
        &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[2.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![MetalGaussianStats {
        mean2d_grad: RunningMoments {
            mean: 1.0,
            m2: 0.0,
            count: 1,
        },
        age: 1,
        ..Default::default()
    }];

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert_eq!(gaussians.len(), 1);
    assert_eq!(trainer.topology_metrics.topology_freeze_epoch, Some(2));
    assert_eq!(trainer.topology_metrics.densify_events, 0);
    assert_eq!(trainer.topology_metrics.prune_events, 0);
    assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
}

#[test]
fn litegs_topology_warmup_blocks_epoch_based_updates() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 20,
        topology_warmup: 10,
        max_initial_gaussians: 4,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(6),
            refine_every: 1,
            densification_interval: 1,
            opacity_reset_interval: 1,
            prune_min_age: 1,
            prune_invisible_epochs: 1,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    trainer.iteration = 6;

    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0],
        &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[2.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![MetalGaussianStats {
        mean2d_grad: RunningMoments {
            mean: 1.0,
            m2: 0.0,
            count: 1,
        },
        age: 1,
        ..Default::default()
    }];

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert_eq!(gaussians.len(), 1);
    assert_eq!(trainer.topology_metrics.densify_events, 0);
    assert_eq!(trainer.topology_metrics.prune_events, 0);
    assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
}

#[test]
fn litegs_topology_skips_prune_without_growth_candidates() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 20,
        topology_warmup: 0,
        max_initial_gaussians: 4,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(6),
            refine_every: 1,
            densification_interval: 1,
            opacity_reset_interval: 1,
            prune_min_age: 1,
            prune_invisible_epochs: 1,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    trainer.iteration = 6;

    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0],
        &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[-10.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![MetalGaussianStats {
        age: 1,
        ..Default::default()
    }];

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert_eq!(gaussians.len(), 1);
    assert_eq!(trainer.topology_metrics.densify_events, 0);
    assert_eq!(trainer.topology_metrics.prune_events, 0);
    assert_eq!(trainer.topology_metrics.opacity_reset_events, 0);
}

#[test]
fn projected_axis_aligned_sigmas_change_with_rotation() {
    let camera_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let identity = projected_axis_aligned_sigmas(
        0.0,
        0.0,
        2.0,
        [0.4, 0.05, 0.05],
        [1.0, 0.0, 0.0, 0.0],
        &camera_rotation,
        64.0,
        64.0,
    );
    let rotated = projected_axis_aligned_sigmas(
        0.0,
        0.0,
        2.0,
        [0.4, 0.05, 0.05],
        [
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
        ],
        &camera_rotation,
        64.0,
        64.0,
    );

    assert!(identity.0 > identity.1 * 5.0, "{identity:?}");
    assert!(rotated.1 > rotated.0 * 5.0, "{rotated:?}");
    assert!(
        (identity.0 - rotated.0).abs() > 1.0,
        "{identity:?} vs {rotated:?}"
    );
}

#[test]
fn project_gaussians_uses_rotation_aware_footprints() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        64.0,
        64.0,
        32.0,
        32.0,
        64,
        64,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();

    let identity = Splats::new(
        &[0.0, 0.0, 2.0],
        &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let rotated = Splats::new(
        &[0.0, 0.0, 2.0],
        &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
        ],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();

    let (identity_projected, _) = trainer
        .project_gaussians(&identity, &camera, false, true, None)
        .unwrap();
    let (rotated_projected, _) = trainer
        .project_gaussians(&rotated, &camera, false, true, None)
        .unwrap();
    let identity_sigma_x = identity_projected.sigma_x.to_vec1::<f32>().unwrap()[0];
    let identity_sigma_y = identity_projected.sigma_y.to_vec1::<f32>().unwrap()[0];
    let rotated_sigma_x = rotated_projected.sigma_x.to_vec1::<f32>().unwrap()[0];
    let rotated_sigma_y = rotated_projected.sigma_y.to_vec1::<f32>().unwrap()[0];

    assert!(identity_sigma_x > identity_sigma_y * 5.0);
    assert!(rotated_sigma_y > rotated_sigma_x * 5.0);
    assert!((identity_sigma_x - rotated_sigma_x).abs() > 1.0);
}

#[test]
fn rotation_parameter_grads_become_nonzero_for_asymmetric_color_error() {
    let device = Device::Cpu;
    let base_z = 0.25f32;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        lr_rotation: 0.1,
        metal_render_scale: 1.0,
        metal_use_native_forward: false,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        96.0,
        48.0,
        32.0,
        32.0,
        64,
        64,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[0.25, 0.0, 2.0],
        &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, base_z],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();

    let (rendered, projected, _) = trainer
        .render(&gaussians, &camera, false, true, None)
        .unwrap();
    let rendered_color_cpu = rendered
        .color
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut target_color = rendered_color_cpu.clone();
    let u = projected.u.to_vec1::<f32>().unwrap()[0];
    let v = projected.v.to_vec1::<f32>().unwrap()[0];
    let target_px = (u + 2.0).round().clamp(0.0, 63.0) as usize;
    let target_py = v.round().clamp(0.0, 63.0) as usize;
    let target_idx = (target_py * 64 + target_px) * 3;
    target_color[target_idx] = (target_color[target_idx] - 0.2).max(0.0);
    let target_depth = vec![0.0; trainer.pixel_count];
    let ssim_grads = vec![0.0; rendered_color_cpu.len()];
    let rotation_grads = trainer
        .rotation_parameter_grads(
            &gaussians,
            &projected,
            &rendered,
            &rendered_color_cpu,
            &target_color,
            &target_depth,
            &ssim_grads,
            MetalBackwardLossScales {
                color: 1.0,
                depth: 0.0,
                ssim: 0.0,
                alpha: 0.0,
            },
            &camera,
        )
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let analytic = rotation_grads[0][3];

    assert!(analytic.abs() > 1e-4, "analytic={analytic}");
}

#[test]
fn apply_backward_grads_updates_rotations_when_rotation_grad_is_present() {
    let device = Device::Cpu;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        lr_rotation: 0.1,
        metal_render_scale: 1.0,
        metal_use_native_forward: false,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.iteration = 1;
    let camera = DiffCamera::new(
        64.0,
        64.0,
        32.0,
        32.0,
        64,
        64,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 2.0],
        &[0.4f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    let (_, projected, _) = trainer
        .render(&gaussians, &camera, false, true, None)
        .unwrap();
    let zero_grads = MetalBackwardGrads {
        positions: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
        log_scales: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
        opacity_logits: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        colors: Tensor::zeros((1, 3), DType::F32, &device).unwrap(),
    };
    let rotation_grads = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0], (1, 4), &device).unwrap();
    let before = gaussians.rotations.as_tensor().to_vec2::<f32>().unwrap();

    trainer
        .apply_backward_grads(
            &mut gaussians,
            &zero_grads,
            &projected,
            &camera,
            0.0,
            None,
            Some(&rotation_grads),
        )
        .unwrap();

    let after = gaussians.rotations.as_tensor().to_vec2::<f32>().unwrap();
    assert!(
        after[0][3] < before[0][3],
        "before={before:?} after={after:?}"
    );
}

#[test]
fn apply_backward_grads_sparse_grad_preserves_invisible_rows_and_moments() {
    let device = Device::Cpu;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            sparse_grad: true,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.iteration = 3;
    let camera = make_test_camera(&device);
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    let adam = trainer.adam.as_mut().unwrap();
    adam.m_pos =
        Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.5, -0.25, 0.75], (2, 3), &device).unwrap();
    adam.v_pos = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.4, 0.2, 0.6], (2, 3), &device).unwrap();
    let projected = projected_with_visible_sources(&device, &[0]);
    let grads = MetalBackwardGrads {
        positions: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &device).unwrap(),
        log_scales: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        opacity_logits: Tensor::zeros((2,), DType::F32, &device).unwrap(),
        colors: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
    };
    let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();

    trainer
        .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
        .unwrap();

    let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
    let after_m_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .m_pos
        .to_vec2::<f32>()
        .unwrap();
    let after_v_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .v_pos
        .to_vec2::<f32>()
        .unwrap();

    assert!(after_positions[0][0] < before_positions[0][0]);
    assert_eq!(after_positions[1], before_positions[1]);
    assert!(after_m_pos[0][0].abs() > 1e-6);
    assert_eq!(after_m_pos[1], vec![0.5, -0.25, 0.75]);
    assert_eq!(after_v_pos[1], vec![0.4, 0.2, 0.6]);
}

#[test]
fn apply_backward_grads_sparse_grad_noops_when_no_gaussians_are_visible() {
    let device = Device::Cpu;
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            sparse_grad: true,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.iteration = 4;
    let camera = make_test_camera(&device);
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    let adam = trainer.adam.as_mut().unwrap();
    adam.m_pos =
        Tensor::from_slice(&[0.3f32, -0.1, 0.2, 0.5, -0.25, 0.75], (2, 3), &device).unwrap();
    adam.v_pos = Tensor::from_slice(&[0.4f32, 0.2, 0.1, 0.4, 0.2, 0.6], (2, 3), &device).unwrap();
    let projected = projected_with_visible_sources(&device, &[]);
    let grads = MetalBackwardGrads {
        positions: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        log_scales: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
        opacity_logits: Tensor::zeros((2,), DType::F32, &device).unwrap(),
        colors: Tensor::zeros((2, 3), DType::F32, &device).unwrap(),
    };
    let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();
    let before_m_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .m_pos
        .to_vec2::<f32>()
        .unwrap();
    let before_v_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .v_pos
        .to_vec2::<f32>()
        .unwrap();

    trainer
        .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
        .unwrap();

    let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
    let after_m_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .m_pos
        .to_vec2::<f32>()
        .unwrap();
    let after_v_pos = trainer
        .adam
        .as_ref()
        .unwrap()
        .v_pos
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(after_positions, before_positions);
    assert_eq!(after_m_pos, before_m_pos);
    assert_eq!(after_v_pos, before_v_pos);
}

#[test]
fn litegs_loss_weights_only_enable_depth_when_requested() {
    let disabled = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                enable_depth: false,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();
    let enabled = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: LiteGsConfig {
                enable_depth: true,
                ..LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();

    assert_eq!(disabled.loss_weights(), (0.8, 0.2, 0.0));
    assert_eq!(enabled.loss_weights(), (0.8, 0.2, LITEGS_DEPTH_LOSS_WEIGHT));
}

#[test]
fn training_step_records_depth_telemetry_with_clustered_sparse_grad() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            cluster_size: 1,
            sparse_grad: true,
            enable_depth: true,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.scene_extent = 16.0;
    let camera = make_test_camera(&device);
    let mut gaussians = Splats::new(
        &[
            0.0, 0.0, 2.0, //
            0.0, 0.0, -2.0,
        ],
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        &[
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ],
        &[0.0, 0.0],
        &[
            1.0, 0.25, 0.25, //
            0.1, 1.0, 0.1,
        ],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.sync_cluster_assignment(&gaussians, false).unwrap();
    let cluster_visible_mask = trainer.cluster_visible_mask_for_camera(gaussians.len(), &camera);
    let (rendered, _, _) = trainer
        .render(
            &gaussians,
            &camera,
            false,
            true,
            cluster_visible_mask.as_deref(),
        )
        .unwrap();
    let target_depth_cpu = rendered
        .depth
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let target_color_cpu = vec![0.0f32; trainer.pixel_count * 3];
    let frame = MetalTrainingFrame {
        camera: camera.clone(),
        target_color: Tensor::from_slice(&target_color_cpu, (trainer.pixel_count, 3), &device)
            .unwrap(),
        target_depth: rendered.depth.clone(),
        target_color_cpu,
        target_depth_cpu,
    };

    let outcome = trainer
        .training_step(&mut gaussians, &frame, 0, 1, false)
        .unwrap();
    let telemetry = trainer.current_telemetry(1);
    let color_moments = trainer
        .adam
        .as_ref()
        .unwrap()
        .m_color
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(outcome.visible_gaussians, 1);
    assert!(telemetry.loss_terms.total.unwrap_or(0.0) > 0.0);
    assert!(telemetry.loss_terms.depth.is_some());
    assert_eq!(telemetry.depth_valid_pixels, Some(trainer.pixel_count));
    assert_eq!(
        telemetry.depth_grad_scale,
        Some(LITEGS_DEPTH_LOSS_WEIGHT / trainer.pixel_count as f32)
    );
    assert_eq!(telemetry.learning_rates.xyz, Some(trainer.compute_lr_pos()));
    assert!(color_moments[0].iter().any(|value| value.abs() > 1e-8));
    assert!(color_moments[1].iter().all(|value| value.abs() < 1e-8));
    assert!(trainer.cluster_assignment.is_some());
}

#[test]
fn training_step_updates_dense_litegs_params_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            cluster_size: 1,
            sparse_grad: false,
            enable_depth: true,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.scene_extent = 16.0;
    let camera = make_test_camera(&device);
    let mut gaussians = Splats::new(
        &[
            0.0, 0.0, 2.0, //
            0.0, 0.0, -2.0,
        ],
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        &[
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ],
        &[0.0, 0.0],
        &[
            1.0, 0.25, 0.25, //
            0.1, 1.0, 0.1,
        ],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.sync_cluster_assignment(&gaussians, false).unwrap();
    let cluster_visible_mask = trainer.cluster_visible_mask_for_camera(gaussians.len(), &camera);
    let (rendered, _, _) = trainer
        .render(
            &gaussians,
            &camera,
            false,
            true,
            cluster_visible_mask.as_deref(),
        )
        .unwrap();
    let target_depth_cpu = rendered
        .depth
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let target_color_cpu = vec![0.0f32; trainer.pixel_count * 3];
    let frame = MetalTrainingFrame {
        camera: camera.clone(),
        target_color: Tensor::from_slice(&target_color_cpu, (trainer.pixel_count, 3), &device)
            .unwrap(),
        target_depth: rendered.depth.clone(),
        target_color_cpu,
        target_depth_cpu,
    };
    let before_positions = gaussians
        .positions()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_scales = gaussians
        .scales
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_opacities = gaussians
        .opacities
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_colors = gaussians
        .colors()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let outcome = trainer
        .training_step(&mut gaussians, &frame, 0, 1, false)
        .unwrap();
    let telemetry = trainer.current_telemetry(1);
    let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
    let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
    let opacity_delta = max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
    let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();
    let color_moments = trainer
        .adam
        .as_ref()
        .unwrap()
        .m_color
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(outcome.visible_gaussians, 1);
    assert!(telemetry.loss_terms.total.unwrap_or(0.0) > 0.0);
    assert!(
        position_delta > 1e-8
            || scale_delta > 1e-8
            || opacity_delta > 1e-8
            || color_delta > 1e-8,
        "position_delta={position_delta:.6e} scale_delta={scale_delta:.6e} opacity_delta={opacity_delta:.6e} color_delta={color_delta:.6e}"
    );
    assert!(
        color_moments.iter().any(|value| value.abs() > 1e-8),
        "color_moments={color_moments:?}"
    );
}

#[test]
fn tum_frame_initialized_backward_probe_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_data/tum/rgbd_dataset_freiburg1_xyz");
    let dataset = crate::load_tum_rgbd_dataset(
        &dataset_path,
        &crate::TumRgbdConfig {
            max_frames: 10,
            frame_stride: 30,
            ..crate::TumRgbdConfig::default()
        },
    )
    .unwrap();
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 40,
        max_initial_gaussians: 100000,
        topology_warmup: 0,
        metal_render_scale: 0.5,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(6),
            refine_every: 10,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let effective_config = effective_metal_config(&config);
    let mut loaded = load_training_data(&dataset, &effective_config, &device).unwrap();
    let mut trainer = MetalTrainer::new(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        &effective_config,
        device.clone(),
    )
    .unwrap();
    let memory_budget = training_memory_budget(&config);
    let affordable_cap = affordable_initial_gaussian_cap(
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_splats.len()),
        trainer.pixel_count,
        trainer.source_pixel_count,
        loaded.cameras.len(),
        trainer.chunk_size,
        &memory_budget,
    );
    if affordable_cap > 0 && loaded.initial_splats.len() > affordable_cap {
        let initial_cap =
            preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
        loaded.initial_splats.downsample_evenly(initial_cap);
    }
    let initial_splats = loaded.initial_splats.clone();
    trainer.scene_extent = initial_splats.scene_extent();
    let mut gaussians = initial_splats.upload(&device).unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.iteration = 1;
    let frames = trainer.prepare_frames(&loaded).unwrap();
    let frame = &frames[0];
    let (rendered, projected, _) = trainer
        .render(&gaussians, &frame.camera, false, true, None)
        .unwrap();
    let rendered_color_cpu = rendered
        .color
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (ssim_value, ssim_grads) = ssim_gradient(
        &rendered_color_cpu,
        &frame.target_color_cpu,
        trainer.render_width,
        trainer.render_height,
    );
    let backward_loss_scales = metal_backward::backward_loss_scales(
        0.8,
        0.2,
        frame.target_color_cpu.len(),
        0.0,
        false,
        trainer.pixel_count,
    );
    let backward = metal_backward::execute_backward_pass(
        &mut trainer.runtime,
        MetalBackwardRequest {
            tile_bins: &projected.tile_bins,
            n_gaussians: gaussians.len(),
            camera: &frame.camera,
            target_color_cpu: &frame.target_color_cpu,
            target_depth_cpu: &frame.target_depth_cpu,
            ssim_grads: &ssim_grads,
            loss_scales: backward_loss_scales,
            refresh_target_buffers: true,
        },
    )
    .unwrap();
    let position_stats = tensor_abs_stats(&backward.grads.positions).unwrap();
    let scale_stats = tensor_abs_stats(&backward.grads.log_scales).unwrap();
    let opacity_stats = tensor_abs_stats(&backward.grads.opacity_logits).unwrap();
    let color_stats = tensor_abs_stats(&backward.grads.colors).unwrap();
    let param_grad_stats = abs_stats(&backward.grad_magnitudes);
    let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
    let before_positions = gaussians
        .positions()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_scales = gaussians
        .scales
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_opacities = gaussians
        .opacities
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_colors = gaussians
        .colors()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    trainer
        .apply_backward_grads(
            &mut gaussians,
            &backward.grads,
            &projected,
            &frame.camera,
            trainer.compute_lr_pos(),
            None,
            None,
        )
        .unwrap();
    let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
    let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
    let opacity_delta = max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
    let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();
    let trained_gaussians = HostSplats::from_runtime(&gaussians)
        .unwrap()
        .to_scene_gaussians()
        .unwrap();
    let mut map_position_delta = 0.0f32;
    let mut map_scale_delta = 0.0f32;
    let mut map_opacity_delta = 0.0f32;
    let mut map_color_delta = 0.0f32;
    let initial_gaussians = loaded.initial_splats.to_scene_gaussians().unwrap();
    for (before, after) in initial_gaussians.iter().zip(trained_gaussians.iter()) {
        map_position_delta = map_position_delta.max(
            before
                .position
                .iter()
                .zip(after.position.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0f32, f32::max),
        );
        map_scale_delta = map_scale_delta.max(
            before
                .scale
                .iter()
                .zip(after.scale.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0f32, f32::max),
        );
        map_opacity_delta = map_opacity_delta.max((before.opacity - after.opacity).abs());
        for channel in 0..3 {
            map_color_delta =
                map_color_delta.max((before.color[channel] - after.color[channel]).abs());
        }
    }

    assert!(
        position_delta > 1e-8
            || scale_delta > 1e-8
            || opacity_delta > 1e-8
            || color_delta > 1e-8,
        "tum probe no-op | gaussians={} visible={} ssim={:.6} | grad_positions={:?} grad_scales={:?} grad_opacity={:?} grad_colors={:?} | param_grad={:?} projected_grad={:?} | deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
        gaussians.len(),
        projected.visible_count,
        ssim_value,
        position_stats,
        scale_stats,
        opacity_stats,
        color_stats,
        param_grad_stats,
        projected_grad_stats,
        position_delta,
        scale_delta,
        opacity_delta,
        color_delta,
    );
    assert!(
        map_position_delta > 1e-8
            || map_scale_delta > 1e-8
            || map_opacity_delta > 1e-8
            || map_color_delta > 1e-8,
        "tum train-loop export no-op | tensor_deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e}) | map_deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
        position_delta,
        scale_delta,
        opacity_delta,
        color_delta,
        map_position_delta,
        map_scale_delta,
        map_opacity_delta,
        map_color_delta,
    );
}

#[test]
fn tum_frame_initialized_train_loop_updates_params_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_data/tum/rgbd_dataset_freiburg1_xyz");
    let dataset = crate::load_tum_rgbd_dataset(
        &dataset_path,
        &crate::TumRgbdConfig {
            max_frames: 10,
            frame_stride: 30,
            ..crate::TumRgbdConfig::default()
        },
    )
    .unwrap();
    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 1,
        max_initial_gaussians: 100000,
        topology_warmup: 0,
        metal_render_scale: 0.5,
        litegs: LiteGsConfig {
            densify_from: 0,
            densify_until: Some(6),
            refine_every: 10,
            ..LiteGsConfig::default()
        },
        ..TrainingConfig::default()
    };
    let effective_config = effective_metal_config(&config);
    let mut loaded = load_training_data(&dataset, &effective_config, &device).unwrap();
    let mut trainer = MetalTrainer::new(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        &effective_config,
        device.clone(),
    )
    .unwrap();
    let memory_budget = training_memory_budget(&config);
    let affordable_cap = affordable_initial_gaussian_cap(
        effective_config
            .max_initial_gaussians
            .max(loaded.initial_splats.len()),
        trainer.pixel_count,
        trainer.source_pixel_count,
        loaded.cameras.len(),
        trainer.chunk_size,
        &memory_budget,
    );
    if affordable_cap > 0 && loaded.initial_splats.len() > affordable_cap {
        let initial_cap =
            preflight_initial_gaussian_cap(effective_config.training_profile, affordable_cap);
        loaded.initial_splats.downsample_evenly(initial_cap);
    }
    let initial_splats = loaded.initial_splats.clone();
    trainer.scene_extent = initial_splats.scene_extent();
    let mut gaussians = initial_splats.upload(&device).unwrap();
    let before_positions = gaussians
        .positions()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_scales = gaussians
        .scales
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_opacities = gaussians
        .opacities
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let before_colors = gaussians
        .colors()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    trainer.train_loaded(&mut gaussians, &loaded, 1).unwrap();

    let position_delta = max_abs_delta(&before_positions, gaussians.positions()).unwrap();
    let scale_delta = max_abs_delta(&before_scales, gaussians.scales.as_tensor()).unwrap();
    let opacity_delta = max_abs_delta(&before_opacities, gaussians.opacities.as_tensor()).unwrap();
    let color_delta = max_abs_delta(&before_colors, &gaussians.colors()).unwrap();

    assert!(
        position_delta > 1e-8 || scale_delta > 1e-8 || opacity_delta > 1e-8 || color_delta > 1e-8,
        "tum train-loop no-op | deltas=({:.6e}, {:.6e}, {:.6e}, {:.6e})",
        position_delta,
        scale_delta,
        opacity_delta,
        color_delta,
    );
}

#[test]
fn adam_step_var_sparse_preserves_invisible_rows_for_tensor3_params() {
    let device = Device::Cpu;
    let var = Var::from_tensor(
        &Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (2, 2, 3),
            &device,
        )
        .unwrap(),
    )
    .unwrap();
    let grad = Tensor::from_slice(
        &[
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75,
        ],
        (2, 2, 3),
        &device,
    )
    .unwrap();
    let mut m = Tensor::from_slice(
        &[
            0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (2, 2, 3),
        &device,
    )
    .unwrap();
    let mut v = Tensor::from_slice(
        &[
            0.25f32, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (2, 2, 3),
        &device,
    )
    .unwrap();
    let indices = Tensor::from_slice(&[1u32], 1, &device).unwrap();
    let before_param = var.as_tensor().to_vec3::<f32>().unwrap();
    let before_m = m.to_vec3::<f32>().unwrap();
    let before_v = v.to_vec3::<f32>().unwrap();

    adam_step_var_sparse(
        &var, &grad, &mut m, &mut v, &indices, 0.1, 0.9, 0.999, 1e-8, 2,
    )
    .unwrap();

    let after_param = var.as_tensor().to_vec3::<f32>().unwrap();
    let after_m = m.to_vec3::<f32>().unwrap();
    let after_v = v.to_vec3::<f32>().unwrap();

    assert_eq!(after_param[0], before_param[0]);
    assert_eq!(after_m[0], before_m[0]);
    assert_eq!(after_v[0], before_v[0]);
    assert_ne!(after_param[1], before_param[1]);
    assert_ne!(after_m[1], before_m[1]);
    assert_ne!(after_v[1], before_v[1]);
}

#[test]
fn adam_step_var_fused_matches_cpu_update_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let mut runtime = MetalRuntime::new(1, 1, device.clone()).unwrap();
    let shape = (2, 3);
    let initial = [1.0f32, -2.0, 0.25, 3.0, -4.0, 5.0];
    let grads = [0.5f32, -1.5, 2.0, -0.25, 0.75, -3.0];
    let var = Var::from_tensor(&Tensor::from_slice(&initial, shape, &device).unwrap()).unwrap();
    let grad = Tensor::from_slice(&grads, shape, &device).unwrap();
    let mut m = Tensor::zeros(shape, DType::F32, &device).unwrap();
    let mut v = Tensor::zeros(shape, DType::F32, &device).unwrap();

    let cpu = Device::Cpu;
    let (expected_param, expected_m, expected_v) = adam_updated_tensors(
        &Tensor::from_slice(&initial, shape, &cpu).unwrap(),
        &Tensor::from_slice(&grads, shape, &cpu).unwrap(),
        &Tensor::zeros(shape, DType::F32, &cpu).unwrap(),
        &Tensor::zeros(shape, DType::F32, &cpu).unwrap(),
        0.01,
        0.9,
        0.999,
        1e-8,
        1,
    )
    .unwrap();

    adam_step_var_fused(
        &var,
        &grad,
        &mut m,
        &mut v,
        &mut runtime,
        0.01,
        0.9,
        0.999,
        1e-8,
        1,
        MetalBufferSlot::AdamGradPos,
        MetalBufferSlot::AdamMPos,
        MetalBufferSlot::AdamVPos,
        MetalBufferSlot::AdamParamPos,
    )
    .unwrap();

    let actual_param = var
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let actual_m = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let actual_v = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected_param = expected_param
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected_m = expected_m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected_v = expected_v.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert!(
        max_slice_delta(&actual_param, &expected_param) < 1e-6,
        "actual_param={actual_param:?} expected_param={expected_param:?}"
    );
    assert!(
        max_slice_delta(&actual_m, &expected_m) < 1e-6,
        "actual_m={actual_m:?} expected_m={expected_m:?}"
    );
    assert!(
        max_slice_delta(&actual_v, &expected_v) < 1e-6,
        "actual_v={actual_v:?} expected_v={expected_v:?}"
    );
}

#[test]
fn apply_backward_grads_dense_updates_metal_params() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            sparse_grad: false,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(64, 64, &config, device.clone()).unwrap();
    trainer.iteration = 3;
    let camera = make_test_camera(&device);
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 2.0, 3.0, 0.0, 2.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[0.5, 0.1, 0.2, 0.9, 0.2, 0.1],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    let projected = projected_with_visible_sources(&device, &[0]);
    let grads = MetalBackwardGrads {
        positions: Tensor::from_slice(&[1.0f32, -0.5, 0.25, 0.0, 0.0, 0.0], (2, 3), &device)
            .unwrap(),
        log_scales: Tensor::from_slice(&[0.2f32, -0.1, 0.05, 0.0, 0.0, 0.0], (2, 3), &device)
            .unwrap(),
        opacity_logits: Tensor::from_slice(&[0.3f32, 0.0], 2, &device).unwrap(),
        colors: Tensor::from_slice(&[0.4f32, -0.2, 0.1, 0.0, 0.0, 0.0], (2, 3), &device).unwrap(),
    };
    let before_positions = gaussians.positions().to_vec2::<f32>().unwrap();
    let before_scales = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap();
    let before_opacity = gaussians.opacities.as_tensor().to_vec1::<f32>().unwrap();
    let before_colors = gaussians.colors().to_vec2::<f32>().unwrap();

    trainer
        .apply_backward_grads(&mut gaussians, &grads, &projected, &camera, 0.1, None, None)
        .unwrap();

    let after_positions = gaussians.positions().to_vec2::<f32>().unwrap();
    let after_scales = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap();
    let after_opacity = gaussians.opacities.as_tensor().to_vec1::<f32>().unwrap();
    let after_colors = gaussians.colors().to_vec2::<f32>().unwrap();
    let adam = trainer.adam.as_ref().unwrap();
    let m_pos = adam.m_pos.to_vec2::<f32>().unwrap();
    let m_color = adam.m_color.to_vec2::<f32>().unwrap();

    assert_ne!(after_positions[0], before_positions[0]);
    assert_eq!(after_positions[1], before_positions[1]);
    assert_ne!(after_scales[0], before_scales[0]);
    assert_eq!(after_scales[1], before_scales[1]);
    assert_ne!(after_opacity[0], before_opacity[0]);
    assert_eq!(after_opacity[1], before_opacity[1]);
    assert_ne!(after_colors[0], before_colors[0]);
    assert_eq!(after_colors[1], before_colors[1]);
    assert!(m_pos[0].iter().any(|value| value.abs() > 1e-8));
    assert!(m_pos[1].iter().all(|value| value.abs() < 1e-8));
    assert!(m_color[0].iter().any(|value| value.abs() > 1e-8));
    assert!(m_color[1].iter().all(|value| value.abs() < 1e-8));
}

#[test]
fn peak_estimate_scales_with_problem_size() {
    let small = estimate_peak_memory(4_096, 4_800, 5, 32);
    let large = estimate_peak_memory(57_474, 4_800, 5, 32);
    assert!(large.total_bytes() > small.total_bytes());
    assert!(large.projection_bytes > small.projection_bytes);
    assert!(large.runtime_bytes > small.runtime_bytes);
}

#[test]
fn peak_estimate_accounts_for_frames_but_not_chunk_size() {
    let baseline = estimate_peak_memory(4_096, 4_800, 5, 32);
    let more_frames = estimate_peak_memory(4_096, 4_800, 25, 32);
    let larger_chunk = estimate_peak_memory(4_096, 4_800, 5, 128);
    assert!(more_frames.total_bytes() > baseline.total_bytes());
    assert_eq!(larger_chunk.total_bytes(), baseline.total_bytes());
}

#[test]
fn peak_estimate_accounts_for_retained_source_resolution_staging() {
    let render_pixels = 320 * 180;
    let source_pixels = 1920 * 1080;
    let baseline = estimate_peak_memory(4_096, render_pixels, 5, 32);
    let staged =
        estimate_peak_memory_with_source_pixels(4_096, render_pixels, source_pixels, 5, 32);
    assert!(staged.frame_bytes > baseline.frame_bytes);
    assert!(staged.total_bytes() > baseline.total_bytes());
}

#[test]
fn detected_budget_prefers_fraction_of_system_memory() {
    let physical = 16 * GIB;
    let safe = apply_ratio(
        physical,
        METAL_SYSTEM_MEMORY_BUDGET_NUMERATOR,
        METAL_SYSTEM_MEMORY_BUDGET_DENOMINATOR,
    )
    .min(DEFAULT_METAL_MEMORY_BUDGET_BYTES);
    assert!((bytes_to_gib(safe) - 10.4).abs() < 0.05);
}

#[test]
fn preflight_warns_when_close_to_budget() {
    let estimate = estimate_peak_memory(8_000, 4_800, 5, 32);
    let budget = MetalMemoryBudget {
        safe_bytes: estimate.total_bytes().saturating_add(1),
        physical_bytes: Some(16 * GIB),
    };
    assert_eq!(
        assess_memory_estimate(&estimate, &budget),
        MetalMemoryDecision::Warn
    );
}

#[test]
fn preflight_blocks_when_estimate_exceeds_budget() {
    let estimate = estimate_peak_memory(57_474, 4_800, 1, 32);
    let budget = MetalMemoryBudget {
        safe_bytes: estimate.total_bytes().saturating_sub(1),
        physical_bytes: Some(24 * GIB),
    };
    assert_eq!(
        assess_memory_estimate(&estimate, &budget),
        MetalMemoryDecision::Block
    );
}

#[test]
fn affordable_initial_gaussian_cap_finds_non_blocking_limit() {
    let requested = 57_474;
    let minimum_estimate = estimate_peak_memory(1, 4_800, 1, 32);
    let requested_estimate = estimate_peak_memory(requested, 4_800, 1, 32);
    let budget = MetalMemoryBudget {
        safe_bytes: minimum_estimate
            .total_bytes()
            .saturating_add(requested_estimate.total_bytes())
            / 2,
        physical_bytes: Some(24 * GIB),
    };
    let cap = affordable_initial_gaussian_cap(requested, 4_800, 4_800, 1, 32, &budget);
    assert!(cap > 0);
    assert!(cap < requested);
    assert_ne!(
        assess_memory_estimate(&estimate_peak_memory(cap, 4_800, 1, 32), &budget),
        MetalMemoryDecision::Block
    );
    assert_eq!(
        assess_memory_estimate(&estimate_peak_memory(cap + 1, 4_800, 1, 32), &budget),
        MetalMemoryDecision::Block
    );
}

#[test]
fn preflight_initial_gaussian_cap_reserves_litegs_headroom() {
    assert_eq!(
        preflight_initial_gaussian_cap(TrainingProfile::LiteGsMacV1, 552),
        525
    );
    assert_eq!(
        preflight_initial_gaussian_cap(TrainingProfile::LegacyMetal, 552),
        552
    );
}

#[test]
fn splats_downsample_evenly_spreads_samples_across_map() {
    let mut splats = HostSplats::with_sh_degree_capacity(0, 10);
    for idx in 0..10 {
        splats.push_rgb(
            [idx as f32, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            0.0,
            [0.5, 0.5, 0.5],
        );
    }
    splats.downsample_evenly(4);
    let sampled_positions: Vec<f32> = splats
        .positions_vec3()
        .into_iter()
        .map(|position| position[0])
        .collect();
    assert_eq!(sampled_positions, vec![0.0, 2.0, 5.0, 7.0]);
}

#[test]
fn resolve_chunk_memory_budget_caps_requested_budget_to_system_limit() {
    let system_budget = MetalMemoryBudget {
        safe_bytes: 10 * GIB,
        physical_bytes: Some(16 * GIB),
    };
    let resolved = resolve_chunk_memory_budget(12 * GIB, system_budget);
    assert_eq!(resolved.safe_bytes, 10 * GIB);
    assert_eq!(resolved.physical_bytes, Some(16 * GIB));
}

#[test]
fn gib_to_bytes_rejects_non_positive_values() {
    assert_eq!(gib_to_bytes(0.0), 0);
    assert_eq!(gib_to_bytes(-1.0), 0);
}

#[test]
fn chunk_capacity_marks_over_budget_requests_for_subdivision() {
    let config = TrainingConfig {
        chunked_training: true,
        // Keep this deliberately tight so the estimator reliably requests
        // subdivision even as per-gaussian accounting evolves.
        chunk_budget_gb: 0.25,
        metal_render_scale: 1.0,
        max_initial_gaussians: 57_474,
        ..TrainingConfig::default()
    };
    let dataset = TrainingDataset {
        intrinsics: crate::Intrinsics::from_focal(500.0, 1920, 1080),
        depth_scale: 1000.0,
        poses: vec![crate::ScenePose::new(
            0,
            std::path::PathBuf::from("frame.png"),
            crate::SE3::identity(),
            0.0,
        )],
        initial_points: Vec::new(),
    };
    let estimate = estimate_chunk_capacity(&dataset, &config).unwrap();
    assert!(estimate.requires_subdivision_or_degradation());
    assert!(estimate.affordable_initial_gaussians < estimate.requested_initial_gaussians);
    assert!(estimate
        .recommendations()
        .first()
        .expect("recommendations should not be empty")
        .contains("subdivide the chunk"));
}

#[test]
fn chunk_capacity_uses_existing_initial_points_as_requested_scale() {
    let config = TrainingConfig {
        chunked_training: true,
        chunk_budget_gb: 1.0,
        max_initial_gaussians: 16,
        ..TrainingConfig::default()
    };
    let dataset = TrainingDataset {
        intrinsics: crate::Intrinsics::from_focal(500.0, 32, 32),
        depth_scale: 1000.0,
        poses: vec![crate::ScenePose::new(
            0,
            std::path::PathBuf::from("frame.png"),
            crate::SE3::identity(),
            0.0,
        )],
        initial_points: vec![([0.0, 0.0, 1.0], None); 64],
    };
    let estimate = estimate_chunk_capacity(&dataset, &config).unwrap();
    assert_eq!(estimate.requested_initial_gaussians, 16);
}

#[test]
fn profile_tracks_visible_gaussians() {
    let render = MetalRenderProfile {
        visible_gaussians: 12,
        total_gaussians: 40,
        ..Default::default()
    };
    let profile = MetalStepProfile::from_render(render);
    assert_eq!(profile.visible_gaussians, 12);
    assert_eq!(profile.total_gaussians, 40);
}

#[test]
fn chunk_rect_area_matches_bounds() {
    let rect = ScreenRect {
        min_x: 2,
        max_x: 5,
        min_y: 3,
        max_y: 4,
    };
    assert_eq!(rect.max_x - rect.min_x + 1, 4);
    assert_eq!(rect.max_y - rect.min_y + 1, 2);
}

#[test]
fn tile_bins_only_include_overlapping_gaussians() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let projected = ProjectedGaussians {
        source_indices: Tensor::from_slice(&[0u32, 1], 2, &device).unwrap(),
        u: Tensor::from_slice(&[8.0f32, 18.0], 2, &device).unwrap(),
        v: Tensor::from_slice(&[8.0f32, 8.0], 2, &device).unwrap(),
        sigma_x: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
        sigma_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
        raw_sigma_x: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
        raw_sigma_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
        depth: Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap(),
        opacity: Tensor::from_slice(&[0.5f32, 0.5], 2, &device).unwrap(),
        opacity_logits: Tensor::from_slice(&[0.0f32, 0.0], 2, &device).unwrap(),
        scale3d: Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &device).unwrap(),
        colors: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device).unwrap(),
        min_x: Tensor::from_slice(&[2.0f32, 14.0], 2, &device).unwrap(),
        max_x: Tensor::from_slice(&[15.0f32, 18.0], 2, &device).unwrap(),
        min_y: Tensor::from_slice(&[1.0f32, 1.0], 2, &device).unwrap(),
        max_y: Tensor::from_slice(&[14.0f32, 14.0], 2, &device).unwrap(),
        visible_source_indices: vec![0, 1],
        visible_count: 2,
        tile_bins: ProjectedTileBins::default(),
        staging_source: ProjectionStagingSource::TensorReadback,
    };

    let bins = trainer.build_tile_bins(&projected).unwrap();
    assert_eq!(bins.active_tile_count(), 2);
    assert_eq!(bins.total_assignments(), 3);
    assert_eq!(bins.max_gaussians_per_tile(), 2);
    assert_eq!(bins.indices_for_tile(0), &[0, 1]);
    assert_eq!(bins.indices_for_tile(1), &[1]);
}

#[test]
fn native_forward_matches_baseline_on_tiny_scene() {
    let Ok(device) = crate::try_metal_device() else {
        return;
    };
    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        1.0,
        1.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.runtime.stage_camera(&camera).unwrap();
    let projected = ProjectedGaussians {
        source_indices: Tensor::from_slice(&[0u32, 1], 2, &device).unwrap(),
        u: Tensor::from_slice(&[8.0f32, 10.0], 2, &device).unwrap(),
        v: Tensor::from_slice(&[8.0f32, 8.5], 2, &device).unwrap(),
        sigma_x: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
        sigma_y: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
        raw_sigma_x: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
        raw_sigma_y: Tensor::from_slice(&[2.0f32, 2.5], 2, &device).unwrap(),
        depth: Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap(),
        opacity: Tensor::from_slice(&[0.6f32, 0.4], 2, &device).unwrap(),
        opacity_logits: Tensor::from_slice(&[0.0f32, 0.0], 2, &device).unwrap(),
        scale3d: Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &device).unwrap(),
        colors: Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device).unwrap(),
        min_x: Tensor::from_slice(&[2.0f32, 3.0], 2, &device).unwrap(),
        max_x: Tensor::from_slice(&[14.0f32, 17.0], 2, &device).unwrap(),
        min_y: Tensor::from_slice(&[2.0f32, 2.0], 2, &device).unwrap(),
        max_y: Tensor::from_slice(&[14.0f32, 15.0], 2, &device).unwrap(),
        visible_source_indices: vec![0, 1],
        visible_count: 2,
        tile_bins: ProjectedTileBins::default(),
        staging_source: ProjectionStagingSource::TensorReadback,
    };

    let tile_bins = trainer.build_tile_bins(&projected).unwrap();
    let (baseline, _) = trainer.rasterize(&projected, &tile_bins).unwrap();
    let parity = trainer
        .profile_native_forward(&projected, &tile_bins, &baseline)
        .unwrap();

    assert!(
        parity.color_max_abs < 5e-4,
        "color diff={}",
        parity.color_max_abs
    );
    assert!(
        parity.depth_max_abs < 5e-4,
        "depth diff={}",
        parity.depth_max_abs
    );
    assert!(
        parity.alpha_max_abs < 5e-4,
        "alpha diff={}",
        parity.alpha_max_abs
    );
}

#[test]
fn metal_visible_set_stays_on_device() {
    let Ok(device) = crate::try_metal_device() else {
        return;
    };
    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        1.0,
        1.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[0.0, 0.0, 2.0, 1.0, 0.0, 3.0],
        &[
            0.1f32.ln(),
            0.1f32.ln(),
            0.1f32.ln(),
            0.1f32.ln(),
            0.1f32.ln(),
            0.1f32.ln(),
        ],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();

    let (projected, _) = trainer
        .project_gaussians(&gaussians, &camera, false, true, None)
        .unwrap();

    assert!(matches!(
        projected.staging_source,
        ProjectionStagingSource::RuntimeBufferRead
    ));
    assert_eq!(projected.visible_count, 2);
    assert_eq!(projected.visible_source_indices, vec![0, 1]);
}

#[test]
fn prune_interval_is_independent_from_densify_interval() {
    let trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            densify_interval: 128,
            prune_interval: 200,
            topology_warmup: 0,
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();

    let policy = trainer.topology_policy();
    let prune = topology::schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 200,
            frame_count: 1,
        },
    );
    let densify = topology::schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 128,
            frame_count: 1,
        },
    );

    assert!(prune.prune);
    assert!(!prune.densify);
    assert!(densify.densify);
    assert!(!densify.prune);
}

#[test]
fn profile_schedule_honors_interval() {
    assert!(should_profile_iteration(true, 25, 0));
    assert!(should_profile_iteration(true, 25, 4));
    assert!(!should_profile_iteration(true, 25, 5));
    assert!(should_profile_iteration(true, 25, 25));
    assert!(!should_profile_iteration(false, 25, 25));
}

#[test]
fn summarized_final_loss_uses_last_epoch_mean() {
    let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
    let metrics = summarize_training_metrics(&history, 2);
    assert!((metrics.final_loss - 0.55).abs() < 1e-6);
    assert!((metrics.final_step_loss - 0.5).abs() < 1e-6);
}

#[test]
fn project_gaussians_handles_zero_visible_without_index_select() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        1.0,
        1.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[0.0, 0.0, -1.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();

    let (projected, profile) = trainer
        .project_gaussians(&gaussians, &camera, false, true, None)
        .unwrap();

    assert_eq!(profile.total_gaussians, 1);
    assert_eq!(profile.visible_gaussians, 0);
    assert_eq!(projected.source_indices.dim(0).unwrap(), 0);
    assert_eq!(projected.u.dim(0).unwrap(), 0);
    assert_eq!(projected.colors.dim(0).unwrap(), 0);
}

#[test]
fn project_gaussians_keeps_distinct_visible_indices_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let trainer_config = TrainingConfig {
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        16.0,
        16.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[
            0.0, 0.0, 1.0, //
            0.1, 0.0, 0.5, //
            -0.1, 0.0, 2.0,
        ],
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        &[
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ],
        &[0.0, 0.0, 0.0],
        &[
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
        &device,
    )
    .unwrap();

    let (projected, profile) = trainer
        .project_gaussians(&gaussians, &camera, false, true, None)
        .unwrap();
    let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

    assert_eq!(profile.visible_gaussians, 3);
    assert_eq!(source_indices, vec![1, 0, 2]);
}

#[test]
fn project_gaussians_applies_cluster_visible_mask_on_metal() {
    let device = crate::preferred_device();
    if !device.is_metal() {
        return;
    }

    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        litegs: LiteGsConfig {
            cluster_size: 1,
            ..LiteGsConfig::default()
        },
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let camera = DiffCamera::new(
        16.0,
        16.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[
            0.0, 0.0, 1.0, //
            0.1, 0.0, 0.5, //
            -0.1, 0.0, 2.0,
        ],
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        &[
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ],
        &[0.0, 0.0, 0.0],
        &[
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
        &device,
    )
    .unwrap();
    let cluster_visible_mask = [true, false, true];

    let (projected, profile) = trainer
        .project_gaussians(
            &gaussians,
            &camera,
            false,
            true,
            Some(cluster_visible_mask.as_slice()),
        )
        .unwrap();
    let source_indices = projected.source_indices.to_vec1::<u32>().unwrap();

    assert_eq!(profile.visible_gaussians, 2);
    assert_eq!(projected.visible_source_indices, vec![0, 2]);
    assert_eq!(source_indices, vec![0, 2]);
}

#[test]
fn topology_updates_can_grow_beyond_initial_gaussian_count_limit() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        densify_interval: 1,
        prune_interval: 0,
        topology_warmup: 0,
        prune_threshold: 0.05,
        max_initial_gaussians: 2,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        &[
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
        ],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[2.0, 2.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 1,
            ..Default::default()
        },
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 1,
            ..Default::default()
        },
    ];
    trainer.iteration = 1;

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert!(gaussians.len() > 2);
}

#[test]
fn topology_updates_preserve_sh_representation_for_litegs_trainables() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        densify_interval: 1,
        prune_interval: 0,
        topology_warmup: 0,
        prune_threshold: 0.05,
        max_initial_gaussians: 2,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.topology_memory_budget = None;
    let mut gaussians = Splats::new_with_sh(
        &[0.0, 0.0, 1.0],
        &[0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[2.0],
        &[
            crate::diff::diff_splat::rgb_to_sh0_value(0.2),
            crate::diff::diff_splat::rgb_to_sh0_value(0.4),
            crate::diff::diff_splat::rgb_to_sh0_value(0.6),
        ],
        &vec![0.0; 15 * 3],
        3,
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![MetalGaussianStats {
        mean2d_grad: RunningMoments {
            mean: 1.0,
            m2: 0.0,
            count: 1,
        },
        age: 1,
        ..Default::default()
    }];
    trainer.iteration = 1;

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert!(gaussians.len() > 1);
    assert!(gaussians.uses_spherical_harmonics());
    assert_eq!(gaussians.sh_degree(), 3);
    assert_eq!(gaussians.sh_rest().dims()[0], gaussians.len());
}

#[test]
fn litegs_refine_decay_reduces_runtime_opacity_and_scale() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        training_profile: TrainingProfile::LiteGsMacV1,
        iterations: 500,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    trainer.iteration = 160;

    let mut gaussians = Splats::new_with_sh(
        &[0.0, 0.0, 1.0],
        &[0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[
            crate::diff::diff_splat::rgb_to_sh0_value(0.2),
            crate::diff::diff_splat::rgb_to_sh0_value(0.4),
            crate::diff::diff_splat::rgb_to_sh0_value(0.6),
        ],
        &vec![0.0; 15 * 3],
        3,
        &device,
    )
    .unwrap();
    let before_opacity = gaussians.opacities().unwrap().to_vec1::<f32>().unwrap()[0];
    let before_log_scale = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap()[0][0];

    trainer.apply_litegs_refine_decay(&mut gaussians).unwrap();

    let after_opacity = gaussians.opacities().unwrap().to_vec1::<f32>().unwrap()[0];
    let after_log_scale = gaussians.scales.as_tensor().to_vec2::<f32>().unwrap()[0][0];

    assert!(after_opacity < before_opacity);
    assert!(after_log_scale < before_log_scale);
}

#[test]
fn topology_update_densifies_and_prunes_with_matching_adam_state() {
    let device = Device::Cpu;
    let trainer_config = TrainingConfig {
        densify_interval: 1,
        prune_interval: 1,
        topology_warmup: 0,
        prune_threshold: 0.05,
        max_initial_gaussians: 4,
        metal_render_scale: 1.0,
        ..TrainingConfig::default()
    };
    let mut trainer = MetalTrainer::new(32, 16, &trainer_config, device.clone()).unwrap();
    let mut gaussians = Splats::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        &[
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
        ],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[2.0, -10.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();
    trainer.adam = Some(MetalAdamState::new(&gaussians).unwrap());
    trainer.gaussian_stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 5,
            ..Default::default()
        },
        MetalGaussianStats {
            mean2d_grad: RunningMoments::default(),
            age: 7,
            ..Default::default()
        },
    ];
    trainer.iteration = 1;

    trainer
        .maybe_apply_topology_updates(&mut gaussians, 0, 1)
        .unwrap();

    assert_eq!(gaussians.len(), 2);
    assert_eq!(trainer.gaussian_stats.len(), 2);
    assert!(trainer.gaussian_stats.iter().any(|stats| stats.age == 0));
    let opacities = gaussians.opacities().unwrap().to_vec1::<f32>().unwrap();
    assert!(opacities
        .iter()
        .all(|opacity| *opacity >= trainer_config.prune_threshold));
    let positions = gaussians.positions().to_vec2::<f32>().unwrap();
    assert!((positions[1][0] - positions[0][0]).abs() > 1e-6);

    let adam = trainer.adam.as_ref().unwrap();
    assert_eq!(adam.m_pos.dim(0).unwrap(), 2);
    assert_eq!(adam.v_pos.dim(0).unwrap(), 2);
    let m_pos = adam.m_pos.to_vec2::<f32>().unwrap();
    assert!(m_pos[1].iter().all(|value| value.abs() < 1e-6));
}

#[test]
fn rebuild_adam_state_preserves_reordered_rows() {
    let device = Device::Cpu;
    let trainer = MetalTrainer::new(32, 16, &TrainingConfig::default(), device.clone()).unwrap();
    let gaussians = Splats::new(
        &[0.0, 0.0, 1.0, 1.0, 0.0, 2.0],
        &[
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.06f32.ln(),
            0.06f32.ln(),
            0.06f32.ln(),
        ],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();
    let mut adam = MetalAdamState::new(&gaussians).unwrap();
    adam.m_pos = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();
    adam.v_pos =
        Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (2, 3), &device).unwrap();

    let reordered = trainer
        .rebuild_adam_state(
            &adam,
            &topology::TopologyMutationPlan {
                rows: vec![
                    topology::TopologyPlanRow::Existing { source_idx: 1 },
                    topology::TopologyPlanRow::Existing { source_idx: 0 },
                ],
                ..topology::TopologyMutationPlan::default()
            },
        )
        .unwrap();
    assert_eq!(
        reordered.m_pos.to_vec2::<f32>().unwrap(),
        vec![vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0]]
    );
    assert_eq!(
        reordered.v_pos.to_vec2::<f32>().unwrap(),
        vec![vec![10.0, 11.0, 12.0], vec![7.0, 8.0, 9.0]]
    );
}

#[test]
fn sync_cluster_assignment_updates_aabbs_from_current_positions() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: super::LiteGsConfig {
                cluster_size: 2,
                ..super::LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    trainer.scene_extent = 16.0;
    let gaussians = Splats::new(
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();

    trainer.sync_cluster_assignment(&gaussians, false).unwrap();
    let initial_aabb = trainer.cluster_assignment.as_ref().unwrap().aabbs[0];

    gaussians
        .positions
        .set(&Tensor::from_slice(&[10.0f32, 0.0, 0.0, 11.0, 0.0, 0.0], (2, 3), &device).unwrap())
        .unwrap();
    trainer.sync_cluster_assignment(&gaussians, false).unwrap();

    let updated_aabb = trainer.cluster_assignment.as_ref().unwrap().aabbs[0];
    assert!(initial_aabb.min[0] < 1.0);
    assert!((updated_aabb.min[0] - 10.0).abs() < 1e-6);
    assert!((updated_aabb.max[0] - 11.0).abs() < 1e-6);
}

#[test]
fn sync_cluster_assignment_reassigns_after_topology_change() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: super::LiteGsConfig {
                cluster_size: 1,
                ..super::LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    trainer.scene_extent = 16.0;
    let gaussians_one = Splats::new(
        &[0.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[1.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    trainer
        .sync_cluster_assignment(&gaussians_one, false)
        .unwrap();
    assert_eq!(
        trainer
            .cluster_assignment
            .as_ref()
            .unwrap()
            .cluster_indices
            .len(),
        1
    );

    let gaussians_two = Splats::new(
        &[0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &device,
    )
    .unwrap();
    trainer
        .sync_cluster_assignment(&gaussians_two, true)
        .unwrap();

    let assignment = trainer.cluster_assignment.as_ref().unwrap();
    assert_eq!(assignment.cluster_indices.len(), 2);
    assert_eq!(assignment.cluster_sizes.iter().sum::<usize>(), 2);
    assert_eq!(assignment.aabbs.len(), assignment.num_clusters);
}

#[test]
fn lr_pos_exponential_decay_is_correct() {
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            lr_position: 0.001,
            lr_pos_final: 0.00001,
            iterations: 1000,
            ..TrainingConfig::default()
        },
        Device::Cpu,
    )
    .unwrap();

    // At iteration 0, effective LR equals initial LR.
    let lr_init = trainer.compute_lr_pos();
    assert!(
        (lr_init - 0.001).abs() < 1e-7,
        "at t=0 expected 0.001, got {lr_init}"
    );

    // At iteration = max, effective LR equals final LR.
    trainer.iteration = 1000;
    let lr_end = trainer.compute_lr_pos();
    assert!(
        (lr_end - 0.00001).abs() < 1e-9,
        "at t=T expected 0.00001, got {lr_end}"
    );

    // At t = T/2, effective LR is geometric mean of init and final.
    trainer.iteration = 500;
    let lr_mid = trainer.compute_lr_pos();
    let expected_mid = (0.001f32 * 0.00001f32).sqrt();
    assert!(
        (lr_mid - expected_mid).abs() < expected_mid * 1e-4,
        "at t=T/2 expected {expected_mid}, got {lr_mid}"
    );

    // LR must strictly decrease over time.
    trainer.iteration = 100;
    let lr_100 = trainer.compute_lr_pos();
    trainer.iteration = 900;
    let lr_900 = trainer.compute_lr_pos();
    assert!(lr_100 > lr_900, "LR should decrease: {lr_100} > {lr_900}");
}

#[test]
fn sh_render_colors_follow_view_direction_terms() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    trainer.active_sh_degree = 1;
    let camera = DiffCamera::new(
        16.0,
        16.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let mut sh_rest = vec![0.0f32; 3 * 3];
    sh_rest[0] = -0.5;
    let gaussians = Splats::new_with_sh(
        &[0.0, 1.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[0.0, 0.0, 0.0],
        &sh_rest,
        1,
        &device,
    )
    .unwrap();

    let positions = gaussians.positions().detach();
    let colors = trainer
        .render_colors_for_camera(&gaussians, &positions, &camera)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let expected_red = 0.5 + (-metal_forward::SH_C1) * -0.5;

    assert!((colors[0][0] - expected_red).abs() < 1e-5);
    assert!((colors[0][1] - 0.5).abs() < 1e-6);
    assert!((colors[0][2] - 0.5).abs() < 1e-6);
}

#[test]
fn litegs_scale_regularization_uses_activated_scales() {
    let device = Device::Cpu;
    let trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: super::LiteGsConfig {
                reg_weight: 0.5,
                ..super::LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    let visible_log_scales =
        Tensor::from_slice(&[2.0f32.ln(), 1.0f32.ln(), 0.5f32.ln()], (1, 3), &device).unwrap();

    let term = scale_regularization_term(&visible_log_scales)
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    let grad = test_scale_regularization_grad(&visible_log_scales, trainer.litegs.reg_weight)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert!((term - 1.75).abs() < 1e-6);
    assert!((grad[0][0] - (4.0 / 3.0)).abs() < 1e-6);
    assert!((grad[0][1] - (1.0 / 3.0)).abs() < 1e-6);
    assert!((grad[0][2] - (1.0 / 12.0)).abs() < 1e-6);
}

#[test]
fn pose_parameter_grads_returns_tensor_pair() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: super::LiteGsConfig {
                learnable_viewproj: true,
                lr_pose: 1e-4,
                ..super::LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    let camera = DiffCamera::new(
        16.0,
        16.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new(
        &[0.0, 0.0, 3.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[1.0, 0.5, 0.25],
        &device,
    )
    .unwrap();
    let (rendered, _, _) = trainer
        .render(&gaussians, &camera, false, true, None)
        .unwrap();
    let target_color_cpu = rendered
        .color
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let target_depth_cpu = rendered
        .depth
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let frame = MetalTrainingFrame {
        camera: camera.clone(),
        target_color: rendered.color.clone(),
        target_depth: rendered.depth.clone(),
        target_color_cpu,
        target_depth_cpu,
    };
    trainer.pose_embeddings = Some(
        crate::training::pose_embedding::PoseEmbeddings::from_dataset(
            &[crate::ScenePose::new(
                0,
                std::path::PathBuf::from("frame.png"),
                crate::SE3::identity(),
                0.0,
            )],
            1e-4,
            &device,
        )
        .unwrap(),
    );
    let frame_pose_embedding =
        pose_embedding::cloned_frame_pose_embedding(trainer.pose_embeddings.as_ref(), 0);

    let (quaternion_grad, translation_grad) = pose_embedding::optional_pose_parameter_grads_fd(
        frame_pose_embedding.as_ref(),
        &camera,
        |render_camera| trainer.loss_for_camera(&gaussians, &frame, render_camera),
        &device,
    )
    .unwrap()
    .unwrap();
    let quaternion_grad = quaternion_grad.to_vec1::<f32>().unwrap();
    let translation_grad = translation_grad.to_vec1::<f32>().unwrap();

    assert_eq!(quaternion_grad.len(), 4);
    assert_eq!(translation_grad.len(), 3);
    assert!(quaternion_grad.iter().all(|value| value.is_finite()));
    assert!(translation_grad.iter().all(|value| value.is_finite()));
}

#[test]
fn scale_regularization_scatter_only_updates_visible_rows() {
    let device = Device::Cpu;
    let trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            litegs: super::LiteGsConfig {
                reg_weight: 0.5,
                ..super::LiteGsConfig::default()
            },
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    let gaussians = Splats::new(
        &[0.0, 0.0, 3.0, 1.0, 0.0, 3.0],
        &[2.0f32.ln(), 1.0f32.ln(), 0.5f32.ln(), 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.5, 0.25, 0.2, 0.4, 0.6],
        &device,
    )
    .unwrap();
    let projected = projected_with_visible_sources(&device, &[0]);

    let grad = test_optional_full_scale_regularization_grad(
        &gaussians,
        &projected,
        true,
        trainer.litegs.reg_weight,
    )
    .unwrap()
    .unwrap()
    .to_vec2::<f32>()
    .unwrap();

    assert_eq!(grad.len(), 2);
    assert!((grad[0][0] - (4.0 / 3.0)).abs() < 1e-6);
    assert!((grad[0][1] - (1.0 / 3.0)).abs() < 1e-6);
    assert!((grad[0][2] - (1.0 / 12.0)).abs() < 1e-6);
    assert!(grad[1].iter().all(|value| value.abs() < 1e-6));
}

#[test]
fn sh_parameter_grads_populate_sh_rest_terms() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        32,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    trainer.active_sh_degree = 1;
    let camera = DiffCamera::new(
        16.0,
        16.0,
        16.0,
        8.0,
        32,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new_with_sh(
        &[0.0, 1.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[
            crate::diff::diff_splat::rgb_to_sh0_value(0.5),
            crate::diff::diff_splat::rgb_to_sh0_value(0.5),
            crate::diff::diff_splat::rgb_to_sh0_value(0.5),
        ],
        &vec![0.0; 3 * 3],
        1,
        &device,
    )
    .unwrap();
    let projected = ProjectedGaussians {
        source_indices: Tensor::from_slice(&[0u32], 1, &device).unwrap(),
        u: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        v: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
        sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
        raw_sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
        raw_sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
        depth: Tensor::ones((1,), DType::F32, &device).unwrap(),
        opacity: Tensor::ones((1,), DType::F32, &device).unwrap(),
        opacity_logits: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        scale3d: Tensor::ones((1, 3), DType::F32, &device).unwrap(),
        colors: Tensor::from_slice(&[0.5f32, 0.5, 0.5], (1, 3), &device).unwrap(),
        min_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        max_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        min_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        max_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        visible_source_indices: vec![0],
        visible_count: 1,
        tile_bins: ProjectedTileBins::default(),
        staging_source: ProjectionStagingSource::TensorReadback,
    };
    let render_grads = Tensor::from_slice(&[1.0f32, 2.0, -0.5], (1, 3), &device).unwrap();

    let (position_grads, sh_0_grads, sh_rest_grads) = trainer
        .parameter_grads_from_render_color_grads(&gaussians, &projected, &render_grads, &camera)
        .unwrap();
    let position_grads = position_grads.to_vec2::<f32>().unwrap();
    let sh_0_grads = sh_0_grads.to_vec2::<f32>().unwrap();
    let sh_rest_grads = sh_rest_grads.to_vec3::<f32>().unwrap();
    let expected_basis = -metal_forward::SH_C1;

    assert!(position_grads[0].iter().all(|value| value.abs() < 1e-6));
    assert!((sh_0_grads[0][0] - SH_C0).abs() < 1e-6);
    assert!((sh_0_grads[0][1] - 2.0 * SH_C0).abs() < 1e-6);
    assert!((sh_0_grads[0][2] + 0.5 * SH_C0).abs() < 1e-6);
    assert!((sh_rest_grads[0][0][0] - expected_basis).abs() < 1e-6);
    assert!((sh_rest_grads[0][0][1] - 2.0 * expected_basis).abs() < 1e-6);
    assert!((sh_rest_grads[0][0][2] + 0.5 * expected_basis).abs() < 1e-6);
    assert!(sh_rest_grads[0][1].iter().all(|value| value.abs() < 1e-6));
    assert!(sh_rest_grads[0][2].iter().all(|value| value.abs() < 1e-6));
}

#[test]
fn sh_parameter_grads_include_position_terms_from_view_direction() {
    let device = Device::Cpu;
    let mut trainer = MetalTrainer::new(
        16,
        16,
        &TrainingConfig {
            training_profile: TrainingProfile::LiteGsMacV1,
            ..TrainingConfig::default()
        },
        device.clone(),
    )
    .unwrap();
    trainer.active_sh_degree = 1;
    let camera = DiffCamera::new(
        8.0,
        8.0,
        8.0,
        8.0,
        16,
        16,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[0.0, 0.0, 0.0],
        &device,
    )
    .unwrap();
    let gaussians = Splats::new_with_sh(
        &[1.2, 0.8, 2.5],
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0],
        &[
            crate::diff::diff_splat::rgb_to_sh0_value(0.6),
            crate::diff::diff_splat::rgb_to_sh0_value(0.55),
            crate::diff::diff_splat::rgb_to_sh0_value(0.5),
        ],
        &[
            0.05, 0.0, 0.0, //
            0.10, 0.0, 0.0, //
            -0.08, 0.0, 0.0,
        ],
        1,
        &device,
    )
    .unwrap();
    let projected = ProjectedGaussians {
        source_indices: Tensor::from_slice(&[0u32], 1, &device).unwrap(),
        u: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        v: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
        sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
        raw_sigma_x: Tensor::ones((1,), DType::F32, &device).unwrap(),
        raw_sigma_y: Tensor::ones((1,), DType::F32, &device).unwrap(),
        depth: Tensor::ones((1,), DType::F32, &device).unwrap(),
        opacity: Tensor::ones((1,), DType::F32, &device).unwrap(),
        opacity_logits: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        scale3d: Tensor::ones((1, 3), DType::F32, &device).unwrap(),
        colors: Tensor::from_slice(&[0.6f32, 0.55, 0.5], (1, 3), &device).unwrap(),
        min_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        max_x: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        min_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        max_y: Tensor::zeros((1,), DType::F32, &device).unwrap(),
        visible_source_indices: vec![0],
        visible_count: 1,
        tile_bins: ProjectedTileBins::default(),
        staging_source: ProjectionStagingSource::TensorReadback,
    };
    let render_grads = Tensor::from_slice(&[0.8f32, 0.0, 0.0], (1, 3), &device).unwrap();

    let (position_grads, _, _) = trainer
        .parameter_grads_from_render_color_grads(&gaussians, &projected, &render_grads, &camera)
        .unwrap();
    let analytic = position_grads.to_vec2::<f32>().unwrap()[0].clone();
    assert!(analytic.iter().any(|value| value.abs() > 1e-5));

    let base_position = gaussians.positions().to_vec2::<f32>().unwrap();
    let eps = 1e-3f32;
    for axis in 0..3 {
        let mut plus = base_position.clone();
        let mut minus = base_position.clone();
        plus[0][axis] += eps;
        minus[0][axis] -= eps;
        let plus_positions = Tensor::from_slice(&plus.concat(), (1, 3), &device).unwrap();
        let minus_positions = Tensor::from_slice(&minus.concat(), (1, 3), &device).unwrap();
        let plus_color = trainer
            .render_colors_for_camera(&gaussians, &plus_positions, &camera)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let minus_color = trainer
            .render_colors_for_camera(&gaussians, &minus_positions, &camera)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let fd = (plus_color[0][0] - minus_color[0][0]) * 0.8 / (2.0 * eps);
        assert!(
            (analytic[axis] - fd).abs() < 2e-2,
            "axis={axis} analytic={} fd={fd}",
            analytic[axis]
        );
    }
}
