use super::*;

#[test]
fn test_statistics_helper_mean() {
    let mut stats = StatisticsHelper::new(3);

    stats.mean2d_grad[1] = (0.4, 2.0);

    // Mean should be 0.2
    assert!((stats.get_mean2d_grad(1) - 0.2).abs() < 1e-6);
    assert_eq!(stats.get_mean2d_grad(0), 0.0);
}

#[test]
fn test_prune_mask_threshold_mode() {
    let config = DensityControllerConfig {
        opacity_threshold: 0.1,
        prune_mode: PruneMode::Threshold,
        ..Default::default()
    };
    let mut controller = DensityController::new(config, false);

    // Set up stats: index 0 has low opacity, index 1 is invisible, index 2 is healthy
    controller.stats.opacity = vec![0.05, 0.5, 0.8];
    controller.stats.visible_count = vec![1, 0, 5];
    controller.stats.resize(3);

    let mask = controller.get_prune_mask();

    // Index 0: low opacity -> prune
    assert!(mask[0]);
    // Index 1: invisible -> prune
    assert!(mask[1]);
    // Index 2: healthy -> keep
    assert!(!mask[2]);
}

#[test]
fn test_prune_mask_weight_mode() {
    let config = DensityControllerConfig {
        prune_mode: PruneMode::Weight,
        ..Default::default()
    };
    let mut controller = DensityController::new(config, false);

    // Set up stats: index 0 has weight_sum > 0, index 1 has weight_sum = 0
    controller.stats.resize(2);
    controller.stats.visible_count = vec![5, 3];
    // fragment_weight sum for index 0
    controller.stats.fragment_weight[0] = (0.5, 1.0); // mean = 0.5, weight_sum = 0.5 * 5 = 2.5
                                                      // fragment_weight for index 1 is 0
    controller.stats.fragment_weight[1] = (0.0, 0.0); // weight_sum = 0

    let mask = controller.get_prune_mask();

    // Index 0: weight_sum > 0 -> keep
    assert!(!mask[0]);
    // Index 1: weight_sum = 0 -> prune
    assert!(mask[1]);
}

#[test]
fn test_clone_split_masks() {
    let config = DensityControllerConfig {
        densify_grad_threshold: 0.001,
        percent_dense: 0.01,
        screen_extent: 1.0,
        opacity_threshold: 0.1,
        ..Default::default()
    };
    let mut controller = DensityController::new(config, false);

    controller.stats.resize(3);

    // Index 0: high grad, small scale -> clone
    controller.stats.mean2d_grad[0] = (0.01, 1.0); // mean = 0.01 >= 0.001
    controller.stats.max_scale[0] = 0.005; // <= 0.01
    controller.stats.opacity[0] = 0.5;

    // Index 1: high grad, large scale -> split
    controller.stats.mean2d_grad[1] = (0.01, 1.0);
    controller.stats.max_scale[1] = 0.05; // > 0.01
    controller.stats.opacity[1] = 0.5;

    // Index 2: low grad -> neither
    controller.stats.mean2d_grad[2] = (0.0001, 1.0); // < 0.001
    controller.stats.max_scale[2] = 0.005;
    controller.stats.opacity[2] = 0.5;

    let clone_mask = controller.get_clone_mask();
    let split_mask = controller.get_split_mask();

    assert!(clone_mask[0]);
    assert!(!split_mask[0]);

    assert!(!clone_mask[1]);
    assert!(split_mask[1]);

    assert!(!clone_mask[2]);
    assert!(!split_mask[2]);
}

#[test]
fn test_densify_budget() {
    let config = DensityControllerConfig {
        init_points_num: 100,
        target_primitives: 1000,
        densify_from: 0,
        densify_until: 100,
        ..Default::default()
    };
    let controller = DensityController::new(config, true);

    // At epoch 50: cur_target = 100 + (900/100)*50 = 550
    // If current = 400, deficit = 150
    // budget = min(150 + prune(10), 400) = 160
    let budget = controller.compute_densify_budget(400, 10, 50);
    assert_eq!(budget, 160);
}
