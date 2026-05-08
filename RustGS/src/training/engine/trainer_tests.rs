use std::sync::Arc;

use burn::tensor::{DType, Tensor};

use super::super::backend::{GsBackendBase, GsDevice};
use super::{
    target_image_tensor_data, target_image_tensor_data_owned, TrainingLoopObserver, WgpuTrainer,
};
use crate::core::HostSplats;
use crate::training::engine::{host_splats_to_device, GsDiffBackend};
use crate::training::TrainingConfig;

#[test]
fn test_position_lr_decay() {
    let config = TrainingConfig {
        iterations: 1000,
        optimizer: crate::training::TrainingOptimizerConfig {
            lr_position: 1.6e-4_f32,
            lr_pos_final: 1.6e-6_f32,
            ..crate::training::TrainingOptimizerConfig::default()
        },
        ..TrainingConfig::default()
    };

    let trainer = WgpuTrainer::new(config.clone(), GsDevice::default(), 1, 1);
    let at_0 = trainer.position_lr_at(0);
    let at_mid = trainer.position_lr_at(500);
    let at_end = trainer.position_lr_at(config.iterations);

    assert!(
        (at_0 - config.optimizer.lr_position).abs() < 1e-8,
        "initial LR should equal lr_position"
    );
    assert!(
        (at_end - config.optimizer.lr_pos_final).abs() < config.optimizer.lr_pos_final * 0.01,
        "final LR should ≈ lr_pos_final"
    );
    assert!(
        at_mid < config.optimizer.lr_position && at_mid > config.optimizer.lr_pos_final,
        "mid LR should be between bounds"
    );
}

#[test]
fn test_group_lr_decay_uses_configured_horizon() {
    let config = TrainingConfig {
        iterations: 30_000,
        optimizer: crate::training::TrainingOptimizerConfig {
            lr_decay_iterations: Some(10_000),
            lr_scale: 5e-3,
            lr_scale_final: 5e-4,
            lr_rotation: 1e-3,
            lr_rotation_final: 1e-4,
            lr_opacity: 5e-2,
            lr_opacity_final: 5e-3,
            lr_color: 2.5e-3,
            lr_color_final: 2.5e-4,
            ..crate::training::TrainingOptimizerConfig::default()
        },
        ..TrainingConfig::default()
    };

    let trainer = WgpuTrainer::new(config, GsDevice::default(), 1, 1);

    assert!((trainer.scale_lr_at(10_000) - 5e-4).abs() < 1e-8);
    assert!((trainer.rotation_lr_at(10_000) - 1e-4).abs() < 1e-8);
    assert!((trainer.opacity_lr_at(10_000) - 5e-3).abs() < 1e-8);
    assert!((trainer.color_lr_at(10_000) - 2.5e-4).abs() < 1e-8);
    assert!((trainer.color_lr_at(30_000) - 2.5e-4).abs() < 1e-8);
}

#[tokio::test]
async fn train_with_observer_stops_early_when_cancelled() {
    struct CancelledObserver;
    impl TrainingLoopObserver for CancelledObserver {
        fn should_cancel(&self) -> bool {
            true
        }
    }

    let config = TrainingConfig {
        iterations: 100,
        ..TrainingConfig::default()
    };

    let device = GsDevice::default();
    let host = HostSplats::default();
    let mut splats = host_splats_to_device::<GsDiffBackend>(&host, &device);
    let mut trainer = WgpuTrainer::new(config, device, 0, 1);
    let mut observer = CancelledObserver;
    let target_images: Vec<Arc<Vec<f32>>> = Vec::new();

    let report = trainer
        .train_with_observer(&mut splats, &[], &target_images, (0, 0), 100, &mut observer)
        .await;

    assert!(report.cancelled);
    assert_eq!(report.completed_iterations, 0);
    assert_eq!(report.final_loss, None);
}

#[test]
fn target_image_tensor_data_wraps_shared_image_without_reformatting() {
    let target_image = Arc::new(vec![0.0f32, 0.25, 0.5, 0.75, 1.0, 0.125]);
    let tensor_data = target_image_tensor_data(&target_image, (1, 2));

    assert_eq!(tensor_data.shape.dims(), [2, 1, 3]);
    assert_eq!(tensor_data.dtype, DType::F32);
    let values = tensor_data
        .as_slice::<f32>()
        .expect("target image tensor data should decode as f32");
    assert_eq!(values, target_image.as_slice());
}

#[tokio::test(flavor = "current_thread")]
async fn target_image_tensor_data_round_trips_through_gpu_upload() {
    let target_image = Arc::new(vec![0.0f32, 0.25, 0.5, 0.75, 1.0, 0.125]);
    let device = GsDevice::default();
    let tensor = Tensor::<GsBackendBase, 3>::from_data(
        target_image_tensor_data(&target_image, (1, 2)),
        &device,
    );

    let readback = tensor
        .into_data_async()
        .await
        .expect("gpu readback should succeed");
    let values = readback
        .as_slice::<f32>()
        .expect("readback tensor data should decode as f32");
    assert_eq!(values, target_image.as_slice());
}

#[tokio::test(flavor = "current_thread")]
async fn shared_and_owned_target_uploads_match_for_full_frame() {
    let image_dims = (640usize, 480usize);
    let len = image_dims.0 * image_dims.1 * 3;
    let owned = (0..len)
        .map(|idx| ((idx % 251) as f32) / 250.0)
        .collect::<Vec<_>>();
    let shared = Arc::new(owned.clone());
    let device = GsDevice::default();

    let shared_tensor = Tensor::<GsBackendBase, 3>::from_data(
        target_image_tensor_data(&shared, image_dims),
        &device,
    );
    let owned_tensor = Tensor::<GsBackendBase, 3>::from_data(
        target_image_tensor_data_owned(&owned, image_dims),
        &device,
    );

    let shared_data = shared_tensor
        .into_data_async()
        .await
        .expect("shared upload readback");
    let owned_data = owned_tensor
        .into_data_async()
        .await
        .expect("owned upload readback");

    let shared_values = shared_data.as_slice::<f32>().expect("shared readback f32");
    let owned_values = owned_data.as_slice::<f32>().expect("owned readback f32");
    assert_eq!(shared_values, owned_values);
}
