//! Burn+wgpu training backend

pub mod backend;
pub mod gpu_primitives;
pub mod loss;
pub mod optimizer;
pub mod render;
pub mod render_bwd;
pub mod splats;
pub mod topology_apply;
pub mod topology_bridge;
pub mod trainer;

use burn::prelude::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use std::time::Instant;

use crate::core::GaussianCamera;
use crate::training::data::frame_loader::{
    ordered_frame_indices, FrameLoaderOptions, PrefetchFrameLoader,
};
use crate::training::data::frame_targets::resize_rgb_u8_to_f32;
use crate::training::data::init_map::build_initial_splats;
use crate::training::pipeline::events::{
    emit_training_event, TrainingEvent, TrainingEventRoute, TrainingPlanSelected, TrainingRun,
    TrainingRunCompleted, TrainingRunReport, TrainingRunStarted,
};
use crate::training::{scaled_dimensions, HostSplats, TrainingConfig};
use crate::{Intrinsics, TrainingDataset, TrainingError};

pub use backend::{GsBackend, GsBackendBase, GsDevice, GsDiffBackend};
pub use render_bwd::render_splats;
pub use splats::{device_splats_to_host, host_splats_to_device, DeviceSplats};
pub use trainer::{WgpuTrainer, WgpuTrainingReport};

pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    train_splats_with_report(dataset, config).map(TrainingRun::into_splats)
}

pub fn train_splats_with_report(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    let mut sink = |_event| {};
    train_splats_with_events(dataset, config, &mut sink)
}

pub fn train_splats_with_events<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut on_event: F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent),
{
    emit_training_event(
        &mut on_event,
        TrainingEvent::RunStarted(TrainingRunStarted {
            profile: config.training_profile,
            iterations: config.iterations,
            frame_count: dataset.poses.len(),
            input_point_count: dataset.initial_points.len(),
        }),
    );
    emit_training_event(
        &mut on_event,
        TrainingEvent::PlanSelected(TrainingPlanSelected {
            route: TrainingEventRoute::Standard,
        }),
    );

    let run = run_training(dataset, config)?;

    emit_training_event(
        &mut on_event,
        TrainingEvent::RunCompleted(TrainingRunCompleted {
            report: run.report.clone(),
        }),
    );
    Ok(run)
}

fn run_training(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingRun, TrainingError> {
    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training dataset does not contain any poses".to_string(),
        ));
    }

    let started_at = Instant::now();
    let input_width = dataset.intrinsics.width as usize;
    let input_height = dataset.intrinsics.height as usize;
    let (target_width, target_height) =
        scaled_dimensions(input_width, input_height, config.metal_render_scale);
    let initial_splats = build_initial_splats(dataset, config)?;

    let mut loader = PrefetchFrameLoader::new(
        dataset,
        config,
        FrameLoaderOptions {
            cache_capacity: config.frame_cache_capacity,
            prefetch_ahead: config.frame_prefetch_ahead,
        },
    )?;

    let frame_order = ordered_frame_indices(dataset.poses.len(), 1, config.frame_shuffle_seed);
    let mut cameras = Vec::with_capacity(frame_order.len());
    let mut target_images = Vec::with_capacity(frame_order.len());

    for (cursor, &pose_idx) in frame_order.iter().enumerate() {
        loader.prefetch_order_window(&frame_order, cursor)?;
        let pose = &dataset.poses[pose_idx];
        let decoded = loader.get(pose_idx)?;

        cameras.push(gaussian_camera_from_scene_pose(
            &pose.pose,
            dataset.intrinsics,
            target_width,
            target_height,
        ));
        target_images.push(resize_rgb_u8_to_f32(
            &decoded.color_u8,
            input_width,
            input_height,
            target_width,
            target_height,
        ));
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| TrainingError::TrainingFailed(format!("failed to build tokio runtime: {err}")))?
        .block_on(async move {
            let device = GsDevice::default();
            let mut device_splats = host_splats_to_device::<GsDiffBackend>(&initial_splats, &device);
            let target_tensors =
                convert_images_to_tensors::<GsDiffBackend>(&target_images, target_width, target_height, &device);
            let sh_coeffs = device_splats.sh_coeffs.val().dims()[1];
            let mut trainer =
                WgpuTrainer::new(config.clone(), device.clone(), device_splats.num_splats(), sh_coeffs);
            let report = trainer
                .train(&mut device_splats, &cameras, &target_tensors, config.iterations)
                .await;
            let splats = device_splats_to_host(&device_splats).await;
            Ok(build_training_run(splats, report, started_at.elapsed()))
        })
}

fn gaussian_camera_from_scene_pose(
    pose: &crate::SE3,
    intrinsics: crate::Intrinsics,
    target_width: usize,
    target_height: usize,
) -> GaussianCamera {
    let sx = target_width as f32 / intrinsics.width as f32;
    let sy = target_height as f32 / intrinsics.height as f32;
    let scaled_intrinsics = Intrinsics::new(
        intrinsics.fx * sx,
        intrinsics.fy * sy,
        intrinsics.cx * sx,
        intrinsics.cy * sy,
        target_width as u32,
        target_height as u32,
    );
    GaussianCamera::new(scaled_intrinsics, pose.inverse())
}

fn build_training_run(
    splats: HostSplats,
    report: WgpuTrainingReport,
    elapsed: std::time::Duration,
) -> TrainingRun {
    let final_loss = report.losses.last().copied();
    let gaussian_count = report.num_splats.last().copied().unwrap_or_else(|| splats.len());

    TrainingRun {
        report: TrainingRunReport {
            elapsed,
            final_loss,
            final_step_loss: final_loss,
            gaussian_count,
            sh_degree: splats.sh_degree(),
            telemetry: None,
        },
        splats,
    }
}

fn convert_images_to_tensors<B: Backend>(
    images: &[Vec<f32>],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Vec<Tensor<B, 3>> {
    images
        .iter()
        .map(|image| {
            Tensor::<B, 3>::from_data(
                TensorData::new(image.clone(), Shape::new([height, width, 3])),
                device,
            )
        })
        .collect()
}
