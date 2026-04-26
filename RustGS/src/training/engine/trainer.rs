#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::{s, AllocationProperty, Bytes as BurnBytes, DType, Shape, TensorData};
use bytes::Bytes as SharedBytes;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::training::backward;
use crate::training::data::frame_loader::PrefetchFrameLoader;
use crate::training::metrics::{ParityLossCurveSample, ParityTopologyMetrics};
use crate::training::telemetry::{LiteGsOptimizerLrs, LiteGsTrainingTelemetry};
use crate::training::topology::TopologyMutationPlan;
use crate::training::topology::{apply_mutations, plan_mutations, snapshot_for_topology};
use crate::training::topology::{apply_topology_metrics_delta, should_apply_topology_step};
use crate::training::{LiteGsPruneMode, TrainingConfig};
use crate::TrainingError;

use super::backend::{GsBackendBase, GsDevice, GsDiffBackend};
use super::loss::{combined_loss_with_kernel, gaussian_kernel_1d, SsimConfig};
use super::optimizer::{AdamScaled, AdamScaledConfig};
use super::splats::{device_splats_to_host, DeviceSplats};

#[derive(Debug, Clone, Default)]
pub struct WgpuTrainingReport {
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub final_gaussian_count: usize,
    pub completed_iterations: usize,
    pub cancelled: bool,
    pub telemetry: LiteGsTrainingTelemetry,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TrainingIterationMetrics {
    pub iteration: usize,
    pub loss: f32,
    pub gaussian_count: usize,
}

pub(crate) trait TrainingLoopObserver {
    fn should_cancel(&self) -> bool {
        false
    }

    fn should_emit_progress(&self, _iteration: usize) -> bool {
        false
    }

    fn should_emit_snapshot(&self, _iteration: usize) -> bool {
        false
    }

    fn on_iteration(&mut self, _metrics: TrainingIterationMetrics) {}

    fn on_snapshot(&mut self, _metrics: TrainingIterationMetrics, _splats: HostSplats) {}
}

pub struct WgpuTrainer {
    config: TrainingConfig,
    optimizer: AdamScaled<GsBackendBase>,
    device: GsDevice,
    grad_2d_accum: Tensor<GsDiffBackend, 1>,
    grad_color_accum: Tensor<GsDiffBackend, 1>,
    num_observations: Tensor<GsDiffBackend, 1>,
    visible_observations: Tensor<GsDiffBackend, 1>,
    splat_birth_iterations: Vec<usize>,
    splat_invisible_windows: Vec<usize>,
    ssim_config: SsimConfig,
    ssim_kernel: Tensor<GsDiffBackend, 1>,
    telemetry: LiteGsTrainingTelemetry,
}

#[derive(Clone)]
struct SharedTargetImageBytes {
    data: Arc<Vec<f32>>,
}

impl AsRef<[u8]> for SharedTargetImageBytes {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
    }
}

fn target_image_tensor_data(
    target_image: &Arc<Vec<f32>>,
    image_dims: (usize, usize),
) -> TensorData {
    let (width, height) = image_dims;
    let shared = SharedBytes::from_owner(SharedTargetImageBytes {
        data: Arc::clone(target_image),
    });
    TensorData::from_bytes(
        BurnBytes::from_shared(shared, AllocationProperty::Native),
        Shape::new([height, width, 3]),
        DType::F32,
    )
}

#[cfg(test)]
fn target_image_tensor_data_owned(target_image: &[f32], image_dims: (usize, usize)) -> TensorData {
    let (width, height) = image_dims;
    TensorData::new(target_image.to_vec(), Shape::new([height, width, 3]))
}

impl WgpuTrainer {
    pub fn new(
        config: TrainingConfig,
        device: GsDevice,
        initial_splats: usize,
        sh_coeffs: usize,
    ) -> Self {
        let mut optimizer = AdamScaled::<GsBackendBase>::new(AdamScaledConfig {
            lr: 1.0,
            ..AdamScaledConfig::default()
        });

        let transform_scales = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[
                config.lr_position,
                config.lr_position,
                config.lr_position,
                config.lr_rotation,
                config.lr_rotation,
                config.lr_rotation,
                config.lr_rotation,
                config.lr_scale,
                config.lr_scale,
                config.lr_scale,
            ]]),
            &device,
        );
        let sh_scale_values = vec![config.lr_color; sh_coeffs.max(1)];
        let sh_scales = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(sh_scale_values, [1, sh_coeffs.max(1), 1]),
            &device,
        );
        let opacity_scales = Tensor::<GsBackendBase, 1>::from_floats([config.lr_opacity], &device);

        optimizer.set_transform_scaling(transform_scales);
        optimizer.set_sh_scaling(sh_scales);
        optimizer.set_opacity_scaling(opacity_scales);
        let ssim_config = SsimConfig::default();
        let ssim_kernel = gaussian_kernel_1d::<GsDiffBackend>(&ssim_config, &device);

        let telemetry = initial_training_telemetry(&config, initial_splats);

        Self {
            config,
            optimizer,
            device: device.clone(),
            grad_2d_accum: Tensor::zeros([initial_splats], &device),
            grad_color_accum: Tensor::zeros([initial_splats], &device),
            num_observations: Tensor::zeros([initial_splats], &device),
            visible_observations: Tensor::zeros([initial_splats], &device),
            splat_birth_iterations: vec![0; initial_splats],
            splat_invisible_windows: vec![0; initial_splats],
            ssim_config,
            ssim_kernel,
            telemetry,
        }
    }

    fn lr_at(&self, initial: f32, final_value: f32, iteration: usize) -> f32 {
        if final_value <= 0.0 || final_value >= initial || self.config.iterations == 0 {
            return initial;
        }

        let decay_iterations = self
            .config
            .lr_decay_iterations
            .unwrap_or(self.config.iterations)
            .max(1);
        let t = (iteration.min(decay_iterations) as f32) / (decay_iterations as f32);
        initial * ((final_value / initial).ln() * t).exp()
    }

    fn position_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(self.config.lr_position, self.config.lr_pos_final, iteration)
    }

    fn scale_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(self.config.lr_scale, self.config.lr_scale_final, iteration)
    }

    fn rotation_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.lr_rotation,
            self.config.lr_rotation_final,
            iteration,
        )
    }

    fn opacity_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.lr_opacity,
            self.config.lr_opacity_final,
            iteration,
        )
    }

    fn color_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(self.config.lr_color, self.config.lr_color_final, iteration)
    }

    fn update_optimizer_lrs(&mut self, iteration: usize, sh_coeffs: usize) {
        let pos_lr = self.position_lr_at(iteration);
        let rotation_lr = self.rotation_lr_at(iteration);
        let scale_lr = self.scale_lr_at(iteration);
        let transform_scales = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[
                pos_lr,
                pos_lr,
                pos_lr,
                rotation_lr,
                rotation_lr,
                rotation_lr,
                rotation_lr,
                scale_lr,
                scale_lr,
                scale_lr,
            ]]),
            &self.device,
        );
        let sh_scale_values = vec![self.color_lr_at(iteration); sh_coeffs.max(1)];
        let sh_scales = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(sh_scale_values, [1, sh_coeffs.max(1), 1]),
            &self.device,
        );
        let opacity_scales =
            Tensor::<GsBackendBase, 1>::from_floats([self.opacity_lr_at(iteration)], &self.device);

        self.optimizer.set_transform_scaling(transform_scales);
        self.optimizer.set_sh_scaling(sh_scales);
        self.optimizer.set_opacity_scaling(opacity_scales);
    }

    pub async fn train_step(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        target_image: &Arc<Vec<f32>>,
        image_dims: (usize, usize),
        iteration: usize,
        frame_count: usize,
        read_loss_scalar: bool,
    ) -> Option<f32> {
        let (width, height) = image_dims;
        let background = [0.0, 0.0, 0.0];
        let target_img = Tensor::<GsDiffBackend, 3>::from_data(
            target_image_tensor_data(target_image, image_dims),
            &self.device,
        );

        let rendered = backward::render_splats_with_visibility(
            splats,
            camera,
            (width as u32, height as u32),
            background,
        )
        .await;
        let pred_rgb = rendered.image.slice(s![.., .., 0..3]);
        let loss = combined_loss_with_kernel(
            pred_rgb,
            target_img,
            0.8,
            0.2,
            &self.ssim_config,
            self.ssim_kernel.clone(),
        );
        let loss_value = if read_loss_scalar {
            let loss_value = loss.clone().into_scalar_async().await.expect("loss scalar");
            if !loss_value.is_finite() {
                log::warn!(
                    "Non-finite loss ({loss_value:.6}) at iteration {iteration}; skipping gradient update"
                );
                return Some(loss_value);
            }
            Some(loss_value)
        } else {
            None
        };
        let mut grads = loss.backward();

        let transforms_grad = splats
            .transforms
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.transforms.val().inner().zeros_like());
        let sh_grad = splats
            .sh_coeffs
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.sh_coeffs.val().inner().zeros_like());
        let opacity_grad = splats
            .raw_opacities
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.raw_opacities.val().inner().zeros_like());

        // Brush keeps a strong gradient-validation path; mirror that observability here
        // so we can quickly spot silent no-op training regressions.
        let should_log_diagnostics = iteration <= 3 || iteration.is_multiple_of(100);
        let grad_transforms_for_diag = if should_log_diagnostics {
            Some(transforms_grad.clone())
        } else {
            None
        };
        let grad_sh_for_diag = if should_log_diagnostics {
            Some(sh_grad.clone())
        } else {
            None
        };
        let grad_opacity_for_diag = if should_log_diagnostics {
            Some(opacity_grad.clone())
        } else {
            None
        };
        let prev_transforms = if should_log_diagnostics {
            Some(splats.transforms.val().inner())
        } else {
            None
        };
        let prev_sh = if should_log_diagnostics {
            Some(splats.sh_coeffs.val().inner())
        } else {
            None
        };
        let prev_opacity = if should_log_diagnostics {
            Some(splats.raw_opacities.val().inner())
        } else {
            None
        };

        self.update_optimizer_lrs(
            iteration.saturating_sub(1),
            splats.sh_coeffs.val().dims()[1],
        );
        self.accumulate_gradients(
            &transforms_grad,
            &sh_grad,
            &rendered.visible,
            self.uses_visibility_pruning(),
        );
        self.optimizer
            .step_device_splats(splats, transforms_grad, sh_grad, opacity_grad);

        if should_log_diagnostics {
            let grad_transforms_mean_abs = grad_transforms_for_diag
                .expect("transforms grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("transforms grad mean");
            let grad_sh_mean_abs = grad_sh_for_diag
                .expect("sh grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("sh grad mean");
            let grad_opacity_mean_abs = grad_opacity_for_diag
                .expect("opacity grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("opacity grad mean");

            let delta_transforms_mean_abs = (splats.transforms.val().inner()
                - prev_transforms.expect("prev transforms for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("transforms delta mean");
            let delta_sh_mean_abs = (splats.sh_coeffs.val().inner()
                - prev_sh.expect("prev sh for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("sh delta mean");
            let delta_opacity_mean_abs = (splats.raw_opacities.val().inner()
                - prev_opacity.expect("prev opacity for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("opacity delta mean");

            log::info!(
                "WGPU train diagnostics step {} | grad_mean_abs: transforms={:.6e}, sh={:.6e}, opacity={:.6e} | delta_mean_abs: transforms={:.6e}, sh={:.6e}, opacity={:.6e}",
                iteration,
                grad_transforms_mean_abs,
                grad_sh_mean_abs,
                grad_opacity_mean_abs,
                delta_transforms_mean_abs,
                delta_sh_mean_abs,
                delta_opacity_mean_abs,
            );
        }

        if self.should_apply_topology(iteration, frame_count) {
            self.apply_topology_mutations(splats, iteration, frame_count)
                .await;
        }

        loss_value
    }

    #[cfg(test)]
    pub(crate) async fn train_with_observer(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        cameras: &[GaussianCamera],
        target_images: &[Arc<Vec<f32>>],
        image_dims: (usize, usize),
        num_iterations: usize,
        observer: &mut dyn TrainingLoopObserver,
    ) -> WgpuTrainingReport {
        let mut report = WgpuTrainingReport::default();
        let mut rng = StdRng::seed_from_u64(self.config.frame_shuffle_seed);

        for iteration in 0..num_iterations {
            if observer.should_cancel() {
                report.cancelled = true;
                break;
            }

            let sample_idx = if cameras.len() == 1 {
                0
            } else {
                rng.gen_range(0..cameras.len())
            };
            let iteration_idx = iteration + 1;
            let emit_progress = observer.should_emit_progress(iteration_idx);
            let emit_snapshot = observer.should_emit_snapshot(iteration_idx);
            let should_log_step = iteration % 100 == 0;
            let should_read_loss = emit_progress
                || emit_snapshot
                || should_log_step
                || iteration_idx == num_iterations;
            let loss = self
                .train_step(
                    splats,
                    &cameras[sample_idx],
                    &target_images[sample_idx],
                    image_dims,
                    iteration_idx,
                    cameras.len(),
                    should_read_loss,
                )
                .await;
            report.completed_iterations = iteration_idx;
            report.final_gaussian_count = splats.num_splats();

            if let Some(loss) = loss {
                report.final_loss = Some(loss);
                report.final_step_loss = Some(loss);
                self.record_loss_sample(
                    iteration_idx,
                    sample_idx,
                    loss,
                    should_log_step || iteration_idx == num_iterations,
                );
                let metrics = TrainingIterationMetrics {
                    iteration: iteration_idx,
                    loss,
                    gaussian_count: splats.num_splats(),
                };
                if emit_progress {
                    observer.on_iteration(metrics);
                }
                if emit_snapshot {
                    let host = device_splats_to_host(splats).await;
                    observer.on_snapshot(metrics, host);
                }
                if should_log_step {
                    log::info!(
                        "WGPU training step {} | loss={:.6} | splats={}",
                        iteration_idx,
                        loss,
                        splats.num_splats()
                    );
                }
            } else if should_log_step {
                log::info!(
                    "WGPU training step {} | splats={}",
                    iteration_idx,
                    splats.num_splats()
                );
            }

            if observer.should_cancel() {
                report.cancelled = true;
                break;
            }
        }

        self.finish_report(&mut report);
        report
    }

    pub(crate) async fn train_with_frame_loader(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        cameras: &[GaussianCamera],
        frame_order: &[usize],
        frame_loader: &mut PrefetchFrameLoader,
        image_dims: (usize, usize),
        num_iterations: usize,
        observer: &mut dyn TrainingLoopObserver,
    ) -> Result<WgpuTrainingReport, TrainingError> {
        if cameras.is_empty() || cameras.len() != frame_order.len() {
            return Err(TrainingError::InvalidInput(format!(
                "training frame order length ({}) must match camera count ({}) and be non-empty",
                frame_order.len(),
                cameras.len()
            )));
        }

        let mut report = WgpuTrainingReport::default();
        let mut rng = StdRng::seed_from_u64(self.config.frame_shuffle_seed);
        self.telemetry.topology.total_epochs =
            Some(training_epoch_count(num_iterations, cameras.len()));

        for iteration in 0..num_iterations {
            if observer.should_cancel() {
                report.cancelled = true;
                break;
            }

            let sample_idx = if cameras.len() == 1 {
                0
            } else {
                rng.gen_range(0..cameras.len())
            };
            let frame_idx = frame_order[sample_idx];
            frame_loader.prefetch_order_window(frame_order, sample_idx)?;
            let decoded = frame_loader.get(frame_idx)?;
            let target_image = decoded.target_rgb.clone().ok_or_else(|| {
                TrainingError::TrainingFailed(format!(
                    "frame loader did not prepare target_rgb for frame {frame_idx}"
                ))
            })?;

            let iteration_idx = iteration + 1;
            let emit_progress = observer.should_emit_progress(iteration_idx);
            let emit_snapshot = observer.should_emit_snapshot(iteration_idx);
            let should_log_step = iteration.is_multiple_of(100);
            let should_read_loss = emit_progress
                || emit_snapshot
                || should_log_step
                || iteration_idx == num_iterations;
            let loss = self
                .train_step(
                    splats,
                    &cameras[sample_idx],
                    &target_image,
                    image_dims,
                    iteration_idx,
                    cameras.len(),
                    should_read_loss,
                )
                .await;
            report.completed_iterations = iteration_idx;
            report.final_gaussian_count = splats.num_splats();

            if let Some(loss) = loss {
                report.final_loss = Some(loss);
                report.final_step_loss = Some(loss);
                self.record_loss_sample(
                    iteration_idx,
                    frame_idx,
                    loss,
                    should_log_step || iteration_idx == num_iterations,
                );
                let metrics = TrainingIterationMetrics {
                    iteration: iteration_idx,
                    loss,
                    gaussian_count: splats.num_splats(),
                };
                if emit_progress {
                    observer.on_iteration(metrics);
                }
                if emit_snapshot {
                    let host = device_splats_to_host(splats).await;
                    observer.on_snapshot(metrics, host);
                }
                if should_log_step {
                    log::info!(
                        "WGPU training step {} | loss={:.6} | splats={}",
                        iteration_idx,
                        loss,
                        splats.num_splats()
                    );
                }
            } else if should_log_step {
                log::info!(
                    "WGPU training step {} | splats={}",
                    iteration_idx,
                    splats.num_splats()
                );
            }

            if observer.should_cancel() {
                report.cancelled = true;
                break;
            }
        }

        self.finish_report(&mut report);
        Ok(report)
    }

    fn should_apply_topology(&self, iteration: usize, frame_count: usize) -> bool {
        should_apply_topology_step(&self.config, iteration.max(1), frame_count)
    }

    fn accumulate_gradients(
        &mut self,
        transforms_grad: &Tensor<GsBackendBase, 2>,
        sh_grad: &Tensor<GsBackendBase, 3>,
        visible: &Tensor<GsDiffBackend, 1>,
        use_actual_visibility: bool,
    ) {
        let grad_2d = transforms_grad
            .clone()
            .slice(s![.., 0..3])
            .abs()
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let grad_color = sh_grad
            .clone()
            .abs()
            .mean_dim(2)
            .mean_dim(1)
            .squeeze_dim::<2>(2)
            .squeeze_dim::<1>(1);

        let grad_2d = Tensor::<GsDiffBackend, 1>::from_inner(grad_2d);
        let grad_color = Tensor::<GsDiffBackend, 1>::from_inner(grad_color);

        self.grad_2d_accum = self.grad_2d_accum.clone() + grad_2d;
        self.grad_color_accum = self.grad_color_accum.clone() + grad_color;
        self.num_observations = self.num_observations.clone().add_scalar(1.0);
        let visible_increment = if use_actual_visibility {
            visible.clone()
        } else {
            Tensor::<GsDiffBackend, 1>::ones([visible.dims()[0]], &self.device)
        };
        self.visible_observations = self.visible_observations.clone() + visible_increment;
    }

    fn uses_visibility_pruning(&self) -> bool {
        matches!(self.config.litegs.prune_mode, LiteGsPruneMode::Threshold)
    }

    async fn apply_topology_mutations(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        iteration: usize,
        frame_count: usize,
    ) {
        let mut snapshot = snapshot_for_topology(
            splats,
            &self.grad_2d_accum,
            &self.grad_color_accum,
            &self.num_observations,
            &self.visible_observations,
        )
        .await;
        self.update_topology_visibility_state(
            snapshot.splats.len(),
            &snapshot.visible_observations,
            iteration,
        );
        snapshot.splat_ages = self.splat_ages_at(iteration, snapshot.splats.len());
        snapshot.invisible_windows = self
            .splat_invisible_windows
            .iter()
            .copied()
            .take(snapshot.splats.len())
            .collect();
        let plan = plan_mutations(&snapshot, &self.config, iteration, frame_count);
        apply_topology_metrics_delta(&mut self.telemetry.topology, plan.aftermath.metrics_delta);
        if plan.mutates_splats() {
            apply_mutations(splats, &snapshot.splats, &plan, &self.device);
            self.remap_topology_visibility_state(&plan, iteration);
        }
        self.optimizer.reset();
        self.reset_accumulators(
            splats.num_splats(),
            splats.sh_coeffs.val().dims()[1],
            iteration,
        );
    }

    fn reset_accumulators(&mut self, num_splats: usize, sh_coeffs: usize, iteration: usize) {
        self.grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.grad_color_accum = Tensor::zeros([num_splats], &self.device);
        self.num_observations = Tensor::zeros([num_splats], &self.device);
        self.visible_observations = Tensor::zeros([num_splats], &self.device);

        self.update_optimizer_lrs(iteration.saturating_sub(1), sh_coeffs);
    }

    fn update_topology_visibility_state(
        &mut self,
        num_splats: usize,
        visible_observations: &[f32],
        iteration: usize,
    ) {
        self.ensure_topology_visibility_state(num_splats, iteration);
        for idx in 0..num_splats {
            let visible = visible_observations
                .get(idx)
                .copied()
                .is_some_and(|value| value.is_finite() && value > 0.0);
            if visible {
                self.splat_invisible_windows[idx] = 0;
            } else {
                self.splat_invisible_windows[idx] =
                    self.splat_invisible_windows[idx].saturating_add(1);
            }
        }
    }

    fn ensure_topology_visibility_state(&mut self, num_splats: usize, iteration: usize) {
        if self.splat_birth_iterations.len() < num_splats {
            self.splat_birth_iterations.resize(num_splats, iteration);
        } else {
            self.splat_birth_iterations.truncate(num_splats);
        }

        if self.splat_invisible_windows.len() < num_splats {
            self.splat_invisible_windows.resize(num_splats, 0);
        } else {
            self.splat_invisible_windows.truncate(num_splats);
        }
    }

    fn splat_ages_at(&self, iteration: usize, num_splats: usize) -> Vec<usize> {
        (0..num_splats)
            .map(|idx| {
                iteration.saturating_sub(
                    self.splat_birth_iterations
                        .get(idx)
                        .copied()
                        .unwrap_or(iteration),
                )
            })
            .collect()
    }

    fn remap_topology_visibility_state(&mut self, plan: &TopologyMutationPlan, iteration: usize) {
        let previous_birth_iterations = self.splat_birth_iterations.clone();
        let previous_invisible_windows = self.splat_invisible_windows.clone();
        let origins = plan.origins();
        self.splat_birth_iterations = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_birth_iterations.get(idx).copied())
                    .unwrap_or(iteration)
            })
            .collect();
        self.splat_invisible_windows = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_invisible_windows.get(idx).copied())
                    .unwrap_or(0)
            })
            .collect();
    }

    fn record_loss_sample(
        &mut self,
        iteration: usize,
        frame_idx: usize,
        loss: f32,
        keep_curve_sample: bool,
    ) {
        self.telemetry.final_loss = Some(loss);
        self.telemetry.final_step_loss = Some(loss);
        self.telemetry.loss_terms.total = Some(loss);
        if keep_curve_sample {
            self.telemetry
                .loss_curve_samples
                .push(ParityLossCurveSample {
                    iteration,
                    frame_idx,
                    l1: None,
                    ssim: None,
                    depth: None,
                    total: Some(loss),
                    depth_valid_pixels: self.telemetry.depth_valid_pixels,
                });
        }
    }

    fn finish_report(&mut self, report: &mut WgpuTrainingReport) {
        self.telemetry.final_loss = report.final_loss;
        self.telemetry.final_step_loss = report.final_loss;
        self.telemetry.topology.final_gaussians = Some(report.final_gaussian_count);
        report.telemetry = self.telemetry.clone();
    }
}

fn initial_training_telemetry(
    config: &TrainingConfig,
    initial_splats: usize,
) -> LiteGsTrainingTelemetry {
    LiteGsTrainingTelemetry {
        active_sh_degree: Some(config.litegs.sh_degree),
        rotation_frozen: config.lr_rotation == 0.0,
        learning_rates: LiteGsOptimizerLrs {
            xyz: Some(config.lr_position),
            sh_0: Some(config.lr_color),
            sh_rest: Some(config.lr_color),
            opacity: Some(config.lr_opacity),
            scale: Some(config.lr_scale),
            rot: Some(config.lr_rotation),
        },
        topology: ParityTopologyMetrics {
            initialization_gaussians: Some(initial_splats),
            topology_freeze_epoch: config.litegs.topology_freeze_after_epoch,
            ..ParityTopologyMetrics::default()
        },
        ..LiteGsTrainingTelemetry::default()
    }
}

fn training_epoch_count(iterations: usize, frame_count: usize) -> usize {
    if frame_count == 0 {
        0
    } else {
        (iterations / frame_count).max(1)
    }
}

#[cfg(test)]
mod tests {
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
            lr_position: 1.6e-4_f32,
            lr_pos_final: 1.6e-6_f32,
            ..TrainingConfig::default()
        };

        let trainer = WgpuTrainer::new(config.clone(), GsDevice::default(), 1, 1);
        let at_0 = trainer.position_lr_at(0);
        let at_mid = trainer.position_lr_at(500);
        let at_end = trainer.position_lr_at(config.iterations);

        assert!(
            (at_0 - config.lr_position).abs() < 1e-8,
            "initial LR should equal lr_position"
        );
        assert!(
            (at_end - config.lr_pos_final).abs() < config.lr_pos_final * 0.01,
            "final LR should ≈ lr_pos_final"
        );
        assert!(
            at_mid < config.lr_position && at_mid > config.lr_pos_final,
            "mid LR should be between bounds"
        );
    }

    #[test]
    fn test_group_lr_decay_uses_configured_horizon() {
        let config = TrainingConfig {
            iterations: 30_000,
            lr_decay_iterations: Some(10_000),
            lr_scale: 5e-3,
            lr_scale_final: 5e-4,
            lr_rotation: 1e-3,
            lr_rotation_final: 1e-4,
            lr_opacity: 5e-2,
            lr_opacity_final: 5e-3,
            lr_color: 2.5e-3,
            lr_color_final: 2.5e-4,
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
}
