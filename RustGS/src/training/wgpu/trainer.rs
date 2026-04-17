use burn::prelude::*;
use burn::tensor::{s, Int, TensorData};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::training::topology::should_apply_topology_step;
use crate::training::TrainingConfig;

use super::backend::{GsBackendBase, GsDevice, GsDiffBackend};
use super::loss::{combined_loss, SsimConfig};
use super::optimizer::{AdamScaled, AdamScaledConfig};
use super::splats::{device_splats_to_host, DeviceSplats};
use super::{render_bwd, topology_apply, topology_bridge};

#[derive(Debug, Clone, Default)]
pub struct WgpuTrainingReport {
    pub losses: Vec<f32>,
    pub num_splats: Vec<usize>,
    pub completed_iterations: usize,
    pub cancelled: bool,
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
    num_observations: Tensor<GsDiffBackend, 1, Int>,
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

        Self {
            config,
            optimizer,
            device: device.clone(),
            grad_2d_accum: Tensor::zeros([initial_splats], &device),
            grad_color_accum: Tensor::zeros([initial_splats], &device),
            num_observations: Tensor::zeros([initial_splats], &device),
        }
    }

    fn position_lr_at(&self, iteration: usize) -> f32 {
        let lr_init = self.config.lr_position;
        let lr_final = self.config.lr_pos_final;
        if lr_final <= 0.0 || lr_final >= lr_init || self.config.iterations == 0 {
            return lr_init;
        }

        let t = (iteration.min(self.config.iterations) as f32) / (self.config.iterations as f32);
        lr_init * ((lr_final / lr_init).ln() * t).exp()
    }

    fn update_position_lr(&mut self, lr: f32) {
        let pos_cols =
            Tensor::<GsBackendBase, 2>::from_data(TensorData::from([[lr, lr, lr]]), &self.device);
        let rest = self.optimizer.transform_scaling().slice(s![.., 3..10]);
        self.optimizer
            .set_transform_scaling(Tensor::cat(vec![pos_cols, rest], 1));
    }

    pub async fn train_step(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        target_img: Tensor<GsDiffBackend, 3>,
        iteration: usize,
        frame_count: usize,
    ) -> f32 {
        let [height, width, _] = target_img.dims();
        let background = [0.0, 0.0, 0.0];

        let pred =
            render_bwd::render_splats(splats, camera, (width as u32, height as u32), background)
                .await;
        let pred_rgb = pred.slice(s![.., .., 0..3]);
        let loss = combined_loss(
            pred_rgb,
            target_img,
            0.8,
            0.2,
            &SsimConfig::default(),
            &self.device,
        );
        let loss_value = loss.clone().into_scalar_async().await.expect("loss scalar");
        if !loss_value.is_finite() {
            log::warn!(
                "Non-finite loss ({loss_value:.6}) at iteration {iteration}; skipping gradient update"
            );
            return loss_value;
        }
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
        let should_log_diagnostics = iteration <= 3 || iteration % 100 == 0;
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

        let pos_lr = self.position_lr_at(iteration.saturating_sub(1));
        self.update_position_lr(pos_lr);
        self.accumulate_gradients(&transforms_grad, &sh_grad);
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

    pub async fn train(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        cameras: &[GaussianCamera],
        target_images: &[Tensor<GsDiffBackend, 3>],
        num_iterations: usize,
    ) -> WgpuTrainingReport {
        struct NoopObserver;
        impl TrainingLoopObserver for NoopObserver {}
        let mut observer = NoopObserver;
        self.train_with_observer(
            splats,
            cameras,
            target_images,
            num_iterations,
            &mut observer,
        )
        .await
    }

    pub(crate) async fn train_with_observer(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        cameras: &[GaussianCamera],
        target_images: &[Tensor<GsDiffBackend, 3>],
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
            let loss = self
                .train_step(
                    splats,
                    &cameras[sample_idx],
                    target_images[sample_idx].clone(),
                    iteration + 1,
                    cameras.len(),
                )
                .await;

            report.losses.push(loss);
            report.num_splats.push(splats.num_splats());
            report.completed_iterations = report.losses.len();

            let metrics = TrainingIterationMetrics {
                iteration: iteration + 1,
                loss,
                gaussian_count: splats.num_splats(),
            };
            observer.on_iteration(metrics);

            if observer.should_emit_snapshot(metrics.iteration) {
                let host = device_splats_to_host(splats).await;
                observer.on_snapshot(metrics, host);
            }

            if iteration % 100 == 0 {
                log::info!(
                    "WGPU training step {} | loss={:.6} | splats={}",
                    iteration + 1,
                    loss,
                    splats.num_splats()
                );
            }

            if observer.should_cancel() {
                report.cancelled = true;
                break;
            }
        }

        report
    }

    fn should_apply_topology(&self, iteration: usize, frame_count: usize) -> bool {
        should_apply_topology_step(&self.config, iteration.max(1), frame_count)
    }

    fn accumulate_gradients(
        &mut self,
        transforms_grad: &Tensor<GsBackendBase, 2>,
        sh_grad: &Tensor<GsBackendBase, 3>,
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
        self.num_observations = self.num_observations.clone().add_scalar(1);
    }

    async fn apply_topology_mutations(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        iteration: usize,
        frame_count: usize,
    ) {
        let snapshot = topology_bridge::snapshot_for_topology(
            splats,
            &self.grad_2d_accum,
            &self.grad_color_accum,
            &self.num_observations,
        )
        .await;
        let plan = topology_bridge::plan_mutations(&snapshot, &self.config, iteration, frame_count);
        topology_apply::apply_mutations(splats, &plan, &self.device).await;
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

        let transform_scales = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[
                self.config.lr_position,
                self.config.lr_position,
                self.config.lr_position,
                self.config.lr_rotation,
                self.config.lr_rotation,
                self.config.lr_rotation,
                self.config.lr_rotation,
                self.config.lr_scale,
                self.config.lr_scale,
                self.config.lr_scale,
            ]]),
            &self.device,
        );
        let sh_scale_values = vec![self.config.lr_color; sh_coeffs.max(1)];
        let sh_scales = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(sh_scale_values, [1, sh_coeffs.max(1), 1]),
            &self.device,
        );
        let opacity_scales =
            Tensor::<GsBackendBase, 1>::from_floats([self.config.lr_opacity], &self.device);

        self.optimizer.set_transform_scaling(transform_scales);
        self.optimizer.set_sh_scaling(sh_scales);
        self.optimizer.set_opacity_scaling(opacity_scales);
        self.update_position_lr(self.position_lr_at(iteration.saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::{TrainingLoopObserver, WgpuTrainer};
    use crate::core::HostSplats;
    use crate::training::wgpu::backend::GsDevice;
    use crate::training::wgpu::{host_splats_to_device, GsDiffBackend};
    use crate::training::TrainingConfig;

    #[test]
    fn test_position_lr_decay() {
        let mut config = TrainingConfig::default();
        config.iterations = 1000;
        config.lr_position = 1.6e-4_f32;
        config.lr_pos_final = 1.6e-6_f32;

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

    #[tokio::test]
    async fn train_with_observer_stops_early_when_cancelled() {
        struct CancelledObserver;
        impl TrainingLoopObserver for CancelledObserver {
            fn should_cancel(&self) -> bool {
                true
            }
        }

        let mut config = TrainingConfig::default();
        config.iterations = 100;

        let device = GsDevice::default();
        let host = HostSplats::default();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host, &device);
        let mut trainer = WgpuTrainer::new(config, device, 0, 1);
        let mut observer = CancelledObserver;

        let report = trainer
            .train_with_observer(&mut splats, &[], &[], 100, &mut observer)
            .await;

        assert!(report.cancelled);
        assert_eq!(report.completed_iterations, 0);
        assert!(report.losses.is_empty());
    }
}
