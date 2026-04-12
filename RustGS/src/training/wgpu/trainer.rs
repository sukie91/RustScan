use burn::prelude::*;
use burn::tensor::{s, Int, TensorData};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::core::GaussianCamera;
use crate::training::TrainingConfig;

use super::backend::{GsBackendBase, GsDevice, GsDiffBackend};
use super::loss::{combined_loss, SsimConfig};
use super::optimizer::{AdamScaled, AdamScaledConfig};
use super::splats::DeviceSplats;
use super::{render_bwd, topology_apply, topology_bridge};

#[derive(Debug, Clone, Default)]
pub struct WgpuTrainingReport {
    pub losses: Vec<f32>,
    pub num_splats: Vec<usize>,
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
    pub fn new(config: TrainingConfig, device: GsDevice, initial_splats: usize, sh_coeffs: usize) -> Self {
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
        let opacity_scales =
            Tensor::<GsBackendBase, 1>::from_floats([config.lr_opacity], &device);

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

        let pred = render_bwd::render_splats(splats, camera, (width as u32, height as u32), background).await;
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

        self.accumulate_gradients(&transforms_grad, &sh_grad);
        self.optimizer
            .step_device_splats(splats, transforms_grad, sh_grad, opacity_grad);

        if self.should_apply_topology(iteration) {
            self.apply_topology_mutations(splats, iteration, frame_count).await;
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
        let mut report = WgpuTrainingReport::default();
        let mut rng = StdRng::seed_from_u64(self.config.frame_shuffle_seed);

        for iteration in 0..num_iterations {
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

            if iteration % 100 == 0 {
                log::info!(
                    "WGPU training step {} | loss={:.6} | splats={}",
                    iteration + 1,
                    loss,
                    splats.num_splats()
                );
            }
        }

        report
    }

    fn should_apply_topology(&self, iteration: usize) -> bool {
        let next_iter = iteration.max(1);
        (self.config.densify_interval > 0 && next_iter % self.config.densify_interval == 0)
            || (self.config.prune_interval > 0 && next_iter % self.config.prune_interval == 0)
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
        self.reset_accumulators(splats.num_splats(), splats.sh_coeffs.val().dims()[1]);
    }

    fn reset_accumulators(&mut self, num_splats: usize, sh_coeffs: usize) {
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
    }
}
