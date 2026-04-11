use super::*;

impl MetalTrainer {
    pub(super) fn is_litegs_mode(&self) -> bool {
        self.training_profile == TrainingProfile::LiteGsMacV1
    }

    pub(super) fn current_learning_rates(&self) -> LiteGsOptimizerLrs {
        LiteGsOptimizerLrs {
            xyz: Some(self.compute_lr_pos()),
            sh_0: Some(self.lr_color),
            sh_rest: Some(self.lr_sh_rest),
            opacity: Some(self.lr_opacity),
            scale: Some(self.lr_scale),
            rot: Some(self.lr_rotation),
        }
    }

    pub(super) fn record_loss_curve_sample(
        &mut self,
        iter: usize,
        frame_idx: usize,
        max_iterations: usize,
    ) {
        if !should_record_loss_curve_sample(iter, max_iterations) {
            return;
        }
        if self
            .loss_curve_samples
            .last()
            .map(|sample| sample.iteration == iter)
            .unwrap_or(false)
        {
            return;
        }

        self.loss_curve_samples.push(ParityLossCurveSample {
            iteration: iter,
            frame_idx,
            l1: self.last_loss_terms.l1,
            ssim: self.last_loss_terms.ssim,
            depth: self.last_loss_terms.depth,
            total: self.last_loss_terms.total,
            depth_valid_pixels: self.last_depth_valid_pixels,
        });
    }

    pub(super) fn current_telemetry(&self, frame_count: usize) -> LiteGsTrainingTelemetry {
        let final_metrics = if self.loss_history.is_empty() {
            None
        } else {
            Some(summarize_training_metrics(&self.loss_history, frame_count))
        };
        LiteGsTrainingTelemetry {
            loss_terms: self.last_loss_terms.clone(),
            loss_curve_samples: self.loss_curve_samples.clone(),
            topology: self.topology_metrics.clone(),
            active_sh_degree: Some(self.active_sh_degree),
            final_loss: final_metrics.map(|metrics| metrics.final_loss),
            final_step_loss: final_metrics.map(|metrics| metrics.final_step_loss),
            depth_valid_pixels: self.last_depth_valid_pixels,
            depth_grad_scale: self.last_depth_grad_scale,
            rotation_frozen: self.rotation_frozen,
            learning_rates: self.last_learning_rates.clone(),
        }
    }

    pub(super) fn loss_weights(&self) -> (f32, f32, f32) {
        let color_weight = if self.is_litegs_mode() {
            1.0 - LITEGS_LAMBDA_DSSIM
        } else {
            0.8
        };
        let ssim_weight = if self.is_litegs_mode() {
            LITEGS_LAMBDA_DSSIM
        } else {
            0.2
        };
        let depth_weight = if self.is_litegs_mode() {
            if self.litegs.enable_depth {
                LITEGS_DEPTH_LOSS_WEIGHT
            } else {
                0.0
            }
        } else {
            LITEGS_DEPTH_LOSS_WEIGHT
        };
        (color_weight, ssim_weight, depth_weight)
    }

    pub(super) fn loss_config(&self) -> MetalLossConfig {
        let (color_weight, ssim_weight, depth_weight) = self.loss_weights();
        MetalLossConfig {
            color_weight,
            ssim_weight,
            depth_weight,
            scale_regularization_weight: if self.is_litegs_mode() {
                self.litegs.reg_weight
            } else {
                0.0
            },
            enable_transmittance_loss: self.is_litegs_mode() && self.litegs.enable_transmittance,
            render_width: self.render_width,
            render_height: self.render_height,
        }
    }
}
