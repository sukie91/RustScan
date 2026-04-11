use super::*;

impl MetalTrainer {
    fn total_loss_for_render_result(
        &self,
        gaussians: &Splats,
        rendered: &RenderedFrame,
        projected: &ProjectedGaussians,
        frame: &MetalTrainingFrame,
    ) -> candle_core::Result<f32> {
        evaluate_training_step_loss(
            gaussians,
            rendered,
            projected,
            &frame.target_color,
            &frame.target_depth,
            &frame.target_color_cpu,
            &frame.target_depth_cpu,
            self.loss_config(),
        )
        .map(|context| context.total_loss)
    }

    pub(super) fn loss_for_camera(
        &mut self,
        gaussians: &Splats,
        frame: &MetalTrainingFrame,
        camera: &DiffCamera,
    ) -> candle_core::Result<f32> {
        let collect_visible_indices = self.is_litegs_mode() && self.litegs.reg_weight > 0.0;
        let cluster_visible_mask = self.cluster_visible_mask_for_camera(gaussians.len(), camera);
        let (rendered, projected, _) = self.render(
            gaussians,
            camera,
            false,
            collect_visible_indices,
            cluster_visible_mask.as_deref(),
        )?;
        self.total_loss_for_render_result(gaussians, &rendered, &projected, frame)
    }

    pub(crate) fn training_step(
        &mut self,
        gaussians: &mut Splats,
        frame: &MetalTrainingFrame,
        frame_idx: usize,
        frame_count: usize,
        should_profile: bool,
    ) -> candle_core::Result<MetalStepOutcome> {
        self.iteration += 1;
        self.active_sh_degree = self.litegs_active_sh_degree_for_iteration(frame_count);
        self.last_learning_rates = self.current_learning_rates();
        let total_start = Instant::now();
        let collect_visible_indices = topology::should_collect_visible_indices(
            &self.topology_policy(),
            topology::schedule_topology(
                &self.topology_policy(),
                TopologyStepContext {
                    iteration: self.iteration,
                    frame_count,
                },
            ),
        );
        let frame_pose_embedding =
            pose_embedding::cloned_frame_pose_embedding(self.pose_embeddings.as_ref(), frame_idx);

        if self.litegs.cluster_size > 0 {
            self.sync_cluster_assignment(gaussians, false)?;
        }

        let render_camera = pose_embedding::resolve_render_camera(
            frame_pose_embedding.as_ref(),
            &frame.camera,
            &self.device,
        )?;

        let cluster_visible_mask =
            self.cluster_visible_mask_for_camera(gaussians.len(), &render_camera);

        let (rendered, projected, render_profile) = self.render(
            gaussians,
            &render_camera,
            should_profile,
            collect_visible_indices,
            cluster_visible_mask.as_deref(),
        )?;
        let mut profile = MetalStepProfile::from_render(render_profile);

        let loss_start = Instant::now();
        let step_loss = evaluate_training_step_loss(
            gaussians,
            &rendered,
            &projected,
            &frame.target_color,
            &frame.target_depth,
            &frame.target_color_cpu,
            &frame.target_depth_cpu,
            self.loss_config(),
        )?;
        self.last_loss_terms = step_loss.telemetry.loss_terms.clone();
        self.last_depth_valid_pixels = step_loss.telemetry.depth_valid_pixels;
        self.last_depth_grad_scale = step_loss.telemetry.depth_grad_scale;
        self.synchronize_if_needed(should_profile)?;
        profile.loss = loss_start.elapsed();

        let backward_start = Instant::now();
        let refresh_target_buffers = self.cached_target_frame_idx != Some(frame_idx);
        let backward = metal_backward::execute_backward_pass(
            &mut self.runtime,
            MetalBackwardRequest {
                tile_bins: &projected.tile_bins,
                n_gaussians: gaussians.len(),
                camera: &render_camera,
                target_color_cpu: &frame.target_color_cpu,
                target_depth_cpu: &frame.target_depth_cpu,
                ssim_grads: &step_loss.ssim_grads,
                loss_scales: step_loss.backward_loss_scales,
                refresh_target_buffers,
            },
        )?;
        if refresh_target_buffers {
            self.cached_target_frame_idx = Some(frame_idx);
        }
        profile.backward = backward_start.elapsed();

        if debug_training_step_probe_enabled()
            && (self.iteration <= 6 || self.iteration % self.litegs.refine_every.max(1) == 0)
        {
            let param_grad_stats = abs_stats(&backward.grad_magnitudes);
            let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
            log::info!(
                "Backward stats step {} | visible={} | param_grad_max={:.6e} | param_grad_mean={:.6e} | param_grad_nonzero={} | projected_grad_max={:.6e} | projected_grad_mean={:.6e} | projected_grad_nonzero={}",
                self.iteration,
                projected.visible_count,
                param_grad_stats.0,
                param_grad_stats.1,
                param_grad_stats.2,
                projected_grad_stats.0,
                projected_grad_stats.1,
                projected_grad_stats.2,
            );
        }

        let debug_probe = debug_training_step_probe_enabled() && self.iteration <= 2;
        let debug_param_snapshots = if debug_probe {
            Some((
                gaussians.positions().flatten_all()?.to_vec1::<f32>()?,
                gaussians
                    .scales
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                gaussians
                    .opacities
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                gaussians.colors().flatten_all()?.to_vec1::<f32>()?,
            ))
        } else {
            None
        };
        if debug_probe {
            let position_stats = tensor_abs_stats(&backward.grads.positions)?;
            let scale_stats = tensor_abs_stats(&backward.grads.log_scales)?;
            let opacity_stats = tensor_abs_stats(&backward.grads.opacity_logits)?;
            let color_stats = tensor_abs_stats(&backward.grads.colors)?;
            let param_grad_stats = abs_stats(&backward.grad_magnitudes);
            let projected_grad_stats = abs_stats(&backward.projected_grad_magnitudes);
            log::info!(
                "Metal debug backward step {} | frame={} | grad_positions(max={:.6e} mean={:.6e} nz={}) | grad_scales(max={:.6e} mean={:.6e} nz={}) | grad_opacity(max={:.6e} mean={:.6e} nz={}) | grad_colors(max={:.6e} mean={:.6e} nz={}) | param_grad_mag(max={:.6e} mean={:.6e} nz={}) | projected_grad_mag(max={:.6e} mean={:.6e} nz={})",
                self.iteration,
                frame_idx,
                position_stats.0,
                position_stats.1,
                position_stats.2,
                scale_stats.0,
                scale_stats.1,
                scale_stats.2,
                opacity_stats.0,
                opacity_stats.1,
                opacity_stats.2,
                color_stats.0,
                color_stats.1,
                color_stats.2,
                param_grad_stats.0,
                param_grad_stats.1,
                param_grad_stats.2,
                projected_grad_stats.0,
                projected_grad_stats.1,
                projected_grad_stats.2,
            );
        }

        let optimizer_start = Instant::now();
        let effective_lr_pos = self.compute_lr_pos();
        let scale_reg_grad = optional_full_scale_regularization_grad(
            gaussians,
            &projected,
            self.is_litegs_mode(),
            self.litegs.reg_weight,
        )?;
        let parameter_grads = metal_backward::assemble_parameter_grads(
            &self.device,
            MetalParameterGradInputs {
                gaussians,
                raw_grads: &backward.grads,
                projected: &projected,
                rendered: &rendered,
                rendered_color_cpu: &step_loss.rendered_color_cpu,
                target_color_cpu: &frame.target_color_cpu,
                target_depth_cpu: &frame.target_depth_cpu,
                ssim_grads: &step_loss.ssim_grads,
                loss_scales: step_loss.backward_loss_scales,
                camera: &render_camera,
                active_sh_degree: self.active_sh_degree,
                render_width: self.render_width,
                render_height: self.render_height,
                include_rotation_grads: self.lr_rotation > 0.0 && projected.visible_count > 0,
            },
        )?;
        let pose_grad_device = self.device.clone();
        let pose_parameter_grads = pose_embedding::optional_pose_parameter_grads_fd(
            frame_pose_embedding.as_ref(),
            &frame.camera,
            |camera| self.loss_for_camera(gaussians, frame, camera),
            &pose_grad_device,
        )?;
        self.apply_parameter_grads(
            gaussians,
            &parameter_grads,
            &projected,
            effective_lr_pos,
            scale_reg_grad.as_ref(),
        )?;
        if let Some((before_positions, before_scales, before_opacities, before_colors)) =
            debug_param_snapshots.as_ref()
        {
            let position_delta = max_abs_delta(before_positions, gaussians.positions())?;
            let scale_delta = max_abs_delta(before_scales, gaussians.scales.as_tensor())?;
            let opacity_delta = max_abs_delta(before_opacities, gaussians.opacities.as_tensor())?;
            let color_delta = max_abs_delta(before_colors, &gaussians.colors())?;
            log::info!(
                "Metal debug optimizer step {} | frame={} | delta_positions={:.6e} | delta_scales={:.6e} | delta_opacity={:.6e} | delta_colors={:.6e}",
                self.iteration,
                frame_idx,
                position_delta,
                scale_delta,
                opacity_delta,
                color_delta,
            );
        }
        pose_embedding::apply_optional_pose_update(
            self.pose_embeddings.as_mut(),
            frame_idx,
            pose_parameter_grads,
        )?;

        self.update_gaussian_stats(
            &backward.grad_magnitudes,
            &backward.projected_grad_magnitudes,
            &projected,
            gaussians.len(),
        )?;
        self.synchronize_if_needed(should_profile)?;
        profile.optimizer = optimizer_start.elapsed();
        profile.total = total_start.elapsed();
        self.last_step_duration = Some(profile.total);
        self.loss_history.push(step_loss.total_loss);

        Ok(MetalStepOutcome {
            loss: step_loss.total_loss,
            visible_gaussians: profile.visible_gaussians,
            total_gaussians: profile.total_gaussians,
            profile: if should_profile { Some(profile) } else { None },
        })
    }

    /// Compute the current effective position learning rate using exponential decay.
    ///
    /// eta(t) = eta0 * (eta_end / eta0)^(t / T)
    pub(super) fn compute_lr_pos(&self) -> f32 {
        let lr0 = self.lr_pos;
        let lr_end = self.lr_pos_final;
        let t = self.iteration as f32;
        let total = self.max_iterations as f32;
        if total <= 0.0 || lr0 <= 0.0 || lr_end <= 0.0 {
            return lr0;
        }
        lr0 * (lr_end / lr0).powf(t / total)
    }

    pub(super) fn render_colors_for_camera(
        &self,
        gaussians: &Splats,
        positions: &Tensor,
        camera: &DiffCamera,
    ) -> candle_core::Result<Tensor> {
        metal_forward::render_colors_for_camera(
            gaussians,
            positions,
            camera,
            &self.device,
            self.active_sh_degree,
        )
    }

    #[cfg(test)]
    pub(super) fn rotation_parameter_grads(
        &self,
        gaussians: &Splats,
        projected: &ProjectedGaussians,
        rendered: &RenderedFrame,
        rendered_color_cpu: &[f32],
        target_color_cpu: &[f32],
        target_depth_cpu: &[f32],
        ssim_grads: &[f32],
        loss_scales: MetalBackwardLossScales,
        camera: &DiffCamera,
    ) -> candle_core::Result<Tensor> {
        metal_backward::rotation_parameter_grads(
            &self.device,
            gaussians,
            projected,
            rendered,
            rendered_color_cpu,
            target_color_cpu,
            target_depth_cpu,
            ssim_grads,
            loss_scales,
            camera,
            self.render_width,
            self.render_height,
        )
    }

    #[cfg(test)]
    pub(super) fn parameter_grads_from_render_color_grads(
        &self,
        gaussians: &Splats,
        projected: &ProjectedGaussians,
        render_color_grads: &Tensor,
        camera: &DiffCamera,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        metal_backward::parameter_grads_from_render_color_grads(
            &self.device,
            self.active_sh_degree,
            gaussians,
            projected,
            render_color_grads,
            camera,
        )
    }

    fn apply_parameter_grads(
        &mut self,
        gaussians: &mut Splats,
        parameter_grads: &MetalParameterGrads,
        projected: &ProjectedGaussians,
        effective_lr_pos: f32,
        scale_reg_grad: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let optimizer_config = MetalOptimizerConfig {
            effective_lr_pos,
            lr_scale: self.lr_scale,
            lr_rotation: self.lr_rotation,
            lr_opacity: self.lr_opacity,
            lr_color: self.lr_color,
            lr_sh_rest: self.lr_sh_rest,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            step: self.iteration,
            use_sparse_updates: self.is_litegs_mode() && self.litegs.sparse_grad,
        };
        let adam = self
            .adam
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("adam state not initialized".into()))?;
        let scale_grads = if let Some(extra) = scale_reg_grad {
            parameter_grads.log_scales.broadcast_add(extra)?
        } else {
            parameter_grads.log_scales.clone()
        };
        let parameter_grads = MetalParameterGrads {
            positions: parameter_grads.positions.clone(),
            log_scales: scale_grads,
            rotations: parameter_grads.rotations.clone(),
            opacity_logits: parameter_grads.opacity_logits.clone(),
            colors: parameter_grads.colors.clone(),
            sh_rest: parameter_grads.sh_rest.clone(),
        };
        metal_optimizer::apply_optimizer_step(
            gaussians,
            &mut self.runtime,
            adam,
            &parameter_grads,
            Some(&projected.source_indices),
            optimizer_config,
        )
    }

    #[cfg(test)]
    pub(super) fn apply_backward_grads(
        &mut self,
        gaussians: &mut Splats,
        grads: &MetalBackwardGrads,
        projected: &ProjectedGaussians,
        camera: &DiffCamera,
        effective_lr_pos: f32,
        scale_reg_grad: Option<&Tensor>,
        rotation_parameter_grads: Option<&Tensor>,
    ) -> candle_core::Result<()> {
        let (_, color_parameter_grads, sh_rest_parameter_grads) = self
            .parameter_grads_from_render_color_grads(gaussians, projected, &grads.colors, camera)?;
        let parameter_grads = MetalParameterGrads {
            positions: grads.positions.clone(),
            log_scales: grads.log_scales.clone(),
            rotations: rotation_parameter_grads.cloned(),
            opacity_logits: grads.opacity_logits.clone(),
            colors: color_parameter_grads,
            sh_rest: sh_rest_parameter_grads,
        };
        self.apply_parameter_grads(
            gaussians,
            &parameter_grads,
            projected,
            effective_lr_pos,
            scale_reg_grad,
        )
    }

    fn forward_settings(&self) -> MetalForwardSettings {
        MetalForwardSettings {
            pixel_count: self.pixel_count,
            render_width: self.render_width,
            render_height: self.render_height,
            chunk_size: self.chunk_size,
            use_native_forward: self.use_native_forward,
            litegs_mode: self.is_litegs_mode(),
        }
    }

    pub(super) fn render(
        &mut self,
        gaussians: &Splats,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(RenderedFrame, ProjectedGaussians, MetalRenderProfile)> {
        let positions = gaussians.positions().detach();
        let render_colors = self.render_colors_for_camera(gaussians, &positions, camera)?;

        if should_profile && self.device.is_metal() && gaussians.len() > 0 {
            let gaussian_bindings = self.runtime.bind_gaussians(gaussians, &render_colors)?;
            let _ = (
                gaussian_bindings.positions.byte_offset(),
                gaussian_bindings.positions.element_count(),
                gaussian_bindings.positions.dtype(),
                gaussian_bindings.positions.buffer()?,
                gaussian_bindings.scales.byte_offset(),
                gaussian_bindings.rotations.byte_offset(),
                gaussian_bindings.opacities.byte_offset(),
                gaussian_bindings.colors.byte_offset(),
            );
        }

        let settings = self.forward_settings();
        metal_forward::execute_forward_pass_on_runtime(
            &mut self.runtime,
            &self.device,
            settings,
            MetalForwardInputs {
                gaussians,
                positions: &positions,
                colors: &render_colors,
                camera,
                should_profile,
                collect_visible_indices,
                cluster_visible_mask,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn project_gaussians(
        &mut self,
        gaussians: &Splats,
        camera: &DiffCamera,
        should_profile: bool,
        collect_visible_indices: bool,
        cluster_visible_mask: Option<&[bool]>,
    ) -> candle_core::Result<(ProjectedGaussians, MetalRenderProfile)> {
        let positions = gaussians.positions().detach();
        let render_colors = self.render_colors_for_camera(gaussians, &positions, camera)?;
        let settings = self.forward_settings();
        metal_forward::project_gaussians_on_runtime(
            &mut self.runtime,
            &self.device,
            settings,
            MetalForwardInputs {
                gaussians,
                positions: &positions,
                colors: &render_colors,
                camera,
                should_profile,
                collect_visible_indices,
                cluster_visible_mask,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn rasterize(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &ProjectedTileBins,
    ) -> candle_core::Result<(RenderedFrame, TileBinningStats)> {
        metal_forward::rasterize(
            &mut self.runtime,
            &self.device,
            self.pixel_count,
            self.chunk_size,
            projected,
            tile_bins,
        )
    }

    #[cfg(test)]
    pub(super) fn build_tile_bins(
        &mut self,
        projected: &ProjectedGaussians,
    ) -> candle_core::Result<ProjectedTileBins> {
        metal_forward::build_tile_bins(&mut self.runtime, &self.device, projected)
    }

    #[cfg(test)]
    pub(super) fn profile_native_forward(
        &mut self,
        projected: &ProjectedGaussians,
        tile_bins: &ProjectedTileBins,
        baseline: &RenderedFrame,
    ) -> candle_core::Result<NativeParityProfile> {
        metal_forward::profile_native_forward(
            &mut self.runtime,
            self.render_width,
            self.render_height,
            projected,
            tile_bins,
            baseline,
        )
    }

    fn synchronize_if_needed(&self, should_profile: bool) -> candle_core::Result<()> {
        if should_profile {
            self.device.synchronize()?;
        }
        Ok(())
    }
}
