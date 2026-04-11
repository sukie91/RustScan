use super::*;

pub(crate) struct MetalTrainingFrame {
    pub(crate) camera: DiffCamera,
    pub(crate) target_color: Tensor,
    pub(crate) target_depth: Tensor,
    pub(crate) target_color_cpu: Vec<f32>,
    pub(crate) target_depth_cpu: Vec<f32>,
}

impl MetalTrainer {
    pub(crate) fn prepare_frame(
        &self,
        loaded: &LoadedTrainingData,
        idx: usize,
    ) -> Result<MetalTrainingFrame, TrainingError> {
        let src_camera = &loaded.cameras[idx];
        let target_color_cpu = if loaded.target_width == self.render_width
            && loaded.target_height == self.render_height
        {
            loaded.colors[idx].clone()
        } else {
            resize_rgb(
                &loaded.colors[idx],
                loaded.target_width,
                loaded.target_height,
                self.render_width,
                self.render_height,
            )
        };
        let target_depth_cpu = if loaded.target_width == self.render_width
            && loaded.target_height == self.render_height
        {
            loaded.depths[idx].clone()
        } else {
            resize_depth(
                &loaded.depths[idx],
                loaded.target_width,
                loaded.target_height,
                self.render_width,
                self.render_height,
            )
        };
        let scaled_camera = scale_camera(
            src_camera,
            self.render_width,
            self.render_height,
            &self.device,
        )?;
        Ok(MetalTrainingFrame {
            camera: scaled_camera,
            target_color: Tensor::from_slice(
                &target_color_cpu,
                (self.pixel_count, 3),
                &self.device,
            )?,
            target_depth: Tensor::from_slice(&target_depth_cpu, (self.pixel_count,), &self.device)?,
            target_color_cpu,
            target_depth_cpu,
        })
    }

    pub(crate) fn prepare_frames(
        &self,
        loaded: &LoadedTrainingData,
    ) -> Result<Vec<MetalTrainingFrame>, TrainingError> {
        let mut frames = Vec::with_capacity(loaded.cameras.len());
        for idx in 0..loaded.cameras.len() {
            frames.push(self.prepare_frame(loaded, idx)?);
        }
        Ok(frames)
    }

    pub(crate) fn initialize_training_session(
        &mut self,
        gaussians: &mut Splats,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        if frame_count == 0 {
            candle_core::bail!("metal backend received zero training frames");
        }
        self.loss_curve_samples.clear();
        self.adam = Some(MetalAdamState::new(gaussians)?);
        self.reset_gaussian_stats(gaussians.len());
        self.topology_metrics.initialization_gaussians = Some(gaussians.len());
        self.topology_metrics.final_gaussians = Some(gaussians.len());
        self.refresh_litegs_topology_window_metrics(frame_count);
        self.runtime.reserve_core_buffers(gaussians.len())?;
        self.runtime.prime_tile_index_buffer()?;
        Ok(())
    }

    pub(crate) fn train(
        &mut self,
        gaussians: &mut Splats,
        frames: &[MetalTrainingFrame],
        max_iterations: usize,
    ) -> candle_core::Result<MetalTrainingStats> {
        self.initialize_training_session(gaussians, frames.len())?;
        let runtime_stats = self.runtime.stats();

        log::info!(
            "MetalTrainer running at {}x{} | gaussian_batch_size={} | native_forward={} | topology(densify={} prune={} warmup={} log={}) | frames={} | initial_gaussians={} | tiles={} | runtime_buffers={} | pipeline_warmups={} | tile_index_capacity={}B",
            self.render_width,
            self.render_height,
            self.batch_size,
            self.use_native_forward,
            self.densify_interval,
            self.prune_interval,
            self.topology_warmup,
            self.topology_log_interval,
            frames.len(),
            gaussians.len(),
            runtime_stats.tile_windows,
            runtime_stats.buffer_allocations,
            runtime_stats.pipeline_compilations,
            self.runtime.tile_index_capacity_bytes(),
        );

        let train_start = Instant::now();
        for iter in 0..max_iterations {
            let frame_idx = iter % frames.len();
            let should_log = iter < 5 || iter % 25 == 0;
            let should_profile =
                should_profile_iteration(self.profile_steps, self.profile_interval, iter);
            let step_start = Instant::now();
            let outcome = self.training_step(
                gaussians,
                &frames[frame_idx],
                frame_idx,
                frames.len(),
                should_profile,
            )?;
            self.record_loss_curve_sample(iter, frame_idx, max_iterations);
            self.maybe_apply_topology_updates(gaussians, frame_idx, frames.len())?;
            if should_log {
                log::info!(
                    "Metal iter {:5}/{:5} | frame {:3}/{:3} | visible {:5}/{:5} | loss {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    frames.len(),
                    outcome.visible_gaussians,
                    outcome.total_gaussians,
                    outcome.loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }
            if let Some(profile) = outcome.profile {
                profile.log(iter, max_iterations);
            }
        }
        self.topology_metrics.final_gaussians = Some(gaussians.len());

        let final_metrics = summarize_training_metrics(&self.loss_history, frames.len());
        Ok(MetalTrainingStats {
            final_loss: final_metrics.final_loss,
            final_step_loss: final_metrics.final_step_loss,
            telemetry: self.current_telemetry(frames.len()),
        })
    }

    pub(crate) fn train_loaded(
        &mut self,
        gaussians: &mut Splats,
        loaded: &LoadedTrainingData,
        max_iterations: usize,
    ) -> Result<MetalTrainingStats, TrainingError> {
        let frame_count = loaded.cameras.len();
        self.initialize_training_session(gaussians, frame_count)?;
        let runtime_stats = self.runtime.stats();

        log::info!(
            "MetalTrainer running at {}x{} | gaussian_batch_size={} | native_forward={} | topology(densify={} prune={} warmup={} log={}) | frames={} | initial_gaussians={} | tiles={} | runtime_buffers={} | pipeline_warmups={} | tile_index_capacity={}B",
            self.render_width,
            self.render_height,
            self.batch_size,
            self.use_native_forward,
            self.densify_interval,
            self.prune_interval,
            self.topology_warmup,
            self.topology_log_interval,
            frame_count,
            gaussians.len(),
            runtime_stats.tile_windows,
            runtime_stats.buffer_allocations,
            runtime_stats.pipeline_compilations,
            self.runtime.tile_index_capacity_bytes(),
        );

        let train_start = Instant::now();
        for iter in 0..max_iterations {
            let frame_idx = iter % frame_count;
            let should_log = iter < 5 || iter % 25 == 0;
            let should_profile =
                should_profile_iteration(self.profile_steps, self.profile_interval, iter);
            let frame = self.prepare_frame(loaded, frame_idx)?;
            let step_start = Instant::now();
            let outcome =
                self.training_step(gaussians, &frame, frame_idx, frame_count, should_profile)?;
            self.record_loss_curve_sample(iter, frame_idx, max_iterations);
            self.maybe_apply_topology_updates(gaussians, frame_idx, frame_count)?;
            if should_log {
                log::info!(
                    "Metal iter {:5}/{:5} | frame {:3}/{:3} | visible {:5}/{:5} | loss {:.6} | step_time={:.2}s | elapsed={:.2}s",
                    iter,
                    max_iterations,
                    frame_idx + 1,
                    frame_count,
                    outcome.visible_gaussians,
                    outcome.total_gaussians,
                    outcome.loss,
                    step_start.elapsed().as_secs_f64(),
                    train_start.elapsed().as_secs_f64()
                );
            }
            if let Some(profile) = outcome.profile {
                profile.log(iter, max_iterations);
            }
        }
        self.topology_metrics.final_gaussians = Some(gaussians.len());

        let final_metrics = summarize_training_metrics(&self.loss_history, frame_count);
        Ok(MetalTrainingStats {
            final_loss: final_metrics.final_loss,
            final_step_loss: final_metrics.final_step_loss,
            telemetry: self.current_telemetry(frame_count),
        })
    }
}
