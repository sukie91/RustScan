use super::super::LiteGsOpacityResetMode;
use super::*;

impl MetalTrainer {
    pub(super) fn topology_policy(&self) -> TopologyPolicy {
        TopologyPolicy {
            training_profile: self.training_profile,
            litegs: self.litegs.clone(),
            prune_threshold: self.prune_threshold,
            densify_interval: self.densify_interval,
            prune_interval: self.prune_interval,
            topology_warmup: self.topology_warmup,
            topology_log_interval: self.topology_log_interval,
            legacy_densify_grad_threshold: self.legacy_densify_grad_threshold,
            legacy_clone_scale_threshold: self.legacy_clone_scale_threshold,
            legacy_split_scale_threshold: self.legacy_split_scale_threshold,
            legacy_prune_scale_threshold: self.legacy_prune_scale_threshold,
            legacy_max_densify_per_update: self.legacy_max_densify_per_update,
            max_gaussian_budget: self.max_gaussian_budget,
            scene_extent: self.scene_extent,
            max_iterations: self.max_iterations,
        }
    }

    pub(super) fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        self.topology_policy().litegs_total_epochs(frame_count)
    }

    pub(super) fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        self.topology_policy()
            .litegs_densify_until_epoch(frame_count)
    }

    pub(super) fn litegs_effective_densify_from_epoch(&self, frame_count: usize) -> usize {
        self.topology_policy()
            .litegs_effective_densify_from_epoch(frame_count)
    }

    pub(super) fn litegs_late_stage_start_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        let densify_until = self.litegs_densify_until_epoch(frame_count);
        let derived = total_epochs.saturating_mul(2) / 3;
        derived
            .max(self.litegs_effective_densify_from_epoch(frame_count))
            .min(densify_until.saturating_sub(1))
    }

    pub(super) fn refresh_litegs_topology_window_metrics(&mut self, frame_count: usize) {
        if !self.is_litegs_mode() {
            return;
        }

        self.topology_metrics.total_epochs = Some(self.litegs_total_epochs(frame_count));
        self.topology_metrics.densify_until_epoch =
            Some(self.litegs_densify_until_epoch(frame_count));
        self.topology_metrics.late_stage_start_epoch =
            Some(self.litegs_late_stage_start_epoch(frame_count));
        self.topology_metrics.topology_freeze_epoch = self.litegs.topology_freeze_after_epoch;
    }

    fn is_litegs_late_stage_epoch(&self, epoch: usize, frame_count: usize) -> bool {
        self.is_litegs_mode() && epoch >= self.litegs_late_stage_start_epoch(frame_count)
    }

    pub(super) fn litegs_active_sh_degree_for_iteration(&self, frame_count: usize) -> usize {
        if !self.is_litegs_mode() || self.litegs.sh_degree == 0 {
            return self.litegs.sh_degree;
        }
        let _ = frame_count;
        self.litegs.sh_degree
    }

    pub(super) fn reset_gaussian_stats(&mut self, gaussian_count: usize) {
        self.gaussian_stats = vec![MetalGaussianStats::default(); gaussian_count];
    }

    fn reset_litegs_refine_window_stats(&mut self) {
        if !self.is_litegs_mode() {
            return;
        }

        for stats in &mut self.gaussian_stats {
            stats.mean2d_grad = RunningMoments::default();
            stats.fragment_weight = RunningMoments::default();
            stats.fragment_err = RunningMoments::default();
            stats.refine_weight_max = 0.0;
            stats.visible_count = 0;
        }
    }

    pub(super) fn update_gaussian_stats(
        &mut self,
        param_grad_magnitudes: &[f32],
        refine_weights: &[f32],
        projected: &ProjectedGaussians,
        gaussian_count: usize,
    ) -> candle_core::Result<()> {
        if self.gaussian_stats.len() != gaussian_count {
            self.gaussian_stats
                .resize(gaussian_count, MetalGaussianStats::default());
        }

        for stats in &mut self.gaussian_stats {
            stats.age = stats.age.saturating_add(1);
            stats.consecutive_invisible_epochs =
                stats.consecutive_invisible_epochs.saturating_add(1);
        }

        if !self.is_litegs_mode() {
            for idx in 0..gaussian_count.min(param_grad_magnitudes.len()) {
                let grad_mag = param_grad_magnitudes[idx] * self.pixel_count.max(1) as f32;
                let stats = &mut self.gaussian_stats[idx];
                stats.mean2d_grad.update(grad_mag.min(10.0));
            }
            for source_idx in projected.visible_source_indices().iter().copied() {
                if let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) {
                    stats.visible_count = stats.visible_count.saturating_add(1);
                    stats.consecutive_invisible_epochs = 0;
                }
            }
            return Ok(());
        }

        let sigma_x = projected.sigma_x.to_vec1::<f32>()?;
        let sigma_y = projected.sigma_y.to_vec1::<f32>()?;
        let opacity = projected.opacity.to_vec1::<f32>()?;

        for (visible_idx, source_idx) in projected
            .visible_source_indices()
            .iter()
            .copied()
            .enumerate()
        {
            let Some(stats) = self.gaussian_stats.get_mut(source_idx as usize) else {
                continue;
            };
            stats.consecutive_invisible_epochs = 0;

            let refine_weight = refine_weights
                .get(source_idx as usize)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let sigma_x = sigma_x
                .get(visible_idx)
                .copied()
                .unwrap_or(0.0)
                .abs()
                .max(1e-4);
            let sigma_y = sigma_y
                .get(visible_idx)
                .copied()
                .unwrap_or(0.0)
                .abs()
                .max(1e-4);
            let opacity = opacity.get(visible_idx).copied().unwrap_or(0.0).max(0.0);
            let fragment_weight = opacity * sigma_x * sigma_y;
            let fragment_err = refine_weight * fragment_weight;

            stats.mean2d_grad.update(refine_weight);
            stats.fragment_weight.update(fragment_weight);
            stats.fragment_err.update(fragment_err);
            stats.refine_weight_max = stats.refine_weight_max.max(refine_weight);
            stats.visible_count = stats.visible_count.saturating_add(1);
        }

        if debug_training_step_probe_enabled()
            && (self.iteration <= 6 || self.iteration % self.litegs.refine_every.max(1) == 0)
        {
            let mut visible_raw_max = 0.0f32;
            let mut visible_nonzero = 0usize;
            for source_idx in projected.visible_source_indices().iter().copied() {
                let raw = refine_weights
                    .get(source_idx as usize)
                    .copied()
                    .unwrap_or_default()
                    .max(0.0);
                if raw > 0.0 {
                    visible_nonzero += 1;
                    visible_raw_max = visible_raw_max.max(raw);
                }
            }
            let (all_raw_max, all_raw_mean, all_raw_nonzero) = abs_stats(refine_weights);
            log::info!(
                "Growth stats step {} | visible={} | visible_nonzero={} | visible_refine_max={:.6e} | all_refine_max={:.6e} | all_refine_mean={:.6e} | all_refine_nonzero={}",
                self.iteration,
                projected.visible_source_indices().len(),
                visible_nonzero,
                visible_raw_max,
                all_raw_max,
                all_raw_mean,
                all_raw_nonzero,
            );
        }

        Ok(())
    }

    fn reset_opacity_state(&mut self, only_opacity: bool) -> candle_core::Result<()> {
        let Some(adam) = self.adam.as_mut() else {
            return Ok(());
        };
        adam.reset_moments(only_opacity)
    }

    fn apply_litegs_opacity_reset(&mut self, gaussians: &mut Splats) -> candle_core::Result<()> {
        let opacities = gaussians.opacities()?;
        let new_opacity_values = match self.litegs.opacity_reset_mode {
            LiteGsOpacityResetMode::Decay => inverse_sigmoid_tensor(
                &opacities
                    .affine(LITEGS_OPACITY_DECAY_RATE as f64, 0.0)?
                    .clamp(LITEGS_OPACITY_DECAY_MIN, 1.0 - 1e-6)?,
            )?,
            LiteGsOpacityResetMode::Reset => {
                inverse_sigmoid_tensor(&opacities.clamp(1e-6, LITEGS_OPACITY_THRESHOLD)?)?
            }
        };
        gaussians.opacities.set(&new_opacity_values)?;
        self.reset_opacity_state(matches!(
            self.litegs.opacity_reset_mode,
            LiteGsOpacityResetMode::Reset
        ))?;
        Ok(())
    }

    pub(super) fn apply_litegs_refine_decay(
        &self,
        gaussians: &mut Splats,
    ) -> candle_core::Result<()> {
        if !self.is_litegs_mode() || self.max_iterations == 0 {
            return Ok(());
        }

        let train_t = (self.iteration as f32 / self.max_iterations as f32).clamp(0.0, 1.0);
        let shrink_strength = 1.0 - train_t;
        if shrink_strength <= 0.0 {
            return Ok(());
        }

        let opacity_sub = LITEGS_REFINE_OPACITY_DECAY * shrink_strength;
        if opacity_sub > 0.0 {
            let opacities = gaussians.opacities()?;
            let decayed_opacity = opacities
                .affine(1.0, -(opacity_sub as f64))?
                .clamp(1e-6, 1.0 - 1e-6)?;
            gaussians
                .opacities
                .set(&inverse_sigmoid_tensor(&decayed_opacity)?)?;
        }

        let scale_scaling = (1.0 - LITEGS_REFINE_SCALE_DECAY * shrink_strength).max(1e-6);
        if scale_scaling < 1.0 {
            let log_scale_delta = scale_scaling.ln() as f64;
            let decayed_log_scales = gaussians.scales.as_tensor().affine(1.0, log_scale_delta)?;
            gaussians.scales.set(&decayed_log_scales)?;
        }

        Ok(())
    }

    fn max_topology_gaussians(
        &self,
        requested_cap: usize,
        current_len: usize,
        frame_count: usize,
    ) -> usize {
        let min_cap = current_len.max(1);
        let requested_cap = requested_cap.max(min_cap);
        let Some(memory_budget) = self.topology_memory_budget else {
            return requested_cap;
        };
        if assess_memory_estimate(
            &estimate_peak_memory_with_source_pixels(
                requested_cap,
                self.pixel_count,
                self.source_pixel_count,
                frame_count,
                self.batch_size,
            ),
            &memory_budget,
        ) != MetalMemoryDecision::Block
        {
            return requested_cap;
        }

        let mut low = min_cap;
        let mut high = requested_cap;
        while low < high {
            let mid = low + (high - low + 1) / 2;
            let decision = assess_memory_estimate(
                &estimate_peak_memory_with_source_pixels(
                    mid,
                    self.pixel_count,
                    self.source_pixel_count,
                    frame_count,
                    self.batch_size,
                ),
                &memory_budget,
            );
            if decision == MetalMemoryDecision::Block {
                high = mid - 1;
            } else {
                low = mid;
            }
        }
        low
    }

    pub(super) fn rebuild_adam_state(
        &self,
        old_state: &MetalAdamState,
        plan: &topology::TopologyMutationPlan,
    ) -> candle_core::Result<MetalAdamState> {
        metal_optimizer::rebuild_adam_state_with_plan(&self.device, old_state, plan)
    }

    fn apply_topology_mutation_aftermath(
        &mut self,
        gaussians: &mut Splats,
        rebuilt: Option<Splats>,
        stats: Vec<MetalGaussianStats>,
        plan: Option<&topology::TopologyMutationPlan>,
        aftermath: topology::TopologyMutationAftermath,
    ) -> candle_core::Result<()> {
        if aftermath.requires_runtime_rebuild {
            let rebuilt = rebuilt.ok_or_else(|| {
                candle_core::Error::Msg("topology aftermath expected rebuilt runtime splats".into())
            })?;
            let new_adam = match self.adam.take() {
                Some(old_state) if aftermath.requires_adam_rebuild => self.rebuild_adam_state(
                    &old_state,
                    plan.ok_or_else(|| {
                        candle_core::Error::Msg(
                            "topology aftermath expected mutation plan for adam rebuild".into(),
                        )
                    })?,
                )?,
                _ => MetalAdamState::new(&rebuilt)?,
            };

            *gaussians = rebuilt;
            self.adam = Some(new_adam);
        }

        match aftermath.gaussian_stats_action {
            TopologyStatsAction::KeepCurrent => {}
            TopologyStatsAction::UseMutated => {
                self.gaussian_stats = stats;
            }
            TopologyStatsAction::ResetAll => {
                self.reset_gaussian_stats(aftermath.metrics_delta.final_gaussians);
            }
        }

        if aftermath.requires_cluster_resync {
            self.sync_cluster_assignment(gaussians, true)?;
        }
        if aftermath.requires_runtime_reserve {
            self.runtime.reserve_core_buffers(gaussians.len())?;
        }
        if aftermath.apply_opacity_reset {
            self.apply_litegs_opacity_reset(gaussians)?;
        }
        if aftermath.reset_refine_window_stats {
            self.reset_litegs_refine_window_stats();
        }
        topology::apply_topology_metrics_delta(&mut self.topology_metrics, aftermath.metrics_delta);

        Ok(())
    }

    fn clustering_positions(&self, gaussians: &Splats) -> candle_core::Result<Vec<[f32; 3]>> {
        gaussians
            .positions()
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| {
                if row.len() != 3 {
                    candle_core::bail!("expected gaussian positions with row width 3");
                }
                Ok([row[0], row[1], row[2]])
            })
            .collect()
    }

    pub(super) fn sync_cluster_assignment(
        &mut self,
        gaussians: &Splats,
        topology_changed: bool,
    ) -> candle_core::Result<()> {
        if self.litegs.cluster_size == 0 {
            self.cluster_assignment = None;
            return Ok(());
        }

        let positions = self.clustering_positions(gaussians)?;
        if positions.is_empty() {
            self.cluster_assignment = None;
            return Ok(());
        }

        match self.cluster_assignment.as_mut() {
            Some(assignment)
                if !topology_changed && assignment.cluster_indices.len() == positions.len() =>
            {
                assignment.update_aabbs(&positions);
            }
            Some(assignment) => {
                assignment.reassign(&positions, self.litegs.cluster_size, self.scene_extent);
            }
            None => {
                self.cluster_assignment = Some(ClusterAssignment::assign_spatial_hash(
                    &positions,
                    self.litegs.cluster_size,
                    self.scene_extent,
                ));
            }
        }

        Ok(())
    }

    pub(super) fn cluster_visible_mask_for_camera(
        &self,
        gaussians_len: usize,
        camera: &DiffCamera,
    ) -> Option<Vec<bool>> {
        let assignment = self.cluster_assignment.as_ref()?;
        let view_proj = camera.view_projection_mat4();
        let visible_clusters = assignment.get_visible_clusters(&view_proj);

        let mut mask = vec![false; gaussians_len];
        for &cluster in &visible_clusters {
            for (i, &c) in assignment.cluster_indices.iter().enumerate() {
                if c == cluster {
                    mask[i] = true;
                }
            }
        }

        let visible_count = mask.iter().filter(|&v| *v).count();
        if visible_count < gaussians_len {
            log::debug!(
                "Cluster culling: {} / {} Gaussians visible ({} / {} clusters)",
                visible_count,
                gaussians_len,
                visible_clusters.len(),
                assignment.num_clusters
            );
        }

        Some(mask)
    }

    pub(super) fn maybe_apply_topology_updates(
        &mut self,
        gaussians: &mut Splats,
        _frame_idx: usize,
        frame_count: usize,
    ) -> candle_core::Result<()> {
        if gaussians.len() == 0 {
            return Ok(());
        }

        self.refresh_litegs_topology_window_metrics(frame_count);
        let policy = self.topology_policy();
        let schedule = topology::schedule_topology(
            &policy,
            TopologyStepContext {
                iteration: self.iteration,
                frame_count,
            },
        );
        let should_log_topology = policy.should_log_topology(self.iteration);

        let completed_epoch = schedule.completed_epoch;
        let mut should_reset_opacity = schedule.reset_opacity;
        let allow_extra_growth = schedule.allow_extra_growth;
        let (mut should_densify, mut should_prune) = (schedule.densify, schedule.prune);
        if (!should_densify && !should_prune && !should_reset_opacity) || gaussians.len() == 0 {
            return Ok(());
        }

        let topology_start = Instant::now();
        let old_len = gaussians.len();
        let mut current_stats = self.gaussian_stats.clone();
        if current_stats.len() != old_len {
            current_stats.resize(old_len, MetalGaussianStats::default());
        }
        let topology_splats = TopologySplatMetrics::from_runtime(gaussians)?;
        let analysis =
            topology::analyze_topology_candidates(&policy, &topology_splats, &current_stats);
        let density_reference = if self.is_litegs_mode() {
            Some(topology::density_controller_reference_summary(
                &policy,
                &topology_splats,
                &current_stats,
                completed_epoch,
            ))
        } else {
            None
        };
        let reference_clone_candidates = density_reference
            .as_ref()
            .map(|summary| summary.clone_candidates())
            .unwrap_or(0);
        let reference_split_candidates = density_reference
            .as_ref()
            .map(|summary| summary.split_candidates())
            .unwrap_or(0);
        let reference_prune_candidates = density_reference
            .as_ref()
            .map(|summary| summary.prune_candidates())
            .unwrap_or(0);
        let reference_densify_budget = density_reference
            .as_ref()
            .and_then(|summary| summary.densify_budget)
            .unwrap_or(0);
        let litegs_requested_additions = if self.is_litegs_mode() && should_densify {
            topology::litegs_requested_additions(
                &analysis.infos,
                self.litegs.growth_select_fraction,
                allow_extra_growth,
            )
        } else {
            0
        };
        let requested_cap =
            topology::requested_gaussian_cap(&policy, old_len, litegs_requested_additions);
        let max_gaussians = self.max_topology_gaussians(requested_cap, old_len, frame_count);
        let litegs_selection = if self.is_litegs_mode() && should_densify {
            topology::litegs_select_densify_candidates(
                &analysis.infos,
                max_gaussians.saturating_sub(old_len),
                self.litegs.growth_select_fraction,
                allow_extra_growth,
            )
        } else {
            topology::LiteGsDensifySelection::default()
        };
        let execution =
            topology::plan_topology_execution(&policy, schedule, &analysis, &litegs_selection);
        should_reset_opacity = execution.should_reset_opacity;
        should_densify = execution.should_densify;
        should_prune = execution.should_prune;

        if execution.disposition == TopologyExecutionDisposition::SkipDestructiveLiteGs
            && should_log_topology
        {
            log::info!(
                "Metal topology check at iter {} skipped destructive LiteGS prune/reset because no replacement or growth sources were available | epoch={:?} | prune_candidates={} | growth_candidates={} | reference_clone_candidates={} | reference_split_candidates={} | reference_prune_candidates={} | reference_budget={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
                self.iteration,
                execution.completed_epoch,
                analysis.prune_candidates,
                analysis.growth_candidates,
                reference_clone_candidates,
                reference_split_candidates,
                reference_prune_candidates,
                reference_densify_budget,
                analysis.max_grad,
                analysis.mean_grad,
            );
        }

        if execution.disposition == TopologyExecutionDisposition::SkipNoEligibleCandidates {
            if should_log_topology {
                log::info!(
                    "Metal topology check at iter {} found no eligible candidates | densify={} | prune={} | reset_opacity={} | gaussians={} | budget_cap={} | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | reference_clone_candidates={} | reference_split_candidates={} | reference_prune_candidates={} | reference_budget={}",
                    self.iteration,
                    should_densify,
                    should_prune,
                    should_reset_opacity,
                    old_len,
                    max_gaussians,
                    analysis.max_grad,
                    analysis.mean_grad,
                    analysis.active_grad_stats,
                    analysis.small_scale_stats,
                    analysis.opacity_ready_stats,
                    analysis.clone_candidates,
                    analysis.split_candidates,
                    analysis.prune_candidates,
                    reference_clone_candidates,
                    reference_split_candidates,
                    reference_prune_candidates,
                    reference_densify_budget,
                );
            }
            if self.is_litegs_mode() {
                self.apply_litegs_refine_decay(gaussians)?;
                self.reset_litegs_refine_window_stats();
            }
            return Ok(());
        }

        let late_stage = completed_epoch
            .map(|epoch| self.is_litegs_late_stage_epoch(epoch, frame_count))
            .unwrap_or(false);

        let mutation = topology::plan_topology_mutation(
            &topology_splats,
            TopologyMutationRequest {
                policy: &policy,
                should_densify,
                should_prune,
                should_reset_opacity,
                completed_epoch,
                late_stage,
                max_gaussians,
                infos: &analysis.infos,
                litegs_selection: &litegs_selection,
            },
        );
        let added = mutation.added;
        let pruned = mutation.pruned;
        let morton_sorted = mutation.morton_sorted;
        let aftermath = mutation.aftermath;
        let stats = mutation.remap_stats(&current_stats);

        let topology_duration = topology_start.elapsed();
        let topology_ms = duration_ms(topology_duration);
        let topology_ratio = self
            .last_step_duration
            .map(|step| {
                let step_ms = duration_ms(step);
                if step_ms > 0.0 {
                    topology_ms / step_ms
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let guardrail_triggered = topology_ms >= 50.0 || topology_ratio >= 0.35;

        if added == 0 && pruned == 0 {
            self.apply_topology_mutation_aftermath(gaussians, None, stats, None, aftermath)?;
            self.apply_litegs_refine_decay(gaussians)?;
            if should_log_topology || guardrail_triggered {
                log::info!(
                    "Metal topology check at iter {} | epoch={:?} | late_stage={} | made no changes | densify={} | prune={} | reset_opacity={} | gaussians={} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | max_grad_accum={:.6} | mean_grad_accum={:.6} | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | reference_clone_candidates={} | reference_split_candidates={} | reference_prune_candidates={} | reference_budget={}",
                    self.iteration,
                    completed_epoch,
                    late_stage,
                    should_densify,
                    should_prune,
                    should_reset_opacity,
                    old_len,
                    max_gaussians,
                    topology_ms,
                    topology_ratio * 100.0,
                    analysis.max_grad,
                    analysis.mean_grad,
                    analysis.active_grad_stats,
                    analysis.small_scale_stats,
                    analysis.opacity_ready_stats,
                    analysis.clone_candidates,
                    analysis.split_candidates,
                    analysis.prune_candidates,
                    reference_clone_candidates,
                    reference_split_candidates,
                    reference_prune_candidates,
                    reference_densify_budget,
                );
            }
            if guardrail_triggered {
                log::warn!(
                    "Metal topology guardrail triggered at iter {} | topology={:.2}ms | previous_step={:.2}ms | share={:.0}% | consider increasing --topology-warmup, --densify-interval, or --prune-interval",
                    self.iteration,
                    topology_ms,
                    self.last_step_duration.map(duration_ms).unwrap_or(0.0),
                    topology_ratio * 100.0,
                );
            }
            if self.is_litegs_mode() {
                self.reset_litegs_refine_window_stats();
            }
            return Ok(());
        }

        let rebuilt = apply_topology_plan(gaussians, &mutation, &self.device)?;
        self.apply_topology_mutation_aftermath(
            gaussians,
            Some(rebuilt),
            stats,
            Some(&mutation),
            aftermath,
        )?;
        self.apply_litegs_refine_decay(gaussians)?;

        log::info!(
            "Metal topology update at iter {} | epoch={:?} | late_stage={} | densify={} | prune={} | reset_opacity={} | added {} | pruned {} | morton={} | gaussians {} -> {} | budget_cap={} | topology={:.2}ms | step_share={:.0}% | active_grad_stats={} | small_scale_stats={} | opacity_ready_stats={} | clone_candidates={} | split_candidates={} | prune_candidates={} | reference_clone_candidates={} | reference_split_candidates={} | reference_prune_candidates={} | reference_budget={} | max_grad_accum={:.6} | mean_grad_accum={:.6}",
            self.iteration,
            completed_epoch,
            late_stage,
            should_densify,
            should_prune,
            should_reset_opacity,
            added,
            pruned,
            morton_sorted,
            old_len,
            gaussians.len(),
            max_gaussians,
            topology_ms,
            topology_ratio * 100.0,
            analysis.active_grad_stats,
            analysis.small_scale_stats,
            analysis.opacity_ready_stats,
            analysis.clone_candidates,
            analysis.split_candidates,
            analysis.prune_candidates,
            reference_clone_candidates,
            reference_split_candidates,
            reference_prune_candidates,
            reference_densify_budget,
            analysis.max_grad,
            analysis.mean_grad,
        );
        if guardrail_triggered {
            log::warn!(
                "Metal topology guardrail triggered at iter {} | topology={:.2}ms | previous_step={:.2}ms | share={:.0}% | added={} | pruned={}",
                self.iteration,
                topology_ms,
                self.last_step_duration.map(duration_ms).unwrap_or(0.0),
                topology_ratio * 100.0,
                added,
                pruned,
            );
        }
        Ok(())
    }
}
