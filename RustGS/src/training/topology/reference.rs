use super::density_controller::DensityController;
use super::splat_metrics::TopologySplatMetrics;
use super::{MetalGaussianStats, TopologyPolicy};
use crate::training::LiteGsPruneMode;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DensityControllerReferenceSummary {
    pub(super) prune_mask: Vec<bool>,
    pub(super) clone_mask: Vec<bool>,
    pub(super) split_mask: Vec<bool>,
    pub(super) densify_budget: Option<usize>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DensityControllerReferenceSummary {
    pub(super) fn prune_candidates(&self) -> usize {
        self.prune_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }

    pub(super) fn clone_candidates(&self) -> usize {
        self.clone_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }

    pub(super) fn split_candidates(&self) -> usize {
        self.split_mask
            .iter()
            .filter(|candidate| **candidate)
            .count()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub(super) struct DensityControllerReferenceAdapter {
    controller: DensityController,
}

#[cfg_attr(not(test), allow(dead_code))]
impl DensityControllerReferenceAdapter {
    pub(super) fn from_topology_state(
        policy: &TopologyPolicy,
        splats: &TopologySplatMetrics,
        stats: &[MetalGaussianStats],
    ) -> Self {
        let mut controller = DensityController::new(
            policy.density_controller_reference_config(splats.len()),
            density_controller_taming_mode(policy),
        );
        controller.stats.resize(splats.len());

        for idx in 0..splats.len() {
            let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
            let mean2d_grad_count = gaussian_stats.mean2d_grad.count as f32;
            let fragment_weight_count = gaussian_stats.fragment_weight.count as f32;
            let fragment_err_count = gaussian_stats.fragment_err.count as f32;

            controller.stats.visible_count[idx] =
                gaussian_stats.visible_count.min(u32::MAX as usize) as u32;
            controller.stats.mean2d_grad[idx] = (
                gaussian_stats.mean2d_grad.mean * mean2d_grad_count,
                mean2d_grad_count,
            );
            controller.stats.fragment_weight[idx] = (
                gaussian_stats.fragment_weight.mean * fragment_weight_count,
                fragment_weight_count,
            );
            controller.stats.fragment_err[idx] = (
                gaussian_stats.fragment_err.mean * fragment_err_count,
                gaussian_stats.fragment_err.m2
                    + fragment_err_count
                        * gaussian_stats.fragment_err.mean
                        * gaussian_stats.fragment_err.mean,
                fragment_err_count,
            );
            controller.stats.opacity[idx] = splats.opacity(idx);
            controller.stats.max_scale[idx] = splats.max_scale(idx);
        }

        Self { controller }
    }

    pub(super) fn summary(
        &self,
        completed_epoch: Option<usize>,
    ) -> DensityControllerReferenceSummary {
        let prune_mask = self.controller.get_prune_mask();
        let clone_mask = self.controller.get_clone_mask();
        let split_mask = self.controller.get_split_mask();
        let prune_candidates = prune_mask.iter().filter(|candidate| **candidate).count();
        let densify_budget = completed_epoch
            .filter(|epoch| self.controller.is_densify_active(*epoch))
            .map(|epoch| {
                self.controller.compute_densify_budget(
                    self.controller.stats.len(),
                    prune_candidates,
                    epoch,
                )
            });

        DensityControllerReferenceSummary {
            prune_mask,
            clone_mask,
            split_mask,
            densify_budget,
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn density_controller_reference_summary(
    policy: &TopologyPolicy,
    splats: &TopologySplatMetrics,
    stats: &[MetalGaussianStats],
    completed_epoch: Option<usize>,
) -> DensityControllerReferenceSummary {
    DensityControllerReferenceAdapter::from_topology_state(policy, splats, stats)
        .summary(completed_epoch)
}

#[cfg_attr(not(test), allow(dead_code))]
fn density_controller_taming_mode(policy: &TopologyPolicy) -> bool {
    matches!(policy.litegs.prune_mode, LiteGsPruneMode::Weight)
}
