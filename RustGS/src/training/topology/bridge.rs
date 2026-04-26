use burn::prelude::*;
use burn::tensor::Transaction;

use crate::core::HostSplats;
use crate::training::engine::{device_splats_to_host, DeviceSplats};

use super::{plan_topology_from_host_snapshot, TopologyMutationPlan};

pub(crate) struct TopologySnapshot {
    pub splats: HostSplats,
    pub grad_2d_accum: Vec<f32>,
    pub grad_color_accum: Vec<f32>,
    pub num_observations: Vec<f32>,
    pub visible_observations: Vec<f32>,
    pub splat_ages: Vec<usize>,
    pub invisible_windows: Vec<usize>,
}

pub(crate) async fn snapshot_for_topology<B: Backend>(
    splats: &DeviceSplats<B>,
    grad_2d_accum: &Tensor<B, 1>,
    grad_color_accum: &Tensor<B, 1>,
    num_observations: &Tensor<B, 1>,
    visible_observations: &Tensor<B, 1>,
) -> TopologySnapshot {
    let splats_host = device_splats_to_host(splats).await;
    let data = Transaction::default()
        .register(grad_2d_accum.clone())
        .register(grad_color_accum.clone())
        .register(num_observations.clone())
        .register(visible_observations.clone())
        .execute_async()
        .await
        .expect("topology accumulator readback");

    let grad_2d_accum = data[0]
        .clone()
        .into_vec::<f32>()
        .expect("topology grad_2d readback");
    let grad_color_accum = data[1]
        .clone()
        .into_vec::<f32>()
        .expect("topology grad_color readback");
    let num_observations = data[2]
        .clone()
        .into_vec::<f32>()
        .expect("topology num_observations: expected f32 scalar tensor")
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let visible_observations = data[3]
        .clone()
        .into_vec::<f32>()
        .expect("topology visible_observations: expected f32 scalar tensor")
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    TopologySnapshot {
        splat_ages: vec![0; splats_host.len()],
        invisible_windows: vec![0; splats_host.len()],
        splats: splats_host,
        grad_2d_accum,
        grad_color_accum,
        num_observations,
        visible_observations,
    }
}

pub(crate) fn plan_mutations(
    snapshot: &TopologySnapshot,
    config: &crate::training::TrainingConfig,
    iteration: usize,
    frame_count: usize,
) -> TopologyMutationPlan {
    plan_topology_from_host_snapshot(
        config,
        &snapshot.splats,
        &snapshot.grad_2d_accum,
        &snapshot.grad_color_accum,
        &snapshot.num_observations,
        &snapshot.visible_observations,
        &snapshot.splat_ages,
        &snapshot.invisible_windows,
        iteration,
        frame_count,
    )
}
