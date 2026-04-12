use burn::prelude::*;
use burn::tensor::{Int, Transaction};

use crate::training::topology::{plan_topology_from_host_snapshot, TopologyMutationPlan};

use super::splats::{device_splats_to_host, DeviceSplats};

pub struct TopologySnapshot {
    pub splats: crate::training::HostSplats,
    pub grad_2d_accum: Vec<f32>,
    pub grad_color_accum: Vec<f32>,
    pub num_observations: Vec<u32>,
}

pub async fn snapshot_for_topology<B: Backend>(
    splats: &DeviceSplats<B>,
    grad_2d_accum: &Tensor<B, 1>,
    grad_color_accum: &Tensor<B, 1>,
    num_observations: &Tensor<B, 1, Int>,
) -> TopologySnapshot {
    let splats_host = device_splats_to_host(splats).await;
    let data = Transaction::default()
        .register(grad_2d_accum.clone())
        .register(grad_color_accum.clone())
        .register(num_observations.clone())
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
    let num_observations = if let Ok(values) = data[2].clone().into_vec::<u32>() {
        values
    } else {
        data[2]
            .clone()
            .into_vec::<i32>()
            .expect("topology num_observations readback")
            .into_iter()
            .map(|value| value.max(0) as u32)
            .collect()
    };

    TopologySnapshot {
        splats: splats_host,
        grad_2d_accum,
        grad_color_accum,
        num_observations,
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
        iteration,
        frame_count,
    )
}
