mod backend;
mod loss;
mod optimizer;
mod runtime;
mod splats;
mod trainer;

pub(crate) use backend::{GsBackendBase, GsDiffBackend};
pub(crate) use runtime::train_splats_with_controlled_events;
pub(crate) use splats::{device_splats_to_host, host_splats_to_device, DeviceSplats};
