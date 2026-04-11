use super::eval::scaled_dimensions;
use super::splats;
use super::TrainingConfig;

#[cfg(feature = "gpu")]
pub(crate) mod data_loading;
#[cfg(feature = "gpu")]
pub(crate) mod frame_loader;
#[cfg(feature = "gpu")]
pub(crate) mod frame_targets;
#[cfg(feature = "gpu")]
pub(crate) mod init_map;
