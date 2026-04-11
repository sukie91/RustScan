use super::data::data_loading;
use super::metal;
use super::splats;
use super::telemetry;
use super::{
    materialize_chunk_dataset, plan_spatial_chunks, ChunkBounds, ChunkDisposition, ChunkPlan,
    LiteGsConfig, MaterializedChunkDataset, PlannedChunk, TrainingConfig, TrainingProfile,
    MIN_RENDER_SCALE,
};

#[cfg(feature = "gpu")]
pub(crate) mod benchmark;
#[cfg(feature = "gpu")]
pub(crate) mod chunk_training;
#[cfg(feature = "gpu")]
pub(crate) mod events;
#[cfg(feature = "gpu")]
mod execution_plan;
#[cfg(feature = "gpu")]
mod export;
#[cfg(feature = "gpu")]
pub(crate) mod orchestrator;
