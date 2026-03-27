use glam::Vec3;

use crate::{TrainingDataset, TrainingError};

use super::TrainingConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBoundsSource {
    SparsePoints,
    CameraTrajectory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkDisposition {
    Trainable,
    SkippedInsufficientCameras,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedChunk {
    pub chunk_id: usize,
    pub core_bounds: ChunkBounds,
    pub overlap_bounds: ChunkBounds,
    pub pose_indices: Vec<usize>,
    pub point_indices: Vec<usize>,
    pub disposition: ChunkDisposition,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPlan {
    pub bounds_source: ChunkBoundsSource,
    pub scene_bounds: ChunkBounds,
    pub chunk_axis: usize,
    pub requested_chunks: usize,
    pub chunks: Vec<PlannedChunk>,
}

#[derive(Debug, Clone)]
pub struct MaterializedChunkDataset {
    pub chunk_id: usize,
    pub dataset: TrainingDataset,
    pub used_frame_based_initialization: bool,
}

impl ChunkBounds {
    fn from_points(points: impl Iterator<Item = [f32; 3]>) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut seen = false;

        for point in points {
            let point = Vec3::from_array(point);
            min = min.min(point);
            max = max.max(point);
            seen = true;
        }

        if !seen {
            return None;
        }

        Some(Self {
            min: min.to_array(),
            max: max.to_array(),
        })
    }

    fn contains(&self, point: [f32; 3]) -> bool {
        (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] <= self.max[axis])
    }

    fn size(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]).max(0.0),
            (self.max[1] - self.min[1]).max(0.0),
            (self.max[2] - self.min[2]).max(0.0),
        ]
    }

    fn diagonal_length(&self) -> f32 {
        let size = self.size();
        (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt()
    }

    fn longest_axis(&self) -> usize {
        let size = self.size();
        if size[1] > size[0] && size[1] >= size[2] {
            1
        } else if size[2] > size[0] && size[2] > size[1] {
            2
        } else {
            0
        }
    }

    fn expand_relative(&self, overlap_ratio: f32, clamp_to: &Self) -> Self {
        let mut min = self.min;
        let mut max = self.max;
        let size = self.size();
        let clamp_size = clamp_to.size();

        for axis in 0..3 {
            let base_extent = size[axis].max(clamp_size[axis] * 0.05).max(1e-3);
            let expand = base_extent * overlap_ratio.max(0.0);
            min[axis] = (min[axis] - expand).max(clamp_to.min[axis]);
            max[axis] = (max[axis] + expand).min(clamp_to.max[axis]);
        }

        Self { min, max }
    }
}

impl ChunkPlan {
    pub fn trainable_chunks(&self) -> impl Iterator<Item = &PlannedChunk> {
        self.chunks
            .iter()
            .filter(|chunk| chunk.disposition == ChunkDisposition::Trainable)
    }
}

pub fn plan_spatial_chunks(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    affordable_initial_gaussians: Option<usize>,
) -> Result<ChunkPlan, TrainingError> {
    let (bounds_source, scene_bounds) = derive_scene_bounds(dataset)?;
    let requested_chunks = target_chunk_count(dataset, config, affordable_initial_gaussians);
    let chunk_axis = scene_bounds.longest_axis();
    let scene_extent = scene_bounds.size()[chunk_axis];
    let mut chunks = Vec::with_capacity(requested_chunks);

    for chunk_id in 0..requested_chunks {
        let mut core_bounds = scene_bounds;
        let start_t = chunk_id as f32 / requested_chunks as f32;
        let end_t = (chunk_id + 1) as f32 / requested_chunks as f32;
        core_bounds.min[chunk_axis] = scene_bounds.min[chunk_axis] + scene_extent * start_t;
        core_bounds.max[chunk_axis] = scene_bounds.min[chunk_axis] + scene_extent * end_t;
        let overlap_bounds = core_bounds.expand_relative(config.chunk_overlap_ratio, &scene_bounds);
        let point_indices = collect_chunk_point_indices(dataset, &overlap_bounds);
        let pose_indices =
            collect_chunk_pose_indices(dataset, &overlap_bounds, &point_indices, &scene_bounds);
        let disposition = if pose_indices.len() >= config.min_cameras_per_chunk.max(1) {
            ChunkDisposition::Trainable
        } else {
            ChunkDisposition::SkippedInsufficientCameras
        };

        chunks.push(PlannedChunk {
            chunk_id,
            core_bounds,
            overlap_bounds,
            pose_indices,
            point_indices,
            disposition,
        });
    }

    Ok(ChunkPlan {
        bounds_source,
        scene_bounds,
        chunk_axis,
        requested_chunks,
        chunks,
    })
}

pub fn materialize_chunk_dataset(
    dataset: &TrainingDataset,
    chunk: &PlannedChunk,
) -> Result<MaterializedChunkDataset, TrainingError> {
    if chunk.pose_indices.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "chunk {} has no assigned poses and cannot be materialized",
            chunk.chunk_id
        )));
    }

    let mut chunk_dataset =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
    chunk_dataset.poses = chunk
        .pose_indices
        .iter()
        .map(|&idx| dataset.poses[idx].clone())
        .collect();
    chunk_dataset.initial_points = chunk
        .point_indices
        .iter()
        .map(|&idx| dataset.initial_points[idx])
        .collect();

    Ok(MaterializedChunkDataset {
        chunk_id: chunk.chunk_id,
        used_frame_based_initialization: chunk_dataset.initial_points.is_empty(),
        dataset: chunk_dataset,
    })
}

fn derive_scene_bounds(
    dataset: &TrainingDataset,
) -> Result<(ChunkBoundsSource, ChunkBounds), TrainingError> {
    if let Some(bounds) =
        ChunkBounds::from_points(dataset.initial_points.iter().map(|(pos, _)| *pos))
    {
        return Ok((ChunkBoundsSource::SparsePoints, bounds));
    }

    if let Some(bounds) =
        ChunkBounds::from_points(dataset.poses.iter().map(|pose| pose.pose.translation()))
    {
        return Ok((ChunkBoundsSource::CameraTrajectory, bounds));
    }

    Err(TrainingError::InvalidInput(
        "chunk planning requires either initial_points or camera poses".to_string(),
    ))
}

fn target_chunk_count(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    affordable_initial_gaussians: Option<usize>,
) -> usize {
    let requested_scale = if dataset.initial_points.is_empty() {
        config.max_initial_gaussians.max(1)
    } else {
        dataset.initial_points.len().max(1)
    };
    let affordable_scale = affordable_initial_gaussians
        .map(|estimate| estimate.max(1))
        .unwrap_or(requested_scale);
    let mut chunk_count = requested_scale.div_ceil(affordable_scale.max(1)).max(1);
    if config.max_chunks > 0 {
        chunk_count = chunk_count.min(config.max_chunks);
    }
    chunk_count.max(1)
}

fn collect_chunk_point_indices(
    dataset: &TrainingDataset,
    overlap_bounds: &ChunkBounds,
) -> Vec<usize> {
    dataset
        .initial_points
        .iter()
        .enumerate()
        .filter_map(|(idx, (position, _))| overlap_bounds.contains(*position).then_some(idx))
        .collect()
}

fn collect_chunk_pose_indices(
    dataset: &TrainingDataset,
    overlap_bounds: &ChunkBounds,
    point_indices: &[usize],
    scene_bounds: &ChunkBounds,
) -> Vec<usize> {
    let scene_size = scene_bounds.size();
    let scene_diagonal = scene_bounds.diagonal_length().max(1e-3);
    let fallback_radius = overlap_bounds.diagonal_length().mul_add(0.5, 0.0).max(1e-3);
    let fallback_radius_sq = fallback_radius * fallback_radius;

    dataset
        .poses
        .iter()
        .enumerate()
        .filter_map(|(idx, pose)| {
            let center = pose.pose.translation();
            if center_intersects_overlap(center, overlap_bounds, &scene_size, scene_diagonal) {
                return Some(idx);
            }

            let center = Vec3::from_array(center);
            point_indices
                .iter()
                .copied()
                .any(|point_idx| {
                    let point = Vec3::from_array(dataset.initial_points[point_idx].0);
                    center.distance_squared(point) <= fallback_radius_sq
                })
                .then_some(idx)
        })
        .collect()
}

fn center_intersects_overlap(
    center: [f32; 3],
    overlap_bounds: &ChunkBounds,
    scene_size: &[f32; 3],
    scene_diagonal: f32,
) -> bool {
    (0..3).all(|axis| {
        let tolerance = if scene_size[axis] <= 1e-6 {
            scene_diagonal * 0.25
        } else {
            0.0
        };
        center[axis] >= overlap_bounds.min[axis] - tolerance
            && center[axis] <= overlap_bounds.max[axis] + tolerance
    })
}

#[cfg(test)]
mod tests {
    use super::{
        materialize_chunk_dataset, plan_spatial_chunks, ChunkBoundsSource, ChunkDisposition,
    };
    use crate::{Intrinsics, ScenePose, TrainingConfig, SE3};
    use rustscan_types::TrainingDataset;
    use std::path::PathBuf;

    fn pose(frame_id: u64, position: [f32; 3]) -> ScenePose {
        ScenePose::new(
            frame_id,
            PathBuf::from(format!("frame_{frame_id:04}.png")),
            SE3::new(&[0.0, 0.0, 0.0, 1.0], &position),
            frame_id as f64,
        )
    }

    fn dataset_with_points() -> TrainingDataset {
        let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 640, 480));
        dataset.add_pose(pose(0, [0.0, 0.0, 0.0]));
        dataset.add_pose(pose(1, [4.0, 0.0, 0.0]));
        dataset.add_pose(pose(2, [8.0, 0.0, 0.0]));
        dataset.add_point([0.0, 0.0, 1.0], None);
        dataset.add_point([1.0, 0.0, 1.0], None);
        dataset.add_point([7.5, 0.0, 1.0], None);
        dataset.add_point([8.0, 0.0, 1.0], None);
        dataset
    }

    fn two_chunk_capacity() -> usize {
        2
    }

    #[test]
    fn chunk_generation_prefers_sparse_point_bounds() {
        let dataset = dataset_with_points();
        let plan = plan_spatial_chunks(
            &dataset,
            &TrainingConfig::default(),
            Some(two_chunk_capacity()),
        )
        .unwrap();
        assert_eq!(plan.bounds_source, ChunkBoundsSource::SparsePoints);
        assert_eq!(plan.scene_bounds.min, [0.0, 0.0, 1.0]);
        assert_eq!(plan.scene_bounds.max, [8.0, 0.0, 1.0]);
    }

    #[test]
    fn chunk_generation_falls_back_to_camera_trajectory_bounds() {
        let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 640, 480));
        dataset.add_pose(pose(0, [1.0, 2.0, 3.0]));
        dataset.add_pose(pose(1, [5.0, 4.0, 3.0]));
        let plan = plan_spatial_chunks(&dataset, &TrainingConfig::default(), None).unwrap();
        assert_eq!(plan.bounds_source, ChunkBoundsSource::CameraTrajectory);
        assert_eq!(plan.scene_bounds.min, [1.0, 2.0, 3.0]);
        assert_eq!(plan.scene_bounds.max, [5.0, 4.0, 3.0]);
    }

    #[test]
    fn chunk_count_and_overlap_are_deterministic() {
        let config = TrainingConfig {
            chunk_overlap_ratio: 0.25,
            ..TrainingConfig::default()
        };
        let plan = plan_spatial_chunks(&dataset_with_points(), &config, Some(two_chunk_capacity()))
            .unwrap();
        assert_eq!(plan.requested_chunks, 2);
        assert_eq!(plan.chunk_axis, 0);
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].core_bounds.min[0], 0.0);
        assert_eq!(plan.chunks[0].core_bounds.max[0], 4.0);
        assert_eq!(plan.chunks[0].overlap_bounds.max[0], 5.0);
        assert_eq!(plan.chunks[1].core_bounds.min[0], 4.0);
        assert_eq!(plan.chunks[1].core_bounds.max[0], 8.0);
        assert_eq!(plan.chunks[1].overlap_bounds.min[0], 3.0);
    }

    #[test]
    fn camera_assignment_uses_overlap_boundary_and_point_proximity() {
        let mut dataset = dataset_with_points();
        dataset.add_pose(pose(3, [5.2, 0.0, 0.0]));
        let config = TrainingConfig {
            chunk_overlap_ratio: 0.25,
            ..TrainingConfig::default()
        };
        let plan = plan_spatial_chunks(&dataset, &config, Some(two_chunk_capacity())).unwrap();

        assert_eq!(plan.chunks[0].pose_indices, vec![0, 1]);
        assert_eq!(plan.chunks[1].pose_indices, vec![1, 2, 3]);
    }

    #[test]
    fn weak_chunks_are_skipped_deterministically() {
        let config = TrainingConfig {
            min_cameras_per_chunk: 3,
            chunk_overlap_ratio: 0.0,
            ..TrainingConfig::default()
        };
        let plan = plan_spatial_chunks(&dataset_with_points(), &config, Some(two_chunk_capacity()))
            .unwrap();

        assert_eq!(
            plan.chunks
                .iter()
                .map(|chunk| chunk.disposition)
                .collect::<Vec<_>>(),
            vec![
                ChunkDisposition::SkippedInsufficientCameras,
                ChunkDisposition::SkippedInsufficientCameras,
            ]
        );
    }

    #[test]
    fn materialization_keeps_only_local_poses_and_points() {
        let plan = plan_spatial_chunks(
            &dataset_with_points(),
            &TrainingConfig::default(),
            Some(two_chunk_capacity()),
        )
        .unwrap();
        let chunk_dataset =
            materialize_chunk_dataset(&dataset_with_points(), &plan.chunks[0]).unwrap();

        assert_eq!(chunk_dataset.chunk_id, 0);
        assert_eq!(chunk_dataset.dataset.poses.len(), 2);
        assert_eq!(chunk_dataset.dataset.initial_points.len(), 2);
        assert!(!chunk_dataset.used_frame_based_initialization);
    }

    #[test]
    fn materialization_falls_back_to_frame_initialization_when_local_points_absent() {
        let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 640, 480));
        dataset.add_pose(pose(0, [0.0, 0.0, 0.0]));
        dataset.add_pose(pose(1, [1.0, 0.0, 0.0]));
        let chunk = super::PlannedChunk {
            chunk_id: 0,
            core_bounds: super::ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 0.0, 0.0],
            },
            overlap_bounds: super::ChunkBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 0.0, 0.0],
            },
            pose_indices: vec![0, 1],
            point_indices: vec![],
            disposition: ChunkDisposition::Trainable,
        };
        let chunk_dataset = materialize_chunk_dataset(&dataset, &chunk).unwrap();

        assert_eq!(chunk_dataset.dataset.poses.len(), 2);
        assert!(chunk_dataset.dataset.initial_points.is_empty());
        assert!(chunk_dataset.used_frame_based_initialization);
    }
}
