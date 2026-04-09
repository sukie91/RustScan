use std::mem::size_of;

use candle_core::Tensor;
use objc2_metal::MTLSize;

use crate::diff::diff_splat::Splats;

use super::metal_kernels::MetalKernel;
use super::metal_runtime::{
    MetalBufferSlot, MetalProjectionRecord, MetalRuntime, MetalTileBins, MetalTileDispatchRecord,
    ProjectedGpuBatch, METAL_TILE_SIZE,
};

pub(crate) fn project_gaussians(
    runtime: &mut MetalRuntime,
    gaussians: &Splats,
    render_colors: &Tensor,
    extract_visible_source_indices: bool,
) -> candle_core::Result<ProjectedGpuBatch> {
    let gaussian_count = gaussians.len();
    if gaussian_count == 0 {
        return Ok(ProjectedGpuBatch {
            visible_count: 0,
            visible_source_indices: Vec::new(),
        });
    }

    let padded_count = gaussian_count.next_power_of_two();
    runtime.ensure_buffer(
        MetalBufferSlot::ProjectionRecords,
        padded_count.saturating_mul(size_of::<MetalProjectionRecord>()),
    )?;
    runtime.ensure_buffer(MetalBufferSlot::VisibleCount, size_of::<u32>())?;
    let pipeline = runtime
        .ensure_pipeline(MetalKernel::ProjectGaussians)?
        .clone();
    let bindings = runtime.bind_gaussians(gaussians, render_colors)?;
    let metal = runtime.device.as_metal_device()?.clone();
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::ProjectGaussians.function_name());
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::CameraUniforms)?
            .map(|buffer| buffer.as_ref()),
        0,
    );
    encoder.set_buffer(
        1,
        Some(bindings.positions.buffer()?),
        bindings.positions.byte_offset(),
    );
    encoder.set_buffer(
        2,
        Some(bindings.scales.buffer()?),
        bindings.scales.byte_offset(),
    );
    encoder.set_buffer(
        3,
        Some(bindings.rotations.buffer()?),
        bindings.rotations.byte_offset(),
    );
    encoder.set_buffer(
        4,
        Some(bindings.opacities.buffer()?),
        bindings.opacities.byte_offset(),
    );
    encoder.set_buffer(
        5,
        Some(bindings.colors.buffer()?),
        bindings.colors.byte_offset(),
    );
    encoder.set_buffer(
        6,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .ok_or_else(|| candle_core::Error::Msg("missing projection records".into()))?
                .as_ref(),
        ),
        0,
    );
    let count = gaussian_count as u32;
    encoder.set_bytes(7, &count);
    let threads_per_group = pipeline
        .max_total_threads_per_threadgroup()
        .min(gaussian_count)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: gaussian_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    runtime.device.synchronize()?;

    if padded_count > gaussian_count {
        let pipeline = runtime
            .ensure_pipeline(MetalKernel::FillProjectionPadding)?
            .clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::FillProjectionPadding.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            runtime
                .buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        let start_idx = gaussian_count as u32;
        let total_count = padded_count as u32;
        encoder.set_bytes(1, &start_idx);
        encoder.set_bytes(2, &total_count);
        let padding_count = padded_count - gaussian_count;
        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(padding_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: padding_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        runtime.device.synchronize()?;
    }

    let sort_pipeline = runtime
        .ensure_pipeline(MetalKernel::BitonicSortProjections)?
        .clone();
    let metal = runtime.device.as_metal_device()?.clone();
    let total_count = padded_count as u32;
    let threads_per_group = sort_pipeline
        .max_total_threads_per_threadgroup()
        .min(padded_count)
        .max(1);
    let mut k = 2usize;
    while k <= padded_count {
        let mut j = k >> 1;
        while j > 0 {
            let encoder = metal.command_encoder()?;
            encoder.set_label(MetalKernel::BitonicSortProjections.function_name());
            encoder.set_compute_pipeline_state(&sort_pipeline);
            encoder.set_buffer(
                0,
                runtime
                    .buffer_handle(MetalBufferSlot::ProjectionRecords)?
                    .map(|buffer| buffer.as_ref()),
                0,
            );
            let stage_j = j as u32;
            let stage_k = k as u32;
            encoder.set_bytes(1, &stage_j);
            encoder.set_bytes(2, &stage_k);
            encoder.set_bytes(3, &total_count);
            encoder.dispatch_threads(
                MTLSize {
                    width: padded_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group,
                    height: 1,
                    depth: 1,
                },
            );
            drop(encoder);
            j >>= 1;
        }
        k <<= 1;
    }
    runtime.device.synchronize()?;

    runtime.dispatch_fill_u32(MetalBufferSlot::VisibleCount, 0, 1)?;
    let pipeline = runtime.ensure_pipeline(MetalKernel::CountVisible)?.clone();
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::CountVisible.function_name());
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::ProjectionRecords)?
            .map(|buffer| buffer.as_ref()),
        0,
    );
    encoder.set_buffer(
        1,
        runtime
            .buffer_handle(MetalBufferSlot::VisibleCount)?
            .map(|buffer| buffer.as_ref()),
        0,
    );
    let record_count = gaussian_count as u32;
    encoder.set_bytes(2, &record_count);
    let threads_per_group = pipeline
        .max_total_threads_per_threadgroup()
        .min(gaussian_count)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: gaussian_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    runtime.device.synchronize()?;

    let visible_count = runtime.read_buffer_structs::<u32>(MetalBufferSlot::VisibleCount, 1)?;
    let visible_count = visible_count.first().copied().unwrap_or(0) as usize;
    let visible_source_indices = if extract_visible_source_indices && visible_count > 0 {
        runtime.ensure_buffer(
            MetalBufferSlot::VisibleSourceIndices,
            visible_count.saturating_mul(size_of::<u32>()),
        )?;
        let pipeline = runtime
            .ensure_pipeline(MetalKernel::ExtractSourceIndices)?
            .clone();
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::ExtractSourceIndices.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            runtime
                .buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        encoder.set_buffer(
            1,
            runtime
                .buffer_handle(MetalBufferSlot::VisibleSourceIndices)?
                .map(|buffer| buffer.as_ref()),
            0,
        );
        let count = visible_count as u32;
        encoder.set_bytes(2, &count);
        let threads_per_group = pipeline
            .max_total_threads_per_threadgroup()
            .min(visible_count)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: visible_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        runtime.device.synchronize()?;
        runtime.read_buffer_structs::<u32>(MetalBufferSlot::VisibleSourceIndices, visible_count)?
    } else {
        Vec::new()
    };

    Ok(ProjectedGpuBatch {
        visible_count,
        visible_source_indices,
    })
}

pub(crate) fn ensure_projection_record_buffer(
    runtime: &mut MetalRuntime,
    count: usize,
) -> candle_core::Result<()> {
    runtime.ensure_buffer(
        MetalBufferSlot::ProjectionRecords,
        count.saturating_mul(size_of::<MetalProjectionRecord>()),
    )
}

pub(crate) fn write_projection_records(
    runtime: &mut MetalRuntime,
    records: &[MetalProjectionRecord],
) -> candle_core::Result<()> {
    runtime.write_slice(MetalBufferSlot::ProjectionRecords, records)
}

pub(crate) fn build_tile_bins(
    runtime: &MetalRuntime,
    min_x_values: &[f32],
    max_x_values: &[f32],
    min_y_values: &[f32],
    max_y_values: &[f32],
) -> candle_core::Result<MetalTileBins> {
    let total = min_x_values.len();
    if max_x_values.len() != total || min_y_values.len() != total || max_y_values.len() != total {
        candle_core::bail!("tile binning expects matching bound lengths");
    }

    let (num_tiles_x, num_tiles_y) = runtime.tile_grid_dims();
    let tile_count = num_tiles_x.saturating_mul(num_tiles_y);
    let mut tile_counts = vec![0usize; tile_count];
    let mut total_assignments = 0usize;

    for idx in 0..total {
        let tile_x_min = (min_x_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_x_max = (max_x_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_y_min = (min_y_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_y_max = (max_y_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;

        for ty in tile_y_min..=tile_y_max.min(num_tiles_y.saturating_sub(1)) {
            for tx in tile_x_min..=tile_x_max.min(num_tiles_x.saturating_sub(1)) {
                tile_counts[ty * num_tiles_x + tx] += 1;
                total_assignments += 1;
            }
        }
    }

    let mut records = Vec::with_capacity(tile_count);
    let mut active_tiles = Vec::new();
    let mut max_gaussians_per_tile = 0usize;
    let mut start = 0usize;
    for (tile_idx, count) in tile_counts.iter().copied().enumerate() {
        records.push(MetalTileDispatchRecord::new(start as u32, count as u32));
        if count > 0 {
            active_tiles.push(tile_idx);
            max_gaussians_per_tile = max_gaussians_per_tile.max(count);
        }
        start += count;
    }

    let mut packed_indices = vec![0u32; total_assignments];
    let mut write_offsets: Vec<usize> = records.iter().map(|record| record.start()).collect();
    for idx in 0..total {
        let tile_x_min = (min_x_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_x_max = (max_x_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_y_min = (min_y_values[idx].floor().max(0.0) as usize) / METAL_TILE_SIZE;
        let tile_y_max = (max_y_values[idx].ceil().max(0.0) as usize) / METAL_TILE_SIZE;

        for ty in tile_y_min..=tile_y_max.min(num_tiles_y.saturating_sub(1)) {
            for tx in tile_x_min..=tile_x_max.min(num_tiles_x.saturating_sub(1)) {
                let tile_idx = ty * num_tiles_x + tx;
                let write_idx = write_offsets[tile_idx];
                packed_indices[write_idx] = idx as u32;
                write_offsets[tile_idx] += 1;
            }
        }
    }

    Ok(MetalTileBins::from_parts(
        records,
        active_tiles,
        packed_indices,
        total_assignments,
        max_gaussians_per_tile,
    ))
}

pub(crate) fn build_tile_bins_gpu(
    runtime: &mut MetalRuntime,
    gaussian_count: usize,
) -> candle_core::Result<MetalTileBins> {
    let (num_tiles_x, num_tiles_y) = runtime.tile_grid_dims();
    let tile_count = num_tiles_x.saturating_mul(num_tiles_y);
    if gaussian_count == 0 {
        return Ok(MetalTileBins::default());
    }

    runtime.ensure_buffer(
        MetalBufferSlot::TileCounts,
        tile_count.saturating_mul(size_of::<u32>()),
    )?;
    runtime.dispatch_fill_u32(MetalBufferSlot::TileCounts, 0, tile_count)?;

    let count_pipeline = runtime.ensure_pipeline(MetalKernel::TileCount)?.clone();
    let metal = runtime.device.as_metal_device()?.clone();
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::TileCount.function_name());
    encoder.set_compute_pipeline_state(&count_pipeline);
    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::CameraUniforms)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        1,
        runtime
            .buffer_handle(MetalBufferSlot::ProjectionRecords)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        2,
        runtime
            .buffer_handle(MetalBufferSlot::TileCounts)?
            .map(|b| b.as_ref()),
        0,
    );
    let count = gaussian_count as u32;
    encoder.set_bytes(3, &count);
    let threads_per_group = count_pipeline
        .max_total_threads_per_threadgroup()
        .min(gaussian_count)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: gaussian_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    runtime.device.synchronize()?;

    let tile_count_u32 = tile_count as u32;
    runtime.ensure_buffer(
        MetalBufferSlot::TileOffsets,
        tile_count.saturating_mul(size_of::<u32>()),
    )?;
    runtime.ensure_buffer(MetalBufferSlot::TotalAssignments, size_of::<u32>())?;

    let prefix_pipeline = runtime.ensure_pipeline(MetalKernel::TilePrefixSum)?.clone();
    let metal2 = runtime.device.as_metal_device()?.clone();
    let prefix_encoder = metal2.command_encoder()?;
    prefix_encoder.set_label(MetalKernel::TilePrefixSum.function_name());
    prefix_encoder.set_compute_pipeline_state(&prefix_pipeline);
    prefix_encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::TileCounts)?
            .map(|b| b.as_ref()),
        0,
    );
    prefix_encoder.set_buffer(
        1,
        runtime
            .buffer_handle(MetalBufferSlot::TileOffsets)?
            .map(|b| b.as_ref()),
        0,
    );
    prefix_encoder.set_buffer(
        2,
        runtime
            .buffer_handle(MetalBufferSlot::TotalAssignments)?
            .map(|b| b.as_ref()),
        0,
    );
    prefix_encoder.set_bytes(3, &tile_count_u32);
    prefix_encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    drop(prefix_encoder);
    runtime.device.synchronize()?;

    let total_vec = runtime.read_buffer_structs::<u32>(MetalBufferSlot::TotalAssignments, 1)?;
    let total_assignments = total_vec.first().copied().unwrap_or(0) as usize;
    let tile_counts_u32 =
        runtime.read_buffer_structs::<u32>(MetalBufferSlot::TileCounts, tile_count)?;
    let tile_offsets_u32 =
        runtime.read_buffer_structs::<u32>(MetalBufferSlot::TileOffsets, tile_count)?;
    let mut records = Vec::with_capacity(tile_count);
    let mut active_tiles = Vec::new();
    let mut max_gaussians_per_tile = 0usize;
    for tile_idx in 0..tile_count {
        let cnt = tile_counts_u32[tile_idx];
        let off = tile_offsets_u32[tile_idx];
        records.push(MetalTileDispatchRecord::new(off, cnt));
        if cnt > 0 {
            active_tiles.push(tile_idx);
            max_gaussians_per_tile = max_gaussians_per_tile.max(cnt as usize);
        }
    }

    runtime.ensure_buffer(
        MetalBufferSlot::TileIndices,
        total_assignments.saturating_mul(size_of::<u32>()),
    )?;

    let assign_pipeline = runtime.ensure_pipeline(MetalKernel::TileAssign)?.clone();
    let metal = runtime.device.as_metal_device()?.clone();
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::TileAssign.function_name());
    encoder.set_compute_pipeline_state(&assign_pipeline);
    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::CameraUniforms)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        1,
        runtime
            .buffer_handle(MetalBufferSlot::ProjectionRecords)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        2,
        runtime
            .buffer_handle(MetalBufferSlot::TileOffsets)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        3,
        runtime
            .buffer_handle(MetalBufferSlot::TileIndices)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_bytes(4, &count);
    let threads_per_group = assign_pipeline
        .max_total_threads_per_threadgroup()
        .min(gaussian_count)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: gaussian_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    runtime.device.synchronize()?;
    let packed_indices =
        runtime.read_buffer_structs::<u32>(MetalBufferSlot::TileIndices, total_assignments)?;

    Ok(MetalTileBins::from_parts(
        records,
        active_tiles,
        packed_indices,
        total_assignments,
        max_gaussians_per_tile,
    ))
}
