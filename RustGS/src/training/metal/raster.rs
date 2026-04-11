use std::mem::size_of;

use candle_core::DType;
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

use super::kernels::MetalKernel;
use super::runtime::{
    MetalBufferSlot, MetalProjectedGaussian, MetalRuntime, MetalTileBins, MetalTileDispatchRecord,
    NativeBackwardFrame, NativeForwardFrame, NativeForwardProfile, METAL_TILE_SIZE,
};

pub(crate) fn reserve_forward_buffers(
    runtime: &mut MetalRuntime,
    gaussian_count: usize,
    tile_ref_count: usize,
    pixel_count: usize,
) -> candle_core::Result<()> {
    runtime.ensure_buffer(
        MetalBufferSlot::ProjectedGaussians,
        gaussian_count.saturating_mul(size_of::<MetalProjectedGaussian>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::TileMetadata,
        runtime
            .tile_window_count()
            .saturating_mul(size_of::<MetalTileDispatchRecord>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::TileIndices,
        tile_ref_count.saturating_mul(size_of::<u32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::OutputColor,
        pixel_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::OutputDepth,
        pixel_count.saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::OutputAlpha,
        pixel_count.saturating_mul(size_of::<f32>()),
    )?;
    Ok(())
}

pub(crate) fn rasterize_forward(
    runtime: &mut MetalRuntime,
    gaussian_count: usize,
    tile_bins: &MetalTileBins,
    render_width: usize,
    render_height: usize,
) -> candle_core::Result<(NativeForwardFrame, NativeForwardProfile)> {
    let total_start = std::time::Instant::now();
    let pixel_count = render_width.saturating_mul(render_height);
    let setup_start = std::time::Instant::now();
    runtime.reserve_forward_buffers(gaussian_count, tile_bins.total_assignments(), pixel_count)?;
    let pipeline = runtime.ensure_pipeline(MetalKernel::TileForward)?.clone();
    let color_buffer = runtime
        .buffer_handle(MetalBufferSlot::OutputColor)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing output color buffer".into()))?;
    let depth_buffer = runtime
        .buffer_handle(MetalBufferSlot::OutputDepth)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing output depth buffer".into()))?;
    let alpha_buffer = runtime
        .buffer_handle(MetalBufferSlot::OutputAlpha)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing output alpha buffer".into()))?;
    let tile_buffer = runtime
        .buffer_handle(MetalBufferSlot::TileMetadata)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing tile metadata buffer".into()))?;
    let tile_index_buffer = runtime
        .buffer_handle(MetalBufferSlot::TileIndices)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing tile index buffer".into()))?;
    let gaussian_buffer = runtime
        .buffer_handle(MetalBufferSlot::ProjectionRecords)?
        .cloned()
        .ok_or_else(|| candle_core::Error::Msg("missing gaussian buffer".into()))?;
    let setup = setup_start.elapsed();

    let staging_start = std::time::Instant::now();
    runtime.write_slice(MetalBufferSlot::TileMetadata, tile_bins.records())?;
    runtime.write_slice(MetalBufferSlot::TileIndices, tile_bins.packed_indices())?;
    let staging = staging_start.elapsed();

    let kernel_start = std::time::Instant::now();
    let metal = runtime.device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::TileForward.function_name());
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::CameraUniforms)?
            .map(|buffer| buffer.as_ref()),
        0,
    );
    encoder.set_buffer(1, Some(tile_buffer.as_ref()), 0);
    encoder.set_buffer(2, Some(tile_index_buffer.as_ref()), 0);
    encoder.set_buffer(3, Some(gaussian_buffer.as_ref()), 0);
    encoder.set_buffer(4, Some(color_buffer.as_ref()), 0);
    encoder.set_buffer(5, Some(depth_buffer.as_ref()), 0);
    encoder.set_buffer(6, Some(alpha_buffer.as_ref()), 0);
    let threads_per_group = tile_group_dims(&pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: render_width,
            height: render_height,
            depth: 1,
        },
        threads_per_group,
    );
    drop(encoder);
    runtime.device.synchronize()?;
    let kernel = kernel_start.elapsed();

    let frame = NativeForwardFrame {
        color: runtime.tensor_from_buffer(
            MetalBufferSlot::OutputColor,
            pixel_count.saturating_mul(3),
            DType::F32,
            (pixel_count, 3),
        )?,
        depth: runtime.tensor_from_buffer(
            MetalBufferSlot::OutputDepth,
            pixel_count,
            DType::F32,
            (pixel_count,),
        )?,
        alpha: runtime.tensor_from_buffer(
            MetalBufferSlot::OutputAlpha,
            pixel_count,
            DType::F32,
            (pixel_count,),
        )?,
    };

    Ok((
        frame,
        NativeForwardProfile {
            setup,
            staging,
            kernel,
            total: total_start.elapsed(),
        },
    ))
}

pub(crate) fn reserve_backward_buffers(
    runtime: &mut MetalRuntime,
    gaussian_count: usize,
    pixel_count: usize,
) -> candle_core::Result<()> {
    runtime.ensure_buffer(
        MetalBufferSlot::OutputColor,
        pixel_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::OutputDepth,
        pixel_count.saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::OutputAlpha,
        pixel_count.saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::TargetColor,
        pixel_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::TargetDepth,
        pixel_count.saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(MetalBufferSlot::LossScalars, 4 * size_of::<f32>())?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradPositions,
        gaussian_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradProjectedPositions,
        gaussian_count
            .saturating_mul(2)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradScales,
        gaussian_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradOpacity,
        gaussian_count.saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradColors,
        gaussian_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::GradMagnitudes,
        gaussian_count.saturating_mul(size_of::<f32>()),
    )?;
    Ok(())
}

pub(crate) fn reserve_ssim_grad_buffer(
    runtime: &mut MetalRuntime,
    pixel_count: usize,
) -> candle_core::Result<()> {
    runtime.ensure_buffer(
        MetalBufferSlot::SsimColorGrad,
        pixel_count
            .saturating_mul(3)
            .saturating_mul(size_of::<f32>()),
    )
}

pub(crate) fn write_ssim_grad(
    runtime: &mut MetalRuntime,
    ssim_grad: &[f32],
) -> candle_core::Result<()> {
    runtime.write_slice(MetalBufferSlot::SsimColorGrad, ssim_grad)
}

pub(crate) fn write_target_data(
    runtime: &mut MetalRuntime,
    target_color: &[f32],
    target_depth: &[f32],
    color_scale: f32,
    depth_scale: f32,
    ssim_scale: f32,
    alpha_scale: f32,
) -> candle_core::Result<()> {
    runtime.write_slice(MetalBufferSlot::TargetColor, target_color)?;
    runtime.write_slice(MetalBufferSlot::TargetDepth, target_depth)?;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LossScalars {
        color: f32,
        depth: f32,
        ssim: f32,
        alpha: f32,
    }
    runtime.write_struct(
        MetalBufferSlot::LossScalars,
        &LossScalars {
            color: color_scale,
            depth: depth_scale,
            ssim: ssim_scale,
            alpha: alpha_scale,
        },
    )
}

pub(crate) fn rasterize_backward(
    runtime: &mut MetalRuntime,
    gaussian_count: usize,
    tile_bins: &MetalTileBins,
    render_width: usize,
    render_height: usize,
) -> candle_core::Result<(NativeBackwardFrame, NativeForwardProfile)> {
    let total_start = std::time::Instant::now();
    let pixel_count = render_width.saturating_mul(render_height);

    runtime.reserve_backward_buffers(gaussian_count, pixel_count)?;
    runtime.reserve_ssim_grad_buffer(pixel_count)?;
    runtime.ensure_buffer(
        MetalBufferSlot::TileMetadata,
        runtime
            .tile_window_count()
            .saturating_mul(size_of::<MetalTileDispatchRecord>()),
    )?;
    runtime.ensure_buffer(
        MetalBufferSlot::TileIndices,
        tile_bins
            .total_assignments()
            .saturating_mul(size_of::<u32>()),
    )?;

    let pipeline = runtime.ensure_pipeline(MetalKernel::TileBackward)?.clone();

    runtime.dispatch_fill_u32(MetalBufferSlot::GradPositions, 0, gaussian_count * 3)?;
    runtime.dispatch_fill_u32(
        MetalBufferSlot::GradProjectedPositions,
        0,
        gaussian_count * 2,
    )?;
    runtime.dispatch_fill_u32(MetalBufferSlot::GradScales, 0, gaussian_count * 3)?;
    runtime.dispatch_fill_u32(MetalBufferSlot::GradOpacity, 0, gaussian_count)?;
    runtime.dispatch_fill_u32(MetalBufferSlot::GradColors, 0, gaussian_count * 3)?;

    let setup = total_start.elapsed();

    let staging_start = std::time::Instant::now();
    runtime.write_slice(MetalBufferSlot::TileMetadata, tile_bins.records())?;
    runtime.write_slice(MetalBufferSlot::TileIndices, tile_bins.packed_indices())?;
    let staging = staging_start.elapsed();

    let kernel_start = std::time::Instant::now();
    let metal = runtime.device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::TileBackward.function_name());
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(
        0,
        runtime
            .buffer_handle(MetalBufferSlot::CameraUniforms)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        1,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::TileMetadata)?
                .ok_or_else(|| candle_core::Error::Msg("missing tile metadata".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        2,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::TileIndices)?
                .ok_or_else(|| candle_core::Error::Msg("missing tile indices".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        3,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::ProjectionRecords)?
                .ok_or_else(|| candle_core::Error::Msg("missing projected gaussians".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        4,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::OutputColor)?
                .ok_or_else(|| candle_core::Error::Msg("missing output color".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        5,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::OutputDepth)?
                .ok_or_else(|| candle_core::Error::Msg("missing output depth".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        6,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::OutputAlpha)?
                .ok_or_else(|| candle_core::Error::Msg("missing output alpha".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        7,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::TargetColor)?
                .ok_or_else(|| candle_core::Error::Msg("missing target color".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        8,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::TargetDepth)?
                .ok_or_else(|| candle_core::Error::Msg("missing target depth".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        9,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::GradPositions)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad positions".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        10,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::GradScales)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad scales".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        11,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::GradOpacity)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad opacity".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        12,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::GradColors)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad colors".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        13,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::LossScalars)?
                .ok_or_else(|| candle_core::Error::Msg("missing loss scalars".into()))?
                .as_ref(),
        ),
        0,
    );
    encoder.set_buffer(
        14,
        runtime
            .buffer_handle(MetalBufferSlot::SsimColorGrad)?
            .map(|b| b.as_ref()),
        0,
    );
    encoder.set_buffer(
        15,
        Some(
            runtime
                .buffer_handle(MetalBufferSlot::GradProjectedPositions)?
                .ok_or_else(|| candle_core::Error::Msg("missing projected position grads".into()))?
                .as_ref(),
        ),
        0,
    );

    let threads_per_group = tile_group_dims(&pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: render_width,
            height: render_height,
            depth: 1,
        },
        threads_per_group,
    );
    drop(encoder);
    runtime.device.synchronize()?;
    let kernel = kernel_start.elapsed();

    let frame = NativeBackwardFrame {
        grad_positions: runtime.tensor_from_buffer(
            MetalBufferSlot::GradPositions,
            gaussian_count * 3,
            DType::F32,
            (gaussian_count, 3),
        )?,
        grad_log_scales: runtime.tensor_from_buffer(
            MetalBufferSlot::GradScales,
            gaussian_count * 3,
            DType::F32,
            (gaussian_count, 3),
        )?,
        grad_opacity_logits: runtime.tensor_from_buffer(
            MetalBufferSlot::GradOpacity,
            gaussian_count,
            DType::F32,
            (gaussian_count,),
        )?,
        grad_colors: runtime.tensor_from_buffer(
            MetalBufferSlot::GradColors,
            gaussian_count * 3,
            DType::F32,
            (gaussian_count, 3),
        )?,
    };

    Ok((
        frame,
        NativeForwardProfile {
            setup,
            staging,
            kernel,
            total: total_start.elapsed(),
        },
    ))
}

fn tile_group_dims(pipeline: &ComputePipeline) -> MTLSize {
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let side = (max_threads as f64).sqrt().floor() as usize;
    let width = side.clamp(1, METAL_TILE_SIZE);
    let height = (max_threads / width).max(1).min(METAL_TILE_SIZE);
    MTLSize {
        width,
        height,
        depth: 1,
    }
}
