use candle_core::Device;
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::MTLSize;

use super::kernels::MetalKernel;

pub(crate) fn dispatch_fill_u32(
    device: &Device,
    pipeline: &ComputePipeline,
    buffer: &Buffer,
    value: u32,
    len: usize,
) -> candle_core::Result<()> {
    let metal = device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::FillU32.function_name());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_bytes(1, &value);
    let count = len as u32;
    encoder.set_bytes(2, &count);
    let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(len).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: len,
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
    device.synchronize()?;
    Ok(())
}

pub(crate) fn dispatch_adam_step(
    device: &Device,
    pipeline: &ComputePipeline,
    params: &Buffer,
    grads: &Buffer,
    m: &Buffer,
    v: &Buffer,
    hyperparams: &Buffer,
    element_count: usize,
) -> candle_core::Result<()> {
    let metal = device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::AdamStep.function_name());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(params), 0);
    encoder.set_buffer(1, Some(grads), 0);
    encoder.set_buffer(2, Some(m), 0);
    encoder.set_buffer(3, Some(v), 0);
    encoder.set_buffer(4, Some(hyperparams), 0);

    let threads_per_group = pipeline
        .max_total_threads_per_threadgroup()
        .min(element_count)
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: element_count,
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
    Ok(())
}

pub(crate) fn dispatch_grad_magnitudes(
    device: &Device,
    pipeline: &ComputePipeline,
    grad_positions: &Buffer,
    grad_scales: &Buffer,
    grad_opacity: &Buffer,
    grad_colors: &Buffer,
    grad_magnitudes: &Buffer,
    gaussian_count: usize,
) -> candle_core::Result<()> {
    let metal = device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::GradMagnitudes.function_name());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(grad_positions), 0);
    encoder.set_buffer(1, Some(grad_scales), 0);
    encoder.set_buffer(2, Some(grad_opacity), 0);
    encoder.set_buffer(3, Some(grad_colors), 0);
    encoder.set_buffer(4, Some(grad_magnitudes), 0);
    let count = gaussian_count as u32;
    encoder.set_bytes(5, &count);
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
    device.synchronize()?;
    Ok(())
}

pub(crate) fn dispatch_projected_grad_magnitudes(
    device: &Device,
    pipeline: &ComputePipeline,
    grad_projected_positions: &Buffer,
    grad_magnitudes: &Buffer,
    gaussian_count: usize,
) -> candle_core::Result<()> {
    let metal = device.as_metal_device()?;
    let encoder = metal.command_encoder()?;
    encoder.set_label(MetalKernel::ProjectedGradMagnitudes.function_name());
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(grad_projected_positions), 0);
    encoder.set_buffer(1, Some(grad_magnitudes), 0);
    let count = gaussian_count as u32;
    encoder.set_bytes(2, &count);
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
    device.synchronize()?;
    Ok(())
}
