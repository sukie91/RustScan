use std::mem::size_of;

use candle_core::{DType, Device, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;

use crate::diff::diff_splat::{DiffCamera, Splats};

use super::dispatch as metal_dispatch;
use super::kernels::MetalKernel;
use super::pipelines::MetalPipelineCache;
use super::projection as metal_projection;
use super::raster as metal_raster;
use super::resources::MetalResources;
pub(crate) use super::resources::{MetalBufferSlot, MetalGaussianBindings};

pub(crate) const METAL_TILE_SIZE: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenRect {
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
}

pub(crate) struct BatchPixelWindow {
    pub pixel_x: Tensor,
    pub pixel_y: Tensor,
    pub indices: Tensor,
    pub pixel_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalCameraUniform {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    tile_size: u32,
    num_tiles_x: u32,
    rot00: f32,
    rot01: f32,
    rot02: f32,
    rot10: f32,
    rot11: f32,
    rot12: f32,
    rot20: f32,
    rot21: f32,
    rot22: f32,
    tx: f32,
    ty: f32,
    tz: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalTileDispatchRecord {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

impl MetalTileDispatchRecord {
    pub(crate) fn new(start: u32, count: u32) -> Self {
        Self {
            start,
            count,
            ..Default::default()
        }
    }

    pub(crate) fn start(&self) -> usize {
        self.start as usize
    }

    pub(crate) fn count(&self) -> usize {
        self.count as usize
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MetalTileBins {
    records: Vec<MetalTileDispatchRecord>,
    active_tiles: Vec<usize>,
    packed_indices: Vec<u32>,
    total_assignments: usize,
    max_gaussians_per_tile: usize,
}

impl MetalTileBins {
    pub(crate) fn from_parts(
        records: Vec<MetalTileDispatchRecord>,
        active_tiles: Vec<usize>,
        packed_indices: Vec<u32>,
        total_assignments: usize,
        max_gaussians_per_tile: usize,
    ) -> Self {
        Self {
            records,
            active_tiles,
            packed_indices,
            total_assignments,
            max_gaussians_per_tile,
        }
    }

    pub(crate) fn active_tiles(&self) -> &[usize] {
        &self.active_tiles
    }

    pub(crate) fn active_tile_count(&self) -> usize {
        self.active_tiles.len()
    }

    pub(crate) fn total_assignments(&self) -> usize {
        self.total_assignments
    }

    pub(crate) fn max_gaussians_per_tile(&self) -> usize {
        self.max_gaussians_per_tile
    }

    pub(crate) fn packed_indices(&self) -> &[u32] {
        &self.packed_indices
    }

    pub(crate) fn records(&self) -> &[MetalTileDispatchRecord] {
        &self.records
    }

    pub(crate) fn record(&self, tile_idx: usize) -> Option<MetalTileDispatchRecord> {
        self.records.get(tile_idx).copied()
    }

    #[cfg(test)]
    pub(crate) fn indices_for_tile(&self, tile_idx: usize) -> &[u32] {
        let Some(record) = self.records.get(tile_idx) else {
            return &[];
        };
        let start = record.start();
        let end = start.saturating_add(record.count());
        self.packed_indices.get(start..end).unwrap_or(&[])
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalProjectedGaussian {
    pub u: f32,
    pub v: f32,
    pub sigma_x: f32,
    pub sigma_y: f32,
    pub depth: f32,
    pub opacity: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetalProjectionRecord {
    // Hot fields (forward inner loop)
    pub u: f32,
    pub v: f32,
    pub sigma_x: f32,
    pub sigma_y: f32,
    pub depth: f32,
    pub opacity: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    // Warm fields (backward)
    pub raw_sigma_x: f32,
    pub raw_sigma_y: f32,
    pub opacity_logit: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    // Cold fields (index / tile binning)
    pub source_idx: u32,
    pub visible: u32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

pub(crate) struct NativeForwardFrame {
    pub color: Tensor,
    pub depth: Tensor,
    pub alpha: Tensor,
}

pub(crate) struct NativeBackwardFrame {
    pub grad_positions: Tensor,
    pub grad_log_scales: Tensor,
    pub grad_opacity_logits: Tensor,
    pub grad_colors: Tensor,
}

pub(crate) struct ProjectedGpuBatch {
    pub visible_count: usize,
    pub visible_source_indices: Vec<u32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NativeForwardProfile {
    pub setup: std::time::Duration,
    pub staging: std::time::Duration,
    pub kernel: std::time::Duration,
    pub total: std::time::Duration,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MetalRuntimeStats {
    pub tile_windows: usize,
    pub buffer_allocations: usize,
    #[allow(dead_code)]
    pub buffer_reuses: usize,
    pub pipeline_compilations: usize,
}

pub(crate) struct MetalRuntime {
    pub(crate) device: Device,
    tile_windows: Vec<BatchPixelWindow>,
    num_tiles_x: usize,
    num_tiles_y: usize,
    resources: MetalResources,
    pipelines: MetalPipelineCache,
    pipeline_compilations: usize,
}

impl MetalRuntime {
    pub(crate) fn new(
        render_width: usize,
        render_height: usize,
        device: Device,
    ) -> candle_core::Result<Self> {
        let num_tiles_x = render_width.div_ceil(METAL_TILE_SIZE);
        let num_tiles_y = render_height.div_ceil(METAL_TILE_SIZE);
        let mut tile_windows = Vec::with_capacity(num_tiles_x * num_tiles_y);

        for tile_y in 0..num_tiles_y {
            for tile_x in 0..num_tiles_x {
                let min_x = tile_x * METAL_TILE_SIZE;
                let min_y = tile_y * METAL_TILE_SIZE;
                let max_x = (min_x + METAL_TILE_SIZE)
                    .min(render_width)
                    .saturating_sub(1);
                let max_y = (min_y + METAL_TILE_SIZE)
                    .min(render_height)
                    .saturating_sub(1);
                tile_windows.push(build_batch_pixel_window(
                    &device,
                    render_width,
                    ScreenRect {
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    },
                )?);
            }
        }

        Ok(Self {
            device,
            tile_windows,
            num_tiles_x,
            num_tiles_y,
            resources: MetalResources::default(),
            pipelines: MetalPipelineCache::default(),
            pipeline_compilations: 0,
        })
    }

    pub(crate) fn tile_window(&self, tile_idx: usize) -> candle_core::Result<&BatchPixelWindow> {
        self.tile_windows
            .get(tile_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("invalid tile index {tile_idx}")))
    }

    #[cfg(test)]
    pub(crate) fn tile_grid(&self) -> (usize, usize) {
        (self.num_tiles_x, self.num_tiles_y)
    }

    pub(crate) fn tile_grid_dims(&self) -> (usize, usize) {
        (self.num_tiles_x, self.num_tiles_y)
    }

    pub(crate) fn tile_window_count(&self) -> usize {
        self.tile_windows.len()
    }

    pub(crate) fn stats(&self) -> MetalRuntimeStats {
        MetalRuntimeStats {
            tile_windows: self.tile_windows.len(),
            buffer_allocations: self.resources.buffer_allocations(),
            buffer_reuses: self.resources.buffer_reuses(),
            pipeline_compilations: self.pipeline_compilations,
        }
    }

    pub(crate) fn tile_index_capacity_bytes(&self) -> usize {
        self.resources.buffer_capacity(MetalBufferSlot::TileIndices)
    }

    pub(crate) fn prime_tile_index_buffer(&mut self) -> candle_core::Result<()> {
        if self.device.is_metal() {
            self.dispatch_fill_u32(MetalBufferSlot::TileIndices, 0, 1)?;
        }
        Ok(())
    }

    pub(crate) fn reserve_core_buffers(
        &mut self,
        gaussian_capacity: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::CameraUniforms,
            size_of::<MetalCameraUniform>(),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::VisibleIndices,
            gaussian_capacity.saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileCounts,
            self.tile_windows.len().saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileOffsets,
            self.tile_windows.len().saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileDispatchRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            gaussian_capacity
                .saturating_mul(4)
                .saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradPositions,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradProjectedPositions,
            gaussian_capacity
                .saturating_mul(2)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradRotations,
            gaussian_capacity
                .saturating_mul(4)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradScales,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradOpacity,
            gaussian_capacity.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::GradColors,
            gaussian_capacity
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        Ok(())
    }

    pub(crate) fn stage_camera(&mut self, camera: &DiffCamera) -> candle_core::Result<()> {
        let camera_uniform = MetalCameraUniform {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
            width: camera.width as u32,
            height: camera.height as u32,
            tile_size: METAL_TILE_SIZE as u32,
            num_tiles_x: self.num_tiles_x as u32,
            rot00: camera.rotation[0][0],
            rot01: camera.rotation[0][1],
            rot02: camera.rotation[0][2],
            rot10: camera.rotation[1][0],
            rot11: camera.rotation[1][1],
            rot12: camera.rotation[1][2],
            rot20: camera.rotation[2][0],
            rot21: camera.rotation[2][1],
            rot22: camera.rotation[2][2],
            tx: camera.translation[0],
            ty: camera.translation[1],
            tz: camera.translation[2],
        };
        self.write_struct(MetalBufferSlot::CameraUniforms, &camera_uniform)
    }

    pub(crate) fn bind_gaussians<'a>(
        &self,
        gaussians: &'a Splats,
        render_colors: &'a Tensor,
    ) -> candle_core::Result<MetalGaussianBindings<'a>> {
        self.resources.bind_gaussians(gaussians, render_colors)
    }

    pub(crate) fn project_gaussians(
        &mut self,
        gaussians: &Splats,
        render_colors: &Tensor,
        extract_visible_source_indices: bool,
    ) -> candle_core::Result<ProjectedGpuBatch> {
        metal_projection::project_gaussians(
            self,
            gaussians,
            render_colors,
            extract_visible_source_indices,
        )
    }

    pub(crate) fn ensure_projection_record_buffer(
        &mut self,
        count: usize,
    ) -> candle_core::Result<()> {
        metal_projection::ensure_projection_record_buffer(self, count)
    }

    pub(crate) fn write_projection_records(
        &mut self,
        records: &[MetalProjectionRecord],
    ) -> candle_core::Result<()> {
        metal_projection::write_projection_records(self, records)
    }

    pub(crate) fn build_tile_bins(
        &self,
        min_x_values: &[f32],
        max_x_values: &[f32],
        min_y_values: &[f32],
        max_y_values: &[f32],
    ) -> candle_core::Result<MetalTileBins> {
        metal_projection::build_tile_bins(
            self,
            min_x_values,
            max_x_values,
            min_y_values,
            max_y_values,
        )
    }

    pub(crate) fn build_tile_bins_gpu(
        &mut self,
        gaussian_count: usize,
    ) -> candle_core::Result<MetalTileBins> {
        metal_projection::build_tile_bins_gpu(self, gaussian_count)
    }

    pub(crate) fn reserve_forward_buffers(
        &mut self,
        gaussian_count: usize,
        tile_ref_count: usize,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        metal_raster::reserve_forward_buffers(self, gaussian_count, tile_ref_count, pixel_count)
    }

    pub(crate) fn rasterize_forward(
        &mut self,
        gaussian_count: usize,
        tile_bins: &MetalTileBins,
        render_width: usize,
        render_height: usize,
    ) -> candle_core::Result<(NativeForwardFrame, NativeForwardProfile)> {
        metal_raster::rasterize_forward(
            self,
            gaussian_count,
            tile_bins,
            render_width,
            render_height,
        )
    }

    pub(crate) fn reserve_backward_buffers(
        &mut self,
        gaussian_count: usize,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        metal_raster::reserve_backward_buffers(self, gaussian_count, pixel_count)
    }

    pub(crate) fn reserve_ssim_grad_buffer(
        &mut self,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        metal_raster::reserve_ssim_grad_buffer(self, pixel_count)
    }

    pub(crate) fn write_ssim_grad(&mut self, ssim_grad: &[f32]) -> candle_core::Result<()> {
        metal_raster::write_ssim_grad(self, ssim_grad)
    }

    pub(crate) fn write_target_data(
        &mut self,
        target_color: &[f32],
        target_depth: &[f32],
        color_scale: f32,
        depth_scale: f32,
        ssim_scale: f32,
        alpha_scale: f32,
    ) -> candle_core::Result<()> {
        metal_raster::write_target_data(
            self,
            target_color,
            target_depth,
            color_scale,
            depth_scale,
            ssim_scale,
            alpha_scale,
        )
    }

    pub(crate) fn rasterize_backward(
        &mut self,
        gaussian_count: usize,
        tile_bins: &MetalTileBins,
        render_width: usize,
        render_height: usize,
    ) -> candle_core::Result<(NativeBackwardFrame, NativeForwardProfile)> {
        metal_raster::rasterize_backward(
            self,
            gaussian_count,
            tile_bins,
            render_width,
            render_height,
        )
    }

    /// Fused Adam update: runs entirely on GPU, no intermediate Tensor allocations.
    /// `param_buf` is the raw Metal buffer of a `Var`'s storage.
    pub(crate) fn adam_step_fused(
        &mut self,
        param_slot: MetalBufferSlot,
        grad_slot: MetalBufferSlot,
        m_slot: MetalBufferSlot,
        v_slot: MetalBufferSlot,
        element_count: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
    ) -> candle_core::Result<()> {
        if element_count == 0 {
            return Ok(());
        }
        let bc1 = 1.0f32 / (1.0 - beta1.powi(step as i32));
        let bc2 = 1.0f32 / (1.0 - beta2.powi(step as i32));

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AdamHyperparams {
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            bc1: f32,
            bc2: f32,
        }
        let hp = AdamHyperparams {
            lr,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
        };

        let pipeline = self.ensure_pipeline(MetalKernel::AdamStep)?.clone();
        let get_buf = |slot: MetalBufferSlot, name: &'static str, rt: &MetalRuntime| {
            rt.buffer_handle(slot)?.cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("adam_step_fused: missing {name}").into())
            })
        };

        self.write_struct(MetalBufferSlot::AdamHyperparams, &hp)?;
        let hyperparams = self
            .buffer_handle(MetalBufferSlot::AdamHyperparams)?
            .cloned()
            .ok_or_else(|| {
                candle_core::Error::Msg("adam_step_fused: missing hyperparams".into())
            })?;
        metal_dispatch::dispatch_adam_step(
            &self.device,
            &pipeline,
            get_buf(param_slot, "params", self)?.as_ref(),
            get_buf(grad_slot, "grads", self)?.as_ref(),
            get_buf(m_slot, "m", self)?.as_ref(),
            get_buf(v_slot, "v", self)?.as_ref(),
            hyperparams.as_ref(),
            element_count,
        )
    }

    pub(crate) fn compute_grad_magnitudes(
        &mut self,
        gaussian_count: usize,
    ) -> candle_core::Result<Tensor> {
        if gaussian_count == 0 {
            return Tensor::zeros((0,), DType::F32, &self.device);
        }
        self.ensure_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count.saturating_mul(size_of::<f32>()),
        )?;
        let pipeline = self.ensure_pipeline(MetalKernel::GradMagnitudes)?.clone();
        metal_dispatch::dispatch_grad_magnitudes(
            &self.device,
            &pipeline,
            self.buffer_handle(MetalBufferSlot::GradPositions)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad positions".into()))?
                .as_ref(),
            self.buffer_handle(MetalBufferSlot::GradScales)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad scales".into()))?
                .as_ref(),
            self.buffer_handle(MetalBufferSlot::GradOpacity)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad opacity".into()))?
                .as_ref(),
            self.buffer_handle(MetalBufferSlot::GradColors)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad colors".into()))?
                .as_ref(),
            self.buffer_handle(MetalBufferSlot::GradMagnitudes)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad magnitudes".into()))?
                .as_ref(),
            gaussian_count,
        )?;
        self.tensor_from_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count,
            DType::F32,
            (gaussian_count,),
        )
    }

    pub(crate) fn compute_projected_grad_magnitudes(
        &mut self,
        gaussian_count: usize,
    ) -> candle_core::Result<Tensor> {
        if gaussian_count == 0 {
            return Tensor::zeros((0,), DType::F32, &self.device);
        }
        self.ensure_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count.saturating_mul(size_of::<f32>()),
        )?;
        let pipeline = self
            .ensure_pipeline(MetalKernel::ProjectedGradMagnitudes)?
            .clone();
        metal_dispatch::dispatch_projected_grad_magnitudes(
            &self.device,
            &pipeline,
            self.buffer_handle(MetalBufferSlot::GradProjectedPositions)?
                .ok_or_else(|| candle_core::Error::Msg("missing projected position grads".into()))?
                .as_ref(),
            self.buffer_handle(MetalBufferSlot::GradMagnitudes)?
                .ok_or_else(|| candle_core::Error::Msg("missing grad magnitudes".into()))?
                .as_ref(),
            gaussian_count,
        )?;
        self.tensor_from_buffer(
            MetalBufferSlot::GradMagnitudes,
            gaussian_count,
            DType::F32,
            (gaussian_count,),
        )
    }

    pub(crate) fn dispatch_fill_u32(
        &mut self,
        slot: MetalBufferSlot,
        value: u32,
        len: usize,
    ) -> candle_core::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.ensure_buffer(slot, len.saturating_mul(size_of::<u32>()))?;
        let pipeline = self.ensure_pipeline(MetalKernel::FillU32)?.clone();
        let Some(buffer) = self.buffer_handle(slot)?.cloned() else {
            candle_core::bail!("fill_u32 requires a Metal device");
        };
        metal_dispatch::dispatch_fill_u32(&self.device, &pipeline, buffer.as_ref(), value, len)
    }

    #[cfg(test)]
    pub(crate) fn stage_tensor_from_slice<T: candle_core::WithDType + Copy, S: Into<Shape>>(
        &mut self,
        slot: MetalBufferSlot,
        values: &[T],
        shape: S,
    ) -> candle_core::Result<Tensor> {
        self.resources
            .stage_tensor_from_slice(&self.device, slot, values, shape)
    }

    pub(crate) fn read_tensor_flat<T: candle_core::WithDType + Copy>(
        &self,
        tensor: &Tensor,
    ) -> candle_core::Result<Vec<T>> {
        self.resources.read_tensor_flat(&self.device, tensor)
    }

    pub(crate) fn read_buffer_structs<T: Copy>(
        &self,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<T>> {
        self.resources.read_buffer_structs(&self.device, slot, len)
    }

    #[cfg(test)]
    pub(crate) fn read_u32_buffer(
        &self,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<u32>> {
        self.resources.read_u32_buffer(&self.device, slot, len)
    }

    pub(crate) fn buffer_handle(
        &self,
        slot: MetalBufferSlot,
    ) -> candle_core::Result<Option<&std::sync::Arc<candle_metal_kernels::metal::Buffer>>> {
        self.resources.buffer_handle(slot)
    }

    pub(super) fn ensure_pipeline(
        &mut self,
        kernel: MetalKernel,
    ) -> candle_core::Result<&ComputePipeline> {
        if self.pipelines.ensure(&self.device, kernel)? {
            self.pipeline_compilations += 1;
        }
        Ok(self
            .pipelines
            .get(kernel)
            .expect("pipeline inserted before lookup"))
    }

    pub(crate) fn ensure_buffer(
        &mut self,
        slot: MetalBufferSlot,
        byte_capacity: usize,
    ) -> candle_core::Result<()> {
        self.resources
            .ensure_buffer(&self.device, slot, byte_capacity)
    }

    pub(crate) fn tensor_from_buffer<S: Into<Shape>>(
        &self,
        slot: MetalBufferSlot,
        element_count: usize,
        dtype: DType,
        shape: S,
    ) -> candle_core::Result<Tensor> {
        self.resources
            .tensor_from_buffer(&self.device, slot, element_count, dtype, shape)
    }

    pub(crate) fn write_struct<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        value: &T,
    ) -> candle_core::Result<()> {
        self.resources.write_struct(&self.device, slot, value)
    }

    pub(crate) fn write_slice<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        values: &[T],
    ) -> candle_core::Result<()> {
        self.resources.write_slice(&self.device, slot, values)
    }
}

fn build_batch_pixel_window(
    device: &Device,
    render_width: usize,
    rect: ScreenRect,
) -> candle_core::Result<BatchPixelWindow> {
    let width = rect.max_x - rect.min_x + 1;
    let height = rect.max_y - rect.min_y + 1;
    let mut xs = Vec::with_capacity(width * height);
    let mut ys = Vec::with_capacity(width * height);
    let mut indices = Vec::with_capacity(width * height);

    for y in rect.min_y..=rect.max_y {
        for x in rect.min_x..=rect.max_x {
            xs.push(x as f32 + 0.5);
            ys.push(y as f32 + 0.5);
            indices.push((y * render_width + x) as u32);
        }
    }

    let pixel_count = indices.len();
    Ok(BatchPixelWindow {
        pixel_x: Tensor::from_slice(&xs, (1, pixel_count), device)?,
        pixel_y: Tensor::from_slice(&ys, (1, pixel_count), device)?,
        indices: Tensor::from_slice(&indices, pixel_count, device)?,
        pixel_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_windows_cover_entire_surface() {
        let runtime = MetalRuntime::new(32, 17, Device::Cpu).unwrap();
        assert_eq!(runtime.tile_grid(), (2, 2));
        assert_eq!(runtime.stats().tile_windows, 4);
        assert_eq!(runtime.tile_window(0).unwrap().pixel_count, 16 * 16);
        assert_eq!(runtime.tile_window(3).unwrap().pixel_count, 16);
    }

    #[test]
    fn core_buffer_reservations_reuse_capacity() {
        let mut runtime = MetalRuntime::new(32, 16, Device::Cpu).unwrap();
        runtime.reserve_core_buffers(64).unwrap();
        let initial = runtime.stats();
        assert!(initial.buffer_allocations > 0);

        runtime.reserve_core_buffers(32).unwrap();
        let reused = runtime.stats();
        assert_eq!(reused.buffer_allocations, initial.buffer_allocations);
        assert!(reused.buffer_reuses >= initial.buffer_allocations);
    }

    #[test]
    fn fill_kernel_writes_shared_buffer() {
        let Ok(device) = crate::try_metal_device() else {
            return;
        };

        let mut runtime = MetalRuntime::new(16, 16, device).unwrap();
        runtime
            .dispatch_fill_u32(MetalBufferSlot::TileIndices, 7, 8)
            .unwrap();
        let values = runtime
            .read_u32_buffer(MetalBufferSlot::TileIndices, 8)
            .unwrap();
        assert_eq!(values, vec![7; 8]);
    }

    #[test]
    fn tile_bins_pack_indices_by_tile() {
        let runtime = MetalRuntime::new(32, 16, Device::Cpu).unwrap();
        let bins = runtime
            .build_tile_bins(
                &[2.0f32, 14.0],
                &[15.0f32, 18.0],
                &[1.0f32, 1.0],
                &[14.0f32, 14.0],
            )
            .unwrap();

        assert_eq!(bins.active_tile_count(), 2);
        assert_eq!(bins.total_assignments(), 3);
        assert_eq!(bins.max_gaussians_per_tile(), 2);
        assert_eq!(bins.packed_indices(), &[0, 1, 1]);
        assert_eq!(bins.indices_for_tile(0), &[0, 1]);
        assert_eq!(bins.indices_for_tile(1), &[1]);
    }

    #[test]
    fn stage_tensor_from_slice_falls_back_to_cpu_tensor() {
        let mut runtime = MetalRuntime::new(16, 16, Device::Cpu).unwrap();
        let tensor = runtime
            .stage_tensor_from_slice(MetalBufferSlot::TileIndices, &[1u32, 2, 3], 3)
            .unwrap();
        assert_eq!(tensor.to_vec1::<u32>().unwrap(), vec![1, 2, 3]);
    }
}
