use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, RwLockReadGuard};

use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_foundation::NSRange;
use objc2_metal::MTLSize;

use crate::diff::diff_splat::{DiffCamera, TrainableGaussians};

pub(crate) const METAL_TILE_SIZE: usize = 16;

const METAL_FILL_U32_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fill_u32(
    device uint* dst [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    dst[gid] = value;
}
"#;

const METAL_TILE_FORWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CameraUniform {
    float fx;
    float fy;
    float cx;
    float cy;
    uint width;
    uint height;
    uint tile_size;
    uint num_tiles_x;
};

struct TileRecord {
    uint start;
    uint count;
    uint _pad0;
    uint _pad1;
};

struct ProjectedGaussian {
    float u;
    float v;
    float sigma_x;
    float sigma_y;
    float depth;
    float opacity;
    float color_r;
    float color_g;
    float color_b;
    float _pad0;
    float _pad1;
    float _pad2;
};

kernel void tile_forward(
    constant CameraUniform& camera [[buffer(0)]],
    device const TileRecord* tile_records [[buffer(1)]],
    device const uint* tile_indices [[buffer(2)]],
    device const ProjectedGaussian* gaussians [[buffer(3)]],
    device float* out_color [[buffer(4)]],
    device float* out_depth [[buffer(5)]],
    device float* out_alpha [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= camera.width || gid.y >= camera.height) {
        return;
    }

    const uint pixel_idx = gid.y * camera.width + gid.x;
    const uint tile_x = gid.x / camera.tile_size;
    const uint tile_y = gid.y / camera.tile_size;
    const uint tile_idx = tile_y * camera.num_tiles_x + tile_x;
    const TileRecord record = tile_records[tile_idx];

    const float px = float(gid.x) + 0.5f;
    const float py = float(gid.y) + 0.5f;
    float trans = 1.0f;
    float3 color = float3(0.0f);
    float depth = 0.0f;
    float alpha_acc = 0.0f;

    for (uint i = 0; i < record.count; ++i) {
        const ProjectedGaussian gaussian = gaussians[tile_indices[record.start + i]];
        const float dx = (px - gaussian.u) / gaussian.sigma_x;
        const float dy = (py - gaussian.v) / gaussian.sigma_y;
        const float exponent = -0.5f * (dx * dx + dy * dy);
        const float alpha = clamp(exp(exponent) * gaussian.opacity, 0.0f, 0.99f);
        const float contrib = alpha * trans;
        color += contrib * float3(gaussian.color_r, gaussian.color_g, gaussian.color_b);
        depth += contrib * gaussian.depth;
        alpha_acc += contrib;
        trans *= (1.0f - alpha);
        if (trans <= 1e-4f) {
            break;
        }
    }

    out_color[pixel_idx * 3 + 0] = clamp(color.x, 0.0f, 1.0f);
    out_color[pixel_idx * 3 + 1] = clamp(color.y, 0.0f, 1.0f);
    out_color[pixel_idx * 3 + 2] = clamp(color.z, 0.0f, 1.0f);
    out_alpha[pixel_idx] = alpha_acc;
    out_depth[pixel_idx] = depth / (alpha_acc + 1e-6f);
}
"#;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenRect {
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
}

pub(crate) struct ChunkPixelWindow {
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
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

impl MetalProjectedGaussian {
    pub(crate) fn new(
        u: f32,
        v: f32,
        sigma_x: f32,
        sigma_y: f32,
        depth: f32,
        opacity: f32,
        color_r: f32,
        color_g: f32,
        color_b: f32,
    ) -> Self {
        Self {
            u,
            v,
            sigma_x,
            sigma_y,
            depth,
            opacity,
            color_r,
            color_g,
            color_b,
            ..Default::default()
        }
    }
}

pub(crate) struct NativeForwardFrame {
    pub color: Tensor,
    pub depth: Tensor,
    pub alpha: Tensor,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NativeForwardProfile {
    pub setup: std::time::Duration,
    pub staging: std::time::Duration,
    pub kernel: std::time::Duration,
    pub total: std::time::Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MetalBufferSlot {
    CameraUniforms,
    TileMetadata,
    TileIndices,
    ProjectedGaussians,
    OutputColor,
    OutputDepth,
    OutputAlpha,
}

impl MetalBufferSlot {
    fn label(self) -> &'static str {
        match self {
            Self::CameraUniforms => "camera_uniforms",
            Self::TileMetadata => "tile_metadata",
            Self::TileIndices => "tile_indices",
            Self::ProjectedGaussians => "projected_gaussians",
            Self::OutputColor => "output_color",
            Self::OutputDepth => "output_depth",
            Self::OutputAlpha => "output_alpha",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetalKernel {
    FillU32,
    TileForward,
}

impl MetalKernel {
    fn function_name(self) -> &'static str {
        match self {
            Self::FillU32 => "fill_u32",
            Self::TileForward => "tile_forward",
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::FillU32 => METAL_FILL_U32_KERNEL,
            Self::TileForward => METAL_TILE_FORWARD_KERNEL,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MetalRuntimeStats {
    pub tile_windows: usize,
    pub buffer_allocations: usize,
    pub buffer_reuses: usize,
    pub pipeline_compilations: usize,
}

struct PersistentMetalBuffer {
    byte_capacity: usize,
    backing: Option<Arc<Buffer>>,
}

impl PersistentMetalBuffer {
    fn new(byte_capacity: usize, backing: Option<Arc<Buffer>>) -> Self {
        Self {
            byte_capacity,
            backing,
        }
    }
}

pub(crate) struct MetalTensorView<'a> {
    storage: RwLockReadGuard<'a, Storage>,
    byte_offset: usize,
    element_count: usize,
    dtype: DType,
}

impl<'a> MetalTensorView<'a> {
    pub(crate) fn buffer(&self) -> candle_core::Result<&Buffer> {
        match &*self.storage {
            Storage::Metal(storage) => Ok(storage.buffer()),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }
}

pub(crate) struct MetalGaussianBindings<'a> {
    pub positions: MetalTensorView<'a>,
    pub scales: MetalTensorView<'a>,
    pub rotations: MetalTensorView<'a>,
    pub opacities: MetalTensorView<'a>,
    pub colors: MetalTensorView<'a>,
}

pub(crate) struct MetalRuntime {
    device: Device,
    tile_windows: Vec<ChunkPixelWindow>,
    num_tiles_x: usize,
    num_tiles_y: usize,
    buffers: HashMap<MetalBufferSlot, PersistentMetalBuffer>,
    pipelines: HashMap<MetalKernel, ComputePipeline>,
    stats: MetalRuntimeStats,
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
                tile_windows.push(build_chunk_pixel_window(
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
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            stats: MetalRuntimeStats {
                tile_windows: num_tiles_x * num_tiles_y,
                ..Default::default()
            },
        })
    }

    pub(crate) fn tile_window(&self, tile_idx: usize) -> candle_core::Result<&ChunkPixelWindow> {
        self.tile_windows
            .get(tile_idx)
            .ok_or_else(|| candle_core::Error::Msg(format!("invalid tile index {tile_idx}")))
    }

    pub(crate) fn tile_grid(&self) -> (usize, usize) {
        (self.num_tiles_x, self.num_tiles_y)
    }

    pub(crate) fn stats(&self) -> MetalRuntimeStats {
        self.stats
    }

    pub(crate) fn buffer_capacity(&self, slot: MetalBufferSlot) -> usize {
        self.buffers
            .get(&slot)
            .map(|buffer| buffer.byte_capacity)
            .unwrap_or(0)
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
        Ok(())
    }

    pub(crate) fn reserve_tile_index_capacity(
        &mut self,
        tile_refs: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            tile_refs.saturating_mul(size_of::<u32>()),
        )
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
        };
        self.write_struct(MetalBufferSlot::CameraUniforms, &camera_uniform)
    }

    pub(crate) fn bind_tensor<'a>(
        &self,
        tensor: &'a Tensor,
    ) -> candle_core::Result<MetalTensorView<'a>> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            candle_core::bail!("metal tensor binding requires contiguous layout");
        }
        let byte_offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        let element_count = layout.shape().elem_count();
        match &*storage {
            Storage::Metal(_) => Ok(MetalTensorView {
                storage,
                byte_offset,
                element_count,
                dtype: tensor.dtype(),
            }),
            _ => candle_core::bail!("tensor is not backed by Metal storage"),
        }
    }

    pub(crate) fn bind_gaussians<'a>(
        &self,
        gaussians: &'a TrainableGaussians,
    ) -> candle_core::Result<MetalGaussianBindings<'a>> {
        Ok(MetalGaussianBindings {
            positions: self.bind_tensor(gaussians.positions())?,
            scales: self.bind_tensor(gaussians.scales.as_tensor())?,
            rotations: self.bind_tensor(gaussians.rotations.as_tensor())?,
            opacities: self.bind_tensor(gaussians.opacities.as_tensor())?,
            colors: self.bind_tensor(gaussians.colors())?,
        })
    }

    pub(crate) fn reserve_forward_buffers(
        &mut self,
        gaussian_count: usize,
        tile_ref_count: usize,
        pixel_count: usize,
    ) -> candle_core::Result<()> {
        self.ensure_buffer(
            MetalBufferSlot::ProjectedGaussians,
            gaussian_count.saturating_mul(size_of::<MetalProjectedGaussian>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileMetadata,
            self.tile_windows
                .len()
                .saturating_mul(size_of::<MetalTileDispatchRecord>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::TileIndices,
            tile_ref_count.saturating_mul(size_of::<u32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputColor,
            pixel_count
                .saturating_mul(3)
                .saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputDepth,
            pixel_count.saturating_mul(size_of::<f32>()),
        )?;
        self.ensure_buffer(
            MetalBufferSlot::OutputAlpha,
            pixel_count.saturating_mul(size_of::<f32>()),
        )?;
        Ok(())
    }

    pub(crate) fn rasterize_forward(
        &mut self,
        gaussians: &[MetalProjectedGaussian],
        tile_records: &[MetalTileDispatchRecord],
        tile_indices: &[u32],
        render_width: usize,
        render_height: usize,
    ) -> candle_core::Result<(NativeForwardFrame, NativeForwardProfile)> {
        let total_start = std::time::Instant::now();
        let pixel_count = render_width.saturating_mul(render_height);
        let setup_start = std::time::Instant::now();
        self.reserve_forward_buffers(gaussians.len(), tile_indices.len(), pixel_count)?;
        let pipeline = self.ensure_pipeline(MetalKernel::TileForward)?.clone();
        let color_buffer = self
            .buffer_handle(MetalBufferSlot::OutputColor)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output color buffer".into()))?;
        let depth_buffer = self
            .buffer_handle(MetalBufferSlot::OutputDepth)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output depth buffer".into()))?;
        let alpha_buffer = self
            .buffer_handle(MetalBufferSlot::OutputAlpha)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing output alpha buffer".into()))?;
        let tile_buffer = self
            .buffer_handle(MetalBufferSlot::TileMetadata)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing tile metadata buffer".into()))?;
        let tile_index_buffer = self
            .buffer_handle(MetalBufferSlot::TileIndices)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing tile index buffer".into()))?;
        let gaussian_buffer = self
            .buffer_handle(MetalBufferSlot::ProjectedGaussians)?
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing gaussian buffer".into()))?;
        let setup = setup_start.elapsed();

        let staging_start = std::time::Instant::now();
        self.write_slice(MetalBufferSlot::ProjectedGaussians, gaussians)?;
        self.write_slice(MetalBufferSlot::TileMetadata, tile_records)?;
        self.write_slice(MetalBufferSlot::TileIndices, tile_indices)?;
        let staging = staging_start.elapsed();

        let kernel_start = std::time::Instant::now();
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::TileForward.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            self.buffer_handle(MetalBufferSlot::CameraUniforms)?
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
        self.device.synchronize()?;
        let kernel = kernel_start.elapsed();

        let frame = NativeForwardFrame {
            color: self.tensor_from_buffer(
                MetalBufferSlot::OutputColor,
                pixel_count.saturating_mul(3),
                DType::F32,
                (pixel_count, 3),
            )?,
            depth: self.tensor_from_buffer(
                MetalBufferSlot::OutputDepth,
                pixel_count,
                DType::F32,
                (pixel_count,),
            )?,
            alpha: self.tensor_from_buffer(
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
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("fill_u32 requires a Metal device");
        };
        let metal = self.device.as_metal_device()?;
        let encoder = metal.command_encoder()?;
        encoder.set_label(MetalKernel::FillU32.function_name());
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(buffer.as_ref()), 0);
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
        self.device.synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn read_u32_buffer(
        &self,
        slot: MetalBufferSlot,
        len: usize,
    ) -> candle_core::Result<Vec<u32>> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        self.device.synchronize()?;
        let values = unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, len) };
        Ok(values.to_vec())
    }

    fn buffer_handle(&self, slot: MetalBufferSlot) -> candle_core::Result<Option<&Arc<Buffer>>> {
        Ok(self
            .buffers
            .get(&slot)
            .map(|buffer| buffer.backing.as_ref())
            .flatten())
    }

    fn ensure_pipeline(&mut self, kernel: MetalKernel) -> candle_core::Result<&ComputePipeline> {
        if !self.pipelines.contains_key(&kernel) {
            let metal = self.device.as_metal_device()?;
            let library = metal
                .new_library_with_source(kernel.source(), None)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            let function = library
                .get_function(kernel.function_name(), None)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            let pipeline = metal
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            self.pipelines.insert(kernel, pipeline);
            self.stats.pipeline_compilations += 1;
        }
        Ok(self
            .pipelines
            .get(&kernel)
            .expect("pipeline inserted before lookup"))
    }

    fn ensure_buffer(
        &mut self,
        slot: MetalBufferSlot,
        byte_capacity: usize,
    ) -> candle_core::Result<()> {
        let required = byte_capacity.max(1);
        if let Some(existing) = self.buffers.get(&slot) {
            if existing.byte_capacity >= required {
                self.stats.buffer_reuses += 1;
                return Ok(());
            }
        }

        let backing = if self.device.is_metal() {
            let metal = self.device.as_metal_device()?;
            Some(metal.new_buffer(required, DType::U8, slot.label())?)
        } else {
            None
        };
        self.buffers
            .insert(slot, PersistentMetalBuffer::new(required, backing));
        self.stats.buffer_allocations += 1;
        Ok(())
    }

    fn tensor_from_buffer<S: Into<Shape>>(
        &self,
        slot: MetalBufferSlot,
        element_count: usize,
        dtype: DType,
        shape: S,
    ) -> candle_core::Result<Tensor> {
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref().cloned())
        else {
            candle_core::bail!("buffer {:?} is not backed by Metal", slot);
        };
        let metal = self.device.as_metal_device()?.clone();
        let storage = Storage::Metal(MetalStorage::new(buffer, metal, element_count, dtype));
        Ok(Tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    fn write_struct<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        value: &T,
    ) -> candle_core::Result<()> {
        let data =
            unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.write_bytes(slot, data)
    }

    fn write_slice<T: Copy>(
        &mut self,
        slot: MetalBufferSlot,
        values: &[T],
    ) -> candle_core::Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        self.write_bytes(slot, data)
    }

    fn write_bytes(&mut self, slot: MetalBufferSlot, data: &[u8]) -> candle_core::Result<()> {
        self.ensure_buffer(slot, data.len())?;
        let Some(buffer) = self
            .buffers
            .get(&slot)
            .and_then(|buffer| buffer.backing.as_ref())
        else {
            return Ok(());
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.contents().cast::<u8>(),
                data.len(),
            );
        }
        buffer.did_modify_range(NSRange::new(0, data.len()));
        Ok(())
    }
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

fn build_chunk_pixel_window(
    device: &Device,
    render_width: usize,
    rect: ScreenRect,
) -> candle_core::Result<ChunkPixelWindow> {
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
    Ok(ChunkPixelWindow {
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
        assert_eq!(initial.buffer_allocations, 3);

        runtime.reserve_core_buffers(32).unwrap();
        let reused = runtime.stats();
        assert_eq!(reused.buffer_allocations, initial.buffer_allocations);
        assert!(reused.buffer_reuses >= 3);
    }

    #[test]
    fn fill_kernel_writes_shared_buffer() {
        let Ok(device) = Device::new_metal(0) else {
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
}
